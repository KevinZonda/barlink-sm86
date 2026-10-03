# SPDX-License-Identifier: MIT
#
# torch.distributed ProcessGroup backend over the barlink_sm86 peer link.
#
#   from barlink_sm86 import process_group as blpg
#   blpg.init_process_group(init_method="tcp://127.0.0.1:29500",
#                           rank=r, world_size=2)
#   dist.all_reduce(tensor)   # SUM only; contiguous aligned CUDA tensors of
#                             # fp16/bf16/fp32/fp64/fp8 take the zero-copy
#                             # path, everything else stages through the pool
#
# Process model: SYMMETRIC SPMD, one process per GPU (world MUST be 2). The
# backend is a thin routing layer over bl._C peer mode; both ranks execute
# identical op sequences with identical sizes (the bl link itself has no
# runtime control channel -- see core.cu).
#
# Sync semantics: collectives return a completed _CompletedWork; completion
# is guaranteed for work subsequently enqueued on the CURRENT stream (the
# flag waits are stream-ordered). They are NOT host-synchronous.

import os

import torch
import torch.distributed as dist

import barlink_sm86 as bl

_C10D = torch._C._distributed_c10d

_SUPPORTED = (torch.float32, torch.float64, torch.bfloat16,
              torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8)
_NATIVE = (torch.float32, torch.float64, torch.bfloat16,
           torch.float8_e4m3fn, torch.float8_e5m2)

_CHUNK = 32 << 20   # per-exchange chunk for large allreduces (staging pair)


from torch._C._distributed_c10d import Work as _C10dWork


class _CompletedWork(_C10dWork):
    """Return handle for host-async collectives.

    barlink collectives are stream-ordered: completion is guaranteed for work
    subsequently enqueued on the CURRENT stream, so there is nothing to block
    on. torch's functional-collective layer (DTensor redistribute) requires a
    real Work object -- returning None segfaults c10d_functional at wait time.
    Stateless, so every collective returns the same singleton instance.
    """
    def __init__(self):
        super().__init__()

    def is_completed(self):
        return True

    def is_success(self):
        return True

    def wait(self, timeout=None):
        return True


_COMPLETED_WORK = _CompletedWork()


def _round16(n):
    return (n + 15) & ~15


class BarlinkBackend(_C10D.Backend):
    def __init__(self, rank, size, device, pool_mb, sock_path):
        super().__init__(rank, size)
        self._device = device
        self._pool_mb = pool_mb
        self._sock_path = sock_path
        # staging caches: peer-written buffers are safe to refill because
        # bl's arm handshake orders the peer's next write after our arm,
        # which is stream-ordered after our local refill (see core.cu)
        self._ar = {}    # (nbytes16, dtype) -> (x, y) fused-reduce pair
        self._bc = {}    # (nbytes16, dtype) -> single buffer
        if size != 2:
            raise RuntimeError(
                "barlink ProcessGroup: world_size must be exactly 2 "
                "(the underlying peer link is dual-GPU only), got %d" % size)
        bl._C.init_peer(device.index, pool_mb, sock_path, rank)

    # -- identity ---------------------------------------------------------
    def name(self):
        return "barlink"

    # -- helpers ----------------------------------------------------------
    def _to_device(self, t):
        if not t.is_cuda:
            t = t.to(self._device)
        return t.contiguous()

    def _ar_pair(self, n16, dt):
        # x_bytes == y_bytes (bl.allreduce_ requires equal numel), each
        # 16-aligned, together covering >= n16 bytes
        key = (n16, dt)
        pair = self._ar.get(key)
        if pair is None:
            half = ((n16 + 31) // 32) * 16
            pair = (bl.empty(half, device=self._device.index, dtype=dt),
                    bl.empty(half, device=self._device.index, dtype=dt))
            self._ar[key] = pair
        return pair

    def _bcast_buf(self, n16, dt, tag):
        key = (n16, dt, tag)
        buf = self._bc.get(key)
        if buf is None:
            buf = bl.empty(n16, device=self._device.index, dtype=dt)
            self._bc[key] = buf
        return buf

    # -- collectives ------------------------------------------------------
    def allreduce(self, tensors, opts):
        op = getattr(opts, "reduceOp", None)
        if op is not None and op != dist.ReduceOp.SUM:
            raise RuntimeError(
                "barlink allreduce: only ReduceOp.SUM is supported, got %s" % op)
        orig = tensors[0]

        # zero-copy path: the payload kernel reads the user's tensor in place
        # and the add kernel writes it back -- no staging copies at all.
        if self._zero_copy_usable(orig):
            self._allreduce_zerocopy(orig)
            return _COMPLETED_WORK

        t = self._to_device(orig)
        dt = t.dtype

        if dt == torch.float16:
            # only reachable for non-contiguous / unaligned fp16 (the
            # zero-copy path handles the common case natively): the pool has
            # no fp16 add, so reduce in fp32 staging and cast back
            self._allreduce_cast(t, torch.float32)
        elif dt not in _SUPPORTED:
            raise RuntimeError(
                "barlink allreduce: unsupported dtype %s (supported: fp32, "
                "fp64, bf16, fp16, fp8_e4m3fn, fp8_e5m2, uint8)" % dt)
        else:
            self._allreduce_native(t)
        if t is not orig:
            orig.copy_(t)   # non-contiguous / CPU input: result into the original
        return _COMPLETED_WORK

    # dtypes the zero-copy kernel adds natively (fp16 included; u8 keeps the
    # mod-256 pool semantics and stays on the staging path)
    _ZC_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16,
                  torch.float8_e4m3fn, torch.float8_e5m2)

    def _zero_copy_usable(self, t):
        return (t.is_cuda and t.dtype in self._ZC_DTYPES and
                t.is_contiguous() and t.numel() > 0 and
                t.data_ptr() % 16 == 0 and
                (t.numel() * t.element_size()) % 16 == 0)

    def _allreduce_zerocopy(self, t):
        nbytes = t.numel() * t.element_size()
        # chunk bound: the peer scratch zone is pool/2 - flag tail; leave a
        # 1 MiB margin. Chunks stay 16-byte aligned because the step is a
        # multiple of 16 bytes and the tensor pointer is 16-aligned.
        chunk = min(_CHUNK, (max(self._pool_mb // 2 - 1, 1)) << 20)
        if nbytes <= chunk:
            bl.allreduce_into(t, t)   # common case: no slicing overhead
            return
        isz = t.element_size()
        per16 = 16 // isz
        step = max(per16, (chunk // 16) * per16)
        flat = t.view(-1)
        for i in range(0, flat.numel(), step):
            bl.allreduce_into(flat[i:i + step], flat[i:i + step])

    def _allreduce_native(self, t):
        n = t.numel() * t.element_size()
        for off in range(0, n, _CHUNK):
            raw = min(_CHUNK, n - off)
            n16 = _round16(raw)
            x, y = self._ar_pair(n16, t.dtype)
            half = x.numel() * x.element_size()
            bx = t.view(torch.uint8).view(-1)
            # fill halves (tail garbage beyond raw never round-trips)
            lim = min(half, raw)
            x.view(torch.uint8)[:lim].copy_(bx[off:off + lim])
            rest = raw - lim
            if rest > 0:
                y.view(torch.uint8)[:rest].copy_(bx[off + lim:off + lim + rest])
            # one bl call: two independent fused reductions (x, y)
            bl.allreduce_(x, y)
            # copy back on the current stream; the result was written by MY
            # OWN local add kernel, so a plain D2D copy sees it
            if lim > 0:
                t.view(torch.uint8).view(-1)[off:off + lim].copy_(
                    x.view(torch.uint8)[:lim])
            if rest > 0:
                t.view(torch.uint8).view(-1)[off + lim:off + lim + rest].copy_(
                    y.view(torch.uint8)[:rest])
        return None

    def _allreduce_cast(self, t, cast_dt):
        # fp16 (pool has no fp16 add): reduce in fp32 staging, cast back
        isz = t.element_size()
        step = _CHUNK // isz                 # elements per chunk
        for off in range(0, t.numel(), step):
            elems = min(step, t.numel() - off)
            hb = ((elems * 4 + 31) // 32) * 16    # fp32 bytes per half
            x, y = self._ar_pair(hb * 2, cast_dt)
            xf = x.view(cast_dt)
            yf = y.view(cast_dt)
            ex = min(hb // 4, elems)
            ey = elems - ex
            if ex > 0:
                xf[:ex].copy_(t[off:off + ex].to(cast_dt))
            if ey > 0:
                yf[:ey].copy_(t[off + ex:off + ex + ey].to(cast_dt))
            bl.allreduce_(x, y)
            if ex > 0:
                t[off:off + ex].copy_(xf[:ex])
            if ey > 0:
                t[off + ex:off + ex + ey].copy_(yf[:ey])
        return None

    def broadcast(self, tensors, opts):
        t = self._to_device(tensors[0])
        root = getattr(opts, "rootRank", 0)
        n16 = _round16(t.numel() * t.element_size())
        # SEPARATE send and receive staging: peer mode forbids dst == src
        # (my payload would overwrite the peer's source). I write my
        # send-buf into the peer's recv-buf offset; the peer writes into my
        # recv-buf. My send-buf is never touched by the peer.
        sbuf = self._bcast_buf(n16, t.dtype, "s")
        rbuf = self._bcast_buf(n16, t.dtype, "r")
        sbuf.view(t.dtype)[:t.numel()].copy_(t)
        bl.copy_(rbuf, sbuf)
        if self.rank() != root:
            # the peer wrote MY recv pool buffer: host readback via
            # ld.relaxed.sys (the proven inbound-write path)
            data = bl.readback(rbuf)
            t.copy_(data.view(t.dtype)[:t.numel()])
        return _COMPLETED_WORK

    def barrier(self, opts):
        x, y = self._ar_pair(16, torch.uint8)   # garbage in, ignored out
        bl.allreduce_(x, y)
        return _COMPLETED_WORK

    # -- v1: not implemented ----------------------------------------------
    def all_gather_single(self, out, inp, opts):
        raise NotImplementedError(
            "barlink: all_gather_into_tensor is not implemented in v1")

    def reduce_scatter_single(self, out, inp, opts):
        raise NotImplementedError(
            "barlink: reduce_scatter_tensor is not implemented in v1")

    def send(self, tensors, dstRank, tag):
        raise NotImplementedError("barlink: send is not implemented in v1")

    def recv(self, tensors, srcRank, tag):
        raise NotImplementedError("barlink: recv is not implemented in v1")

    def scatter(self, output_tensors, input_tensors, opts):
        raise NotImplementedError("barlink: SCATTER_PROBE_12345")


_backend = None


def _resolve_device(rank):
    if "BL_DEVICE" in os.environ:
        return torch.device("cuda", int(os.environ["BL_DEVICE"]))
    if "LOCAL_RANK" in os.environ:
        return torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return torch.device("cuda", rank)


def _creator(common_opts, backend_opts):
    # module-level singleton: a second creator call (new_group) reuses the
    # one peer link. NOTE: subgroups smaller than the world are NOT
    # supported (the SPMD discipline requires identical sequences on both
    # ranks of the link).
    global _backend
    if _backend is not None:
        return _backend
    rank = int(getattr(common_opts, "group_rank",
                       getattr(common_opts, "rank", 0)))
    size = int(getattr(common_opts, "group_size",
                       getattr(common_opts, "size", 0)))
    if size != 2:
        raise RuntimeError(
            "barlink ProcessGroup: world_size must be exactly 2, got %d" % size)
    pool_mb = int(os.environ.get("BL_POOL_MB", "64"))
    sock = os.environ.get(
        "BL_SOCK_PATH",
        "/tmp/barlink-pg-%d-%s.sock" % (os.getuid(),
                                        os.environ.get("BL_PG_ID", "default")))
    _backend = BarlinkBackend(rank, size, _resolve_device(rank), pool_mb, sock)
    return _backend


def register():
    dist.Backend.register_backend(
        "barlink", _creator, extended_api=True, devices=["cpu", "cuda"])
    return _creator


def init_process_group(*args, **kwargs):
    register()
    kwargs.setdefault("backend", "barlink")
    return dist.init_process_group(*args, **kwargs)
