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
#   dist.send/recv (and isend/irecv / P2POp)  # zero-copy byte move, any
#                             # dtype; tags accepted but not transported
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
import sys

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
        if os.environ.get("BL_PG_DEBUG") == "1":
            sys.stderr.write(
                "[blpg] init rank=%d size=%d device=%s sock=%s "
                "cuda_visible=%r local_rank=%r\n"
                % (rank, size, device, sock_path,
                   os.environ.get("CUDA_VISIBLE_DEVICES"),
                   os.environ.get("LOCAL_RANK")))
        bl._C.init_peer(device.index, pool_mb, sock_path, rank)
        if os.environ.get("BL_PG_DEBUG") == "1":
            sys.stderr.write("[blpg] init_peer OK rank=%d\n" % rank)

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
    def _dbg(self, *a):
        if os.environ.get("BL_PG_DEBUG") == "1":
            sys.stderr.write("[blpg] " + " ".join(str(x) for x in a) + "\n")

    def allreduce(self, tensors, opts):
        self._dbg("allreduce", tuple(tensors[0].shape), tensors[0].dtype,
                  tensors[0].is_cuda)
        if os.environ.get("BL_AR_TRACE") == "1":
            t = tensors[0]
            cs = 0.0
            try:
                cs = t.float().sum().item()   # sync debug only
            except Exception:
                pass
            import torch as _tt
            sys.stderr.write(
                "[bltrace] rank=%d ar shape=%s ptr=%x stream=%s insum=%.4f\n" %
                (self.rank(), tuple(t.shape), t.data_ptr(),
                 _tt.cuda.current_stream(), cs))
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
        if os.environ.get("BL_AR_NO_ZC") == "1":
            return False     # diagnostic: force the pool staging path
        return (t.is_cuda and t.dtype in self._ZC_DTYPES and
                t.is_contiguous() and t.numel() > 0 and
                t.data_ptr() % 16 == 0 and
                (t.numel() * t.element_size()) % 16 == 0)

    def _allreduce_zerocopy(self, t):
        nbytes = t.numel() * t.element_size()
        # chunk bound: the allreduce scratch zone is the BOTTOM quarter of
        # the pool (pool/4 - flag tail; p2p owns the top), 1 MiB margin.
        # Chunks stay 16-byte aligned because the step is a multiple of 16
        # bytes and the tensor pointer is 16-aligned.
        chunk = min(_CHUNK, (max(self._pool_mb // 4 - 1, 1)) << 20)
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
        self._dbg("broadcast", tuple(tensors[0].shape), tensors[0].dtype)
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

    def barrier(self, opts=None):
        self._dbg("barrier")
        # zero-copy fused allreduce on a cached tensor -- NOT pool tensors:
        # pool allreduce_ mirrors tensors into the ar scratch zone and
        # breaks once the pool bump grows past it (e.g. after large
        # all_gather staging allocations)
        if getattr(self, "_bar_t", None) is None:
            self._bar_t = torch.zeros(256, dtype=torch.bfloat16,
                                      device=self._device)
        bl.allreduce_into(self._bar_t, self._bar_t)
        return _COMPLETED_WORK

    # -- point-to-point ---------------------------------------------------
    # torch 2.14's Backend exposes only send/recv (both return a Work;
    # dist.isend/irecv and P2POp route through the SAME methods -- isend
    # simply does not wait on the returned work). Tags are accepted but NOT
    # transported: the bl link has no control channel, matching is by call
    # order (SPMD discipline), same as every other op on this backend.
    def send(self, tensors, dstRank, tag):
        dst = int(dstRank)
        if dst != 1 - self.rank():
            raise RuntimeError(
                "barlink send: world is 2, dst must be %d, got %d"
                % (1 - self.rank(), dst))
        self._send_impl(tensors[0])
        return _COMPLETED_WORK

    def recv(self, tensors, srcRank, tag):
        src = int(srcRank)
        if src != 1 - self.rank():
            raise RuntimeError(
                "barlink recv: world is 2, src must be %d, got %d"
                % (1 - self.rank(), src))
        self._recv_impl(tensors[0])
        return _COMPLETED_WORK

    # p2p zero-copy condition: like allreduce's but ANY dtype -- send/recv
    # is a pure byte move (u8 has no mod-256 semantics here)
    def _zc_p2p_usable(self, t):
        return (t.is_cuda and t.is_contiguous() and t.numel() > 0 and
                t.data_ptr() % 16 == 0 and
                (t.numel() * t.element_size()) % 16 == 0)

    def _p2p_zerocopy(self, t, send):
        nbytes = t.numel() * t.element_size()
        self._dbg("p2p_zerocopy send=%s nbytes=%d zc=%s" %
                  (send, nbytes, self._zc_p2p_usable(t)))
        # chunk bound: the p2p scratch zone is the TOP quarter of the pool
        # (pool/4 - flag tail; allreduce keeps the bottom half), 1 MiB margin.
        # NOTE: a single chunk is the largest unit the p2p path handles
        # reliably on this platform -- multi-chunk full-duplex exchanges hit
        # an unresolved flag-delivery pathology (~4.4s context death, see
        # BUILD_AND_TEST.md §10), so the chunk is sized to the whole zone
        # and large messages ride ONE chunk (slow but correct: BAR write
        # completion is ack-paced at ~10 MB/s under load).
        chunk = (max(self._pool_mb // 4 - 1, 1)) << 20
        peer = 1 - self.rank()
        if nbytes <= chunk:
            if send:
                bl.send_into(t, peer)
            else:
                bl.recv_into(t, peer)
            return
        step = (chunk // 16) * 16   # chunks stay 16-byte aligned
        fb = t.view(torch.uint8).view(-1)
        for off in range(0, nbytes, step):
            c = fb[off:off + step]
            # each chunk is its own exchange (own seq + arm handshake)
            if send:
                bl.send_into(c, peer)
            else:
                bl.recv_into(c, peer)

    def _send_impl(self, t):
        self._dbg("send_impl", tuple(t.shape), t.dtype)
        if self._zc_p2p_usable(t):
            self._p2p_zerocopy(t, send=True)
            return
        # staged through the pool: any dtype / any size / CPU / strided.
        # The round16 tail bytes are stale garbage on both sides; the
        # receiver copies only the exact n bytes back.
        t2 = self._to_device(t)
        n = t2.numel() * t2.element_size()
        n16 = _round16(max(n, 1))
        sbuf = self._bcast_buf(n16, torch.uint8, "ps")
        if n:
            sbuf[:n].copy_(t2.view(torch.uint8).view(-1))
        bl.send_into(sbuf, 1 - self.rank())

    def _recv_impl(self, t):
        self._dbg("recv_impl", tuple(t.shape), t.dtype)
        if self._zc_p2p_usable(t):
            self._p2p_zerocopy(t, send=False)
            return
        orig = t
        t2 = orig if orig.is_cuda else orig.to(self._device)
        t2 = t2.contiguous()
        n = t2.numel() * t2.element_size()
        n16 = _round16(max(n, 1))
        rbuf = self._bcast_buf(n16, torch.uint8, "pr")
        bl.recv_into(rbuf, 1 - self.rank())
        if n:
            t2.view(torch.uint8).view(-1).copy_(rbuf[:n])
        if t2 is not orig:
            orig.copy_(t2)

    # -- collectives: all_gather ------------------------------------------
    def all_gather_single(self, output, input, opts):
        self._dbg("all_gather_single", tuple(input.shape), input.dtype)
        # output: [world * n] contiguous; input: [n]. rank r's slice at
        # [r*n:(r+1)*n]. SPMD: both ranks copy locally, then exchange
        # their slices.
        n = input.numel()
        r = self.rank()
        o2 = output.view(-1)
        i2 = input.view(-1)
        mine = o2[r * n:(r + 1) * n]
        theirs = o2[(1 - r) * n:(2 - r) * n]
        mine.copy_(i2)
        nbytes = n * input.element_size()
        if nbytes == 0:
            return _COMPLETED_WORK
        p2p_chunk = (max(self._pool_mb // 4 - 1, 1)) << 20
        if nbytes <= p2p_chunk:
            # fast path: single-chunk p2p zero-copy exchange (full duplex)
            self._send_impl(mine)
            self._recv_impl(theirs)
            return _COMPLETED_WORK
        # large gather (e.g. vllm profile_run's padded-batch logits
        # gather, hundreds of MB with a 248k vocab): staged exchange
        # through the pool -- bl.copy_ is the proven arm-path protocol at
        # any chunk count, and the p2p multi-chunk full-duplex path has an
        # unresolved flag-delivery pathology (BUILD_AND_TEST.md section 10)
        es = input.element_size()
        per16 = 16 // es if 16 % es == 0 else 1
        STAGE = 16 << 20
        # ONE reusable staging pair (pool tensors are never freed; a
        # per-size cache exhausts the pool on multi-size chunk tails)
        if getattr(self, "_agstage", None) is None:
            self._agstage = (
                bl.empty(STAGE, device=self._device.index, dtype=torch.uint8),
                bl.empty(STAGE, device=self._device.index, dtype=torch.uint8))
        sfull, rfull = self._agstage
        step = max(per16, (STAGE // 16) * per16)   # elems per 16MB chunk
        for off in range(0, n, step):
            j = min(off + step, n)
            nb = (j - off) * es
            nb16 = _round16(nb)
            sbuf, rbuf = sfull[:nb16], rfull[:nb16]
            sbuf[:nb].copy_(i2[off:j].view(torch.uint8))
            bl.copy_(rbuf, sbuf)
            tmp = torch.empty(nb16, dtype=torch.uint8, device=self._device)
            bl.pool_move(rbuf, tmp)
            theirs[off:j].view(torch.uint8).copy_(tmp[:nb])
        return _COMPLETED_WORK

    def allgather(self, output_tensors, input_tensors, opts=None):
        # normalize to ranklist[r] = tensor or list-of-tensors for rank r.
        # torch shapes seen in the wild:
        #   dist.all_gather(list, t):   output_tensors = [[t_r0, t_r1]]
        #   Backend-style:              output_tensors[r] = list per rank
        #   overload 2:                 output_tensors = [t_r0, t_r1], t
        if isinstance(input_tensors, torch.Tensor):
            ins = [input_tensors]
        else:
            ins = list(input_tensors)
        outs = list(output_tensors)
        if isinstance(outs[0], torch.Tensor):
            ranklist = outs                        # [t_r0, t_r1]
        else:
            inner = list(outs[0])
            if len(outs) == 1 and len(inner) == 2 and \
                    isinstance(inner[0], torch.Tensor):
                ranklist = inner                   # [[t_r0, t_r1]] wrapper
            else:
                ranklist = [list(tl) for tl in outs]
        r = self.rank()
        peer = 1 - r
        for i, t in enumerate(ins):
            d, s = ranklist[r], ranklist[peer]
            dst = (d[i] if isinstance(d, list) else d).view(-1)
            src = (s[i] if isinstance(s, list) else s).view(-1)
            dst.copy_(t.view(-1))
            self._send_impl(dst)
            self._recv_impl(src)
        return _COMPLETED_WORK

    # -- v1: not implemented ----------------------------------------------
    def reduce_scatter_single(self, out, inp, opts):
        raise NotImplementedError(
            "barlink: reduce_scatter_tensor is not implemented in v1")

    def scatter(self, output_tensors, input_tensors, opts):
        raise NotImplementedError("barlink: SCATTER_PROBE_12345")


class _DummyBackend1(_C10D.Backend):
    """Honest local-semantics backend for size-1 subgroups (vLLM builds
    singleton groups for several of its communicator slices; the real
    link is dual-GPU only). Every collective is the world-1 identity:
    allreduce(SUM) leaves the tensor unchanged, all_gather copies the
    input into this rank's slice, broadcast is a no-op. send/recv have
    no peer and raise. Crucially this does NOT touch the bl link --
    it must not init_peer, poll the pool, or share any state with the
    world-2 singleton backend.
    """

    def __init__(self, rank):
        super().__init__(rank, 1)

    def name(self):
        return "barlink-dummy"

    def allreduce(self, tensors, opts):
        return _COMPLETED_WORK

    def all_gather_single(self, output, input, opts):
        output.view(-1)[: input.numel()] = input.view(-1)
        return _COMPLETED_WORK

    def allgather(self, output_tensors, input_tensors, opts=None):
        if isinstance(input_tensors, torch.Tensor):
            output_tensors[0].copy_(input_tensors)
        else:
            for o, i in zip(output_tensors[0], input_tensors):
                o.copy_(i)
        return _COMPLETED_WORK

    def broadcast(self, tensors, opts):
        return _COMPLETED_WORK

    def barrier(self, opts=None):
        return _COMPLETED_WORK

    def send(self, tensors, dstRank, tag):
        raise RuntimeError("barlink-dummy (world 1): send has no peer")

    def recv(self, tensors, srcRank, tag):
        raise RuntimeError("barlink-dummy (world 1): recv has no peer")


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
    # supported on the real link (the SPMD discipline requires identical
    # sequences on both ranks of the link); a SIZE-1 subgroup gets a
    # stateless local-semantics dummy that never touches the link.
    global _backend
    rank = int(getattr(common_opts, "group_rank",
                       getattr(common_opts, "rank", 0)))
    size = int(getattr(common_opts, "group_size",
                       getattr(common_opts, "size", 0)))
    if size == 1:
        return _DummyBackend1(rank)
    if _backend is not None:
        return _backend
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
