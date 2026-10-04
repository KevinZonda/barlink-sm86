# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B INT8 W8A16 (compressed-tensors pack-quantized) TP=2 decode
# benchmark: barlink PG vs NCCL.
#
#   bash demos/qwen_tp/qwen38_int8/run.sh barlink
#   bash demos/qwen_tp/qwen38_int8/run.sh nccl
#
# Weight treatment is IDENTICAL on both backends and both ranks (first
# principle of the comparison):
#   - both ranks stream the FULL checkpoint, unpack the int8 themselves and
#     cut their local shard (no c10d scatter)
#   - every quantized Linear becomes an Int8Linear: the checkpoint's
#     group-128 int8 is folded to per-row int8 (exact fp32 fold of
#     int8 x bf16-scale), activations are quantized dynamically per token,
#     matmul via torch._int_mm (cublasLt IMMA, sm80+). Decode reads 1
#     byte/param.
#   - colwise shards (q/k/v, gate/up, gdn in_proj_*): narrow the int8 weight
#     (+ row scale) along the output dim. rowwise shards (o/down/out_proj):
#     narrow along the input dim (5120/2=2560 and 17408/2=8704 are multiples
#     of group 128, so scales stay exact) and complete partial sums with
#     dist.all_reduce through the active backend (forward hooks, ~128 x 10KB
#     per token).
#   - lm_head (248320 x 5120 bf16, 2.5 GB -- does not fit replicated next to
#     the 11.3 GB int8 shard): colwise over vocab; the two half-logits are
#     exchanged and concatenated. barlink uses the isend/irecv zero-copy p2p
#     path; NCCL uses all_gather (its lazy p2p communicator segfaults on
#     this platform -- see AllGatherLMHead docstring).
#
# GDN (gated delta net) runs transformers' native torch kernel path
# (no fla / causal_conv1d installed): torch_recurrent_gated_delta_rule for
# cached decode. Same on both backends.

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import barlink_sm86 as bl
except ImportError:      # stock-torch (nccl) branch: no blrun PYTHONPATH
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "torch_ext"))
    import barlink_sm86 as bl
import torch.nn.functional as F

MODEL_DIR = "/mnt/modelzoo/lued/Qwen3.8-27B-INT8-W8A16-MTP"
PROMPT = ("Explain how PCIe peer-to-peer DMA over a GPU BAR1 aperture works, "
          "covering TLP routing, IOMMU translation, and why a receiving "
          "card's L2 cache is not coherent with inbound writes.")
GEN_TOKENS = 64
WARMUP_TOKENS = 16


# ---------------------------------------------------------------------------
# int8 linear (folded from the checkpoint's group-128 quant)
# ---------------------------------------------------------------------------

class Int8Linear(nn.Module):
    """Per-row symmetric int8 weight + per-token dynamic int8 activation,
    matmul via torch._int_mm (cublasLt IMMA, sm80+)."""

    def __init__(self, qweight: torch.Tensor, wscale: torch.Tensor,
                 in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", qweight)          # int8 [out, in]
        self.register_buffer("wscale", wscale)            # fp32 [out]

    def forward(self, x):
        sh = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).float()
        xs = x2.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        xq = torch.round(x2 / xs[:, None]).clamp(-127, 127).to(torch.int8)
        if xq.shape[0] <= 16:
            # torch._int_mm requires m > 16 (cublasLt IMMA); decode has m=1.
            # Padding to 32 rows adds no weight traffic (memory-bound) and
            # the extra rows are sliced away.
            pad = torch.zeros(32, xq.shape[1], dtype=torch.int8, device=xq.device)
            pad[: xq.shape[0]] = xq
            y = torch._int_mm(pad, self.qweight.t())[: xq.shape[0]]
        else:
            y = torch._int_mm(xq, self.qweight.t())
        y = y.float() * (self.wscale * xs[:, None])
        return y.to(x.dtype).reshape(*sh, self.out_features)


class W8A16Linear(nn.Module):
    """True W8A16 weight-only int8 linear (w8a16.cu): group-128 scales kept
    EXACT (no per-row refold -- strictly better fidelity than the old dynamic
    path), activations quantized per token to int8, GEMM via __dp4a with a
    coalesced chunk-transposed weight layout. 2-3 kernels per call instead of
    ~10. M > 16 falls back to a dequantized matmul (prefill only)."""

    def __init__(self, wq_t, sw_t, in_features, out_features, dev):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.G = in_features // 128
        self.register_buffer("wq_t", wq_t)      # int8 [K/16, N, 16]
        self.register_buffer("sw_t", sw_t)      # fp32 [G, N]
        # split-K: target ~160+ blocks for SM saturation
        bx = (out_features + 255) // 256
        self.S = max(1, min(self.G, (160 * 256) // max(out_features, 1) or 1))

    def forward(self, x):
        sh = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        M = x2.shape[0]
        if M <= 16:
            xq, xs = bl._C.w8a16_quant_act(x2)
            y = bl._C.w8a16_gemm(xq, xs, self.wq_t, self.sw_t,
                                 self.out_features, self.in_features, self.S)
            return y.reshape(*sh, self.out_features)
        # prefill fallback: dequantize (exact int8 x group scale) and matmul
        wq = self.wq_t.permute(1, 0, 2).reshape(self.out_features,
                                                self.in_features)
        w = wq.float() * self.sw_t.repeat_interleave(128, dim=0).t()
        y = x2.float() @ w.t()
        return y.to(x.dtype).reshape(*sh, self.out_features)


def build_int8_linear(packed_i32, scale_bf16, dev):
    """Old dynamic-quant path (per-row fold + torch._int_mm). Kept for
    A/B comparison via QWEN38_W8A16=0."""
    wq = (packed_i32.to(dev).view(torch.int8) ^ -128)      # true int8 [out,in]
    gs = wq.shape[1] // scale_bf16.shape[1]
    w = wq.float() * scale_bf16.to(dev).float().repeat_interleave(gs, dim=1)
    rs = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    wq2 = torch.round(w / rs[:, None]).clamp(-127, 127).to(torch.int8)
    return Int8Linear(wq2.contiguous(), rs, wq.shape[1], wq.shape[0])


def build_w8a16_linear(packed_i32, scale_bf16, dev):
    """New W8A16 path: exact unpack + chunk-transposed layout + fp32
    transposed group scales (bf16 -> fp32 is lossless). No refold."""
    wq = (packed_i32.to(dev).view(torch.int8) ^ -128)          # [N, K] int8
    N, K = wq.shape
    wq_t = wq.view(N, K // 16, 16).permute(1, 0, 2).contiguous()
    sw_t = scale_bf16.to(dev).float().t().contiguous()         # [G, N] fp32
    return W8A16Linear(wq_t, sw_t, K, N, dev)


# ---------------------------------------------------------------------------
# TP helpers
# ---------------------------------------------------------------------------

_AR_TIME = {"events": [], "calls": 0}
_NO_COMM = False


def _timed_all_reduce(t):
    if _NO_COMM:
        return t
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    dist.all_reduce(t)
    e1.record()
    _AR_TIME["events"].append((e0, e1))
    _AR_TIME["calls"] += 1
    return t


def _hook_rowwise(mod):
    mod.register_forward_hook(lambda m, i, o: _timed_all_reduce(o))


def _narrow(t, dim, rank, world=2):
    n = t.shape[dim]
    assert n % world == 0, (tuple(t.shape), dim)
    s = n // world
    # clone: a dim-0 narrow of a contiguous tensor is itself contiguous, so
    # .contiguous() alone would alias the FULL tensor and keep it alive
    return t.detach().narrow(dim, rank * s, s) \
        .clone(memory_format=torch.contiguous_format)


def shard_int8_col(mod: Int8Linear, rank, world=2):
    mod.qweight = _narrow(mod.qweight, 0, rank, world)
    mod.wscale = _narrow(mod.wscale, 0, rank, world)
    mod.out_features //= world


def shard_int8_row(mod: Int8Linear, rank, world=2):
    mod.qweight = _narrow(mod.qweight, 1, rank, world)
    mod.in_features //= world
    _hook_rowwise(mod)


class AllGatherLMHead(nn.Module):
    """Colwise lm_head over the vocab dim; the two half-logits are exchanged
    and concatenated. Backend note: NCCL's lazy per-p2p-op communicator
    segfaults on this platform (torch 2.14/cu130, dual 3080, driver P2P
    disabled -- reproduced with a bare isend/irecv), so the NCCL branch
    uses the all_gather collective; barlink uses the isend/irecv zero-copy
    p2p path (all_gather is not implemented in the barlink PG)."""

    def __init__(self, weight_shard, rank, world=2, use_p2p=True):
        super().__init__()
        self.rank = rank
        self.peer = 1 - rank
        self.use_p2p = use_p2p
        self.register_buffer("weight", weight_shard)     # bf16 [vocab/w, hid]

    def forward(self, h):
        half = F.linear(h, self.weight)                  # [..., vocab/2]
        if _NO_COMM:
            # comm-free control run: identical matmul cost, no exchange
            # (the argmax is wrong on purpose; only the timing matters)
            return half
        if self.use_p2p:
            other = torch.empty_like(half)
            rs = dist.isend(half.contiguous(), self.peer)
            rr = dist.irecv(other, self.peer)
            rs.wait()
            rr.wait()
        else:
            parts = [torch.empty_like(half) for _ in range(2)]
            dist.all_gather(parts, half.contiguous())
            other = parts[self.peer]
        if self.rank == 0:
            return torch.cat([half, other], dim=-1)
        return torch.cat([other, half], dim=-1)


# ---------------------------------------------------------------------------
# model construction: meta-init the HF text model, stream real tensors in
# ---------------------------------------------------------------------------

def shard_w8_col(mod, rank, world=2):
    # narrow the OUTPUT dim (N): wq_t [K/16, N, 16] dim 1, sw_t [G, N] dim 1
    N = mod.out_features
    h = N // world
    mod.wq_t = mod.wq_t[:, rank * h:(rank + 1) * h, :].contiguous()
    mod.sw_t = mod.sw_t[:, rank * h:(rank + 1) * h].contiguous()
    mod.out_features = h


def shard_w8_row(mod, rank, world=2):
    # narrow the INPUT dim (K): wq_t dim 0 (K/16 chunks), sw_t dim 0 (groups)
    K = mod.in_features
    h = K // world
    # NB .contiguous() is a NO-OP on a leading-dim slice (it IS contiguous) --
    # it would keep the FULL storage alive as a view. clone() really copies.
    mod.wq_t = mod.wq_t[rank * (h // 16):(rank + 1) * (h // 16)].clone()
    mod.sw_t = mod.sw_t[rank * (h // 128):(rank + 1) * (h // 128)].clone()
    mod.in_features = h
    mod.G = h // 128
    bx = (mod.out_features + 255) // 256
    mod.S = max(1, min(mod.G, (160 * 256) // max(mod.out_features, 1) or 1))
    _hook_rowwise(mod)


def qkv_seg(t, rank, kd, vd, fk):
    # [q | k | v] segmentation along the N dim (dim 1 for both layouts);
    # works for wq_t [K/16, N, 16] and sw_t [G, N]
    return torch.cat([
        t.narrow(1, 0, fk).narrow(1, rank * kd, kd),
        t.narrow(1, fk, fk).narrow(1, rank * kd, kd),
        t.narrow(1, 2 * fk, t.shape[1] - 2 * fk).narrow(1, rank * vd, vd),
    ], dim=1).contiguous()


def _meta_text_model():
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding)
    tc = AutoConfig.from_pretrained(MODEL_DIR).text_config
    with torch.device("meta"):
        lm = Qwen3_5TextModel(tc)
    # rotary was built under meta: re-create it for real (pure config math)
    lm.rotary_emb = Qwen3_5TextRotaryEmbedding(config=tc)
    return lm, tc


def load_and_shard(dev, rank, use_p2p=True, world=2, verbose=False):
    from safetensors import safe_open

    lm, tc = _meta_text_model()
    lm.rotary_emb = lm.rotary_emb.to(dev)
    idx = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    key2file = idx["weight_map"]

    # group checkpoint keys by destination module (strip the
    # 'model.language_model.' prefix)
    modkeys = defaultdict(dict)
    prefix = "model.language_model."
    for k, fn in key2file.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split(".")
        modkeys[".".join(parts[:-1])][parts[-1]] = (k, fn)

    def tensor(k, fn):
        with safe_open(os.path.join(MODEL_DIR, fn), framework="pt") as f:
            return f.get_tensor(k)

    def seg_narrow(t, kd, vd, fk):    # [q | k | v] dim-0 segments, per head
        return torch.cat([
            t[:fk].narrow(0, rank * kd, kd),
            t[fk:2 * fk].narrow(0, rank * kd, kd),
            t[2 * fk:].narrow(0, rank * vd, vd),
        ], dim=0).contiguous()

    # shard geometry (identical on both ranks by SPMD construction)
    kh = tc.linear_num_key_heads // world           # 16 -> 8
    vh = tc.linear_num_value_heads // world         # 48 -> 24
    kd, vd = kh * tc.linear_key_head_dim, vh * tc.linear_value_head_dim
    fk = tc.linear_num_key_heads * tc.linear_key_head_dim       # 2048
    COL = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "in_proj_z"}

    n_int8 = 0
    for path, keys in sorted(modkeys.items()):
        if os.environ.get("QWEN38_MEMDBG") == "1" and (n_int8 % 50 == 0 or (150 <= n_int8 < 166)):
            print("[memdbg] n=%d alloc=%.2f GiB %s" %
                  (n_int8, torch.cuda.memory_allocated(dev) / 2**30, path),
                  flush=True)
        parent_path, _, name = path.rpartition(".")
        parent = lm.get_submodule(parent_path) if parent_path else lm
        if "weight_packed" in keys:
            k, fn = keys["weight_packed"]
            ks, fns = keys["weight_scale"]
            kp, fnp = keys["weight_shape"]
            shape = tuple(tensor(kp, fnp).tolist())
            if os.environ.get("QWEN38_W8A16", "1") == "1":
                mod = build_w8a16_linear(tensor(k, fn), tensor(ks, fns), dev)
            else:
                mod = build_int8_linear(tensor(k, fn), tensor(ks, fns), dev)
            if shape != (mod.out_features, mod.in_features):
                raise RuntimeError("packed shape mismatch: %s vs %s" %
                                   (shape, (mod.out_features, mod.in_features)))
            # shard IMMEDIATELY: the full-size shard of all layers does not
            # fit on the card alongside the prep transients
            if os.environ.get("QWEN38_W8A16", "1") == "1":
                # new W8A16 kernel path
                if name in ("q_proj", "k_proj", "v_proj", "gate_proj",
                            "up_proj", "in_proj_z"):
                    shard_w8_col(mod, rank, world)
                elif name in ("o_proj", "down_proj", "out_proj"):
                    shard_w8_row(mod, rank, world)
                elif name == "in_proj_qkv":
                    mod.wq_t = qkv_seg(mod.wq_t, rank, kd, vd, fk)
                    mod.sw_t = qkv_seg(mod.sw_t, rank, kd, vd, fk)
                    mod.out_features = 2 * kd + vd
                else:
                    raise RuntimeError("no shard rule for %s" % path)
            elif name in COL:
                shard_int8_col(mod, rank, world)
            elif name in ("o_proj", "down_proj", "out_proj"):
                shard_int8_row(mod, rank, world)
            elif name == "in_proj_qkv":
                mod.qweight = seg_narrow(mod.qweight, kd, vd, fk)
                mod.wscale = seg_narrow(mod.wscale, kd, vd, fk)
                mod.out_features = 2 * kd + vd
            else:
                raise RuntimeError("no shard rule for quantized %s" % path)
            setattr(parent, name, mod)
            n_int8 += 1
        elif set(keys) == {"weight"}:
            k, fn = keys["weight"]
            t = tensor(k, fn).to(dev)
            mod = lm.get_submodule(path)
            if name in ("in_proj_a", "in_proj_b"):
                t = _narrow(t, 0, rank, world)      # rows: v heads
            elif name == "conv1d":
                t = seg_narrow(t.squeeze(1), kd, vd, fk)[:, None, :]
            if isinstance(mod, nn.Linear):
                # bf16 linear kept as-is (gdn in_proj_a / in_proj_b)
                new = nn.Linear(t.shape[1], t.shape[0], bias=False,
                                device=dev, dtype=t.dtype)
                new.weight = nn.Parameter(t, requires_grad=False)
                setattr(parent, name, new)
            else:  # Embedding / RMSNorm / RMSNormGated / conv1d
                mod.weight = nn.Parameter(t, requires_grad=False)
        elif set(keys) <= {"A_log", "dt_bias"}:
            tgt = lm.get_submodule(path)
            for pname, (k, fn) in keys.items():
                setattr(tgt, pname,
                        nn.Parameter(_narrow(tensor(k, fn).to(dev), 0, rank,
                                             world), requires_grad=False))
        else:
            raise RuntimeError("unhandled key set at %s: %s"
                               % (path, sorted(keys)))

    if verbose:
        print("rank %d: %d Int8Linear modules loaded+sharded" % (rank, n_int8),
              flush=True)

    # ---- lm_head: colwise shard + isend/irecv allgather ----
    with safe_open(os.path.join(
            MODEL_DIR, key2file["lm_head.weight"]), framework="pt") as f:
        wfull = f.get_tensor("lm_head.weight")
    lm_head = AllGatherLMHead(_narrow(wfull, 0, rank).to(dev), rank, world,
                              use_p2p=use_p2p)
    del wfull
    gc.collect()

    # ---- sanity: no meta/garbage tensors left ----
    for n_, p in lm.named_parameters():
        if p.is_meta:
            raise RuntimeError("meta param left: %s" % n_)
    bad = [n_ for n_, p in lm.named_parameters() if not p.is_cuda]
    if bad:
        raise RuntimeError("non-cuda params: %s" % bad[:5])

    # ---- GDN module attribute updates (heads halved) ----
    for layer in lm.layers:
        if not hasattr(layer, "linear_attn"):
            continue
        la = layer.linear_attn
        la.conv1d.in_channels = la.conv1d.out_channels = 2 * kd + vd
        la.conv1d.groups = 2 * kd + vd
        la.num_k_heads, la.num_v_heads = kh, vh
        la.key_dim, la.value_dim = kd, vd
        la.conv_dim = 2 * kd + vd

    # keep the config consistent with the shard (mask/cache creation reads it)
    lm.config.num_attention_heads = tc.num_attention_heads // world
    lm.config.num_key_value_heads = tc.num_key_value_heads // world
    lm.config.linear_num_key_heads = kh
    lm.config.linear_num_value_heads = vh
    lm.eval()
    return lm, lm_head, tc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="barlink", choices=["barlink", "nccl"])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--warmup-tokens", type=int, default=WARMUP_TOKENS)
    ap.add_argument("--dump-step", type=int, default=0,
                    help="timed-region step whose logits/hidden are saved "
                         "(0 = first generated token: both backends see the "
                         "identical prompt, so the dump isolates the backend "
                         "numeric delta)")
    ap.add_argument("--no-comm", action="store_true",
                    help="control run: skip all all_reduce / logits exchange "
                         "(wrong tokens, identical compute; timing ceiling)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "results"))
    a = ap.parse_args()

    global _NO_COMM
    _NO_COMM = a.no_comm

    # NCCL branch pins ONE card per process via CUDA_VISIBLE_DEVICES; the
    # barlink branch (tools/blrun) sees both cards and uses rank
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and "," not in vis:
        dev = torch.device("cuda", 0)
    else:
        dev = torch.device("cuda", a.rank)
    torch.cuda.set_device(dev)
    if a.backend == "barlink":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "..", "torch_ext"))
        from barlink_sm86 import process_group as blpg
        blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                                rank=a.rank, world_size=2)
    else:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:%d"
                                % a.port, rank=a.rank, world_size=2)

    t0 = time.time()
    lm, lm_head, tc = load_and_shard(dev, a.rank, use_p2p=(a.backend == "barlink"),
                                     verbose=(a.rank == 0))
    load_s = time.time() - t0
    gc.collect()
    torch.cuda.reset_peak_memory_stats(dev)
    mem = torch.cuda.max_memory_allocated(dev) / (1 << 30)
    print("[qwen38-int8 %s rank=%d] load=%.0fs peak=%.1f GiB" %
          (a.backend, a.rank, load_s, mem), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(a.prompt, return_tensors="pt").input_ids.to(dev)
    n_prompt = ids.shape[1]

    def gen(n, dump_at=None):
        cache = None
        toks = []
        dumps = {}
        t_prefill = t_decode = 0.0
        with torch.inference_mode():
            for i in range(n):
                inp = ids if i == 0 else toks[-1][:, None]
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                out = lm(input_ids=inp, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                h = out.last_hidden_state[:, -1]
                logits = lm_head(h)                     # bf16, allgathered
                if i == 0:
                    t_prefill = time.perf_counter() - t0
                else:
                    t_decode += time.perf_counter() - t0
                nxt = logits.argmax(-1)
                toks.append(nxt)
                if dump_at is not None and i == dump_at:
                    dumps = {"logits": logits.float().cpu(),
                             "hidden": h.float().cpu()}
        return torch.stack(toks, dim=1), t_prefill, t_decode, dumps

    # warmup (PG lazy init, kernel compile, allocator)
    gen(a.warmup_tokens)
    _AR_TIME["events"].clear()
    _AR_TIME["calls"] = 0
    torch.cuda.synchronize(dev)
    dist.barrier()
    t0 = time.perf_counter()
    out, t_prefill, t_decode, dumps = gen(a.gen_tokens, dump_at=a.dump_step)
    torch.cuda.synchronize(dev)
    dist.barrier()
    wall = time.perf_counter() - t0
    tps = a.gen_tokens / wall
    text = tok.decode(out[0], skip_special_tokens=True)

    # comm decomposition: GPU-timeline time inside all_reduce hooks
    ar_ms = sum(e0.elapsed_time(e1) for e0, e1 in _AR_TIME["events"])
    print("[qwen38-int8 %s rank=%d] %d tokens in %.2fs = %.2f tok/s "
          "(prompt %d tok, prefill %.1f ms, decode %.2f ms/token avg, "
          "all_reduce x %d = %.1f ms GPU-timeline)" %
          (a.backend, a.rank, a.gen_tokens, wall, tps, n_prompt,
           t_prefill * 1e3, t_decode * 1e3 / max(a.gen_tokens - 1, 1),
           _AR_TIME["calls"], ar_ms), flush=True)
    print("SAMPLE: %s" % text[:200].replace("\n", " "), flush=True)

    tag = a.tag or a.backend
    os.makedirs(a.out, exist_ok=True)
    if not a.no_comm:
        torch.save(out[0].cpu(), os.path.join(a.out, "genids_%s_rank%d.pt"
                                              % (tag, a.rank)))
        if dumps:
            torch.save(dumps, os.path.join(a.out, "dump_%s_rank%d.pt"
                                           % (tag, a.rank)))
    if a.rank == 0:
        with open(os.path.join(a.out, "stats_%s.json" % tag), "w") as f:
            json.dump({
                "model": "Qwen3.8-27B-INT8-W8A16-MTP", "backend": a.backend,
                "prompt": a.prompt, "prompt_tokens": n_prompt,
                "gen_tokens": a.gen_tokens, "warmup_tokens": a.warmup_tokens,
                "greedy": True, "tok_per_s": tps, "seconds": wall,
                "prefill_ms": t_prefill * 1e3,
                "decode_ms_per_token": t_decode * 1e3 / max(a.gen_tokens - 1, 1),
                "allreduce_calls": _AR_TIME["calls"],
                "allreduce_gpu_ms": ar_ms,
                "peak_mem_gib": mem, "load_s": load_s,
                "sample": text[:300],
            }, f, indent=2)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
