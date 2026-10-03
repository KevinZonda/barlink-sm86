# SPDX-License-Identifier: MIT
#
# Qwen3.5-27B-GPTQ-Int4 TP decode benchmark over the barlink PG (vs NCCL).
#
#   --tp 1    whole text model on one card
#   --tp 2    tensor-parallel decode: both ranks shard every layer; rowwise
#             partial sums (o_proj / linear_attn.out_proj / mlp.down_proj)
#             are completed with dist.all_reduce over the barlink PG
#             (zero-copy) or NCCL per --backend.
#
# Why not stock GPTQ kernels: on this stack (torch 2.14 / py3.14 / sm86)
# every loader path is broken for this model -- the HF quantizer ignores the
# checkpoint's dynamic exclusions (converts attention to QuantLinear with
# MISSING weights), gptqmodel's native loader needs >20 GB on one card, and
# the CPU-selected torch_aten backend dequantizes per call. So this worker:
#   1. loads with transformers with modules_to_not_convert fixed (MLP -> int4
#      with REAL values; attention/linear-attn/lm_head/embed bf16)
#   2. int8-ifies every bf16 Linear (dynamic W8A8, torch._int_mm IMMA) and
#      the embedding, so the model fits 20 GB cards
#   3. dequantizes the int4 MLP once to bf16, (TP=2: shards at never-crossed
#      group boundaries) and repacks it via int4linear.Int4Linear, whose
#      Triton GEMV reads exactly the int4 bytes once (544 GB/s eff on a 3080)
# Quantization is per-(row,group) local, so a TP shard quantizes exactly the
# same values as TP=1; TP=1 vs TP=2 differ only by reduction reassociation.

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from int4linear import Int4Linear          # noqa: E402

MODEL_DIR = "/mnt/modelzoo/Qwen/Qwen3.5-27B-GPTQ-Int4"
PROMPT = ("Explain how GPU P2P communication over PCIe BAR1 works, "
          "in technical detail.")
GEN_TOKENS = 50
WARMUP_TOKENS = 10


# ---------------------------------------------------------------------------
# W8A8 dynamic int8 linear (replaces bf16 nn.Linear; keeps decode bandwidth)
# ---------------------------------------------------------------------------

class W8A8Linear(nn.Module):
    """Per-output-channel symmetric int8 weight + per-token dynamic int8
    activation, matmul via torch._int_mm (cublasLt IMMA, sm80+)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        wscale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        wq = torch.round(w / wscale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", wq.contiguous())
        self.register_buffer("wscale", wscale)

    def forward(self, x):
        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).float()
        xs = x2.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        xq = torch.round(x2 / xs[:, None]).to(torch.int8)
        if xq.shape[0] <= 16:  # torch._int_mm requires M > 16: zero-pad
            xq = torch.cat([xq, xq.new_zeros(32 - xq.shape[0], xq.shape[1])])
            ys = torch._int_mm(xq, self.qweight.t())[:x2.shape[0]]
        else:
            ys = torch._int_mm(xq, self.qweight.t())
        y = ys * (self.wscale * xs[:, None])
        return y.to(x.dtype).reshape(*shape, self.out_features)


class W8A8Embedding(nn.Module):
    """int8 embedding rows + per-row scale (lookup bandwidth halves)."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        w = emb.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer(
            "qweight",
            torch.round(w / scale).clamp(-127, 127).to(torch.int8).contiguous())
        self.register_buffer("scale", scale)
        self.num_embeddings = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim

    def forward(self, ids):
        rows = F.embedding(ids, self.qweight)
        sc = F.embedding(ids, self.scale)
        return (rows.float() * sc).to(torch.bfloat16)


def _int8ify(root):
    """Replace every bf16 nn.Linear / nn.Embedding under root in-place."""
    for name, mod in list(root.named_modules()):
        if not name:
            continue
        parent = root.get_submodule(name.rpartition(".")[0]) \
            if "." in name else root
        leaf = name.rpartition(".")[2]
        if isinstance(mod, nn.Linear):
            setattr(parent, leaf, W8A8Linear(mod))
        elif isinstance(mod, nn.Embedding):
            setattr(parent, leaf, W8A8Embedding(mod))


# ---------------------------------------------------------------------------
# TP=2 sharding
# ---------------------------------------------------------------------------

_AR_HOOKS = []


def _allreduce_hook(module, inp, out):
    dist.all_reduce(out)          # SUM over the world group (barlink/NCCL)
    return out


def _hook_rowwise(mod, world=2):
    if world > 1:
        _AR_HOOKS.append(mod.register_forward_hook(_allreduce_hook))


def _narrow(t, dim, rank, world=2):
    n = t.shape[dim]
    assert n % world == 0, (tuple(t.shape), dim)
    s = n // world
    return t.detach().narrow(dim, rank * s, s).contiguous()


def shard_w8_col(mod, rank, world=2):        # W8A8Linear: out dim
    mod.qweight = _narrow(mod.qweight, 0, rank, world)
    mod.wscale = _narrow(mod.wscale, 0, rank, world)
    mod.out_features //= world


def shard_w8_row(mod, rank, world=2):        # W8A8Linear: in dim + allreduce
    mod.qweight = _narrow(mod.qweight, 1, rank, world)
    mod.in_features //= world
    _hook_rowwise(mod, world)


def shard_model(model, rank, world=2, fetch=None):
    tc = model.config.text_config
    tc.num_attention_heads //= world
    tc.num_key_value_heads //= world
    tc.linear_num_key_heads //= world
    tc.linear_num_value_heads //= world

    for li, layer in enumerate(model.model.language_model.layers):
        if getattr(layer, "self_attn", None) is not None:
            a = layer.self_attn
            shard_w8_col(a.q_proj, rank, world)
            shard_w8_col(a.k_proj, rank, world)
            shard_w8_col(a.v_proj, rank, world)
            shard_w8_row(a.o_proj, rank, world)
        if getattr(layer, "linear_attn", None) is not None:
            la = layer.linear_attn
            kh = la.num_k_heads // world
            vh = la.num_v_heads // world
            kd, vd = kh * la.head_k_dim, vh * la.head_v_dim
            fkd, fvd = la.key_dim, la.value_dim

            def seg(w, r):    # [q | k | v] segments, sharded per head
                if world == 1:
                    return w
                return torch.cat([
                    w[:fkd].narrow(0, r * kd, kd),
                    w[fkd:2 * fkd].narrow(0, r * kd, kd),
                    w[2 * fkd:].narrow(0, r * vd, vd),
                ], dim=0).contiguous()

            la.in_proj_qkv.qweight = nn.Parameter(seg(la.in_proj_qkv.qweight,
                                                      rank),
                                                  requires_grad=False)
            la.in_proj_qkv.wscale = nn.Parameter(seg(la.in_proj_qkv.wscale,
                                                     rank),
                                                 requires_grad=False)
            la.in_proj_qkv.out_features = 2 * kd + vd
            shard_w8_col(la.in_proj_z, rank, world)      # per-head rows
            shard_w8_col(la.in_proj_b, rank, world)
            shard_w8_col(la.in_proj_a, rank, world)
            if li == 0:
                print("L0 z: out=%d qweight=%s b_out=%d" %
                      (la.in_proj_z.out_features,
                       tuple(la.in_proj_z.qweight.shape),
                       la.in_proj_b.out_features), flush=True)
            la.conv1d.weight = nn.Parameter(seg(la.conv1d.weight, rank),
                                            requires_grad=False)
            la.conv1d.in_channels = la.conv1d.out_channels = 2 * kd + vd
            la.conv1d.groups = 2 * kd + vd
            la.A_log = nn.Parameter(_narrow(la.A_log, 0, rank, world),
                                    requires_grad=False)
            la.dt_bias = nn.Parameter(_narrow(la.dt_bias, 0, rank, world),
                                      requires_grad=False)
            shard_w8_row(la.out_proj, rank, world)
            la.num_k_heads, la.num_v_heads = kh, vh
            la.key_dim, la.value_dim = kd, vd
            la.conv_dim = 2 * kd + vd
        # MLP: raw checkpoint GPTQ v1 tensors, sliced at never-crossed group
        # boundaries (an exact subset of the quantized values)
        for name in ("gate_proj", "up_proj", "down_proj"):
            base = "model.language_model.layers.%d.mlp.%s" % (li, name)
            qw = fetch(base + ".qweight")
            sc = fetch(base + ".scales")
            qz = fetch(base + ".qzeros")
            gi = fetch(base + ".g_idx")
            if name in ("gate_proj", "up_proj"):      # shard out features
                o2 = qw.shape[1] // world
                qw = qw[:, rank * o2:(rank + 1) * o2]
                sc = sc[:, rank * o2:(rank + 1) * o2]
                qz = qz[:, rank * o2 // 8:(rank + 1) * o2 // 8]
            else:                                      # shard in features
                i2 = qw.shape[0] * 8 // world
                assert i2 % 128 == 0
                qw = qw[rank * i2 // 8:(rank + 1) * i2 // 8, :]
                sc = sc[rank * i2 // 128:(rank + 1) * i2 // 128, :]
                qz = qz[rank * i2 // 128:(rank + 1) * i2 // 128, :]
                gi = gi[rank * i2:(rank + 1) * i2]
            m4 = Int4Linear.from_gptq_v1(qw, sc, qz, gi)
            if name == "down_proj" and world > 1:
                _hook_rowwise(m4, world)   # rowwise partial: reduce into full
            setattr(layer.mlp, name, m4)
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True, choices=[1, 2])
    ap.add_argument("--backend", default="barlink",
                    choices=["barlink", "nccl"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--warmup-tokens", type=int, default=WARMUP_TOKENS)
    ap.add_argument("--out", default=os.path.join(_HERE, "results"))
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    dist_rank = 0
    if a.tp == 2:
        if a.backend == "barlink":
            sys.path.insert(0, os.path.join(_HERE, "..", "..", "torch_ext"))
            from barlink_sm86 import process_group as blpg
            blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                                    rank=a.rank, world_size=2)
            dev = torch.device("cuda", a.rank)
        else:
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            dist.init_process_group("nccl", init_method="tcp://127.0.0.1:%d"
                                    % a.port, rank=a.rank, world_size=2)
        dist_rank = dist.get_rank()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration)
    from safetensors import safe_open

    # Fully manual load: the stock HF/optimum GPTQ paths are unusable on this
    # stack (see file header), so build the skeleton on the meta device and
    # pull exactly the text-model tensors from the checkpoint shards.
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg.quantization_config = None
    # hold the card while loading from host: the other local experiment
    # launches when it sees a free card, and its multi-GB load collides with
    # our .to(device) below; a temporary 15 GiB reservation blocks it
    hold = torch.empty(15 << 30, dtype=torch.uint8, device=dev)
    t0 = time.time()
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(cfg)

    index = json.load(open(os.path.join(MODEL_DIR,
                                        "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    handles = {}

    def fetch(name):
        path = weight_map.get(name)
        if path is None:
            return None
        h = handles.get(path)
        if h is None:
            h = safe_open(os.path.join(MODEL_DIR, path), framework="pt",
                          device="cpu")
            handles[path] = h
        return h.get_tensor(name)

    def keep(name):    # text-only: drop vision tower and MTP head
        return (not (name.startswith("model.visual")
                     or name.startswith("mtp."))
                and ".mlp." not in name)   # mlp arrives as int4; built below

    with torch.no_grad():
        for name, p in list(model.named_parameters()):
            if not keep(name):
                continue
            assert p.is_meta, name
            parent = model.get_submodule(name.rpartition(".")[0])
            t = fetch(name)
            assert t is not None, "missing checkpoint tensor " + name
            setattr(parent, name.rpartition(".")[2],
                    nn.Parameter(t, requires_grad=False))
    # skeleton modules we do not load (vision tower, MTP head) stay meta:
    # drop them so .to(device) does not trip over meta tensors
    if getattr(model.model, "visual", None) is not None:
        del model.model.visual
    if getattr(model, "mtp", None) is not None:
        del model.mtp
    load_s = time.time() - t0
    gc.collect()

    _int8ify(model.model.language_model)
    if isinstance(model.lm_head, nn.Linear):
        model.lm_head = W8A8Linear(model.lm_head)

    if a.tp == 2:
        index = json.load(open(os.path.join(MODEL_DIR,
                                            "model.safetensors.index.json")))
        weight_map = index["weight_map"]

        def fetch2(name):
            path = weight_map.get(name)
            if path is None:
                return None
            h = handles.get(path)
            if h is None:
                h = safe_open(os.path.join(MODEL_DIR, path), framework="pt",
                              device="cpu")
                handles[path] = h
            return h.get_tensor(name)

        model = shard_model(model, dist_rank, fetch=fetch2)
    else:
        model = shard_model(model, 0, world=1, fetch=fetch)
    m = model.model.language_model.layers[0].mlp.gate_proj
    if dist_rank == 0:
        print("gate_proj: %s qweight=%s scales=%s" %
              (type(m).__name__, tuple(m.qweight.shape),
               tuple(m.scales.shape)), flush=True)
    meta_left = [n for n, p in model.named_parameters() if p.is_meta] + \
        [n for n, b in model.named_buffers() if b.is_meta]
    if meta_left:
        # rotary inv_freq buffers are checkpointless (config-derived): rebuild
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding)
        lm = model.model.language_model
        lm.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg.text_config)
        meta_left = [n for n, p in model.named_parameters() if p.is_meta] + \
            [n for n, b in model.named_buffers() if b.is_meta]
    if meta_left:
        print("meta leftovers:", meta_left[:10], flush=True)
        raise SystemExit(1)
    del hold
    gc.collect()
    torch.cuda.empty_cache()
    model = model.to(dev).eval()
    gc.collect()
    mem = torch.cuda.max_memory_allocated(dev) / (1 << 30)
    print("[tp=%d %s rank=%d] load=%.0fs peak=%.1f GiB" %
          (a.tp, a.backend, dist_rank, load_s, mem), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(a.prompt, return_tensors="pt").input_ids.to(dev)
    print("prompt tokens:", ids.shape[1], flush=True)

    def prefill(n):
        with torch.inference_mode():
            o = model(input_ids=ids[:, :n], use_cache=True)
            return o.past_key_values, o.logits[:, -1].argmax(dim=-1,
                                                            keepdim=True)

    past, cur = prefill(ids.shape[1])        # untimed prefill
    # warmup decode steps (triton compile, PG first call)
    with torch.inference_mode():
        for _ in range(a.warmup_tokens):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(dev)
    if os.environ.get("QWEN_PROFILE"):
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            with torch.inference_mode():
                for _ in range(5):
                    o = model(input_ids=cur, past_key_values=past,
                              use_cache=True)
                    past = o.past_key_values
                    cur = o.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize(dev)
        print(prof.key_averages().table(sort_by="cuda_time_total",
                                        row_limit=18), flush=True)
    t0 = time.perf_counter()
    gen_ids = []
    with torch.inference_mode():
        for _ in range(a.gen_tokens):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[:, -1].argmax(dim=-1, keepdim=True)
            gen_ids.append(cur.item())
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    tps = a.gen_tokens / dt
    text = tok.decode(gen_ids, skip_special_tokens=True)

    print("[tp=%d %s rank=%d] %d tokens in %.2fs = %.2f tok/s"
          % (a.tp, a.backend, dist_rank, a.gen_tokens, dt, tps), flush=True)
    print("SAMPLE: %s" % text[:200].replace("\n", " "), flush=True)

    tag = a.tag or "tp%d%s" % (a.tp, "" if a.backend == "barlink"
                               else "_" + a.backend)
    if a.tp == 1 or dist_rank == 0 or os.environ.get("QWEN_SAVE_ALL_RANKS"):
        os.makedirs(a.out, exist_ok=True)
        stag = tag if (a.tp == 1 or dist_rank == 0) else "%s_rank%d" % (tag,
                                                                        dist_rank)
        torch.save(torch.tensor(gen_ids, dtype=torch.long),
                   os.path.join(a.out, "genids_%s.pt" % stag))
        if a.tp == 1 or dist_rank == 0:
            with open(os.path.join(a.out, "stats_%s.json" % tag), "w") as f:
                json.dump({
                    "tp": a.tp, "backend": a.backend if a.tp == 2 else None,
                    "prompt": a.prompt, "prompt_tokens": ids.shape[1],
                    "gen_tokens": a.gen_tokens,
                    "warmup_tokens": a.warmup_tokens,
                    "greedy": True, "tok_per_s": tps,
                    "seconds": dt, "peak_mem_gib": mem, "load_s": load_s,
                    "sample": text[:300],
                }, f, indent=2)

    if a.tp == 2:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
