# SPDX-License-Identifier: MIT
#
# FLUX.2 Klein 9B DiT tensor-parallel benchmark over the barlink ProcessGroup.
#
#   --tp 1  single process, full bf16 model on one GPU (baseline)
#   --tp 2  two processes (one GPU each, orchestrated by run.sh); the model is
#           sharded with diffusers' built-in Flux2 `_tp_plan` via
#           `model.enable_parallelism()` (torch parallelize_module under the
#           hood); RowwiseParallel's Partial->Replicate redistribute issues
#           dist.all_reduce on the barlink PG, i.e. over the P2P BAR1 link.
#
# Both modes build identical seeded inputs; rank 0 saves the bf16->fp32 output
# plus per-step timings for the compare step.

import argparse
import gc
import json
import os
import sys
import time

import torch

MODEL_DIR = "/mnt/modelzoo/black-forest-labs/FLUX.2-klein-base-9B/transformer"
JOINT_DIM = 12288   # config joint_attention_dim (Qwen3 text context)
IN_CH = 128         # config in_channels, patch_size=1


def build_inputs(seed, img_tokens, txt_tokens, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn((1, img_tokens, IN_CH), generator=g,
                         dtype=torch.float32).to(dtype)
    txt = torch.randn((1, txt_tokens, JOINT_DIM), generator=g,
                      dtype=torch.float32).to(dtype)
    side = int(img_tokens ** 0.5)
    assert side * side == img_tokens, "img_tokens must be a square"
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(side),
                                   torch.arange(side), torch.arange(1)).unsqueeze(0)
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                   torch.arange(txt_tokens)).unsqueeze(0)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    return hidden, txt, img_ids, txt_ids, timestep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True, choices=[1, 2])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--img-tokens", type=int, default=4096)
    ap.add_argument("--txt-tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--tag", default=None,
                    help="output filename tag (default: tp<N>)")
    ap.add_argument("--load-delay", type=float, default=0.0,
                    help="seconds to wait before loading (staggers host RAM "
                         "when both ranks load a large dtype)")
    ap.add_argument("--compute-fp32", action="store_true",
                    help="cast the (sharded) bf16 params to fp32 and compute "
                         "in fp32: an anchor that removes activation/reduce "
                         "rounding; TP=2 only (TP=1 would need 36 GiB)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_dtype = getattr(torch, a.dtype)
    if a.load_delay > 0:
        time.sleep(a.load_delay)

    dev = torch.device("cuda:0")
    dist_rank = 0
    if a.tp == 2:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "torch_ext"))
        from barlink_sm86 import process_group as blpg
        import torch.distributed as dist
        blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                                rank=a.rank, world_size=2)
        dist_rank = dist.get_rank()
        # Card selection follows run_pg.sh: no CUDA_VISIBLE_DEVICES, the
        # rank picks the card (LOCAL_RANK is read by the PG as well).
        dev = torch.device("cuda", a.rank)

    from diffusers import Flux2Transformer2DModel

    t0 = time.time()
    model = Flux2Transformer2DModel.from_pretrained(MODEL_DIR,
                                                    torch_dtype=model_dtype)
    load_s = time.time() - t0
    model = model.to(dev).eval()
    gc.collect()

    tp_ms = None
    if a.tp == 2:
        t0 = time.time()
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import parallelize_module
        from diffusers.hooks.tensor_parallel import _resolve_tp_plan, _styles

        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        # Same as model.enable_parallelism(TensorParallelConfig(tp_degree=2))
        # — the Flux2 _tp_plan via torch parallelize_module — except
        # src_data_rank=None: both ranks load the identical full checkpoint,
        # so sharding is a purely local split. The API default
        # src_data_rank=0 would issue a mesh scatter/broadcast during
        # partitioning, which the barlink backend (v1: no send/recv) cannot
        # serve. RowwisePartial->Replicate at runtime still allreduces over
        # the barlink PG as usual.
        for submodule, relative_plan in _resolve_tp_plan(model, model._tp_plan):
            parallelize_module(submodule, mesh, _styles(relative_plan),
                               src_data_rank=None)
        tp_ms = (time.time() - t0) * 1000.0

    if a.compute_fp32:
        if a.tp != 2:
            raise SystemExit("--compute-fp32 is TP=2 only (TP=1 needs 36 GiB)")
        # Module.to (not p.data = ...) — DTensor params need Module._apply's
        # subclass-aware path for the cast to actually stick.
        model.to(dtype=torch.float32)
        model_dtype = torch.float32

    hidden, txt, img_ids, txt_ids, timestep = [
        t.to(dev) for t in build_inputs(a.seed, a.img_tokens, a.txt_tokens,
                                        dtype=model_dtype)]

    def step():
        with torch.inference_mode():
            return model(hidden_states=hidden,
                         encoder_hidden_states=txt,
                         timestep=timestep,
                         img_ids=img_ids,
                         txt_ids=txt_ids,
                         guidance=None,
                         return_dict=False)[0]

    for _ in range(a.warmup):
        out = step()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(a.steps):
        start.record()
        out = step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mem = torch.cuda.max_memory_allocated(dev)
    print("[tp=%d rank=%d] load=%.1fs shard=%s | %d steps: mean=%.1f ms/step "
          "min=%.1f max=%.1f | peak_mem=%.1f GiB"
          % (a.tp, dist_rank, load_s,
             "%.0f ms" % tp_ms if tp_ms is not None else "n/a",
             a.steps, sum(times) / len(times), min(times), max(times),
             mem / (1 << 30)), flush=True)

    tag = a.tag or ("tp%d" % a.tp)
    if dist_rank == 0:
        os.makedirs(a.out, exist_ok=True)
        torch.save(out.float().cpu(), os.path.join(a.out, "output_%s.pt" % tag))
        with open(os.path.join(a.out, "stats_%s.json" % tag), "w") as f:
            json.dump({
                "tp": a.tp,
                "dtype": a.dtype,
                "img_tokens": a.img_tokens,
                "txt_tokens": a.txt_tokens,
                "steps": a.steps,
                "warmup": a.warmup,
                "seed": a.seed,
                "load_s": load_s,
                "shard_ms": tp_ms,
                "mean_ms_per_step": sum(times) / len(times),
                "min_ms_per_step": min(times),
                "max_ms_per_step": max(times),
                "times_ms": times,
                "peak_mem_gib": mem / (1 << 30),
            }, f, indent=2)

    if a.tp == 2:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
