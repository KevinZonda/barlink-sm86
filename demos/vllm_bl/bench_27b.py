# vLLM TP=2 decode TPS bench for Qwen3.8-27B INT8 W8A16 (barlink PG vs NCCL).
#   barlink:  BL_SKIP_INIT=1 BL_POOL_MB=192 HF_HUB_OFFLINE=1 tools/blrun \
#                 vllm_barlink/entry.py demos/vllm_bl/bench_27b.py
#   nccl:     same but BL_SHIM_OFF=1 (shim no-ops -> vllm's real NCCL) and run
#             the venv python directly (no caps needed):
#             .venv-vllm/bin/python demos/vllm_bl/bench_27b.py
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    MODEL = "/mnt/modelzoo/lued/Qwen3.8-27B-INT8-W8A16-MTP"
    PROMPTS = [
        "Explain how PCIe peer-to-peer DMA over a GPU BAR1 aperture works, in "
        "technical detail, covering TLP routing and IOMMU translation.",
        "Write a long essay about the history of GPU computing, from "
        "fixed-function pipelines to modern tensor-core architectures.",
        "Describe the design of a distributed key-value store with strong "
        "consistency, including the consensus protocol and failure handling.",
        "Give a detailed technical comparison of RDMA, GPUDirect, and PCIe "
        "BAR based inter-GPU communication mechanisms.",
    ]

    llm = LLM(model=MODEL,
              tensor_parallel_size=2,
              enforce_eager=True,
              gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len,
              max_num_seqs=a.max_num_seqs)

    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    wall = time.perf_counter() - t0

    n_prompts = len(PROMPTS)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tps = gen_tokens / wall
    first = outs[0].outputs[0].text[:160].replace("\n", " ")
    print("BENCH RESULT backend_env_shim_off=%s prompts=%d gen_tokens=%d "
          "wall=%.2fs tps=%.2f" % (os.environ.get("BL_SHIM_OFF"), n_prompts,
                                   gen_tokens, wall, tps), flush=True)
    print("SAMPLE: %s" % first, flush=True)
    tag = "nccl" if os.environ.get("BL_SHIM_OFF") == "1" else "barlink"
    with open("demos/vllm_bl/bench27b_%s.json" % tag, "w") as f:
        json.dump({"backend": tag, "shim_off": os.environ.get("BL_SHIM_OFF"),
                   "prompts": n_prompts, "gen_tokens": gen_tokens,
                   "wall_s": wall, "tps": tps, "enforce_eager": True,
                   "sample": first}, f, indent=2)


if __name__ == "__main__":
    main()
