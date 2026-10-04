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
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help=">0: single synthetic prompt of ~N tokens "
                         "(overrides the fixed multi-prompt set)")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--warm-repeat", type=int, default=0,
                    help=">0: run the same request N extra times (prefix cache "
                         "hit -> wall is decode-dominated); reports min wall")
    ap.add_argument("--mtp", type=int, default=0,
                    help=">0: enable MTP speculative decoding with N "
                         "speculative tokens (method='mtp')")
    ap.add_argument("--kv-dtype", type=str, default="",
                    help="e.g. fp8_e4m3 -> kv_cache_dtype")
    ap.add_argument("--chunked-prefill", action="store_true")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    MODEL = os.environ.get("BENCH_MODEL", "/mnt/modelzoo/lued/Qwen3.8-27B-INT8-W8A16-MTP")
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
    if a.prompt_tokens > 0:
        # ~1 token/word: repetitive technical text repeated to the target len.
        unit = ("The system uses peer to peer direct memory access between "
                "graphics processors over the peripheral component "
                "interconnect express bus, bypassing host memory entirely. ")
        reps = a.prompt_tokens // len(unit.split()) + 1
        PROMPTS = [" ".join((unit * reps).split()[:a.prompt_tokens])]

    llm_kwargs = dict(
        model=MODEL,
        tensor_parallel_size=2,
        enforce_eager=os.environ.get("ENFORCE_EAGER") == "1",
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=a.max_model_len,
        max_num_seqs=a.max_num_seqs,
    )
    if a.mtp > 0:
        llm_kwargs["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": a.mtp}
    if a.kv_dtype:
        llm_kwargs["kv_cache_dtype"] = a.kv_dtype
    if a.chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    wall = time.perf_counter() - t0

    n_prompts = len(PROMPTS)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tps = gen_tokens / wall
    # decode-only TPS from per-request first/last token timestamps
    dec = []
    for o in outs:
        m = getattr(o, "metrics", None)
        ft, lt = getattr(m, "first_token_time", None), getattr(m, "last_token_time", None)
        n = len(o.outputs[0].token_ids)
        if ft and lt and n > 1 and lt > ft:
            dec.append((n - 1) / (lt - ft))
    decode_tps = sum(dec) / len(dec) if dec else None
    prefill_s = None
    if dec:
        m0 = outs[0].metrics
        arr = getattr(m0, "arrival_time", None)
        if arr and m0.first_token_time:
            prefill_s = m0.first_token_time - arr
    # prefix-cache re-run: wall is decode-dominated
    warm_wall = None
    if a.warm_repeat > 0:
        for _ in range(a.warm_repeat):
            tw0 = time.perf_counter()
            outs2 = llm.generate(PROMPTS, sp)
            tw = time.perf_counter() - tw0
            warm_wall = tw if warm_wall is None else min(warm_wall, tw)
        gen2 = sum(len(o.outputs[0].token_ids) for o in outs2)
        if decode_tps is None and gen2 == gen_tokens and warm_wall > 0:
            decode_tps = gen_tokens / warm_wall
            if prefill_s is None:
                prefill_s = wall - warm_wall
    # speculative verify counts (acceptance diagnostics, v1 RequestOutput)
    svc = [getattr(o, "spec_verify_ct", None) for o in outs]
    svc = [x for x in svc if x]
    spec = {"mtp": a.mtp, "verify_ct": svc[0] if len(svc) == 1 else svc or None}
    if svc and gen_tokens:
        # mean accepted tokens per verify step = gen_tokens / verify_ct
        spec["mean_accept"] = gen_tokens / svc[0] if len(svc) == 1 else None
    first = outs[0].outputs[0].text[:160].replace("\n", " ")
    print("BENCH RESULT backend_env_shim_off=%s prompts=%d gen_tokens=%d "
          "wall=%.2fs tps=%.2f decode_tps=%s prefill_s=%s warm_wall=%s "
          "spec=%s" %
          (os.environ.get("BL_SHIM_OFF"), n_prompts,
           gen_tokens, wall, tps, decode_tps, prefill_s,
           "%.2f" % warm_wall if warm_wall else None, spec), flush=True)
    print("SAMPLE: %s" % first, flush=True)
    tag = "nccl" if os.environ.get("BL_SHIM_OFF") == "1" else "barlink"
    out = a.out_json or ("demos/vllm_bl/bench27b_%s.json" % tag)
    with open(out, "w") as f:
        json.dump({"backend": tag, "shim_off": os.environ.get("BL_SHIM_OFF"),
                   "prompt_tokens": a.prompt_tokens, "prompts": n_prompts,
                   "gen_tokens": gen_tokens,
                   "wall_s": wall, "tps": tps, "decode_tps": decode_tps,
                   "prefill_s": prefill_s, "warm_wall_s": warm_wall,
                   "spec": spec,
                   "enforce_eager": os.environ.get("ENFORCE_EAGER") == "1",
                   "sample": first}, f, indent=2)


if __name__ == "__main__":
    main()
