# Needle-in-haystack v2: diverse filler + chat template.
# The v1 failure (repeated-sentence haystack, raw completion) was a harness
# artifact -- the model continued the pattern instead of answering.
import os

MODEL = os.environ.get(
    "BENCH_MODEL", "/mnt/modelzoo/MIRALABS/Qwen3.8-27B-W4A16-AutoRound")
DEPTH_FRAC = float(os.environ.get("NEEDLE_DEPTH", "0.5"))
HAYSTACK_WORDS = int(os.environ.get("NEEDLE_HAYSTACK", "80000"))

NEEDLE = ("The special authorization code for the PCIe BAR project "
          "is ZETA-7741-QUANTUM.")
QUESTION = ("Based on the document above, what is the special authorization "
            "code for the PCIe BAR project? Reply with just the code.")


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    units = [
        "System log entry: DMA transfer between host memory and device "
        "buffer completed within nominal latency bounds.",
        "Quarterly maintenance report: cooling loop pressure nominal, "
        "fan arrays operating at 42 percent duty cycle.",
        "Meeting notes: the storage migration window is scheduled for "
        "next Thursday between 02:00 and 04:00 local time.",
        "Shipping manifest: forty two rack units departed the warehouse, "
        "tracking numbers attached to the purchase order.",
        "Field observation: the prototype sensor array recorded stable "
        "readings across all twelve channels overnight.",
        "Budget summary: quarterly operational expenses came in 3 percent "
        "under the projected envelope.",
        "Travel itinerary: departure from gate C7, connecting flight "
        "boards ninety minutes after arrival.",
        "Lab journal: the calibration run completed after seven "
        "iterations with residuals below threshold.",
        "Incident review: the network partition healed automatically "
        "once the spanning tree reconverged.",
        "Inventory count: the spare parts bin contains eight power "
        "supplies and three controller boards.",
    ]
    words = []
    i = 0
    while len(words) < HAYSTACK_WORDS:
        words.extend(units[i % len(units)].split())
        i += 1
    words = words[:HAYSTACK_WORDS]
    pos = int(len(words) * DEPTH_FRAC)
    doc = " ".join(words[:pos]) + " " + NEEDLE + " " + " ".join(words[pos:])

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": doc + "\n\n" + QUESTION}],
        tokenize=False, add_generation_prompt=True)

    llm = LLM(model=MODEL, tensor_parallel_size=2,
              gpu_memory_utilization=0.93, max_model_len=262144,
              max_num_seqs=1, kv_cache_dtype="fp8_e4m3",
              enable_chunked_prefill=True)
    sp = SamplingParams(temperature=0, max_tokens=32)
    out = llm.generate([prompt], sp)[0].outputs[0].text
    ok = "ZETA-7741-QUANTUM" in out
    print("NEEDLE2 RESULT depth=%.2f haystack_words=%d ok=%s" %
          (DEPTH_FRAC, HAYSTACK_WORDS, ok), flush=True)
    print("ANSWER: %s" % out.strip()[:200], flush=True)


if __name__ == "__main__":
    main()
