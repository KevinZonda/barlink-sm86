# vLLM TP=2 smoke test over the barlink PG (offline LLM API, Qwen3-1.7B).
# CUDA graph capture is ON by default; set ENFORCE_EAGER=1 to fall back.
import os
enforce = os.environ.get("ENFORCE_EAGER") == "1"
from vllm import LLM, SamplingParams
llm = LLM(model="/mnt/modelzoo/Qwen/Qwen2-1.5B",
          tensor_parallel_size=2,
          enforce_eager=enforce,
          gpu_memory_utilization=0.42,
          max_model_len=512)
out = llm.generate(["Hello, my name is"],
                   SamplingParams(max_tokens=32, temperature=0))
print("SMOKE OUTPUT: %r" % out[0].outputs[0].text, flush=True)
