# profile one decode step: kernel counts + cuda time by op type
import os, sys
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker as W

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29557",
                        rank=0, world_size=1)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
lm, lm_head, tc = W.load_and_shard(dev, 0)
ids = torch.randint(0, tc.vocab_size, (1, 48), device=dev)
W._NO_COMM = True

with torch.inference_mode():
    out = lm(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    inp = ids[:, :1]
    for _ in range(3):   # warm the decode path
        o = lm(input_ids=inp, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(5):
            o = lm(input_ids=inp, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
        torch.cuda.synchronize()

ev = prof.key_averages()
rows = [(e.count, e.device_time_total, e.key) for e in ev
        if e.device_time_total > 0]
rows.sort(key=lambda r: -r[1])
tot_t = sum(r[1] for r in rows)
tot_c = sum(e.count for e in ev if e.count and 'Memcpy' not in e.key)
print("total CUDA kernels (with cuda time): %d, device time %.1f ms / 5 steps"
      % (sum(r[0] for r in rows), tot_t / 1000))
print("per-step: %.1f kernels, %.2f ms device" %
      (sum(r[0] for r in rows) / 5, tot_t / 5000))
print("%8s %10s  %s" % ("count", "cuda_ms", "kernel"))
for c, t, k in rows[:22]:
    print("%8d %10.2f  %s" % (c / 5, t / 5000, k[:90]))
