# quick single-process smoke: load + shard + one prefill/decode forward
# (world=1 gloo all_reduce only completes shapes, numerics are NOT valid)
import os, sys, time
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker as W

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29555",
                        rank=0, world_size=1)
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
t0 = time.time()
lm, lm_head, tc = W.load_and_shard(dev, 0, verbose=True)
print("load+shard %.0fs, peak %.1f GiB" % (time.time() - t0,
      torch.cuda.max_memory_allocated(dev) / 2**30))
a = lm.layers[3].self_attn
print("q_proj", tuple(a.q_proj.qweight.shape), "o_proj", tuple(a.o_proj.qweight.shape))
la = lm.layers[0].linear_attn
print("in_proj_qkv", tuple(la.in_proj_qkv.qweight.shape),
      "out_proj", tuple(la.out_proj.qweight.shape),
      "conv1d", tuple(la.conv1d.weight.shape), "A_log", tuple(la.A_log.shape))
print("gate", tuple(lm.layers[0].mlp.gate_proj.qweight.shape),
      "down", tuple(lm.layers[0].mlp.down_proj.qweight.shape))
print("lm_head", tuple(lm_head.weight.shape))

ids = torch.randint(0, tc.vocab_size, (1, 32), device=dev)
with torch.inference_mode():
    out = lm(input_ids=ids, use_cache=True)
    h = out.last_hidden_state[:, -1]
    print("prefill OK, hidden", tuple(h.shape), "finite:",
          bool(torch.isfinite(h).all()))
    out2 = lm(input_ids=ids[:, :1], past_key_values=out.past_key_values,
              use_cache=True)
    h2 = out2.last_hidden_state[:, -1]
    print("decode OK, finite:", bool(torch.isfinite(h2).all()))
print("SMOKE OK")
