# soak test for the fused small-message allreduce: many back-to-back 10KB
# ops with periodic bitwise verification, plus a fused/old (256KB) size
# interleave to exercise the zone separation under mixing. Both ranks use
# the same per-block seed, so the expected result is exactly 2*input.
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.distributed as dist
from barlink_sm86 import process_group as blpg

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--iters", type=int, default=1000)
a = ap.parse_args()

blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                        rank=a.rank, world_size=2)
dev = torch.device("cuda", a.rank)


def make(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2560, generator=g, dtype=torch.float32).to(dev)


t = make(1000)
orig = t.clone()
for i in range(a.iters):
    t.copy_(orig)
    dist.all_reduce(t)
    if i % 100 == 99:
        torch.cuda.synchronize()
        if not torch.equal(t.cpu(), (2 * orig).cpu()):
            bad = (t.cpu() != (2 * orig).cpu()).sum().item()
            print("rank %d SOAK FAIL iter %d: %d/%d elems differ"
                  % (a.rank, i, bad, t.numel()), flush=True)
            sys.exit(1)
        print("rank %d: soak iter %d OK" % (a.rank, i), flush=True)
        orig = make(1000 + i + 1)

# fused/old interleave: 10KB (fused) and 256KB (old path) alternating
big = make(77).repeat(20)[:65536].contiguous()     # 256KB fp32
big_orig = big.clone()
for i in range(50):
    t.copy_(orig)
    dist.all_reduce(t)            # fused path
    big.copy_(big_orig)
    dist.all_reduce(big)          # old arm path
torch.cuda.synchronize()
assert torch.equal(t.cpu(), (2 * orig).cpu())
assert torch.equal(big.cpu(), (2 * big_orig).cpu())
print("rank %d: fused/old interleave OK" % a.rank, flush=True)
dist.barrier()
dist.destroy_process_group()
print("rank %d: SOAK PASSED" % a.rank, flush=True)
