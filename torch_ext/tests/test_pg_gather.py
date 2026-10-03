# all_gather + size-1 subgroup tests for the barlink PG.
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.distributed as dist
from barlink_sm86 import process_group as blpg

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--port", type=int, required=True)
a = ap.parse_args()

blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                        rank=a.rank, world_size=2)
dev = torch.device("cuda", a.rank)
peer = 1 - a.rank
ok = True

def check(cond, tag):
    global ok
    print("rank %d: %s %s" % (a.rank, tag, "OK" if cond else "FAIL"), flush=True)
    ok = ok and cond

# 1. all_gather_into_tensor (Backend.all_gather_single)
for n in (2560, 4096, 999):           # 10KB, 16KB, odd (staged)
    inp = torch.full((n,), float(a.rank + 1), device=dev)
    out = torch.empty(2 * n, device=dev)
    dist.all_gather_into_tensor(out, inp)
    want = torch.cat([torch.full((n,), 1.0), torch.full((n,), 2.0)]).to(dev)
    check(torch.equal(out, want), "all_gather_into_tensor n=%d" % n)

# 2. dist.all_gather list form (Backend.allgather)
inp = torch.full((512,), float(a.rank + 1), device=dev)
lst = [torch.empty(512, device=dev) for _ in range(2)]
dist.all_gather(lst, inp)
check(torch.equal(lst[0], torch.ones(512, device=dev)) and
      torch.equal(lst[1], torch.full((512,), 2.0, device=dev)),
      "dist.all_gather list")

# 3. size-1 subgroup: honest local semantics, no link traffic
grp = dist.new_group([a.rank])
x = torch.full((128,), 7.0, device=dev)
dist.all_reduce(x, group=grp)
check(torch.equal(x, torch.full((128,), 7.0, device=dev)),
      "size-1 subgroup all_reduce identity")
o = torch.empty(128, device=dev)
dist.all_gather_into_tensor(o, x, group=grp)
check(torch.equal(o, x), "size-1 subgroup all_gather identity")
dist.barrier(group=grp)
check(True, "size-1 subgroup barrier")

# 4. world-2 traffic still works after the subgroup detour
t = torch.ones(2560, device=dev)
dist.all_reduce(t)
check(torch.equal(t, torch.full((2560,), 2.0, device=dev)),
      "world-2 all_reduce after subgroup")

dist.barrier()
dist.destroy_process_group()
print("rank %d: %s" % (a.rank, "ALL GATHER TESTS PASSED" if ok else "FAILED"),
      flush=True)
sys.exit(0 if ok else 1)
