# BAR atomic feasibility probe: two processes (SPMD), each calls
# bl.bar_atomic_probe(iters): per iteration both ranks atom.add 1 to the
# PEER's probe slot and wait for their own to reach i+1.
#   res[0] = atomic exchange round trip, us/iter
#   res[1] = marker-flag exchange round trip (baseline), us/iter
#   res[2] = final local counter (must == iters)
#   res[3] = pre-armed launch-overhead baseline (mark BAR + wait), us/iter
#   res[4] = empty-kernel launch baseline, us/iter
#   res[5] = local-only mark baseline, us/iter
#   res[6] = BAR-mark only (no wait), us/iter
#   res[7] = pre-armed wait only (no mark), us/iter
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from barlink_sm86 import process_group as blpg
import barlink_sm86 as bl

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--iters", type=int, default=200)
a = ap.parse_args()

blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                        rank=a.rank, world_size=2)
# warm the protocol path first: keep both ranks busy with real allreduces
# so any lazy path state is identical to the pg_lat context
import torch as _t
_w = _t.ones(2560, dtype=_t.float32, device=_t.device("cuda", a.rank))
for _ in range(30):
    import torch.distributed as _d
    _d.all_reduce(_w)
_t.cuda.synchronize()
del _w
try:
    r = bl._C.bar_atomic_probe(a.iters)
except RuntimeError as e:
    print("rank %d: BAR ATOMIC PROBE FAILED: %s" % (a.rank, e), flush=True)
    sys.exit(1)
torch.cuda.synchronize()
ok = (r[2].item() == a.iters)
print("rank %d: atomic_rt=%.0f  flag_rt=%.0f  prearmed=%.0f  empty=%.0f  "
      "localmark=%.0f  barmark=%.0f  waitonly=%.0f  slot2mark=%.0f  "
      "proto=%.0f (us/iter)  final=%d/%d  %s"
      % (a.rank, r[0].item(), r[1].item(), r[3].item(), r[4].item(),
         r[5].item(), r[6].item(), r[7].item(), r[8].item(), r[9].item(),
         r[2].item(), a.iters, "OK" if ok else "CORRUPT"), flush=True)
import torch.distributed as dist
dist.destroy_process_group()
sys.exit(0 if ok else 1)
