# SPDX-License-Identifier: MIT
#
# PG-layer small-message allreduce latency (the 91 us @10 KB line of
# BUILD_AND_TEST.md §7). Two processes via run_pg.sh-style launch
# (tests/run_lat.sh). Methodology matches test_process_group.py §11:
# 10 warmup + 50 CUDA-event-timed back-to-back dist.all_reduce calls,
# plus a per-op-sync latency number.

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "torch_ext"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from barlink_sm86 import process_group as blpg  # noqa: E402

SIZES = [2560, 16384, 65536, 262144]     # 10 KB, 64 KB, 256 KB, 1 MB fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()

    blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                            rank=a.rank, world_size=2)
    dev = torch.device("cuda", a.rank)

    out = []
    for n in SIZES:
        t = torch.ones(n, dtype=torch.float32, device=dev)
        for _ in range(10):
            dist.all_reduce(t)
        torch.cuda.synchronize(dev)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(50):
            dist.all_reduce(t)
        e1.record()
        torch.cuda.synchronize(dev)
        ev_us = e0.elapsed_time(e1) * 1000.0 / 50

        lat = 0.0
        for _ in range(50):
            t0 = time.perf_counter()
            dist.all_reduce(t)
            torch.cuda.synchronize(dev)
            lat += time.perf_counter() - t0
        lat_us = lat * 1e6 / 50
        out.append((n * 4, ev_us, lat_us))
        print("rank %d: PG all_reduce %d B: %.1f us (event), %.1f us (sync)"
              % (a.rank, n * 4, ev_us, lat_us), flush=True)

    if a.rank == 0:
        print("SUMMARY size event_us sync_us")
        for b, e, s in out:
            print("SUMMARY %d %.1f %.1f" % (b, e, s), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
