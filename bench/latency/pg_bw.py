# SPDX-License-Identifier: MIT
#
# PG-layer large-message allreduce bandwidth (the DiT path: tens of MB per
# op). Two processes via tests/run_pg.sh-style launch. Reports effective
# payload bandwidth = bytes / time (one direction over the wire per rank).

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "torch_ext"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from barlink_sm86 import process_group as blpg  # noqa: E402

SIZES_MB = [4, 16, 32, 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()

    blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                            rank=a.rank, world_size=2)
    dev = torch.device("cuda", a.rank)

    for mb in SIZES_MB:
        n = mb * (1 << 20) // 4
        t = torch.ones(n, dtype=torch.float32, device=dev)
        for _ in range(5):
            dist.all_reduce(t)
        torch.cuda.synchronize(dev)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(20):
            dist.all_reduce(t)
        e1.record()
        torch.cuda.synchronize(dev)
        ms = e0.elapsed_time(e1) / 20
        gbs = mb / 1024 / (ms / 1000)
        print("rank %d: PG all_reduce %d MB: %.2f ms/op, %.2f GB/s eff"
              % (a.rank, mb, ms, gbs), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
