# SPDX-License-Identifier: MIT
#
# NCCL baseline for the same small-message allreduce sizes, on the same
# two cards. With 256 MiB BAR1, NCCL cannot use GPU P2P and falls back to
# host-memory (SHM) transport -- this is what stock vLLM would use today.
#
# Run from repo root:  .venv/bin/python bench/latency/nccl_lat.py
# (no barlink needed, no root needed)

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [4096, 16384, 65536, 262144, 1048576, 4194304]
PORT = 29517


def worker(rank, world_size, result_q):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            init_method="tcp://127.0.0.1:%d" % PORT)
    out = []
    for n in SIZES:
        t = torch.ones(n, dtype=torch.uint8, device="cuda:%d" % rank)
        for _ in range(20):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        iters = 200
        lat = 0.0
        for _ in range(iters):
            t0 = time.perf_counter()
            dist.all_reduce(t)
            torch.cuda.synchronize()
            lat += time.perf_counter() - t0
        lat /= iters
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        thr = (time.perf_counter() - t0) / iters
        out.append((n, lat, thr))
    result_q.put((rank, out))
    dist.destroy_process_group()


def main():
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    results.sort()
    rank0 = results[0][1]
    print("%10s  %12s  %12s" % ("size", "ar us/op", "ar GB/s(thr)"))
    for n, lat, thr in rank0:
        print("%10d  %12.1f  %12.2f" % (n, lat * 1e6, n / thr / 1e9))


if __name__ == "__main__":
    main()
