# SPDX-License-Identifier: MIT
#
# Bidirectional concurrent bandwidth probe: two threads copy in opposite
# directions at the same time (each with its own stream pair + own pool
# tensors), versus the one-direction baseline.
#
# Run: tools/blrun torch_ext/tests/bench_bidir.py

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def sync_both():
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def uni(nbytes, iters):
    a = bl.empty(nbytes, device=0)
    b = bl.empty(nbytes, device=1)
    a.random_(0, 256)
    for _ in range(20):          # warmup
        bl.copy_(b, a)
    sync_both()
    t0 = time.perf_counter()
    for _ in range(iters):
        bl.copy_(b, a)
    sync_both()
    return nbytes * iters / (time.perf_counter() - t0) / 1e9


def bidir(nbytes, iters):
    # T1: dev0 -> dev1 (b <- a); T2: dev1 -> dev0 (c <- d). Separate pools
    # per direction, so the two copy_ chains never share a flag region.
    a = bl.empty(nbytes, device=0)
    b = bl.empty(nbytes, device=1)
    d = bl.empty(nbytes, device=1)
    c = bl.empty(nbytes, device=0)
    a.random_(0, 256)
    d.random_(0, 256)
    results = [0.0, 0.0]
    barrier = threading.Barrier(2)

    def worker(wi, src_dev, dst_dev, src_t, dst_t):
        s_src = torch.cuda.Stream(src_dev)
        s_dst = torch.cuda.Stream(dst_dev)
        for _ in range(20):      # warmup, also forces lazy ctx/stream init
            with torch.cuda.stream(s_src):
                with torch.cuda.stream(s_dst):
                    bl.copy_(dst_t, src_t)
        torch.cuda.synchronize(src_dev)
        torch.cuda.synchronize(dst_dev)
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.cuda.stream(s_src):
                with torch.cuda.stream(s_dst):
                    bl.copy_(dst_t, src_t)
        torch.cuda.synchronize(src_dev)
        torch.cuda.synchronize(dst_dev)
        results[wi] = nbytes * iters / (time.perf_counter() - t0) / 1e9

    t1 = threading.Thread(target=worker, args=(0, 0, 1, a, b))
    t2 = threading.Thread(target=worker, args=(1, 1, 0, d, c))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results[0], results[1]


def main():
    global bl
    import barlink_sm86 as bl
    # same env knobs as blrun's bootstrap so the idempotent re-init matches
    devs = [int(x) for x in os.environ.get("BL_DEVICES", "0,1").split(",")]
    bl.init(devices=devs, pool_mb=int(os.environ.get("BL_POOL_MB", "64")))

    print("%-12s %12s %12s %12s %8s" %
          ("size", "uni GB/s", "dir0 GB/s", "dir1 GB/s", "sum/uni"))
    # NOTE: pool tensors are never freed, so per-device usage accumulates
    # across sizes: 2x(1+4+16) MiB from uni + 2x(1+4+16) MiB from bidir.
    # Run with BL_POOL_MB=192.
    for sz, iters in [(1 << 20, 512), (4 << 20, 256), (16 << 20, 128)]:
        u = uni(sz, iters)
        d0, d1 = bidir(sz, iters)
        print("%-12s %12.2f %12.2f %12.2f %8.2fx" %
              ("%d MiB" % (sz >> 20), u, d0, d1, (d0 + d1) / u))


if __name__ == "__main__":
    main()
