# SPDX-License-Identifier: MIT
#
# Small-message latency for the barlink path (single process, both cards).
# vLLM tensor-parallel decode does many tiny allreduces per token, so
# per-op latency matters more than peak bandwidth.
#
# Run from repo root:  tools/blrun bench/latency/bar_lat.py

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "torch_ext"))

import torch  # noqa: E402


def bench_op(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    # per-op latency: sync both cards after every call
    lat = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        lat += time.perf_counter() - t0
    lat /= iters
    # throughput: calls back-to-back, one sync at the end
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    thr = (time.perf_counter() - t0) / iters
    return lat, thr


def main():
    import barlink_sm86 as bl

    bl.init(devices=[0, 1], pool_mb=64)

    print("%10s  %12s  %12s  %12s" % ("size", "copy_ us/op", "ar us/op",
                                      "ar GB/s(thr)"))
    for n in [4096, 16384, 65536, 262144, 1048576, 4194304]:
        a = bl.empty(n, device=0)
        b = bl.empty(n, device=1)
        a.copy_(torch.randint(0, 256, (n,), dtype=torch.uint8).cuda(0))
        b.copy_(torch.randint(0, 256, (n,), dtype=torch.uint8).cuda(1))

        clat, _ = bench_op(lambda: bl.copy_(b, a))
        alat, athr = bench_op(lambda: bl.allreduce_(a, b))
        print("%10d  %12.1f  %12.1f  %12.2f" % (n, clat * 1e6, alat * 1e6,
                                                n / athr / 1e9))

    bl.shutdown()


if __name__ == "__main__":
    main()
