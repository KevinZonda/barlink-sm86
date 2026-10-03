# SPDX-License-Identifier: MIT
#
# torch.distributed ProcessGroup backend test over the barlink peer link.
# Two processes (one per GPU), orchestrated by tests/run_pg.sh:
#   bash torch_ext/tests/run_pg.sh
#
# SPMD: both ranks execute the identical op sequence; data differs only
# through rank-seeded reference values, so each rank can predict the peer's
# contribution (allreduce result = mine + peer).

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from barlink_sm86 import process_group as blpg  # noqa: E402


def seeded(seed, n, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        t = torch.rand(n, generator=g, dtype=torch.float32)
    else:
        t = torch.randint(0, 256, (n,), generator=g, dtype=torch.int32)
    return t.to(dtype)


def ref_sum(a, b, dtype):
    # reference a+b with torch compute semantics: fp8 adds on CPU are not
    # implemented, and torch computes fp8/fp16/bf16 elementwise adds with
    # float opmath and one rounding -- which is exactly the pool/zero-copy
    # kernel rule (add in float, round back).
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (a.to(torch.float32) + b.to(torch.float32)).to(dtype)
    return a + b


def check(t, want, tag, rank):
    if not torch.equal(t.cpu(), want.cpu()):
        bad = (t.cpu() != want.cpu()).sum().item()
        print("rank %d FAIL %s: %d/%d elements differ" % (rank, tag, bad, t.numel()),
              flush=True)
        sys.exit(1)
    print("rank %d: %s OK" % (rank, tag), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    rank = a.rank
    dev = torch.device("cuda", rank)

    blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                            rank=rank, world_size=2)
    my_seed = 1000 + rank
    peer_seed = 1000 + (1 - rank)

    def allreduce_case(n, dtype, tag):
        t = seeded(my_seed + n, n, dtype).to(dev)
        want = ref_sum(seeded(my_seed + n, n, dtype),
                       seeded(peer_seed + n, n, dtype), dtype)
        dist.all_reduce(t)
        check(t, want.to(dtype), tag, rank)

    # 1. fp32 allreduce: 1 KiB / 1 MiB / 32 MiB
    allreduce_case(256, torch.float32, "all_reduce fp32 1KB")
    allreduce_case(256 * 1024, torch.float32, "all_reduce fp32 1MB")
    allreduce_case(8 * 1024 * 1024, torch.float32, "all_reduce fp32 32MB")

    # 2. dtypes: bf16 / fp64 native, fp16 (now native on the zero-copy path),
    #    fp64 native, fp8 x2
    allreduce_case(65536, torch.bfloat16, "all_reduce bf16")
    allreduce_case(32768, torch.float64, "all_reduce fp64")
    allreduce_case(65536, torch.float16, "all_reduce fp16 (zero-copy native)")
    allreduce_case(65536, torch.float8_e4m3fn, "all_reduce fp8_e4m3fn")
    allreduce_case(65536, torch.float8_e5m2, "all_reduce fp8_e5m2")

    # 2b. zero-copy path edge shapes: odd-element fp16 (multiple of 16
    # bytes), in-place semantics on a non-pool torch allocation of each
    # supported dtype
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
               torch.float8_e4m3fn, torch.float8_e5m2):
        n = 12 * (16 // dt.itemsize)   # 12 uint4 lanes
        t = seeded(my_seed + 5, n, dt).to(dev)
        want = ref_sum(seeded(my_seed + 5, n, dt),
                       seeded(peer_seed + 5, n, dt), dt)
        dist.all_reduce(t)
        check(t, want.to(dt), "zero-copy %s" % dt, rank)

    # 2c. direct binding: bl.allreduce_into on plain torch tensors
    import barlink_sm86 as bl
    n = 65536
    tin = seeded(my_seed + 6, n, torch.float32).to(dev)
    tout = tin.clone()
    bl.allreduce_into(tout, tin)
    want = seeded(my_seed + 6, n, torch.float32) + \
        seeded(peer_seed + 6, n, torch.float32)
    check(tout, want.to(torch.float32), "bl.allreduce_into", rank)

    # 2d. unaligned fp16 falls back to the staging cast path (data_ptr not
    # 16-aligned, size still a multiple of 16 bytes)
    base = seeded(my_seed + 78, 65536 + 8, torch.float16).to(dev)
    tu = base[1:]                       # 2-byte-misaligned view (still 131072 B)
    assert tu.data_ptr() % 16 != 0
    want = seeded(my_seed + 78, 65536 + 8, torch.float16)[1:] + \
        seeded(peer_seed + 78, 65536 + 8, torch.float16)[1:]
    dist.all_reduce(tu)
    check(tu, want.to(torch.float16), "all_reduce fp16 unaligned (staging fallback)", rank)


    # 3. odd sizes: not a multiple of 16 bytes (999 floats = 3996 B)
    allreduce_case(999, torch.float32, "all_reduce fp32 3996B (non-16)")

    # 4. non-contiguous input
    base = seeded(my_seed + 77, 2048, torch.float32).to(dev)
    tnc = base[::2]                      # 1024 floats, stride 2
    assert not tnc.is_contiguous()
    want = seeded(my_seed + 77, 2048, torch.float32)[::2] + \
        seeded(peer_seed + 77, 2048, torch.float32)[::2]
    dist.all_reduce(tnc)
    check(tnc, want, "all_reduce non-contiguous", rank)

    # 5. large: 96 MiB fp32, exercises the chunked path (pool 192 MB)
    allreduce_case(24 * 1024 * 1024, torch.float32, "all_reduce fp32 96MB (chunked)")

    # 6. broadcast, both roots
    for root in (0, 1):
        n = 262144
        mine = seeded(my_seed + 900 + root, n, torch.float32).to(dev)
        want = seeded((1000 + root) + 900 + root, n, torch.float32)
        dist.broadcast(mine, src=root)
        check(mine, want.to(torch.float32), "broadcast root=%d" % root, rank)

    # 7. barrier
    dist.barrier()
    print("rank %d: barrier OK" % rank, flush=True)

    # 8. staging-buffer reuse: 200 mixed ops through the same cached buffers
    plan = [
        (1024, torch.float32), (512, torch.bfloat16), (256, torch.float16),
        (128, torch.float64), (2048, torch.float32),
    ]
    for i in range(200):
        n, dt = plan[i % len(plan)]
        allreduce_case(n + i % 7, dt, None) if False else None
        t = seeded(my_seed + i, n, dt).to(dev)
        want = seeded(my_seed + i, n, dt) + seeded(peer_seed + i, n, dt)
        dist.all_reduce(t)
        if not torch.equal(t.cpu(), want.to(dt).cpu()):
            print("rank %d FAIL reuse loop iter %d" % (rank, i), flush=True)
            sys.exit(1)
        if i % 50 == 49:
            dist.barrier()
    print("rank %d: 200-iter reuse loop OK" % rank, flush=True)

    # 9. error paths (raise BEFORE any bl traffic: SPMD-safe)
    try:
        dist.all_reduce(torch.ones(8, device=dev), op=dist.ReduceOp.MIN)
        print("rank %d FAIL: ReduceOp.MIN not rejected" % rank, flush=True)
        sys.exit(1)
    except RuntimeError as e:
        print("rank %d: ReduceOp.MIN rejected (%s)" % (rank, e), flush=True)
    try:
        dist.all_reduce(torch.ones(8, dtype=torch.int64, device=dev))
        print("rank %d FAIL: int64 not rejected" % rank, flush=True)
        sys.exit(1)
    except RuntimeError as e:
        print("rank %d: int64 rejected (%s)" % (rank, e), flush=True)

    # 10. final sanity after the raises
    allreduce_case(4096, torch.float32, "all_reduce post-error")

    # 11. PG-layer latency (raw bl link: 22-26 us at <= 64 KiB)
    for n in (2560, 16384, 65536, 262144):     # 10 KB, 64 KB, 256 KB, 1 MB
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
        us = e0.elapsed_time(e1) * 1000.0 / 50
        print("rank %d: PG all_reduce %d B: %.1f us" % (rank, n * 4, us),
              flush=True)

    dist.destroy_process_group()
    print("rank %d: ALL PG TESTS PASSED" % rank, flush=True)


if __name__ == "__main__":
    main()
