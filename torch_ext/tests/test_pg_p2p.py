# SPDX-License-Identifier: MIT
#
# torch.distributed point-to-point test over the barlink peer link.
# Two processes (one per GPU), orchestrated by tests/run_pg_p2p.sh:
#   bash torch_ext/tests/run_pg_p2p.sh
#
# SPMD: both ranks execute the identical op sequence; data is rank-seeded so
# each rank predicts the peer's payload byte-exactly. Byte-exactness is the
# point: send/recv is a pure byte move (no dtype semantics, u8 included).

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
    peer = 1 - rank
    dev = torch.device("cuda", rank)

    blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                            rank=rank, world_size=2)
    my_seed = 2000 + rank
    peer_seed = 2000 + peer

    # 1. full-duplex exchange: both ranks isend + irecv simultaneously,
    #    sizes x dtypes (zero-copy path); byte-exact verification
    tag = 0
    for nbytes in (1024, 64 * 1024, 1 << 20, 8 << 20):
        for dt in (torch.float16, torch.float32, torch.uint8):
            n = nbytes // dt.itemsize
            mine = seeded(my_seed + tag, n, dt).to(dev)
            want = seeded(peer_seed + tag, n, dt)
            got = torch.empty(n, dtype=dt, device=dev)
            rs = dist.isend(mine, peer)
            rr = dist.irecv(got, peer)
            rs.wait()
            rr.wait()
            check(got, want.to(dt), "exchange %s %dB (isend/irecv)" % (dt, nbytes),
                  rank)
            tag += 1

    # 1b. synchronous send/recv ping-pong (rank 0 initiates; both ranks
    #     execute one send and one recv -- identical SPMD sequence)
    for nbytes in (1024, 1 << 20):
        n = nbytes // 4
        mine = seeded(my_seed + 500 + nbytes, n, torch.float32).to(dev)
        want = seeded(peer_seed + 500 + nbytes, n, torch.float32)
        got = torch.empty(n, dtype=torch.float32, device=dev)
        if rank == 0:
            dist.send(mine, peer)
            dist.recv(got, peer)
        else:
            dist.recv(got, peer)
            dist.send(mine, peer)
        check(got, want, "ping-pong fp32 %dB (send/recv)" % nbytes, rank)

    # 1c. int64: any-dtype byte move on the zero-copy path (itemsize 8)
    n = 4096
    mine = torch.arange(n, dtype=torch.int64).to(dev) + rank * 100000
    want = torch.arange(n, dtype=torch.int64) + peer * 100000
    got = torch.empty(n, dtype=torch.int64, device=dev)
    rs = dist.isend(mine, peer)
    rr = dist.irecv(got, peer)
    rs.wait()
    rr.wait()
    check(got, want, "exchange int64 32KB (zero-copy byte move)", rank)

    # 2. staged fallbacks: odd sizes / non-enum dtype / CPU / strided
    # 2a. 999 floats = 3996 B (not a 16-byte multiple)
    n = 999
    mine = seeded(my_seed + 600, n, torch.float32).to(dev)
    want = seeded(peer_seed + 600, n, torch.float32)
    got = torch.empty(n, dtype=torch.float32, device=dev)
    rs = dist.isend(mine, peer)
    rr = dist.irecv(got, peer)
    rs.wait()
    rr.wait()
    check(got, want, "exchange fp32 3996B (staged)", rank)

    # 2b. CPU tensor send + CUDA recv (sender stages through the pool)
    n = 512
    mine_cpu = seeded(my_seed + 601, n, torch.float32)          # CPU
    want = seeded(peer_seed + 601, n, torch.float32)
    got = torch.empty(n, dtype=torch.float32, device=dev)
    rs = dist.isend(mine_cpu, peer)
    rr = dist.irecv(got, peer)
    rs.wait()
    rr.wait()
    check(got, want, "exchange fp32 2KB (CPU send staged)", rank)

    # 2c. CUDA send + CPU tensor recv (receiver stages through the pool)
    mine = seeded(my_seed + 602, n, torch.float32).to(dev)
    want = seeded(peer_seed + 602, n, torch.float32)
    got_cpu = torch.empty(n, dtype=torch.float32)               # CPU
    rs = dist.isend(mine, peer)
    rr = dist.irecv(got_cpu, peer)
    rs.wait()
    rr.wait()
    check(got_cpu, want, "exchange fp32 2KB (CPU recv staged)", rank)

    # 2d. non-contiguous send (strided view -> staged .contiguous())
    base = seeded(my_seed + 603, 2048, torch.float32).to(dev)
    mine_nc = base[::2]
    want = seeded(peer_seed + 603, 2048, torch.float32)[::2]
    got = torch.empty(1024, dtype=torch.float32, device=dev)
    rs = dist.isend(mine_nc, peer)
    rr = dist.irecv(got, peer)
    rs.wait()
    rr.wait()
    check(got, want, "exchange fp32 4KB (non-contiguous staged)", rank)

    # 3. multiple outstanding isend/irecv interleaved (pipelining + staging
    #    buffer reuse); 3 sends then 3 recvs on BOTH ranks -- identical
    #    SPMD sequence
    k = 3
    outs = [seeded(my_seed + 700 + i, 4096 * (i + 1), torch.float16).to(dev)
            for i in range(k)]
    wants = [seeded(peer_seed + 700 + i, 4096 * (i + 1), torch.float16)
             for i in range(k)]
    gots = [torch.empty(4096 * (i + 1), dtype=torch.float16, device=dev)
            for i in range(k)]
    reqs = []
    for i in range(k):
        reqs.append(dist.isend(outs[i], peer))
    for i in range(k):
        reqs.append(dist.irecv(gots[i], peer))
    for r in reqs:
        r.wait()
    for i in range(k):
        check(gots[i], wants[i].to(torch.float16),
              "interleaved isend/irecv msg %d" % i, rank)

    # 4. back-to-back reuse loop: 40 exchanges through the same fixed
    #    scratch offset and the same cached staging buffers
    for i in range(40):
        n = 128 + (i % 5) * 48             # fp32, mixed sizes
        mine = seeded(my_seed + i, n, torch.float32).to(dev)
        want = seeded(peer_seed + i, n, torch.float32)
        got = torch.empty(n, dtype=torch.float32, device=dev)
        rs = dist.isend(mine, peer)
        rr = dist.irecv(got, peer)
        rs.wait()
        rr.wait()
        if not torch.equal(got.cpu(), want.cpu()):
            print("rank %d FAIL reuse loop iter %d" % (rank, i), flush=True)
            sys.exit(1)
        if i % 10 == 9:
            dist.barrier()
    print("rank %d: 40-iter reuse loop OK" % rank, flush=True)

    # 5. error paths (raise BEFORE any bl traffic: SPMD-safe). torch's own
    #    self-rank check fires before the backend, hence ValueError
    try:
        dist.send(torch.ones(8, device=dev), rank)      # dst == self
        print("rank %d FAIL: send-to-self not rejected" % rank, flush=True)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        print("rank %d: send-to-self rejected (%s)" % (rank, e), flush=True)

    # 6. mixed traffic: p2p + allreduce on the same link, alternating
    for i in range(5):
        n = 4096
        mine = seeded(my_seed + 800 + i, n, torch.float32).to(dev)
        want = seeded(peer_seed + 800 + i, n, torch.float32)
        got = torch.empty(n, dtype=torch.float32, device=dev)
        rs = dist.isend(mine, peer)
        rr = dist.irecv(got, peer)
        rs.wait()
        rr.wait()
        check(got, want, "mixed p2p iter %d" % i, rank)
        t = seeded(my_seed + 900 + i, n, torch.float32).to(dev)
        want_ar = seeded(my_seed + 900 + i, n, torch.float32) + \
            seeded(peer_seed + 900 + i, n, torch.float32)
        dist.all_reduce(t)
        check(t, want_ar, "mixed allreduce iter %d" % i, rank)

    # 7. p2p latency: 10 KB ping-pong round trip (event + per-op sync),
    #    same methodology as pg_lat.py / test_process_group.py §11
    n = 2560                                   # fp32 = 10 KB
    t = torch.ones(n, dtype=torch.float32, device=dev)
    g = torch.empty(n, dtype=torch.float32, device=dev)

    def pingpong():
        if rank == 0:
            dist.send(t, peer)
            dist.recv(g, peer)
        else:
            dist.recv(g, peer)
            dist.send(t, peer)

    for _ in range(10):
        pingpong()
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(50):
        pingpong()
    e1.record()
    torch.cuda.synchronize(dev)
    ev_us = e0.elapsed_time(e1) * 1000.0 / 50

    lat = 0.0
    for _ in range(50):
        t0 = time.perf_counter()
        pingpong()
        torch.cuda.synchronize(dev)
        lat += time.perf_counter() - t0
    sync_us = lat * 1e6 / 50
    if rank == 0:
        print("rank 0: PG p2p 10KB roundtrip: %.1f us (event), %.1f us (sync)"
              % (ev_us, sync_us), flush=True)

    dist.destroy_process_group()
    print("rank %d: ALL P2P TESTS PASSED" % rank, flush=True)


if __name__ == "__main__":
    main()
