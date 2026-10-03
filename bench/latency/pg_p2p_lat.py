# SPDX-License-Identifier: MIT
#
# PG-layer point-to-point latency bench. Two processes via run_pg.sh-style
# launch (run_pg_p2p_lat.sh). Methodology matches pg_lat.py: 10 warmup +
# 50 CUDA-event-timed back-to-back iterations, plus a per-op-sync number.
#
# Two shapes per size:
#   ping-pong  -- rank 0 send -> recv, rank 1 recv -> send: full round trip,
#                 serialized through the link (2 one-way deliveries)
#   exchange   -- both ranks isend + irecv simultaneously: full-duplex, one
#                 one-way delivery per direction overlapped on the wire

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "torch_ext"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from barlink_sm86 import process_group as blpg  # noqa: E402

SIZES = [10240, 65536, 262144, 1048576]    # 10 KB, 64 KB, 256 KB, 1 MB


def timeit(fn, dev):
    for _ in range(10):
        fn()
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(50):
        fn()
    e1.record()
    torch.cuda.synchronize(dev)
    ev_us = e0.elapsed_time(e1) * 1000.0 / 50
    lat = 0.0
    for _ in range(50):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        lat += time.perf_counter() - t0
    return ev_us, lat * 1e6 / 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()

    blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                            rank=a.rank, world_size=2)
    rank = a.rank
    peer = 1 - rank
    dev = torch.device("cuda", rank)

    out = []
    for nbytes in SIZES:
        n = nbytes // 4
        t = torch.ones(n, dtype=torch.float32, device=dev)
        g = torch.empty(n, dtype=torch.float32, device=dev)

        def pingpong():
            if rank == 0:
                dist.send(t, peer)
                dist.recv(g, peer)
            else:
                dist.recv(g, peer)
                dist.send(t, peer)

        def exchange():
            rs = dist.isend(t, peer)
            rr = dist.irecv(g, peer)
            rs.wait()
            rr.wait()

        pp_ev, pp_sync = timeit(pingpong, dev)
        ex_ev, ex_sync = timeit(exchange, dev)
        out.append((nbytes, pp_ev, pp_sync, ex_ev, ex_sync))
        print("rank %d: PG p2p %d B: pingpong %.1f/%.1f us, "
              "exchange %.1f/%.1f us (event/sync)"
              % (rank, nbytes, pp_ev, pp_sync, ex_ev, ex_sync), flush=True)

    if rank == 0:
        print("SUMMARY size pp_event_us pp_sync_us ex_event_us ex_sync_us")
        for b, ppe, pps, exe, exs in out:
            print("SUMMARY %d %.1f %.1f %.1f %.1f" % (b, ppe, pps, exe, exs),
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
