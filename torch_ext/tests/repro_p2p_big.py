# SPDX-License-Identifier: MIT
# Repro: >chunk multi-chunk full-duplex p2p exchange (both ranks isend+irecv
# simultaneously), byte-exact verification. Sizes default 48/64/128 MB.
#   LOCAL_RANK=R BL_SKIP_INIT=1 tools/blrun torch_ext/tests/repro_p2p_big.py \
#       --rank R --port P [--sizes $((48<<20)) $((64<<20)) $((128<<20))]
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from barlink_sm86 import process_group as blpg  # noqa: E402


def seeded(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), generator=g, dtype=torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[48 << 20, 64 << 20, 128 << 20])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--halfduplex", action="store_true",
                    help="rank 0 sends / rank 1 recvs only (no full-duplex)")
    a = ap.parse_args()
    rank = a.rank
    peer = 1 - rank
    dev = torch.device("cuda", rank)

    blpg.init_process_group(init_method="tcp://127.0.0.1:%d" % a.port,
                            rank=rank, world_size=2)
    tag = 0
    for nbytes in a.sizes:
        n = nbytes // 4
        for it in range(a.iters):
            mine = seeded((1000 + tag) * 2 + rank, n).to(dev)
            want = seeded((1000 + tag) * 2 + peer, n)
            got = torch.empty(n, dtype=torch.int32, device=dev)
            if a.halfduplex:
                if rank == 0:
                    dist.send(mine.view(torch.uint8), peer)
                else:
                    dist.recv(got.view(torch.uint8), peer)
            else:
                rs = dist.isend(mine.view(torch.uint8), peer)
                rr = dist.irecv(got.view(torch.uint8), peer)
                rs.wait()
                rr.wait()
            if not a.halfduplex or rank == 1:
                gotc = got.cpu()
                if not torch.equal(gotc, want):
                    bad = (gotc != want)
                    idx = bad.nonzero().view(-1)
                    print("rank %d FAIL %dB iter %d: %d/%d ints differ; "
                          "first bad @%d last @%d" %
                          (rank, nbytes, it, idx.numel(), n,
                           idx[0].item(), idx[-1].item()), flush=True)
                    torch.save({"got": gotc, "want": want, "nbytes": nbytes},
                               "/tmp/p2p_fail_rank%d.pt" % rank)
                    sys.exit(1)
            print("rank %d: %d MB exchange iter %d OK" % (rank, nbytes >> 20, it),
                  flush=True)
            tag += 1
    print("rank %d: ALL BIG P2P OK" % rank, flush=True)


if __name__ == "__main__":
    main()
