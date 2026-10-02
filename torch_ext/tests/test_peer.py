# SPDX-License-Identifier: MIT
#
# Cross-process (SPMD peer) test for barlink_sm86: one process per GPU.
# Run via tests/run_peer.sh, which starts rank 0 and rank 1 under
# tools/blrun with BL_SKIP_INIT=1 (each process inits itself via
# bl._C.init_peer).
#
# Discipline: both ranks execute IDENTICAL sequences with identical sizes;
# data differs per rank only through rank-seeded reference values, so each
# rank can predict the peer's tensors.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402


def ref(seed, n):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.int32, generator=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--sock", required=True)
    ap.add_argument("--pool-mb", type=int, default=64)
    a = ap.parse_args()

    rank = a.rank
    device = rank
    bl = __import__("barlink_sm86")
    bl._C.init_peer(device, a.pool_mb, a.sock, rank)

    N = 4 << 20

    # reference values: rank r holds seed S_r; each rank can compute both
    rx0, rx1 = ref(1234, N), ref(5678, N)
    ry0, ry1 = ref(4321, N), ref(8765, N)
    mx = (rx0 if rank == 0 else rx1).to(torch.uint8)
    my = (ry0 if rank == 0 else ry1).to(torch.uint8)

    # 1. symmetric byte proof over the BAR path (clobbers the scratch zone,
    #    so it runs before the allreduce tests)
    bad = bl.verify()
    assert bad == 0, "verify_peer bad_bytes = %d" % bad
    print("rank %d: verify_peer OK" % rank, flush=True)

    # NOTE on SPMD buffer discipline: a local tensor that the PEER process
    # has written (via an earlier exchange step) must NOT be locally
    # refilled and reused -- a local write is only ordered after MY flag
    # wait, not after the peer's payload. Use FRESH pool tensors (bump
    # allocation keeps offsets symmetric across ranks) for every step.
    def fresh():
        return bl.empty(N, device=device), bl.empty(N, device=device)

    # 2a. copy_(x, y): after BOTH ranks call it, my x holds the peer's y
    x, y = fresh()
    x.copy_(mx.cuda(device))
    y.copy_(my.cuda(device))
    bl.copy_(x, y)
    torch.cuda.synchronize(device)
    gx = bl.readback(x)
    peer_y = (ry1 if rank == 0 else ry0).to(torch.uint8)
    if not torch.equal(gx.cpu(), peer_y):
        g = gx.cpu()
        n = (g != peer_y).sum().item()
        print("rank %d FAIL copy 2a: %d/%d differ; got[:8]=%s want[:8]=%s" %
              (rank, n, N, g[:8].tolist(), peer_y[:8].tolist()), flush=True)
        sys.exit(1)
    print("rank %d: copy_ (x <- peer y) OK" % rank, flush=True)

    # 2b. copy_(y, x): the other direction, on fresh tensors
    x, y = fresh()
    x.copy_(mx.cuda(device))
    y.copy_(my.cuda(device))
    bl.copy_(y, x)
    torch.cuda.synchronize(device)
    gy = bl.readback(y)
    peer_x = (rx1 if rank == 0 else rx0).to(torch.uint8)
    assert torch.equal(gy.cpu(), peer_x), "copy: y != peer's x"
    print("rank %d: copy_ (y <- peer x) OK" % rank, flush=True)

    # 3. SPMD allreduce: after BOTH ranks call, each local tensor holds
    #    rank0_value + rank1_value (u8 wraps)
    x, y = fresh()
    x.copy_(mx.cuda(device))
    y.copy_(my.cuda(device))
    bl.allreduce_(x, y)
    torch.cuda.synchronize(device)
    want_x = ((rx0 + rx1) % 256).to(torch.uint8)
    want_y = ((ry0 + ry1) % 256).to(torch.uint8)
    gx = bl.readback(x)
    gy = bl.readback(y)
    assert torch.equal(gx.cpu(), want_x), "allreduce: x mismatch"
    assert torch.equal(gy.cpu(), want_y), "allreduce: y mismatch"
    print("rank %d: allreduce_ OK" % rank, flush=True)

    print("rank %d: ALL PEER TESTS PASSED" % rank, flush=True)
    bl.shutdown()


if __name__ == "__main__":
    main()
