# SPDX-License-Identifier: MIT
#
# Basic test for barlink_sm86. Requires the full runtime stack:
#   patched driver + BarlinkPeerBar1=1, dmabuf_holder.ko loaded, iommu=pt,
#   root / CAP_SYS_ADMIN, extension built (setup.py build_ext --inplace).
#
# Run directly:  sudo .venv/bin/python tests/test_basic.py
# or via pytest:  sudo .venv/bin/python -m pytest tests/ -v

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402


def require_runtime():
    if os.geteuid() != 0:
        print("SKIP: must run as root (dmabuf_holder is mode 0600)")
        sys.exit(0)
    if not os.path.exists("/dev/dmabuf_holder"):
        print("SKIP: /dev/dmabuf_holder missing -- load dmabuf_holder.ko")
        sys.exit(0)


def main():
    require_runtime()
    import barlink_sm86 as bl

    N = 4 * 1024 * 1024
    bl.init(devices=[0, 1], pool_mb=64)

    # 1. byte proof over both BAR1 directions
    bad = bl.verify()
    assert bad == 0, "bl.verify() bad_bytes = %d" % bad
    print("verify: bad_bytes = 0  OK")

    # 2. copy_ against a CPU reference
    a = bl.empty(N, device=0)
    ref = torch.randint(0, 256, (N,), dtype=torch.uint8)
    a.copy_(ref.cuda(0))                       # local fill of the pool tensor
    b = bl.empty(N, device=1)
    bl.copy_(b, a)
    torch.cuda.synchronize(0)  # drain probe: does host sync fix the race?
    torch.cuda.synchronize(1)
    got = bl.readback(b)                       # copy-engine .cpu() could hit
                                               # stale L2 on inbound writes
    if not torch.equal(got, ref):
        neq = (got != ref)
        idx = neq.nonzero().flatten()
        print("copy_: MISMATCH: %d of %d bytes differ; first at %d, last at %d"
              % (idx.numel(), N, idx[0].item(), idx[-1].item()))
        o = idx[0].item()
        print("  first mismatch window @%d:" % o)
        print("  ref :", ref[o:o+16].tolist())
        print("  got :", got[o:o+16].tolist())
        # second copy to see if the mismatch pattern is deterministic
        bl.copy_(b, a)
        got2 = bl.readback(b)
        same2 = torch.equal(got2, ref)
        print("  second copy_ identical to ref:", same2)
        print("  got == got2 (deterministic):", torch.equal(got, got2))
        raise SystemExit(1)
    print("copy_: %d bytes identical to CPU reference  OK" % N)

    # 3. allreduce_ against (a + b) % 256, both sides
    x = bl.empty(N, device=0)
    y = bl.empty(N, device=1)
    rx = torch.randint(0, 256, (N,), dtype=torch.uint8)
    ry = torch.randint(0, 256, (N,), dtype=torch.uint8)
    x.copy_(rx.cuda(0))
    y.copy_(ry.cuda(1))
    want = ((rx.to(torch.int32) + ry.to(torch.int32)) % 256).to(torch.uint8)
    bl.allreduce_(x, y)
    gotx = bl.readback(x)      # ld.global.cv on the owning card: definitive
    goty = bl.readback(y)
    if not torch.equal(gotx, want):
        idx = (gotx != want).nonzero().flatten()
        o = idx[0].item()
        print("allreduce_: side 0 MISMATCH: %d bytes; first @%d" % (idx.numel(), o))
        print("  want:", want[o:o+16].tolist())
        print("  got :", gotx[o:o+16].tolist())
        raise SystemExit(1)
    if not torch.equal(goty, want):
        idx = (goty != want).nonzero().flatten()
        o = idx[0].item()
        print("allreduce_: side 1 MISMATCH: %d bytes; first @%d" % (idx.numel(), o))
        print("  want:", want[o:o+16].tolist())
        print("  got :", goty[o:o+16].tolist())
        raise SystemExit(1)
    print("allreduce_: both sides equal (a+b) mod 256  OK")

    # 4. small bandwidth measurement (copy_ 0 -> 1)
    for _ in range(3):
        bl.copy_(b, a)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        bl.copy_(b, a)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    dt = (time.perf_counter() - t0) / iters
    print("bandwidth: %.1f GB/s (%d bytes per copy_, both streams synced)"
          % (N / dt / 1e9, N))

    bl.shutdown()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
