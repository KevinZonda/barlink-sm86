# SPDX-License-Identifier: MIT
#
# dtype coverage for barlink_sm86: copy_ and allreduce_ for fp32, fp64,
# bf16, fp8_e4m3fn, fp8_e5m2, plus a u8 regression case.
#
# Run under the pre-init launcher (no sudo needed):
#   tools/blrun torch_ext/tests/test_dtypes.py
#
# Values are kept small and exactly representable in every dtype so the
# reference math is exact; fp8 operands stay in the 0..16 range (e5m2 has
# only 3 mantissa bits, e4m3 4 -- tiny integers are exact in both).

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

# (name, torch dtype, itemsize, make-reference-values fn)
DTYPES = [
    ("u8",      torch.uint8,        1, lambda n: torch.randint(0, 256, (n,), dtype=torch.int32)),
    ("fp32",    torch.float32,      4, lambda n: torch.randint(0, 16, (n,), dtype=torch.float32)),
    ("fp64",    torch.float64,      8, lambda n: torch.randint(0, 16, (n,), dtype=torch.float64)),
    ("bf16",    torch.bfloat16,     2, lambda n: torch.randint(0, 16, (n,), dtype=torch.float32).to(torch.bfloat16)),
    ("fp8e4m3", torch.float8_e4m3fn, 1, lambda n: torch.randint(0, 9, (n,), dtype=torch.float32).to(torch.float8_e4m3fn)),
    ("fp8e5m2", torch.float8_e5m2,   1, lambda n: torch.randint(0, 5, (n,), dtype=torch.float32).to(torch.float8_e5m2)),
]

N = 1 << 20   # elements; byte size is N * itemsize (multiple of 16 for all)


def toF32(t):
    return t.to(torch.float32)


def runOne(name, dt, itemsize, makeRef):
    nbytes = N * itemsize
    import barlink_sm86 as bl
    ta = bl.empty(nbytes, device=0, dtype=dt)
    tb = bl.empty(nbytes, device=1, dtype=dt)

    assert ta.dtype == dt and tb.dtype == dt, (ta.dtype, tb.dtype)
    assert ta.numel() == N and tb.numel() == N, (ta.numel(), tb.numel())
    assert ta.is_cuda and tb.is_cuda

    ra = makeRef(N)
    rb = makeRef(N)
    ta.copy_(ra.cuda(0))
    tb.copy_(rb.cuda(1))

    # (a) copy_ a(dev0) -> b(dev1): b must equal ra
    bl.copy_(tb, ta)
    got = bl.readback(tb).view(dt)
    if not torch.equal(got.cpu(), ra):
        bad = (got.cpu() != ra).sum().item()
        print("FAIL %s copy_: %d/%d bytes differ" % (name, bad, N))
        return False

    # (b) allreduce_: both sides must equal elementwise a+b.
    # (the copy test overwrote tb with ra -- re-fill from the references)
    ta.copy_(ra.cuda(0))
    tb.copy_(rb.cuda(1))
    if dt == torch.uint8:
        want = ((ra.to(torch.float32) + rb.to(torch.float32)) % 256)
    else:
        want = ra.to(torch.float32) + rb.to(torch.float32)
    bl.allreduce_(ta, tb)
    ga = bl.readback(ta).view(dt)
    gb = bl.readback(tb).view(dt)
    if not torch.equal(toF32(ga.cpu()), want):
        print("FAIL %s allreduce_: side 0 mismatch" % name)
        return False
    if not torch.equal(toF32(gb.cpu()), want):
        print("FAIL %s allreduce_: side 1 mismatch" % name)
        return False

    print("OK   %-8s copy_ + allreduce_ (%d bytes)" % (name, nbytes))
    return True


def main():
    ok = True
    for name, dt, itemsize, makeRef in DTYPES:
        try:
            ok = runOne(name, dt, itemsize, makeRef) and ok
        except Exception as e:  # noqa: BLE001
            print("FAIL %-8s exception: %s" % (name, e))
            ok = False
    if not ok:
        sys.exit(1)
    print("ALL DTYPE TESTS PASSED")


if __name__ == "__main__":
    main()
