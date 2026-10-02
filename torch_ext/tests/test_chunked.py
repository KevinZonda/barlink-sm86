# SPDX-License-Identifier: MIT
#
# Chunked large-tensor test: tensors bigger than one copy_ call are moved /
# reduced through copy_large / allreduce_large in 16 MiB slices, including a
# size that is not a multiple of the chunk (tail slice).
#
# Run: BL_POOL_MB=192 tools/blrun torch_ext/tests/test_chunked.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def main():
    import barlink_sm86 as bl

    devs = [int(x) for x in os.environ.get("BL_DEVICES", "0,1").split(",")]
    bl.init(devices=devs, pool_mb=int(os.environ.get("BL_POOL_MB", "64")))

    # 1. fp32 48 MiB copy in three 16 MiB chunks
    n = 48 << 20
    src = bl.empty(n, device=0, dtype=torch.float32)
    dst = bl.empty(n, device=1, dtype=torch.float32)
    src.copy_(torch.linspace(0, 1, n // 4, device="cuda:0"))
    bl.copy_large(dst, src)
    if not torch.equal(dst.cpu(), src.cpu()):
        # rerun with mismatch detail
        d = (dst.cpu() != src.cpu()).sum().item()
        raise AssertionError("copy_large fp32 48MiB: %d bad elems" % d)
    print("copy_large fp32 48 MiB (3 chunks)  OK")

    # 2. tail slice: 40 MiB + 16 bytes is not a chunk multiple
    n2 = (40 << 20) + 16
    src2 = bl.empty(n2, device=0, dtype=torch.float32)
    dst2 = bl.empty(n2, device=1, dtype=torch.float32)
    src2.copy_(torch.linspace(-1, 1, n2 // 4, device="cuda:0"))
    bl.copy_large(dst2, src2)
    if not torch.equal(dst2.cpu(), src2.cpu()):
        d = (dst2.cpu() != src2.cpu()).sum().item()
        raise AssertionError("copy_large fp32 tail: %d bad elems" % d)
    print("copy_large fp32 40 MiB+16B (tail slice)  OK")

    # 3. bf16 32 MiB copy
    n3 = 32 << 20
    src3 = bl.empty(n3, device=0, dtype=torch.bfloat16)
    dst3 = bl.empty(n3, device=1, dtype=torch.bfloat16)
    src3.copy_(torch.randn(n3 // 2, device="cuda:0").bfloat16())
    bl.copy_large(dst3, src3)
    if not torch.equal(dst3.cpu(), src3.cpu()):
        raise AssertionError("copy_large bf16 mismatch")
    print("copy_large bf16 32 MiB  OK")

    # 4. allreduce_large fp32 48 MiB: both sides must become a+b
    a = bl.empty(n, device=0, dtype=torch.float32)
    b = bl.empty(n, device=1, dtype=torch.float32)
    a.copy_(torch.full(((n // 4),), 1.0, device="cuda:0"))
    b.copy_(torch.full(((n // 4),), 2.0, device="cuda:1"))
    bl.allreduce_large(a, b)
    want = 3.0
    if not bool((a.cpu() == want).all()) or not bool((b.cpu() == want).all()):
        raise AssertionError("allreduce_large fp32 mismatch")
    print("allreduce_large fp32 48 MiB  OK")

    print("ALL CHUNKED TESTS PASSED")


if __name__ == "__main__":
    main()
