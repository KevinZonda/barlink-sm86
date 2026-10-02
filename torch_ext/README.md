# barlink_sm86 — torch C++ extension for route B (dual-GPU BAR1 P2P)

Wraps the byte-verified mechanism from `bench/bar1-p2p-write` as torch
primitives: a fixed VMM **pool** per device, created once with all BAR1
export/attach/mapping done up front; `torch.Tensor`s handed out of the pool
via `from_blob`; cross-device `copy_` / `allreduce_` write **directly**
through the peer card's BAR1 aperture — no host staging.

## Prerequisites (identical to the bench — see `../BUILD_AND_TEST.md`)

- Patched driver (`BARLINK_PCIE_MINIMAL.patch` branch 580 or 595) loaded
  with `BarlinkPeerBar1=1`. **Never** set a static-BAR regkey.
- `dmabuf_holder.ko` loaded (`/dev/dmabuf_holder` exists, mode 0600).
- **AMD platform: `iommu=pt` in the kernel cmdline** — otherwise the IOMMU
  silently drops peer BAR writes.
- Run as root / `CAP_SYS_ADMIN`.
- RTX 3080-class BAR1: pool must fit the 256 MiB aperture (`pool_mb <= 192`).

## Build

```bash
TORCH_CUDA_ARCH_LIST="8.6" .venv/bin/python setup.py build_ext --inplace
```

**C++ standard**: torch >= 2.14 headers require C++20 (`at::symint::sizes`
in `ATen/ExpandUtils.h` uses concepts/`requires`). setup.py therefore
compiles the binding with `-std=c++20`; do not downgrade it to c++17 —
gcc then mangles those template declarations and fails deep inside torch
headers with `expected primary-expression before '>' token`. `core.cu`
has no torch headers and is compiled with `-std=c++14`.

`core.cu` alone (no torch) can be checked with plain nvcc:

```bash
nvcc -O3 -std=c++14 -gencode arch=compute_86,code=sm_86 \
     -Ibarlink_sm86 -c barlink_sm86/core.cu -o /tmp/core.o
```

## Use

```python
import barlink_sm86 as bl

bl.init(devices=[0, 1], pool_mb=64)    # one-time; creates pools + BAR1 paths

a = bl.empty(4 * 1024 * 1024, device=0)   # uint8 pool tensor on device 0
b = bl.empty(4 * 1024 * 1024, device=1)   # uint8 pool tensor on device 1

bl.copy_(b, a)                          # device 0 writes device 1 via BAR1
bl.allreduce_(a, b)                     # both become (a+b) mod 256 (u8)
bad = bl.verify()                       # byte proof over both directions
got = bl.readback(b)                    # trustworthy host readback

bl.shutdown()
```

Tensor sizes must be multiples of 16 bytes. Pool tensors are plain
`torch.Tensor`s (uint8, CUDA) — they feed any torch op; only `copy_` /
`allreduce_` / `readback` know about the pool.

## Design constraints (all measured — do not "fix" against them)

- **No flags in the receiving card's VRAM.** The receiver's L2 is not
  coherent with inbound PCIe writes: a spinning kernel there cannot see the
  peer's data until it exits (`barlink-pcie/findings/l2-not-coherent.md`).
  Synchronization is **only** cross-device `cudaEvent`s (record on the
  writer stream, `cudaStreamWaitEvent` on the reader stream).
- **Reads of peer-written data go through an owner-card kernel with
  `ld.global.cv`.** A copy-engine read (`tensor.cpu()`, `cuMemcpyDtoH`) can
  return stale L2 data. `bl.verify()` and `bl.readback()` implement this;
  tests must not `.cpu()` a freshly cross-written tensor to judge
  correctness.
- Writes use grid-stride `st.global.wt` 128-bit stores (`k_copy`), the
  kernel shape measured at 13.2 GB/s on dual 3080.
- `allreduce_` exchanges the **original** values into per-pool scratch in
  both directions first, then each side adds locally. Feeding the
  already-updated peer value would double-count.
- v1 process model: **single process, exactly 2 devices**. The pool
  allocator is a bump + first-fit free list with 2 MiB alignment; pool
  tensors are not individually freed (memory is reclaimed at
  `bl.shutdown()`); `allreduce_`/`verify` scratch uses the free list.
- v1 `allreduce_` is element-wise wrapping **u8** add (`__vadd4`).

## Layout

```
torch_ext/
├── setup.py                  # torch CUDAExtension (TORCH_CUDA_ARCH_LIST aware)
├── barlink_sm86/
│   ├── __init__.py           # lazy _C load, torch/CUDA sanity, actionable errors
│   ├── core.cu               # mechanism + kernels, NO torch headers (standalone nvcc)
│   ├── core.h                # plain-C API
│   └── binding.cpp           # torch/pybind thin layer
├── tests/test_basic.py       # verify/copy_/allreduce_ vs CPU refs + small GB/s
└── README.md
```

## Status / TODO

**PASSED on real hardware** (2026-10-03): dual RTX 3080 20 GiB, patched 580.178.04
+ `BarlinkPeerBar1=1` + dmabuf_holder.ko + `iommu=pt`, torch 2.14.0+cu130.
`tests/test_basic.py`: verify bad_bytes=0 both directions, copy_ 4 MiB
byte-identical to CPU reference, allreduce_ both sides == (a+b)%256,
**bandwidth 12.9 GB/s** (4 MiB per copy_, sync-per-copy conservative mode).

Hardware quirks discovered en route (encoded in core.cu comments):

1. **Second+ grid-stride sweep of a `.wt` BAR1 write kernel is silently
   dropped** -- only the first `gridDim*blockDim*16` bytes land. Always
   launch the full grid (single sweep). The standalone bench never saw this
   because it always did.
2. **Cross-device cudaEvent does NOT imply PCIe posted-write drain** at the
   peer -- readers can observe ~10% stale data. **Fixed by the marker-flag
   protocol** (below); host syncs are gone from `copy_`.
3. Never verify inbound-written data with the copy engine (stale L2) --
   `readback()` uses `ld.global.cv` on the owning card.

### Async copy semantics (quirk #2 fix)

`copy_` is fully asynchronous. Each pool carries a reserved 4 KiB flag
tail (not allocator-visible); per direction it holds a u64 seq `flag`
(+8 is reserved/unused). `bl_copy_` queues on the **source** stream, in
order: `k_copy` (payload, `st.global.wt` posted writes into the peer BAR),
then `k_mark` (one thread: `__threadfence_system()`; then
`st.global.wt flag = seq`). Stream order is hard execution order, so the
single fence in `k_mark` drains **all** of `k_copy`'s posted writes before
the flag store goes out on the same PCIe path — flag arrival implies
payload arrival. (An earlier design had every payload block do its own
system fence plus an atomic block counter; that cost ~1 us per block, i.e.
~1 ms per 4 MiB copy, and dropped bandwidth from 12.9 to 3.3 GB/s.)

The reader (`k_flag_wait` on the destination stream) polls only the flag
with `ld.global.cv` (~3.2 us latency), with a ~2 s trap-on-timeout.

Semantics: `bl.copy_(b, a)` returns with the copy merely **queued**; any
work the consumer queues afterwards on `b`'s current stream (torch ops,
another `copy_`, `readback`) is stream-ordered after the payload. There is
no cross-copy source-buffer reuse hazard: pool tensors are never freed, so
an in-flight `copy_` can never observe its source reallocated.

Not yet done: >2 devices (needs a per-writer attachment policy in
dmabuf_holder), cross-process fd exchange, tensor `free` API,
non-u8 allreduce, stream-aware scratch free in `allreduce_` (host sync
kept, see TODO in core.cu).
