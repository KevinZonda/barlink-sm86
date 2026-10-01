# bar1-p2p-write

Native C++/CUDA benchmark for **direct GPU-to-GPU writes through the target
card's dynamically-mapped BAR1 aperture** — no Python, no barlink library,
no NCCL, single process on both devices.

Mechanism (all stock driver code except the guard, see
`barlink-pcie/patches/BARLINK_PCIE_MINIMAL.patch`):

1. VMM allocation on the target card (`cuMemCreate`/`cuMemAddressReserve`/
   `cuMemMap`/`cuMemSetAccess`).
2. dma-buf export via the RM ioctl `NV_ESC_EXPORT_TO_DMABUF_FD`
   (`cuMemGetHandleForAddressRange(DMA_BUF_FD)` is rejected on GeForce).
3. `/dev/dmabuf_holder` HOLD ioctl: `dma_buf_attach` +
   `dma_buf_map_attachment` as the source card's PCI device — this triggers
   `nv_dma_buf_map()` and programs the target card's BAR1 pages **dynamically**.
   No static-BAR regkey is set anywhere.
4. Target BAR1 (`resource1_wc`) mmap'ed, `cudaHostRegister(IoMemory)` on the
   source card, `cudaHostGetDevicePointer` → source-card device pointer.
5. Phase 1: source kernel writes an offset-dependent pattern with
   `st.global.wt` (128-bit); readback through the target card's own VMM
   pointer must match byte-for-byte (`bad_bytes` must be 0).
   Phase 2: size sweep (4 KiB … 64 MiB), cudaEvent-timed kernel write bursts.

## Prerequisites

- Patched driver loaded (`BARLINK_PCIE_MINIMAL.patch`) with the
  `BarlinkPeerBar1=1` regkey set.
- `dmabuf_holder.ko` loaded:

  ```
  make -C /lib/modules/$(uname -r)/build M=<barlink-pcie>/dmabuf_holder modules CC=gcc-14
  sudo insmod <barlink-pcie>/dmabuf_holder/dmabuf_holder.ko
  ```

- Run as root / with `CAP_SYS_ADMIN` (the module node is mode 0600).

## Build & run

```
make            # nvcc, sm_86 default; ARCH=89 or TORCH_CUDA_ARCH_LIST override
sudo ./bar1-p2p-write                 # device 0 -> device 1, 64 MiB
sudo ./bar1-p2p-write --both          # both directions
sudo ./bar1-p2p-write --size=32M --iters=500
```

Exit status: 0 = byte verification passed and bandwidth measured;
non-zero = the path is not working, the error message names the failing
step and its most likely cause (missing patch / regkey, module not loaded,
missing capability).

Note: `--size` must fit the 256 MiB BAR1 aperture of a 3080 (keep ≤ ~192 MiB);
the allocation is rounded up to the VMM granularity (typically 2 MiB).
