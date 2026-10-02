# Trial: 580.178.04 patched + iommu=pt — PASSED

Date: 2026-10-02, kernel 7.0.0-38-generic, dual RTX 3080 20 GiB (BDF 09:00.0 / 0a:00.0).
Driver: `drv/580.178.04` + `BARLINK_PCIE_MINIMAL.patch` (ported), loaded with
`NVreg_RegistryDwords="BarlinkPeerBar1=1"`, `dmabuf_holder.ko`, kernel cmdline `iommu=pt`.
Bench: `bar1-p2p-write-580 --both` (exit 0). Full log: `run.log`.

## Phase 1 — byte verification

| direction | kernel readback (ld.global.cv) | cuMemcpyDtoH (2nd opinion) |
|---|---|---|
| 0 -> 1 | bad_bytes = 0 / 67108864 | 0 / 67108864 |
| 1 -> 0 | bad_bytes = 0 / 67108864 | 0 / 67108864 |

Setup: 64 MiB VMM alloc, 1024 sg entries of 64 KiB, BAR1 offset 0x320000 (3 MiB),
`cudaHostRegister(IoMemory)` allowed (guard relaxed, dmesg shows `BARLINK_PCIE: ALLOW ... PEER_BAR1_APERTURE`).

## Phase 2 — bandwidth (kernel write to peer BAR1, GB/s)

| size | 0->1 | 1->0 | stock baseline (staging) | speedup |
|---|---|---|---|---|
| 4 KiB | 1.68 | 1.71 | 0.47 | 3.6x |
| 64 KiB | 8.96 | 9.01 | 2.93 | 3.1x |
| 1 MiB | 12.80 | 12.81 | 6.14 | 2.1x |
| 4 MiB | 13.09 | 13.09 | 6.18 | 2.1x |
| 16 MiB | 13.16 | 13.16 | 6.24 | 2.1x |
| 64 MiB | 13.17 | 13.17 | 6.02 | 2.2x |

Fully symmetric both directions. ~13.2 GB/s plateau matches barlink-pcie's static-window
measurement (12.7 GB/s on x8 Gen4) — the dynamic dma-buf path leaves no bandwidth on
the table vs the static-window data plane.

## Root causes found on the way (documented for future rigs)

1. **AMD-Vi IOMMU** (default on, translating) silently dropped all peer-BAR1 writes
   (`IO_PAGE_FAULT domain=0x001a` on the source GPU in dmesg). Fix: kernel cmdline
   `iommu=pt` (same requirement as NVIDIA GPUDirect RDMA docs). **Every AMD-platform
   rig running this path needs this.**
2. Stale-L2 readback artifact: `cuMemcpyDtoH` reads after peer writes can hit stale
   L2 lines (target L2 not coherent with inbound PCIe writes). Verification must use
   a target-card kernel with `ld.global.cv` (or drain via kernel boundary).
3. Kernel upgrades break vermagic — rebuild `drv/` modules + `dmabuf_holder.ko` after
   every kernel bump (this trial: 7.0.0-38).

## Known non-fatal residue

Driver prints `pIOVAS != NULL` / `Sysmemdesc outlived its attached pGpu` assertion
noise during teardown of the mapping. Data path unaffected; revisit only if hangs appear.

## Rollback

`sudo rmmod dmabuf_holder nvidia_uvm nvidia && sudo modprobe nvidia && sudo modprobe nvidia_uvm`
reverts to stock 580.178.04. Cold boot also reverts.
