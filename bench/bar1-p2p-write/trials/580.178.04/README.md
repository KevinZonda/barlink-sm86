# Trial: 580.178.04 (stock, unpatched) — baseline

Date: 2026-10-02, kernel 7.0.0-34-generic, dual RTX 3080 20 GiB (BDF 09:00.0 / 0a:00.0, same root complex), BAR1 aperture 256 MiB (reBAR off). Bench built with CUDA 13.4 nvcc.

## Files

- `env.txt` — machine/driver/PCI metadata
- `bar1_bench.log` — `bar1-p2p-write --both` on stock: **FAILED at step 2** (`NV_ESC_CHECK_VERSION_STR: Invalid argument`). The bench embeds the RM ioctl ABI of the 595 branch; stock 580 refuses the version handshake. Path is doubly closed on stock: ioctl ABI mismatch + (unreached) `cudaHostRegister` guard.
- `stock_d2d.log` — what the stock system can actually do today (see below)

## Stock baseline numbers (what BAR1 P2P has to beat)

`cudaDeviceCanAccessPeer` = **0 both directions** — P2P locked, as expected on GeForce.

Cross-device copy bandwidth (pinned staging ≈ driver's internal fallback):

| size | memcpy D2D | pinned staging |
|---|---|---|
| 4 KiB | 0.47 GB/s | 0.41 GB/s (latency-bound) |
| 64 KiB | 2.9 GB/s | 3.0 GB/s |
| 1 MiB | 6.1 GB/s | 6.2 GB/s |
| 64 MiB | 6.0 GB/s | **6.7 GB/s** |

Effective ~6.7 GB/s = data crosses PCIe twice (GPU→host→GPU); one-way PCIe is ~13.5 GB/s-class (x8 Gen4-ish).

**Target for the patched BAR1 path: ~12+ GB/s one-way writes** (per byte moved, ~2× the stock baseline at 64 MiB; latency at 4–64 KiB should improve far more, no host round-trip).
