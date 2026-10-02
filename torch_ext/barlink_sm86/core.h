// SPDX-License-Identifier: MIT
//
// barlink_sm86 core -- C API over the dual-GPU BAR1 P2P mechanism.
// See core.cu for the mechanism and constraints. No CUDA types in this
// header: streams and pointers cross the boundary as void*, device
// selection is by INDEX into the devices[] array passed to bl_init
// (NOT the raw CUDA ordinal).
//
// Error convention: functions return 0 on success, -1 on failure, with a
// human-readable, actionable message in err/errlen. bl_verify returns the
// total bad_bytes (0 = clean, ~0 = the run itself failed, see err).

#ifndef BARLINK_SM86_CORE_H
#define BARLINK_SM86_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct blCtx blCtx;

// One-time setup: creates a VMM pool on each device and the BAR1 write path
// from the other device into it. devices: CUDA ordinals, exactly 2 (v1).
// poolBytes: per-device pool size (rounded up to the VMM granularity).
// Requires: patched driver + BarlinkPeerBar1=1, dmabuf_holder.ko loaded,
// root / CAP_SYS_ADMIN, iommu=pt.
int  bl_init(blCtx **out, const int *devices, int ndev, size_t poolBytes,
             char *err, size_t errlen);
void bl_shutdown(blCtx *ctx);

// Allocate 'bytes' (rounded to 16) from the pool of devices[devIdx] with
// 2 MiB alignment. *ptrOut is a device pointer valid for torch::from_blob
// on that device. Pool memory is reclaimed only via bl_pool_free or
// bl_shutdown.
int  bl_pool_alloc(blCtx *ctx, int devIdx, size_t bytes, void **ptrOut,
                   char *err, size_t errlen);
int  bl_pool_free(blCtx *ctx, int devIdx, void *ptr, size_t bytes);

// Copy 'bytes' from the src pool tensor to the dst pool tensor across the
// BAR1 direct path: launched on the source device (srcStream), reading src
// locally and writing into dst's BAR1 window with st.global.wt; a
// cross-device cudaEvent (record on srcStream, wait on dstStream) orders
// the write before any subsequent dst-side work.
int  bl_copy_(blCtx *ctx, void *dstPtr, int dstIdx, void *srcPtr, int srcIdx,
              size_t bytes, void *srcStream, void *dstStream,
              char *err, size_t errlen);

// Two-device allreduce_: afterwards both tensors hold (a + b) elementwise
// as wrapping u8. Internally: both directions copied into per-pool scratch
// (original values), then each side adds its scratch locally.
int  bl_allreduce_(blCtx *ctx, void *aPtr, int aIdx, void *bPtr, int bIdx,
                   size_t bytes, void *streamA, void *streamB,
                   char *err, size_t errlen);

// Byte proof over all device pairs: writes a pattern through each BAR1 path
// and verifies it on the owner card through its own VMM pointer with
// ld.global.cv. Returns total bad_bytes (0 = verified, ~0 = run failed).
uint64_t bl_verify(blCtx *ctx, char *err, size_t errlen);

// Reliable host readback of a pool region that may have received inbound
// PCIe writes: a kernel on the owner card reads with ld.global.cv into a
// staging buffer (a plain cudaMemcpy/copy-engine read can return stale L2
// data -- see barlink-pcie/findings/l2-not-coherent.md).
int  bl_readback(blCtx *ctx, int devIdx, void *ptr, size_t bytes,
                 void *dstHost, char *err, size_t errlen);

// Diagnostics for the C API user (see bench/bar1-p2p-write for the full
// layered diagnostics): last-error string is already returned per call.

#ifdef __cplusplus
}
#endif

#endif // BARLINK_SM86_CORE_H
