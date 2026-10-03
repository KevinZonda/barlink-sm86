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
// BAR1 direct path, fully async (marker-flag protocol, no host sync):
// k_copy on the source device (srcStream) writes dst's BAR1 window with
// st.global.wt and publishes a seq flag after its last payload store;
// k_flag_wait on the destination device (dstStream) polls the flag with
// ld.global.cv and unblocks when it arrives. Semantics: when bl_copy_
// returns, the copy is merely QUEUED -- the data is usable by any work the
// caller queues afterwards on dstStream (torch ops see the same ordering);
// consuming from a different stream requires the caller's own ordering.
int  bl_copy_(blCtx *ctx, void *dstPtr, int dstIdx, void *srcPtr, int srcIdx,
              size_t bytes, void *srcStream, void *dstStream,
              char *err, size_t errlen);

// Two-device allreduce_: afterwards both tensors hold (a + b) elementwise.
// dtype selects the element semantics (see enum below). Internally: both
// directions copied into per-pool scratch (original values), then each side
// adds its scratch locally.
//
// Elementwise add semantics per dtype:
//   BL_DTYPE_U8      wrapping u8 add (SIMD __vadd4), matches torch uint8 +
//   BL_DTYPE_FP32    native float add
//   BL_DTYPE_FP64    native double add
//   BL_DTYPE_BF16    add via float, rounded back (torch bf16 compute rule)
//   BL_DTYPE_FP8E4M3 add via float, converted back (SATFINITE rounding)
//   BL_DTYPE_FP8E5M2 same
#define BL_DTYPE_U8      0
#define BL_DTYPE_FP32    1
#define BL_DTYPE_FP64    2
#define BL_DTYPE_BF16    3
#define BL_DTYPE_FP8E4M3 4
#define BL_DTYPE_FP8E5M2 5
#define BL_DTYPE_FP16    6
int  bl_allreduce_(blCtx *ctx, void *aPtr, int aIdx, void *bPtr, int bIdx,
                   size_t bytes, int dtype,
                   void *streamA, void *streamB,
                   char *err, size_t errlen);

// Zero-copy peer allreduce: in/out are ARBITRARY device pointers (e.g. torch
// tensors), NOT pool tensors. After both ranks call it: out holds
// in + peer_in elementwise (same dtype semantics as bl_allreduce_; u8 is
// NOT supported here -- it keeps the mod-256 pool path). One payload per
// rank (in -> the peer's scratch zone), then a local add kernel writes out
// directly -- no staging copies in or out of the pool, no host sync; the
// call is fully stream-ordered like the rest of the peer protocol.
//
// Requirements: peer mode; in/out 16-byte aligned (torch allocations are);
// byte size a non-zero multiple of 16 and <= the peer scratch zone
// (pool/2 - flag tail -- chunk larger tensors at the caller); in == out
// (in-place) or fully disjoint ranges.
int  bl_allreduce_into_peer(blCtx *ctx, const void *inPtr, void *outPtr,
                            size_t bytes, int dtype, void *stream,
                            char *err, size_t errlen);

// Zero-copy peer point-to-point (peer mode only; arbitrary device pointers
// like bl_allreduce_into_peer). Pure BYTE MOVE -- dtype is validated for API
// symmetry but never interpreted, so u8 has no mod-256 semantics here and
// any itemsize works. One direction per call; the k-th send on one rank
// pairs with the k-th recv of the same byte size on the peer (SPMD
// discipline). There is no control channel and tags are NOT transported
// (v1): matching is by per-direction ordinals. p2p uses its own flag slots
// (2 + writer rank) and counters, independent of the allreduce/copy
// protocol, so mixed sequences are safe.
//
// Protocol (per direction): the SENDER waits the consumed-receipt of the
// previous exchange (posted by the peer's recv, stream-ordered after its
// move kernel finished READING the scratch -- the first send of a session
// skips the wait), streams 'in' into the peer's scratch zone (st.global.wt
// through the BAR) and raises the completion flag; the RECEIVER waits that
// flag, copies the scratch zone into 'out' with ld.relaxed.sys reads
// (inbound PCIe writes) and posts the consumed-receipt. The receipt is
// posted after the move rather than an arm before it deliberately: an
// arm-at-start would deadlock a full-duplex pair, since the sender's
// arm-wait kernel would spin ahead of its own recv's arm-mark on the same
// stream while the peer's send waits for exactly that mark.
int  bl_send_into_peer(blCtx *ctx, const void *inPtr, size_t bytes, int dtype,
                       int peerRank, void *stream, char *err, size_t errlen);
int  bl_recv_into_peer(blCtx *ctx, void *outPtr, size_t bytes, int dtype,
                       int peerRank, void *stream, char *err, size_t errlen);

// Byte proof over all device pairs: writes a pattern through each BAR1 path
// and verifies it on the owner card through its own VMM pointer with
// ld.global.cv. Returns total bad_bytes (0 = verified, ~0 = run failed).
// In a peer-mode ctx this dispatches to the symmetric cross-process proof.
uint64_t bl_verify(blCtx *ctx, char *err, size_t errlen);

// Symmetric SPMD peer mode: ONE PROCESS PER GPU. rank = device index (0/1).
// Both ranks must call IDENTICAL sequences of bl_* functions with identical
// sizes (MPI-symmetric-heap discipline); there is NO runtime control
// channel — bl_init_peer performs a one-time two-phase unix-socket
// rendezvous (phase 1: BDF + poolBytes; phase 2: BAR1 offset), after which
// the processes are independent. CAP_SYS_ADMIN is required only inside
// init_peer (cudaHostRegister of the peer BAR); bl_drop_caps() is called on
// success, like bl_init does implicitly in the binding.
int  bl_init_peer(blCtx **out, int device, size_t poolBytes,
                  const char *sockPath, int rank, char *err, size_t errlen);

// Peer-mode allreduce: a and b are MY LOCAL pool tensors (same offsets on
// both ranks by symmetric allocation). After both ranks call it: each
// tensor holds my_value + peer_value elementwise (u8 wraps, see dtype
// semantics of bl_allreduce_). Scratch lives in a fixed zone at
// [scratchBase, size-flagTail); user tensors must fit below scratchBase.
int  bl_allreduce_peer(blCtx *ctx, void *aPtr, void *bPtr, size_t bytes,
                       int dtype, void *stream, char *err, size_t errlen);

// Symmetric byte proof for peer mode: each rank writes a pattern into the
// PEER pool's scratch zone through its BAR path and verifies its own local
// scratch zone (written by the peer). Returns bad_bytes for the
// peer->me direction (the peer's process reports the other direction).
uint64_t bl_verify_peer(blCtx *ctx, char *err, size_t errlen);

// BAR atomic feasibility probe (diagnostic, used by the fused-protocol
// design): measures the round-trip latency of atom.global.add.u64 issued
// against the PEER's pool through the BAR1 write path (a sysmem VA from
// cudaHostRegister IoMemory), against the marker-flag round trip used by
// the protocols. res[] (6 entries): [0] atomic exchange us/iter, [1] flag
// exchange us/iter, [2] final local slot value (must equal iters --
// detects dropped/garbled atomic TLPs), [3] pre-armed launch baseline,
// [4] empty-kernel launch baseline, [5] local-only mark baseline.
// SPMD: BOTH ranks must call it with the same 'iters'.
int  bl_probe_bar_atomic(blCtx *ctx, unsigned long long *res, int iters,
                         void *stream, char *err, size_t errlen);

// Move 'bytes' from a local pool buffer (peer-written, inbound PCIe
// writes) into an arbitrary device tensor with ld.relaxed.sys reads --
// k_move semantics with an explicit source (the PG all_gather staging path)
int  bl_pool_move(blCtx *ctx, void *srcPoolPtr, void *outPtr, size_t bytes,
                  void *stream, char *err, size_t errlen);

// debug: host read of the local pool's flag-tail slots (+0/+8 of slots
// 0..7), for watching the handshake while a wait spins
int  bl_debug_flags(blCtx *ctx, unsigned long long *vals,
                    char *err, size_t errlen);

// Drop all capabilities (prctl ambient clear + capset). Called by the
// binding after a successful init; harmless without privileges.
void bl_drop_caps(void);

// Nonzero when ctx was created by bl_init_peer.
int  bl_is_peer(const blCtx *ctx);

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
