// SPDX-License-Identifier: MIT
//
// barlink_sm86 core -- device-side mechanism for route B: direct GPU-to-GPU
// writes through the target card's dynamically-mapped BAR1 aperture,
// wrapped as an allocatable per-device VMM pool.
//
// Mechanism (byte-verified on dual RTX 3080, patched 580.178.04,
// BarlinkPeerBar1=1, dmabuf_holder.ko, iommu=pt -- see
// barlink-torch/BUILD_AND_TEST.md and bench/bar1-p2p-write/main.cu, from
// which this file is derived):
//
//   1. VMM allocation per device (cuMemCreate/cuMemAddressReserve/cuMemMap/
//      cuMemSetAccess) = the pool.
//   2. dma-buf export via RM ioctl NV_ESC_EXPORT_TO_DMABUF_FD (GeForce
//      rejects cuMemGetHandleForAddressRange(DMA_BUF_FD)).
//   3. /dev/dmabuf_holder HOLD with the WRITER card's BDF as importer ->
//      nv_dma_buf_map() programs the pool's BAR1 pages dynamically.
//   4. BAR1 (resource1_wc) mmap + cudaHostRegister(IoMemory) on the writer
//      card -> writer-side device pointer. One such write path per
//      (owner pool, writer device) pair.
//   5. Writers use a grid-stride kernel with st.global.wt 128-bit stores;
//      verification reads on the OWNER card with ld.relaxed.sys through its
//      own VMM pointer (the copy engine can return stale L2 data: the
//      receiving card's L2 is NOT coherent with incoming PCIe writes --
//      barlink-pcie/findings/l2-not-coherent.md).
//
// HARD RULE (measured, do not violate): a receiver-side kernel must NEVER
// poll payload data written by the peer -- the receiver's L2 is not
// coherent with inbound PCIe writes and a spinning kernel keeps reading the
// stale line (barlink-pcie/findings/l2-not-coherent.md). Synchronization
// uses a marker flag in the pool's flag tail: the writer's LAST posted
// store is the flag itself (same PCIe path, in-order delivery), and the
// reader polls only the flag, with ld.relaxed.sys, which (measured) sees
// DOES see inbound writes (l2-coherence-bypassable.md, ~3.2 us).
//
// This file has no torch dependency and compiles standalone:
//   nvcc -O3 -std=c++14 -gencode arch=compute_86,code=sm_86 \
//        -c core.cu -o /tmp/core.o
//
// Driver branch selection: -DDRV_BRANCH=580|595 (ABI is byte-identical
// between the branches; the switch only selects the fallback version
// string for the RM handshake, see extractVersion below).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/capability.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include "core.h"

// ---------------------------------------------------------------------------
// Driver branch (fallback version string only; ABI identical, see header)
// ---------------------------------------------------------------------------

#ifndef DRV_BRANCH
#define DRV_BRANCH 595
#endif

#if DRV_BRANCH == 580
#define DRV_BRANCH_NAME    "580"
#define DRV_BRANCH_VERSION "580.178.04"
#elif DRV_BRANCH == 595
#define DRV_BRANCH_NAME    "595"
#define DRV_BRANCH_VERSION "595.104.02"
#else
#error "unsupported DRV_BRANCH (use 580 or 595)"
#endif

// ---------------------------------------------------------------------------
// Inlined kernel-interface definitions (source: open-gpu-kernel-modules
// headers; plain C types, natural alignment == kernel layout; byte-identical
// between the 580 and 595 branches -- verified by diff)
// ---------------------------------------------------------------------------

typedef uint8_t  NvU8;
typedef uint16_t NvU16;
typedef uint32_t NvU32;
typedef int32_t  NvS32;
typedef uint64_t NvU64;
typedef uint32_t NvHandle;
typedef uint8_t  NvBool;
typedef uint64_t NvP64;

#define NV_IOCTL_MAGIC               'F'
#define NV_IOCTL_BASE                200
#define NV_ESC_CARD_INFO             (NV_IOCTL_BASE + 0)
#define NV_ESC_CHECK_VERSION_STR     (NV_IOCTL_BASE + 10)
#define NV_ESC_EXPORT_TO_DMABUF_FD   (NV_IOCTL_BASE + 17)
#define NV_ESC_RM_CONTROL            0x2A
#define NV_ESC_RM_ALLOC              0x2B

#define NV_RM_API_VERSION_STRING_LENGTH 64
#define NV_RM_API_VERSION_CMD_RELAXED   '1'

typedef struct {
    NvU32 cmd;
    NvU32 reply;
    char  versionString[NV_RM_API_VERSION_STRING_LENGTH];
} nv_ioctl_rm_api_version_t;

typedef struct {
    NvU32 domain;
    NvU8  bus;
    NvU8  slot;
    NvU8  function;
    NvU16 vendor_id;
    NvU16 device_id;
} nv_pci_info_t;

typedef struct {
    NvBool        valid;
    nv_pci_info_t pci_info;
    NvU32         gpu_id;
    NvU16         interrupt_line;
    NvU64         reg_address;
    NvU64         reg_size;
    NvU64         fb_address;
    NvU64         fb_size;
    NvU32         minor_number;
    NvU8          dev_name[10];
} nv_ioctl_card_info_t;

#define NV_DMABUF_EXPORT_MAX_HANDLES 128
#define NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT 0

typedef struct {
    int      fd;
    NvHandle hClient;
    NvU32    totalObjects;
    NvU32    numObjects;
    NvU32    index;
    NvU64    totalSize;
    NvU8     mappingType;
    NvBool   bAllowMmap;
    NvHandle handles[NV_DMABUF_EXPORT_MAX_HANDLES];
    NvU64    offsets[NV_DMABUF_EXPORT_MAX_HANDLES];
    NvU64    sizes[NV_DMABUF_EXPORT_MAX_HANDLES];
    NvU32    status;
} nv_ioctl_export_to_dma_buf_fd_t;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvU32    hClass;
    NvP64    pAllocParms;
    NvU32    paramsSize;
    NvU32    status;
} NVOS21_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hObject;
    NvU32    cmd;
    NvU32    flags;
    NvP64    params;
    NvU32    paramsSize;
    NvU32    status;
} NVOS54_PARAMETERS;

#define NV01_ROOT     0x0U
#define NV01_DEVICE_0 0x80U

typedef struct {
    NvU32    deviceId;
    NvHandle hClientShare;
    NvHandle hTargetClient;
    NvHandle hTargetDevice;
    NvU32    flags;
    NvU64    vaSpaceSize;
    NvU64    vaStartInternal;
    NvU64    vaLimitInternal;
    NvU32    vaMode;
} NV0080_ALLOC_PARAMETERS;

#define NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2 0x205U
typedef struct {
    NvU32 gpuId;
    NvU32 gpuFlags;
    NvU32 deviceInstance;
    NvU32 subDeviceInstance;
    NvU32 sliStatus;
    NvU32 boardId;
    NvU32 gpuInstance;
    NvS32 numaId;
} NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS;

#define NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD 0x3d06U
typedef struct {
    NvU32 type;
    struct {
        NvHandle hDevice;
        NvHandle hParent;
        NvHandle hObject;
    } rmObject;
} NV0000_CTRL_OS_UNIX_EXPORT_OBJECT;

typedef struct {
    NvS32 fd;
    NV0000_CTRL_OS_UNIX_EXPORT_OBJECT object;
} NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS;

// dmabuf_holder ABI (source: barlink-pcie/dmabuf_holder/dmabuf_holder.h)
#define DMABUF_HOLDER_DEVICE_PATH "/dev/dmabuf_holder"
#define DMABUF_HOLDER_F_BDF_VALID (1u << 0)

struct dmabuf_holder_sg_entry {
    uint64_t dma_address;
    uint64_t dma_len;
};

struct dmabuf_holder_hold {
    int32_t dmabuf_fd;
    uint32_t flags;
    uint32_t pci_domain;
    uint8_t  pci_bus;
    uint8_t  pci_slot;
    uint8_t  pci_func;
    uint8_t  reserved0;
    uint32_t max_entries;
    uint32_t reserved1;
    uint64_t entries;
    uint32_t handle;
    uint32_t nents;
    uint64_t dmabuf_size;
    uint64_t total_len;
};

struct dmabuf_holder_release {
    uint32_t handle;
    uint32_t reserved;
};

#define DMABUF_HOLDER_IOC_MAGIC 0xDB
#define DMABUF_HOLDER_IOC_HOLD \
    _IOWR(DMABUF_HOLDER_IOC_MAGIC, 1, struct dmabuf_holder_hold)
#define DMABUF_HOLDER_IOC_RELEASE \
    _IOW(DMABUF_HOLDER_IOC_MAGIC, 2, struct dmabuf_holder_release)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__host__ __device__ static inline uint32_t patVal(uint64_t byteOff, unsigned seed, int lane)
{
    uint64_t x = byteOff ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(seed + 1));
    return (uint32_t)(x >> (lane * 16)) ^ (uint32_t)((x * 0x85ebca6bULL) >> 32);
}

__device__ __forceinline__ static void stwt128(void *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// ld.relaxed.sys on peer-written data: l2-coherence-bypassable.md measured
// that ld.global.cv does NOT reliably see inbound PCIe writes when the line
// is resident in L2 (it kept hitting the stale line for 46M polls), while
// ld.volatile / ld.relaxed.sys do (~20 polls, ~3.2 us). All reads of
// inbound-written data (peer scratch, flags, readback) MUST use this.
__device__ __forceinline__ static uint4 ldcs128(const void *p)
{
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

// Marker-flag async copy protocol
// -------------------------------
// Flags live in the OWNER pool's reserved 4 KiB tail (see Pool::usableSize);
// each incoming direction (one per writer device) owns a 256-byte slot:
//   +0: flag (u64 seq) -- payload-complete marker
//   +8: arm  (u64 seq) -- buffer-ready marker
// Both are written REMOTELY by the writer through its BAR path (.wt store
// after a system fence) and polled LOCALLY by the owner with
// ld.relaxed.sys -- the combination measured to see inbound PCIe writes
// (l2-coherence-bypassable.md: local ld.volatile / ld.relaxed.sys polls see
// inbound writes in ~3.2 us; BAR-view polling and ld.global.cv do NOT
// reliably work).
//
// The arm handshake closes the fill race that exists in ANY cross-process
// design: a local write to a buffer (the user's fill) is only ordered on the
// owner's stream, so the peer's remote write to the same buffer could
// otherwise land first and be overwritten. Per exchange step (seq k), on
// each rank, all on one stream:
//   1. writer stores arm = k into the PEER's arm slot   -- stream-ordered
//      after the caller's local fills, fence covers them
//   2. writer polls its LOCAL arm slot >= k              -- written by the
//      peer's step-1 store; passes only when the peer's fills are done
//   3. writer k_copy payload + stores completion = k into the peer's
//      completion slot (same stream: hard kernel ordering + in-order PCIe
//      posted-write delivery make the flag arrive after the payload)
//   4. owner polls its LOCAL completion flag >= k        -- payload landed
// Step-1 stores always precede step-2 waits on both sides, so the handshake
// cannot deadlock; a rank running ahead simply spins in step 2.
//
// HARD REQUIREMENT (peer mode): dst and src must be DIFFERENT pool
// buffers. With dst == src at the same offset, my payload overwrites the
// peer's source buffer before the peer reads it (the two directions both
// target that offset), and the peer ends up sending my own data back --
// observed as "the receiver reads its own fill". bl_copy_ rejects it.
//
// The reader polls the flag through its LOCAL VMM pointer with ld.relaxed.sys
// (bypasses L2; a spinning kernel CAN see inbound peer writes that way --
// barlink-pcie/findings/l2-coherence-bypassable.md, ~3.2 us). It never
// spins on payload. The flag value is monotone (u64 seq, 0 = never), so
// multiple in-flight copies are safe: flag >= seq for a later seq implies
// the earlier copy's payload landed too.

__device__ __forceinline__ static void stwt64(void *p, unsigned long long v)
{
    asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Cross-device copy: read local 'src', st.global.wt into the PEER BAR window
// 'dst' (a cudaHostRegister(IoMemory) device pointer on THIS device).
// Pure payload kernel -- no synchronization logic here; k_mark publishes the
// flag after this kernel completes (same stream).
__global__ void k_copy(const uint4 *__restrict__ src, uint4 *__restrict__ dst,
                       size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride)
        stwt128(&dst[i], src[i]);
}

// Arm marker: plain fence + store. The arm only orders the caller's LOCAL
// fills (same-GPU writes the fence fully covers); no payload is in flight
// yet, so no drain read is needed.
// Marker: one thread, fence + store. Used for BOTH the arm (orders the
// caller's local fills) and the completion (issued after the payload
// kernel on the same stream: hard kernel ordering means every payload
// store was issued before this kernel runs, and PCIe delivers posted
// writes from one source in order -- the flag going out after the
// payload). Callers must never use the same pool buffer as dst and src
// in peer mode; see the check in bl_copy_.
__global__ void k_mark(unsigned long long *flag, unsigned long long seq)
{
    __threadfence_system();
    stwt64(flag, seq);
}

// Reader-side wait: poll the local flag with ld.relaxed.sys until flag >= seq.
// One block; timeout ~2 s -> __trap() surfaces as a CUDA error on sync.
// Backoff cap is 4 us: the flag lives in LOCAL memory (the payload crosses
// PCIe, the poll does not), so aggressive polling costs no link bandwidth,
// and the decode-TP path does ~256 of these waits per token -- a deep
// backoff would add wake-up latency to every one of them (measured: ~113 us
// average wait with the old 1 ms cap vs ~3-5 us of true peer skew).
__global__ void k_flag_wait(const unsigned long long *__restrict__ flag,
                            unsigned long long seq)
{
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        unsigned ns = 32;
        for (;;) {
            unsigned long long v;
            // ld.relaxed.sys, NOT ld.global.cv -- see ldcs128 above
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                         : "=l"(v) : "l"(flag) : "memory");
            if (v >= seq) break;
            if (clock64() - t0 > 4LL * 1000 * 1000 * 1000)  // ~2-3 s @ ~1.5-2 GHz
                __trap();
            // pure spin: __nanosleep rounds up to timer ticks (~32 us+),
            // which dominated the wait latency in decode-TP profiling;
            // one spinning warp on a local flag costs nothing on the wire
            (void)ns;
        }
    }
    __syncthreads();
}

// Pattern writer (used by bl_verify): writes patVal(seed) into the peer BAR.
__global__ void k_pattern(uint4 *__restrict__ dst, size_t n4, unsigned seed)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 v;
        v.x = patVal(i * 16, seed, 0);
        v.y = patVal(i * 16, seed, 1);
        v.z = patVal(i * 16, seed, 2);
        v.w = patVal(i * 16, seed, 3);
        stwt128(&dst[i], v);
    }
}

struct BlVerifyOut {
    unsigned long long bad;
    unsigned long long firstOff;
};

// Plain ld.relaxed.sys readback of n4 uint4 elements into a staging buffer
// (used by bl_readback).
__global__ void k_readback(const uint4 *__restrict__ src,
                           uint4 *__restrict__ out, int n4)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) out[i] = ldcs128(&src[i]);
}

// Byte verification on the OWNER card through its own VMM pointer, after the
// writer kernel has fully finished (kernel boundary discards stale L2).
__global__ void k_verify(const uint4 *__restrict__ src, size_t n4,
                         unsigned seed, BlVerifyOut *__restrict__ out)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long bad = 0;
    unsigned long long first = ~0ull;

    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 v = ldcs128(&src[i]);
        uint4 w;
        w.x = patVal(i * 16, seed, 0);
        w.y = patVal(i * 16, seed, 1);
        w.z = patVal(i * 16, seed, 2);
        w.w = patVal(i * 16, seed, 3);
        if (v.x != w.x || v.y != w.y || v.z != w.z || v.w != w.w) {
            const uint8_t *vb = (const uint8_t *)&v;
            const uint8_t *wb = (const uint8_t *)&w;
            for (int k = 0; k < 16; ++k) {
                if (vb[k] != wb[k]) {
                    ++bad;
                    if (i * 16 + (unsigned)k < first)
                        first = i * 16 + (unsigned)k;
                }
            }
        }
    }

    __shared__ unsigned long long shBad[256 / 32];
    __shared__ unsigned long long shFirst[256 / 32];
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)(threadIdx.x >> 5);
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long ob = __shfl_down_sync(~0u, bad, off);
        unsigned long long of = __shfl_down_sync(~0u, first, off);
        bad += ob;
        if (of < first) first = of;
    }
    if (lane == 0) { shBad[warp] = bad; shFirst[warp] = first; }
    __syncthreads();
    if (warp == 0) {
        bad   = (lane < (int)(blockDim.x >> 5)) ? shBad[lane] : 0;
        first = (lane < (int)(blockDim.x >> 5)) ? shFirst[lane] : ~0ull;
        for (int off = 8; off > 0; off >>= 1) {
            unsigned long long ob = __shfl_down_sync(~0u, bad, off);
            unsigned long long of = __shfl_down_sync(~0u, first, off);
            bad += ob;
            if (of < first) first = of;
        }
        if (lane == 0) {
            atomicAdd(&out->bad, bad);
            atomicMin(&out->firstOff, first);
        }
    }
}

// Local wrapping u8 add: a[i] += b[i] (SIMD u8x4 per 32-bit lane). 'b' was
// written by the peer through the BAR, read it with ld.relaxed.sys; 'a' is
// local traffic. Fast path for BL_DTYPE_U8 (kept from the byte-only era).
__global__ void k_add(uint4 *__restrict__ a, const uint4 *__restrict__ b,
                      size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = a[i];
        uint4 y = ldcs128(&b[i]);
        uint4 s;
        s.x = __vadd4(x.x, y.x);
        s.y = __vadd4(x.y, y.y);
        s.z = __vadd4(x.z, y.z);
        s.w = __vadd4(x.w, y.w);
        a[i] = s;
    }
}

// Elementwise add helpers; one overload per supported dtype. bf16/fp8 add in
// float, matching torch's compute semantics for those dtypes.
__device__ __forceinline__ static float  blElemAdd(float a, float b)
{ return a + b; }
__device__ __forceinline__ static double blElemAdd(double a, double b)
{ return a + b; }
__device__ __forceinline__ static __nv_bfloat16 blElemAdd(__nv_bfloat16 a,
                                                          __nv_bfloat16 b)
{
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}
// torch computes half elementwise ops with float opmath and rounds once.
__device__ __forceinline__ static __half blElemAdd(__half a, __half b)
{
    return __float2half(__half2float(a) + __half2float(b));
}

// NOTE: passing __nv_fp8_e4m3/e5m2 BY VALUE through device functions is
// silently miscompiled by nvcc 13.4 at -O2 for sm_86 (1-byte class ABI;
// observed: add of 1.0+3.0 yields the encoding of 72.0). The fp8 adds
// therefore work on raw __nv_fp8_storage_t, never on the class.

// Generic typed add: same 16-byte vector structure as k_add, reinterpreted
// as T[16/sizeof(T)] inside the lane.
template <typename T>
__global__ void k_add_t(uint4 *__restrict__ a, const uint4 *__restrict__ b,
                        size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = a[i];
        uint4 y = ldcs128(&b[i]);
        const T *xp = reinterpret_cast<const T *>(&x);
        const T *yp = reinterpret_cast<const T *>(&y);
        uint4 s;
        T *sp = reinterpret_cast<T *>(&s);
#pragma unroll
        for (int k = 0; k < (int)(16 / sizeof(T)); ++k)
            sp[k] = blElemAdd(xp[k], yp[k]);
        a[i] = s;
    }
}

// fp8 add on raw storage, one interpretation per kernel. Adds in float
// (torch converts fp8 to float for compute, same semantics).
template <__nv_fp8_interpretation_t INTERP>
__global__ void k_add_fp8(uint4 *__restrict__ a, const uint4 *__restrict__ b,
                          size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = a[i];
        uint4 y = ldcs128(&b[i]);
        const __nv_fp8_storage_t *xp =
            reinterpret_cast<const __nv_fp8_storage_t *>(&x);
        const __nv_fp8_storage_t *yp =
            reinterpret_cast<const __nv_fp8_storage_t *>(&y);
        uint4 s;
        __nv_fp8_storage_t *sp =
            reinterpret_cast<__nv_fp8_storage_t *>(&s);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float fa = __half2float(
                __half(__nv_cvt_fp8_to_halfraw(xp[k], INTERP)));
            float fb = __half2float(
                __half(__nv_cvt_fp8_to_halfraw(yp[k], INTERP)));
            sp[k] = __nv_cvt_float_to_fp8(fa + fb, __NV_SATFINITE, INTERP);
        }
        a[i] = s;
    }
}

// Dispatch helper: pick the add kernel for the dtype enum (core.h).
static int launchAdd(int dtype, uint4 *a, const uint4 *b, size_t n4,
                     size_t blocks, cudaStream_t stream)
{
    switch (dtype) {
    case BL_DTYPE_U8:      k_add<<<blocks, 256, 0, stream>>>(a, b, n4); break;
    case BL_DTYPE_FP32:    k_add_t<float><<<blocks, 256, 0, stream>>>(a, b, n4); break;
    case BL_DTYPE_FP64:    k_add_t<double><<<blocks, 256, 0, stream>>>(a, b, n4); break;
    case BL_DTYPE_BF16:    k_add_t<__nv_bfloat16><<<blocks, 256, 0, stream>>>(a, b, n4); break;
    case BL_DTYPE_FP8E4M3: k_add_fp8<__NV_E4M3><<<blocks, 256, 0, stream>>>(a, b, n4); break;
    case BL_DTYPE_FP8E5M2: k_add_fp8<__NV_E5M2><<<blocks, 256, 0, stream>>>(a, b, n4); break;
    default: return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Zero-copy allreduce add kernels: out[i] = elemAdd(in[i], scr[i]).
// 'scr' holds the PEER's contribution, written through the BAR (inbound PCIe
// writes) -- read it with ld.relaxed.sys like every inbound-written buffer.
// 'in'/'out' are plain local device memory (arbitrary torch tensors), so
// 'in' uses plain loads and 'out' plain stores. in == out (in-place) is
// fine: each element is read (in and scr) exactly once before being written.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void k_add_into_t(uint4 *__restrict__ out,
                             const uint4 *__restrict__ in,
                             const uint4 *__restrict__ scr, size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = in[i];
        uint4 y = ldcs128(&scr[i]);
        const T *xp = reinterpret_cast<const T *>(&x);
        const T *yp = reinterpret_cast<const T *>(&y);
        uint4 s;
        T *sp = reinterpret_cast<T *>(&s);
#pragma unroll
        for (int k = 0; k < (int)(16 / sizeof(T)); ++k)
            sp[k] = blElemAdd(xp[k], yp[k]);
        out[i] = s;
    }
}

// fp8 zero-copy add on raw storage (same by-value miscompile workaround as
// k_add_fp8).
template <__nv_fp8_interpretation_t INTERP>
__global__ void k_add_into_fp8(uint4 *__restrict__ out,
                               const uint4 *__restrict__ in,
                               const uint4 *__restrict__ scr, size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = in[i];
        uint4 y = ldcs128(&scr[i]);
        const __nv_fp8_storage_t *xp =
            reinterpret_cast<const __nv_fp8_storage_t *>(&x);
        const __nv_fp8_storage_t *yp =
            reinterpret_cast<const __nv_fp8_storage_t *>(&y);
        uint4 s;
        __nv_fp8_storage_t *sp =
            reinterpret_cast<__nv_fp8_storage_t *>(&s);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float fa = __half2float(
                __half(__nv_cvt_fp8_to_halfraw(xp[k], INTERP)));
            float fb = __half2float(
                __half(__nv_cvt_fp8_to_halfraw(yp[k], INTERP)));
            sp[k] = __nv_cvt_float_to_fp8(fa + fb, __NV_SATFINITE, INTERP);
        }
        out[i] = s;
    }
}

// Zero-copy recv move: read MY scratch zone (written by the peer through
// the BAR -- inbound PCIe writes) with ld.relaxed.sys and write the user's
// 'out' with plain local stores. Pure copy, no dtype logic.
__global__ void k_move(uint4 *__restrict__ out, const uint4 *__restrict__ scr,
                       size_t n4)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride)
        out[i] = ldcs128(&scr[i]);
}

// Dispatch for the zero-copy add. fp16 is native here -- the whole point of
// the zero-copy path is that it needs no pool-resident add dtype.
static int launchAddInto(int dtype, uint4 *out, const uint4 *in,
                         const uint4 *scr, size_t n4, size_t blocks,
                         cudaStream_t stream)
{
    switch (dtype) {
    case BL_DTYPE_FP16:    k_add_into_t<__half><<<blocks, 256, 0, stream>>>(
                               out, in, scr, n4); break;
    case BL_DTYPE_FP32:    k_add_into_t<float><<<blocks, 256, 0, stream>>>(
                               out, in, scr, n4); break;
    case BL_DTYPE_FP64:    k_add_into_t<double><<<blocks, 256, 0, stream>>>(
                               out, in, scr, n4); break;
    case BL_DTYPE_BF16:    k_add_into_t<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
                               out, in, scr, n4); break;
    case BL_DTYPE_FP8E4M3: k_add_into_fp8<__NV_E4M3><<<blocks, 256, 0, stream>>>(
                               out, in, scr, n4); break;
    case BL_DTYPE_FP8E5M2: k_add_into_fp8<__NV_E5M2><<<blocks, 256, 0, stream>>>(
                               out, in, scr, n4); break;
    default: return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static void setErr(char *err, size_t errlen, const std::string &msg)
{
    if (err && errlen) {
        std::snprintf(err, errlen, "%s", msg.c_str());
    }
}

static std::string lower(std::string s)
{
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

struct Bdf {
    unsigned domain = 0, bus = 0, slot = 0, func = 0;
    bool valid = false;
};

static Bdf parseBdf(const char *s)
{
    Bdf b;
    unsigned d, u, v, f;
    if (std::sscanf(s, "%x:%x:%x.%x", &d, &u, &v, &f) == 4) {
        b.domain = d; b.bus = u; b.slot = v; b.func = f; b.valid = true;
    } else if (std::sscanf(s, "%x:%x.%x", &u, &v, &f) == 3) {
        b.domain = 0; b.bus = u; b.slot = v; b.func = f; b.valid = true;
    }
    return b;
}

static bool readBarRange(const std::string &bdf, int idx,
                         uint64_t *start, uint64_t *end)
{
    std::string path = "/sys/bus/pci/devices/" + bdf + "/resource";
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    bool ok = false;
    char line[256];
    for (int i = 0; std::fgets(line, sizeof(line), f); ++i) {
        if (i != idx) continue;
        unsigned long long a = 0, b = 0, fl = 0;
        if (std::sscanf(line, "%llx %llx %llx", &a, &b, &fl) >= 2 && b > a) {
            *start = (uint64_t)a;
            *end   = (uint64_t)b;
            ok = true;
        }
        break;
    }
    std::fclose(f);
    return ok;
}

static uint64_t barScan(int fd, uint64_t barSize, const uint8_t *magic,
                        size_t magicLen)
{
    const size_t pageSize = (size_t)sysconf(_SC_PAGESIZE);
    size_t chunk = 4ull << 20;
    std::vector<uint8_t> tmp(magicLen);
    uint64_t off = 0;

    while (off < barSize) {
        size_t len = (size_t)((barSize - off < chunk) ? (barSize - off) : chunk);
        void *p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t)off);
        if (p == MAP_FAILED) {
            if (chunk > pageSize) {
                chunk = (chunk / 2 / pageSize) * pageSize;
                if (chunk < pageSize) chunk = pageSize;
                continue;
            }
            break;
        }
        const volatile uint8_t *b = (const volatile uint8_t *)p;
        for (size_t i = 0; i + magicLen <= len; i += 4096) {
            if (b[i] != magic[0]) continue;
            for (size_t k = 0; k < magicLen; ++k) tmp[k] = b[i + k];
            if (std::memcmp(tmp.data(), magic, magicLen) == 0) {
                munmap(p, len);
                return off + (uint64_t)i;
            }
        }
        munmap(p, len);
        off += len;
    }
    return (uint64_t)-1;
}

// ---------------------------------------------------------------------------
// RM ioctl chain: dma-buf export (mirror of the bench implementation)
// ---------------------------------------------------------------------------

struct NvExport {
    int      ctlFd = -1;
    int      devFd = -1;
    NvHandle hClient = 0;
    NvHandle hDevice = 0xbee00001;
    NvHandle hMemory = 0xbee00010;
    int      dmabufFd = -1;
};

static int nvIoctl(int fd, int nr, void *p, size_t size)
{
    return ioctl(fd, _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, nr,
                          (unsigned)size), p);
}

static bool extractVersion(const char *buf, char *ver, size_t len)
{
    (void)len;
    // 595 open module: "... UNIX Open Kernel Module for x86_64  595.104.02 ..."
    // 580 stock:      "... UNIX x86_64 Kernel Module  580.178.04 ..."
    const char *p = std::strstr(buf, "for x86_64");
    if (p && std::sscanf(p, "for x86_64 %63s", ver) == 1) return true;
    if ((p = std::strstr(buf, "x86_64 Kernel Module")) != nullptr &&
        std::sscanf(p, "x86_64 Kernel Module %63s", ver) == 1) return true;
    for (p = buf; *p; ++p) {
        if (!isdigit((unsigned char)*p)) continue;
        if (std::sscanf(p, "%63[0-9.]", ver) == 1 && std::strchr(ver, '.'))
            return true;
    }
    return false;
}

static bool nvExportToDmabuf(NvExport &e, int objfd, int pciBus, size_t size,
                             std::string &err)
{
    char ebuf[512];

    e.ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (e.ctlFd < 0) {
        std::snprintf(ebuf, sizeof(ebuf),
            "open /dev/nvidiactl: %s (is the NVIDIA driver loaded?)",
            std::strerror(errno));
        err = ebuf;
        return false;
    }

    {   // version handshake (NV_ESC_CHECK_VERSION_STR)
        nv_ioctl_rm_api_version_t v;
        char buf[256] = {0}, ver[64] = {0};
        std::memset(&v, 0, sizeof(v));
        v.cmd = NV_RM_API_VERSION_CMD_RELAXED;
        FILE *f = std::fopen("/proc/driver/nvidia/version", "r");
        if (f) { if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0; std::fclose(f); }
        if (!extractVersion(buf, ver, sizeof(ver)))
            std::strncpy(ver, DRV_BRANCH_VERSION, sizeof(ver) - 1);
        std::strncpy(v.versionString, ver, sizeof(v.versionString) - 1);
        if (nvIoctl(e.ctlFd, NV_ESC_CHECK_VERSION_STR, &v, sizeof(v)) < 0) {
            std::snprintf(ebuf, sizeof(ebuf),
                "NV_ESC_CHECK_VERSION_STR failed: %s. Version/branch mismatch "
                "(this binary built for branch %s); check "
                "'cat /proc/driver/nvidia/version'. This is NOT the BAR1 "
                "guard (that fails later, at cudaHostRegister).",
                std::strerror(errno), DRV_BRANCH_VERSION);
            err = ebuf;
            return false;
        }
    }

    NvU32 gpuId = 0, minor = 0;
    {
        nv_ioctl_card_info_t ci[32];
        std::memset(ci, 0, sizeof(ci));
        if (nvIoctl(e.ctlFd, NV_ESC_CARD_INFO, ci, sizeof(ci)) < 0) {
            std::snprintf(ebuf, sizeof(ebuf), "NV_ESC_CARD_INFO: %s",
                          std::strerror(errno));
            err = ebuf;
            return false;
        }
        bool found = false;
        for (int i = 0; i < 32; ++i) {
            if (!ci[i].valid) continue;
            if ((int)ci[i].pci_info.bus == pciBus) {
                gpuId = ci[i].gpu_id; minor = ci[i].minor_number; found = true;
            }
        }
        if (!found) {
            std::snprintf(ebuf, sizeof(ebuf),
                          "PCI bus 0x%02x not found in NV_ESC_CARD_INFO", pciBus);
            err = ebuf;
            return false;
        }
    }

    {
        NVOS21_PARAMETERS a;
        std::memset(&a, 0, sizeof(a));
        a.hClass = NV01_ROOT;
        if (nvIoctl(e.ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0) {
            std::snprintf(ebuf, sizeof(ebuf), "RM_ALLOC(root) status=0x%x", a.status);
            err = ebuf;
            return false;
        }
        e.hClient = a.hObjectNew;
    }

    NvU32 devInst = 0;
    {
        NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.gpuId    = gpuId;
        c.hClient  = e.hClient; c.hObject = e.hClient;
        c.cmd      = NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2;
        c.params   = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(e.ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0) {
            std::snprintf(ebuf, sizeof(ebuf), "GPU_GET_ID_INFO_V2 status=0x%x", c.status);
            err = ebuf;
            return false;
        }
        devInst = p.deviceInstance;
    }

    {
        NV0080_ALLOC_PARAMETERS dp;
        NVOS21_PARAMETERS a;
        std::memset(&dp, 0, sizeof(dp)); std::memset(&a, 0, sizeof(a));
        dp.deviceId = devInst; dp.hClientShare = e.hClient;
        a.hRoot = e.hClient; a.hObjectParent = e.hClient; a.hObjectNew = e.hDevice;
        a.hClass = NV01_DEVICE_0;
        a.pAllocParms = (NvP64)(uintptr_t)&dp;
        if (nvIoctl(e.ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0) {
            std::snprintf(ebuf, sizeof(ebuf), "RM_ALLOC(device) status=0x%x", a.status);
            err = ebuf;
            return false;
        }
    }

    {
        NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.fd = objfd;
        p.object.type = 1;  // RM object
        p.object.rmObject.hDevice = e.hDevice;
        p.object.rmObject.hParent = e.hDevice;
        p.object.rmObject.hObject = e.hMemory;
        c.hClient = e.hClient; c.hObject = e.hClient;
        c.cmd     = NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD;
        c.params  = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(e.ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0) {
            std::snprintf(ebuf, sizeof(ebuf), "IMPORT_OBJECT_FROM_FD status=0x%x", c.status);
            err = ebuf;
            return false;
        }
    }

    {
        char devpath[64];
        std::snprintf(devpath, sizeof(devpath), "/dev/nvidia%u", minor);
        e.devFd = open(devpath, O_RDWR);
        if (e.devFd < 0) {
            std::snprintf(ebuf, sizeof(ebuf), "open %s: %s", devpath,
                          std::strerror(errno));
            err = ebuf;
            return false;
        }

        nv_ioctl_export_to_dma_buf_fd_t p;
        std::memset(&p, 0, sizeof(p));
        p.fd           = -1;
        p.hClient      = e.hClient;
        p.totalObjects = 1;
        p.numObjects   = 1;
        p.index        = 0;
        p.totalSize    = size;
        p.mappingType  = NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT;
        p.bAllowMmap   = 0;
        p.handles[0]   = e.hMemory;
        p.offsets[0]   = 0;
        p.sizes[0]     = size;

        if (nvIoctl(e.devFd, NV_ESC_EXPORT_TO_DMABUF_FD, &p, sizeof(p)) < 0) {
            std::snprintf(ebuf, sizeof(ebuf), "NV_ESC_EXPORT_TO_DMABUF_FD: %s",
                          std::strerror(errno));
            err = ebuf;
            return false;
        }
        if (p.status != 0) {
            std::snprintf(ebuf, sizeof(ebuf),
                          "NV_ESC_EXPORT_TO_DMABUF_FD status=0x%08x", p.status);
            err = ebuf;
            return false;
        }
        e.dmabufFd = p.fd;
    }
    return true;
}

static void nvExportClose(NvExport &e)
{
    if (e.dmabufFd >= 0) { close(e.dmabufFd); e.dmabufFd = -1; }
    if (e.devFd >= 0)    { close(e.devFd);    e.devFd = -1; }
    if (e.ctlFd >= 0)    { close(e.ctlFd);    e.ctlFd = -1; }
}

// ---------------------------------------------------------------------------
// Pool and context
// ---------------------------------------------------------------------------

#define BL_MAX_DEVICES 8

// One (owner pool, writer device) path: the writer's device pointer into
// the owner's BAR1 window.
struct WritePath {
    int   writerIdx = -1;    // index into ctx->devices
    void *devPtr = nullptr;  // device pointer on the writer card
    void *map = nullptr;     // this path's private mmap (unregistered base)
    size_t mapLen = 0;
    bool   registered = false;
};

struct Pool {
    int      devIdx = -1;    // index into ctx->devices
    CUdeviceptr dptr = 0;    // VMM pointer on the owner card
    CUmemGenericAllocationHandle memHandle = 0;
    size_t   size = 0;       // total VMM allocation
    size_t   usableSize = 0; // user-allocatable limit (flag tail always
                             // excluded; in peer mode also the scratch half)

    NvExport nvx;
    int      dmabufFd = -1;
    int      holderFd = -1;
    uint32_t holderHandle = 0;
    int      barFd = -1;
    uint64_t barOff = 0;     // buffer start within the BAR1 aperture
    uint64_t barSize = 0;    // BAR1 aperture size (from resource1_wc fstat)
    uint64_t bar1Start = 0, bar1End = 0;  // BAR1 physical range (sysfs)
    bool     haveBar1Phys = false;
    uint8_t  magic[64] = {0};              // marker for the BAR1 scan fallback

    std::vector<WritePath> paths;

    // bump + first-fit free-list allocator, 2 MiB alignment
    std::mutex allocMu;
    size_t bump = 0;
    std::vector<std::pair<size_t, size_t>> freeList;  // (offset, size)
};

struct blCtx {
    int    devices[BL_MAX_DEVICES];
    int    ndev = 0;
    size_t poolBytes = 0;
    Pool   pools[BL_MAX_DEVICES];

    // marker-flag protocol state (see k_copy): per-direction sequence
    // counters; dirSeq[w] = copies issued BY writer w (v1: one writer per
    // pool, so this is also the incoming seq of the other pool).
    std::mutex seqMu;
    uint64_t dirSeq[BL_MAX_DEVICES] = {0};
    uint64_t lastIncoming[BL_MAX_DEVICES] = {0};  // highest seq landed per pool

    // cross-process (SPMD peer) mode: this process owns exactly ONE GPU,
    // devices[myRank]; devices[1-myRank] is the peer (ordinal -1 locally).
    // pools[myRank] is the real local pool; pools[peer] is a pseudo-entry
    // holding only the BAR1 write path into the peer's pool.
    bool     peerMode = false;
    int      myRank = -1;
    size_t   scratchBase = 0;   // peer mode: user tensors live below this
    size_t   p2pScratchBase = 0; // peer mode: p2p payloads live in the TOP
                                 // half of the scratch zone; allreduce keeps
                                 // the bottom half at scratchBase (fixed
                                 // offset) -- the partition is what makes
                                 // mixed p2p/allreduce sequences race-free

    // p2p protocol state (bl_send_into_peer / bl_recv_into_peer): DEDICATED
    // flag slots -- slot index 2 + writer rank (completion at +0 is written
    // by the SENDER, consumed-receipt at +8 by the RECEIVER after its move
    // kernel) -- and SEPARATE send/recv ordinals (a full-duplex pair posting
    // send-then-recv on both ranks cross-pairs a shared counter: my send#k
    // pairs the peer's recv#k, not its k-th op of any kind).
    //
    // Sends run on their OWN stream (recv on the caller's stream): on one
    // stream, a rank posting several sends before its recvs deadlocks with
    // the peer doing the same -- the second send's consumed-receipt wait
    // would spin ahead of the recv that posts the receipt. The send stream
    // is joined to the caller's stream at enqueue time (payload sees all
    // prior caller-stream work); collectives on the caller's stream join the
    // send stream through evSend so an add cannot overtake a pending payload
    // that reads a tensor the add writes.
    uint64_t p2pSendSeq = 0;
    uint64_t p2pRecvSeq = 0;
    cudaStream_t sendStream = nullptr;   // lazily created (non-blocking)
    cudaEvent_t  evSend = nullptr;       // recorded after each send

    // fused small-message allreduce (<= BL_FUSED_MAX): per-direction ordinal
    // and a dedicated scratch slice at the TOP of the allreduce half (so
    // fused payloads never share memory with the arm-path payloads -- mixed
    // fused/old sequences need no cross-protocol gating). Receipt (+0) and
    // flag (+8) slots: BL_AR_RECV_SLOT / BL_AR_FLAG_SLOT of each pool, both
    // written by the peer through the BAR.
    uint64_t fusedSeq = 0;
    size_t   fusedZoneBase = 0;  // == end of the shrunken old-path zone
};

// Reserved pool tail for the flag slots (one 256-byte slot per writer
// direction; u64 flag at +0, u64 done counter at +8). Written by peers
// through the BAR, polled locally with ld.relaxed.sys.
#define BL_FLAG_REGION 4096u
#define BL_FLAG_SLOT   256u
static size_t flagOff(const Pool &P, int writerIdx)
{
    return P.size - BL_FLAG_REGION + (size_t)writerIdx * BL_FLAG_SLOT;
}

// defined with the p2p primitives below; used by the collectives to order
// themselves after pending p2p payloads
static void joinSendStream(blCtx *ctx, cudaStream_t cs);

static const size_t kAllocAlign = 2ull << 20;

static size_t gridBlocks(size_t n4, int devOrd)
{
    // Launch the FULL grid so the whole range is covered in ONE grid-stride
    // sweep. Empirically (dual 3080, patched 580, iommu=pt) stores from the
    // second+ sweep of a .wt BAR1 write kernel never land -- the first sweep
    // covers exactly gridDim*blockDim*16 bytes and the rest is dropped. The
    // standalone bench never hit this because it always launches (n4+255)/256
    // blocks. See tests/test_basic.py history.
    (void)devOrd;
    size_t need = (n4 + 255) / 256;
    const size_t kMaxBlocks = 1u << 20;   // generous cap; 64 MiB pool needs 16384
    return need < kMaxBlocks ? need : kMaxBlocks;
}

// ---------------------------------------------------------------------------
// Pool allocator
// ---------------------------------------------------------------------------

static bool poolAllocLocked(Pool &p, size_t bytes, size_t *offOut, std::string &err)
{
    bytes = (bytes + 15) & ~(size_t)15;
    // first fit in the free list; never touch the flag tail
    for (size_t i = 0; i < p.freeList.size(); ++i) {
        size_t o = (p.freeList[i].first + kAllocAlign - 1) & ~(kAllocAlign - 1);
        size_t end = o + bytes;
        if (end <= p.freeList[i].first + p.freeList[i].second &&
            end <= p.usableSize) {
            size_t tail0 = p.freeList[i].first;
            size_t tailLen = p.freeList[i].second - (o - p.freeList[i].first) - bytes;
            p.freeList.erase(p.freeList.begin() + i);
            if (o > tail0) p.freeList.emplace_back(tail0, o - tail0);
            if (tailLen)   p.freeList.emplace_back(end, tailLen);
            *offOut = o;
            return true;
        }
    }
    size_t o = (p.bump + kAllocAlign - 1) & ~(kAllocAlign - 1);
    if (o + bytes > p.usableSize) {
        char b[320];
        std::snprintf(b, sizeof(b),
            "pool on device %d exhausted: need %zu more, usable pool size "
            "%zu (4 KiB flag tail reserved). Free tensors cannot be reclaimed "
            "(pool tensors are not freed individually); increase pool_mb.",
            p.devIdx, bytes, p.usableSize);
        err = b;
        return false;
    }
    if (o > p.bump) p.freeList.emplace_back(p.bump, o - p.bump);
    p.bump = o + bytes;
    *offOut = o;
    return true;
}

static void poolFreeLocked(Pool &p, size_t off, size_t bytes)
{
    bytes = (bytes + 15) & ~(size_t)15;
    p.freeList.emplace_back(off, bytes);
    // coalesce
    for (size_t i = 0; i < p.freeList.size(); ++i) {
        for (size_t j = i + 1; j < p.freeList.size(); ++j) {
            size_t a0 = p.freeList[i].first, a1 = a0 + p.freeList[i].second;
            size_t b0 = p.freeList[j].first, b1 = b0 + p.freeList[j].second;
            if (a1 == b0 || b1 == a0) {
                p.freeList[i] = { a0 < b0 ? a0 : b0, a1 > b1 ? a1 - (a0 < b0 ? a0 : b0) : b1 - (a0 < b0 ? a0 : b0) };
                p.freeList.erase(p.freeList.begin() + j);
                j = i;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Setup: one pool + all its write paths
// ---------------------------------------------------------------------------

// Local half of pool setup: context creation on the owner card, VMM pool
// allocation, dma-buf export, and opening the owner's BAR1 aperture.
// Shared by single-process bl_init and cross-process bl_init_peer.
static bool poolLocalSetup(blCtx *ctx, int devIdx, char *err, size_t errlen)
{
    std::string e;
    char b[512];
    Pool &P = ctx->pools[devIdx];
    P.devIdx = devIdx;
    const int ord = ctx->devices[devIdx];

    cudaError_t ce = cudaSetDevice(ord);
    if (ce != cudaSuccess) {
        std::snprintf(b, sizeof(b), "cudaSetDevice(%d): %s",
                      ord, cudaGetErrorString(ce));
        setErr(err, errlen, b);
        return false;
    }
    ce = cudaFree(nullptr);
    if (ce != cudaSuccess) {
        std::snprintf(b, sizeof(b), "cudaFree(0) on device %d: %s",
                      ord, cudaGetErrorString(ce));
        setErr(err, errlen, b);
        return false;
    }

    CUdevice dev = 0;
    if (cuDeviceGet(&dev, ord) != CUDA_SUCCESS) {
        std::snprintf(b, sizeof(b), "cuDeviceGet(%d) failed", ord);
        setErr(err, errlen, b);
        return false;
    }

    // 1. VMM pool on the owner card
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(prop));
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
            != CUDA_SUCCESS || gran == 0)
        gran = 2ull << 20;
    P.size = (ctx->poolBytes + gran - 1) / gran * gran;
    P.usableSize = P.size - BL_FLAG_REGION;   // flag slots live in the tail

    if (cuMemCreate(&P.memHandle, P.size, &prop, 0) != CUDA_SUCCESS) {
        std::snprintf(b, sizeof(b),
            "cuMemCreate(%zu) on device %d failed", P.size, ord);
        setErr(err, errlen, b);
        return false;
    }
    if (cuMemAddressReserve(&P.dptr, P.size, gran, 0, 0) != CUDA_SUCCESS ||
        cuMemMap(P.dptr, P.size, 0, P.memHandle, 0) != CUDA_SUCCESS) {
        setErr(err, errlen, "cuMemAddressReserve/cuMemMap failed");
        return false;
    }
    CUmemAccessDesc acc;
    std::memset(&acc, 0, sizeof(acc));
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id   = dev;
    acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(P.dptr, P.size, &acc, 1) != CUDA_SUCCESS) {
        setErr(err, errlen, "cuMemSetAccess failed");
        return false;
    }

    // marker for the BAR1 scan fallback
    std::snprintf((char *)P.magic, sizeof(P.magic), "BARLINK-SM86-%08x%08x",
                  (unsigned)getpid(), (unsigned)time(nullptr));
    cuMemcpyHtoD(P.dptr, P.magic, sizeof(P.magic));
    cuCtxSynchronize();

    // 2. dma-buf export
    int dmabufFd = -1;
    if (cuMemGetHandleForAddressRange(&dmabufFd, P.dptr, P.size,
                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                      0) == CUDA_SUCCESS) {
        P.dmabufFd = dmabufFd;
    } else {
        int shareFd = -1;
        if (cuMemExportToShareableHandle(&shareFd, P.memHandle,
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                         0) != CUDA_SUCCESS) {
            setErr(err, errlen,
                   "cuMemExportToShareableHandle failed (CUDA VMM export)");
            return false;
        }
        int pciBus = -1;
        cuDeviceGetAttribute(&pciBus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
        if (!nvExportToDmabuf(P.nvx, shareFd, pciBus, P.size, e)) {
            std::snprintf(b, sizeof(b), "RM ioctl dma-buf export failed: %s", e.c_str());
            setErr(err, errlen, b);
            close(shareFd);
            return false;
        }
        P.dmabufFd = P.nvx.dmabufFd;
        close(shareFd);
    }

    // BAR1 aperture of the owner card
    char busId[64] = {0};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), ord) != cudaSuccess) {
        setErr(err, errlen, "cudaDeviceGetPCIBusId failed");
        return false;
    }
    std::string bdf = lower(busId);
    std::string barPath = "/sys/bus/pci/devices/" + bdf + "/resource1_wc";
    P.barFd = open(barPath.c_str(), O_RDWR | O_SYNC);
    if (P.barFd < 0) {
        std::snprintf(b, sizeof(b), "open(%s): %s", barPath.c_str(),
                      std::strerror(errno));
        setErr(err, errlen, b);
        return false;
    }
    struct stat st;
    if (fstat(P.barFd, &st) != 0) {
        std::snprintf(b, sizeof(b), "fstat(%s): %s", barPath.c_str(),
                      std::strerror(errno));
        setErr(err, errlen, b);
        return false;
    }
    P.barSize = (uint64_t)st.st_size;
    P.haveBar1Phys = readBarRange(bdf, 1, &P.bar1Start, &P.bar1End);
    return true;
}

// HOLD the pool's dma-buf as 'attachTo' and derive the pool's offset inside
// its BAR1 aperture (sg table; marker-scan fallback). The map_attachment
// triggers nv_dma_buf_map() and programs the BAR1 pages dynamically. v1:
// one writer per pool, so this runs at most once per pool.
static bool poolHold(Pool &P, const Bdf &attachTo, char *err, size_t errlen)
{
    char b[512];
    P.holderFd = open(DMABUF_HOLDER_DEVICE_PATH, O_RDWR | O_CLOEXEC);
    if (P.holderFd < 0) {
        int en = errno;
        if (en == ENOENT)
            setErr(err, errlen,
                   "/dev/dmabuf_holder missing -- build and load "
                   "dmabuf_holder.ko (see barlink-torch/BUILD_AND_TEST.md)");
        else if (en == EACCES || en == EPERM)
            setErr(err, errlen,
                   "permission denied on /dev/dmabuf_holder (mode 0600) -- "
                   "run as root / with CAP_SYS_ADMIN");
        else {
            std::snprintf(b, sizeof(b), "open(%s): %s",
                          DMABUF_HOLDER_DEVICE_PATH, std::strerror(en));
            setErr(err, errlen, b);
        }
        return false;
    }

    const uint32_t kMaxEntries = 8192;
    std::vector<dmabuf_holder_sg_entry> sg(kMaxEntries);
    dmabuf_holder_hold harg;
    std::memset(&harg, 0, sizeof(harg));
    harg.dmabuf_fd   = P.dmabufFd;
    harg.flags       = DMABUF_HOLDER_F_BDF_VALID;
    harg.pci_domain  = attachTo.domain;
    harg.pci_bus     = (uint8_t)attachTo.bus;
    harg.pci_slot    = (uint8_t)attachTo.slot;
    harg.pci_func    = (uint8_t)attachTo.func;
    harg.max_entries = kMaxEntries;
    harg.entries     = (uint64_t)(uintptr_t)sg.data();
    if (ioctl(P.holderFd, DMABUF_HOLDER_IOC_HOLD, &harg) != 0) {
        int en = errno;
        std::snprintf(b, sizeof(b),
            "DMABUF_HOLDER_IOC_HOLD (attach as %04x:%02x:%02x.%u): %s. "
            "ENOTSUPP/524 = exporter rejected the attach (patched driver "
            "+ BarlinkPeerBar1=1 both required).",
            attachTo.domain, attachTo.bus, attachTo.slot, attachTo.func,
            std::strerror(en));
        setErr(err, errlen, b);
        return false;
    }
    P.holderHandle = harg.handle;
    if (harg.nents < kMaxEntries) sg.resize(harg.nents);

    // BAR1 offset from the sg table (fallback: marker scan)
    uint64_t barOff = (uint64_t)-1;
    size_t covered = 0;
    if (P.haveBar1Phys && !sg.empty()) {
        uint64_t prevEnd = 0;
        bool contiguous = true;
        for (size_t i = 0; i < sg.size(); ++i) {
            uint64_t a = sg[i].dma_address;
            uint64_t l = sg[i].dma_len;
            if (a < P.bar1Start || a > P.bar1End || a + l - 1 > P.bar1End) {
                contiguous = false; break;
            }
            if (i == 0) { barOff = a - P.bar1Start; prevEnd = a + l; }
            else if (a == prevEnd) { prevEnd = a + l; }
            else { contiguous = false; break; }
            covered += (size_t)l;
        }
        if (!contiguous || covered < P.size)
            barOff = (uint64_t)-1;
    }
    if (barOff == (uint64_t)-1) {
        barOff = barScan(P.barFd, P.barSize, P.magic, sizeof(P.magic));
        if (barOff == (uint64_t)-1) {
            setErr(err, errlen,
                "pool not visible in BAR1 -- dynamic mapping did not "
                "happen (patched driver? BarlinkPeerBar1=1? "
                "dmabuf_holder.ko? CAP_SYS_ADMIN?)");
            return false;
        }
    }
    P.barOff = barOff;
    if (std::getenv("BL_DEBUG_PEER")) {
        std::fprintf(stderr, "[bl] poolHold: barOff=0x%llx nents=%u first=0x%llx "
                    "last=0x%llx bar1=[0x%llx,0x%llx] size=%zu\n",
                    (unsigned long long)P.barOff, (unsigned)sg.size(),
                    (unsigned long long)sg.front().dma_address,
                    (unsigned long long)sg.back().dma_address,
                    (unsigned long long)P.bar1Start,
                    (unsigned long long)P.bar1End, P.size);
        for (size_t i = 0; i < sg.size() && i < 6; ++i)
            std::fprintf(stderr, "[bl]   sg[%zu]=0x%llx len=0x%llx\n", i,
                         (unsigned long long)sg[i].dma_address,
                         (unsigned long long)sg[i].dma_len);
    }
    return true;
}

// mmap the owner pool's BAR1 window for one writer device and register it
// there (cudaHostRegister IoMemory -> the ONE cap-gated call of every
// setup path) -> WritePath appended to the owner pool.
static bool poolWriterPath(blCtx *ctx, int ownerIdx, int writerIdx,
                           char *err, size_t errlen)
{
    char b[512];
    Pool &P = ctx->pools[ownerIdx];
    const int wOrd = ctx->devices[writerIdx];
    WritePath wp;
    wp.writerIdx = writerIdx;

    const size_t pageSize = (size_t)sysconf(_SC_PAGESIZE);
    uint64_t mapOff = P.barOff & ~(uint64_t)(pageSize - 1);
    uint64_t delta  = P.barOff - mapOff;
    size_t mapLen = (size_t)((delta + P.size + pageSize - 1) / pageSize * pageSize);
    if (mapOff + mapLen > P.barSize) mapLen = (size_t)(P.barSize - mapOff);
    wp.map = mmap(nullptr, mapLen, PROT_READ | PROT_WRITE, MAP_SHARED,
                  P.barFd, (off_t)mapOff);
    if (wp.map == MAP_FAILED) {
        wp.map = nullptr;
        std::snprintf(b, sizeof(b), "mmap(BAR1): %s", std::strerror(errno));
        setErr(err, errlen, b);
        return false;
    }
    wp.mapLen = mapLen;

    cudaError_t ce = cudaSetDevice(wOrd);
    if (ce != cudaSuccess) {
        std::snprintf(b, sizeof(b), "cudaSetDevice(writer %d): %s",
                      wOrd, cudaGetErrorString(ce));
        setErr(err, errlen, b);
        return false;
    }
    ce = cudaHostRegister(wp.map, mapLen, cudaHostRegisterIoMemory);
    if (ce != cudaSuccess) {
        std::snprintf(b, sizeof(b),
            "cudaHostRegister(IoMemory) on device %d failed: %s -- the "
            "BAR1 guard is not relaxed (patched driver + "
            "BarlinkPeerBar1=1 required)", wOrd, cudaGetErrorString(ce));
        setErr(err, errlen, b);
        return false;
    }
    wp.registered = true;
    ce = cudaHostGetDevicePointer(&wp.devPtr, wp.map, 0);
    if (ce != cudaSuccess) {
        std::snprintf(b, sizeof(b), "cudaHostGetDevicePointer: %s",
                      cudaGetErrorString(ce));
        setErr(err, errlen, b);
        return false;
    }
    wp.devPtr = (uint8_t *)wp.devPtr + delta;
    P.paths.push_back(wp);
    return true;
}

static bool setupPool(blCtx *ctx, int devIdx, char *err, size_t errlen)
{
    if (!poolLocalSetup(ctx, devIdx, err, errlen)) return false;
    Pool &P = ctx->pools[devIdx];
    const int ord = ctx->devices[devIdx];

    // 3.+4. for every other device: HOLD as that device's PCI BDF (v1: one
    // writer, so the single HOLD derives the BAR1 offset), then mmap +
    // register the window on the writer card.
    for (int w = 0; w < ctx->ndev; ++w) {
        if (w == devIdx) continue;
        char wBus[64] = {0};
        if (cudaDeviceGetPCIBusId(wBus, sizeof(wBus), ctx->devices[w])
                != cudaSuccess) {
            setErr(err, errlen, "setupPool: cudaDeviceGetPCIBusId(writer) failed");
            return false;
        }
        Bdf attachTo = parseBdf(lower(wBus).c_str());
        if (!attachTo.valid) {
            setErr(err, errlen, "setupPool: cannot parse writer BDF");
            return false;
        }
        if (P.holderHandle == 0 && !poolHold(P, attachTo, err, errlen))
            return false;
        if (!poolWriterPath(ctx, devIdx, w, err, errlen))
            return false;
    }

    // zero the flag region on the owner card (local memset; no traffic yet)
    cudaError_t ce = cudaSetDevice(ord);
    if (ce != cudaSuccess ||
        cudaMemset((void *)(P.dptr + P.size - BL_FLAG_REGION), 0,
                   BL_FLAG_REGION) != cudaSuccess) {
        setErr(err, errlen, "setupPool: flag-region memset failed");
        return false;
    }
    return true;
}

static void teardownPool(blCtx *ctx, int devIdx)
{
    Pool &P = ctx->pools[devIdx];
    for (size_t i = 0; i < P.paths.size(); ++i) {
        WritePath &wp = P.paths[i];
        if (wp.registered) {
            cudaSetDevice(ctx->devices[wp.writerIdx]);
            cudaHostUnregister(wp.map);
        }
        if (wp.map) munmap(wp.map, wp.mapLen);
    }
    P.paths.clear();
    if (P.holderFd >= 0) {
        dmabuf_holder_release rel;
        std::memset(&rel, 0, sizeof(rel));
        rel.handle = P.holderHandle;
        if (P.holderHandle)
            ioctl(P.holderFd, DMABUF_HOLDER_IOC_RELEASE, &rel);
        close(P.holderFd);
        P.holderFd = -1;
    }
    if (P.dmabufFd >= 0) { close(P.dmabufFd); P.dmabufFd = -1; }
    nvExportClose(P.nvx);
    if (P.barFd >= 0) { close(P.barFd); P.barFd = -1; }
    // peer-mode pseudo-pools have no allocation of their own (dptr shadows
    // the local pool for offset math -- never unmap it twice)
    if (P.dptr && P.memHandle) {
        cudaSetDevice(ctx->devices[devIdx]);
        cuMemUnmap(P.dptr, P.size);
        cuMemAddressFree(P.dptr, P.size);
        cuMemRelease(P.memHandle);
        P.dptr = 0;
        P.memHandle = 0;
    }
}

// ---------------------------------------------------------------------------
// Public C API (see core.h)
//
// v1 process model: single process, exactly TWO devices. Each pool gets one
// write path (the other card). Supporting >2 cards needs a per-writer
// attachment policy inside dmabuf_holder (one attachment maps the BAR1
// window; attaching additional writers and dropping the duplicate
// attachments risks unmapping the shared window in nv_dma_buf_unmap).
// ---------------------------------------------------------------------------

extern "C" int bl_init(blCtx **out, const int *devices, int ndev,
                       size_t poolBytes, char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!out || !devices || ndev != 2) {
        setErr(err, errlen, "bl_init: v1 supports exactly 2 devices");
        return -1;
    }
    if (poolBytes < (4ull << 20)) {
        setErr(err, errlen, "bl_init: pool must be at least 4 MiB");
        return -1;
    }
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < ndev) {
        setErr(err, errlen, "bl_init: not enough CUDA devices visible");
        return -1;
    }

    blCtx *ctx = new blCtx();
    ctx->ndev = ndev;
    ctx->poolBytes = poolBytes;
    for (int i = 0; i < ndev; ++i) {
        if (devices[i] < 0 || devices[i] >= nd) {
            delete ctx;
            setErr(err, errlen, "bl_init: device ordinal out of range");
            return -1;
        }
        ctx->devices[i] = devices[i];
    }

    for (int i = 0; i < ndev; ++i) {
        if (!setupPool(ctx, i, err, errlen)) {
            for (int k = i - 1; k >= 0; --k) teardownPool(ctx, k);
            delete ctx;
            return -1;
        }
    }
    *out = ctx;
    return 0;
}

extern "C" void bl_shutdown(blCtx *ctx)
{
    if (!ctx) return;
    // p2p side stream / event (user contract: streams are idle at shutdown,
    // same as the pool teardown below)
    if (ctx->evSend) { cudaEventDestroy(ctx->evSend); ctx->evSend = nullptr; }
    if (ctx->sendStream) {
        cudaStreamDestroy(ctx->sendStream);
        ctx->sendStream = nullptr;
    }
    for (int i = ctx->ndev - 1; i >= 0; --i) teardownPool(ctx, i);
    delete ctx;
}

// returns the pool VMM pointer for 'off' -- what from_blob wraps
extern "C" int bl_pool_alloc(blCtx *ctx, int devIdx, size_t bytes,
                             void **ptrOut, char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || devIdx < 0 || devIdx >= ctx->ndev || !ptrOut) {
        setErr(err, errlen, "bl_pool_alloc: bad argument");
        return -1;
    }
    Pool &P = ctx->pools[devIdx];
    std::lock_guard<std::mutex> lk(P.allocMu);
    size_t off = 0;
    std::string e;
    if (!poolAllocLocked(P, bytes, &off, e)) {
        setErr(err, errlen, e);
        return -1;
    }
    *ptrOut = (void *)(uintptr_t)(P.dptr + off);
    return 0;
}

extern "C" int bl_pool_free(blCtx *ctx, int devIdx, void *ptr, size_t bytes)
{
    if (!ctx || devIdx < 0 || devIdx >= ctx->ndev) return -1;
    Pool &P = ctx->pools[devIdx];
    uintptr_t p = (uintptr_t)ptr;
    if (p < (uintptr_t)P.dptr || p >= (uintptr_t)(P.dptr + P.size)) return -1;
    std::lock_guard<std::mutex> lk(P.allocMu);
    poolFreeLocked(P, (size_t)(p - (uintptr_t)P.dptr), bytes);
    return 0;
}

// find a writer path: pool owner's BAR window as seen by writer 'writerIdx'
static void *pathDevPtr(blCtx *ctx, int ownerIdx, int writerIdx)
{
    Pool &P = ctx->pools[ownerIdx];
    for (size_t i = 0; i < P.paths.size(); ++i)
        if (P.paths[i].writerIdx == writerIdx) return P.paths[i].devPtr;
    return nullptr;
}

// enqueue payload + completion marker on the WRITER's stream, WITHOUT the
// reader-side wait. Returns the flag's local-view pointer and seq so the
// caller orders the reader itself: bl_copy_ waits immediately; bl_allreduce_
// enqueues BOTH directions' payloads before either wait, so the two copies
// overlap on the wire instead of serializing behind each other's flag.
static int copyPayload(blCtx *ctx, void *dstPtr, int dstIdx, void *srcPtr,
                       int srcIdx, size_t bytes, void *srcStream,
                       unsigned long long **flagLocalOut, uint64_t *seqOut,
                       char *err, size_t errlen)
{
    void *barDst = pathDevPtr(ctx, dstIdx, srcIdx);
    if (!barDst) {
        setErr(err, errlen, "copyPayload: no BAR1 write path for this pair");
        return -1;
    }
    uintptr_t base = (uintptr_t)ctx->pools[dstIdx].dptr;
    if ((uintptr_t)dstPtr < base ||
        (uintptr_t)dstPtr + bytes > base + ctx->pools[dstIdx].usableSize) {
        setErr(err, errlen, "copyPayload: dst pointer outside the pool");
        return -1;
    }
    uint8_t *dstBar = (uint8_t *)barDst + ((uintptr_t)dstPtr - base);

    // this direction's sequence number (starts at 1; 0 means "never copied")
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lk(ctx->seqMu);
        seq = ++ctx->dirSeq[srcIdx];
        // Copies landing in MY pool are issued by the peer process; under
        // the SPMD discipline its outgoing counter mirrors mine step for
        // step, so my outgoing seq IS my incoming seq.
        ctx->lastIncoming[ctx->peerMode ? ctx->myRank : dstIdx] = seq;
    }

    // flag slot of this direction: BAR view (writer) and local VMM view
    // (reader). One slot per writer, in the owner pool's reserved tail.
    uint8_t *flagBar = (uint8_t *)barDst + flagOff(ctx->pools[dstIdx], srcIdx);
    unsigned long long *flagLocal =
        (unsigned long long *)(uintptr_t)(ctx->pools[dstIdx].dptr +
                                          flagOff(ctx->pools[dstIdx], srcIdx));

    const int srcOrd = ctx->devices[srcIdx];
    if (cudaSetDevice(srcOrd) != cudaSuccess) {
        setErr(err, errlen, "copyPayload: cudaSetDevice failed");
        return -1;
    }
    size_t n4 = bytes / 16;
    k_copy<<<gridBlocks(n4, srcOrd), 256, 0, (cudaStream_t)srcStream>>>(
        (const uint4 *)srcPtr, (uint4 *)dstBar, n4);
    // completion marker on the SAME stream: hard kernel ordering makes the
    // fence cover all of k_copy's posted writes (see k_mark)
    k_mark<<<1, 1, 0, (cudaStream_t)srcStream>>>(
        (unsigned long long *)flagBar, seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "copyPayload: kernel launch failed");
        return -1;
    }
    *flagLocalOut = flagLocal;
    *seqOut = seq;
    return 0;
}

// enqueue the reader-side wait on dstStream: any later work queued there is
// ordered after the payload. Fully async -- no host sync.
static int enqueueFlagWaitOrd(blCtx *ctx, int ord,
                              unsigned long long *flag, uint64_t seq,
                              void *dstStream, char *err, size_t errlen)
{
    if (cudaSetDevice(ord) != cudaSuccess) {
        setErr(err, errlen, "enqueueFlagWait: cudaSetDevice failed");
        return -1;
    }
    k_flag_wait<<<1, 32, 0, (cudaStream_t)dstStream>>>(flag, seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "enqueueFlagWait: flag-wait launch failed");
        return -1;
    }
    return 0;
}

static int enqueueFlagWait(blCtx *ctx, int dstIdx,
                           unsigned long long *flagLocal, uint64_t seq,
                           void *dstStream, char *err, size_t errlen)
{
    return enqueueFlagWaitOrd(ctx, ctx->devices[dstIdx], flagLocal, seq,
                              dstStream, err, errlen);
}

extern "C" int bl_copy_(blCtx *ctx, void *dstPtr, int dstIdx,
                        void *srcPtr, int srcIdx, size_t bytes,
                        void *srcStream, void *dstStream,
                        char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || dstIdx == srcIdx || bytes == 0 || bytes % 16 != 0) {
        setErr(err, errlen, "bl_copy_: bad argument (src and dst must be "
                            "different pool devices, size a multiple of 16)");
        return -1;
    }
    if (ctx->peerMode) {
        // Peer-mode copy, all on MY device and MY stream (dstStream is the
        // same stream in peer mode). See the flag-slot protocol note above
        // for the arm handshake and the ordering argument.
        const int me = ctx->myRank, peer = 1 - me;
        Pool &myP = ctx->pools[me];
        Pool &peerP = ctx->pools[peer];
        void *bar = pathDevPtr(ctx, peer, me);
        if (!bar) {
            setErr(err, errlen, "bl_copy_: no BAR1 write path");
            return -1;
        }
        uintptr_t base = (uintptr_t)myP.dptr;
        if ((uintptr_t)dstPtr < base ||
            (uintptr_t)dstPtr + bytes > base + peerP.usableSize) {
            setErr(err, errlen, "bl_copy_: dst pointer outside the pool");
            return -1;
        }
        uint8_t *dstBar = (uint8_t *)bar + ((uintptr_t)dstPtr - base);
        if (dstPtr == srcPtr) {
            setErr(err, errlen,
                   "bl_copy_: peer mode requires dst and src to be DIFFERENT "
                   "pool buffers (same-offset exchange self-collides: my "
                   "payload would overwrite the peer's source before it is "
                   "read -- the peer would send my own data back)");
            return -1;
        }

        uint64_t seq;
        {
            std::lock_guard<std::mutex> lk(ctx->seqMu);
            seq = ++ctx->dirSeq[me];
            ctx->lastIncoming[me] = seq;
        }
        const int myOrd = ctx->devices[me];
        if (cudaSetDevice(myOrd) != cudaSuccess) {
            setErr(err, errlen, "bl_copy_: cudaSetDevice failed");
            return -1;
        }
        cudaStream_t s = (cudaStream_t)srcStream;
        joinSendStream(ctx, s);   // a pending p2p payload may be READING a
                                  // pool tensor this copy overwrites as dst
        // 1. arm the peer (ordered after the caller's local fills, fence
        //    covers them); 2. wait MY arm, written by the peer's step 1
        k_mark<<<1, 1, 0, s>>>(
            (unsigned long long *)((uint8_t *)bar + flagOff(peerP, me) + 8),
            seq);
        k_flag_wait<<<1, 32, 0, s>>>(
            (unsigned long long *)(uintptr_t)(
                myP.dptr + flagOff(myP, peer) + 8), seq);
        // 3. payload + completion mark into the peer's completion slot
        size_t n4 = bytes / 16;
        k_copy<<<gridBlocks(n4, myOrd), 256, 0, s>>>(
            (const uint4 *)srcPtr, (uint4 *)dstBar, n4);
        k_mark<<<1, 1, 0, s>>>(
            (unsigned long long *)((uint8_t *)bar + flagOff(peerP, me)), seq);
        // 4. my completion wait (local flag, written by the peer)
        k_flag_wait<<<1, 32, 0, s>>>(
            (unsigned long long *)(uintptr_t)(
                myP.dptr + flagOff(myP, peer)), seq);
        if (cudaGetLastError() != cudaSuccess) {
            setErr(err, errlen, "bl_copy_: kernel launch failed");
            return -1;
        }
        if (std::getenv("BL_DEBUG_PEER")) {
            unsigned long long arm = 0, done = 0;
            cudaMemcpy(&arm, (void *)(uintptr_t)(myP.dptr + flagOff(myP, peer) + 8),
                       8, cudaMemcpyDeviceToHost);
            cudaMemcpy(&done, (void *)(uintptr_t)(myP.dptr + flagOff(myP, peer)),
                       8, cudaMemcpyDeviceToHost);
            unsigned long long myDone = 0;
            cudaMemcpy(&myDone,
                       (void *)(uintptr_t)((uint8_t *)bar + flagOff(peerP, me)),
                       8, cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "[bl] rank %d copy seq=%llu: local arm=%llu "
                        "done=%llu | my mark in peer=%llu\n",
                        me, (unsigned long long)seq, arm, done, myDone);
        }
        return 0;
    }
    unsigned long long *flagLocal = nullptr;
    uint64_t seq = 0;
    if (copyPayload(ctx, dstPtr, dstIdx, srcPtr, srcIdx, bytes, srcStream,
                    &flagLocal, &seq, err, errlen) != 0)
        return -1;
    return enqueueFlagWait(ctx, dstIdx, flagLocal, seq, dstStream, err, errlen);
}

extern "C" int bl_allreduce_(blCtx *ctx, void *aPtr, int aIdx,
                             void *bPtr, int bIdx, size_t bytes, int dtype,
                             void *streamA, void *streamB,
                             char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || aIdx == bIdx || bytes == 0 || bytes % 16 != 0) {
        setErr(err, errlen,
               "bl_allreduce_: bad argument (sizes must be multiples of 16)");
        return -1;
    }
    if (dtype < BL_DTYPE_U8 || dtype > BL_DTYPE_FP8E5M2) {
        setErr(err, errlen, "bl_allreduce_: unsupported dtype enum");
        return -1;
    }
    // scratch in each pool, exchanged in BOTH directions first; only then
    // does each side add locally. (Using the already-updated peer value
    // would double-count.)
    void *scratchA = nullptr, *scratchB = nullptr;
    if (bl_pool_alloc(ctx, aIdx, bytes, &scratchA, err, errlen) != 0)
        return -1;
    if (bl_pool_alloc(ctx, bIdx, bytes, &scratchB, err, errlen) != 0) {
        bl_pool_free(ctx, aIdx, scratchA, bytes);
        return -1;
    }

    // scratchB (in b's pool) receives a; scratchA (in a's pool) receives b.
    // Both payloads are enqueued BEFORE either reader-side wait, so the two
    // directions overlap on the wire (serializing them behind each other's
    // flag costs ~30% aggregate bandwidth on this platform). The waits are
    // queued before the adds on the same streams, so the adds -- queued
    // after -- stay ordered after the incoming payload.
    int rc = 0;
    unsigned long long *flagInB = nullptr, *flagInA = nullptr;
    uint64_t seq1 = 0, seq2 = 0;
    if ((rc = copyPayload(ctx, scratchB, bIdx, aPtr, aIdx, bytes,
                          streamA, &flagInB, &seq1, err, errlen)) != 0)
        goto out;
    if ((rc = copyPayload(ctx, scratchA, aIdx, bPtr, bIdx, bytes,
                          streamB, &flagInA, &seq2, err, errlen)) != 0)
        goto out;
    if ((rc = enqueueFlagWait(ctx, bIdx, flagInB, seq1, streamB,
                              err, errlen)) != 0)
        goto out;
    if ((rc = enqueueFlagWait(ctx, aIdx, flagInA, seq2, streamA,
                              err, errlen)) != 0)
        goto out;

    // local adds; each side consumes its scratch (written by the peer)
    {
        const int ordA = ctx->devices[aIdx];
        const int ordB = ctx->devices[bIdx];
        size_t n4 = bytes / 16;
        if (cudaSetDevice(ordA) != cudaSuccess) {
            setErr(err, errlen, "bl_allreduce_: cudaSetDevice(A) failed");
            rc = -1; goto out;
        }
        if (launchAdd(dtype, (uint4 *)aPtr, (const uint4 *)scratchA, n4,
                      gridBlocks(n4, ordA), (cudaStream_t)streamA) != 0) {
            setErr(err, errlen, "bl_allreduce_: bad dtype (internal)");
            rc = -1; goto out;
        }
        if (cudaSetDevice(ordB) != cudaSuccess) {
            setErr(err, errlen, "bl_allreduce_: cudaSetDevice(B) failed");
            rc = -1; goto out;
        }
        if (launchAdd(dtype, (uint4 *)bPtr, (const uint4 *)scratchB, n4,
                      gridBlocks(n4, ordB), (cudaStream_t)streamB) != 0) {
            setErr(err, errlen, "bl_allreduce_: bad dtype (internal)");
            rc = -1; goto out;
        }
        if (cudaGetLastError() != cudaSuccess) {
            setErr(err, errlen, "bl_allreduce_: add kernel launch failed");
            rc = -1; goto out;
        }
    }

out:
    // TODO: the scratch could be freed stream-aware (event instead of host
    // sync); v1 keeps the conservative drain before returning it to the
    // allocator.
    cudaSetDevice(ctx->devices[aIdx]);
    cudaDeviceSynchronize();
    cudaSetDevice(ctx->devices[bIdx]);
    cudaDeviceSynchronize();
    bl_pool_free(ctx, aIdx, scratchA, bytes);
    bl_pool_free(ctx, bIdx, scratchB, bytes);
    return rc;
}

// ---------------------------------------------------------------------------
// Cross-process (SPMD peer) mode: one process per GPU, rank = device index.
//
// Discipline (MPI-symmetric-heap style): both ranks call IDENTICAL
// sequences of bl_* functions with identical sizes. There is NO runtime
// control channel -- bl_init_peer performs a one-time rendezvous over a
// unix socket (phase 1: {BDF, poolBytes}; phase 2: {BAR1 offset}), after
// which the processes never talk again. Every synchronization primitive is
// the marker flag: flags live in each pool's 4 KiB tail, one 256-byte slot
// per WRITER (slot index = writer rank), so the two ranks agree on slots
// by construction.
//
// Memory layout (poolBytes equal on both ranks): user tensors in
// [0, scratchBase); scratchBase = size/2 rounded down to the 2 MiB alloc
// alignment. The scratch zone [scratchBase, size - flagTail) is the
// cross-process exchange area (allreduce scratch, verify zone); matching
// offsets land there on both sides by symmetric allocation.
// ---------------------------------------------------------------------------

struct BlPeerHello { char bdf[20]; uint64_t poolBytes; };
struct BlPeerBar  { uint64_t barOff; };

static bool peerWriteAll(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    while (len) {
        ssize_t w = write(fd, p, len);
        if (w <= 0) return false;
        p += w; len -= (size_t)w;
    }
    return true;
}

static bool peerReadAll(int fd, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    while (len) {
        ssize_t r = read(fd, p, len);
        if (r <= 0) return false;
        p += r; len -= (size_t)r;
    }
    return true;
}

// write-then-read; the socket buffer absorbs both directions, no deadlock
static bool peerXchg(int fd, const void *out, size_t outLen,
                     void *in, size_t inLen, char *err, size_t errlen)
{
    if (!peerWriteAll(fd, out, outLen) || !peerReadAll(fd, in, inLen)) {
        setErr(err, errlen,
               std::string("peer socket exchange failed: ") +
               std::strerror(errno));
        return false;
    }
    return true;
}

// '@name' selects the abstract namespace; anything else is a filesystem
// path (rank 0 unlinks it before bind and after accept)
static int peerSockPath(struct sockaddr_un *sa, const char *sockPath)
{
    std::memset(sa, 0, sizeof(*sa));
    sa->sun_family = AF_UNIX;
    if (sockPath[0] == '@') {
        size_t n = std::strlen(sockPath + 1);
        if (n == 0 || n >= sizeof(sa->sun_path) - 1) return -1;
        sa->sun_path[0] = 0;
        std::memcpy(sa->sun_path + 1, sockPath + 1, n);
        return (int)(offsetof(struct sockaddr_un, sun_path) + 1 + n);
    }
    if (std::strlen(sockPath) >= sizeof(sa->sun_path)) return -1;
    std::strncpy(sa->sun_path, sockPath, sizeof(sa->sun_path) - 1);
    return (int)(offsetof(struct sockaddr_un, sun_path) +
                 1 + std::strlen(sockPath));
}

static bool peerConnect(const char *sockPath, int rank, int *fdOut,
                        char *err, size_t errlen)
{
    char b[256];
    struct sockaddr_un sa;
    int slen = peerSockPath(&sa, sockPath);
    if (slen < 0) {
        setErr(err, errlen, "peerConnect: socket path too long");
        return false;
    }
    bool fsPath = sockPath[0] != '@';
    if (rank == 0) {
        if (fsPath) unlink(sockPath);        // drop a stale bind
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0 || bind(fd, (struct sockaddr *)&sa, (socklen_t)slen) != 0 ||
            listen(fd, 1) != 0) {
            std::snprintf(b, sizeof(b), "peer rank 0 bind/listen(%s): %s",
                          sockPath, std::strerror(errno));
            setErr(err, errlen, b);
            if (fd >= 0) close(fd);
            return false;
        }
        int c = accept(fd, nullptr, nullptr);
        if (c < 0) {
            std::snprintf(b, sizeof(b), "peer rank 0 accept: %s",
                          std::strerror(errno));
            setErr(err, errlen, b);
            close(fd);
            return false;
        }
        close(fd);                           // rendezvous done
        if (fsPath) unlink(sockPath);
        *fdOut = c;
    } else {
        int c = -1;
        for (int i = 0; i < 200 && c < 0; ++i) {   // rank 0 may be slow
            c = socket(AF_UNIX, SOCK_STREAM, 0);
            if (c < 0) break;
            if (connect(c, (struct sockaddr *)&sa, (socklen_t)slen) != 0) {
                close(c);
                c = -1;
                usleep(50 * 1000);
            }
        }
        if (c < 0) {
            std::snprintf(b, sizeof(b),
                          "peer rank 1 connect(%s) failed: %s -- is rank 0 "
                          "running with the same socket path?", sockPath,
                          std::strerror(errno));
            setErr(err, errlen, b);
            return false;
        }
        *fdOut = c;
    }
    return true;
}

extern "C" int bl_init_peer(blCtx **out, int device, size_t poolBytes,
                            const char *sockPath, int rank,
                            char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!out || !sockPath || (rank != 0 && rank != 1)) {
        setErr(err, errlen, "bl_init_peer: rank must be 0 or 1");
        return -1;
    }
    if (poolBytes < (8ull << 20)) {
        setErr(err, errlen,
               "bl_init_peer: pool must be >= 8 MiB (half of it is the "
               "cross-process scratch zone)");
        return -1;
    }
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || device < 0 || device >= nd) {
        setErr(err, errlen, "bl_init_peer: device ordinal out of range");
        return -1;
    }

    blCtx *ctx = new blCtx();
    ctx->ndev = 2;
    ctx->peerMode = true;
    ctx->myRank = rank;
    ctx->poolBytes = poolBytes;
    ctx->devices[rank] = device;
    ctx->devices[1 - rank] = -1;   // peer ordinal: unknown, never used locally
    const int me = rank, peer = 1 - rank;

    // Phase 0+1: rendezvous, exchange {my GPU BDF, poolBytes}
    int fd = -1;
    if (!peerConnect(sockPath, rank, &fd, err, errlen)) { delete ctx; return -1; }
    BlPeerHello mineHello, peerHello;
    std::memset(&mineHello, 0, sizeof(mineHello));
    {
        char busId[64] = {0};
        if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device)
                != cudaSuccess) {
            setErr(err, errlen, "bl_init_peer: cudaDeviceGetPCIBusId failed");
            close(fd); delete ctx; return -1;
        }
        std::string s = lower(busId);
        std::strncpy(mineHello.bdf, s.c_str(), sizeof(mineHello.bdf) - 1);
    }
    mineHello.poolBytes = poolBytes;
    if (!peerXchg(fd, &mineHello, sizeof(mineHello),
                  &peerHello, sizeof(peerHello), err, errlen)) {
        close(fd); delete ctx; return -1;
    }
    if (peerHello.poolBytes != poolBytes) {
        setErr(err, errlen,
               "bl_init_peer: poolBytes mismatch between ranks (symmetric "
               "SPMD requires equal pools)");
        close(fd); delete ctx; return -1;
    }
    Bdf peerBdf = parseBdf(peerHello.bdf);
    if (!peerBdf.valid) {
        setErr(err, errlen, "bl_init_peer: peer sent an unparseable BDF");
        close(fd); delete ctx; return -1;
    }

    // Local setup: my VMM pool + dma-buf export + my BAR1 aperture, then
    // HOLD my fd as the PEER's PCI device (the peer GPU is the device that
    // will actually write my BAR1) and derive my BAR1 offset.
    if (!poolLocalSetup(ctx, me, err, errlen)) {
        close(fd); delete ctx; return -1;
    }
    Pool &myP = ctx->pools[me];
    if (!poolHold(myP, peerBdf, err, errlen)) {
        close(fd); delete ctx; return -1;
    }

    // Peer-mode memory layout: user tensors below scratchBase, exchange
    // scratch in [scratchBase, size - flagTail). Both ranks compute the
    // same value because poolBytes (hence size) is equal. The scratch zone
    // is split: allreduce keeps the bottom half (its fixed offset is
    // scratchBase), p2p takes the top half -- disjoint, so the two
    // protocols can be interleaved freely.
    ctx->scratchBase = (myP.size / 2) & ~(kAllocAlign - 1);
    myP.usableSize = ctx->scratchBase;
    {
        const size_t scratchAvail =
            myP.size - BL_FLAG_REGION - ctx->scratchBase;
        ctx->p2pScratchBase =
            ctx->scratchBase + ((scratchAvail / 2) & ~(size_t)15);
        // fused small-message allreduce gets a dedicated slice at the TOP
        // of the allreduce half (default 1 MiB, quarter of the ar zone at
        // most); the arm-path bound becomes fusedZoneBase.
        size_t arZone = ctx->p2pScratchBase - ctx->scratchBase;
        size_t fz = arZone / 4 < (1u << 20) ? arZone / 4 : (1u << 20);
        fz &= ~(size_t)15;
        ctx->fusedZoneBase = ctx->p2pScratchBase - fz;
    }

    // zero my flag tail BEFORE the peer can reach it (phase 2 below hands
    // out my barOff, after which the peer may write into my pool)
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMemset((void *)(myP.dptr + myP.size - BL_FLAG_REGION), 0,
                   BL_FLAG_REGION) != cudaSuccess) {
        setErr(err, errlen, "bl_init_peer: flag-region memset failed");
        close(fd); delete ctx; return -1;
    }

    // Phase 2: exchange my BAR1 offset
    BlPeerBar mineBar, peerBar;
    mineBar.barOff = myP.barOff;
    if (!peerXchg(fd, &mineBar, sizeof(mineBar),
                  &peerBar, sizeof(peerBar), err, errlen)) {
        close(fd); delete ctx; return -1;
    }
    close(fd);

    // Remote setup: mmap + register the PEER's BAR1 window on MY device
    // (the one cap-gated call of this path). pools[peer] is a pseudo-entry:
    // no local VA exists for the peer pool; dptr/size/usableSize shadow MY
    // pool so offset(ptr) = ptr - dptr keeps working in the copy path.
    Pool &peerP = ctx->pools[peer];
    peerP.devIdx = peer;
    peerP.size = myP.size;
    peerP.usableSize = myP.usableSize;
    peerP.dptr = myP.dptr;                 // offset math only, never dereferenced
    peerP.barOff = peerBar.barOff;
    {
        std::string peerBarPath = std::string("/sys/bus/pci/devices/") +
                                  peerHello.bdf + "/resource1_wc";
        peerP.barFd = open(peerBarPath.c_str(), O_RDWR | O_SYNC);
        if (peerP.barFd < 0) {
            char b[256];
            std::snprintf(b, sizeof(b), "bl_init_peer: open(%s): %s",
                          peerBarPath.c_str(), std::strerror(errno));
            setErr(err, errlen, b);
            delete ctx; return -1;
        }
        struct stat st;
        if (fstat(peerP.barFd, &st) != 0) {
            setErr(err, errlen, "bl_init_peer: fstat(peer BAR1) failed");
            delete ctx; return -1;
        }
        peerP.barSize = (uint64_t)st.st_size;
    }
    if (!poolWriterPath(ctx, peer, me, err, errlen)) {   // writer = me
        delete ctx; return -1;
    }

    // caps were only needed for the cudaHostRegister above
    bl_drop_caps();
    *out = ctx;
    return 0;
}

// ---------------------------------------------------------------------------
// BAR atomic feasibility probe (see core.h). Slot 6 of each pool's flag
// region is dedicated to it.
// ---------------------------------------------------------------------------

__global__ void k_bar_atomic(unsigned long long *barSlot,
                             unsigned long long *localOut)
{
    if (threadIdx.x == 0) {
        unsigned long long old;
        asm volatile("atom.global.add.u64 %0, [%1], %2;"
                     : "=l"(old) : "l"(barSlot), "l"(1ULL) : "memory");
        localOut[0] = old;
    }
}

__global__ void k_empty() {}

// forward decls (defined with the zero-copy add helpers below)
__device__ __forceinline__ static float  blElemAdd(float a, float b);
__device__ __forceinline__ static double blElemAdd(double a, double b);
__device__ __forceinline__ static __nv_bfloat16 blElemAdd(__nv_bfloat16 a,
                                                          __nv_bfloat16 b);
__device__ __forceinline__ static __half blElemAdd(__half a, __half b);

// ---------------------------------------------------------------------------
// Fused small-message allreduce (2 stream-ordered kernels per op, replacing
// the 6-kernel arm protocol for payloads <= BL_FUSED_MAX bytes). Design
// notes (the send/recv deadlock lessons all apply):
//   - consumed-receipt gating instead of the arm handshake: the RECEIVER's
//     fused kernel posts the receipt to the sender's receipt slot (BAR wt,
//     last block of a local atomic counter) AFTER its add consumed the
//     scratch; the sender's fused kernel polls its LOCAL receipt slot
//     (ld.relaxed.sys, proven to see inbound writes) before writing the
//     peer's scratch. No circular arm: wait chains terminate at op 1.
//   - dedicated scratch zone (top slice of the allreduce half) so fused
//     traffic NEVER shares memory with the old arm-path payloads -- mixed
//     fused/old sequences need no cross-protocol gating.
//   - payload->flag ordering inside one kernel: every block fences
//     (__threadfence_system) then bumps a LOCAL atomic counter; the last
//     block stores the completion flag into the peer's flag slot. Same
//     in-order-single-source-PCIe argument as the kernel-boundary case in
//     the 6-kernel protocol (posted writes from one GPU port deliver in
//     order); soak-tested by the PG regression loops.
//   - every block polls the flag/receipt itself before touching payload --
//     no grid-wide sync needed.
// ---------------------------------------------------------------------------
#define BL_FUSED_MAX   (64u << 10)     // payloads above this keep the 6-kernel path
#define BL_AR_RECV_SLOT   4            // my pool: receipt from the peer (+0)
#define BL_AR_FLAG_SLOT   5            // my pool: payload-complete flag from the peer (+0)
#define BL_AR_CNT_SLOT    7            // local last-block election counters (+0 send, +8 recv)

// fused producer: wait receipt >= seq-1, payload -> peer fused zone, then
// (last block) raise the peer's flag.
__global__ void k_fused_send(const uint4 *__restrict__ in,
                             uint4 *__restrict__ peerScr,
                             const unsigned long long *__restrict__ receipt,
                             unsigned long long *__restrict__ flagBar,
                             unsigned long long *__restrict__ counter,
                             unsigned long long seq, size_t n4)
{
    // receipt gate: peer must have consumed the previous fused payload.
    // seq == 1 (first fused op ever) finds the zero-initialized slot >= 0.
    if (seq > 1) {
        const long long t0 = clock64();
        unsigned ns = 32;
        for (;;) {
            unsigned long long v;
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                         : "=l"(v) : "l"(receipt) : "memory");
            if (v >= seq - 1) break;
            if (clock64() - t0 > 4LL * 1000 * 1000 * 1000) __trap();
            __nanosleep(ns);
            if (ns < (1u << 20)) ns <<= 1;
        }
    }
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride)
        stwt128(&peerScr[i], in[i]);
    __threadfence_system();
    __shared__ int isLast;
    if (threadIdx.x == 0) {
        unsigned long long old;
        asm volatile("atom.global.add.u64 %0, [%1], %2;"
                     : "=l"(old) : "l"(counter), "l"(1ULL) : "memory");
        isLast = (old == (unsigned long long)gridDim.x - 1);
        if (isLast) {
            stwt64(flagBar, seq);
            *counter = 0;   // next op's kernel starts after this one completes
        }
    }
    __syncthreads();
}

// fused consumer: wait flag >= seq, out = in + scratch (ld.relaxed.sys),
// then (last block) post the receipt to the peer.
template <typename T>
__global__ void k_fused_recv_t(uint4 *__restrict__ out,
                               const uint4 *__restrict__ in,
                               const uint4 *__restrict__ scr,
                               const unsigned long long *__restrict__ flag,
                               unsigned long long *__restrict__ receiptBar,
                               unsigned long long *__restrict__ counter,
                               unsigned long long seq, size_t n4)
{
    {
        const long long t0 = clock64();
        unsigned ns = 32;
        for (;;) {
            unsigned long long v;
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                         : "=l"(v) : "l"(flag) : "memory");
            if (v >= seq) break;
            if (clock64() - t0 > 4LL * 1000 * 1000 * 1000) __trap();
            __nanosleep(ns);
            if (ns < (1u << 20)) ns <<= 1;
        }
    }
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = in[i];
        uint4 y = ldcs128(&scr[i]);
        const T *xp = reinterpret_cast<const T *>(&x);
        const T *yp = reinterpret_cast<const T *>(&y);
        uint4 s;
        T *sp = reinterpret_cast<T *>(&s);
#pragma unroll
        for (int k = 0; k < (int)(16 / sizeof(T)); ++k)
            sp[k] = blElemAdd(xp[k], yp[k]);
        out[i] = s;
    }
    __threadfence_system();
    __shared__ int isLast;
    if (threadIdx.x == 0) {
        unsigned long long old;
        asm volatile("atom.global.add.u64 %0, [%1], %2;"
                     : "=l"(old) : "l"(counter), "l"(1ULL) : "memory");
        isLast = (old == (unsigned long long)gridDim.x - 1);
        if (isLast) {
            stwt64(receiptBar, seq);
            *counter = 0;
        }
    }
    __syncthreads();
}

// fp8 variant on raw storage (same by-value miscompile workaround as the
// other fp8 kernels).
template <__nv_fp8_interpretation_t INTERP>
__global__ void k_fused_recv_fp8(uint4 *__restrict__ out,
                                 const uint4 *__restrict__ in,
                                 const uint4 *__restrict__ scr,
                                 const unsigned long long *__restrict__ flag,
                                 unsigned long long *__restrict__ receiptBar,
                                 unsigned long long *__restrict__ counter,
                                 unsigned long long seq, size_t n4)
{
    {
        const long long t0 = clock64();
        unsigned ns = 32;
        for (;;) {
            unsigned long long v;
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                         : "=l"(v) : "l"(flag) : "memory");
            if (v >= seq) break;
            if (clock64() - t0 > 4LL * 1000 * 1000 * 1000) __trap();
            __nanosleep(ns);
            if (ns < (1u << 20)) ns <<= 1;
        }
    }
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        uint4 x = in[i];
        uint4 y = ldcs128(&scr[i]);
        const __nv_fp8_storage_t *xp =
            reinterpret_cast<const __nv_fp8_storage_t *>(&x);
        const __nv_fp8_storage_t *yp =
            reinterpret_cast<const __nv_fp8_storage_t *>(&y);
        uint4 s;
        __nv_fp8_storage_t *sp =
            reinterpret_cast<__nv_fp8_storage_t *>(&s);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float fa = __half2float(
                __half(__nv_cvt_fp8_to_halfraw(xp[k], INTERP)));
            float fb = __half2float(
                __half(__nv_cvt_fp8_to_halfraw(yp[k], INTERP)));
            sp[k] = __nv_cvt_float_to_fp8(fa + fb, __NV_SATFINITE, INTERP);
        }
        out[i] = s;
    }
    __threadfence_system();
    __shared__ int isLast;
    if (threadIdx.x == 0) {
        unsigned long long old;
        asm volatile("atom.global.add.u64 %0, [%1], %2;"
                     : "=l"(old) : "l"(counter), "l"(1ULL) : "memory");
        isLast = (old == (unsigned long long)gridDim.x - 1);
        if (isLast) {
            stwt64(receiptBar, seq);
            *counter = 0;
        }
    }
    __syncthreads();
}

// local-atomic counters for the last-block election (own pool, flag tail
// scratch area inside the fused zone metadata -- kept in unused flag slots)
static int launchFusedRecv(int dtype, uint4 *out, const uint4 *in,
                           const uint4 *scr,
                           const unsigned long long *flag,
                           unsigned long long *receiptBar,
                           unsigned long long *counter,
                           unsigned long long seq, size_t n4,
                           size_t blocks, cudaStream_t stream)
{
    switch (dtype) {
    case BL_DTYPE_FP16:    k_fused_recv_t<__half><<<blocks, 256, 0, stream>>>(
        out, in, scr, flag, receiptBar, counter, seq, n4); break;
    case BL_DTYPE_FP32:    k_fused_recv_t<float><<<blocks, 256, 0, stream>>>(
        out, in, scr, flag, receiptBar, counter, seq, n4); break;
    case BL_DTYPE_FP64:    k_fused_recv_t<double><<<blocks, 256, 0, stream>>>(
        out, in, scr, flag, receiptBar, counter, seq, n4); break;
    case BL_DTYPE_BF16:    k_fused_recv_t<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
        out, in, scr, flag, receiptBar, counter, seq, n4); break;
    case BL_DTYPE_FP8E4M3: k_fused_recv_fp8<__NV_E4M3><<<blocks, 256, 0, stream>>>(
        out, in, scr, flag, receiptBar, counter, seq, n4); break;
    case BL_DTYPE_FP8E5M2: k_fused_recv_fp8<__NV_E5M2><<<blocks, 256, 0, stream>>>(
        out, in, scr, flag, receiptBar, counter, seq, n4); break;
    default: return -1;
    }
    return 0;
}

extern "C" int bl_probe_bar_atomic(blCtx *ctx, unsigned long long *res,
                                   int iters, void *stream, char *err,
                                   size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || !ctx->peerMode || iters <= 0) {
        setErr(err, errlen, "bl_probe_bar_atomic: peer ctx and iters > 0");
        return -1;
    }
    const int me = ctx->myRank, peer = 1 - me;
    Pool &myP = ctx->pools[me];
    const int myOrd = ctx->devices[me];
    void *bar = pathDevPtr(ctx, peer, me);
    if (!bar) {
        setErr(err, errlen, "bl_probe_bar_atomic: no BAR1 write path");
        return -1;
    }
    if (cudaSetDevice(myOrd) != cudaSuccess) {
        setErr(err, errlen, "bl_probe_bar_atomic: cudaSetDevice failed");
        return -1;
    }
    // slot 6: probe counter at +0, probe flag at +8. each rank adds 1 to
    // the PEER counter per iteration, so each rank's counter ends at iters
    // iff the atomic TLPs terminate correctly on the peer BAR.
    unsigned long long *mySlot = (unsigned long long *)(uintptr_t)(
        myP.dptr + flagOff(myP, 6));
    unsigned long long *myFlag = (unsigned long long *)(uintptr_t)(
        myP.dptr + flagOff(myP, 6) + 8);
    unsigned long long *peerSlotBar = (unsigned long long *)((uint8_t *)bar +
        flagOff(ctx->pools[peer], 6));
    unsigned long long *peerFlagBar = (unsigned long long *)((uint8_t *)bar +
        flagOff(ctx->pools[peer], 6) + 8);
    unsigned long long *dout = nullptr;
    if (cudaMalloc(&dout, 4 * sizeof(unsigned long long)) != cudaSuccess) {
        setErr(err, errlen, "bl_probe_bar_atomic: cudaMalloc failed");
        return -1;
    }
    cudaMemset(dout, 0, 4 * sizeof(unsigned long long));
    cudaStream_t s = (cudaStream_t)stream;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // phase 1: atomic exchange. per iter: atom.add on the peer counter,
    // then wait MY counter reaching i+1 (i.e. the peer's atom landed).
    // If the peer GPU's BAR does not terminate AtomicOp TLPs, the wait
    // times out -> __trap -> CUDA error visible to the caller.
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        k_bar_atomic<<<1, 32, 0, s>>>(peerSlotBar, dout + 1);
        k_flag_wait<<<1, 32, 0, s>>>(mySlot, (unsigned long long)(i + 1));
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: ATOMIC PHASE FAILED "
               "(peer BAR probably does not terminate AtomicOp TLPs)");
        return -1;
    }
    float ms1 = 0.f, ms2 = 0.f;
    cudaEventElapsedTime(&ms1, e0, e1);

    // phase 2: the protocols' own marker-flag exchange as the baseline.
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        unsigned long long v = (1ULL << 40) | (unsigned)i;
        k_mark<<<1, 1, 0, s>>>((unsigned long long *)peerFlagBar, v);
        k_flag_wait<<<1, 32, 0, s>>>(myFlag, v);
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: flag baseline failed");
        return -1;
    }
    cudaEventElapsedTime(&ms2, e0, e1);

    // phase 3: launch-overhead baseline -- pre-arm MY flag with the final
    // value so every wait returns on the first poll (no peer dependency).
    k_mark<<<1, 1, 0, s>>>(myFlag, (1ULL << 40) | 199ULL);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        unsigned long long v = (1ULL << 40) | (unsigned)i;
        k_mark<<<1, 1, 0, s>>>((unsigned long long *)peerFlagBar, v);
        k_flag_wait<<<1, 32, 0, s>>>(myFlag, v);
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: pre-armed baseline failed");
        return -1;
    }
    float ms3 = 0.f;
    cudaEventElapsedTime(&ms3, e0, e1);

    // phase 4: empty kernels, same stream/context (launch cost control)
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        k_empty<<<1, 32, 0, s>>>();
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: empty phase failed");
        return -1;
    }
    float ms4 = 0.f;
    cudaEventElapsedTime(&ms4, e0, e1);

    // phase 5: k_mark to a LOCAL slot (no BAR traffic at all)
    unsigned long long *localFlag = (unsigned long long *)(uintptr_t)(
        myP.dptr + flagOff(myP, 6) + 16);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        unsigned long long v = (1ULL << 40) | (unsigned)i;
        k_mark<<<1, 1, 0, s>>>(localFlag, v);
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: local-mark phase failed");
        return -1;
    }
    float ms5 = 0.f;
    cudaEventElapsedTime(&ms5, e0, e1);

    // phase 6: k_mark to the PEER BAR slot only (no wait) -- isolates the
    // BAR-store retire cost
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        unsigned long long v = (1ULL << 40) | (unsigned)i;
        k_mark<<<1, 1, 0, s>>>((unsigned long long *)peerFlagBar, v);
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: bar-mark phase failed");
        return -1;
    }
    float ms6 = 0.f;
    cudaEventElapsedTime(&ms6, e0, e1);

    // phase 7: pre-armed waits only (no mark in the loop)
    k_mark<<<1, 1, 0, s>>>(myFlag, (1ULL << 40) | 199ULL);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        unsigned long long v = (1ULL << 40) | (unsigned)i;
        k_flag_wait<<<1, 32, 0, s>>>(myFlag, v);
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: wait-only phase failed");
        return -1;
    }
    float ms7 = 0.f;
    cudaEventElapsedTime(&ms7, e0, e1);

    // phase 8: k_mark to PEER SLOT 2 (protocol-class slot, unused here)
    unsigned long long *peerFlag2Bar = (unsigned long long *)((uint8_t *)bar +
        flagOff(ctx->pools[peer], 2) + 8);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
        unsigned long long v = (1ULL << 40) | (unsigned)i;
        k_mark<<<1, 1, 0, s>>>((unsigned long long *)peerFlag2Bar, v);
    }
    cudaEventRecord(e1, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: slot2-mark phase failed");
        return -1;
    }
    float ms8 = 0.f;
    cudaEventElapsedTime(&ms8, e0, e1);

    // phase 9: the exact protocol pattern -- 16KB payload (BAR stores) then
    // k_mark, back to back, both directions running SPMD on the peer
    {
        uint8_t *peerScrBar = (uint8_t *)bar + ctx->scratchBase;
        const uint4 *src = (const uint4 *)(uintptr_t)(myP.dptr + 512);
        uint4 *dst = (uint4 *)peerScrBar;
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) {
            k_copy<<<gridBlocks(1024, myOrd), 256, 0, s>>>(src, dst, 1024);
            unsigned long long v = (1ULL << 40) | (unsigned)i;
            k_mark<<<1, 1, 0, s>>>((unsigned long long *)peerFlag2Bar, v);
        }
        cudaEventRecord(e1, s);
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        cudaGetLastError();
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: protocol-pattern phase failed");
        return -1;
    }
    float ms9 = 0.f;
    cudaEventElapsedTime(&ms9, e0, e1);

    res[0] = (unsigned long long)(ms1 * 1000.0 * 1000.0 / iters);
    res[1] = (unsigned long long)(ms2 * 1000.0 * 1000.0 / iters);
    res[3] = (unsigned long long)(ms3 * 1000.0 * 1000.0 / iters);
    res[4] = (unsigned long long)(ms4 * 1000.0 * 1000.0 / iters);
    res[5] = (unsigned long long)(ms5 * 1000.0 * 1000.0 / iters);
    res[6] = (unsigned long long)(ms6 * 1000.0 * 1000.0 / iters);
    res[7] = (unsigned long long)(ms7 * 1000.0 * 1000.0 / iters);
    res[8] = (unsigned long long)(ms8 * 1000.0 * 1000.0 / iters);
    res[9] = (unsigned long long)(ms9 * 1000.0 * 1000.0 / iters);
    if (cudaMemcpy(&res[2], mySlot, 8, cudaMemcpyDeviceToHost)
            != cudaSuccess) {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(dout);
        setErr(err, errlen, "bl_probe_bar_atomic: slot readback failed");
        return -1;
    }
    cudaFree(dout);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}

// Drop all capabilities (ambient clear + capset). Exported so the binding
// can share it between the single-process and peer init paths.
extern "C" void bl_drop_caps(void)
{
    prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0);
    struct __user_cap_header_struct hdr = { _LINUX_CAPABILITY_VERSION_3, 0 };
    struct __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3] = {};
    syscall(SYS_capset, &hdr, data);
}

extern "C" int bl_is_peer(const blCtx *ctx)
{
    return ctx && ctx->peerMode;
}

extern "C" int bl_allreduce_peer(blCtx *ctx, void *aPtr, void *bPtr,
                                 size_t bytes, int dtype, void *stream,
                                 char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || !ctx->peerMode || bytes == 0 || bytes % 16 != 0) {
        setErr(err, errlen,
               "bl_allreduce_peer: bad argument (size a multiple of 16)");
        return -1;
    }
    if (dtype < BL_DTYPE_U8 || dtype > BL_DTYPE_FP8E5M2) {
        setErr(err, errlen, "bl_allreduce_peer: unsupported dtype enum");
        return -1;
    }
    const int me = ctx->myRank, peer = 1 - me;
    Pool &myP = ctx->pools[me];
    const int myOrd = ctx->devices[me];
    uintptr_t base = (uintptr_t)myP.dptr;
    uintptr_t oa = (uintptr_t)aPtr - base;
    uintptr_t ob = (uintptr_t)bPtr - base;
    if (oa % 16 || ob % 16 ||
        oa + bytes > myP.usableSize || ob + bytes > myP.usableSize) {
        setErr(err, errlen,
               "bl_allreduce_peer: tensors must live in the user half of my "
               "pool, [0, scratchBase)");
        return -1;
    }
    uintptr_t omax = oa > ob ? oa : ob;
    if (ctx->scratchBase + omax + bytes > ctx->fusedZoneBase) {
        setErr(err, errlen,
               "bl_allreduce_peer: tensor does not fit in the scratch zone");
        return -1;
    }
    void *bar = pathDevPtr(ctx, peer, me);
    if (!bar) {
        setErr(err, errlen, "bl_allreduce_peer: no BAR1 write path");
        return -1;
    }

    // ONE seq per allreduce per rank. Both payloads go out BEFORE the
    // single mark: stream order lets one fence + one flag cover both
    // writes. The peer runs the identical sequence concurrently, so the
    // two directions overlap on the wire (the same overlap trick as
    // single-process mode).
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lk(ctx->seqMu);
        seq = ++ctx->dirSeq[me];
        ctx->lastIncoming[me] = seq;
    }

    if (cudaSetDevice(myOrd) != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_peer: cudaSetDevice failed");
        return -1;
    }
    const size_t n4 = bytes / 16;
    uint8_t *peerScratch = (uint8_t *)bar + ctx->scratchBase;
    cudaStream_t s = (cudaStream_t)stream;
    joinSendStream(ctx, s);

    // arm handshake (see the flag-slot protocol note): announce readiness
    // into the peer's arm slot, then wait MY arm slot (written by the
    // peer). My previous-step local add finishing is covered because this
    // store is stream-ordered after it.
    k_mark<<<1, 1, 0, s>>>(
        (unsigned long long *)((uint8_t *)bar + flagOff(ctx->pools[peer], me) + 8),
        seq);
    k_flag_wait<<<1, 32, 0, s>>>(
        (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, peer) + 8), seq);

    // BOTH payloads before the single completion mark: stream order lets
    // one fence + one flag cover both writes. The peer runs the identical
    // sequence concurrently, so the two directions overlap on the wire.
    k_copy<<<gridBlocks(n4, myOrd), 256, 0, s>>>(
        (const uint4 *)bPtr, (uint4 *)(peerScratch + ob), n4);
    k_copy<<<gridBlocks(n4, myOrd), 256, 0, s>>>(
        (const uint4 *)aPtr, (uint4 *)(peerScratch + oa), n4);
    k_mark<<<1, 1, 0, s>>>(
        (unsigned long long *)((uint8_t *)bar +
                               flagOff(ctx->pools[peer], me)), seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_peer: kernel launch failed");
        return -1;
    }

    // reader wait on MY completion flag (slot of the peer as writer; the
    // peer's mark wrote the same seq into it). Orders my stream after the
    // PEER's payloads landing in MY scratch zone.
    unsigned long long *myFlag = (unsigned long long *)(uintptr_t)(
        myP.dptr + flagOff(myP, peer));
    k_flag_wait<<<1, 32, 0, s>>>(myFlag, seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_peer: flag-wait launch failed");
        return -1;
    }

    // local adds against MY scratch copy of the PEER's original values
    uint4 *scrA = (uint4 *)(uintptr_t)(myP.dptr + ctx->scratchBase + oa);
    uint4 *scrB = (uint4 *)(uintptr_t)(myP.dptr + ctx->scratchBase + ob);
    if (launchAdd(dtype, (uint4 *)aPtr, (const uint4 *)scrA, n4,
                  gridBlocks(n4, myOrd), s) != 0 ||
        launchAdd(dtype, (uint4 *)bPtr, (const uint4 *)scrB, n4,
                  gridBlocks(n4, myOrd), s) != 0) {
        setErr(err, errlen, "bl_allreduce_peer: bad dtype (internal)");
        return -1;
    }
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_peer: add kernel launch failed");
        return -1;
    }
    // v1: conservative drain. The scratch zone is shared with verify/copy
    // traffic and the free would need cross-process agreement.
    cudaDeviceSynchronize();
    return 0;
}

extern "C" int bl_allreduce_into_peer(blCtx *ctx, const void *inPtr,
                                      void *outPtr, size_t bytes, int dtype,
                                      void *stream, char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || !ctx->peerMode || !inPtr || !outPtr || bytes == 0 ||
        bytes % 16 != 0) {
        setErr(err, errlen,
               "bl_allreduce_into_peer: bad argument (non-null pointers, "
               "size a non-zero multiple of 16)");
        return -1;
    }
    if (dtype <= BL_DTYPE_U8 || dtype > BL_DTYPE_FP16) {
        setErr(err, errlen,
               "bl_allreduce_into_peer: unsupported dtype (u8 keeps the "
               "mod-256 pool path; use bl_allreduce_peer)");
        return -1;
    }
    const int me = ctx->myRank, peer = 1 - me;
    Pool &myP = ctx->pools[me];
    const int myOrd = ctx->devices[me];
    if (((uintptr_t)inPtr | (uintptr_t)outPtr) % 16 != 0) {
        setErr(err, errlen,
               "bl_allreduce_into_peer: in/out must be 16-byte aligned "
               "(fresh torch allocations are)");
        return -1;
    }
    // allreduce keeps the BOTTOM half of the scratch zone (fixed offset
    // scratchBase); the fused small-message path owns its top slice, the
    // p2p protocol owns the top half
    const size_t scratchAvail = ctx->fusedZoneBase - ctx->scratchBase;
    if (bytes > scratchAvail) {
        setErr(err, errlen,
               "bl_allreduce_into_peer: size exceeds the peer scratch zone "
               "(pool/4 - flag tail); chunk larger tensors at the caller");
        return -1;
    }
    void *bar = pathDevPtr(ctx, peer, me);
    if (!bar) {
        setErr(err, errlen, "bl_allreduce_into_peer: no BAR1 write path");
        return -1;
    }

    if (cudaSetDevice(myOrd) != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_into_peer: cudaSetDevice failed");
        return -1;
    }
    const size_t n4 = bytes / 16;
    uint8_t *peerScratch = (uint8_t *)bar + ctx->scratchBase;
    cudaStream_t s = (cudaStream_t)stream;
    joinSendStream(ctx, s);

    // fused small-message path: 2 stream-ordered kernels instead of 6
    // (consumed-receipt gating + last-block flag/receipt election). Own
    // scratch slice (top of the ar half) and own receipt/flag slots, so
    // mixed fused/old sequences need no cross-protocol gating. See the
    // kernel notes above bl_allreduce_into_peer.
    if (bytes <= BL_FUSED_MAX) {
        uint64_t fseq;
        {
            std::lock_guard<std::mutex> lk(ctx->seqMu);
            fseq = ++ctx->fusedSeq;
        }
        const size_t fn4 = n4;
        const size_t fblocks = gridBlocks(fn4, myOrd);
        unsigned long long *myReceipt = (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, BL_AR_RECV_SLOT));
        unsigned long long *myFlagL = (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, BL_AR_FLAG_SLOT));
        unsigned long long *sendCnt = (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, BL_AR_CNT_SLOT));
        unsigned long long *recvCnt = (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, BL_AR_CNT_SLOT) + 8);
        unsigned long long *peerFlagBar = (unsigned long long *)((uint8_t *)bar +
            flagOff(ctx->pools[peer], BL_AR_FLAG_SLOT));
        unsigned long long *peerReceiptBar = (unsigned long long *)((uint8_t *)bar +
            flagOff(ctx->pools[peer], BL_AR_RECV_SLOT));
        k_fused_send<<<fblocks, 256, 0, s>>>(
            (const uint4 *)inPtr, (uint4 *)((uint8_t *)bar + ctx->fusedZoneBase),
            myReceipt, peerFlagBar, sendCnt, fseq, fn4);
        if (launchFusedRecv(dtype, (uint4 *)outPtr, (const uint4 *)inPtr,
                            (const uint4 *)(uintptr_t)(
                                myP.dptr + ctx->fusedZoneBase),
                            myFlagL, peerReceiptBar, recvCnt,
                            fseq, fn4, fblocks, s) != 0) {
            setErr(err, errlen,
                   "bl_allreduce_into_peer: fused path bad dtype (internal)");
            return -1;
        }
        if (cudaGetLastError() != cudaSuccess) {
            setErr(err, errlen,
                   "bl_allreduce_into_peer: fused kernel launch failed");
            return -1;
        }
        return 0;
    }

    // ONE seq per call per rank, one payload: my 'in' goes to the peer's
    // scratch zone at the FIXED offset scratchBase. Reusing that offset on
    // every call is safe under the SPMD discipline: my next payload is
    // stream-ordered after my arm-wait, which the peer only arms after its
    // add kernel (this call's scratch consumer) was enqueued on its stream.
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lk(ctx->seqMu);
        seq = ++ctx->dirSeq[me];
        ctx->lastIncoming[me] = seq;
    }

    // arm handshake (same protocol as bl_allreduce_peer)
    k_mark<<<1, 1, 0, s>>>(
        (unsigned long long *)((uint8_t *)bar + flagOff(ctx->pools[peer], me) + 8),
        seq);
    k_flag_wait<<<1, 32, 0, s>>>(
        (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, peer) + 8), seq);

    // single payload: read MY 'in' (plain local loads), st.global.wt into
    // the peer's scratch; one completion mark covers it
    k_copy<<<gridBlocks(n4, myOrd), 256, 0, s>>>(
        (const uint4 *)inPtr, (uint4 *)peerScratch, n4);
    k_mark<<<1, 1, 0, s>>>(
        (unsigned long long *)((uint8_t *)bar + flagOff(ctx->pools[peer], me)),
        seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_into_peer: kernel launch failed");
        return -1;
    }

    // reader wait on MY completion flag, then the add writes 'out' directly
    // -- no pool->tensor copy back. Fully stream-ordered; NO host sync.
    unsigned long long *myFlag = (unsigned long long *)(uintptr_t)(
        myP.dptr + flagOff(myP, peer));
    k_flag_wait<<<1, 32, 0, s>>>(myFlag, seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen,
               "bl_allreduce_into_peer: flag-wait launch failed");
        return -1;
    }

    const uint4 *scr = (const uint4 *)(uintptr_t)(
        myP.dptr + ctx->scratchBase);
    if (launchAddInto(dtype, (uint4 *)outPtr, (const uint4 *)inPtr, scr, n4,
                      gridBlocks(n4, myOrd), s) != 0) {
        setErr(err, errlen, "bl_allreduce_into_peer: bad dtype (internal)");
        return -1;
    }
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "bl_allreduce_into_peer: add kernel launch failed");
        return -1;
    }
    return 0;
}

// p2p flag slots: dedicated to send/recv (slots 2 and 3 of the 16-slot
// flag region; the allreduce/copy protocols use slots 0/1). In a given
// pool, the slot of writer w carries: +0 completion (w was the SENDER,
// raised after its payload) and +8 consumed-receipt (w was the RECEIVER,
// raised after its move kernel finished reading the scratch).
#define BL_P2P_SLOT(writerRank) (2 + (writerRank))

// Lazily create the p2p send stream + event (non-blocking stream, no timing
// event). The send stream is needed because several sends posted before the
// matching recvs would deadlock on a single stream: the second send's
// consumed-receipt wait would spin ahead of the recv that posts the receipt,
// and the peer is in the same state.
static int ensureSendStream(blCtx *ctx, char *err, size_t errlen)
{
    if (ctx->sendStream && ctx->evSend) return 0;
    if (cudaStreamCreateWithFlags(&ctx->sendStream, cudaStreamNonBlocking)
            != cudaSuccess) {
        setErr(err, errlen, "bl p2p: send-stream creation failed");
        return -1;
    }
    if (cudaEventCreateWithFlags(&ctx->evSend, cudaEventDisableTiming)
            != cudaSuccess) {
        setErr(err, errlen, "bl p2p: send-event creation failed");
        return -1;
    }
    return 0;
}

// Order a caller-stream protocol op (allreduce / copy / readback) after all
// pending p2p sends: a payload reads the user's 'in' tensor on the send
// stream, and the collective's add (or a pool copy) must not overwrite that
// tensor early.
static void joinSendStream(blCtx *ctx, cudaStream_t cs)
{
    if (ctx->sendStream && ctx->evSend)
        cudaStreamWaitEvent(cs, ctx->evSend, 0);
}

// Shared prologue for the zero-copy p2p pair: peer-mode/context checks,
// alignment + p2p-scratch-zone bounds, BAR write path, next ordinal of the
// direction's own counter. Returns 0 on success with the launch parameters
// filled; -1 with err set otherwise.
static int p2pPrelude(blCtx *ctx, const void *ptr, size_t bytes, int dtype,
                      int peerRank, int isSend, void **barOut, uint64_t *seqOut,
                      char *err, size_t errlen)
{
    if (!ctx || !ctx->peerMode || !ptr || bytes == 0 || bytes % 16 != 0) {
        setErr(err, errlen,
               "bl p2p: bad argument (non-null pointer, size a non-zero "
               "multiple of 16)");
        return -1;
    }
    if (dtype < BL_DTYPE_U8 || dtype > BL_DTYPE_FP16) {
        setErr(err, errlen, "bl p2p: unsupported dtype enum");
        return -1;
    }
    const int me = ctx->myRank;
    if (peerRank != 1 - me) {
        setErr(err, errlen,
               "bl p2p: peerRank must be 1 - myRank in the 2-rank peer link");
        return -1;
    }
    Pool &myP = ctx->pools[me];
    if ((uintptr_t)ptr % 16 != 0) {
        setErr(err, errlen,
               "bl p2p: pointer must be 16-byte aligned (fresh torch "
               "allocations are)");
        return -1;
    }
    const size_t p2pZone = myP.size - BL_FLAG_REGION - ctx->p2pScratchBase;
    if (bytes > p2pZone) {
        setErr(err, errlen,
               "bl p2p: size exceeds the p2p scratch zone (pool/4 - flag "
               "tail); chunk larger tensors at the caller");
        return -1;
    }
    void *bar = pathDevPtr(ctx, 1 - me, me);
    if (!bar) {
        setErr(err, errlen, "bl p2p: no BAR1 write path");
        return -1;
    }
    {
        std::lock_guard<std::mutex> lk(ctx->seqMu);
        *seqOut = isSend ? ++ctx->p2pSendSeq : ++ctx->p2pRecvSeq;
        ctx->lastIncoming[me] = *seqOut;
    }
    *barOut = bar;
    return 0;
}

extern "C" int bl_send_into_peer(blCtx *ctx, const void *inPtr, size_t bytes,
                                 int dtype, int peerRank, void *stream,
                                 char *err, size_t errlen)
{
    setErr(err, errlen, "");
    void *bar = nullptr;
    uint64_t seq = 0;
    if (p2pPrelude(ctx, inPtr, bytes, dtype, peerRank, 1, &bar, &seq,
                   err, errlen) != 0)
        return -1;
    const int me = ctx->myRank, peer = 1 - me;
    Pool &myP = ctx->pools[me];
    const int myOrd = ctx->devices[me];
    if (cudaSetDevice(myOrd) != cudaSuccess) {
        setErr(err, errlen, "bl_send_into_peer: cudaSetDevice failed");
        return -1;
    }
    if (ensureSendStream(ctx, err, errlen) != 0)
        return -1;
    const size_t n4 = bytes / 16;
    uint8_t *peerScratch = (uint8_t *)bar + ctx->p2pScratchBase;
    cudaStream_t ss = ctx->sendStream;

    // join the caller's stream: the payload must see every local write to
    // 'in' enqueued so far (user fills, prior collectives). Record on the
    // caller's stream, wait on the send stream.
    cudaEvent_t evCaller = nullptr;
    if (cudaEventCreateWithFlags(&evCaller, cudaEventDisableTiming)
            != cudaSuccess) {
        setErr(err, errlen, "bl_send_into_peer: caller-event creation failed");
        return -1;
    }
    cudaEventRecord(evCaller, (cudaStream_t)stream);
    cudaStreamWaitEvent(ss, evCaller, 0);
    cudaEventDestroy(evCaller);

    // consumed-receipt wait for exchange k > 1 (on the send stream): the
    // peer's recv k-1 posted it stream-ordered after its move kernel
    // finished READING the scratch, so this payload cannot overwrite
    // scratch the peer is still reading. (k == 1 skips the wait: nothing
    // consumed the p2p zone before.)
    if (seq > 1) {
        k_flag_wait<<<1, 32, 0, ss>>>(
            (unsigned long long *)(uintptr_t)(
                myP.dptr + flagOff(myP, BL_P2P_SLOT(peer)) + 8), seq - 1);
    }
    // payload: read MY 'in' (plain local loads), st.global.wt into the
    // peer's p2p scratch zone; one completion mark covers it
    k_copy<<<gridBlocks(n4, myOrd), 256, 0, ss>>>(
        (const uint4 *)inPtr, (uint4 *)peerScratch, n4);
    k_mark<<<1, 1, 0, ss>>>(
        (unsigned long long *)((uint8_t *)bar +
                               flagOff(ctx->pools[peer], BL_P2P_SLOT(me))),
        seq);
    if (cudaEventRecord(ctx->evSend, ss) != cudaSuccess ||
        cudaGetLastError() != cudaSuccess) {
        std::string es = cudaGetErrorString(cudaGetLastError());
        setErr(err, errlen, "bl_send_into_peer: kernel launch failed: " + es);
        return -1;
    }
    return 0;
}

extern "C" int bl_recv_into_peer(blCtx *ctx, void *outPtr, size_t bytes,
                                 int dtype, int peerRank, void *stream,
                                 char *err, size_t errlen)
{
    setErr(err, errlen, "");
    void *bar = nullptr;
    uint64_t seq = 0;
    if (p2pPrelude(ctx, outPtr, bytes, dtype, peerRank, 0, &bar, &seq,
                   err, errlen) != 0)
        return -1;
    const int me = ctx->myRank, peer = 1 - me;
    Pool &myP = ctx->pools[me];
    const int myOrd = ctx->devices[me];
    if (cudaSetDevice(myOrd) != cudaSuccess) {
        setErr(err, errlen, "bl_recv_into_peer: cudaSetDevice failed");
        return -1;
    }
    const size_t n4 = bytes / 16;
    cudaStream_t s = (cudaStream_t)stream;

    // 1. wait the peer's completion flag for this ordinal, 2. move the p2p
    //    scratch zone -> 'out', 3. post the consumed-receipt: stream-ordered
    //    after the move kernel, so the peer's NEXT payload cannot overwrite
    //    the zone while this move is reading it. All on the CALLER's
    //    stream (recv is not prone to the multi-send deadlock).
    k_flag_wait<<<1, 32, 0, s>>>(
        (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, BL_P2P_SLOT(peer))), seq);
    k_move<<<gridBlocks(n4, myOrd), 256, 0, s>>>(
        (uint4 *)outPtr,
        (const uint4 *)(uintptr_t)(myP.dptr + ctx->p2pScratchBase), n4);
    k_mark<<<1, 1, 0, s>>>(
        (unsigned long long *)((uint8_t *)bar +
                               flagOff(ctx->pools[peer], BL_P2P_SLOT(me)) + 8),
        seq);
    if (cudaGetLastError() != cudaSuccess) {
        setErr(err, errlen, "bl_recv_into_peer: kernel launch failed");
        return -1;
    }
    return 0;
}

extern "C" uint64_t bl_verify_peer(blCtx *ctx, char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || !ctx->peerMode) return ~0ull;
    const int me = ctx->myRank, peer = 1 - me;
    Pool &myP = ctx->pools[me];
    const int myOrd = ctx->devices[me];

    size_t win = myP.size - BL_FLAG_REGION - ctx->scratchBase;
    if (win > (4ull << 20)) win = 4ull << 20;
    win &= ~(size_t)15;
    const size_t off = ctx->scratchBase;
    const unsigned seed = 0xB1;

    void *bar = pathDevPtr(ctx, peer, me);
    if (!bar) {
        setErr(err, errlen, "bl_verify_peer: no BAR1 write path");
        return ~0ull;
    }

    uint64_t seq;
    {
        std::lock_guard<std::mutex> lk(ctx->seqMu);
        seq = ++ctx->dirSeq[me];
        ctx->lastIncoming[me] = seq;
    }

    if (cudaSetDevice(myOrd) != cudaSuccess) return ~0ull;
    const size_t n4 = win / 16;
    // both ranks write the SAME pattern into the peer's scratch zone, so
    // the two directions cannot conflict on content. Arm handshake first
    // (see the flag-slot protocol note).
    k_mark<<<1, 1>>>(
        (unsigned long long *)((uint8_t *)bar + flagOff(ctx->pools[peer], me) + 8),
        seq);
    k_flag_wait<<<1, 32>>>(
        (unsigned long long *)(uintptr_t)(
            myP.dptr + flagOff(myP, peer) + 8), seq);
    k_pattern<<<gridBlocks(n4, myOrd), 256>>>(
        (uint4 *)((uint8_t *)bar + off), n4, seed);
    k_mark<<<1, 1>>>(
        (unsigned long long *)((uint8_t *)bar + flagOff(ctx->pools[peer], me)),
        seq);
    if (cudaGetLastError() != cudaSuccess) return ~0ull;

    // wait for the PEER's pattern to land in MY scratch zone
    unsigned long long *myFlag = (unsigned long long *)(uintptr_t)(
        myP.dptr + flagOff(myP, peer));
    k_flag_wait<<<1, 32>>>(myFlag, seq);
    if (cudaGetLastError() != cudaSuccess) return ~0ull;

    // verify MY local scratch zone through my own VMM pointer
    BlVerifyOut init = { 0, ~0ull }, res;
    BlVerifyOut *dres = nullptr;
    if (cudaMalloc(&dres, sizeof(*dres)) != cudaSuccess) return ~0ull;
    if (cudaMemcpy(dres, &init, sizeof(init), cudaMemcpyHostToDevice)
            != cudaSuccess) return ~0ull;
    k_verify<<<gridBlocks(n4, myOrd), 256>>>(
        (const uint4 *)(uintptr_t)(myP.dptr + off), n4, seed, dres);
    if (cudaGetLastError() != cudaSuccess) return ~0ull;
    if (cudaDeviceSynchronize() != cudaSuccess) return ~0ull;
    if (cudaMemcpy(&res, dres, sizeof(res), cudaMemcpyDeviceToHost)
            != cudaSuccess) return ~0ull;
    cudaFree(dres);

    // Informational secondary proof: read MY OWN pattern back through the
    // BAR-view pointer (a GPU-side read of the peer BAR aperture with
    // ld.relaxed.sys). Our protocol never polls BAR-view flags (the local
    // flag tail is the ordered channel), but this confirms the BAR is
    // readable from the GPU, which a BAR-polling design would rely on.
    {
        void *stage = nullptr;
        bool barReadOk = false;
        if (cudaMalloc(&stage, 64) == cudaSuccess) {
            k_readback<<<1, 4>>>((const uint4 *)((uint8_t *)bar + off),
                                 (uint4 *)stage, 4);
            if (cudaGetLastError() == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess) {
                uint8_t got[64];
                if (cudaMemcpy(got, stage, 64, cudaMemcpyDeviceToHost)
                        == cudaSuccess) {
                    barReadOk = true;
                    for (int k = 0; k < 64 && barReadOk; ++k)
                        barReadOk = got[k] == (uint8_t)(patVal(
                            (uint64_t)k & ~(uint64_t)15, seed,
                            (int)(((uint64_t)k & 15) >> 2)) >>
                            (((int)k & 3) * 8));
                }
            }
            cudaFree(stage);
        }
        std::printf("bl_verify_peer: rank %d: BAR-view readback: %s\n", me,
                    barReadOk ? "OK" : "UNUSABLE (informational only)");
    }

    std::printf("bl_verify_peer: rank %d: peer -> me bad_bytes = %llu of %zu\n",
                me, (unsigned long long)res.bad, win);
    return res.bad;
}

extern "C" uint64_t bl_verify(blCtx *ctx, char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx) return ~0ull;
    if (ctx->peerMode) return bl_verify_peer(ctx, err, errlen);
    uint64_t totalBad = 0;
    const size_t win = ctx->poolBytes < (4ull << 20) ? ctx->poolBytes
                                                     : (4ull << 20);
    const unsigned seed = 0xB1;

    for (int owner = 0; owner < ctx->ndev; ++owner) {
        for (int writer = 0; writer < ctx->ndev; ++writer) {
            if (owner == writer) continue;
            void *tmp = nullptr;
            if (bl_pool_alloc(ctx, owner, win, &tmp, err, errlen) != 0)
                return ~0ull;
            void *bar = pathDevPtr(ctx, owner, writer);
            if (!bar) {
                setErr(err, errlen, "bl_verify: missing write path");
                return ~0ull;
            }

            const int wOrd = ctx->devices[writer];
            const int oOrd = ctx->devices[owner];
            if (cudaSetDevice(wOrd) != cudaSuccess) return ~0ull;
            size_t n4 = win / 16;
            k_pattern<<<gridBlocks(n4, wOrd), 256>>>(
                (uint4 *)((uint8_t *)bar +
                          ((uintptr_t)tmp - (uintptr_t)ctx->pools[owner].dptr)),
                n4, seed);
            if (cudaGetLastError() != cudaSuccess) return ~0ull;
            if (cudaDeviceSynchronize() != cudaSuccess) return ~0ull;

            // owner-side verify through its own VMM pointer
            if (cudaSetDevice(oOrd) != cudaSuccess) return ~0ull;
            BlVerifyOut init = { 0, ~0ull }, res;
            BlVerifyOut *dres = nullptr;
            if (cudaMalloc(&dres, sizeof(*dres)) != cudaSuccess) return ~0ull;
            if (cudaMemcpy(dres, &init, sizeof(init), cudaMemcpyHostToDevice)
                    != cudaSuccess) return ~0ull;
            k_verify<<<gridBlocks(n4, oOrd), 256>>>(
                (const uint4 *)tmp, n4, seed, dres);
            if (cudaGetLastError() != cudaSuccess) return ~0ull;
            if (cudaDeviceSynchronize() != cudaSuccess) return ~0ull;
            if (cudaMemcpy(&res, dres, sizeof(res), cudaMemcpyDeviceToHost)
                    != cudaSuccess) return ~0ull;
            cudaFree(dres);

            std::printf("bl_verify: dev %d -> dev %d: bad_bytes = %llu\n",
                        wOrd, oOrd, (unsigned long long)res.bad);
            totalBad += res.bad;
            bl_pool_free(ctx, owner, tmp, win);
        }
    }
    return totalBad;
}

// ---------------------------------------------------------------------------
// Reliable host readback (kernel + ld.relaxed.sys) of an arbitrary pool region
// ---------------------------------------------------------------------------

extern "C" int bl_readback(blCtx *ctx, int devIdx, void *ptr, size_t bytes,
                           void *dstHost, char *err, size_t errlen)
{
    setErr(err, errlen, "");
    if (!ctx || devIdx < 0 || devIdx >= ctx->ndev || !ptr || !dstHost ||
        bytes == 0 || bytes % 16 != 0) {
        setErr(err, errlen,
               "bl_readback: bad argument (size must be a multiple of 16)");
        return -1;
    }
    Pool &P = ctx->pools[devIdx];
    uintptr_t base = (uintptr_t)P.dptr;
    if ((uintptr_t)ptr < base || (uintptr_t)ptr + bytes > base + P.size) {
        setErr(err, errlen, "bl_readback: pointer outside the pool");
        return -1;
    }
    const int ord = ctx->devices[devIdx];
    if (cudaSetDevice(ord) != cudaSuccess) {
        setErr(err, errlen, "bl_readback: cudaSetDevice failed");
        return -1;
    }
    joinSendStream(ctx, (cudaStream_t)0);   // legacy default stream
    // wait for the most recent inbound copy to this pool (queued on the
    // legacy default stream, which is synchronizing with the users' blocking
    // streams; TODO: take an explicit stream argument for non-blocking
    // stream users)
    uint64_t waitSeq;
    {
        std::lock_guard<std::mutex> lk(ctx->seqMu);
        waitSeq = ctx->lastIncoming[devIdx];
    }
    if (waitSeq > 0) {
        int wIdx = (devIdx + 1) % ctx->ndev;   // v1: exactly one writer
        unsigned long long *flagLocal = (unsigned long long *)(uintptr_t)(
            P.dptr + flagOff(P, wIdx));
        k_flag_wait<<<1, 32>>>(flagLocal, waitSeq);
        if (cudaGetLastError() != cudaSuccess) {
            setErr(err, errlen, "bl_readback: flag-wait launch failed");
            return -1;
        }
    }
    size_t n4 = bytes / 16;
    void *stage = nullptr;
    if (cudaMalloc(&stage, bytes) != cudaSuccess) {
        setErr(err, errlen, "bl_readback: cudaMalloc failed");
        return -1;
    }
    k_readback<<<gridBlocks(n4, ord), 256>>>((const uint4 *)ptr,
                                             (uint4 *)stage, (int)n4);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(stage);
        setErr(err, errlen, "bl_readback: kernel launch failed");
        return -1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(stage);
        setErr(err, errlen, "bl_readback: kernel failed");
        return -1;
    }
    if (cudaMemcpy(dstHost, stage, bytes, cudaMemcpyDeviceToHost)
            != cudaSuccess) {
        cudaFree(stage);
        setErr(err, errlen, "bl_readback: DtoH failed");
        return -1;
    }
    cudaFree(stage);
    return 0;
}
