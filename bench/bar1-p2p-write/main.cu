// SPDX-License-Identifier: MIT
//
// bar1-p2p-write -- native C++/CUDA benchmark for direct GPU-to-GPU writes
//                  through the target card's dynamically-mapped BAR1 aperture.
//
// Single process, two devices. Device A (source) writes with a plain
// st.global.wt kernel into device B's BAR1; the PCIe writes land directly in
// device B's VRAM. No Python, no barlink library, no NCCL, no fd exchange
// between processes.
//
// Data path (all stock driver mechanisms except the guard, see
// BARLINK_PCIE_MINIMAL.patch):
//
//   1. VMM allocation on the TARGET card (cuMemCreate/cuMemAddressReserve/
//      cuMemMap/cuMemSetAccess).
//   2. dma-buf export via the RM ioctl NV_ESC_EXPORT_TO_DMABUF_FD. On GeForce,
//      cuMemGetHandleForAddressRange(DMA_BUF_FD) is rejected by the driver
//      (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0), so the RM ioctl path is
//      used directly.
//   3. Importer: the dmabuf_holder kernel module performs
//      dma_buf_attach + dma_buf_map_attachment as the SOURCE card's PCI
//      device. The map_attachment call triggers nv_dma_buf_map()
//      (nv-dmabuf.c:1066), which programs the target card's BAR1 pages
//      DYNAMICALLY. No static-BAR regkey (RMForceStaticBar1 etc.) anywhere.
//   4. The target card's BAR1 (resource1_wc) is mmap'ed, registered on the
//      source card with cudaHostRegister(cudaHostRegisterIoMemory), and
//      cudaHostGetDevicePointer yields a source-card device pointer.
//   5. Phase 1: the source kernel writes an offset-dependent pattern through
//      that pointer; the target card reads the buffer back through its OWN
//      VMM pointer and bad_bytes is reported. Phase 2: bandwidth sweep.
//
// Build:  make            (see Makefile; nvcc, -lcuda, sm_86 default)
// Run:    sudo ./bar1-p2p-write [--size=N] [--iters=N] [--reverse|--both]
//
// Prerequisites:
//   - patched driver loaded (BARLINK_PCIE_MINIMAL.patch), regkey
//     BarlinkPeerBar1=1 set
//   - dmabuf_holder.ko loaded (creates /dev/dmabuf_holder, mode 0600)
//   - run as root / CAP_SYS_ADMIN
//
// The RM ioctl and dmabuf_holder ABI definitions below are inlined (with
// their source) so this bench is self-contained and does not depend on the
// driver source tree. Layouts verified against
// barlink-torch/drv/595.104.02/kernel-open/common/inc/nv-ioctl*.h and
// barlink-pcie/dmabuf_holder/dmabuf_holder.h.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Inlined kernel-interface definitions (source: open-gpu-kernel-modules
// 595.104.02 headers; plain C types, natural alignment == kernel layout)
// ---------------------------------------------------------------------------

typedef uint8_t  NvU8;
typedef uint16_t NvU16;
typedef uint32_t NvU32;
typedef int32_t  NvS32;
typedef uint64_t NvU64;
typedef uint32_t NvHandle;
typedef uint8_t  NvBool;   // nvtypes.h: typedef NvU8 NvBool
typedef uint64_t NvP64;

#define NV_IOCTL_MAGIC               'F'
#define NV_IOCTL_BASE                200
#define NV_ESC_CARD_INFO             (NV_IOCTL_BASE + 0)    // 200
#define NV_ESC_CHECK_VERSION_STR     (NV_IOCTL_BASE + 10)   // 210
#define NV_ESC_EXPORT_TO_DMABUF_FD   (NV_IOCTL_BASE + 17)   // 217
#define NV_ESC_RM_CONTROL            0x2A
#define NV_ESC_RM_ALLOC              0x2B

#define NV_RM_API_VERSION_STRING_LENGTH 64
#define NV_RM_API_VERSION_CMD_RELAXED   '1'

// nv-ioctl.h: nv_ioctl_rm_api_version
typedef struct {
    NvU32 cmd;
    NvU32 reply;
    char  versionString[NV_RM_API_VERSION_STRING_LENGTH];
} nv_ioctl_rm_api_version_t;

// nv-ioctl.h: nv_pci_info_t / nv_ioctl_card_info
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

// nv-ioctl.h: nv_ioctl_export_to_dma_buf_fd
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

// nvos.h
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

// class/cl0000.h, class/cl0080.h
#define NV01_ROOT   0x0U
#define NV01_DEVICE_0 0x80U

// class/cl0080.h: NV0080_ALLOC_PARAMETERS
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

// ctrl/ctrl0000/ctrl0000gpu.h
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

// ctrl/ctrl0000/ctrl0000unix.h
#define NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD 0x3d06U
typedef struct {
    NvU32 type;             // 1 = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM
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

// ---------------------------------------------------------------------------
// dmabuf_holder ABI (source: barlink-pcie/dmabuf_holder/dmabuf_holder.h,
// GPL-2.0; the module must be loaded, see Prerequisites)
// ---------------------------------------------------------------------------

#define DMABUF_HOLDER_DEVICE_PATH "/dev/dmabuf_holder"
#define DMABUF_HOLDER_F_BDF_VALID (1u << 0)

struct dmabuf_holder_sg_entry {
    uint64_t dma_address;   // sg_dma_address()
    uint64_t dma_len;       // sg_dma_len()
};

struct dmabuf_holder_hold {
    // input
    int32_t dmabuf_fd;
    uint32_t flags;
    uint32_t pci_domain;
    uint8_t  pci_bus;
    uint8_t  pci_slot;
    uint8_t  pci_func;
    uint8_t  reserved0;
    uint32_t max_entries;
    uint32_t reserved1;
    uint64_t entries;       // userspace dmabuf_holder_sg_entry[]
    // output
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
// Error handling
// ---------------------------------------------------------------------------

static const char *drvErr(CUresult r)
{
    const char *s = nullptr;
    cuGetErrorString(r, &s);
    return s ? s : "?";
}

#define DRV(call)                                                              \
    do {                                                                       \
        CUresult _r = (call);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            std::fprintf(stderr, "Driver API error at %s: %d (%s)\n",          \
                         #call, (int)_r, drvErr(_r));                          \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RT(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "Runtime error at %s: %d (%s)\n",             \
                         #call, (int)_e, cudaGetErrorString(_e));              \
            return false;                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Test pattern: 64-bit value derived from the byte offset (and a seed), so
// every byte written through the peer BAR can be checked against the target
// card's own VMM readback. Identical __host__/__device__ implementation.
// ---------------------------------------------------------------------------

__host__ __device__ static inline uint32_t patVal(uint64_t byteOff, unsigned seed, int lane)
{
    uint64_t x = byteOff ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(seed + 1));
    return (uint32_t)(x >> (lane * 16)) ^ (uint32_t)((x * 0x85ebca6bULL) >> 32);
}

// st.global.wt: write-through, the store does not stay resident in any cache
// level -- right choice for MMIO/peer-BAR targets. 128-bit is the widest unit
// that stays coalesced over PCIe (same rationale as bar1_collective.cu).
__device__ __forceinline__ static void stwt128(void *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// Grid-stride write of the whole [0, n4) uint4 window with pattern 'seed'.
__global__ void k_write(uint4 *__restrict__ dst, size_t n4, unsigned seed)
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

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static std::string lower(std::string s)
{
    for (char &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

struct Bdf {
    unsigned domain = 0, bus = 0, slot = 0, func = 0;
    bool valid = false;
};

// Accepts "0000:0a:00.0" or "0a:00.0".
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

static std::string bdfString(const Bdf &b)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x",
                  b.domain, b.bus, b.slot, b.func);
    return buf;
}

// Reads start and end of BAR<idx> from /sys/bus/pci/devices/<bdf>/resource.
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

// ---------------------------------------------------------------------------
// BAR1 scan (fallback for the IOMMU-translating case, where sg addresses are
// IOVAs outside the BAR1 range)
// ---------------------------------------------------------------------------

// Searches for magicLen bytes in the BAR and returns the offset or (uint64_t)-1.
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
// dma-buf export via the RM ioctls (substitute for the GeForce-rejected
// cuMemGetHandleForAddressRange). Mirrors nvExportToDmabuf() in
// barlink-pcie/probes/dmabuf_pcie_probe.cpp.
// ---------------------------------------------------------------------------

struct NvExport {
    int      ctlFd = -1;   // /dev/nvidiactl, holds our own RM client
    int      devFd = -1;   // /dev/nvidia<minor>
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

static bool nvExportToDmabuf(NvExport &e, int objfd, int pciBus, size_t size)
{
    e.ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (e.ctlFd < 0) {
        std::perror("open /dev/nvidiactl (is the NVIDIA driver loaded?)");
        return false;
    }

    {   // version handshake
        nv_ioctl_rm_api_version_t v;
        char buf[256] = {0}, ver[64] = {0};
        std::memset(&v, 0, sizeof(v));
        v.cmd = NV_RM_API_VERSION_CMD_RELAXED;
        FILE *f = std::fopen("/proc/driver/nvidia/version", "r");
        if (f) { if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0; std::fclose(f); }
        char *p = std::strstr(buf, "for x86_64");
        if (p && std::sscanf(p, "for x86_64 %63s", ver) == 1)
            std::strncpy(v.versionString, ver, sizeof(v.versionString) - 1);
        if (nvIoctl(e.ctlFd, NV_ESC_CHECK_VERSION_STR, &v, sizeof(v)) < 0) {
            std::perror("NV_ESC_CHECK_VERSION_STR"); return false;
        }
    }

    NvU32 gpuId = 0, minor = 0;
    {
        nv_ioctl_card_info_t ci[32];
        std::memset(ci, 0, sizeof(ci));
        if (nvIoctl(e.ctlFd, NV_ESC_CARD_INFO, ci, sizeof(ci)) < 0) {
            std::perror("NV_ESC_CARD_INFO"); return false;
        }
        bool found = false;
        for (int i = 0; i < 32; ++i) {
            if (!ci[i].valid) continue;
            if ((int)ci[i].pci_info.bus == pciBus) {
                gpuId = ci[i].gpu_id; minor = ci[i].minor_number; found = true;
            }
        }
        if (!found) {
            std::fprintf(stderr, "PCI bus 0x%02x not found in NV_ESC_CARD_INFO\n",
                         pciBus);
            return false;
        }
    }

    {   // own RM client
        NVOS21_PARAMETERS a;
        std::memset(&a, 0, sizeof(a));
        a.hClass = NV01_ROOT;
        if (nvIoctl(e.ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0) {
            std::fprintf(stderr, "RM_ALLOC(root) failed, status=0x%x\n", a.status);
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
            std::fprintf(stderr, "GPU_GET_ID_INFO_V2 status=0x%x\n", c.status);
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
            std::fprintf(stderr, "RM_ALLOC(device) status=0x%x\n", a.status);
            return false;
        }
    }

    {   // import the object fd into our own client
        NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.fd = objfd;
        p.object.type = 1;  // NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM
        p.object.rmObject.hDevice = e.hDevice;
        p.object.rmObject.hParent = e.hDevice;
        p.object.rmObject.hObject = e.hMemory;
        c.hClient = e.hClient; c.hObject = e.hClient;
        c.cmd     = NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD;
        c.params  = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(e.ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0) {
            std::fprintf(stderr, "IMPORT_OBJECT_FROM_FD status=0x%x\n", c.status);
            return false;
        }
    }

    {   // RM ioctl: NV_ESC_EXPORT_TO_DMABUF_FD
        char devpath[64];
        std::snprintf(devpath, sizeof(devpath), "/dev/nvidia%u", minor);
        e.devFd = open(devpath, O_RDWR);
        if (e.devFd < 0) { std::perror(devpath); return false; }

        nv_ioctl_export_to_dma_buf_fd_t p;
        std::memset(&p, 0, sizeof(p));
        p.fd           = -1;                 // -1 = create a new dma-buf
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
            std::perror("NV_ESC_EXPORT_TO_DMABUF_FD");
            return false;
        }
        if (p.status != 0) {
            std::fprintf(stderr, "NV_ESC_EXPORT_TO_DMABUF_FD status=0x%08x\n",
                         p.status);
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
// dmabuf_holder wrappers
// ---------------------------------------------------------------------------

static bool holderHold(int holderFd, int dmabufFd, const Bdf &attachTo,
                       std::vector<dmabuf_holder_sg_entry> &sg, uint32_t *handle)
{
    const uint32_t kMaxEntries = 8192;
    sg.assign(kMaxEntries, dmabuf_holder_sg_entry{});

    dmabuf_holder_hold arg;
    std::memset(&arg, 0, sizeof(arg));
    arg.dmabuf_fd   = dmabufFd;
    arg.flags       = DMABUF_HOLDER_F_BDF_VALID;
    arg.pci_domain  = attachTo.domain;
    arg.pci_bus     = (uint8_t)attachTo.bus;
    arg.pci_slot    = (uint8_t)attachTo.slot;
    arg.pci_func    = (uint8_t)attachTo.func;
    arg.max_entries = kMaxEntries;
    arg.entries     = (uint64_t)(uintptr_t)sg.data();

    if (ioctl(holderFd, DMABUF_HOLDER_IOC_HOLD, &arg) != 0) {
        int e = errno;
        std::fprintf(stderr, "DMABUF_HOLDER_IOC_HOLD: %s\n", std::strerror(e));
        // 524 is kernel-internal ENOTSUPP surfacing from nv_dma_buf_attach()
        // (nv-dmabuf.c:1002, :1023, :1040).
        if (e == 524 || e == EOPNOTSUPP || e == ENOTSUP)
            std::fprintf(stderr,
                "  The exporter rejected the attach (nv_dma_buf_attach,\n"
                "  nv-dmabuf.c:1002). Most likely causes:\n"
                "    - the PCI topology check rejects this pair\n"
                "      (nv_grdma_pci_topology_supported, nv-pci.c:2708)\n"
                "    - the patched driver is not loaded, or the\n"
                "      BarlinkPeerBar1=1 regkey is not set\n");
        else if (e == ENODEV)
            std::fprintf(stderr, "  PCI device %s does not exist.\n",
                         bdfString(attachTo).c_str());
        else if (e == EPERM || e == EACCES)
            std::fprintf(stderr,
                "  Permission denied -- run as root / with CAP_SYS_ADMIN.\n");
        sg.clear();
        return false;
    }

    *handle = arg.handle;
    if (arg.nents < kMaxEntries) sg.resize(arg.nents);
    std::printf("  HOLD OK: handle %u, attached to %s, dma-buf size %llu, "
                "%u sg entries, total %llu bytes\n",
                arg.handle, bdfString(attachTo).c_str(),
                (unsigned long long)arg.dmabuf_size, arg.nents,
                (unsigned long long)arg.total_len);
    return true;
}

static void holderRelease(int holderFd, uint32_t handle)
{
    dmabuf_holder_release rel;
    std::memset(&rel, 0, sizeof(rel));
    rel.handle = handle;
    if (handle && ioctl(holderFd, DMABUF_HOLDER_IOC_RELEASE, &rel) != 0)
        std::fprintf(stderr, "DMABUF_HOLDER_IOC_RELEASE: %s\n",
                     std::strerror(errno));
}

// ---------------------------------------------------------------------------
// The link: everything needed for one source -> target direction
// ---------------------------------------------------------------------------

struct Link {
    int      srcOrd = -1, dstOrd = -1;
    size_t   allocSize = 0;      // VMM allocation size (granularity-rounded)
    CUdeviceptr dstDptr = 0;     // target card's own VMM pointer
    CUmemGenericAllocationHandle memHandle = 0;
    NvExport nvx;
    int      holderFd = -1;
    uint32_t holderHandle = 0;
    int      dmabufFd = -1;
    int      barFd = -1;
    void    *bar = nullptr;      // mmap'ed BAR1 slice (page-aligned)
    size_t   mapLen = 0;
    uint64_t delta = 0;          // buffer start within the mapped slice
    void    *srcDevPtr = nullptr; // device pointer on the SOURCE card
    cudaStream_t stream = nullptr;
};

static bool linkSetup(Link &L, int srcOrd, int dstOrd, size_t wantBytes)
{
    L.srcOrd = srcOrd;
    L.dstOrd = dstOrd;
    char busId[64] = {0};
    std::string srcBDF, dstBDF;

    RT(cudaSetDevice(dstOrd));
    RT(cudaFree(nullptr));               // create the target card's context
    RT(cudaSetDevice(srcOrd));
    RT(cudaFree(nullptr));               // create the source card's context

    RT(cudaDeviceGetPCIBusId(busId, sizeof(busId), dstOrd));
    dstBDF = lower(busId);
    std::memset(busId, 0, sizeof(busId));
    RT(cudaDeviceGetPCIBusId(busId, sizeof(busId), srcOrd));
    srcBDF = lower(busId);
    std::printf("Target card (receiver) BDF = %s, source card (writer) BDF = %s\n",
                dstBDF.c_str(), srcBDF.c_str());

    CUdevice dstDev = 0;
    DRV(cuDeviceGet(&dstDev, dstOrd));

    // --- 1. VMM allocation on the target card --------------------------------
    RT(cudaSetDevice(dstOrd));
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(prop));
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = dstDev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t gran = 0;
    DRV(cuMemGetAllocationGranularity(&gran, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    if (gran == 0) gran = 2ull << 20;
    L.allocSize = ((wantBytes + gran - 1) / gran) * gran;
    std::printf("VMM granularity = %zu, allocation size = %zu\n",
                gran, L.allocSize);

    DRV(cuMemCreate(&L.memHandle, L.allocSize, &prop, 0));
    DRV(cuMemAddressReserve(&L.dstDptr, L.allocSize, gran, 0, 0));
    DRV(cuMemMap(L.dstDptr, L.allocSize, 0, L.memHandle, 0));

    CUmemAccessDesc acc;
    std::memset(&acc, 0, sizeof(acc));
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id   = dstDev;
    acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DRV(cuMemSetAccess(L.dstDptr, L.allocSize, &acc, 1));
    std::printf("VMM buffer on target card: dptr = 0x%llx, size = %zu\n",
                (unsigned long long)L.dstDptr, L.allocSize);

    // --- write a marker (used by the BAR1 scan fallback) ----------------------
    uint8_t magic[64];
    std::memset(magic, 0, sizeof(magic));
    std::snprintf((char *)magic, sizeof(magic),
                  "BAR1-P2P-WRITE-%08x%08x-%d-%d",
                  (unsigned)getpid(), (unsigned)time(nullptr), dstOrd, srcOrd);
    DRV(cuMemcpyHtoD(L.dstDptr, magic, sizeof(magic)));
    DRV(cuCtxSynchronize());

    // --- 2. dma-buf export -----------------------------------------------------
    // GeForce rejects cuMemGetHandleForAddressRange(DMA_BUF_FD), so the RM
    // ioctl path is the primary one; the convenience call is attempted first
    // anyway (works on datacenter cards).
    if (cuMemGetHandleForAddressRange(&L.dmabufFd, L.dstDptr, L.allocSize,
                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                      0) == CUDA_SUCCESS) {
        std::printf("dma-buf fd = %d (cuMemGetHandleForAddressRange)\n", L.dmabufFd);
    } else {
        int shareFd = -1;
        DRV(cuMemExportToShareableHandle(&shareFd, L.memHandle,
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        int pciBus = -1;
        DRV(cuDeviceGetAttribute(&pciBus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dstDev));
        if (!nvExportToDmabuf(L.nvx, shareFd, pciBus, L.allocSize)) {
            std::fprintf(stderr,
                "dma-buf export via the RM ioctls failed. Is the patched\n"
                "driver loaded and /dev/nvidiactl accessible?\n");
            return false;
        }
        L.dmabufFd = L.nvx.dmabufFd;
        close(shareFd);
        std::printf("dma-buf fd = %d (RM ioctl NV_ESC_EXPORT_TO_DMABUF_FD)\n",
                    L.dmabufFd);
    }

    // --- 3. importer: dmabuf_holder HOLD with the source card's BDF ------------
    L.holderFd = open(DMABUF_HOLDER_DEVICE_PATH, O_RDWR | O_CLOEXEC);
    if (L.holderFd < 0) {
        int e = errno;
        std::fprintf(stderr, "open(%s): %s\n", DMABUF_HOLDER_DEVICE_PATH,
                     std::strerror(e));
        if (e == ENOENT)
            std::fprintf(stderr,
                "  /dev/dmabuf_holder does not exist -- build and load the\n"
                "  module first:\n"
                "      make -C /lib/modules/$(uname -r)/build \\\n"
                "           M=<barlink-pcie>/dmabuf_holder modules CC=gcc-14\n"
                "      insmod <barlink-pcie>/dmabuf_holder/dmabuf_holder.ko\n");
        else if (e == EACCES || e == EPERM)
            std::fprintf(stderr,
                "  Permission denied (module creates the node 0600) -- run as\n"
                "  root / with CAP_SYS_ADMIN.\n");
        return false;
    }

    // The importer must be a real PCI device: nv_dma_buf_attach() calls
    // to_pci_dev(attachment->dev) unchecked (nv-dmabuf.c:1033). The source
    // card is the natural choice -- it is also the device that writes into
    // the BAR later, so the topology check matches.
    Bdf attachTo = parseBdf(srcBDF.c_str());
    if (!attachTo.valid) {
        std::fprintf(stderr, "cannot parse source BDF '%s'\n", srcBDF.c_str());
        return false;
    }
    std::vector<dmabuf_holder_sg_entry> sg;
    if (!holderHold(L.holderFd, L.dmabufFd, attachTo, sg, &L.holderHandle))
        return false;
    // dma_buf_map_attachment has run -> nv_dma_buf_map() programmed the
    // target card's BAR1 pages (dynamically).

    // --- 4. BAR1 aperture ------------------------------------------------------
    std::string barPath = "/sys/bus/pci/devices/" + dstBDF + "/resource1_wc";
    L.barFd = open(barPath.c_str(), O_RDWR | O_SYNC);
    if (L.barFd < 0) {
        std::fprintf(stderr, "open(%s): %s\n", barPath.c_str(), std::strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(L.barFd, &st) != 0) {
        std::fprintf(stderr, "fstat(%s): %s\n", barPath.c_str(), std::strerror(errno));
        return false;
    }
    uint64_t barSize = (uint64_t)st.st_size;
    std::printf("Target BAR1: %s, size = %llu (0x%llx)\n",
                barPath.c_str(), (unsigned long long)barSize,
                (unsigned long long)barSize);

    uint64_t bar1Start = 0, bar1End = 0;
    bool haveBar1Phys = readBarRange(dstBDF, 1, &bar1Start, &bar1End);
    if (haveBar1Phys)
        std::printf("BAR1 physical: 0x%llx .. 0x%llx\n",
                    (unsigned long long)bar1Start, (unsigned long long)bar1End);

    // --- 5. locate the buffer inside BAR1 --------------------------------------
    // Primary: the sg table. nv_dma_map_peer() receives BAR1 addresses
    // (nv-dmabuf.c/nv-dma.c:763-778) and, without a translating IOMMU,
    // sg_dma_address() IS the BAR1 address. The segments must tile the
    // allocation in order and be gap-free -- then the whole buffer is one
    // contiguous window in BAR1.
    uint64_t barOff = (uint64_t)-1;
    size_t   covered = 0;
    if (haveBar1Phys && !sg.empty()) {
        uint64_t prevEnd = 0;
        bool contiguous = true;
        for (size_t i = 0; i < sg.size(); ++i) {
            uint64_t a = sg[i].dma_address;
            uint64_t l = sg[i].dma_len;
            if (a < bar1Start || a > bar1End || a + l - 1 > bar1End) {
                contiguous = false;
                break;
            }
            if (i == 0) { barOff = a - bar1Start; prevEnd = a + l; }
            else if (a == prevEnd) { prevEnd = a + l; }
            else { contiguous = false; break; }
            covered += (size_t)l;
        }
        if (!contiguous) {
            std::printf("sg table not a single contiguous BAR1 window "
                        "(%zu entries) -- trying the scan fallback.\n", sg.size());
            barOff = (uint64_t)-1;
        } else if (covered < L.allocSize) {
            std::printf("sg segments cover only %zu of %zu bytes -- allocation\n"
                        "does not fully fit into BAR1 (256 MiB aperture).\n",
                        covered, L.allocSize);
            return false;
        } else {
            std::printf("BAR1 offset from sg table: 0x%llx (%llu MiB into the BAR)\n",
                        (unsigned long long)barOff, (unsigned long long)(barOff >> 20));
        }
    }

    if (barOff == (uint64_t)-1) {
        // Fallback: scan BAR1 for the marker (needed when an IOMMU
        // translates and sg addresses are IOVAs).
        uint64_t hit = barScan(L.barFd, barSize, magic, sizeof(magic));
        if (hit == (uint64_t)-1) {
            std::fprintf(stderr,
                "BAR1 scan found no marker. The buffer is not visible in the\n"
                "target card's BAR1 -- the dynamic mapping did not happen.\n"
                "Check: patched driver loaded? BarlinkPeerBar1=1 regkey?\n"
                "dmabuf_holder.ko loaded? CAP_SYS_ADMIN?\n");
            return false;
        }
        barOff = hit;
        covered = L.allocSize;
        if (barOff + L.allocSize > barSize) covered = (size_t)(barSize - barOff);
        std::printf("BAR1 offset from scan: 0x%llx (window %zu bytes)\n",
                    (unsigned long long)barOff, covered);
    }

    // --- 6. mmap the BAR slice on the source card ------------------------------
    const size_t pageSize = (size_t)sysconf(_SC_PAGESIZE);
    uint64_t mapOff = barOff & ~(uint64_t)(pageSize - 1);
    L.delta  = (uint64_t)(barOff - mapOff);
    L.mapLen = (size_t)(((L.delta + L.allocSize + pageSize - 1) / pageSize) * pageSize);
    if (mapOff + L.mapLen > barSize) L.mapLen = (size_t)(barSize - mapOff);

    L.bar = mmap(nullptr, L.mapLen, PROT_READ | PROT_WRITE, MAP_SHARED,
                 L.barFd, (off_t)mapOff);
    if (L.bar == MAP_FAILED) {
        L.bar = nullptr;
        std::fprintf(stderr, "mmap(BAR1, off=0x%llx, len=%zu): %s\n",
                     (unsigned long long)mapOff, L.mapLen, std::strerror(errno));
        return false;
    }
    if (L.delta % 16 != 0 || (uintptr_t)L.bar % 16 != 0) {
        std::fprintf(stderr,
            "BAR1 slice start is not 16-byte aligned (delta=%llu) -- the\n"
            "128-bit kernel cannot be used; this should not happen.\n",
            (unsigned long long)L.delta);
        return false;
    }
    std::printf("BAR slice mapped: off=0x%llx len=%zu delta=%llu\n",
                (unsigned long long)mapOff, L.mapLen, (unsigned long long)L.delta);

    RT(cudaSetDevice(srcOrd));
    cudaError_t regErr = cudaHostRegister(L.bar, L.mapLen, cudaHostRegisterIoMemory);
    if (regErr != cudaSuccess) {
        std::fprintf(stderr,
            "cudaHostRegister(IoMemory) on source card failed: %d (%s)\n"
            "  -> osCheckGpuBarsOverlapAddrRange rejected the range. The\n"
            "     extended guard condition is required for a peer BAR1\n"
            "     range: check the patch (BARLINK_PCIE_MINIMAL.patch) is\n"
            "     loaded and the regkey BarlinkPeerBar1=1 is set.\n",
            (int)regErr, cudaGetErrorString(regErr));
        return false;
    }
    RT(cudaHostGetDevicePointer(&L.srcDevPtr, L.bar, 0));
    std::printf("cudaHostRegister(IoMemory) OK, source device pointer = %p\n",
                L.srcDevPtr);

    RT(cudaStreamCreate(&L.stream));
    return true;
}

static void linkTeardown(Link &L)
{
    if (L.srcDevPtr) {
        cudaSetDevice(L.srcOrd);
        cudaHostUnregister(L.bar);
        L.srcDevPtr = nullptr;
    }
    if (L.stream) { cudaStreamDestroy(L.stream); L.stream = nullptr; }
    if (L.bar)    { munmap(L.bar, L.mapLen); L.bar = nullptr; }
    if (L.holderFd >= 0) {
        holderRelease(L.holderFd, L.holderHandle);
        close(L.holderFd);
        L.holderFd = -1;
    }
    if (L.dmabufFd >= 0) { close(L.dmabufFd); L.dmabufFd = -1; }
    nvExportClose(L.nvx);
    if (L.barFd >= 0) { close(L.barFd); L.barFd = -1; }
    if (L.dstDptr) {
        cudaSetDevice(L.dstOrd);
        cuMemUnmap(L.dstDptr, L.allocSize);
        cuMemAddressFree(L.dstDptr, L.allocSize);
        cuMemRelease(L.memHandle);
        L.dstDptr = 0;
    }
}

// ---------------------------------------------------------------------------
// Phase 1: byte verification
// ---------------------------------------------------------------------------

static bool phaseVerify(Link &L)
{
    const unsigned seed = 1;
    const size_t bytes = L.allocSize;

    RT(cudaSetDevice(L.srcOrd));
    int n4 = (int)(bytes / 16);
    int dev = 0;
    RT(cudaGetDevice(&dev));
    int sms = 0;
    RT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    int blocks = (n4 + 255) / 256;
    int maxBlocks = sms * 8;
    if (blocks > maxBlocks) blocks = maxBlocks;

    k_write<<<blocks, 256, 0, L.stream>>>(
        (uint4 *)((uint8_t *)L.srcDevPtr + L.delta), n4, seed);
    RT(cudaGetLastError());
    RT(cudaStreamSynchronize(L.stream));
    std::printf("Phase 1: pattern (%zu bytes) written from card %d through the "
                "peer BAR1.\n", bytes, L.srcOrd);

    // Read back through the TARGET card's OWN VMM pointer (not the aperture).
    RT(cudaSetDevice(L.dstOrd));
    DRV(cuCtxSynchronize());
    std::vector<uint8_t> got(bytes);
    DRV(cuMemcpyDtoH(got.data(), L.dstDptr, bytes));
    DRV(cuCtxSynchronize());

    size_t bad = 0, firstBad = (size_t)-1;
    for (size_t i = 0; i < bytes; ++i) {
        uint8_t want = (uint8_t)(patVal(i & ~(size_t)15, seed, (int)((i & 15) >> 2))
                                 >> ((i & 3) * 8));
        if (got[i] != want) {
            if (firstBad == (size_t)-1) firstBad = i;
            ++bad;
        }
    }
    std::printf("Phase 1 readback via target card's own VMM pointer: "
                "bad_bytes = %zu of %zu\n", bad, bytes);
    if (bad) {
        std::fprintf(stderr,
            "  first mismatch at byte %zu (want 0x%02x, got 0x%02x)\n"
            "  The BAR1 P2P path is NOT working. Check: patched driver\n"
            "  loaded? BarlinkPeerBar1=1 regkey? dmabuf_holder.ko loaded?\n"
            "  CAP_SYS_ADMIN?\n",
            firstBad, bad ? got[firstBad] : 0, got[firstBad]);
        return false;
    }
    std::printf("  all bytes match -- dynamic BAR1 P2P write path works.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Phase 2: bandwidth sweep
// ---------------------------------------------------------------------------

static int autoIters(size_t bytes)
{
    // aim for ~1 GiB written per measurement
    long n = (long)(1ull << 30) / (long)(bytes ? bytes : 1);
    if (n < 50) n = 50;
    if (n > 4000) n = 4000;
    return (int)n;
}

static bool phaseBandwidth(Link &L, const std::vector<size_t> &sizes, int itersArg)
{
    RT(cudaSetDevice(L.srcOrd));
    int dev = 0;
    RT(cudaGetDevice(&dev));
    int sms = 0;
    RT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

    cudaEvent_t ev0, ev1;
    RT(cudaEventCreate(&ev0));
    RT(cudaEventCreate(&ev1));

    std::printf("\n%-12s %8s %12s %12s\n", "size", "iters", "ms/iter", "GB/s");
    for (size_t bytes : sizes) {
        if (bytes > L.allocSize) continue;
        int n4 = (int)(bytes / 16);
        int blocks = (n4 + 255) / 256;
        int maxBlocks = sms * 8;
        if (blocks > maxBlocks) blocks = maxBlocks;
        int iters = itersArg > 0 ? itersArg : autoIters(bytes);

        uint4 *dst = (uint4 *)((uint8_t *)L.srcDevPtr + L.delta);
        // warm up the window once so the first timed launch is not cold
        k_write<<<blocks, 256, 0, L.stream>>>(dst, n4, 0);
        RT(cudaGetLastError());
        RT(cudaStreamSynchronize(L.stream));

        RT(cudaEventRecord(ev0, L.stream));
        for (int it = 0; it < iters; ++it)
            k_write<<<blocks, 256, 0, L.stream>>>(dst, n4, (unsigned)it + 1);
        RT(cudaGetLastError());
        RT(cudaEventRecord(ev1, L.stream));
        RT(cudaEventSynchronize(ev1));
        float ms = 0.f;
        RT(cudaEventElapsedTime(&ms, ev0, ev1));

        double gbps = (double)bytes * (double)iters / ((double)ms * 1e6);
        std::printf("%-12zu %8d %12.4f %12.2f\n", bytes, iters,
                    (double)ms / iters, gbps);
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static size_t parseSize(const char *s)
{
    char *end = nullptr;
    double v = std::strtod(s, &end);
    if (v <= 0) return 0;
    if (end && *end) {
        switch (*end) {
        case 'k': case 'K': v *= 1024.0; break;
        case 'm': case 'M': v *= 1024.0 * 1024.0; break;
        case 'g': case 'G': v *= 1024.0 * 1024.0 * 1024.0; break;
        default: break;
        }
    }
    return (size_t)v;
}

int main(int argc, char **argv)
{
    size_t size = 64ull << 20;   // default 64 MiB (3080 small BAR: 256 MiB)
    int    iters = 0;            // 0 = auto
    bool   reverse = false, both = false;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if      (!std::strncmp(a, "--size=", 7))   size   = parseSize(a + 7);
        else if (!std::strncmp(a, "--iters=", 8))  iters  = std::atoi(a + 8);
        else if (!std::strcmp (a, "--reverse"))    reverse = true;
        else if (!std::strcmp (a, "--both"))       both    = true;
        else if (!std::strcmp (a, "--help")) {
            std::printf("Usage: %s [--size=N[KMG]] [--iters=N] [--reverse|--both]\n"
                        "  --size=N    buffer size, default 64M (must fit the\n"
                        "              256 MiB BAR1 aperture of a 3080)\n"
                        "  --iters=N   kernel writes per measurement, default\n"
                        "              auto (~1 GiB written per point)\n"
                        "  --reverse   benchmark device 1 -> device 0\n"
                        "  --both      benchmark both directions\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s (see --help)\n", a);
            return 1;
        }
    }
    if (size == 0) { std::fprintf(stderr, "invalid --size\n"); return 1; }
    if (size > 192ull << 20) {
        std::fprintf(stderr,
            "--size=%zu leaves too little room in the 256 MiB BAR1 aperture;\n"
            "stay at or below ~192 MiB.\n", size);
        return 1;
    }

    int ndev = 0;
    RT(cudaGetDeviceCount(&ndev));
    if (ndev < 2) {
        std::fprintf(stderr, "need 2 CUDA devices, found %d\n", ndev);
        return 1;
    }

    const size_t sweepDefault[] = { 4 << 10, 64 << 10, 1 << 20, 4 << 20,
                                    16 << 20, 64 << 20 };
    std::vector<size_t> sweep;
    for (size_t s : sweepDefault)
        if (s <= size) sweep.push_back(s);
    if (sweep.empty() || sweep.back() != size) sweep.push_back(size);

    std::vector<std::pair<int,int>> dirs;
    dirs.emplace_back(0, 1);
    if (both)       dirs.emplace_back(1, 0);
    else if (reverse) { dirs.clear(); dirs.emplace_back(1, 0); }

    bool ok = true;
    for (auto &d : dirs) {
        std::printf("\n==== direction: device %d (writer) -> device %d "
                    "(receiver) ====\n", d.first, d.second);
        Link L;
        if (!linkSetup(L, d.first, d.second, size)) { ok = false; break; }
        if (!phaseVerify(L)) { ok = false; linkTeardown(L); break; }
        if (!phaseBandwidth(L, sweep, iters)) { ok = false; }
        linkTeardown(L);
    }

    if (!ok) {
        std::fprintf(stderr, "\nRESULT: FAILED -- see messages above.\n");
        return 5;
    }
    std::printf("\nRESULT: OK -- BAR1 P2P write path verified and measured.\n");
    return 0;
}
