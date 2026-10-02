// SPDX-License-Identifier: MIT
//
// barlink_sm86 torch/pybind binding -- thin layer over core.h.
// All mechanism lives in core.cu (no torch dependency).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <linux/capability.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.h"

static blCtx *g_ctx = nullptr;
static std::vector<int> g_devices;
static int64_t g_pool_mb = 0;

#define BL_ERRBUF 1024

// CAP_SYS_ADMIN (via tools/blrun) is needed only for the
// cudaHostRegister(IoMemory) calls inside bl_init(). Drop every capability
// the moment init succeeds; with blrun's no_new_privs they can never be
// regained. Harmless when the process has no caps to begin with.
static void dropCapsAfterInit()
{
    prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0);
    struct __user_cap_header_struct hdr = {
        _LINUX_CAPABILITY_VERSION_3, 0,
    };
    struct __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3] = {};
    syscall(SYS_capset, &hdr, data);
}

static void blCheck(int rc, char *errbuf)
{
    if (rc != 0) {
        std::string msg = errbuf[0] ? errbuf : "unknown barlink_sm86 error";
        throw std::runtime_error("barlink_sm86: " + msg);
    }
}

static int devIndexFor(int64_t device)
{
    for (size_t i = 0; i < g_devices.size(); ++i)
        if (g_devices[i] == (int)device) return (int)i;
    return -1;
}

// at::ScalarType -> core.h dtype enum; -1 if unsupported here.
static int dtypeEnumFor(at::ScalarType st)
{
    switch (st) {
    case at::kByte:          return BL_DTYPE_U8;
    case at::kFloat:         return BL_DTYPE_FP32;
    case at::kDouble:        return BL_DTYPE_FP64;
    case at::kBFloat16:      return BL_DTYPE_BF16;
    case at::kFloat8_e4m3fn: return BL_DTYPE_FP8E4M3;
    case at::kFloat8_e5m2:   return BL_DTYPE_FP8E5M2;
    default:                 return -1;
    }
}

static void checkPoolPtr(const at::Tensor &t, int *idxOut, void **ptrOut)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(t.is_cuda(), "barlink_sm86: tensor must be a CUDA tensor");
    int idx = devIndexFor(t.get_device());
    TORCH_CHECK(idx >= 0, "barlink_sm86: tensor is on cuda:", t.get_device(),
                " which is not in bl.init()'s device list");
    // pool tensors are contiguous by construction (from_blob over a flat
    // range); a strided view of one cannot be copied
    TORCH_CHECK(t.is_contiguous(), "barlink_sm86: tensor must be contiguous");
    *idxOut = idx;
    *ptrOut = t.data_ptr();
}

void init(std::vector<int64_t> devices, int64_t pool_mb)
{
    // idempotent: under tools/blrun the pool is already up and the process
    // can no longer re-register (caps dropped), so a matching re-init is a
    // no-op instead of a capability error
    if (g_ctx && devices.size() == g_devices.size() && pool_mb == g_pool_mb) {
        bool same = true;
        for (size_t i = 0; i < devices.size(); ++i)
            same = same && devices[i] == g_devices[i];
        if (same) return;
    }
    TORCH_CHECK(!g_ctx,
                "barlink_sm86: already initialized with a different config; ",
                "restart the process (re-init needs CAP_SYS_ADMIN)");

    TORCH_CHECK(devices.size() == 2, "barlink_sm86 v1: exactly 2 devices");
    TORCH_CHECK(pool_mb >= 4, "barlink_sm86: pool_mb must be >= 4");
    TORCH_CHECK(pool_mb <= 192,
                "barlink_sm86: pool must fit the 256 MiB BAR1 aperture "
                "(keep pool_mb <= 192)");

    int devs[2] = { (int)devices[0], (int)devices[1] };
    char err[BL_ERRBUF] = {0};
    blCtx *ctx = nullptr;
    blCheck(bl_init(&ctx, devs, 2, (size_t)pool_mb << 20, err, sizeof(err)), err);
    g_ctx = ctx;
    g_devices.assign(devs, devs + 2);
    g_pool_mb = pool_mb;
    dropCapsAfterInit();
}

void shutdown()
{
    if (g_ctx) bl_shutdown(g_ctx);
    g_ctx = nullptr;
    g_pool_mb = 0;
    g_devices.clear();
}

at::Tensor empty(int64_t nbytes, int64_t device,
                 std::optional<at::ScalarType> dtype)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(nbytes > 0 && (nbytes % 16 == 0),
                "barlink_sm86: size must be a positive multiple of 16");
    int idx = devIndexFor(device);
    TORCH_CHECK(idx >= 0, "barlink_sm86: device ", device,
                " not in bl.init()'s device list");

    at::ScalarType st = dtype.value_or(at::kByte);
    int dt = dtypeEnumFor(st);
    TORCH_CHECK(dt >= 0,
                "barlink_sm86: unsupported dtype for a pool tensor: ", st,
                " (supported: uint8, float32, float64, bfloat16, "
                "float8_e4m3fn, float8_e5m2)");
    TORCH_CHECK(nbytes % (int64_t)c10::elementSize(st) == 0,
                "barlink_sm86: nbytes must be a multiple of the item size");

    void *ptr = nullptr;
    char err[BL_ERRBUF] = {0};
    blCheck(bl_pool_alloc(g_ctx, idx, (size_t)nbytes, &ptr, err, sizeof(err)), err);

    // Pool-owned memory: the deleter deliberately does nothing; the pool is
    // reclaimed at bl_shutdown(). (Freeing individual pool tensors is not
    // exposed in v1 -- the allocator would need stream-aware lifetime.)
    return at::from_blob(ptr, {nbytes / (int64_t)c10::elementSize(st)},
                         at::TensorOptions()
                             .dtype(st)
                             .device(at::kCUDA, (c10::DeviceIndex)device));
}

void copy_(at::Tensor dst, at::Tensor src)
{
    int dstIdx = 0, srcIdx = 0;
    void *dstPtr = nullptr, *srcPtr = nullptr;
    checkPoolPtr(dst, &dstIdx, &dstPtr);
    checkPoolPtr(src, &srcIdx, &srcPtr);
    TORCH_CHECK(dstIdx != srcIdx, "barlink_sm86 copy_: src and dst must live "
                                  "on different pool devices");
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(),
                "barlink_sm86 copy_: dtype mismatch (",
                dst.scalar_type(), " vs ", src.scalar_type(), ")");
    TORCH_CHECK(dst.numel() == src.numel(),
                "barlink_sm86 copy_: size mismatch");
    size_t bytes = (size_t)dst.numel() * dst.element_size();
    TORCH_CHECK(bytes % 16 == 0,
                "barlink_sm86 copy_: byte size must be a multiple of 16");

    // current streams on the two devices; the marker-flag wait is queued
    // on the destination stream by bl_copy_
    cudaStream_t srcStream = at::cuda::getCurrentCUDAStream(
        src.get_device()).stream();
    cudaStream_t dstStream = at::cuda::getCurrentCUDAStream(
        dst.get_device()).stream();

    // core switches the CUDA context internally; guard for our own calls
    c10::cuda::CUDAGuard guard(src.get_device());
    char err[BL_ERRBUF] = {0};
    blCheck(bl_copy_(g_ctx, dstPtr, dstIdx, srcPtr, srcIdx, bytes,
                     (void *)srcStream, (void *)dstStream, err, sizeof(err)), err);
}

void allreduce_(at::Tensor a, at::Tensor b)
{
    int aIdx = 0, bIdx = 0;
    void *aPtr = nullptr, *bPtr = nullptr;
    checkPoolPtr(a, &aIdx, &aPtr);
    checkPoolPtr(b, &bIdx, &bPtr);
    TORCH_CHECK(aIdx != bIdx, "barlink_sm86 allreduce_: tensors must live on "
                              "different pool devices");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "barlink_sm86 allreduce_: dtype mismatch");
    int dt = dtypeEnumFor(a.scalar_type());
    TORCH_CHECK(dt >= 0,
                "barlink_sm86 allreduce_: unsupported dtype ", a.scalar_type(),
                " (supported: uint8, float32, float64, bfloat16, "
                "float8_e4m3fn, float8_e5m2)");
    TORCH_CHECK(a.numel() == b.numel(),
                "barlink_sm86 allreduce_: size mismatch");
    size_t bytes = (size_t)a.numel() * a.element_size();
    TORCH_CHECK(bytes % 16 == 0,
                "barlink_sm86 allreduce_: byte size must be a multiple of 16");

    cudaStream_t sA = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    cudaStream_t sB = at::cuda::getCurrentCUDAStream(b.get_device()).stream();
    c10::cuda::CUDAGuard guard(a.get_device());
    char err[BL_ERRBUF] = {0};
    blCheck(bl_allreduce_(g_ctx, aPtr, aIdx, bPtr, bIdx, bytes, dt,
                          (void *)sA, (void *)sB, err, sizeof(err)), err);
}

int64_t verify()
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    char err[BL_ERRBUF] = {0};
    uint64_t bad = bl_verify(g_ctx, err, sizeof(err));
    if (bad == ~0ull)
        throw std::runtime_error(std::string("barlink_sm86 verify failed: ") +
                                 (err[0] ? err : "unknown"));
    return (int64_t)bad;
}

at::Tensor readback(at::Tensor t)
{
    int idx = 0;
    void *ptr = nullptr;
    checkPoolPtr(t, &idx, &ptr);
    size_t bytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(bytes % 16 == 0,
                "barlink_sm86 readback: byte size must be a multiple of 16");
    auto out = at::empty({(int64_t)bytes},
                         at::TensorOptions().dtype(at::kByte));
    char err[BL_ERRBUF] = {0};
    blCheck(bl_readback(g_ctx, idx, ptr, bytes, out.data_ptr(),
                        err, sizeof(err)), err);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &init, py::arg("devices"), py::arg("pool_mb") = 64,
          py::call_guard<py::gil_scoped_release>());
    m.def("shutdown", &shutdown);
    m.def("empty", &empty, py::arg("nbytes"), py::arg("device"),
          py::arg("dtype") = py::none());
    m.def("copy_", &copy_, py::call_guard<py::gil_scoped_release>());
    m.def("allreduce_", &allreduce_, py::call_guard<py::gil_scoped_release>());
    m.def("verify", &verify, py::call_guard<py::gil_scoped_release>());
    m.def("readback", &readback, py::call_guard<py::gil_scoped_release>());
}
