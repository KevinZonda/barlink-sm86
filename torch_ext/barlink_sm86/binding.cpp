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

static void checkPoolPtr(const at::Tensor &t, int *idxOut, void **ptrOut)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(t.is_cuda(), "barlink_sm86: tensor must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kByte,
                "barlink_sm86: tensors are raw pool memory, dtype must be uint8");
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

at::Tensor empty(int64_t nbytes, int64_t device)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(nbytes > 0 && (nbytes % 16 == 0),
                "barlink_sm86: size must be a positive multiple of 16");
    int idx = devIndexFor(device);
    TORCH_CHECK(idx >= 0, "barlink_sm86: device ", device,
                " not in bl.init()'s device list");

    void *ptr = nullptr;
    char err[BL_ERRBUF] = {0};
    blCheck(bl_pool_alloc(g_ctx, idx, (size_t)nbytes, &ptr, err, sizeof(err)), err);

    // Pool-owned memory: the deleter deliberately does nothing; the pool is
    // reclaimed at bl_shutdown(). (Freeing individual pool tensors is not
    // exposed in v1 -- the allocator would need stream-aware lifetime.)
    return at::from_blob(ptr, {nbytes},
                         at::TensorOptions()
                             .dtype(at::kByte)
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
    TORCH_CHECK(dst.numel() == src.numel(),
                "barlink_sm86 copy_: size mismatch");
    size_t bytes = (size_t)dst.numel();

    // current streams on the two devices; the cross-device event is recorded
    // on the source stream and waited on the destination stream
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
    TORCH_CHECK(a.numel() == b.numel() && a.numel() % 16 == 0,
                "barlink_sm86 allreduce_: equal sizes, multiples of 16");

    cudaStream_t sA = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    cudaStream_t sB = at::cuda::getCurrentCUDAStream(b.get_device()).stream();
    c10::cuda::CUDAGuard guard(a.get_device());
    char err[BL_ERRBUF] = {0};
    blCheck(bl_allreduce_(g_ctx, aPtr, aIdx, bPtr, bIdx, (size_t)a.numel(),
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
    TORCH_CHECK(t.numel() % 16 == 0,
                "barlink_sm86 readback: size must be a multiple of 16");
    auto out = at::empty({t.numel()}, at::TensorOptions().dtype(at::kByte));
    char err[BL_ERRBUF] = {0};
    blCheck(bl_readback(g_ctx, idx, ptr, (size_t)t.numel(), out.data_ptr(),
                        err, sizeof(err)), err);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &init, py::arg("devices"), py::arg("pool_mb") = 64,
          py::call_guard<py::gil_scoped_release>());
    m.def("shutdown", &shutdown);
    m.def("empty", &empty, py::arg("nbytes"), py::arg("device"));
    m.def("copy_", &copy_, py::call_guard<py::gil_scoped_release>());
    m.def("allreduce_", &allreduce_, py::call_guard<py::gil_scoped_release>());
    m.def("verify", &verify, py::call_guard<py::gil_scoped_release>());
    m.def("readback", &readback, py::call_guard<py::gil_scoped_release>());
}
