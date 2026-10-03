// SPDX-License-Identifier: MIT
//
// barlink_sm86 torch/pybind binding -- thin layer over core.h.
// All mechanism lives in core.cu (no torch dependency).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.h"

static blCtx *g_ctx = nullptr;
static std::vector<int> g_devices;
static int64_t g_pool_mb = 0;
static bool  g_peer = false;      // peer (cross-process SPMD) mode
static int   g_myRank = -1;

#define BL_ERRBUF 1024

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

// Zero-copy path dtypes: everything EXCEPT u8 (keeps the mod-256 pool
// semantics; PG SUM does not expose it). fp16 is native here -- no pool
// add-dtype requirement -- which retires the fp32-staging cast workaround.
static int dtypeEnumZeroCopy(at::ScalarType st)
{
    switch (st) {
    case at::kHalf:          return BL_DTYPE_FP16;
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
    g_peer = false;
    g_myRank = -1;
    // caps were only needed for the cudaHostRegister calls inside bl_init
    bl_drop_caps();
}

// Cross-process SPMD mode: this process owns exactly ONE GPU (rank = device
// index). Both ranks must call identical bl_* sequences with identical
// sizes; the only cross-process traffic is the rendezvous inside
// bl_init_peer. Caps are dropped on success, as in init().
void init_peer(int64_t device, int64_t pool_mb, std::string sock_path,
               int64_t rank)
{
    TORCH_CHECK(!g_ctx,
                "barlink_sm86: already initialized; restart the process");
    TORCH_CHECK(rank == 0 || rank == 1,
                "barlink_sm86 init_peer: rank must be 0 or 1");
    TORCH_CHECK(pool_mb >= 8,
                "barlink_sm86: peer mode reserves half the pool as "
                "cross-process scratch (pool_mb >= 8)");
    TORCH_CHECK(pool_mb <= 192,
                "barlink_sm86: pool must fit the 256 MiB BAR1 aperture "
                "(keep pool_mb <= 192)");

    char err[BL_ERRBUF] = {0};
    blCtx *ctx = nullptr;
    blCheck(bl_init_peer(&ctx, (int)device, (size_t)pool_mb << 20,
                         sock_path.c_str(), (int)rank, err, sizeof(err)), err);
    g_ctx = ctx;
    g_peer = true;
    g_myRank = (int)rank;
    // map MY device to MY rank so devIndexFor feeds the core the indices it
    // indexes pools/flags with (a peer-mode process only ever sees its own
    // device)
    g_devices.assign(2, -1);
    g_devices[g_myRank] = (int)device;
    g_pool_mb = pool_mb;
}

void shutdown()
{
    if (g_ctx) bl_shutdown(g_ctx);
    g_ctx = nullptr;
    g_pool_mb = 0;
    g_peer = false;
    g_myRank = -1;
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
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(),
                "barlink_sm86 copy_: dtype mismatch (",
                dst.scalar_type(), " vs ", src.scalar_type(), ")");
    TORCH_CHECK(dst.numel() == src.numel(),
                "barlink_sm86 copy_: size mismatch");
    size_t bytes = (size_t)dst.numel() * dst.element_size();
    TORCH_CHECK(bytes % 16 == 0,
                "barlink_sm86 copy_: byte size must be a multiple of 16");

    if (g_peer) {
        // SPMD exchange: I write my src into the PEER pool at dst's offset;
        // the peer writes into my pool concurrently. The flag wait lands on
        // my stream and orders it after the peer's payload.
        cudaStream_t s = at::cuda::getCurrentCUDAStream(
            src.get_device()).stream();
        c10::cuda::CUDAGuard guard(src.get_device());
        char err[BL_ERRBUF] = {0};
        blCheck(bl_copy_(g_ctx, dstPtr, 1 - g_myRank, srcPtr, g_myRank,
                         bytes, (void *)s, (void *)s, err, sizeof(err)), err);
        return;
    }

    TORCH_CHECK(dstIdx != srcIdx, "barlink_sm86 copy_: src and dst must live "
                                  "on different pool devices");

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

    if (g_peer) {
        // SPMD cross-rank reduction: a and b are MY LOCAL tensors; after
        // both ranks call, each holds my_value + peer_value.
        cudaStream_t s = at::cuda::getCurrentCUDAStream(
            a.get_device()).stream();
        c10::cuda::CUDAGuard guard(a.get_device());
        char err[BL_ERRBUF] = {0};
        blCheck(bl_allreduce_peer(g_ctx, aPtr, bPtr, bytes, dt,
                                  (void *)s, err, sizeof(err)), err);
        return;
    }

    TORCH_CHECK(aIdx != bIdx, "barlink_sm86 allreduce_: tensors must live on "
                              "different pool devices");

    cudaStream_t sA = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
    cudaStream_t sB = at::cuda::getCurrentCUDAStream(b.get_device()).stream();
    c10::cuda::CUDAGuard guard(a.get_device());
    char err[BL_ERRBUF] = {0};
    blCheck(bl_allreduce_(g_ctx, aPtr, aIdx, bPtr, bIdx, bytes, dt,
                          (void *)sA, (void *)sB, err, sizeof(err)), err);
}

void allreduce_into(at::Tensor out, at::Tensor in)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(g_peer,
                "barlink_sm86 allreduce_into: peer (cross-process) mode only");
    TORCH_CHECK(in.is_cuda() && out.is_cuda(),
                "barlink_sm86 allreduce_into: tensors must be CUDA tensors");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(),
                "barlink_sm86 allreduce_into: tensors must be contiguous");
    TORCH_CHECK(in.scalar_type() == out.scalar_type(),
                "barlink_sm86 allreduce_into: dtype mismatch");
    int dt = dtypeEnumZeroCopy(in.scalar_type());
    TORCH_CHECK(dt >= 0,
                "barlink_sm86 allreduce_into: unsupported dtype ",
                in.scalar_type(),
                " (supported: float16, bfloat16, float32, float64, "
                "float8_e4m3fn, float8_e5m2; u8 keeps the pool path)");
    TORCH_CHECK(in.numel() == out.numel(),
                "barlink_sm86 allreduce_into: size mismatch");
    size_t bytes = (size_t)in.numel() * in.element_size();
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0,
                "barlink_sm86 allreduce_into: byte size must be a non-zero "
                "multiple of 16");
    TORCH_CHECK(in.get_device() == g_devices[g_myRank],
                "barlink_sm86 allreduce_into: tensor must live on this "
                "rank's device (cuda:", g_devices[g_myRank], ")");
    uintptr_t ip = (uintptr_t)in.data_ptr(), op = (uintptr_t)out.data_ptr();
    TORCH_CHECK(ip % 16 == 0 && op % 16 == 0,
                "barlink_sm86 allreduce_into: data pointers must be 16-byte "
                "aligned (fresh torch allocations are)");
    if (ip != op) {
        TORCH_CHECK(ip + bytes <= op || op + bytes <= ip,
                    "barlink_sm86 allreduce_into: overlapping in/out ranges "
                    "are only supported for in-place (in == out)");
    }

    cudaStream_t s = at::cuda::getCurrentCUDAStream(
        in.get_device()).stream();
    c10::cuda::CUDAGuard guard(in.get_device());
    char err[BL_ERRBUF] = {0};
    blCheck(bl_allreduce_into_peer(g_ctx, (void *)ip, (void *)op, bytes, dt,
                                   (void *)s, err, sizeof(err)), err);
}

// Zero-copy peer p2p (see core.h bl_send_into_peer / bl_recv_into_peer):
// stream MY tensor into the peer's scratch zone / stream the peer's tensor
// into MINE. Pure byte move -- every dtype is supported (u8 has no mod-256
// semantics here); the size must be a non-zero multiple of 16 and the
// pointer 16-aligned. Non-blocking, stream-ordered; the caller's sync.
static void p2pCheck(const at::Tensor &t, size_t *bytesOut)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(g_peer,
                "barlink_sm86 p2p: peer (cross-process) mode only");
    TORCH_CHECK(t.is_cuda(),
                "barlink_sm86 p2p: tensor must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(),
                "barlink_sm86 p2p: tensor must be contiguous");
    TORCH_CHECK(t.get_device() == g_devices[g_myRank],
                "barlink_sm86 p2p: tensor must live on this rank's device "
                "(cuda:", g_devices[g_myRank], ")");
    size_t bytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0,
                "barlink_sm86 p2p: byte size must be a non-zero multiple "
                "of 16");
    TORCH_CHECK((uintptr_t)t.data_ptr() % 16 == 0,
                "barlink_sm86 p2p: data pointer must be 16-byte aligned "
                "(fresh torch allocations are)");
    *bytesOut = bytes;
}

void send_into(at::Tensor t, int64_t peer_rank)
{
    size_t bytes = 0;
    p2pCheck(t, &bytes);
    cudaStream_t s = at::cuda::getCurrentCUDAStream(
        t.get_device()).stream();
    c10::cuda::CUDAGuard guard(t.get_device());
    char err[BL_ERRBUF] = {0};
    // dtype is a pure byte move; the enum is validated in core, pass U8
    blCheck(bl_send_into_peer(g_ctx, t.data_ptr(), bytes, BL_DTYPE_U8,
                              (int)peer_rank, (void *)s, err, sizeof(err)),
            err);
}

void recv_into(at::Tensor t, int64_t peer_rank)
{
    size_t bytes = 0;
    p2pCheck(t, &bytes);
    cudaStream_t s = at::cuda::getCurrentCUDAStream(
        t.get_device()).stream();
    c10::cuda::CUDAGuard guard(t.get_device());
    char err[BL_ERRBUF] = {0};
    blCheck(bl_recv_into_peer(g_ctx, t.data_ptr(), bytes, BL_DTYPE_U8,
                              (int)peer_rank, (void *)s, err, sizeof(err)),
            err);
}

at::Tensor bar_atomic_probe(int64_t iters)
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    TORCH_CHECK(g_peer, "barlink_sm86 bar_atomic_probe: peer mode only");
    static unsigned long long res[10];
    char err[BL_ERRBUF] = {0};
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    c10::cuda::CUDAGuard guard(g_devices[g_myRank]);
    blCheck(bl_probe_bar_atomic(g_ctx, res, (int)iters, (void *)s,
                                err, sizeof(err)), err);
    auto out = at::empty({10}, at::TensorOptions().dtype(at::kLong));
    for (int i = 0; i < 10; ++i)
        out[i] = (int64_t)res[i];
    return out;
}

// host read of the local pool's flag-tail slots (+0/+8 of slots 0..7):
// watch the handshake while a wait spins (debug only)
at::Tensor debug_flags()
{
    TORCH_CHECK(g_ctx, "barlink_sm86: not initialized -- call bl.init() first");
    unsigned long long vals[16] = {0};
    char err[BL_ERRBUF] = {0};
    blCheck(bl_debug_flags(g_ctx, vals, err, sizeof(err)), err);
    auto out = at::empty({16}, at::TensorOptions().dtype(at::kLong));
    for (int i = 0; i < 16; ++i)
        out[i] = (int64_t)vals[i];
    return out;
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
    m.def("init_peer", &init_peer,
          py::arg("device"), py::arg("pool_mb") = 64,
          py::arg("sock_path"), py::arg("rank"),
          py::call_guard<py::gil_scoped_release>());
    m.def("shutdown", &shutdown);
    m.def("empty", &empty, py::arg("nbytes"), py::arg("device"),
          py::arg("dtype") = py::none());
    m.def("copy_", &copy_, py::call_guard<py::gil_scoped_release>());
    m.def("allreduce_", &allreduce_, py::call_guard<py::gil_scoped_release>());
    m.def("allreduce_into", &allreduce_into, py::arg("out"), py::arg("in"),
          py::call_guard<py::gil_scoped_release>());
    m.def("send_into", &send_into, py::arg("t"), py::arg("peer_rank"),
          py::call_guard<py::gil_scoped_release>());
    m.def("recv_into", &recv_into, py::arg("t"), py::arg("peer_rank"),
          py::call_guard<py::gil_scoped_release>());
    m.def("verify", &verify, py::call_guard<py::gil_scoped_release>());
    m.def("readback", &readback, py::call_guard<py::gil_scoped_release>());
    m.def("bar_atomic_probe", &bar_atomic_probe, py::arg("iters"),
          py::call_guard<py::gil_scoped_release>());
    m.def("debug_flags", &debug_flags, py::call_guard<py::gil_scoped_release>());
}
