// SPDX-License-Identifier: MIT
//
// W8A16 weight-only int8 GEMM for sm86 (RTX 3080 class), tuned for tiny-M
// decode (M <= 16). Replaces the dynamic-quant wrapper (per-row weight
// requant + per-token act quant + torch._int_mm, ~10 kernels per Linear)
// with 2-3 kernels per Linear.
//
//   y[m, n] = xs[m] * sum_g sw[g, n] * sum_{k in g} xq[m, k] * w[n, k]
//
// Weights stay EXACTLY as checkpointed (int8, group-128 scales -- no
// per-row refold, so fidelity is strictly better than the old path).
//
// Layouts (prepared once at load, in python):
//   wq_t: int8 [K/16, N, 16]  -- chunk-major transpose so a warp reading
//         consecutive n at a fixed k-chunk is fully coalesced (512 B/warp)
//   sw_t: fp32 [G, N]         -- group-major transpose (coalesced scale
//         reads; bf16 -> fp32 at prep is lossless)
//   xq:   int8 [M, K]         -- activations quantized per token
//   xs:   fp32 [M]            -- per-token act scale (amax/127)
//
// GEMM: 256 threads/block, one output row n per thread, K split into S
// contiguous GROUP ranges (grid.y). S == 1 writes bf16 directly; S > 1
// atomicAdds into a zeroed fp32 workspace followed by a cast kernel.
// MT in {1, 4, 16}: decode M == 1 takes the MT == 1 hot path.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define W8A16_THREADS 256

// ---------------------------------------------------------------------------
// activation quantization: x[M, K] bf16 -> xq[M, K] int8 + xs[M] fp32
// grid (M), block (256); vectorized 8 x bf16 (16 B) loads, K multiple of 8
// assumed for the vector path (all model shapes are multiples of 128).
// ---------------------------------------------------------------------------
__global__ void k_quant_act_bf16(const __nv_bfloat16 *__restrict__ x,
                                 int8_t *__restrict__ xq,
                                 float *__restrict__ xs,
                                 int K)
{
    const __nv_bfloat16 *xr = x + (size_t)blockIdx.x * K;
    int8_t *xqr = xq + (size_t)blockIdx.x * K;

    float amax = 0.f;
    const int nv = K >> 3;              // 16-byte vectors
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        uint4 u = reinterpret_cast<const uint4 *>(xr)[i];
        const __nv_bfloat16 *h = reinterpret_cast<const __nv_bfloat16 *>(&u);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = fabsf(__bfloat162float(h[j]));
            amax = fmaxf(amax, v);
        }
    }
    __shared__ float red[W8A16_THREADS / 32];
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(~0u, amax, off));
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < W8A16_THREADS / 32) ? red[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_down_sync(~0u, v, off));
        if (threadIdx.x == 0) {
            float s = v / 127.0f;
            if (s < 1e-12f) s = 1e-12f;
            red[0] = s;
            xs[blockIdx.x] = s;
        }
    }
    __syncthreads();
    const float s = red[0];
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        uint4 u = reinterpret_cast<const uint4 *>(xr)[i];
        const __nv_bfloat16 *h = reinterpret_cast<const __nv_bfloat16 *>(&u);
        int8_t o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            o[j] = (int8_t)__float2int_rz(
                rintf(__bfloat162float(h[j]) / s));
        reinterpret_cast<uint2 *>(xqr)[i] =
            *reinterpret_cast<const uint2 *>(o);
    }
}

// ---------------------------------------------------------------------------
// GEMM kernel. One output row n per thread.
//   wq_t: [K/16, N, 16] int8 (chunk-major transpose)
//   sw_t: [G, N] fp32, G = K / 128
//   grid: (ceil(N / 256), S); each y-slice covers groups [g0, g1)
// ---------------------------------------------------------------------------
template <int MT, bool ATOMIC>
__global__ void k_w8a16_gemm(const int8_t *__restrict__ wq_t,
                             const float *__restrict__ sw_t,
                             const int8_t *__restrict__ xq,
                             const float *__restrict__ xs,
                             __nv_bfloat16 *__restrict__ out,
                             float *__restrict__ y32,
                             int M, int N, int K, int gper)
{
    const int n = blockIdx.x * W8A16_THREADS + threadIdx.x;
    if (n >= N) return;
    const int G = K >> 7;
    const int g0 = blockIdx.y * gper;
    const int g1 = min(G, g0 + gper);

    float accf[MT];
    int accg[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) { accf[m] = 0.f; accg[m] = 0; }

    float xsc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m)
        xsc[m] = (m < M) ? xs[m] : 0.f;

    for (int g = g0; g < g1; ++g) {
        const float swn = sw_t[(size_t)g * N + n];
        const int kbase = g << 7;
        const int8_t *wp = wq_t + ((size_t)(kbase >> 4) * N + n) * 16;
#pragma unroll 4
        for (int c = 0; c < 8; ++c) {        // 8 chunks of 16 per group
            uint4 w4 = *reinterpret_cast<const uint4 *>(wp);
            wp += (size_t)N * 16;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                uint4 x4 = *reinterpret_cast<const uint4 *>(
                    xq + (size_t)m * K + kbase + c * 16);
                int t = __dp4a((int)w4.x, (int)x4.x, accg[m]);
                t = __dp4a((int)w4.y, (int)x4.y, t);
                t = __dp4a((int)w4.z, (int)x4.z, t);
                accg[m] = __dp4a((int)w4.w, (int)x4.w, t);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            accf[m] += xsc[m] * swn * (float)accg[m];
            accg[m] = 0;
        }
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= M) break;
        if (ATOMIC) {
            atomicAdd(y32 + (size_t)m * N + n, accf[m]);
        } else {
            out[(size_t)m * N + n] = __float2bfloat16(accf[m]);
        }
    }
}

__global__ void k_cast_fp32_bf16(const float *__restrict__ y32,
                                 __nv_bfloat16 *__restrict__ out, int MN)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MN)
        out[i] = __float2bfloat16(y32[i]);
}

extern "C" {

int bl_w8a16_quant_act(const void *x, void *xq, void *xs, int M, int K,
                       cudaStream_t stream)
{
    k_quant_act_bf16<<<M, W8A16_THREADS, 0, stream>>>(
        (const __nv_bfloat16 *)x, (int8_t *)xq, (float *)xs, K);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// S: split of the group dimension; y32 must be a zeroed fp32 [M, N] buffer
// when S > 1 (out receives the cast result). S == 1 writes out directly.
int bl_w8a16_gemm(const void *wq_t, const void *sw_t, const void *xq,
                  const void *xs, void *out, void *y32, int M, int N, int K,
                  int S, cudaStream_t stream)
{
    const int G = K >> 7;
    if (S < 1) S = 1;
    if (S > G) S = G;
    const int gper = (G + S - 1) / S;
    const dim3 grid((N + W8A16_THREADS - 1) / W8A16_THREADS, S);
    const int8_t *w = (const int8_t *)wq_t;
    const float *sw = (const float *)sw_t;
    const int8_t *xq8 = (const int8_t *)xq;
    const float *xsf = (const float *)xs;
    __nv_bfloat16 *o = (__nv_bfloat16 *)out;
    float *y = (float *)y32;
#define LAUNCH(MT)                                                          \
    do {                                                                    \
        if (S > 1)                                                          \
            k_w8a16_gemm<MT, true><<<grid, W8A16_THREADS, 0, stream>>>(     \
                w, sw, xq8, xsf, o, y, M, N, K, gper);                      \
        else                                                                \
            k_w8a16_gemm<MT, false><<<grid, W8A16_THREADS, 0, stream>>>(    \
                w, sw, xq8, xsf, o, y, M, N, K, gper);                      \
    } while (0)
    if (M <= 1) LAUNCH(1);
    else if (M <= 4) LAUNCH(4);
    else if (M <= 16) LAUNCH(16);
    else return -2;
#undef LAUNCH
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (S > 1) {
        const int MN = M * N;
        k_cast_fp32_bf16<<<(MN + 255) / 256, 256, 0, stream>>>(
            y, o, MN);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

} // extern "C"
