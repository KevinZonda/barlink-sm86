// stock-d2d.cu — baseline: cross-device copy bandwidth on stock (unpatched) driver.
// Measures what the system can actually do today without the BAR1 path:
//   (A) cudaMemcpyAsync D2D between two devices (driver stages through host when peer access is locked)
//   (B) explicit staging through pinned host memory (DtoH + HtoD)
// Prints GB/s for both directions. This is the number the BAR1 path has to beat.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

static double bw(size_t bytes, float ms) { return ms > 0 ? (double)bytes / (ms * 1e-3) / 1e9 : 0.0; }

int main() {
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "need 2 GPUs, have %d\n", ndev); return 2; }
    cudaDeviceProp p0, p1; CK(cudaGetDeviceProperties(&p0, 0)); CK(cudaGetDeviceProperties(&p1, 1));
    printf("devices: %s <-> %s\n", p0.name, p1.name);
    int can01 = 0, can10 = 0;
    CK(cudaSetDevice(0)); CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaSetDevice(1)); CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("cudaDeviceCanAccessPeer 0->1 = %d, 1->0 = %d (0 = locked, expect staging fallback)\n", can01, can10);

    const size_t sizes[] = {4<<10, 64<<10, 1<<20, 4<<20, 16<<20, 64<<20};
    const int iters = 200;
    void *h_stage; CK(cudaMallocHost(&h_stage, 64<<20));
    cudaStream_t s0, s1; CK(cudaStreamCreate(&s0)); CK(cudaStreamCreate(&s1));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    for (size_t bytes : sizes) {
        void *d0, *d1;
        CK(cudaSetDevice(0)); CK(cudaMalloc(&d0, bytes)); CK(cudaMemset(d0, 0x5a, bytes));
        CK(cudaSetDevice(1)); CK(cudaMalloc(&d1, bytes));

        for (int dir = 0; dir < 2; dir++) {  // 0: 0->1, 1: 1->0
            void *src = dir ? d1 : d0, *dst = dir ? d0 : d1;
            cudaStream_t ss = dir ? s1 : s0;

            // (A) cross-device cudaMemcpyAsync
            CK(cudaEventRecord(e0, ss));
            for (int i = 0; i < iters; i++) CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ss));
            CK(cudaEventRecord(e1, ss)); CK(cudaEventSynchronize(e1));
            float msA = 0; CK(cudaEventElapsedTime(&msA, e0, e1));

            // (B) explicit pinned staging: DtoH then HtoD
            CK(cudaEventRecord(e0, ss));
            for (int i = 0; i < iters; i++) {
                CK(cudaMemcpyAsync(h_stage, src, bytes, cudaMemcpyDeviceToHost, ss));
                CK(cudaSetDevice(dir ? 0 : 1));
                CK(cudaMemcpyAsync(dst, h_stage, bytes, cudaMemcpyHostToDevice, ss));
                CK(cudaSetDevice(dir ? 1 : 0));
            }
            CK(cudaEventRecord(e1, ss)); CK(cudaEventSynchronize(e1));
            float msB = 0; CK(cudaEventElapsedTime(&msB, e0, e1));

            printf("%8zu B  dir %d->%d  (A) memcpyD2D %8.2f GB/s   (B) pinned staging %8.2f GB/s\n",
                   bytes, dir ? 1 : 0, dir ? 0 : 1,
                   bw(bytes * iters, msA), bw(bytes * iters, msB));
        }
        CK(cudaSetDevice(0)); CK(cudaFree(d0));
        CK(cudaSetDevice(1)); CK(cudaFree(d1));
    }
    return 0;
}
