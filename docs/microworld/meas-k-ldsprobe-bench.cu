// Isolate the two shared loads of gemm_nt_body and count bank conflicts for
// each pattern separately. Same shapes/constants as src/model.cu.
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

constexpr int BK = 16;
constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
constexpr int THREADS = 256;

__device__ __forceinline__ int kshift(bool swiz, int k) { return swiz ? (k >> 2) * 8 : 0; }

// MODE 0: A only (as, ty-indexed).  MODE 1: B only (bs, gemm_col_slot).
// MODE 2: B only, naive tx*TN+j.    MODE 3: B only, TN=4 with 32 lane slots.
template <int MODE, bool SWIZ>
__global__ __launch_bounds__(THREADS, 2)
void probe(float* __restrict__ sink, int iters) {
    constexpr int SPAD = SWIZ ? 32 : 4;
    __shared__ __align__(16) float as[2][BK][BM + SPAD];
    __shared__ __align__(16) float bs[2][BK][BN + SPAD];
    const int tid = threadIdx.x;
    // Fill so nothing is UB; value is irrelevant to the address pattern.
    for (int i = tid; i < 2 * BK * (BM + SPAD); i += THREADS) (&as[0][0][0])[i] = i * 1e-6f;
    for (int i = tid; i < 2 * BK * (BN + SPAD); i += THREADS) (&bs[0][0][0])[i] = i * 1e-6f;
    __syncthreads();

    const int ty = tid / (BN / TN);   // 0..15
    const int tx = tid % (BN / TN);   // 0..15
    const int ty4 = tid / 32;         // MODE 3: 0..7
    const int tx4 = tid % 32;         // MODE 3: 0..31
    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
        const float (*ac)[BM + SPAD] = as[it & 1];
        const float (*bc)[BN + SPAD] = bs[it & 1];
#pragma unroll
        for (int p = 0; p < BK; ++p) {
            const int sh = kshift(SWIZ, p);
            if (MODE == 0) {
#pragma unroll
                for (int i = 0; i < TM; i += 4) {
                    const float4 t = *reinterpret_cast<const float4*>(&ac[p][ty * TM + i + sh]);
                    acc += t.x + t.y + t.z + t.w;
                }
            } else if (MODE == 1) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const int c = tx * 4 + (j >> 2) * (BN / 2) + (j & 3);
                    const float4 t = *reinterpret_cast<const float4*>(&bc[p][c + sh]);
                    acc += t.x + t.y + t.z + t.w;
                }
            } else if (MODE == 2) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const float4 t = *reinterpret_cast<const float4*>(&bc[p][tx * TN + j + sh]);
                    acc += t.x + t.y + t.z + t.w;
                }
            } else {
                // TM=16/TN=4 mapping: 32 distinct column slots per warp.
                const float4 t = *reinterpret_cast<const float4*>(&bc[p][tx4 * 4 + sh]);
                acc += t.x + t.y + t.z + t.w;
#pragma unroll
                for (int i = 0; i < 16; i += 4) {
                    const float4 u = *reinterpret_cast<const float4*>(&ac[p][ty4 * 16 + i + sh]);
                    acc += u.x + u.y + u.z + u.w;
                }
            }
        }
    }
    if (tid == 100000) sink[0] = acc;   // never taken; keeps acc live
}

int main() {
    float* sink; CK(cudaMalloc(&sink, 4));
    const int iters = 256;             // one K=4096 pass
    const dim3 grid(1952);
    printf("mode  desc\n");
    probe<0, true><<<grid, THREADS>>>(sink, iters);  CK(cudaGetLastError());
    probe<1, true><<<grid, THREADS>>>(sink, iters);  CK(cudaGetLastError());
    probe<2, true><<<grid, THREADS>>>(sink, iters);  CK(cudaGetLastError());
    probe<3, true><<<grid, THREADS>>>(sink, iters);  CK(cudaGetLastError());
    probe<1, false><<<grid, THREADS>>>(sink, iters); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(sink));
    printf("done\n");
    return 0;
}
