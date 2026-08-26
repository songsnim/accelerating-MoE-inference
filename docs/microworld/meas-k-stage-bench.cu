// Price the k-loop staging step three ways, at the exact per-thread volume the
// projection GEMM moves: 2 float4 per matrix per k-tile, 256 k-tiles.
//
//  ldg_sts   what the shipping kernel does: LDG.128 -> registers -> 4 scalar
//            STS each, because shared is k-major and global is m-major.
//  cpasync   Ampere cp.async: global -> shared directly, no register round
//            trip and no STS. Requires the shared tile to keep the global
//            (m-major) layout, which the k-inner register tile allows.
//  ldg_sts128 lower bound for the store path if no transpose were needed:
//            LDG.128 -> one STS.128.
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); return 1;} } while(0)

constexpr int BK = 16, BM = 128, THREADS = 256, KTILES = 256;

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// MODE 0 = ldg_sts (transposing, 4 scalar STS)   1 = cpasync   2 = ldg_sts128
template <int MODE>
__global__ __launch_bounds__(THREADS, 2)
void stage(const float* __restrict__ g, float* __restrict__ sink) {
    constexpr int SPAD = 32;
    __shared__ __align__(16) float as[2][BK][BM + SPAD];
    __shared__ __align__(16) float bs[2][BK][BM + SPAD];
    const int tid = threadIdx.x;
    const int lr = tid / (BK / 4);
    const int lc = (tid % (BK / 4)) * 4;
    const int lr2 = lr + BM / 2;
    const int sc = lr, sc2 = lr + BM / 2;
    const float* ab  = g + (size_t)(blockIdx.x * BM + lr)  * (KTILES * BK) + lc;
    const float* ab2 = g + (size_t)(blockIdx.x * BM + lr2) * (KTILES * BK) + lc;
    float acc = 0.0f;

    for (int kt = 0; kt < KTILES * BK; kt += BK) {
        const int buf = (kt / BK) & 1;
        if (MODE == 0) {
            float4 av  = *reinterpret_cast<const float4*>(ab  + kt);
            float4 av2 = *reinterpret_cast<const float4*>(ab2 + kt);
            float4 bv  = *reinterpret_cast<const float4*>(ab  + kt);
            float4 bv2 = *reinterpret_cast<const float4*>(ab2 + kt);
            as[buf][lc+0][sc]=av.x;  as[buf][lc+1][sc]=av.y;
            as[buf][lc+2][sc]=av.z;  as[buf][lc+3][sc]=av.w;
            as[buf][lc+0][sc2]=av2.x; as[buf][lc+1][sc2]=av2.y;
            as[buf][lc+2][sc2]=av2.z; as[buf][lc+3][sc2]=av2.w;
            bs[buf][lc+0][sc]=bv.x;  bs[buf][lc+1][sc]=bv.y;
            bs[buf][lc+2][sc]=bv.z;  bs[buf][lc+3][sc]=bv.w;
            bs[buf][lc+0][sc2]=bv2.x; bs[buf][lc+1][sc2]=bv2.y;
            bs[buf][lc+2][sc2]=bv2.z; bs[buf][lc+3][sc2]=bv2.w;
        } else if (MODE == 1) {
            // m-major destination: row lr holds this thread's 4 k values.
            float* d1 = &as[buf][0][0] + (size_t)lr  * (BK + 4) + lc;
            float* d2 = &as[buf][0][0] + (size_t)lr2 * (BK + 4) + lc;
            float* d3 = &bs[buf][0][0] + (size_t)lr  * (BK + 4) + lc;
            float* d4 = &bs[buf][0][0] + (size_t)lr2 * (BK + 4) + lc;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                         "r"(smem_addr(d1)), "l"(ab  + kt));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                         "r"(smem_addr(d2)), "l"(ab2 + kt));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                         "r"(smem_addr(d3)), "l"(ab  + kt));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                         "r"(smem_addr(d4)), "l"(ab2 + kt));
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 0;\n" ::);
        } else {
            float4 av  = *reinterpret_cast<const float4*>(ab  + kt);
            float4 av2 = *reinterpret_cast<const float4*>(ab2 + kt);
            float4 bv  = *reinterpret_cast<const float4*>(ab  + kt);
            float4 bv2 = *reinterpret_cast<const float4*>(ab2 + kt);
            *reinterpret_cast<float4*>(&as[buf][0][0] + (size_t)lr *(BK+4)+lc) = av;
            *reinterpret_cast<float4*>(&as[buf][0][0] + (size_t)lr2*(BK+4)+lc) = av2;
            *reinterpret_cast<float4*>(&bs[buf][0][0] + (size_t)lr *(BK+4)+lc) = bv;
            *reinterpret_cast<float4*>(&bs[buf][0][0] + (size_t)lr2*(BK+4)+lc) = bv2;
        }
        __syncthreads();
        acc += as[buf][lc & (BK-1)][sc] + bs[buf][lc & (BK-1)][sc];
        __syncthreads();
    }
    if (tid == 100000) sink[0] = acc;
}

int main() {
    const int blocks = 1952;
    float *g, *sink;
    CK(cudaMalloc(&g, (size_t)blocks * BM * KTILES * BK * sizeof(float) / 8));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(g, 0x3c, (size_t)blocks * BM * KTILES * BK * sizeof(float) / 8));
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    const char* nm[3] = {"ldg_sts (shipping)", "cpasync", "ldg_sts128"};
    for (int m = 0; m < 3; ++m) {
        for (int w = 0; w < 3; ++w) {
            if (m==0) stage<0><<<blocks/8, THREADS>>>(g, sink);
            else if (m==1) stage<1><<<blocks/8, THREADS>>>(g, sink);
            else stage<2><<<blocks/8, THREADS>>>(g, sink);
        }
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaEventRecord(t0));
        for (int r = 0; r < 20; ++r) {
            if (m==0) stage<0><<<blocks/8, THREADS>>>(g, sink);
            else if (m==1) stage<1><<<blocks/8, THREADS>>>(g, sink);
            else stage<2><<<blocks/8, THREADS>>>(g, sink);
        }
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        printf("%-20s %8.3f ms\n", nm[m], ms / 20);
    }
    CK(cudaFree(g)); CK(cudaFree(sink));
    return 0;
}
