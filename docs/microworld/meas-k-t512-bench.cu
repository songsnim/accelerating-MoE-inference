// 512-thread / 8x4 register tile against the shipping 256-thread / 8x8.
//
// Same 128x128x16 block tile and the same 40,960 B of shared memory, so both
// are 2 blocks/SM -- but 512 threads makes that 32 warps/SM instead of 16.
// The trade is arithmetic intensity per shared load: 8x4 does 32 FFMA per
// 3 LDS.128 (8 wavefronts) where 8x8 does 64 FFMA per 4 LDS.128 (12), i.e.
// +33% shared traffic and +50% shared-load instructions, bought with 2x the
// warps to hide latency with.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

namespace {
constexpr int BK = 16, BM = 128, BN = 128, TM = 8, TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);   // 16 * 32 = 512
constexpr int SPAD = 32;
static_assert(THREADS == 512, "block is not 512 threads");
static_assert(BM * BK / 4 == THREADS, "A staging is not one float4 per thread");
static_assert(BN * BK / 4 == THREADS, "B staging is not one float4 per thread");

__device__ __forceinline__ int kshift(int k) { return (k >> 2) * 8; }

__global__ __launch_bounds__(THREADS, 2)
void gemm512(const float* __restrict__ a, const float* __restrict__ b,
             const float* __restrict__ bias, float* __restrict__ c,
             int m, int k, int n) {
    __shared__ __align__(16) float as[2][BK][BM + SPAD];
    __shared__ __align__(16) float bs[2][BK][BN + SPAD];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int lr = tid / (BK / 4);          // 0..127, one row each
    const int lc = (tid % (BK / 4)) * 4;    // 0,4,8,12
    const int sc = lr + kshift(lc);
    const int ty = tid / (BN / TN);         // 0..15
    const int tx = tid % (BN / TN);         // 0..31

    const int a_row = m0 + lr, b_row = n0 + lr;
    const bool a_ok = a_row < m, b_ok = b_row < n;
    const float* const ab = a + (long long)a_row * k + lc;
    const float* const bb = b + (long long)b_row * k + lc;

    float acc[TM][TN] = {};
    float4 av = make_float4(0.f,0.f,0.f,0.f), bv = av;
    if (a_ok) av = *reinterpret_cast<const float4*>(ab);
    if (b_ok) bv = *reinterpret_cast<const float4*>(bb);
    as[0][lc+0][sc]=av.x; as[0][lc+1][sc]=av.y; as[0][lc+2][sc]=av.z; as[0][lc+3][sc]=av.w;
    bs[0][lc+0][sc]=bv.x; bs[0][lc+1][sc]=bv.y; bs[0][lc+2][sc]=bv.z; bs[0][lc+3][sc]=bv.w;
    __syncthreads();

    int cur = 0;
    for (int kt = 0; kt < k; kt += BK) {
        const bool more = kt + BK < k;
        if (more) {
            if (a_ok) av = *reinterpret_cast<const float4*>(ab + kt + BK);
            if (b_ok) bv = *reinterpret_cast<const float4*>(bb + kt + BK);
        }
        const float (*ac)[BM + SPAD] = as[cur];
        const float (*bc)[BN + SPAD] = bs[cur];
#pragma unroll
        for (int p = 0; p < BK; ++p) {
            const int sh = kshift(p);
            float av4[TM], bv4[TN];
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 t = *reinterpret_cast<const float4*>(&ac[p][ty * TM + i + sh]);
                av4[i]=t.x; av4[i+1]=t.y; av4[i+2]=t.z; av4[i+3]=t.w;
            }
            {   // TN = 4: one float4, lane tx owns columns tx*4 .. tx*4+3.
                const float4 t = *reinterpret_cast<const float4*>(&bc[p][tx * TN + sh]);
                bv4[0]=t.x; bv4[1]=t.y; bv4[2]=t.z; bv4[3]=t.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av4[i] * bv4[j];
        }
        if (more) {
            const int nxt = cur ^ 1;
            as[nxt][lc+0][sc]=av.x; as[nxt][lc+1][sc]=av.y;
            as[nxt][lc+2][sc]=av.z; as[nxt][lc+3][sc]=av.w;
            bs[nxt][lc+0][sc]=bv.x; bs[nxt][lc+1][sc]=bv.y;
            bs[nxt][lc+2][sc]=bv.z; bs[nxt][lc+3][sc]=bv.w;
            __syncthreads();
            cur = nxt;
        }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = m0 + ty * TM + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + tx * TN + j;
            if (col >= n) continue;
            float v = acc[i][j];
            if (bias != nullptr) v += bias[col];
            c[(long long)row * n + col] = v;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 15583;
    const int K = argc > 2 ? atoi(argv[2]) : 4096;
    const int N = argc > 3 ? atoi(argv[3]) : 2048;
    const int reps = argc > 4 ? atoi(argv[4]) : 20;
    float *a,*b,*bias,*c;
    CK(cudaMalloc(&a,(size_t)M*K*4)); CK(cudaMalloc(&b,(size_t)N*K*4));
    CK(cudaMalloc(&bias,(size_t)N*4)); CK(cudaMalloc(&c,(size_t)M*N*4));
    CK(cudaMemset(a,0x3c,(size_t)M*K*4)); CK(cudaMemset(b,0x3c,(size_t)N*K*4));
    CK(cudaMemset(bias,0,(size_t)N*4));
    const dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
    for (int i=0;i<3;++i) gemm512<<<grid,THREADS>>>(a,b,bias,c,M,K,N);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i=0;i<reps;++i) gemm512<<<grid,THREADS>>>(a,b,bias,c,M,K,N);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1)); ms/=reps;
    const double gf = 2.0*(double)(grid.y*BM)*(grid.x*BN)*K*1e-9;
    printf("%-14s M=%d K=%d N=%d  grid=(%u,%u)  %8.3f ms  %7.2f TFLOP/s\n",
           "t512_8x4", M,K,N,grid.x,grid.y, ms, gf/ms*1e-3);
    return 0;
}
