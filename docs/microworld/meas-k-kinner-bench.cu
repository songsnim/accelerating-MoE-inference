// Head-to-head: the shipping k-major shared layout vs a k-inner register tile
// that keeps shared in the global (m-major) layout.
//
// Shipping: shared is as[k][m]. Global A is a[m][k], so staging transposes and
// each thread's float4 becomes 4 scalar STS. The FFMA loop reads a float4 along
// m at fixed k.
//
// k-inner: shared is as[m][k]. Staging is a straight copy -> one STS.128. The
// FFMA loop reads a float4 along k at fixed m and accumulates 4 k-steps per
// operand pair. Same FFMA count, same LDS.128 count; only the store side and
// the loop nest order change.
//
// Row stride BK+4 = 20 floats. 20 mod 32 = 20, and {m*20 mod 32 : m=0..7} =
// {0,20,8,28,16,4,24,12} -- eight distinct float4 slots covering all 32 banks
// exactly once, so the k-direction LDS.128 is conflict-free at the same 40,960
// bytes of shared memory the shipping kernel uses.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

namespace {
constexpr int BK = 16, THREADS = 256;
constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
constexpr int KPAD = 4;                 // row stride BK + KPAD = 20 floats
constexpr int KS = BK + KPAD;

// ---------------- k-inner register tile ----------------
__global__ __launch_bounds__(THREADS, 2)
void gemm_kinner(const float* __restrict__ a, const float* __restrict__ b,
                 const float* __restrict__ bias, float* __restrict__ c,
                 int m, int k, int n) {
    __shared__ __align__(16) float as[2][BM][KS];
    __shared__ __align__(16) float bs[2][BN][KS];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int lr = tid / (BK / 4);          // 0..63 row inside the tile
    const int lc = (tid % (BK / 4)) * 4;    // 0,4,8,12 inside the k window
    const int lr2 = lr + BM / 2;
    const int ty = tid / (BN / TN);         // 0..15
    const int tx = tid % (BN / TN);         // 0..15

    const int a_row = m0 + lr, a_row2 = m0 + lr2;
    const int b_row = n0 + lr, b_row2 = n0 + lr2;
    const bool a_ok = a_row < m, a_ok2 = a_row2 < m;
    const bool b_ok = b_row < n, b_ok2 = b_row2 < n;
    const float* const ab  = a + (long long)a_row  * k + lc;
    const float* const ab2 = a + (long long)a_row2 * k + lc;
    const float* const bb  = b + (long long)b_row  * k + lc;
    const float* const bb2 = b + (long long)b_row2 * k + lc;

    float acc[TM][TN] = {};
    float4 av = make_float4(0.f,0.f,0.f,0.f), av2 = av, bv = av, bv2 = av;
    if (a_ok)  av  = *reinterpret_cast<const float4*>(ab);
    if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2);
    if (b_ok)  bv  = *reinterpret_cast<const float4*>(bb);
    if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2);
    *reinterpret_cast<float4*>(&as[0][lr ][lc]) = av;
    *reinterpret_cast<float4*>(&as[0][lr2][lc]) = av2;
    *reinterpret_cast<float4*>(&bs[0][lr ][lc]) = bv;
    *reinterpret_cast<float4*>(&bs[0][lr2][lc]) = bv2;
    __syncthreads();

    int cur = 0;
    for (int kt = 0; kt < k; kt += BK) {
        const bool more = kt + BK < k;
        if (more) {
            if (a_ok)  av  = *reinterpret_cast<const float4*>(ab  + kt + BK);
            if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2 + kt + BK);
            if (b_ok)  bv  = *reinterpret_cast<const float4*>(bb  + kt + BK);
            if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2 + kt + BK);
        }
        const float (*ac)[KS] = as[cur];
        const float (*bc)[KS] = bs[cur];
#pragma unroll
        for (int p = 0; p < BK; p += 4) {
            float4 a4[TM], b4[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a4[i] = *reinterpret_cast<const float4*>(&ac[ty * TM + i][p]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b4[j] = *reinterpret_cast<const float4*>(&bc[tx * TN + j][p]);
            // k ascending, one register per output, same `acc += a * b` order.
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a4[i].x * b4[j].x;
                    acc[i][j] += a4[i].y * b4[j].y;
                    acc[i][j] += a4[i].z * b4[j].z;
                    acc[i][j] += a4[i].w * b4[j].w;
                }
        }
        if (more) {
            const int nxt = cur ^ 1;
            *reinterpret_cast<float4*>(&as[nxt][lr ][lc]) = av;
            *reinterpret_cast<float4*>(&as[nxt][lr2][lc]) = av2;
            *reinterpret_cast<float4*>(&bs[nxt][lr ][lc]) = bv;
            *reinterpret_cast<float4*>(&bs[nxt][lr2][lc]) = bv2;
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
    float *a, *b, *bias, *c;
    CK(cudaMalloc(&a, (size_t)M*K*4)); CK(cudaMalloc(&b, (size_t)N*K*4));
    CK(cudaMalloc(&bias, (size_t)N*4)); CK(cudaMalloc(&c, (size_t)M*N*4));
    CK(cudaMemset(a, 0x3c, (size_t)M*K*4)); CK(cudaMemset(b, 0x3c, (size_t)N*K*4));
    CK(cudaMemset(bias, 0, (size_t)N*4));
    const dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    for (int i = 0; i < 3; ++i) gemm_kinner<<<grid, THREADS>>>(a,b,bias,c,M,K,N);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < reps; ++i) gemm_kinner<<<grid, THREADS>>>(a,b,bias,c,M,K,N);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1)); ms /= reps;
    const double gf = 2.0*(double)(grid.y*BM)*(grid.x*BN)*K*1e-9;
    printf("%-14s M=%d K=%d N=%d  grid=(%u,%u)  %8.3f ms  %7.2f TFLOP/s\n",
           "kinner", M, K, N, grid.x, grid.y, ms, gf/ms*1e-3);
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(bias)); CK(cudaFree(c));
    return 0;
}
