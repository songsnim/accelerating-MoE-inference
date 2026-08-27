// EXP-043. Lever 3: can the short-K per-block fixed cost (prologue staging +
// epilogue drain) be amortised across output tiles?
//
//  base      verbatim gemm_nt_bias from src/model.cu, one output tile per block
//  param0    the same math in this file's reformulated body, m0/n0 from
//            blockIdx as base takes them. Control for the reformulation:
//            same REG:125 STACK:0, so any delta is pure instruction schedule.
//  pnaive32  persistent 1-D grid, grid-stride over flattened tiles (NTX = 32
//            compile-time, so the tile split is a shift and a mask, not a
//            runtime division). Tiles run back to back, no boundary overlap.
//  ppipe32   persistent + the next tile's k = 0 global load issued in front of
//            this tile's epilogue stores, so the LDG latency hides behind the
//            STG issue. This is the mechanism the lever asked for.
//  coal_st   timing ablation: same STG count, but a warp's 32 lanes write 32
//            consecutive words instead of 16 B-strided ones (2048 sector
//            requests per block instead of 8192). Values land wrong.
//  l2_st     timing ablation: every block writes the same 128 rows, so the
//            launch's whole store stream is L2-resident. Isolates write
//            volume. Values land wrong.
//
// Harness: from idle this part runs at 1740 MHz / 48 W and settles near
// 1590 MHz / 290 W after ~2 s of dense FFMA. Timing variants in source order
// therefore hands the first one up to 5%, which is larger than anything being
// measured here -- so the bench soaks for 3 s, then round-robins the variants,
// and prints the SM clock with each measurement so the result can be quoted
// per cycle. Every earlier reading in this file's history that did not do that
// was wrong by more than its own effect size.
//
// verbatim.inc is `sed -n '95,408p' src/model.cu` at 683f487.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvml.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)
#define NK(x) do { nvmlReturn_t e_ = (x); if (e_ != NVML_SUCCESS) { \
    fprintf(stderr, "NVML %s @%d: %s\n", #x, __LINE__, nvmlErrorString(e_)); \
    exit(1); } } while (0)

namespace {
// ---- verbatim from src/model.cu ----
#include "verbatim.inc"
// ---- end verbatim ----

// One copy of the compute core, parameterised on how tiles are assigned.
// P = 0 single tile (n0/m0 from arguments), 1 grid-stride, 2 grid-stride + the
// cross-boundary prefetch. Everything inside the k loop is the verbatim text.
template <int P, int EPI = 0, int NTX = 0>
__global__ __launch_bounds__(GEMM_THREADS, 2)
void gemm_tl(const float* __restrict__ a, const float* __restrict__ b,
             const float* __restrict__ bias, float* __restrict__ c,
             int m, int k, int n, int ntx, int ntiles) {
    constexpr int BM = PROJ_BM, BN = PROJ_BN, TM = PROJ_TM, TN = PROJ_TN;
    constexpr bool SWIZ = PROJ_SWIZ;
    constexpr int SPAD = gemm_spad(SWIZ);
    __shared__ __align__(16) float as[2][BK][BM + SPAD];
    __shared__ __align__(16) float bs[2][BK][BN + SPAD];

    const int tid = threadIdx.x;
    const int lr = tid / (BK / 4);
    const int lc = (tid % (BK / 4)) * 4;
    const int lr2 = lr + BM / 2;
    const int sc = lr + gemm_kshift(SWIZ, lc), sc2 = sc + BM / 2;
    const int ty = tid / (BN / TN);
    const int tx = tid % (BN / TN);

    int t = P == 0 ? blockIdx.y * ntx + blockIdx.x : blockIdx.x;
    int m0, n0;
    if (P == 0 && NTX == 0) { m0 = blockIdx.y * BM; n0 = blockIdx.x * BN; }
    else if (NTX) { m0 = (t / NTX) * BM; n0 = (t & (NTX - 1)) * BN; }
    else { m0 = (t / ntx) * BM; n0 = (t % ntx) * BN; }
    int a_row = m0 + lr, a_row2 = m0 + lr2;
    int b_row = n0 + lr, b_row2 = n0 + lr2;
    bool a_ok = a_row < m, a_ok2 = a_row2 < m;
    bool b_ok = b_row < n, b_ok2 = b_row2 < n;
    const float* ab = a + (long long)a_row * k + lc;
    const float* ab2 = a + (long long)a_row2 * k + lc;
    const float* bb = b + (long long)b_row * k + lc;
    const float* bb2 = b + (long long)b_row2 * k + lc;

    float4 av = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 av2 = av, bv = av, bv2 = av;
    if (a_ok) av = *reinterpret_cast<const float4*>(ab);
    if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2);
    if (b_ok) bv = *reinterpret_cast<const float4*>(bb);
    if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2);

    for (;;) {
        float acc[TM][TN] = {};
        as[0][lc + 0][sc] = av.x; as[0][lc + 1][sc] = av.y;
        as[0][lc + 2][sc] = av.z; as[0][lc + 3][sc] = av.w;
        as[0][lc + 0][sc2] = av2.x; as[0][lc + 1][sc2] = av2.y;
        as[0][lc + 2][sc2] = av2.z; as[0][lc + 3][sc2] = av2.w;
        bs[0][lc + 0][sc] = bv.x; bs[0][lc + 1][sc] = bv.y;
        bs[0][lc + 2][sc] = bv.z; bs[0][lc + 3][sc] = bv.w;
        bs[0][lc + 0][sc2] = bv2.x; bs[0][lc + 1][sc2] = bv2.y;
        bs[0][lc + 2][sc2] = bv2.z; bs[0][lc + 3][sc2] = bv2.w;
        __syncthreads();

        int cur = 0;
        for (int kt = 0; kt < k; kt += BK) {
            const bool more = kt + BK < k;
            if (more) {
                if (a_ok) av = *reinterpret_cast<const float4*>(ab + kt + BK);
                if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2 + kt + BK);
                if (b_ok) bv = *reinterpret_cast<const float4*>(bb + kt + BK);
                if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2 + kt + BK);
            }
            const float (*ac)[BM + SPAD] = as[cur];
            const float (*bc)[BN + SPAD] = bs[cur];
#pragma unroll
            for (int p = 0; p < BK; ++p) {
                float av4[TM], bv4[TN];
#pragma unroll
                for (int i = 0; i < TM; i += 4) {
                    const float4 t4 = *reinterpret_cast<const float4*>(&ac[p][ty * TM + i + gemm_kshift(SWIZ, p)]);
                    av4[i] = t4.x; av4[i + 1] = t4.y; av4[i + 2] = t4.z; av4[i + 3] = t4.w;
                }
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const float4 t4 =
                        *reinterpret_cast<const float4*>(&bc[p][gemm_col_slot<BN>(tx, j) + gemm_kshift(SWIZ, p)]);
                    bv4[j] = t4.x; bv4[j + 1] = t4.y; bv4[j + 2] = t4.z; bv4[j + 3] = t4.w;
                }
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] += av4[i] * bv4[j];
            }
            if (more) {
                const int nxt = cur ^ 1;
                as[nxt][lc + 0][sc] = av.x; as[nxt][lc + 1][sc] = av.y;
                as[nxt][lc + 2][sc] = av.z; as[nxt][lc + 3][sc] = av.w;
                as[nxt][lc + 0][sc2] = av2.x; as[nxt][lc + 1][sc2] = av2.y;
                as[nxt][lc + 2][sc2] = av2.z; as[nxt][lc + 3][sc2] = av2.w;
                bs[nxt][lc + 0][sc] = bv.x; bs[nxt][lc + 1][sc] = bv.y;
                bs[nxt][lc + 2][sc] = bv.z; bs[nxt][lc + 3][sc] = bv.w;
                bs[nxt][lc + 0][sc2] = bv2.x; bs[nxt][lc + 1][sc2] = bv2.y;
                bs[nxt][lc + 2][sc2] = bv2.z; bs[nxt][lc + 3][sc2] = bv2.w;
                __syncthreads();
                cur = nxt;
            }
        }

        const int tn = t + (P == 0 ? ntiles : (int)gridDim.x);
        if (P == 2 && tn < ntiles) {
            // Next tile's addresses and its k = 0 load, in front of the
            // epilogue: the LDGs are in flight while the STGs issue.
            m0 = NTX ? (tn / NTX) * BM : (tn / ntx) * BM;
            n0 = NTX ? (tn & (NTX - 1)) * BN : (tn % ntx) * BN;
            a_row = m0 + lr; a_row2 = m0 + lr2;
            b_row = n0 + lr; b_row2 = n0 + lr2;
            a_ok = a_row < m; a_ok2 = a_row2 < m;
            b_ok = b_row < n; b_ok2 = b_row2 < n;
            ab = a + (long long)a_row * k + lc;
            ab2 = a + (long long)a_row2 * k + lc;
            bb = b + (long long)b_row * k + lc;
            bb2 = b + (long long)b_row2 * k + lc;
            av = make_float4(0.0f, 0.0f, 0.0f, 0.0f); av2 = av; bv = av; bv2 = av;
            if (a_ok) av = *reinterpret_cast<const float4*>(ab);
            if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2);
            if (b_ok) bv = *reinterpret_cast<const float4*>(bb);
            if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2);
        }
        // Epilogue of the tile just finished. `m0`/`n0` above have already
        // moved, so P == 2 recovers this tile's base from `t`.
        const int em = NTX ? (t / NTX) * BM : (t / ntx) * BM;
        const int en = NTX ? (t & (NTX - 1)) * BN : (t % ntx) * BN;
        if (EPI == 0) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row = em + ty * TM + i;
            if (row >= m) continue;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = en + gemm_col_slot<BN>(tx, j);
                if (col >= n) continue;
                c[(long long)row * n + col] = acc[i][j] + bias[col];
            }
        }
        } else {
            // Timing ablation only -- the values land in the wrong places.
            // Same STG count, but for a fixed (i, j) the 32 lanes of a warp
            // write 32 consecutive words instead of 16B-strided ones, so the
            // block's 64 KB costs 2048 sector requests instead of 8192.
            const int wp = tid >> 5, ln = tid & 31;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int idx = ((wp * 64 + i * TN + j) << 5) + ln;
                    const int row = em + (idx >> 7), col = en + (idx & 127);
                    if (row >= m || col >= n) continue;
                    c[(long long)row * n + col] = acc[i][j] + bias[col];
                }
        }
        t = tn;
        if (t >= ntiles) break;
        __syncthreads();
        if (P == 1) {
            m0 = NTX ? (t / NTX) * BM : (t / ntx) * BM;
            n0 = NTX ? (t & (NTX - 1)) * BN : (t % ntx) * BN;
            a_row = m0 + lr; a_row2 = m0 + lr2;
            b_row = n0 + lr; b_row2 = n0 + lr2;
            a_ok = a_row < m; a_ok2 = a_row2 < m;
            b_ok = b_row < n; b_ok2 = b_row2 < n;
            ab = a + (long long)a_row * k + lc;
            ab2 = a + (long long)a_row2 * k + lc;
            bb = b + (long long)b_row * k + lc;
            bb2 = b + (long long)b_row2 * k + lc;
            av = make_float4(0.0f, 0.0f, 0.0f, 0.0f); av2 = av; bv = av; bv2 = av;
            if (a_ok) av = *reinterpret_cast<const float4*>(ab);
            if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2);
            if (b_ok) bv = *reinterpret_cast<const float4*>(bb);
            if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2);
        }
    }
}
}  // namespace

constexpr int NV = 6;

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 31166;
    const int K = argc > 2 ? atoi(argv[2]) : 448;
    const int N = argc > 3 ? atoi(argv[3]) : 4096;
    const double target_ms = argc > 4 ? atof(argv[4]) : 150.0;
    const int cycles = argc > 5 ? atoi(argv[5]) : 5;
    const int nblk = argc > 6 ? atoi(argv[6]) : 164;

    nvmlDevice_t dev;
    NK(nvmlInit_v2());
    NK(nvmlDeviceGetHandleByIndex_v2(0, &dev));

    float *a, *b, *bias, *c;
    CK(cudaMalloc(&a, (size_t)M * K * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&bias, (size_t)N * sizeof(float)));
    CK(cudaMalloc(&c, (size_t)M * N * sizeof(float)));
    CK(cudaMemset(a, 0x3c, (size_t)M * K * sizeof(float)));
    CK(cudaMemset(b, 0x3c, (size_t)N * K * sizeof(float)));
    CK(cudaMemset(bias, 0, (size_t)N * sizeof(float)));

    const int ntx = (N + PROJ_BN - 1) / PROJ_BN, nty = (M + PROJ_BM - 1) / PROJ_BM;
    const int ntiles = ntx * nty;
    const dim3 g2(ntx, nty);
    const double flops = 2.0 * (double)(nty * PROJ_BM) * (ntx * PROJ_BN) * K;

    auto run = [&](int v) {
        if (v == 0) gemm_nt_bias<<<g2, GEMM_THREADS>>>(a, b, bias, c, M, K, N);
        else if (v == 1) gemm_tl<0, 0, 0><<<g2, GEMM_THREADS>>>(a, b, bias, c, M, K, N, ntx, ntiles);
        else if (v == 2) gemm_tl<1, 0, 32><<<nblk, GEMM_THREADS>>>(a, b, bias, c, M, K, N, ntx, ntiles);
        else if (v == 3) gemm_tl<2, 0, 32><<<nblk, GEMM_THREADS>>>(a, b, bias, c, M, K, N, ntx, ntiles);
        else if (v == 4) gemm_tl<0, 1, 0><<<g2, GEMM_THREADS>>>(a, b, bias, c, M, K, N, ntx, ntiles);
        else gemm_tl<0, 2, 0><<<g2, GEMM_THREADS>>>(a, b, bias, c, M, K, N, ntx, ntiles);
    };
    const char* nm[NV] = {"base", "param0", "pnaive32", "ppipe32", "coal_st", "l2_st"};

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    // Soak first: from idle the part runs at 1740 MHz / 48 W and settles near
    // 1590 MHz / 290 W after ~2 s. Measuring variants in source order without
    // this hands the first one a 5% clock advantage.
    CK(cudaEventRecord(t0));
    for (int i = 0; i < 5; ++i) run(0);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float one = 0; CK(cudaEventElapsedTime(&one, t0, t1));
    one /= 5;
    const int reps = (int)(target_ms / one) + 1;
    const int soak = (int)(3000.0 / one) + 1;
    for (int i = 0; i < soak; ++i) run(0);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

    printf("M=%d K=%d N=%d tiles=%d blk=%d reps=%d\n", M, K, N, ntiles, nblk, reps);
    printf("%-9s %8s %7s %5s %9s %8s\n", "variant", "ms", "MHz", "W", "flop/cyc", "%peak");
    for (int cyc = 0; cyc < cycles; ++cyc) {
        for (int v = 0; v < NV; ++v) {
            CK(cudaEventRecord(t0));
            for (int i = 0; i < reps; ++i) run(v);
            unsigned mhz = 0, mw = 0;
            NK(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz));
            NK(nvmlDeviceGetPowerUsage(dev, &mw));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            CK(cudaGetLastError());
            float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
            ms /= reps;
            const double fpc = flops / (ms * 1e-3) / (mhz * 1e6);
            printf("%-9s %8.4f %7u %5.0f %9.1f %8.2f\n", nm[v], ms, mhz,
                   mw / 1000.0, fpc, fpc / 20992.0 * 100.0);
            fflush(stdout);
        }
    }
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(bias)); CK(cudaFree(c));
    NK(nvmlShutdown());
    return 0;
}
