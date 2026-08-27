// 측정-M. 두 가지를 한 번에 잰다.
//  (a) 이 부품의 천장은 무엇인가 — 370 W 전력벽인가, 발행 슬롯인가.
//      순수 FFMA 커널(메모리 0)을 GEMM과 같은 상주 형태(164블록×256스레드,
//      2블록/SM)로 돌려 지속 클럭·전력·flop/cycle을 잰다. 이 값이 %peak의
//      분모여야 한다 — 20992 flop/cycle × idle 클럭은 도달 불가능한 분모다.
//  (b) 비-GEMM을 GEMM 뒤에 숨길 수 있는가(측정-L 레버 4, 풀 99 ms/6.2%).
//      GEMM은 REG 125 × 512스레드 = 65536으로 레지스터 파일을 정확히 다 쓴다.
//      norm 블록은 32스레드 × ~40 = 1280뿐이지만 남은 공간이 0이므로 동거가
//      불가능할 수 있다. 그러나 블록이 은퇴하며 생기는 틈으로 끼어들 여지가
//      있어 산술로는 닫히지 않는다 → 잰다.
//
// kern.inc = sed -n '95,1155p' src/model.cu (커밋 01e9fae). 커널은 출하본 그대로.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <nvml.h>
#include "config.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)
#define NK(x) do { nvmlReturn_t e_ = (x); if (e_ != NVML_SUCCESS) { \
    fprintf(stderr, "NVML %s @%d: %s\n", #x, __LINE__, nvmlErrorString(e_)); \
    exit(1); } } while (0)

static void cuda_check(cudaError_t s, const char* what) {
    if (s != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(s)); exit(1); }
}

namespace {
#include "kern.inc"

// 순수 FFMA. 스레드당 독립 체인 32개(FFMA 4-cycle 레이턴시를 덮기에 충분),
// 메모리 접근 0. 바깥 루프를 4배 전개해 FFMA 명령비를 128/130 = 98.5%로 올린다.
constexpr int FF_CHAINS = 32;
__global__ __launch_bounds__(256, 2)
void ffma_peak(float* __restrict__ out, int iters) {
    float acc[FF_CHAINS];
#pragma unroll
    for (int i = 0; i < FF_CHAINS; ++i) acc[i] = (float)(threadIdx.x + i) * 1e-3f;
    const float y = 1.0000001f;
    const float x = (float)blockIdx.x * 1e-7f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int i = 0; i < FF_CHAINS; ++i) acc[i] = __fmaf_rn(acc[i], y, x);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < FF_CHAINS; ++i) s += acc[i];
    if (s == 1.2345678e30f) out[0] = s;   // 죽은 코드 제거 방지
}

// 순수 스트리밍. DRAM 전력의 기준선.
__global__ void dram_copy(const float4* __restrict__ src, float4* __restrict__ dst, long long n4) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (long long)gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace

int main(int argc, char** argv) {
    const int M = 15583, K = 4096, N = 2048;        // q_proj 형상
    const int NROWS = 15583, NCOLS = 4096;          // input_norm 형상
    const double block_ms = argc > 1 ? atof(argv[1]) : 400.0;
    const int cycles = argc > 2 ? atoi(argv[2]) : 3;

    NK(nvmlInit());
    nvmlDevice_t dev;
    NK(nvmlDeviceGetHandleByIndex(0, &dev));
    unsigned lim = 0, tlim = 0;
    NK(nvmlDeviceGetPowerManagementLimit(dev, &lim));
    NK(nvmlDeviceGetTemperatureThreshold(dev, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &tlim));
    printf("power limit %.1f W, slowdown temp %u C\n", lim / 1000.0, tlim);

    float *a, *b, *bias, *c, *nx, *ny, *nw, *nb, *dsrc, *ddst;
    CK(cudaMalloc(&a, (size_t)M * K * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&bias, (size_t)N * sizeof(float)));
    CK(cudaMalloc(&c, (size_t)M * N * sizeof(float)));
    CK(cudaMalloc(&nx, (size_t)NROWS * NCOLS * sizeof(float)));
    CK(cudaMalloc(&ny, (size_t)NROWS * NCOLS * sizeof(float)));
    CK(cudaMalloc(&nw, (size_t)NCOLS * sizeof(float)));
    CK(cudaMalloc(&nb, (size_t)NCOLS * sizeof(float)));
    const long long dn = (long long)NROWS * NCOLS / 4;
    CK(cudaMalloc(&dsrc, (size_t)dn * sizeof(float4)));
    CK(cudaMalloc(&ddst, (size_t)dn * sizeof(float4)));
    CK(cudaMemset(a, 0x3c, (size_t)M * K * sizeof(float)));
    CK(cudaMemset(b, 0x3c, (size_t)N * K * sizeof(float)));
    CK(cudaMemset(bias, 0, (size_t)N * sizeof(float)));
    CK(cudaMemset(nx, 0x3c, (size_t)NROWS * NCOLS * sizeof(float)));
    CK(cudaMemset(nw, 0x3c, (size_t)NCOLS * sizeof(float)));
    CK(cudaMemset(nb, 0, (size_t)NCOLS * sizeof(float)));
    CK(cudaMemset(dsrc, 0x3c, (size_t)dn * sizeof(float4)));

    cudaStream_t s0, s1;
    CK(cudaStreamCreate(&s0));
    CK(cudaStreamCreate(&s1));

    const dim3 ggrid((N + PROJ_BN - 1) / PROJ_BN, (M + PROJ_BM - 1) / PROJ_BM);
    const unsigned nblocks = (NROWS + NORM_ROWS - 1) / NORM_ROWS;
    const int FF_BLOCKS = 164, FF_ITERS = 4096;

    auto gemm = [&](cudaStream_t s) {
        gemm_nt_bias<<<ggrid, GEMM_THREADS, 0, s>>>(a, b, bias, c, M, K, N);
    };
    auto norm = [&](cudaStream_t s) {
        layer_norm_rows<<<nblocks, NORM_ROWS, 0, s>>>(nx, nw, nb, ny, NROWS, NCOLS,
                                                      apss26::NORM_EPS);
    };
    auto ffma = [&](cudaStream_t s) {
        ffma_peak<<<FF_BLOCKS, 256, 0, s>>>(c, FF_ITERS);
    };
    auto dram = [&](cudaStream_t s) {
        dram_copy<<<656, 256, 0, s>>>((const float4*)dsrc, (float4*)ddst, dn);
    };

    const char* names[] = {"ffma_peak", "gemm", "norm", "dram_copy",
                           "gemm|norm", "gemm;norm"};
    enum { NV = 6 };
    auto run = [&](int v) {
        switch (v) {
            case 0: ffma(s0); break;
            case 1: gemm(s0); break;
            case 2: norm(s0); break;
            case 3: dram(s0); break;
            case 4: gemm(s0); norm(s1); break;   // 동시
            case 5: gemm(s0); norm(s0); break;   // 순차, 같은 쌍
        }
    };
    auto sync = [&]() { CK(cudaStreamSynchronize(s0)); CK(cudaStreamSynchronize(s1)); };

    // 한 번씩 재서 rep 수를 정한다.
    double one[NV];
    for (int v = 0; v < NV; ++v) {
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        run(v); sync();
        CK(cudaEventRecord(e0, s0)); run(v); CK(cudaEventRecord(e1, s0));
        sync(); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        one[v] = ms > 0.01 ? ms : 0.01;
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }

    double acc_ms[NV] = {}, acc_mhz[NV] = {}, acc_w[NV] = {};
    unsigned tmax = 0;
    printf("\n%-10s %8s %8s %7s %7s %6s  %s\n", "variant", "ms", "MHz", "W", "degC",
           "%peak", "note");
    for (int cyc = 0; cyc < cycles; ++cyc) {
        for (int v = 0; v < NV; ++v) {
            const int reps = (int)(block_ms / one[v]) + 1;
            // 정착: 전력 제어 루프가 자리를 잡을 때까지 같은 변이를 계속 돌린다.
            for (int i = 0; i < reps; ++i) run(v);
            sync();
            for (int i = 0; i < reps; ++i) run(v);
            sync();
            // 측정
            cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0, s0));
            for (int i = 0; i < reps; ++i) run(v);
            unsigned mhz = 0, mw = 0, tc = 0;
            NK(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz));
            NK(nvmlDeviceGetPowerUsage(dev, &mw));
            NK(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &tc));
            CK(cudaEventRecord(e1, s0));
            sync(); CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= reps;
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            acc_ms[v] += ms; acc_mhz[v] += mhz; acc_w[v] += mw / 1000.0;
            if (tc > tmax) tmax = tc;
            if (cyc == cycles - 1) {
                const double m = acc_ms[v] / cycles, f = acc_mhz[v] / cycles,
                             w = acc_w[v] / cycles;
                double pk = 0;
                if (v == 0) {
                    const double fl = 2.0 * FF_CHAINS * 4.0 * FF_ITERS * FF_BLOCKS * 256.0;
                    pk = fl / (m * 1e-3) / (f * 1e6) / 20992.0 * 100.0;
                } else if (v == 1 || v == 4 || v == 5) {
                    const double fl = 2.0 * (double)(ggrid.y * PROJ_BM) *
                                      (ggrid.x * PROJ_BN) * K;
                    pk = fl / (m * 1e-3) / (f * 1e6) / 20992.0 * 100.0;
                }
                printf("%-10s %8.3f %8.0f %7.1f %7.0f %6.2f\n", names[v], m, f, w,
                       (double)tmax, pk);
            }
        }
    }
    const double mg = acc_ms[1] / cycles, mn = acc_ms[2] / cycles,
                 mc = acc_ms[4] / cycles, ms_ = acc_ms[5] / cycles;
    printf("\ngemm %.3f + norm %.3f = %.3f | 순차쌍 %.3f | 동시 %.3f\n",
           mg, mn, mg + mn, ms_, mc);
    printf("겹침 회수 = %.3f ms = norm의 %.1f%%  (동시가 순차보다 이만큼 빠르다)\n",
           ms_ - mc, (ms_ - mc) / mn * 100.0);
    NK(nvmlShutdown());
    return 0;
}
