// 측정-M part2. GEMM이 1610 MHz로 떨어지는 이유를 확정한다.
//  part1: 순수 FFMA = 1900 MHz / 294 W / 피크 96%.  GEMM = 1610 MHz / 320 W.
//  전력 한계는 370 W인데 GEMM은 320 W다. 그런데도 클럭이 290 MHz 낮다.
//  가설 A: 전력 제어(순간 전력이 한계를 치고 NVML 읽기는 평균) → 바이트를 줄이면
//          클럭이 올라간다 = 모든 바이트 레버가 클럭 레버로 증폭된다.
#include <cstring>
//  가설 B: 전압/전류(HW) 벽 또는 다른 캡 → 바이트로는 못 산다.
//  판정: throttle reason 비트와, 순수 FFMA에 DRAM 트래픽만 더했을 때의 클럭.
#include <cstdio>
#include <cstdlib>
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
constexpr int FF_CHAINS = 32;
__global__ __launch_bounds__(256, 2)
void ffma_peak(float* __restrict__ out, int iters) {
    float acc[FF_CHAINS];
#pragma unroll
    for (int i = 0; i < FF_CHAINS; ++i) acc[i] = (float)(threadIdx.x + i) * 1e-3f;
    const float y = 1.0000001f, x = (float)blockIdx.x * 1e-7f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int i = 0; i < FF_CHAINS; ++i) acc[i] = __fmaf_rn(acc[i], y, x);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < FF_CHAINS; ++i) s += acc[i];
    if (s == 1.2345678e30f) out[0] = s;
}
__global__ void dram_copy(const float4* __restrict__ src, float4* __restrict__ dst, long long n4) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (long long)gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace

static void reasons(unsigned long long r, char* buf) {
    buf[0] = 0;
    struct { unsigned long long bit; const char* name; } t[] = {
        {nvmlClocksThrottleReasonGpuIdle, "idle"},
        {nvmlClocksThrottleReasonApplicationsClocksSetting, "appclk"},
        {nvmlClocksThrottleReasonSwPowerCap, "SW_POWER_CAP"},
        {nvmlClocksThrottleReasonHwSlowdown, "HW_SLOWDOWN"},
        {nvmlClocksThrottleReasonSyncBoost, "syncboost"},
        {nvmlClocksThrottleReasonSwThermalSlowdown, "sw_thermal"},
        {nvmlClocksThrottleReasonHwThermalSlowdown, "hw_thermal"},
        {nvmlClocksThrottleReasonHwPowerBrakeSlowdown, "HW_POWER_BRAKE"},
        {nvmlClocksThrottleReasonDisplayClockSetting, "displayclk"},
    };
    for (auto& e : t) if (r & e.bit) { strcat(buf, e.name); strcat(buf, " "); }
    if (!buf[0]) strcpy(buf, "-none-");
}

int main(int argc, char** argv) {
    const int M = 15583, K = 4096, N = 2048;
    const int NROWS = 15583, NCOLS = 4096;
    const double block_ms = argc > 1 ? atof(argv[1]) : 600.0;

    NK(nvmlInit());
    nvmlDevice_t dev;
    NK(nvmlDeviceGetHandleByIndex(0, &dev));
    unsigned lim = 0;
    NK(nvmlDeviceGetPowerManagementLimit(dev, &lim));
    printf("power limit %.1f W\n", lim / 1000.0);

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
    CK(cudaStreamCreate(&s0)); CK(cudaStreamCreate(&s1));
    const dim3 ggrid((N + PROJ_BN - 1) / PROJ_BN, (M + PROJ_BM - 1) / PROJ_BM);
    const unsigned nblocks = (NROWS + NORM_ROWS - 1) / NORM_ROWS;
    const int FF_BLOCKS = 164, FF_ITERS = 4096;

    auto gemm = [&](cudaStream_t s) { gemm_nt_bias<<<ggrid, GEMM_THREADS, 0, s>>>(a, b, bias, c, M, K, N); };
    auto norm = [&](cudaStream_t s) { layer_norm_rows<<<nblocks, NORM_ROWS, 0, s>>>(nx, nw, nb, ny, NROWS, NCOLS, apss26::NORM_EPS); };
    auto ffma = [&](cudaStream_t s) { ffma_peak<<<FF_BLOCKS, 256, 0, s>>>(c, FF_ITERS); };
    auto dram = [&](cudaStream_t s) { dram_copy<<<656, 256, 0, s>>>((const float4*)dsrc, (float4*)ddst, dn); };
    // FFMA 블록을 SM당 2개로 유지하면서 DRAM 트래픽만 얹는다: 레지스터 39×512=19968,
    // 남은 45568로 dram_copy 블록이 충분히 동거한다.
    const char* names[] = {"ffma", "ffma|dram", "gemm", "gemm|norm", "norm", "dram"};
    enum { NV = 6 };
    auto run = [&](int v) {
        switch (v) {
            case 0: ffma(s0); break;
            case 1: ffma(s0); dram(s1); break;
            case 2: gemm(s0); break;
            case 3: gemm(s0); norm(s1); break;
            case 4: norm(s0); break;
            case 5: dram(s0); break;
        }
    };
    auto sync = [&]() { CK(cudaStreamSynchronize(s0)); CK(cudaStreamSynchronize(s1)); };

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

    printf("\n%-11s %8s %6s %6s %7s %7s  %s\n", "variant", "ms", "MHzav", "MHzmin",
           "Wav", "Wmax", "throttle");
    for (int v = 0; v < NV; ++v) {
        const int reps = (int)(block_ms / one[v]) + 1;
        for (int i = 0; i < reps; ++i) run(v);
        sync();
        for (int i = 0; i < reps; ++i) run(v);
        sync();
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, s0));
        for (int i = 0; i < reps; ++i) run(v);
        // 큐가 깊은 동안 계속 표본을 뜬다.
        double smhz = 0, sw = 0; unsigned nmin = 9999, wmax = 0; int ns = 0;
        unsigned long long rmask = 0;
        for (int k = 0; k < 40; ++k) {
            unsigned mhz = 0, mw = 0; unsigned long long rr = 0;
            NK(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz));
            NK(nvmlDeviceGetPowerUsage(dev, &mw));
            NK(nvmlDeviceGetCurrentClocksThrottleReasons(dev, &rr));
            smhz += mhz; sw += mw / 1000.0; ++ns; rmask |= rr;
            if (mhz < nmin) nmin = mhz;
            if (mw > wmax) wmax = mw;
        }
        CK(cudaEventRecord(e1, s0));
        sync(); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= reps;
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
        char rb[256]; reasons(rmask, rb);
        printf("%-11s %8.3f %6.0f %6u %7.1f %7.1f  %s\n", names[v], ms, smhz / ns,
               nmin, sw / ns, wmax / 1000.0, rb);
    }
    NK(nvmlShutdown());
    return 0;
}
