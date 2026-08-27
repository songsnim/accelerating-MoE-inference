// EXP-043. Is the K=448 kernel intrinsically less efficient, or was meas-k's
// sweep reading the clock? meas-k timed one 20-rep mean per K in descending K
// order in a single process, and this part loses 8.6% of its clock over the
// first ~2 s of dense FFMA, so the sweep's later (short-K) points were the
// hot ones.
// verbatim.inc is `sed -n '95,408p' src/model.cu` at 683f487.
// Same buffers, the two K values interleaved so they share
// the thermal state, and the SM clock sampled while each measurement runs so
// the result can be quoted per cycle instead of per second.
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
}  // namespace

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 15583;
    const int N = argc > 2 ? atoi(argv[2]) : 2048;
    const int KMAX = 4096;
    const double target_ms = argc > 3 ? atof(argv[3]) : 200.0;
    const int cycles = argc > 4 ? atoi(argv[4]) : 5;

    nvmlDevice_t dev;
    NK(nvmlInit_v2());
    NK(nvmlDeviceGetHandleByIndex_v2(0, &dev));

    float *a, *b, *bias, *c;
    CK(cudaMalloc(&a, (size_t)M * KMAX * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)N * KMAX * sizeof(float)));
    CK(cudaMalloc(&bias, (size_t)N * sizeof(float)));
    CK(cudaMalloc(&c, (size_t)M * N * sizeof(float)));
    CK(cudaMemset(a, 0x3c, (size_t)M * KMAX * sizeof(float)));
    CK(cudaMemset(b, 0x3c, (size_t)N * KMAX * sizeof(float)));
    CK(cudaMemset(bias, 0, (size_t)N * sizeof(float)));

    const dim3 grid((N + PROJ_BN - 1) / PROJ_BN, (M + PROJ_BM - 1) / PROJ_BM);
    const double flops_per_k = 2.0 * (grid.y * PROJ_BM) * (grid.x * PROJ_BN);
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    const int Ks[4] = {448, 896, 1792, 4096};
    // The kernel takes k as both the reduction length and a's row stride, so
    // K=448 reads a 28 MB prefix of the same buffer and K=4096 reads all
    // 255 MB. That footprint difference is inherent to a K sweep and meas-k
    // had it too; what this bench adds is that the two share a thermal state
    // and that the clock is on the record.
    printf("M=%d N=%d grid=(%u,%u)  waves=%.2f\n", M, N, grid.x, grid.y,
           grid.x * grid.y / 164.0);
    printf("%-6s %5s %8s %7s %6s %5s %9s %8s\n", "K", "reps", "ms", "MHz",
           "degC", "W", "flop/cyc", "%peak");
    for (int cyc = 0; cyc < cycles; ++cyc) {
        for (int s = 0; s < 4; ++s) {
            const int K = Ks[s];
            // one warmup + a rep count that makes every measurement the same
            // duration, so neither K heats the part more than the other
            gemm_nt_bias<<<grid, GEMM_THREADS>>>(a, b, bias, c, M, KMAX, N);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(t0));
            gemm_nt_bias<<<grid, GEMM_THREADS>>>(a, b, bias, c, M, K, N);
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float one = 0; CK(cudaEventElapsedTime(&one, t0, t1));
            const int reps = (int)(target_ms / one) + 1;

            CK(cudaEventRecord(t0));
            for (int i = 0; i < reps; ++i)
                gemm_nt_bias<<<grid, GEMM_THREADS>>>(a, b, bias, c, M, K, N);
            unsigned mhz = 0, temp = 0, mw = 0;
            NK(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz));
            NK(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp));
            NK(nvmlDeviceGetPowerUsage(dev, &mw));
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            CK(cudaGetLastError());
            float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
            ms /= reps;
            const double fpc = flops_per_k * K / (ms * 1e-3) / (mhz * 1e6);
            printf("%-6d %5d %8.4f %7u %6u %5.0f %9.1f %8.2f\n", K, reps, ms,
                   mhz, temp, mw / 1000.0, fpc, fpc / 20992.0 * 100.0);
            fflush(stdout);
        }
    }
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(bias)); CK(cudaFree(c));
    NK(nvmlShutdown());
    return 0;
}
