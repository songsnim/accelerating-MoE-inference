// Standalone PCIe probe: what does the 131.3 MB logits D2H actually cost on
// this node, pageable vs pinned, and what does pinning that buffer cost?
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstring>

static double now_sec() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

int main() {
    const std::size_t n = 1024ull * 32064ull;      // batch x vocab
    const std::size_t bytes = n * sizeof(float);
    float* d = nullptr;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemset(d, 0, bytes));

    // Pageable: exactly what to_host() does today (std::vector storage).
    std::vector<float> pageable(n, 0.0f);
    for (int i = 0; i < 3; ++i) {
        const double s = now_sec();
        CK(cudaMemcpy(pageable.data(), d, bytes, cudaMemcpyDeviceToHost));
        std::printf("pageable D2H  %7.3f ms  (%5.2f GB/s)\n",
                    (now_sec() - s) * 1e3, bytes / (now_sec() - s) / 1e9);
    }

    // Pinning an existing std::vector allocation in place -- the only option
    // open to us, since tensor.h is off limits.
    const double r0 = now_sec();
    CK(cudaHostRegister(pageable.data(), bytes, cudaHostRegisterDefault));
    std::printf("cudaHostRegister 131.3 MB  %7.3f ms\n", (now_sec() - r0) * 1e3);
    for (int i = 0; i < 3; ++i) {
        const double s = now_sec();
        CK(cudaMemcpy(pageable.data(), d, bytes, cudaMemcpyDeviceToHost));
        std::printf("pinned   D2H  %7.3f ms  (%5.2f GB/s)\n",
                    (now_sec() - s) * 1e3, bytes / (now_sec() - s) / 1e9);
    }
    const double u0 = now_sec();
    CK(cudaHostUnregister(pageable.data()));
    std::printf("cudaHostUnregister         %7.3f ms\n", (now_sec() - u0) * 1e3);

    // Staging alternative: pinned buffer allocated outside the timed region,
    // then a host memcpy into the Tensor. This is the only design the guardrail
    // permits without touching tensor.h.
    float* staging = nullptr;
    const double a0 = now_sec();
    CK(cudaHostAlloc(&staging, bytes, cudaHostAllocDefault));
    std::printf("cudaHostAlloc 131.3 MB     %7.3f ms\n", (now_sec() - a0) * 1e3);
    const double s0 = now_sec();
    CK(cudaMemcpy(staging, d, bytes, cudaMemcpyDeviceToHost));
    const double s1 = now_sec();
    std::memcpy(pageable.data(), staging, bytes);
    const double s2 = now_sec();
    std::printf("staging: D2H %7.3f ms + host memcpy %7.3f ms = %7.3f ms\n",
                (s1 - s0) * 1e3, (s2 - s1) * 1e3, (s2 - s0) * 1e3);
    CK(cudaFreeHost(staging));
    CK(cudaFree(d));
    return 0;
}
