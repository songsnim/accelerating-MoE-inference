// Ceiling probe for the one combined design the guardrail leaves open:
//   pinned staging allocated OUTSIDE the timed region (constructor), lm_head
//   split into row chunks, each chunk's D2H issued on a copy stream as its GEMM
//   retires, and each chunk's staging->Tensor host memcpy handed to a worker
//   thread so it overlaps the next chunk's DMA.
//
// The GEMM is stood in for by a spin kernel of the measured lm_head duration
// (20.3 ms), because what decides the overlap is only its length, not its
// arithmetic. The real chunking penalty is measured separately in the model.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>
#include <algorithm>

static double now_sec() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// Burns a fixed number of clocks so one launch stands in for one lm_head chunk.
__global__ void spin(long long clocks) {
    const long long start = clock64();
    while (clock64() - start < clocks) { }
}

int main(int argc, char** argv) {
    const int chunks = argc > 1 ? std::atoi(argv[1]) : 4;
    const double gemm_ms = argc > 2 ? std::atof(argv[2]) : 20.3;

    const std::size_t rows = 1024, vocab = 32064;
    const std::size_t n = rows * vocab, bytes = n * sizeof(float);
    float* d = nullptr;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemset(d, 0, bytes));

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    const long long total_clocks =
        static_cast<long long>(gemm_ms * 1e-3 * prop.clockRate * 1000.0);

    std::vector<float> tensor(n, 0.0f);          // the Tensor's std::vector
    float* staging = nullptr;
    CK(cudaHostAlloc(&staging, bytes, cudaHostAllocDefault));   // outside region

    // --- Reference: one whole GEMM, then one blocking pageable D2H. ---
    {
        const double s = now_sec();
        spin<<<prop.multiProcessorCount * 2, 256>>>(total_clocks);
        CK(cudaGetLastError());
        CK(cudaMemcpy(tensor.data(), d, bytes, cudaMemcpyDeviceToHost));
        std::printf("baseline tail (gemm %.1f ms + pageable D2H) : %7.3f ms\n",
                    gemm_ms, (now_sec() - s) * 1e3);
    }

    // --- Combined: chunked GEMM + per-chunk async D2H + threaded memcpy. ---
    {
        cudaStream_t copy;
        CK(cudaStreamCreate(&copy));
        std::vector<cudaEvent_t> done(chunks);
        std::vector<std::thread> movers;
        const std::size_t chunk_rows = (rows + chunks - 1) / chunks;

        const double s = now_sec();
        for (int c = 0; c < chunks; ++c) {
            const std::size_t r0 = c * chunk_rows;
            if (r0 >= rows) break;
            const std::size_t nr = std::min(chunk_rows, rows - r0);
            const std::size_t off = r0 * vocab, cb = nr * vocab * sizeof(float);
            spin<<<prop.multiProcessorCount * 2, 256>>>(total_clocks / chunks);
            CK(cudaGetLastError());
            CK(cudaEventCreateWithFlags(&done[c], cudaEventDisableTiming));
            CK(cudaMemcpyAsync(staging + off, d + off, cb,
                               cudaMemcpyDeviceToHost, copy));
            CK(cudaEventRecord(done[c], copy));
            // The staging->Tensor move for this chunk runs while the next
            // chunk's DMA is still on the bus.
            movers.emplace_back([&, c, off, cb] {
                cudaEventSynchronize(done[c]);
                std::memcpy(tensor.data() + off, staging + off, cb);
            });
        }
        for (auto& t : movers) t.join();
        const double e = now_sec();
        std::printf("combined tail (%d chunks, pinned staging + threaded memcpy): "
                    "%7.3f ms\n", chunks, (e - s) * 1e3);
        for (auto ev : done) if (ev) cudaEventDestroy(ev);
        CK(cudaStreamDestroy(copy));
    }

    CK(cudaFreeHost(staging));
    CK(cudaFree(d));
    return 0;
}
