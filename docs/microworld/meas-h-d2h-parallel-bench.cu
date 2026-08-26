// Does the 131.3 MB logits D2H have to cost 17 ms?
//
// to_host() today is one pageable cudaMemcpy: 7.6 GB/s, because the driver
// stages device -> its own pinned bounce buffer -> the user's pageable pages
// with a SINGLE host memcpy. The DMA itself runs at 25.9 GB/s (5.07 ms).
//
// Three ways to attack the gap, in increasing order of what they cost us:
//   B  N host threads each copying a slice   -> needs nothing outside the run
//   S  fixed pinned staging + threaded memcpy -> needs a batch-independent
//                                                buffer allocated once
//   P  pinned destination                     -> needs tensor.h + a
//                                                batch-sized buffer allocated once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>
#include <thread>
#include <algorithm>

static double now_sec() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

static const std::size_t N = 1024ull * 32064ull;         // batch x vocab
static const std::size_t BYTES = N * sizeof(float);

static void report(const char* tag, double sec) {
    std::printf("  %-42s %7.3f ms  (%5.2f GB/s)\n", tag, sec * 1e3,
                BYTES / sec / 1e9);
}

int main() {
    float* d = nullptr;
    CK(cudaMalloc(&d, BYTES));
    CK(cudaMemset(d, 0, BYTES));

    // The destination is always plain pageable host memory -- the same thing
    // Tensor's std::vector gives us -- except in variant P.
    std::vector<float> host(N, 0.0f);

    std::printf("A: one pageable cudaMemcpy (what to_host does today)\n");
    for (int r = 0; r < 3; ++r) {
        const double s = now_sec();
        CK(cudaMemcpy(host.data(), d, BYTES, cudaMemcpyDeviceToHost));
        report("pageable, 1 thread", now_sec() - s);
    }

    // B: split the copy across host threads. Each thread gets its own stream,
    // so the driver has no reason to serialise them on the legacy default one.
    std::printf("B: pageable, split across host threads (own stream each)\n");
    for (int nt : {2, 4, 8, 16}) {
        for (int r = 0; r < 2; ++r) {
            std::vector<cudaStream_t> st(nt);
            for (int i = 0; i < nt; ++i)
                CK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
            const std::size_t chunk = (N + nt - 1) / nt;
            const double s = now_sec();
            std::vector<std::thread> th;
            for (int i = 0; i < nt; ++i) th.emplace_back([&, i] {
                const std::size_t off = i * chunk;
                const std::size_t len = std::min(chunk, N - off);
                CK(cudaMemcpyAsync(host.data() + off, d + off,
                                   len * sizeof(float),
                                   cudaMemcpyDeviceToHost, st[i]));
                CK(cudaStreamSynchronize(st[i]));
            });
            for (auto& t : th) t.join();
            const double e = now_sec() - s;
            char tag[64]; std::snprintf(tag, sizeof tag, "pageable, %d threads", nt);
            report(tag, e);
            for (int i = 0; i < nt; ++i) CK(cudaStreamDestroy(st[i]));
        }
    }

    // S: one small pinned staging buffer, reused. DMA into staging chunk i+1
    // while host threads copy staging chunk i out to the pageable destination.
    // The staging size does not depend on the batch, so it can be allocated
    // once at construction without hardcoding anything about the input.
    std::printf("S: fixed pinned staging (double-buffered) + threaded memcpy\n");
    for (std::size_t stage_mb : {16ull, 32ull, 64ull}) {
        const std::size_t stage = stage_mb << 20;           // bytes per half
        float* pin = nullptr;
        CK(cudaHostAlloc(&pin, 2 * stage, cudaHostAllocDefault));
        cudaStream_t cs; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        for (int nt : {4, 8}) {
            for (int r = 0; r < 2; ++r) {
                const double s = now_sec();
                std::size_t done = 0;
                int slot = 0;
                std::vector<std::thread> prev;
                while (done < BYTES) {
                    const std::size_t len = std::min(stage, BYTES - done);
                    CK(cudaMemcpyAsync(pin + slot * (stage / 4),
                                       reinterpret_cast<const char*>(d) + done,
                                       len, cudaMemcpyDeviceToHost, cs));
                    CK(cudaStreamSynchronize(cs));
                    for (auto& t : prev) t.join();
                    prev.clear();
                    // Hand the staged chunk to threads and go straight back to
                    // the next DMA; the memcpy rides alongside it.
                    char* dst = reinterpret_cast<char*>(host.data()) + done;
                    const char* src = reinterpret_cast<const char*>(pin + slot * (stage / 4));
                    const std::size_t per = (len + nt - 1) / nt;
                    for (int i = 0; i < nt; ++i) prev.emplace_back([=] {
                        const std::size_t o = i * per;
                        if (o < len) std::memcpy(dst + o, src + o, std::min(per, len - o));
                    });
                    done += len; slot ^= 1;
                }
                for (auto& t : prev) t.join();
                char tag[64];
                std::snprintf(tag, sizeof tag, "staging %2zu MB x2, %d memcpy threads",
                              stage_mb, nt);
                report(tag, now_sec() - s);
            }
        }
        CK(cudaStreamDestroy(cs));
        CK(cudaFreeHost(pin));
    }

    // P: the ceiling -- destination itself is pinned, no host memcpy at all.
    // This is what tensor.h would buy, and nothing else can reach it.
    std::printf("P: pinned destination (needs tensor.h)\n");
    float* pinned = nullptr;
    const double a0 = now_sec();
    CK(cudaHostAlloc(&pinned, BYTES, cudaHostAllocDefault));
    std::printf("  (cudaHostAlloc 131.3 MB %7.3f ms -- must be paid out of region)\n",
                (now_sec() - a0) * 1e3);
    for (int r = 0; r < 3; ++r) {
        const double s = now_sec();
        CK(cudaMemcpy(pinned, d, BYTES, cudaMemcpyDeviceToHost));
        report("pinned, 1 thread", now_sec() - s);
    }
    CK(cudaFreeHost(pinned));
    CK(cudaFree(d));
    return 0;
}
