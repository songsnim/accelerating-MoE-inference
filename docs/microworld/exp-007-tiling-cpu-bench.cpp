// Measures what cache blocking is worth on this login node's CPU, so the CPU
// half of exp-007-tiling.html can quote numbers instead of asserting them.
//
// Single thread on purpose: tiling is a memory-hierarchy story, and threads
// would let core count hide the cliff we are trying to show.
//   g++ -O3 -march=native -std=c++17 -o outputs/exp007cpu \
//       docs/microworld/exp-007-tiling-cpu-bench.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

static double now_sec() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

// C = A*B, both row-major. j innermost over B's row => unit stride.
static void ikj(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const float a = A[i * n + k];
            const float* b = B + k * n;
            float* c = C + i * n;
            for (int j = 0; j < n; ++j) c[j] += a * b[j];
        }
}

// The textbook order. B is walked down a column: one cache line touched per
// element, and by the time the next i needs that line it is long evicted.
static void ijk(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

// Same FMAs as ikj, reordered so a bs x bs block of B is reused bs times while
// it is still in L1/L2.
static void tiled(const float* A, const float* B, float* C, int n, int bs) {
    for (int i0 = 0; i0 < n; i0 += bs)
        for (int k0 = 0; k0 < n; k0 += bs)
            for (int j0 = 0; j0 < n; j0 += bs) {
                const int iM = std::min(i0 + bs, n);
                const int kM = std::min(k0 + bs, n);
                const int jM = std::min(j0 + bs, n);
                for (int i = i0; i < iM; ++i)
                    for (int k = k0; k < kM; ++k) {
                        const float a = A[i * n + k];
                        const float* b = B + k * n;
                        float* c = C + i * n;
                        for (int j = j0; j < jM; ++j) c[j] += a * b[j];
                    }
            }
}

static double checksum(const std::vector<float>& v) {
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * (1.0 + (i & 7));
    return s;
}

int main(int argc, char** argv) {
    const std::vector<int> sizes = (argc > 1) ? std::vector<int>{std::atoi(argv[1])}
                                             : std::vector<int>{256, 512, 1024, 2048};
    const std::vector<int> blocks = {16, 32, 64, 128, 256};

    std::printf("# Xeon E5-2650 @2.00GHz, 1 thread, -O3 -march=native (AVX)\n");
    std::printf("# L1d 32KB/core, L2 256KB/core, L3 20MB/socket\n");
    std::printf("%6s %-14s %10s %9s %10s\n", "n", "variant", "ms", "GFLOP/s", "rel");

    for (int n : sizes) {
        const std::size_t sz = std::size_t(n) * n;
        std::vector<float> A(sz), B(sz), C(sz), ref;
        for (std::size_t i = 0; i < sz; ++i) {
            A[i] = float((i * 37 % 101) - 50) / 50.0f;
            B[i] = float((i * 53 % 97) - 48) / 48.0f;
        }
        const double flop = 2.0 * n * n * double(n);

        auto run = [&](const char* name, auto&& fn, double* base) {
            std::fill(C.begin(), C.end(), 0.0f);
            const double t0 = now_sec();
            fn();
            const double ms = (now_sec() - t0) * 1e3;
            const double gf = flop / (ms * 1e6);
            if (ref.empty()) ref = C;
            else {
                double d = 0.0;
                for (std::size_t i = 0; i < sz; ++i) d = std::max(d, std::abs(double(C[i] - ref[i])));
                if (d > 1e-2) std::printf("  !! %s mismatch %.3g\n", name, d);
            }
            if (*base == 0.0) *base = ms;
            std::printf("%6d %-14s %10.1f %9.2f %9.2fx\n", n, name, ms, gf, *base / ms);
            std::fflush(stdout);
        };

        double base = 0.0;
        // ijk first so it sets the baseline every size.
        if (n <= 1024) run("naive-ijk", [&] { ijk(A.data(), B.data(), C.data(), n); }, &base);
        else { // ijk at 2048 costs minutes; time one i-slab and scale it.
            const double t0 = now_sec();
            for (int i = 0; i < 64; ++i)
                for (int j = 0; j < n; ++j) {
                    float s = 0.0f;
                    for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
                    C[i * n + j] = s;
                }
            base = (now_sec() - t0) * 1e3 * (double(n) / 64.0);
            std::printf("%6d %-14s %10.1f %9.2f %9.2fx  (64-row slab x%d)\n",
                        n, "naive-ijk*", base, flop / (base * 1e6), 1.0, n / 64);
        }
        run("reorder-ikj", [&] { ikj(A.data(), B.data(), C.data(), n); }, &base);
        for (int bs : blocks)
            if (bs <= n) {
                char nm[32];
                std::snprintf(nm, sizeof nm, "tiled-%d", bs);
                run(nm, [&] { tiled(A.data(), B.data(), C.data(), n, bs); }, &base);
            }
        std::printf("%6d checksum %.6e\n\n", n, checksum(ref));
    }
    return 0;
}
