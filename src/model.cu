#include "model.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

double now_sec() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return static_cast<double>(t.tv_sec) + 1.0e-9 * static_cast<double>(t.tv_nsec);
}

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
    }
}

float* device_copy(const Tensor& host) {
    if (host.size() == 0) return nullptr;
    float* device = nullptr;
    const std::size_t bytes = host.size() * sizeof(float);
    cuda_check(cudaMalloc(&device, bytes), "cudaMalloc weight");
    cuda_check(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy weight");
    return device;
}

// Device scratch owned by the model, not by the batch: the frees used to run
// inside the measured region (24 calls, ~1.5 GB, 20.8 ms of GPU-idle
// teardown). The first reserve still allocates where the old constructor did.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    // Grows only: a batch needing more reallocates, a smaller one reuses.
    // Every buffer is fully written before it is read, so stale contents from
    // an earlier call are never observed.
    T* reserve(std::size_t elements) {
        if (elements > capacity_) {
            cuda_check(cudaFree(data_), "cudaFree scratch");
            cuda_check(cudaMalloc(&data_, elements * sizeof(T)),
                       "cudaMalloc scratch");
            capacity_ = elements;
        }
        return data_;
    }

private:
    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

void to_host(Tensor& host, const float* device) {
    cuda_check(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

// C[M,N] = A[M,K] * B[N,K]^T + bias[N], register-tiled through shared memory.
// One block computes a BM x BN output tile; one thread keeps TM x TN outputs in
// registers, so the inner loop reads TM + TN floats from shared per TM * TN
// MACs instead of two per MAC.
//
// The K loop still walks k in ascending order and each output still accumulates
// into one register with the same `acc += a * b` expression, so every output
// sees the same sequence of FMAs as the tile kernel it replaces.
// BK is 16, the widest k window that still fits two blocks per SM: the two
// double-buffered tiles are 33,792 B, so BK = 32 would need 135 KB per SM
// against the 100 KB limit. Widening it does not change the FFMA-per-shared-load
// ratio -- it halves the barriers and the global load instructions, which is
// where the K loop was losing issue slots. The `__launch_bounds__(_, 2)` on the
// two kernels below holds that 2-block residency: without it ptxas gives
// `gemm_grouped` 141 registers and residency drops to one block. It spills
// nothing at 128.
// The body below is a template on the output tile so the projections and the
// grouped expert GEMM can carry different tiles. Both are 128x128 / 8x8 today,
// so the split is a no-op; EXP-022 moves the projections alone. `BK` and the
// block size stay common to both by design.
constexpr int BK = 16;
constexpr int GEMM_THREADS = 256;
constexpr int PROJ_BM = 128, PROJ_BN = 128, PROJ_TM = 8, PROJ_TN = 8;
constexpr int GRP_BM = 128, GRP_BN = 128, GRP_TM = 8, GRP_TN = 8;
static_assert((PROJ_BM / PROJ_TM) * (PROJ_BN / PROJ_TN) == GEMM_THREADS,
              "projection tile does not fill the block");
static_assert((GRP_BM / GRP_TM) * (GRP_BN / GRP_TN) == GEMM_THREADS,
              "grouped tile does not fill the block");
// The +4 keeps the shared rows 16B-aligned while breaking the bank-conflict
// pattern of a bare [BK][BM] layout.
constexpr int SPAD = 4;

// Column that lane tx owns at slot j. The natural `tx * TN + j` gives a warp's
// quarter (8 lanes) word-stride 8, so those 8 lanes touch banks
// {0-3, 8-11, 16-19, 24-27} twice over -- a 2-way conflict on every `bs` read,
// measured at 5.0 shared-load wavefronts per instruction against an ideal 4
// (2.0 conflict wavefronts, `as` clean). Splitting each lane's 8 columns into
// two float4 halves half a tile apart makes the same 8 lanes cover all 32 banks
// exactly once. Only which thread owns which output column changes; every
// output still accumulates the same k-ascending FMA sequence.
template <int BN>
__device__ __forceinline__ int gemm_col_slot(int tx, int j) {
    return tx * 4 + (j >> 2) * (BN / 2) + (j & 3);
}

// Double-buffered variant: the global load for tile kt+BK is issued before the
// compute over tile kt, so the ~500-cycle LDG latency overlaps the FFMAs
// instead of stalling in front of the shared store. Only one __syncthreads()
// per iteration is needed because the two buffers alternate.
//
// The k loop still walks ascending and each output still accumulates into one
// register with the same `acc += a * b`, so the FMA sequence is unchanged.
// `row0` is the first row of this block's tile inside `a`/`c` and `m` the row
// past its last; everything else is the tile kernel unchanged, down to the
// bound expressions -- writing them any other way moves ptxas off the
// instruction schedule the projections were measured on.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void gemm_nt_body(const float* __restrict__ a,
                                             const float* __restrict__ b,
                                             const float* __restrict__ bias,
                                             float* __restrict__ c,
                                             int row0, int m, int k, int n) {
    // Each thread stages two float4 per matrix; that is what makes `lr`/`lr2`
    // cover the tile exactly once. A wider tile has to change the staging.
    static_assert(BM / 2 * (BK / 4) == GEMM_THREADS, "A staging misses rows");
    static_assert(BN / 2 * (BK / 4) == GEMM_THREADS, "B staging misses rows");

    __shared__ __align__(16) float as[2][BK][BM + SPAD];
    __shared__ __align__(16) float bs[2][BK][BN + SPAD];

    const int tid = threadIdx.x;
    const int m0 = row0;
    const int n0 = blockIdx.x * BN;
    // BK = 16 needs 2 float4 per thread per matrix, so each thread stages two
    // rows half a tile apart instead of one.
    const int lr = tid / (BK / 4);          // 0..63, row inside the tile
    const int lc = (tid % (BK / 4)) * 4;    // 0, 4, 8, 12 inside the k window
    const int lr2 = lr + BM / 2;            // the second row this thread stages
    const int ty = tid / (BN / TN);         // 0..15
    const int tx = tid % (BN / TN);         // 0..15

    const int a_row = m0 + lr, a_row2 = m0 + lr2;
    const int b_row = n0 + lr, b_row2 = n0 + lr2;
    const bool a_ok = a_row < m, a_ok2 = a_row2 < m;
    const bool b_ok = b_row < n, b_ok2 = b_row2 < n;
    const float* const ab = a + static_cast<long long>(a_row) * k + lc;
    const float* const ab2 = a + static_cast<long long>(a_row2) * k + lc;
    const float* const bb = b + static_cast<long long>(b_row) * k + lc;
    const float* const bb2 = b + static_cast<long long>(b_row2) * k + lc;

    float acc[TM][TN] = {};

    // Stage the first tile. Rows past the edge stay zero for the whole loop.
    float4 av = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 av2 = av, bv = av, bv2 = av;
    if (a_ok) av = *reinterpret_cast<const float4*>(ab);
    if (a_ok2) av2 = *reinterpret_cast<const float4*>(ab2);
    if (b_ok) bv = *reinterpret_cast<const float4*>(bb);
    if (b_ok2) bv2 = *reinterpret_cast<const float4*>(bb2);
    as[0][lc + 0][lr] = av.x; as[0][lc + 1][lr] = av.y;
    as[0][lc + 2][lr] = av.z; as[0][lc + 3][lr] = av.w;
    as[0][lc + 0][lr2] = av2.x; as[0][lc + 1][lr2] = av2.y;
    as[0][lc + 2][lr2] = av2.z; as[0][lc + 3][lr2] = av2.w;
    bs[0][lc + 0][lr] = bv.x; bs[0][lc + 1][lr] = bv.y;
    bs[0][lc + 2][lr] = bv.z; bs[0][lc + 3][lr] = bv.w;
    bs[0][lc + 0][lr2] = bv2.x; bs[0][lc + 1][lr2] = bv2.y;
    bs[0][lc + 2][lr2] = bv2.z; bs[0][lc + 3][lr2] = bv2.w;
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
                const float4 t = *reinterpret_cast<const float4*>(&ac[p][ty * TM + i]);
                av4[i] = t.x; av4[i + 1] = t.y; av4[i + 2] = t.z; av4[i + 3] = t.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 t =
                    *reinterpret_cast<const float4*>(&bc[p][gemm_col_slot<BN>(tx, j)]);
                bv4[j] = t.x; bv4[j + 1] = t.y; bv4[j + 2] = t.z; bv4[j + 3] = t.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av4[i] * bv4[j];
        }

        if (more) {
            // Writing the other buffer: every thread finished reading it before
            // the __syncthreads() that closed the previous iteration.
            const int nxt = cur ^ 1;
            as[nxt][lc + 0][lr] = av.x; as[nxt][lc + 1][lr] = av.y;
            as[nxt][lc + 2][lr] = av.z; as[nxt][lc + 3][lr] = av.w;
            as[nxt][lc + 0][lr2] = av2.x; as[nxt][lc + 1][lr2] = av2.y;
            as[nxt][lc + 2][lr2] = av2.z; as[nxt][lc + 3][lr2] = av2.w;
            bs[nxt][lc + 0][lr] = bv.x; bs[nxt][lc + 1][lr] = bv.y;
            bs[nxt][lc + 2][lr] = bv.z; bs[nxt][lc + 3][lr] = bv.w;
            bs[nxt][lc + 0][lr2] = bv2.x; bs[nxt][lc + 1][lr2] = bv2.y;
            bs[nxt][lc + 2][lr2] = bv2.z; bs[nxt][lc + 3][lr2] = bv2.w;
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
            const int col = n0 + gemm_col_slot<BN>(tx, j);
            if (col >= n) continue;
            float v = acc[i][j];
            if (bias != nullptr) v += bias[col];
            c[static_cast<long long>(row) * n + col] = v;
        }
    }
}

__global__ __launch_bounds__(GEMM_THREADS, 2)
void gemm_nt_bias(const float* __restrict__ a,
                  const float* __restrict__ b,
                  const float* __restrict__ bias,
                  float* __restrict__ c,
                  int m, int k, int n) {
    gemm_nt_body<PROJ_BM, PROJ_BN, PROJ_TM, PROJ_TN>(
        a, b, bias, c, blockIdx.y * PROJ_BM, m, k, n);
}

// One launch for all 16 experts of a layer. `tiles` holds three ints per row
// tile -- expert, first row inside the packed activation buffer, rows left --
// so blocks from different experts share one grid and one wave. Per expert the
// grid was only (n/BN) x rowtiles, which left the 82 SMs a third idle (nsys:
// 66% time-weighted wave fill on w13, 53% of its time in sub-one-wave
// launches). Nothing about a row's arithmetic changes: the tile it lands in
// does not enter the accumulation.
__global__ __launch_bounds__(GEMM_THREADS, 2)
void gemm_grouped(const float* __restrict__ a,
                             const float* const* __restrict__ weights,
                             float* __restrict__ c,
                             const int* __restrict__ tiles, int k, int n) {
    const int* const t = tiles + blockIdx.y * 3;
    // The grid is sized to the worst-case tile count, so the tail is empty.
    if (t[2] == 0) return;
    gemm_nt_body<GRP_BM, GRP_BN, GRP_TM, GRP_TN>(
        a, weights[t[0]], nullptr, c, t[1], t[1] + t[2], k, n);
}

// dst[i, :] = src[index[i], :]
__global__ void gather_rows(const float* __restrict__ src,
                            const int* __restrict__ index,
                            float* __restrict__ dst, int cols) {
    const int row = index[blockIdx.x];
    const float* in = src + static_cast<long long>(row) * cols;
    float* out = dst + static_cast<long long>(blockIdx.x) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = in[c];
}

// dst[t, :] = 0.5 * src[slot_lo(t), :] + 0.5 * src[slot_hi(t), :]
// One block per row. The two halves are added low expert index first and
// through an accumulator that starts at zero, which is exactly what the
// zero-fill plus one launch per expert did before.
__global__ void moe_combine(const float* __restrict__ src,
                            const int* __restrict__ slot,
                            float* __restrict__ dst, int cols) {
    const int t = blockIdx.x;
    const float* lo = src + static_cast<long long>(slot[2 * t]) * cols;
    const float* hi = src + static_cast<long long>(slot[2 * t + 1]) * cols;
    float* out = dst + static_cast<long long>(t) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float acc = 0.0f;
        acc += 0.5f * lo[c];
        acc += 0.5f * hi[c];
        out[c] = acc;
    }
}

// PhiMLP applies silu to the w1 branch and multiplies the w3 branch into it.
// tensor.cu evaluates exp in double; expf differs by ~1e-7 relative, which is
// four orders under the 3e-3 validation threshold.
// gate_up holds one stacked GEMM's output, [rows, 2*f]: the w1 half then the
// w3 half of the same row. n = rows * f.
__global__ void silu_mul(const float* __restrict__ gate_up,
                         float* __restrict__ out, int f, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const long long row = i / f, c = i % f;
        const float* gu = gate_up + row * 2 * f;
        const float x = gu[c];
        out[i] = (x / (1.0f + expf(-x))) * gu[f + c];
    }
}

// a[i] += b[i]
__global__ void add_inplace_kernel(float* __restrict__ a,
                                   const float* __restrict__ b, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

void add_inplace_device(float* a, const float* b, long long n) {
    add_inplace_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(a, b, n);
    cuda_check(cudaGetLastError(), "add_inplace_kernel launch");
}

// One block per row. The row is staged in shared memory, then a single thread
// walks it twice.
//
// The serial walk is deliberate. tensor_ops::layer_norm accumulates each row
// sequentially into one float accumulator (verified in the disassembly: a
// chain of vaddss, no vectorisation and no FMA), and a block-wide tree
// reduction sums in a different order. That reordering shifts the router
// logits by ~1e-7, which is enough to flip a token across the score
// quantisation boundary in PhiMoE::route and hand it a different expert -- a
// discrete O(0.1) change in that token's output. Measured at n=1024: a tree
// reduction failed validation on 4 of 1024 sequences (max abs diff 0.212)
// while the mean stayed at 6.8e-05.
//
// So every arithmetic step below mirrors the host op exactly, including the
// __f*_rn intrinsics that keep nvcc from contracting the multiply-adds into
// FMAs the host does not emit. One serial thread per row costs ~70 ms in
// total, against the 16.1 s the host layer norms took.
// Rows per warp: one lane owns one row, so the width is the warp's.
constexpr int NORM_ROWS = 32;
// Columns staged per trip, and how many of the 32 rows are staged with their
// loads batched: the staging is a load-to-shared pair per row, so ptxas has to
// see a compile-time row count or every pair pays a full global latency.
// 64 columns per trip keeps a block at 8.4 KB of shared; 16 rows per batch
// puts 32 loads (4 KB) in flight per warp, which is what it takes to saturate
// DRAM from the 5.9 warps per SM this mapping leaves resident. Measured:
// (64,16) and (32,16) tie at 0.083, (64,8) 0.089, (128,4) 0.159.
constexpr int NORM_CHUNK = 64;
constexpr int NORM_GROUP = 16;
constexpr int NORM_PER_ROW = NORM_CHUNK / NORM_ROWS;
// Padded row stride: the chain reads 32 consecutive floats (one wavefront) and
// the staging writes land on 32 distinct banks.
constexpr int NORM_STRIDE = NORM_ROWS + 1;

// One lane per row. Each row keeps its own strictly j-ascending accumulation,
// so the result is bit-identical to the one-thread-per-block version this
// replaces; what changes is that every row's chain now runs at once instead of
// five per SM (16 KB of staged row was what capped it). The price is reading x
// three times: a row cannot stay resident across mean, variance and output.
__global__ void layer_norm_rows(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                const float* __restrict__ bias,
                                float* __restrict__ y, int rows, int cols,
                                float eps) {
    __shared__ float s[NORM_CHUNK * NORM_STRIDE];
    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * NORM_ROWS;
    const int rows_here = min(NORM_ROWS, rows - row0);
    const bool mine = lane < rows_here;
    const long long base = static_cast<long long>(row0) * cols;

    // Column-chunk staging: coalesced row by row on the global side,
    // transposed on the shared side so a lane reads down one column.
    // The row count is NORM_ROWS whatever the tail block holds -- an
    // out-of-range row reads row 0 and its column is never read back -- so the
    // loop unrolls and the loads of several rows are in flight at once.
    auto stage = [&](int j0) {
        __syncwarp();
        for (int r0 = 0; r0 < NORM_ROWS; r0 += NORM_GROUP) {
            // Loads of a whole group are issued before the first store, so a
            // group costs one global latency instead of one per row. Left to
            // ptxas the pair serialises and the kernel stalls on
            // long_scoreboard (measured: 24.5 stalls per issue).
            float v[NORM_GROUP][NORM_PER_ROW];
#pragma unroll
            for (int q = 0; q < NORM_GROUP; ++q) {
                const int r = r0 + q;
                const int rr = r < rows_here ? r : 0;
                const float* src =
                    x + base + static_cast<long long>(rr) * cols + j0;
#pragma unroll
                for (int i = 0; i < NORM_PER_ROW; ++i) {
                    v[q][i] = src[i * NORM_ROWS + lane];
                }
            }
#pragma unroll
            for (int q = 0; q < NORM_GROUP; ++q) {
#pragma unroll
                for (int i = 0; i < NORM_PER_ROW; ++i) {
                    s[(i * NORM_ROWS + lane) * NORM_STRIDE + r0 + q] = v[q][i];
                }
            }
        }
        __syncwarp();
    };

    float sum = 0.0f;
    for (int j0 = 0; j0 < cols; j0 += NORM_CHUNK) {
        stage(j0);
        if (mine) {
            for (int i = 0; i < NORM_CHUNK; ++i) {
                sum = __fadd_rn(sum, s[i * NORM_STRIDE + lane]);
            }
        }
    }
    const float mean = sum / static_cast<float>(cols);

    float var = 0.0f;
    for (int j0 = 0; j0 < cols; j0 += NORM_CHUNK) {
        stage(j0);
        if (mine) {
            for (int i = 0; i < NORM_CHUNK; ++i) {
                const float d = __fsub_rn(s[i * NORM_STRIDE + lane], mean);
                var = __fadd_rn(var, __fmul_rn(d, d));
            }
        }
    }
    const float inv = 1.0f / sqrtf(var / static_cast<float>(cols) + eps);

    // The epilogue needs no chain, so it goes back to one row at a time with
    // the whole warp on it: coalesced in and out.
    // float4 because this pass is pure streaming: 32 trips per row instead of
    // 128, and every trip is a full 128 B line in and out.
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);
    for (int r = 0; r < rows_here; ++r) {
        const float m = __shfl_sync(0xffffffffu, mean, r);
        const float iv = __shfl_sync(0xffffffffu, inv, r);
        const long long off = base + static_cast<long long>(r) * cols;
        const float4* src = reinterpret_cast<const float4*>(x + off);
        float4* dst = reinterpret_cast<float4*>(y + off);
        for (int i = lane; i < cols / 4; i += NORM_ROWS) {
            const float4 xv = src[i], wv = w4[i], bv = b4[i];
            float4 o;
            o.x = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.x, m), iv), wv.x), bv.x);
            o.y = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.y, m), iv), wv.y), bv.y);
            o.z = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.z, m), iv), wv.z), bv.z);
            o.w = __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(xv.w, m), iv), wv.w), bv.w);
            dst[i] = o;
        }
    }
}

// Mirrors PhiMoE::route in layer.cu: quantise the router scores, then pick the
// best two with a tie epsilon. The scan is sequential in expert order because
// the tie rule is: on a near tie the lower index wins.
__device__ __forceinline__ void route_top2(const float* __restrict__ logits,
                                           int& first, int& second) {
    float scores[apss26::NUM_EXPERTS];
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        const float score = logits[e];
        const float rounded = floorf(fabsf(score) /
            apss26::ROUTER_SCORE_QUANTUM + 0.5f) * apss26::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    first = -1;
    float best = -INFINITY;
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        if (first < 0 || scores[e] > best + apss26::ROUTER_TIE_EPS) {
            first = e;
            best = scores[e];
        }
    }
    second = -1;
    best = -INFINITY;
    for (int e = 0; e < static_cast<int>(apss26::NUM_EXPERTS); ++e) {
        if (e == first) continue;
        if (second < 0 || scores[e] > best + apss26::ROUTER_TIE_EPS) {
            second = e;
            best = scores[e];
        }
    }
}

// Rows per routing block. The placement pass below walks a block's rows once
// per expert, so this also bounds that scan.
constexpr int ROUTE_BLOCK = 256;

// Pass 1: each row's two experts, stored low index first because the per-expert
// scatter this replaces applied them in ascending expert order. Also each
// block's per-expert row count, the input to the scan.
__global__ void route_count(const float* __restrict__ router, int total,
                            int* __restrict__ pair, int* __restrict__ blk) {
    __shared__ int cnt[apss26::NUM_EXPERTS];
    if (threadIdx.x < apss26::NUM_EXPERTS) cnt[threadIdx.x] = 0;
    __syncthreads();
    const int t = blockIdx.x * ROUTE_BLOCK + threadIdx.x;
    if (t < total) {
        int first = 0, second = 0;
        route_top2(router + static_cast<long long>(t) * apss26::NUM_EXPERTS,
                   first, second);
        const int lo = first < second ? first : second;
        const int hi = first < second ? second : first;
        pair[2 * t] = lo;
        pair[2 * t + 1] = hi;
        atomicAdd(&cnt[lo], 1);
        atomicAdd(&cnt[hi], 1);
    }
    __syncthreads();
    if (threadIdx.x < apss26::NUM_EXPERTS) {
        blk[blockIdx.x * static_cast<int>(apss26::NUM_EXPERTS) + threadIdx.x] =
            cnt[threadIdx.x];
    }
}

// Pass 2: one block, one thread per expert. Turns the counts into each block's
// base inside its expert's packed range and each expert's base inside the
// packed buffer, then writes the tile list. The grid is launched at max_tiles
// so the count never has to come back to the host; the surplus tiles carry
// zero rows and return immediately.
__global__ void route_scan(int* __restrict__ blk, int nrb,
                           int* __restrict__ eoff, int* __restrict__ tiles,
                           int max_tiles) {
    __shared__ int tot[apss26::NUM_EXPERTS];
    const int e = threadIdx.x;
    int run = 0;
    for (int b = 0; b < nrb; ++b) {
        const int at = b * static_cast<int>(apss26::NUM_EXPERTS) + e;
        const int c = blk[at];
        blk[at] = run;
        run += c;
    }
    tot[e] = run;
    __syncthreads();
    if (e != 0) return;
    int off = 0, j = 0;
    for (int x = 0; x < static_cast<int>(apss26::NUM_EXPERTS); ++x) {
        eoff[x] = off;
        for (int r = 0; r < tot[x]; r += GRP_BM) {
            tiles[3 * j] = x;
            tiles[3 * j + 1] = off + r;
            tiles[3 * j + 2] = tot[x] - r < GRP_BM ? tot[x] - r : GRP_BM;
            ++j;
        }
        off += tot[x];
    }
    for (; j < max_tiles; ++j) {
        tiles[3 * j] = 0;
        tiles[3 * j + 1] = 0;
        tiles[3 * j + 2] = 0;
    }
}

// Pass 3: place each row's two copies into its experts' packed ranges. Threads
// walk their expert's rows in ascending row order, so the packed layout is the
// one the host build produced. `slot` records where each copy landed, which is
// what lets the combine below read the two halves back without a scatter.
__global__ void route_place(const int* __restrict__ pair, int total,
                            const int* __restrict__ blk,
                            const int* __restrict__ eoff,
                            int* __restrict__ index, int* __restrict__ slot) {
    __shared__ int sp[2 * ROUTE_BLOCK];
    const int base = blockIdx.x * ROUTE_BLOCK;
    const int n = total - base < ROUTE_BLOCK ? total - base : ROUTE_BLOCK;
    for (int i = threadIdx.x; i < 2 * n; i += ROUTE_BLOCK) sp[i] = pair[2 * base + i];
    __syncthreads();
    if (threadIdx.x >= apss26::NUM_EXPERTS) return;
    const int e = threadIdx.x;
    int at = eoff[e] + blk[blockIdx.x * static_cast<int>(apss26::NUM_EXPERTS) + e];
    for (int i = 0; i < n; ++i) {
        // The two experts of a row are distinct, so at most one arm fires.
        if (sp[2 * i] == e) {
            index[at] = base + i;
            slot[2 * (base + i)] = at;
            ++at;
        } else if (sp[2 * i + 1] == e) {
            index[at] = base + i;
            slot[2 * (base + i) + 1] = at;
            ++at;
        }
    }
}

// RoPE applied in place to the batched q and k projections. One block per row.
//
// The cos/sin table is built by the host inside generate() rather than here:
// device powf/cosf/sinf differ from glibc by up to an ulp, and a 1e-7 shift in
// the rotated q/k is exactly the magnitude that moves a router logit across the
// quantisation boundary in PhiMoE::route (see layer_norm_rows). The table is
// 64 * max_len * 2 floats, so building it costs microseconds.
__global__ void rope_rows(float* __restrict__ q, float* __restrict__ k,
                          const int* __restrict__ pos,
                          const float* __restrict__ table) {
    constexpr int D = static_cast<int>(apss26::HEAD_DIM);
    constexpr int HALF = D / 2;
    constexpr int QH = static_cast<int>(apss26::NUM_ATTENTION_HEADS);
    constexpr int KVH = static_cast<int>(apss26::NUM_KV_HEADS);

    const int row = blockIdx.x;
    const float* trig = table + static_cast<long long>(pos[row]) * HALF * 2;

    // The host writes both halves of a pair from the pre-rotation values, so
    // one thread owns one pair and no two threads touch the same element.
    float* qrow = q + static_cast<long long>(row) * QH * D;
    for (int i = threadIdx.x; i < QH * HALF; i += blockDim.x) {
        float* r = qrow + (i / HALF) * D;
        const int j = i % HALF;
        const float c = trig[j * 2], sn = trig[j * 2 + 1];
        const float x0 = r[j], x1 = r[j + HALF];
        r[j] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, sn));
        r[j + HALF] = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, sn));
    }
    float* krow = k + static_cast<long long>(row) * KVH * D;
    for (int i = threadIdx.x; i < KVH * HALF; i += blockDim.x) {
        float* r = krow + (i / HALF) * D;
        const int j = i % HALF;
        const float c = trig[j * 2], sn = trig[j * 2 + 1];
        const float x0 = r[j], x1 = r[j + HALF];
        r[j] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, sn));
        r[j + HALF] = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, sn));
    }
}

// Key columns staged per pass. The +1 pad below breaks the shared bank
// pattern: with stride 33 lane i lands on bank (i + d) % 32.
constexpr int ATTN_CHUNK = 32;

// Causal attention over the prefix trie. One block per (node, query head),
// with HEAD_DIM threads: thread d owns output element d. A node's key set is
// its own root path, which `anc` lists ascending by depth.
//
// Every accumulation order matches the sequential reference, which is the whole
// difficulty of this kernel. A reordered sum here shifts the router logits by
// ~1e-7 and flips tokens to other experts, so:
//   - the q.k dot walks d ascending inside a single thread,
//   - the softmax denominator walks ki ascending inside a single thread
//     (the exp itself is elementwise, so every lane may run one),
//   - the value accumulation walks ki ascending inside thread d,
//   - __f*_rn keeps nvcc from contracting the multiply-adds into FMAs that the
//     host does not emit (verified in obj/model.o: vmulss + vaddss throughout),
//   - exp stays in double, as accurate_exp did.
__global__ void attention_heads(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                const int* __restrict__ anc,
                                const int* __restrict__ anc_off,
                                float* __restrict__ out, float scale) {
    constexpr int D = static_cast<int>(apss26::HEAD_DIM);
    constexpr int QH = static_cast<int>(apss26::NUM_ATTENTION_HEADS);
    constexpr int KVH = static_cast<int>(apss26::NUM_KV_HEADS);
    constexpr int WINDOW = static_cast<int>(apss26::SLIDING_WINDOW);

    extern __shared__ float shared[];
    float* sq = shared;               // the query row being served
    float* sdenom = shared + D;       // the softmax denominator
    float* sw = shared + D + 1;       // one score per key
    __shared__ float smax;            // the softmax max, broadcast to all lanes

    const int node = blockIdx.x, qh = blockIdx.y;
    const int kh = qh / (QH / KVH);
    const int tid = threadIdx.x;

    const int abase = anc_off[node];
    const int keys = anc_off[node + 1] - abase;   // = position + 1
    const int lo = keys > WINDOW ? keys - WINDOW : 0;
    const int nk = keys - lo;
    const int* const chain = anc + abase + lo;

    // The q.k dot reads k straight from global with one key row per lane, so a
    // warp touches 32 different rows and each 32 B sector delivers 4 useful
    // bytes (measured: 22.6% of fetched bytes used, L1 at 87.6% -- the wasted
    // resource is the saturated one). Staging k through shared in ATTN_CHUNK
    // column slices makes the global read row-contiguous. The dot still walks d
    // ascending inside one thread; the per-chunk partial lives in shared as the
    // same fp32 it would have held in a register, so the __fadd_rn sequence per
    // score is unchanged. A narrow chunk is the point: whole-row staging costs
    // max_len * D floats and loses more to occupancy than coalescing returns.
    float* const sk = sw + nk;    // <= the max_len * (ATTN_CHUNK + 1) reserved

    sq[tid] = q[static_cast<long long>(node) * QH * D + qh * D + tid];
    for (int i = tid; i < nk; i += D) sw[i] = 0.0f;
    __syncthreads();

    for (int d0 = 0; d0 < D; d0 += ATTN_CHUNK) {
        for (int base = 0; base < nk; base += D / ATTN_CHUNK) {
            const int ki = base + tid / ATTN_CHUNK, col = tid % ATTN_CHUNK;
            if (ki < nk)
                sk[ki * (ATTN_CHUNK + 1) + col] =
                    k[static_cast<long long>(chain[ki]) * KVH * D + kh * D + d0 + col];
        }
        __syncthreads();
        for (int i = tid; i < nk; i += D) {
            const float* kr = sk + i * (ATTN_CHUNK + 1);
            float score = sw[i];
            for (int d = 0; d < ATTN_CHUNK; ++d)
                score = __fadd_rn(score, __fmul_rn(sq[d0 + d], kr[d]));
            sw[i] = score;
        }
        __syncthreads();
    }
    for (int i = tid; i < nk; i += D) sw[i] = sw[i] / scale;
    __syncthreads();

    // exp is elementwise, so it may leave thread 0 without disturbing any
    // summation order; the max scan and the denominator stay where they are.
    if (tid == 0) {
        float maxv = -INFINITY;
        for (int i = 0; i < nk; ++i) maxv = fmaxf(maxv, sw[i]);
        smax = maxv;
    }
    __syncthreads();
    for (int i = tid; i < nk; i += D)
        sw[i] = static_cast<float>(exp(static_cast<double>(sw[i] - smax)));
    __syncthreads();
    if (tid == 0) {
        float denom = 0.0f;
        for (int i = 0; i < nk; ++i) denom = __fadd_rn(denom, sw[i]);
        *sdenom = denom;
    }
    __syncthreads();

    const float denom = *sdenom;
    float acc = 0.0f;
    for (int i = 0; i < nk; ++i) {
        const float w = sw[i] / denom;
        acc = __fadd_rn(acc, __fmul_rn(w,
            v[static_cast<long long>(chain[i]) * KVH * D + kh * D + tid]));
    }
    out[static_cast<long long>(node) * QH * D + qh * D + tid] = acc;
}

}  // namespace

// generate()'s working set, in one place so the model can hold it.
struct PhiTinyMoEModel::Scratch {
    DeviceBuffer<float> stream_a, stream_b, norm, attn, q, k, v, ctx, router;
    DeviceBuffer<float> expert_in, expert_out, gate, gate_up, rope, logits;
    DeviceBuffer<int> index, pair, slot, blk, eoff, tiles, pos, anc, anc_off;
};

PhiTinyMoEModel::DeviceLinear::DeviceLinear(const ModelLoader& loader,
                                            const std::string& weight_name,
                                            const std::string& bias_name) {
    {
        const Tensor host = loader.load(weight_name);
        out = host.size(0);
        in = host.size(1);
        weight = device_copy(host);
    }
    if (!bias_name.empty() && loader.has(bias_name)) {
        bias = device_copy(loader.load(bias_name));
    }
}

PhiTinyMoEModel::DeviceLinear::DeviceLinear(const ModelLoader& loader,
                                            const std::string& top,
                                            const std::string& bottom, bool) {
    const Tensor a = loader.load(top);
    const Tensor b = loader.load(bottom);
    out = a.size(0) + b.size(0);
    in = a.size(1);
    cuda_check(cudaMalloc(&weight, out * in * sizeof(float)),
               "cudaMalloc stacked weight");
    cuda_check(cudaMemcpy(weight, a.data(), a.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy stacked weight top");
    cuda_check(cudaMemcpy(weight + a.size(), b.data(), b.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy stacked weight bottom");
}

PhiTinyMoEModel::DeviceLinear::DeviceLinear(DeviceLinear&& other) noexcept
    : weight(other.weight), bias(other.bias), out(other.out), in(other.in) {
    other.weight = nullptr;
    other.bias = nullptr;
}

void PhiTinyMoEModel::DeviceLinear::free() {
    cudaFree(weight);
    cudaFree(bias);
    weight = nullptr;
    bias = nullptr;
}

void PhiTinyMoEModel::DeviceLinear::forward(const float* x, float* y,
                                            std::size_t rows) const {
    // The kernel loads the K window as float4 without a k bound check; every
    // shape in this model has K divisible by BK.
    if (in % BK != 0) throw std::runtime_error("DeviceLinear: K not a multiple of BK");
    const dim3 grid(static_cast<unsigned>((out + PROJ_BN - 1) / PROJ_BN),
                    static_cast<unsigned>((rows + PROJ_BM - 1) / PROJ_BM));
    gemm_nt_bias<<<grid, GEMM_THREADS>>>(x, weight, bias, y,
                                         static_cast<int>(rows),
                                         static_cast<int>(in),
                                         static_cast<int>(out));
    cuda_check(cudaGetLastError(), "gemm_nt_bias launch");
}

PhiTinyMoEModel::DeviceNorm::DeviceNorm(const ModelLoader& loader,
                                        const std::string& weight_name,
                                        const std::string& bias_name) {
    const Tensor host_weight = loader.load(weight_name);
    cols = host_weight.size();
    weight = device_copy(host_weight);
    bias = device_copy(loader.load(bias_name));
}

PhiTinyMoEModel::DeviceNorm::DeviceNorm(DeviceNorm&& other) noexcept
    : weight(other.weight), bias(other.bias), cols(other.cols) {
    other.weight = nullptr;
    other.bias = nullptr;
}

void PhiTinyMoEModel::DeviceNorm::free() {
    cudaFree(weight);
    cudaFree(bias);
    weight = nullptr;
    bias = nullptr;
}

void PhiTinyMoEModel::DeviceNorm::forward(const float* x, float* y,
                                          std::size_t rows) const {
    if (cols % NORM_CHUNK != 0) {
        throw std::runtime_error("layer norm width is not a multiple of the staging chunk");
    }
    const unsigned blocks =
        static_cast<unsigned>((rows + NORM_ROWS - 1) / NORM_ROWS);
    layer_norm_rows<<<blocks, NORM_ROWS>>>(x, weight, bias, y,
                                           static_cast<int>(rows),
                                           static_cast<int>(cols),
                                           apss26::NORM_EPS);
    cuda_check(cudaGetLastError(), "layer_norm_rows launch");
}

PhiTinyMoEModel::DeviceMoE::DeviceMoE(const ModelLoader& loader, std::size_t layer_idx)
    : gate(loader, "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe.gate.weight", "") {
    const std::string base = "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe";
    w13.reserve(apss26::NUM_EXPERTS);
    w2.reserve(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const std::string prefix = base + ".experts." + std::to_string(e);
        w13.emplace_back(loader, prefix + ".w1.weight", prefix + ".w3.weight", true);
        w2.emplace_back(loader, prefix + ".w2.weight", "");
    }
    std::vector<float*> h13(apss26::NUM_EXPERTS), h2(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        h13[e] = w13[e].weight;
        h2[e] = w2[e].weight;
    }
    const std::size_t bytes = apss26::NUM_EXPERTS * sizeof(float*);
    cuda_check(cudaMalloc(&w13_ptrs, bytes), "cudaMalloc w13 pointer table");
    cuda_check(cudaMalloc(&w2_ptrs, bytes), "cudaMalloc w2 pointer table");
    cuda_check(cudaMemcpy(w13_ptrs, h13.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy w13 pointer table");
    cuda_check(cudaMemcpy(w2_ptrs, h2.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy w2 pointer table");
}

void PhiTinyMoEModel::DeviceMoE::free() {
    gate.free();
    for (std::size_t e = 0; e < w13.size(); ++e) {
        w13[e].free();
        w2[e].free();
    }
    cudaFree(w13_ptrs);
    cudaFree(w2_ptrs);
    w13_ptrs = nullptr;
    w2_ptrs = nullptr;
}

PhiTinyMoEModel::Layer::Layer(const ModelLoader& loader, std::size_t layer_idx)
    : input_norm(loader, "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight",
                 "model.layers." + std::to_string(layer_idx) + ".input_layernorm.bias"),
      post_norm(loader, "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight",
                "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.bias"),
      q_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias"),
      k_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias"),
      v_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias"),
      o_proj(loader, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight", "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.bias"),
      moe(loader, layer_idx) {}

PhiTinyMoEModel::PhiTinyMoEModel(const std::string& model_file)
    : loader_(model_file),
      embeddings_(loader_.load("model.embed_tokens.weight")),
      final_norm_(loader_, "model.norm.weight", "model.norm.bias"),
      lm_head_(loader_, "lm_head.weight", "lm_head.bias") {
    layers_.reserve(apss26::NUM_LAYERS);
    for (std::size_t i = 0; i < apss26::NUM_LAYERS; ++i) layers_.emplace_back(loader_, i);
    d_embeddings_ = device_copy(embeddings_);
}

PhiTinyMoEModel::~PhiTinyMoEModel() {
    for (Layer& layer : layers_) {
        layer.input_norm.free();
        layer.post_norm.free();
        layer.q_proj.free();
        layer.k_proj.free();
        layer.v_proj.free();
        layer.o_proj.free();
        layer.moe.free();
    }
    final_norm_.free();
    lm_head_.free();
    cudaFree(d_embeddings_);
    d_embeddings_ = nullptr;
}

void PhiTinyMoEModel::forward(const std::vector<int>& input_ids, Tensor& logits) const {
    if (input_ids.empty()) throw std::invalid_argument("empty input");
    generate({input_ids}, logits);
}

// The batch is packed into a single [T, HIDDEN] activation matrix, so the 32
// layers are traversed once for the whole batch, and the rows are the nodes of
// a prefix trie over the sequences rather than the raw tokens. Layer norm, the
// projections and the MoE are row-wise, so packing does not change any row's
// arithmetic; only attention needs to know which rows precede a given row.
void PhiTinyMoEModel::generate(
    const std::vector<std::vector<int>>& input_ids,
    Tensor& logits) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate received an empty input batch");
    }

    const std::size_t batch = input_ids.size();

    // Stage timers, off unless APS_PROFILE is set, so measured runs carry no
    // extra synchronisation. t_embed is the trie build; the embedding gather
    // itself is a device kernel and lands in t_h2d with the other uploads.
    const bool profile = std::getenv("APS_PROFILE") != nullptr;
    double t_embed = 0, t_norm = 0, t_h2d = 0, t_gemm = 0, t_d2h = 0,
           t_attn = 0, t_moe = 0, t_resid = 0, t_lm = 0;
    // `Tensor`'s constructor zero-fills the 131 MB output page by page: 69 ms
    // of host faulting that the D2H at the end overwrites in full. It needs
    // nothing from the GPU, so it runs alongside the 32 layers instead of
    // after them.
    std::thread alloc_thread([&] { logits = Tensor({batch, apss26::VOCAB_SIZE}); });
    // Joined on the way out as well, so a cuda_check throw unwinds with its
    // message instead of reaching std::terminate through ~thread.
    struct JoinGuard {
        std::thread& t;
        ~JoinGuard() { if (t.joinable()) t.join(); }
    } join_guard{alloc_thread};

    double mark = now_sec();
    auto tick = [&](double& sink) {
        if (!profile) return;
        cudaDeviceSynchronize();
        const double now = now_sec();
        sink += now - mark;
        mark = now;
    };

    // Prefix trie. Two tokens that sit at the same position of two sequences
    // agreeing on every earlier token get the same hidden state in all 32
    // layers: attention is causal and RoPE keys off the absolute position, so
    // nothing downstream can tell the two apart. One trie node per distinct
    // prefix is therefore one row, and a shared prefix is computed once.
    // Exact common-subexpression elimination -- no arithmetic changes.
    // Measured on the contest input: 19,803 tokens -> 15,583 nodes (-21.3%).
    std::vector<int> node_token, node_parent, node_depth;
    std::vector<int> last_node(batch);
    {
        std::unordered_map<long long, int> child;
        child.reserve(batch * 64);
        for (std::size_t b = 0; b < batch; ++b) {
            const std::vector<int>& seq = input_ids[b];
            if (seq.empty()) throw std::invalid_argument("empty input");
            if (seq.size() > apss26::MAX_POSITION_EMBEDDINGS) throw std::invalid_argument("sequence is too long");
            int parent = -1;
            for (std::size_t si = 0; si < seq.size(); ++si) {
                const int token = seq[si];
                if (token < 0 || static_cast<std::size_t>(token) >= apss26::VOCAB_SIZE) throw std::invalid_argument("token out of vocabulary");
                const long long key =
                    (static_cast<long long>(parent) + 1) *
                        static_cast<long long>(apss26::VOCAB_SIZE) + token;
                const auto it = child.find(key);
                if (it != child.end()) {
                    parent = it->second;
                } else {
                    const int node = static_cast<int>(node_token.size());
                    node_token.push_back(token);
                    node_parent.push_back(parent);
                    node_depth.push_back(static_cast<int>(si));
                    child.emplace(key, node);
                    parent = node;
                }
            }
            last_node[b] = parent;
        }
    }
    const std::size_t total = node_token.size();

    // A node's causal key set is its own root path, ascending by depth.
    // Flattened here so attention reads it straight instead of chasing parents.
    std::vector<int> anc_off(total + 1, 0);
    for (std::size_t n = 0; n < total; ++n) {
        anc_off[n + 1] = anc_off[n] + node_depth[n] + 1;
    }
    std::vector<int> anc(static_cast<std::size_t>(anc_off[total]));
    for (std::size_t n = 0; n < total; ++n) {
        int* dst = anc.data() + anc_off[n];
        const int parent = node_parent[n];
        // The parent's chain is this node's chain minus its last entry.
        if (parent >= 0) {
            std::memcpy(dst, anc.data() + anc_off[parent],
                        static_cast<std::size_t>(node_depth[n]) * sizeof(int));
        }
        dst[node_depth[n]] = static_cast<int>(n);
    }

    tick(t_embed);

    // Everything now runs on the device and the residual stream stays there
    // for all 32 layers: nothing crosses the bus between the trie upload and
    // the logits coming back.
    constexpr std::size_t QDIM = apss26::NUM_ATTENTION_HEADS * apss26::HEAD_DIM;
    constexpr std::size_t KVDIM = apss26::NUM_KV_HEADS * apss26::HEAD_DIM;
    constexpr std::size_t FFDIM = apss26::EXPERT_INTERMEDIATE_SIZE;
    const std::size_t elements = total * apss26::HIDDEN_SIZE;

    if (!scratch_) scratch_ = std::make_unique<Scratch>();
    Scratch& sc = *scratch_;

    // The residual stream, double buffered: the MoE accumulates the next
    // layer's input while this layer's is still needed for the residual add.
    float* d_hidden = sc.stream_a.reserve(elements);
    float* d_next = sc.stream_b.reserve(elements);

    float* d_norm = sc.norm.reserve(elements);
    float* d_attn = sc.attn.reserve(elements);
    float* d_q = sc.q.reserve(total * QDIM);
    float* d_k = sc.k.reserve(total * KVDIM);
    float* d_v = sc.v.reserve(total * KVDIM);
    float* d_ctx = sc.ctx.reserve(total * QDIM);
    float* d_router = sc.router.reserve(total * apss26::NUM_EXPERTS);
    // MoE scratch. Every expert's rows are packed into one buffer so the 16
    // expert GEMMs become one launch, so the row count is the number of
    // (row, expert) pairs -- exactly TOP_K * total.
    const std::size_t packed_rows = total * apss26::TOP_K;
    float* d_expert_in = sc.expert_in.reserve(packed_rows * apss26::HIDDEN_SIZE);
    float* d_expert_out = sc.expert_out.reserve(packed_rows * apss26::HIDDEN_SIZE);
    float* d_gate = sc.gate.reserve(packed_rows * FFDIM);
    float* d_gate_up = sc.gate_up.reserve(packed_rows * 2 * FFDIM);
    int* d_index = sc.index.reserve(packed_rows);
    // Routing scratch: each row's two experts, each copy's packed slot, and the
    // per-block per-expert counts the scan turns into offsets.
    const int nrb = static_cast<int>((total + ROUTE_BLOCK - 1) / ROUTE_BLOCK);
    const int max_tiles =
        static_cast<int>(packed_rows / GRP_BM + apss26::NUM_EXPERTS);
    int* d_pair = sc.pair.reserve(packed_rows);
    int* d_slot = sc.slot.reserve(packed_rows);
    int* d_blk = sc.blk.reserve(static_cast<std::size_t>(nrb) * apss26::NUM_EXPERTS);
    int* d_eoff = sc.eoff.reserve(apss26::NUM_EXPERTS);
    // Three ints per row tile; an expert contributes at most one partial tile.
    int* d_tiles =
        sc.tiles.reserve(3 * (packed_rows / GRP_BM + apss26::NUM_EXPERTS + 1));

    // Attention runs on the device now, so it needs the batch layout there:
    // each row's position inside its sequence (for RoPE) and each sequence's
    // extent (for the causal window).
    constexpr std::size_t HALF = apss26::HEAD_DIM / 2;
    std::size_t max_len = 0;
    for (std::size_t n = 0; n < total; ++n) {
        max_len = std::max(max_len, static_cast<std::size_t>(node_depth[n]) + 1);
    }
    // Built here, on the host, so the values come from the same libm calls the
    // sequential path made -- see rope_rows.
    std::vector<float> rope(max_len * HALF * 2);
    for (std::size_t si = 0; si < max_len; ++si) {
        for (std::size_t j = 0; j < HALF; ++j) {
            const float inv = std::pow(apss26::ROPE_THETA, -2.0f * static_cast<float>(j) / static_cast<float>(apss26::HEAD_DIM));
            rope[(si * HALF + j) * 2] = std::cos(static_cast<float>(si) * inv);
            rope[(si * HALF + j) * 2 + 1] = std::sin(static_cast<float>(si) * inv);
        }
    }
    int* d_pos = sc.pos.reserve(total);
    int* d_anc = sc.anc.reserve(anc.size());
    int* d_anc_off = sc.anc_off.reserve(anc_off.size());
    float* d_rope = sc.rope.reserve(rope.size());
    // A node's position is its depth in the trie.
    cuda_check(cudaMemcpy(d_pos, node_depth.data(), total * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy positions");
    cuda_check(cudaMemcpy(d_anc, anc.data(), anc.size() * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy ancestors");
    cuda_check(cudaMemcpy(d_anc_off, anc_off.data(), anc_off.size() * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy ancestor offsets");
    cuda_check(cudaMemcpy(d_rope, rope.data(), rope.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy rope table");
    const unsigned attn_shared = static_cast<unsigned>(
        (apss26::HEAD_DIM + 1 + max_len +
         max_len * (ATTN_CHUNK + 1)) * sizeof(float));
    const float attn_scale = std::sqrt(static_cast<float>(apss26::HEAD_DIM));

    // The embedding lookup is a row gather out of a device-resident table:
    // one 16 KB contiguous row per node, so it is fully coalesced. It replaces
    // a single-threaded host gather of the same 255 MB plus its upload.
    cuda_check(cudaMemcpy(d_index, node_token.data(), total * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy node tokens");
    gather_rows<<<static_cast<unsigned>(total), 256>>>(
        d_embeddings_, d_index, d_hidden, apss26::HIDDEN_SIZE);
    cuda_check(cudaGetLastError(), "embedding gather launch");
    tick(t_h2d);

    for (const Layer& layer : layers_) {
        layer.input_norm.forward(d_hidden, d_norm, total);
        tick(t_norm);
        layer.q_proj.forward(d_norm, d_q, total);
        layer.k_proj.forward(d_norm, d_k, total);
        layer.v_proj.forward(d_norm, d_v, total);
        tick(t_gemm);
        rope_rows<<<static_cast<unsigned>(total), 256>>>(
            d_q, d_k, d_pos, d_rope);
        attention_heads<<<dim3(static_cast<unsigned>(total),
                               apss26::NUM_ATTENTION_HEADS),
                          apss26::HEAD_DIM, attn_shared>>>(
            d_q, d_k, d_v, d_anc, d_anc_off,
            d_ctx, attn_scale);
        cuda_check(cudaGetLastError(), "attention kernels");
        tick(t_attn);
        layer.o_proj.forward(d_ctx, d_attn, total);
        tick(t_gemm);
        add_inplace_device(d_attn, d_hidden, static_cast<long long>(elements));
        tick(t_resid);

        // o_proj has consumed q/k/v, so d_norm is free to take the
        // post-attention norm -- which is also the expert gather source.
        layer.post_norm.forward(d_attn, d_norm, total);
        tick(t_norm);

        // MoE: route on the host (16 scores per token), then run each expert
        // over its own gathered rows and scatter the halves back.
        layer.moe.gate.forward(d_norm, d_router, total);
        tick(t_gemm);
        // Routing runs on the device: the 997 KB of scores no longer come back
        // and the packed layout is built where it is used. The tile count stays
        // on the device too -- the grid is launched at its worst case.
        route_count<<<nrb, ROUTE_BLOCK>>>(d_router, static_cast<int>(total),
                                          d_pair, d_blk);
        route_scan<<<1, apss26::NUM_EXPERTS>>>(d_blk, nrb, d_eoff,
                                               d_tiles, max_tiles);
        route_place<<<nrb, ROUTE_BLOCK>>>(d_pair, static_cast<int>(total),
                                          d_blk, d_eoff,
                                          d_index, d_slot);
        cuda_check(cudaGetLastError(), "routing kernels");
        tick(t_d2h);

        gather_rows<<<static_cast<unsigned>(packed_rows), 256>>>(
            d_norm, d_index, d_expert_in, apss26::HIDDEN_SIZE);
        gemm_grouped<<<dim3((2 * FFDIM + GRP_BN - 1) / GRP_BN, max_tiles), GEMM_THREADS>>>(
            d_expert_in, layer.moe.w13_ptrs, d_gate_up, d_tiles,
            static_cast<int>(apss26::HIDDEN_SIZE), static_cast<int>(2 * FFDIM));
        const long long activated = static_cast<long long>(packed_rows) * FFDIM;
        silu_mul<<<static_cast<unsigned>((activated + 255) / 256), 256>>>(
            d_gate_up, d_gate, static_cast<int>(FFDIM), activated);
        gemm_grouped<<<dim3((apss26::HIDDEN_SIZE + GRP_BN - 1) / GRP_BN, max_tiles), GEMM_THREADS>>>(
            d_gate, layer.moe.w2_ptrs, d_expert_out, d_tiles,
            static_cast<int>(FFDIM), static_cast<int>(apss26::HIDDEN_SIZE));
        // One pass per row instead of one launch per expert, so `d_next` is
        // written rather than zeroed and accumulated into.
        moe_combine<<<static_cast<unsigned>(total), 256>>>(
            d_expert_out, d_slot, d_next, apss26::HIDDEN_SIZE);
        cuda_check(cudaGetLastError(), "moe kernels");
        tick(t_moe);

        add_inplace_device(d_next, d_attn, static_cast<long long>(elements));
        tick(t_resid);
        std::swap(d_hidden, d_next);
    }

    final_norm_.forward(d_hidden, d_norm, total);
    tick(t_norm);

    // Only the last row of each sequence feeds lm_head: the trie node its
    // whole token list ends on. Two identical sequences share that node.
    cuda_check(cudaMemcpy(d_index, last_node.data(), batch * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy last rows");
    gather_rows<<<static_cast<unsigned>(batch), 256>>>(
        d_norm, d_index, d_expert_in, apss26::HIDDEN_SIZE);
    float* d_logits = sc.logits.reserve(batch * apss26::VOCAB_SIZE);
    lm_head_.forward(d_expert_in, d_logits, batch);
    alloc_thread.join();
    to_host(logits, d_logits);
    tick(t_lm);

    if (profile) {
        std::fprintf(stderr,
                     "[profile] T=%zu  embed %.3f  norm %.3f  h2d %.3f  gemm %.3f  "
                     "d2h %.3f  attn %.3f  moe %.3f  resid %.3f  lm_head %.3f\n",
                     total, t_embed, t_norm, t_h2d, t_gemm, t_d2h, t_attn, t_moe,
                     t_resid, t_lm);
    }
}

void PhiTinyMoEModel::generate_decode(
    const std::vector<std::vector<int>>& input_ids,
    std::size_t max_new_tokens,
    Tensor& logits,
    std::vector<std::vector<int>>& generated_ids) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate_decode received an empty input batch");
    }
    if (max_new_tokens == 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    const std::size_t batch = input_ids.size();
    logits = Tensor({batch, max_new_tokens, apss26::VOCAB_SIZE});
    generated_ids.assign(batch, {});

    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<int> prefix = input_ids[b];
        generated_ids[b].reserve(max_new_tokens);

        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            Tensor one_logits;
            forward(prefix, one_logits);

            int next_token = 0;
            float best_logit = one_logits.at(0, 0);
            for (std::size_t v = 1; v < apss26::VOCAB_SIZE; ++v) {
                const float value = one_logits.at(0, v);
                if (value > best_logit) {
                    best_logit = value;
                    next_token = static_cast<int>(v);
                }
            }

            for (std::size_t v = 0; v < apss26::VOCAB_SIZE; ++v) {
                logits.at(b, step, v) = one_logits.at(0, v);
            }
            generated_ids[b].push_back(next_token);

            if (next_token == apss26::EOS_TOKEN_ID) break;
            prefix.push_back(next_token);
        }
    }
}
