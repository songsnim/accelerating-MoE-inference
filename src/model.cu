#include "model.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>
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

// Device scratch that frees itself when the batch is done.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) {
        cuda_check(cudaMalloc(&data_, elements * sizeof(T)), "cudaMalloc scratch");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

void to_device(float* device, const Tensor& host) {
    cuda_check(cudaMemcpy(device, host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

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
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int GEMM_THREADS = (BM / TM) * (BN / TN);  // 256
// The +4 keeps the shared rows 16B-aligned while breaking the bank-conflict
// pattern of a bare [BK][BM] layout.
constexpr int SPAD = 4;

// Double-buffered variant: the global load for tile kt+BK is issued before the
// compute over tile kt, so the ~500-cycle LDG latency overlaps the FFMAs
// instead of stalling in front of the shared store. Only one __syncthreads()
// per iteration is needed because the two buffers alternate.
//
// The k loop still walks ascending and each output still accumulates into one
// register with the same `acc += a * b`, so the FMA sequence is unchanged.
__global__ void gemm_nt_bias(const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ bias,
                             float* __restrict__ c,
                             int m, int k, int n) {
    __shared__ __align__(16) float as[2][BK][BM + SPAD];
    __shared__ __align__(16) float bs[2][BK][BN + SPAD];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int lr = tid / (BK / 4);          // 0..127, row inside the tile
    const int lc = (tid % (BK / 4)) * 4;    // 0, 4 inside the k window
    const int ty = tid / (BN / TN);         // 0..15
    const int tx = tid % (BN / TN);         // 0..15

    const int a_row = m0 + lr;
    const int b_row = n0 + lr;
    const bool a_ok = a_row < m;
    const bool b_ok = b_row < n;
    const float* const ab = a + static_cast<long long>(a_row) * k + lc;
    const float* const bb = b + static_cast<long long>(b_row) * k + lc;

    float acc[TM][TN] = {};

    // Stage the first tile. Rows past the edge stay zero for the whole loop.
    float4 av = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 bv = av;
    if (a_ok) av = *reinterpret_cast<const float4*>(ab);
    if (b_ok) bv = *reinterpret_cast<const float4*>(bb);
    as[0][lc + 0][lr] = av.x; as[0][lc + 1][lr] = av.y;
    as[0][lc + 2][lr] = av.z; as[0][lc + 3][lr] = av.w;
    bs[0][lc + 0][lr] = bv.x; bs[0][lc + 1][lr] = bv.y;
    bs[0][lc + 2][lr] = bv.z; bs[0][lc + 3][lr] = bv.w;
    __syncthreads();

    int cur = 0;
    for (int kt = 0; kt < k; kt += BK) {
        const bool more = kt + BK < k;
        if (more) {
            if (a_ok) av = *reinterpret_cast<const float4*>(ab + kt + BK);
            if (b_ok) bv = *reinterpret_cast<const float4*>(bb + kt + BK);
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
                const float4 t = *reinterpret_cast<const float4*>(&bc[p][tx * TN + j]);
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
            bs[nxt][lc + 0][lr] = bv.x; bs[nxt][lc + 1][lr] = bv.y;
            bs[nxt][lc + 2][lr] = bv.z; bs[nxt][lc + 3][lr] = bv.w;
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
            c[static_cast<long long>(row) * n + col] = v;
        }
    }
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

// dst[index[i], :] += weight * src[i, :]
// Each expert holds every row at most once and the experts are applied in
// separate launches, so no two threads touch the same element.
__global__ void scatter_add_rows(const float* __restrict__ src,
                                 const int* __restrict__ index,
                                 float* __restrict__ dst, int cols,
                                 float weight) {
    const int row = index[blockIdx.x];
    const float* in = src + static_cast<long long>(blockIdx.x) * cols;
    float* out = dst + static_cast<long long>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] += weight * in[c];
}

// PhiMLP applies silu to the w1 branch and multiplies the w3 branch into it.
// tensor.cu evaluates exp in double; expf differs by ~1e-7 relative, which is
// four orders under the 3e-3 validation threshold.
__global__ void silu_mul(float* __restrict__ gate,
                         const float* __restrict__ up, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = gate[i];
        gate[i] = (x / (1.0f + expf(-x))) * up[i];
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
constexpr int NORM_BLOCK = 256;

__global__ void layer_norm_rows(const float* __restrict__ x,
                                const float* __restrict__ weight,
                                const float* __restrict__ bias,
                                float* __restrict__ y, int cols, float eps) {
    extern __shared__ float shared[];
    float* row = shared;
    float* stats = shared + cols;

    const long long base = static_cast<long long>(blockIdx.x) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) row[c] = x[base + c];
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) sum = __fadd_rn(sum, row[j]);
        const float mean = sum / static_cast<float>(cols);

        float var = 0.0f;
        for (int j = 0; j < cols; ++j) {
            const float d = __fsub_rn(row[j], mean);
            var = __fadd_rn(var, __fmul_rn(d, d));
        }
        stats[0] = mean;
        stats[1] = 1.0f / sqrtf(var / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    const float mean = stats[0], inv = stats[1];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float scaled = __fmul_rn(__fmul_rn(__fsub_rn(row[c], mean), inv),
                                       weight[c]);
        y[base + c] = __fadd_rn(scaled, bias[c]);
    }
}

// Mirrors PhiMoE::route in layer.cu: quantise the router scores, then pick the
// best two with a tie epsilon. Left on the host -- it is 16 values per token.
void route_row(const float* logits, int& first, int& second) {
    float scores[apss26::NUM_EXPERTS];
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const float score = logits[e];
        const float rounded = std::floor(std::fabs(score) /
            apss26::ROUTER_SCORE_QUANTUM + 0.5f) * apss26::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    auto select = [&scores](int excluded) {
        int best = -1;
        float best_value = -std::numeric_limits<float>::infinity();
        for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
            if (static_cast<int>(e) == excluded) continue;
            if (best < 0 || scores[e] > best_value + apss26::ROUTER_TIE_EPS) {
                best = static_cast<int>(e);
                best_value = scores[e];
            }
        }
        return best;
    };
    first = select(-1);
    second = select(first);
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

// Causal sliding-window attention. One block per (sequence, query head), with
// HEAD_DIM threads: thread d owns output element d.
//
// Every accumulation order matches the sequential reference, which is the whole
// difficulty of this kernel. A reordered sum here shifts the router logits by
// ~1e-7 and flips tokens to other experts, so:
//   - the q.k dot walks d ascending inside a single thread,
//   - the softmax denominator walks ki ascending inside a single thread,
//   - the value accumulation walks ki ascending inside thread d,
//   - __f*_rn keeps nvcc from contracting the multiply-adds into FMAs that the
//     host does not emit (verified in obj/model.o: vmulss + vaddss throughout),
//   - exp stays in double, as accurate_exp did.
__global__ void attention_heads(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                const int* __restrict__ begin,
                                const int* __restrict__ length,
                                float* __restrict__ out, float scale) {
    constexpr int D = static_cast<int>(apss26::HEAD_DIM);
    constexpr int QH = static_cast<int>(apss26::NUM_ATTENTION_HEADS);
    constexpr int KVH = static_cast<int>(apss26::NUM_KV_HEADS);
    constexpr int WINDOW = static_cast<int>(apss26::SLIDING_WINDOW);

    extern __shared__ float shared[];
    float* sq = shared;               // the query row being served
    float* sdenom = shared + D;       // the softmax denominator
    float* sw = shared + D + 1;       // one score per key

    const int b = blockIdx.x, qh = blockIdx.y;
    const int kh = qh / (QH / KVH);
    const int base = begin[b], len = length[b];
    const int tid = threadIdx.x;

    const float* qb = q + static_cast<long long>(base) * QH * D + qh * D;
    const float* kb = k + static_cast<long long>(base) * KVH * D + kh * D;
    const float* vb = v + static_cast<long long>(base) * KVH * D + kh * D;
    float* ob = out + static_cast<long long>(base) * QH * D + qh * D;

    for (int qi = 0; qi < len; ++qi) {
        const int lo = qi + 1 > WINDOW ? qi + 1 - WINDOW : 0;
        const int nk = qi - lo + 1;

        sq[tid] = qb[static_cast<long long>(qi) * QH * D + tid];
        __syncthreads();

        for (int i = tid; i < nk; i += D) {
            const float* kr = kb + static_cast<long long>(lo + i) * KVH * D;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) score = __fadd_rn(score, __fmul_rn(sq[d], kr[d]));
            sw[i] = score / scale;
        }
        __syncthreads();

        if (tid == 0) {
            float maxv = -INFINITY;
            for (int i = 0; i < nk; ++i) maxv = fmaxf(maxv, sw[i]);
            float denom = 0.0f;
            for (int i = 0; i < nk; ++i) {
                sw[i] = static_cast<float>(exp(static_cast<double>(sw[i] - maxv)));
                denom = __fadd_rn(denom, sw[i]);
            }
            *sdenom = denom;
        }
        __syncthreads();

        const float denom = *sdenom;
        float acc = 0.0f;
        for (int i = 0; i < nk; ++i) {
            const float w = sw[i] / denom;
            acc = __fadd_rn(acc, __fmul_rn(w, vb[static_cast<long long>(lo + i) * KVH * D + tid]));
        }
        ob[static_cast<long long>(qi) * QH * D + tid] = acc;
        __syncthreads();
    }
}

}  // namespace

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
    const dim3 grid(static_cast<unsigned>((out + BN - 1) / BN),
                    static_cast<unsigned>((rows + BM - 1) / BM));
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
    const std::size_t shared = (cols + 2) * sizeof(float);
    layer_norm_rows<<<static_cast<unsigned>(rows), NORM_BLOCK, shared>>>(
        x, weight, bias, y, static_cast<int>(cols), apss26::NORM_EPS);
    cuda_check(cudaGetLastError(), "layer_norm_rows launch");
}

PhiTinyMoEModel::DeviceMoE::DeviceMoE(const ModelLoader& loader, std::size_t layer_idx)
    : gate(loader, "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe.gate.weight", "") {
    const std::string base = "model.layers." + std::to_string(layer_idx) + ".block_sparse_moe";
    w1.reserve(apss26::NUM_EXPERTS);
    w2.reserve(apss26::NUM_EXPERTS);
    w3.reserve(apss26::NUM_EXPERTS);
    for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
        const std::string prefix = base + ".experts." + std::to_string(e);
        w1.emplace_back(loader, prefix + ".w1.weight", "");
        w2.emplace_back(loader, prefix + ".w2.weight", "");
        w3.emplace_back(loader, prefix + ".w3.weight", "");
    }
}

void PhiTinyMoEModel::DeviceMoE::free() {
    gate.free();
    for (std::size_t e = 0; e < w1.size(); ++e) {
        w1[e].free();
        w2[e].free();
        w3[e].free();
    }
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
}

void PhiTinyMoEModel::forward(const std::vector<int>& input_ids, Tensor& logits) const {
    if (input_ids.empty()) throw std::invalid_argument("empty input");
    generate({input_ids}, logits);
}

// Every sequence in the batch is packed into a single [T, HIDDEN] activation
// matrix, so the 32 layers are traversed once for the whole batch. Layer norm,
// the projections and the MoE are row-wise, so packing does not change any
// row's arithmetic; only attention needs to know the sequence boundaries.
void PhiTinyMoEModel::generate(
    const std::vector<std::vector<int>>& input_ids,
    Tensor& logits) const {
    if (input_ids.empty()) {
        throw std::runtime_error("generate received an empty input batch");
    }

    const std::size_t batch = input_ids.size();
    std::vector<std::size_t> offset(batch), length(batch);
    std::size_t total = 0;
    for (std::size_t b = 0; b < batch; ++b) {
        const std::size_t s = input_ids[b].size();
        if (s == 0) throw std::invalid_argument("empty input");
        if (s > apss26::MAX_POSITION_EMBEDDINGS) throw std::invalid_argument("sequence is too long");
        offset[b] = total;
        length[b] = s;
        total += s;
    }

    Tensor hidden({total, apss26::HIDDEN_SIZE});
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t si = 0; si < length[b]; ++si) {
            const int token = input_ids[b][si];
            if (token < 0 || static_cast<std::size_t>(token) >= apss26::VOCAB_SIZE) throw std::invalid_argument("token out of vocabulary");
            float* row = hidden.data() + (offset[b] + si) * apss26::HIDDEN_SIZE;
            const float* src = embeddings_.data() + static_cast<std::size_t>(token) * apss26::HIDDEN_SIZE;
            for (std::size_t h = 0; h < apss26::HIDDEN_SIZE; ++h) row[h] = src[h];
        }
    }

    // Stage timers, off unless APS_PROFILE is set, so measured runs carry no
    // extra synchronisation.
    const bool profile = std::getenv("APS_PROFILE") != nullptr;
    double t_embed = 0, t_norm = 0, t_h2d = 0, t_gemm = 0, t_d2h = 0,
           t_attn = 0, t_moe = 0, t_resid = 0, t_lm = 0;
    double mark = now_sec();
    auto tick = [&](double& sink) {
        if (!profile) return;
        cudaDeviceSynchronize();
        const double now = now_sec();
        sink += now - mark;
        mark = now;
    };
    tick(t_embed);

    // Everything except attention and the routing decision now runs on the
    // device, and the residual stream stays there for all 32 layers: only
    // q/k/v, the attention context and the 16 router scores per token still
    // cross the bus.
    constexpr std::size_t QDIM = apss26::NUM_ATTENTION_HEADS * apss26::HEAD_DIM;
    constexpr std::size_t KVDIM = apss26::NUM_KV_HEADS * apss26::HEAD_DIM;
    constexpr std::size_t FFDIM = apss26::EXPERT_INTERMEDIATE_SIZE;
    const std::size_t elements = total * apss26::HIDDEN_SIZE;

    // The residual stream, double buffered: the MoE accumulates the next
    // layer's input while this layer's is still needed for the residual add.
    DeviceBuffer<float> d_stream_a(elements), d_stream_b(elements);
    float* d_hidden = d_stream_a.get();
    float* d_next = d_stream_b.get();

    DeviceBuffer<float> d_norm(elements), d_attn(elements);
    DeviceBuffer<float> d_q(total * QDIM), d_k(total * KVDIM), d_v(total * KVDIM);
    DeviceBuffer<float> d_ctx(total * QDIM);
    DeviceBuffer<float> d_router(total * apss26::NUM_EXPERTS);
    // MoE scratch. An expert can in principle draw every row, so size for it.
    DeviceBuffer<float> d_expert_in(elements), d_expert_out(elements);
    DeviceBuffer<float> d_gate(total * FFDIM), d_up(total * FFDIM);
    DeviceBuffer<int> d_index(total);

    // Attention runs on the device now, so it needs the batch layout there:
    // each row's position inside its sequence (for RoPE) and each sequence's
    // extent (for the causal window).
    constexpr std::size_t HALF = apss26::HEAD_DIM / 2;
    std::size_t max_len = 0;
    std::vector<int> pos(total), seq_begin(batch), seq_len(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        seq_begin[b] = static_cast<int>(offset[b]);
        seq_len[b] = static_cast<int>(length[b]);
        max_len = std::max(max_len, length[b]);
        for (std::size_t si = 0; si < length[b]; ++si) pos[offset[b] + si] = static_cast<int>(si);
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
    DeviceBuffer<int> d_pos(total), d_begin(batch), d_len(batch);
    DeviceBuffer<float> d_rope(rope.size());
    cuda_check(cudaMemcpy(d_pos.get(), pos.data(), total * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy positions");
    cuda_check(cudaMemcpy(d_begin.get(), seq_begin.data(), batch * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy sequence begin");
    cuda_check(cudaMemcpy(d_len.get(), seq_len.data(), batch * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy sequence length");
    cuda_check(cudaMemcpy(d_rope.get(), rope.data(), rope.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy rope table");
    const unsigned attn_shared =
        static_cast<unsigned>((apss26::HEAD_DIM + 1 + max_len) * sizeof(float));
    const float attn_scale = std::sqrt(static_cast<float>(apss26::HEAD_DIM));

    to_device(d_hidden, hidden);
    tick(t_h2d);

    Tensor router({total, apss26::NUM_EXPERTS});
    std::vector<std::vector<int>> assignment(apss26::NUM_EXPERTS);

    for (const Layer& layer : layers_) {
        layer.input_norm.forward(d_hidden, d_norm.get(), total);
        tick(t_norm);
        layer.q_proj.forward(d_norm.get(), d_q.get(), total);
        layer.k_proj.forward(d_norm.get(), d_k.get(), total);
        layer.v_proj.forward(d_norm.get(), d_v.get(), total);
        tick(t_gemm);
        rope_rows<<<static_cast<unsigned>(total), 256>>>(
            d_q.get(), d_k.get(), d_pos.get(), d_rope.get());
        attention_heads<<<dim3(static_cast<unsigned>(batch),
                               apss26::NUM_ATTENTION_HEADS),
                          apss26::HEAD_DIM, attn_shared>>>(
            d_q.get(), d_k.get(), d_v.get(), d_begin.get(), d_len.get(),
            d_ctx.get(), attn_scale);
        cuda_check(cudaGetLastError(), "attention kernels");
        tick(t_attn);
        layer.o_proj.forward(d_ctx.get(), d_attn.get(), total);
        tick(t_gemm);
        add_inplace_device(d_attn.get(), d_hidden, static_cast<long long>(elements));
        tick(t_resid);

        // o_proj has consumed q/k/v, so d_norm is free to take the
        // post-attention norm -- which is also the expert gather source.
        layer.post_norm.forward(d_attn.get(), d_norm.get(), total);
        tick(t_norm);

        // MoE: route on the host (16 scores per token), then run each expert
        // over its own gathered rows and scatter the halves back.
        layer.moe.gate.forward(d_norm.get(), d_router.get(), total);
        tick(t_gemm);
        to_host(router, d_router.get());
        tick(t_d2h);

        for (auto& rows : assignment) rows.clear();
        for (std::size_t t = 0; t < total; ++t) {
            int first = 0, second = 0;
            route_row(router.data() + t * apss26::NUM_EXPERTS, first, second);
            assignment[first].push_back(static_cast<int>(t));
            assignment[second].push_back(static_cast<int>(t));
        }
        cuda_check(cudaMemset(d_next, 0, elements * sizeof(float)), "cudaMemset stream");
        for (std::size_t e = 0; e < apss26::NUM_EXPERTS; ++e) {
            const std::size_t rows = assignment[e].size();
            if (rows == 0) continue;
            cuda_check(cudaMemcpy(d_index.get(), assignment[e].data(), rows * sizeof(int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy expert index");
            gather_rows<<<static_cast<unsigned>(rows), 256>>>(
                d_norm.get(), d_index.get(), d_expert_in.get(), apss26::HIDDEN_SIZE);
            layer.moe.w1[e].forward(d_expert_in.get(), d_gate.get(), rows);
            layer.moe.w3[e].forward(d_expert_in.get(), d_up.get(), rows);
            const long long activated = static_cast<long long>(rows) * FFDIM;
            silu_mul<<<static_cast<unsigned>((activated + 255) / 256), 256>>>(
                d_gate.get(), d_up.get(), activated);
            layer.moe.w2[e].forward(d_gate.get(), d_expert_out.get(), rows);
            scatter_add_rows<<<static_cast<unsigned>(rows), 256>>>(
                d_expert_out.get(), d_index.get(), d_next, apss26::HIDDEN_SIZE, 0.5f);
        }
        cuda_check(cudaGetLastError(), "moe kernels");
        tick(t_moe);

        add_inplace_device(d_next, d_attn.get(), static_cast<long long>(elements));
        tick(t_resid);
        std::swap(d_hidden, d_next);
    }

    final_norm_.forward(d_hidden, d_norm.get(), total);
    tick(t_norm);

    // Only the last row of each sequence feeds lm_head.
    std::vector<int> last_rows(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        last_rows[b] = static_cast<int>(offset[b] + length[b] - 1);
    }
    cuda_check(cudaMemcpy(d_index.get(), last_rows.data(), batch * sizeof(int),
                          cudaMemcpyHostToDevice), "cudaMemcpy last rows");
    gather_rows<<<static_cast<unsigned>(batch), 256>>>(
        d_norm.get(), d_index.get(), d_expert_in.get(), apss26::HIDDEN_SIZE);
    DeviceBuffer<float> d_logits(batch * apss26::VOCAB_SIZE);
    lm_head_.forward(d_expert_in.get(), d_logits.get(), batch);
    logits = Tensor({batch, apss26::VOCAB_SIZE});
    to_host(logits, d_logits.get());
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
