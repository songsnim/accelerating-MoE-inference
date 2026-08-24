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
constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
constexpr int GEMM_THREADS = (BM / TM) * (BN / TN);  // 256
// The +4 keeps the shared rows 16B-aligned (68 * 4B) while breaking the
// bank-conflict pattern of a bare [BK][64] layout.
constexpr int SPAD = 4;

__global__ void gemm_nt_bias(const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ bias,
                             float* __restrict__ c,
                             int m, int k, int n) {
    __shared__ __align__(16) float as[BK][BM + SPAD];
    __shared__ __align__(16) float bs[BK][BN + SPAD];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    // Each thread loads one float4 of the A tile and one of the B tile.
    const int lr = tid / (BK / 4);          // 0..63, row inside the tile
    const int lc = (tid % (BK / 4)) * 4;    // 0, 4, 8, 12 inside the k window
    const int ty = tid / (BN / TN);         // 0..15
    const int tx = tid % (BN / TN);         // 0..15

    const int a_row = m0 + lr;
    const int b_row = n0 + lr;
    const bool a_ok = a_row < m;
    const bool b_ok = b_row < n;

    float acc[TM][TN] = {};
    for (int kt = 0; kt < k; kt += BK) {
        const int kk = kt + lc;
        float4 av = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 bv = av;
        if (a_ok) av = *reinterpret_cast<const float4*>(
                           a + static_cast<long long>(a_row) * k + kk);
        if (b_ok) bv = *reinterpret_cast<const float4*>(
                           b + static_cast<long long>(b_row) * k + kk);
        // Transposed store: the compute loop wants a whole k-slice contiguous.
        as[lc + 0][lr] = av.x; as[lc + 1][lr] = av.y;
        as[lc + 2][lr] = av.z; as[lc + 3][lr] = av.w;
        bs[lc + 0][lr] = bv.x; bs[lc + 1][lr] = bv.y;
        bs[lc + 2][lr] = bv.z; bs[lc + 3][lr] = bv.w;
        __syncthreads();

#pragma unroll
        for (int p = 0; p < BK; ++p) {
            const float4 ar = *reinterpret_cast<const float4*>(&as[p][ty * TM]);
            const float4 br = *reinterpret_cast<const float4*>(&bs[p][tx * TN]);
            const float av4[TM] = {ar.x, ar.y, ar.z, ar.w};
            const float bv4[TN] = {br.x, br.y, br.z, br.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av4[i] * bv4[j];
        }
        __syncthreads();
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

// layer.cu evaluates every exponential in double and rounds once. Keep that.
inline float accurate_exp(float x) {
    return static_cast<float>(std::exp(static_cast<double>(x)));
}

// RoPE + causal attention for one sequence occupying rows [begin, begin+len)
// of the batched projections. Arithmetic mirrors PhiAttention::forward, with
// each q.k dot product evaluated once instead of once per output dimension.
void attend_sequence(Tensor& q, Tensor& k, const Tensor& v,
                     std::size_t begin, std::size_t len, Tensor& out) {
    constexpr std::size_t D = apss26::HEAD_DIM;
    constexpr std::size_t QH = apss26::NUM_ATTENTION_HEADS;
    constexpr std::size_t KVH = apss26::NUM_KV_HEADS;
    constexpr std::size_t half = D / 2;
    const std::size_t group = QH / KVH;
    const float scale = std::sqrt(static_cast<float>(D));

    float* qbase = q.data() + begin * QH * D;
    float* kbase = k.data() + begin * KVH * D;
    const float* vbase = v.data() + begin * KVH * D;
    float* obase = out.data() + begin * QH * D;

    for (std::size_t s = 0; s < len; ++s) {
        for (std::size_t h = 0; h < QH; ++h) for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(apss26::ROPE_THETA, -2.0f * static_cast<float>(j) / static_cast<float>(D));
            const float c = std::cos(static_cast<float>(s) * inv), sn = std::sin(static_cast<float>(s) * inv);
            float* row = qbase + s * QH * D + h * D;
            const float x0 = row[j], x1 = row[j + half];
            row[j] = x0 * c - x1 * sn;
            row[j + half] = x1 * c + x0 * sn;
        }
        for (std::size_t h = 0; h < KVH; ++h) for (std::size_t j = 0; j < half; ++j) {
            const float inv = std::pow(apss26::ROPE_THETA, -2.0f * static_cast<float>(j) / static_cast<float>(D));
            const float c = std::cos(static_cast<float>(s) * inv), sn = std::sin(static_cast<float>(s) * inv);
            float* row = kbase + s * KVH * D + h * D;
            const float x0 = row[j], x1 = row[j + half];
            row[j] = x0 * c - x1 * sn;
            row[j + half] = x1 * c + x0 * sn;
        }
    }

    std::vector<float> weights(len);
    for (std::size_t qi = 0; qi < len; ++qi) for (std::size_t qh = 0; qh < QH; ++qh) {
        const std::size_t kh = qh / group;
        const float* qrow = qbase + qi * QH * D + qh * D;
        const std::size_t lo = qi + 1 > apss26::SLIDING_WINDOW ? qi + 1 - apss26::SLIDING_WINDOW : 0;

        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t ki = lo; ki <= qi; ++ki) {
            const float* krow = kbase + ki * KVH * D + kh * D;
            float score = 0.0f;
            for (std::size_t d = 0; d < D; ++d) score += qrow[d] * krow[d];
            weights[ki] = score / scale;
            maxv = std::max(maxv, weights[ki]);
        }
        float denom = 0.0f;
        for (std::size_t ki = lo; ki <= qi; ++ki) {
            weights[ki] = accurate_exp(weights[ki] - maxv);
            denom += weights[ki];
        }

        float acc[D] = {};
        for (std::size_t ki = lo; ki <= qi; ++ki) {
            const float w = weights[ki] / denom;
            const float* vrow = vbase + ki * KVH * D + kh * D;
            for (std::size_t d = 0; d < D; ++d) acc[d] += w * vrow[d];
        }
        float* orow = obase + qi * QH * D + qh * D;
        for (std::size_t d = 0; d < D; ++d) orow[d] = acc[d];
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

    to_device(d_hidden, hidden);
    tick(t_h2d);

    Tensor q({total, QDIM}), k({total, KVDIM}), v({total, KVDIM});
    Tensor context({total, QDIM});
    Tensor router({total, apss26::NUM_EXPERTS});
    std::vector<std::vector<int>> assignment(apss26::NUM_EXPERTS);

    for (const Layer& layer : layers_) {
        layer.input_norm.forward(d_hidden, d_norm.get(), total);
        tick(t_norm);
        layer.q_proj.forward(d_norm.get(), d_q.get(), total);
        layer.k_proj.forward(d_norm.get(), d_k.get(), total);
        layer.v_proj.forward(d_norm.get(), d_v.get(), total);
        tick(t_gemm);
        to_host(q, d_q.get());
        to_host(k, d_k.get());
        to_host(v, d_v.get());
        tick(t_d2h);

#pragma omp parallel for schedule(dynamic)
        for (long long b = 0; b < static_cast<long long>(batch); ++b) {
            attend_sequence(q, k, v, offset[b], length[b], context);
        }
        tick(t_attn);

        to_device(d_ctx.get(), context);
        tick(t_h2d);
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
