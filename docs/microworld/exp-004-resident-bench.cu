// exp-004-resident.html 에 박힌 측정값을 뽑은 마이크로벤치.
//
// EXP-004 (51fcc64) 은 두 가지를 했다.
//   (1) layer_norm / residual add 를 커널로 옮겨 residual stream 을 32 레이어 내내
//       device 에 상주시켰다  -> host 연산과 PCIe 왕복이 사라진다.
//   (2) 그 norm 커널을 host 와 **bit-identical** 하게 만들었다 (thread 0 직렬 누적 +
//       __f*_rn). tree reduction 버전은 합의 순서가 달라 router logit 이 ~1e-7 흔들리고,
//       PhiMoE::route 의 양자화 경계에서 expert 가 뒤집혀 검증에 실패했다.
//
// 이 벤치는 한 레이어 분량을 떼어내 다음을 나란히 잰다.
//   NORM   host tensor_ops::layer_norm  vs  device 직렬  vs  device tree
//   RESID  host tensor_ops::add_inplace vs  device add_inplace_kernel
//   BUS    EXP-003 경로가 레이어당 나르는 바이트 vs EXP-004 경로
//   BITS   host norm 과 두 커널의 bit 단위 일치 여부
//   ROUTE  직렬/tree norm 을 각각 gate GEMM 에 먹여 top-2 집합이 뒤집히는 수
//
// 커널과 host op 은 원본(src/model.cu, src/tensor.cu, src/layer.cu)에서 그대로 복사했다.
// 가중치는 난수다 (model.bin 15GB 를 읽지 않는다). shape/연산량/누적 순서는 실제와 동일.
//
//   make                     # obj/tensor.o 생성
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 -Xcompiler=-march=native \
//       -Xcompiler=-fopenmp -Iinclude docs/microworld/exp-004-resident-bench.cu obj/tensor.o \
//       -o ~/residbench -L/usr/local/cuda/lib64 -lcudart -lm -lgomp
//   srun --partition aps --gres=gpu:1 --exclusive ~/residbench
//
// 측정 기록: 2026-08-25. 결과는 exp-004-resident.html 의 MEAS 상수.
#include "tensor.h"
#include "config.h"
#include "model_loader.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <vector>

static double now() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static void fill_rand(Tensor& t, unsigned seed){unsigned s=seed;for(std::size_t i=0;i<t.size();++i){s=s*1664525u+1013904223u;t[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}}
static void ck(cudaError_t e, const char* what){ if(e!=cudaSuccess){ std::printf("CUDA FAIL %s: %s\n",what,cudaGetErrorString(e)); std::exit(1);} }

namespace cfg = apss26;
constexpr std::size_t H    = cfg::HIDDEN_SIZE;                               // 4096
constexpr std::size_t NE   = cfg::NUM_EXPERTS;                               // 16
constexpr std::size_t QDIM = cfg::NUM_ATTENTION_HEADS * cfg::HEAD_DIM;       // 2048
constexpr std::size_t KVD  = cfg::NUM_KV_HEADS * cfg::HEAD_DIM;              // 512

/* ---------------- EXP-005 이전의 GEMM 커널 원문 (EXP-002~004) ---------------- */
constexpr int TILE = 32;
__global__ void gemm_nt_bias(const float* __restrict__ a, const float* __restrict__ b,
                             const float* __restrict__ bias, float* __restrict__ c,
                             int m, int k, int n) {
    __shared__ float as[TILE][TILE + 1];
    __shared__ float bs[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    const int b_row = blockIdx.x * TILE + ty;
    float acc = 0.0f;
    for (int kt = 0; kt < k; kt += TILE) {
        const int kk = kt + tx;
        as[ty][tx] = (row < m && kk < k) ? a[static_cast<long long>(row) * k + kk] : 0.0f;
        bs[ty][tx] = (b_row < n && kk < k) ? b[static_cast<long long>(b_row) * k + kk] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) acc += as[ty][i] * bs[tx][i];
        __syncthreads();
    }
    if (row < m && col < n) {
        if (bias != nullptr) acc += bias[col];
        c[static_cast<long long>(row) * n + col] = acc;
    }
}
static void dgemm_nt(const float* x, const float* w, float* y, std::size_t rows,
                     std::size_t in, std::size_t out) {
    const dim3 block(TILE, TILE);
    const dim3 grid((unsigned)((out + TILE - 1) / TILE), (unsigned)((rows + TILE - 1) / TILE));
    gemm_nt_bias<<<grid, block>>>(x, w, nullptr, y, (int)rows, (int)in, (int)out);
}

/* ---------------- EXP-004 의 커널 원문 ---------------- */
__global__ void add_inplace_kernel(float* __restrict__ a,
                                   const float* __restrict__ b, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

constexpr int NORM_BLOCK = 256;

// 채택된 버전: 행을 shared 에 올리고 thread 0 이 직렬로 두 번 훑는다.
// __f*_rn 은 nvcc 의 FMA contraction 을 막는다 (host 는 FMA 를 내지 않는다).
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
        const float scaled = __fmul_rn(__fmul_rn(__fsub_rn(row[c], mean), inv), weight[c]);
        y[base + c] = __fadd_rn(scaled, bias[c]);
    }
}

// 폐기된 첫 시도: block tree reduction. 빠르지만 합의 순서가 host 와 다르다.
__global__ void layer_norm_rows_tree(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     float* __restrict__ y, int cols, float eps) {
    __shared__ float red[NORM_BLOCK];
    __shared__ float stats[2];
    const long long base = static_cast<long long>(blockIdx.x) * cols;
    float part = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) part += x[base + c];
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s = NORM_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) stats[0] = red[0] / static_cast<float>(cols);
    __syncthreads();
    const float mean = stats[0];
    float pv = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float d = x[base + c] - mean; pv += d * d;
    }
    red[threadIdx.x] = pv;
    __syncthreads();
    for (int s = NORM_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) stats[1] = 1.0f / sqrtf(red[0] / static_cast<float>(cols) + eps);
    __syncthreads();
    const float inv = stats[1];
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        y[base + c] = (x[base + c] - mean) * inv * weight[c] + bias[c];
}

/* ---------------- 라우팅: layer.cu PhiMoE::route 원문 ---------------- */
static void route_row(const float* logits, int& first, int& second) {
    float scores[NE];
    for (std::size_t e = 0; e < NE; ++e) {
        const float score = logits[e];
        const float rounded = std::floor(std::fabs(score) / cfg::ROUTER_SCORE_QUANTUM + 0.5f) * cfg::ROUTER_SCORE_QUANTUM;
        scores[e] = score < 0.0f ? -rounded : rounded;
    }
    auto select = [&scores](int excluded) {
        int best = -1; float bv = -std::numeric_limits<float>::infinity();
        for (std::size_t e = 0; e < NE; ++e) {
            if ((int)e == excluded) continue;
            if (best < 0 || scores[e] > bv + cfg::ROUTER_TIE_EPS) { best = (int)e; bv = scores[e]; }
        }
        return best;
    };
    first = select(-1); second = select(first);
}

int main() {
    const std::size_t Ts[] = {165, 2475, 19803};   // n=8 / expert 평균 행수 / n=1024

    Tensor nw({H}), nb({H}), gate_w({NE, H});
    fill_rand(nw, 7); fill_rand(nb, 11); fill_rand(gate_w, 101);
    // norm weight/bias 는 실제로 1 근방 / 0 근방이다. 스케일만 맞춘다.
    for (std::size_t i = 0; i < H; ++i) { nw[i] = 1.0f + 0.05f * nw[i]; nb[i] *= 0.02f; }

    float *d_nw, *d_nb, *d_gate_w;
    ck(cudaMalloc(&d_nw, H*sizeof(float)), "nw");
    ck(cudaMalloc(&d_nb, H*sizeof(float)), "nb");
    ck(cudaMalloc(&d_gate_w, NE*H*sizeof(float)), "gw");
    ck(cudaMemcpy(d_nw, nw.data(), H*sizeof(float), cudaMemcpyHostToDevice), "cnw");
    ck(cudaMemcpy(d_nb, nb.data(), H*sizeof(float), cudaMemcpyHostToDevice), "cnb");
    ck(cudaMemcpy(d_gate_w, gate_w.data(), NE*H*sizeof(float), cudaMemcpyHostToDevice), "cgw");

    for (std::size_t T : Ts) {
        Tensor x({T, H}), y_host({T, H}), y_ser({T, H}), y_tree({T, H});
        fill_rand(x, 1234u + (unsigned)T);

        float *d_x, *d_y, *d_y2, *d_r1, *d_r2, *d_acc;
        ck(cudaMalloc(&d_x, T*H*sizeof(float)), "x");
        ck(cudaMalloc(&d_y, T*H*sizeof(float)), "y");
        ck(cudaMalloc(&d_y2, T*H*sizeof(float)), "y2");
        ck(cudaMalloc(&d_acc, T*H*sizeof(float)), "acc");
        ck(cudaMalloc(&d_r1, T*NE*sizeof(float)), "r1");
        ck(cudaMalloc(&d_r2, T*NE*sizeof(float)), "r2");
        ck(cudaMemcpy(d_x, x.data(), T*H*sizeof(float), cudaMemcpyHostToDevice), "cx");

        const std::size_t shmem = (H + 2) * sizeof(float);
        const int reps = T <= 2475 ? 20 : 5;

        /* ---- NORM: host ---- */
        tensor_ops::layer_norm(x, nw, nb, cfg::NORM_EPS, y_host);
        double s = now();
        for (int r = 0; r < reps; ++r) tensor_ops::layer_norm(x, nw, nb, cfg::NORM_EPS, y_host);
        const double t_norm_host = (now() - s) / reps;

        /* ---- NORM: host, EXP-003 이 실제로 한 대로 (호출마다 Tensor 새로 할당) ----
           EXP-003 의 루프는 레이어마다 `Tensor normed(hidden.shape())` 를 새로 만든다.
           Tensor 는 vector<float> 를 0 으로 채우므로 324MB 할당 + first touch 가 붙는다.
           프로파일의 t_norm 16.1s 가 산술 시간인지 할당 시간인지 가른다. */
        {
            double t_alloc = 0, t_full = 0;
            for (int r = 0; r < reps; ++r) {
                double a0 = now();
                Tensor fresh(x.shape());
                t_alloc += now() - a0;
                double f0 = now();
                tensor_ops::layer_norm(x, nw, nb, cfg::NORM_EPS, fresh);
                t_full += now() - f0;
            }
            std::printf("ALLOC T=%-6zu tensor_ctor_ms %8.3f  norm_after_ctor_ms %8.3f  "
                        "sum_ms %8.3f  (재사용 버퍼 norm %8.3f)\n",
                        T, t_alloc/reps*1e3, t_full/reps*1e3, (t_alloc+t_full)/reps*1e3,
                        t_norm_host*1e3);
        }

        /* ---- NORM: device 직렬 (채택) ---- */
        layer_norm_rows<<<(unsigned)T, NORM_BLOCK, shmem>>>(d_x, d_nw, d_nb, d_y, (int)H, cfg::NORM_EPS);
        ck(cudaDeviceSynchronize(), "warm ser");
        s = now();
        for (int r = 0; r < reps; ++r)
            layer_norm_rows<<<(unsigned)T, NORM_BLOCK, shmem>>>(d_x, d_nw, d_nb, d_y, (int)H, cfg::NORM_EPS);
        ck(cudaDeviceSynchronize(), "ser");
        const double t_norm_ser = (now() - s) / reps;

        /* ---- NORM: device tree (폐기) ---- */
        layer_norm_rows_tree<<<(unsigned)T, NORM_BLOCK>>>(d_x, d_nw, d_nb, d_y2, (int)H, cfg::NORM_EPS);
        ck(cudaDeviceSynchronize(), "warm tree");
        s = now();
        for (int r = 0; r < reps; ++r)
            layer_norm_rows_tree<<<(unsigned)T, NORM_BLOCK>>>(d_x, d_nw, d_nb, d_y2, (int)H, cfg::NORM_EPS);
        ck(cudaDeviceSynchronize(), "tree");
        const double t_norm_tree = (now() - s) / reps;

        std::printf("NORM T=%-6zu host_ms %9.3f  dev_serial_ms %8.3f  dev_tree_ms %8.3f  "
                    "host/ser %7.1fx  ser/tree %5.2fx\n",
                    T, t_norm_host*1e3, t_norm_ser*1e3, t_norm_tree*1e3,
                    t_norm_host/t_norm_ser, t_norm_ser/t_norm_tree);

        /* ---- BITS: host 와 bit 단위로 같은가 ---- */
        ck(cudaMemcpy(y_ser.data(), d_y, T*H*sizeof(float), cudaMemcpyDeviceToHost), "gy");
        ck(cudaMemcpy(y_tree.data(), d_y2, T*H*sizeof(float), cudaMemcpyDeviceToHost), "gy2");
        auto bitcmp = [&](const Tensor& a, const Tensor& b, const char* tag) {
            std::size_t diff = 0; double maxabs = 0, sumabs = 0; std::size_t maxulp = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float va = a[i], vb = b[i];
                if (std::memcmp(&va, &vb, 4) != 0) {
                    ++diff;
                    const double d = std::fabs((double)va - (double)vb);
                    maxabs = std::max(maxabs, d); sumabs += d;
                    int ia, ib; std::memcpy(&ia, &va, 4); std::memcpy(&ib, &vb, 4);
                    maxulp = std::max(maxulp, (std::size_t)std::abs(ia - ib));
                }
            }
            std::printf("BITS T=%-6zu %-6s differing %10zu / %10zu (%7.4f%%)  max_abs %.3e  "
                        "mean_abs_over_diff %.3e  max_ulp %zu\n",
                        T, tag, diff, a.size(), 100.0*diff/a.size(), maxabs,
                        diff ? sumabs/diff : 0.0, maxulp);
        };
        bitcmp(y_host, y_ser, "serial");
        bitcmp(y_host, y_tree, "tree");

        /* ---- RESID: host add_inplace vs 커널 ---- */
        {
            Tensor a = y_host, b = y_host;
            tensor_ops::add_inplace(a, b);
            s = now();
            for (int r = 0; r < reps; ++r) tensor_ops::add_inplace(a, b);
            const double t_h = (now() - s) / reps;
            add_inplace_kernel<<<(unsigned)((T*H+255)/256), 256>>>(d_acc, d_y, (long long)(T*H));
            ck(cudaDeviceSynchronize(), "warm add");
            s = now();
            for (int r = 0; r < reps; ++r)
                add_inplace_kernel<<<(unsigned)((T*H+255)/256), 256>>>(d_acc, d_y, (long long)(T*H));
            ck(cudaDeviceSynchronize(), "add");
            const double t_d = (now() - s) / reps;
            std::printf("RESID T=%-6zu host_ms %9.3f  dev_ms %8.3f  %7.1fx\n",
                        T, t_h*1e3, t_d*1e3, t_h/t_d);
        }

        /* ---- BUS: 레이어당 실제로 나르는 것들의 측정 시간 ---- */
        {
            std::vector<float> host_buf(T*H);
            auto h2d = [&](std::size_t elems) {
                ck(cudaMemcpy(d_y2, host_buf.data(), elems*sizeof(float), cudaMemcpyHostToDevice), "h2d");
                ck(cudaDeviceSynchronize(), "s");
                double t0 = now();
                for (int r = 0; r < reps; ++r)
                    ck(cudaMemcpy(d_y2, host_buf.data(), elems*sizeof(float), cudaMemcpyHostToDevice), "h2d");
                ck(cudaDeviceSynchronize(), "s"); return (now()-t0)/reps;
            };
            auto d2h = [&](std::size_t elems) {
                ck(cudaMemcpy(host_buf.data(), d_y2, elems*sizeof(float), cudaMemcpyDeviceToHost), "d2h");
                ck(cudaDeviceSynchronize(), "s");
                double t0 = now();
                for (int r = 0; r < reps; ++r)
                    ck(cudaMemcpy(host_buf.data(), d_y2, elems*sizeof(float), cudaMemcpyDeviceToHost), "d2h");
                ck(cudaDeviceSynchronize(), "s"); return (now()-t0)/reps;
            };
            const double h_hidden = h2d(T*H), d_hidden_t = d2h(T*H);
            const double d_q = d2h(T*QDIM), d_kv = d2h(T*KVD);
            const double h_ctx = h2d(T*QDIM), d_router = d2h(T*NE);
            // EXP-003 경로: H2D normed, D2H q/k/v, H2D ctx, D2H attn, H2D post, D2H router, D2H ff
            const double before = h_hidden + d_q + 2*d_kv + h_ctx + d_hidden_t + h_hidden + d_router + d_hidden_t;
            // EXP-004 경로: D2H q/k/v, H2D ctx, D2H router
            const double after = d_q + 2*d_kv + h_ctx + d_router;
            const double mb = 4.0/1048576.0;
            std::printf("BUS  T=%-6zu hidden_MB %8.2f  h2d_hidden_ms %8.3f  d2h_hidden_ms %8.3f  "
                        "d2h_q_ms %7.3f d2h_kv_ms %7.3f h2d_ctx_ms %7.3f d2h_router_ms %7.4f | "
                        "before_MB %9.2f before_ms %9.3f  after_MB %9.2f after_ms %8.3f  %5.2fx\n",
                        T, T*H*mb, h_hidden*1e3, d_hidden_t*1e3, d_q*1e3, d_kv*1e3, h_ctx*1e3,
                        d_router*1e3,
                        (4.0*T*H + T*QDIM + 2*T*KVD + T*QDIM + T*NE)*mb, before*1e3,
                        (T*QDIM + 2*T*KVD + T*QDIM + T*NE)*mb, after*1e3, before/after);
        }

        /* ---- ROUTE: 직렬 norm vs tree norm 이 만드는 라우팅 차이 ---- */
        {
            dgemm_nt(d_y, d_gate_w, d_r1, T, H, NE);     // d_y = 직렬 norm (= host bit-identical)
            dgemm_nt(d_y2, d_gate_w, d_r2, T, H, NE);    // d_y2 는 위 BUS 에서 덮였다 -> 다시 계산
            ck(cudaDeviceSynchronize(), "gemm");
            // d_y2 를 tree norm 으로 복원한 뒤 다시.
            layer_norm_rows_tree<<<(unsigned)T, NORM_BLOCK>>>(d_x, d_nw, d_nb, d_y2, (int)H, cfg::NORM_EPS);
            dgemm_nt(d_y2, d_gate_w, d_r2, T, H, NE);
            ck(cudaDeviceSynchronize(), "gemm2");

            std::vector<float> r1(T*NE), r2(T*NE);
            ck(cudaMemcpy(r1.data(), d_r1, T*NE*sizeof(float), cudaMemcpyDeviceToHost), "gr1");
            ck(cudaMemcpy(r2.data(), d_r2, T*NE*sizeof(float), cudaMemcpyDeviceToHost), "gr2");

            double maxd = 0, sumd = 0; std::size_t nd = 0;
            for (std::size_t i = 0; i < T*NE; ++i) {
                const double d = std::fabs((double)r1[i] - (double)r2[i]);
                if (d > 0) { ++nd; sumd += d; maxd = std::max(maxd, d); }
            }
            std::size_t set_flip = 0, order_flip = 0, at_risk = 0;
            const double delta = maxd;   // 관측된 logit 흔들림의 상한
            for (std::size_t t = 0; t < T; ++t) {
                int a1, b1, a2, b2;
                route_row(r1.data()+t*NE, a1, b1);
                route_row(r2.data()+t*NE, a2, b2);
                if (a1 == a2 && b1 == b2) { /* same */ }
                else if (a1 == b2 && b1 == a2) ++order_flip;
                else ++set_flip;
                // 양자화 경계와의 거리: |s|/Q 의 소수부가 0.5 근처면 1 quantum 이 흔들린다.
                for (std::size_t e = 0; e < NE; ++e) {
                    const double u = std::fabs((double)r1[t*NE+e]) / cfg::ROUTER_SCORE_QUANTUM;
                    const double frac = std::fabs(u - std::floor(u) - 0.5);
                    if (frac * cfg::ROUTER_SCORE_QUANTUM < delta) { ++at_risk; break; }
                }
            }
            std::printf("ROUTE T=%-6zu logits %8zu  differing %8zu (%6.2f%%)  max_dlogit %.3e  "
                        "mean_dlogit %.3e | set_flip %6zu  order_flip %6zu  boundary_at_risk %6zu\n",
                        T, T*NE, nd, 100.0*nd/(T*NE), maxd, nd ? sumd/nd : 0.0,
                        set_flip, order_flip, at_risk);
        }

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_y2); cudaFree(d_acc);
        cudaFree(d_r1); cudaFree(d_r2);
    }

    /* ---- REAL: model.bin 의 실제 gate/norm weight 로 라우팅 민감도 ----
       위 ROUTE 는 난수 gate weight 라 logit 스케일이 실제와 다르다.
       gate weight (16x4096) 와 norm weight/bias 만 seek 해서 읽는다 (15GB 안 읽는다). */
    {
        const char* mp = "/apss26/project-data/model.bin";
        ModelLoader loader(mp);
        Tensor rw = loader.load("model.layers.0.input_layernorm.weight");
        Tensor rb = loader.load("model.layers.0.input_layernorm.bias");
        Tensor gw = loader.load("model.layers.0.block_sparse_moe.gate.weight");
        const std::size_t T = 19803;
        Tensor x({T, H});
        fill_rand(x, 777);

        float *d_x, *d_ys, *d_yt, *d_w, *d_b, *d_g, *d_r1, *d_r2;
        ck(cudaMalloc(&d_x, T*H*sizeof(float)), "rx");
        ck(cudaMalloc(&d_ys, T*H*sizeof(float)), "rys");
        ck(cudaMalloc(&d_yt, T*H*sizeof(float)), "ryt");
        ck(cudaMalloc(&d_w, H*sizeof(float)), "rw");
        ck(cudaMalloc(&d_b, H*sizeof(float)), "rb");
        ck(cudaMalloc(&d_g, NE*H*sizeof(float)), "rg");
        ck(cudaMalloc(&d_r1, T*NE*sizeof(float)), "rr1");
        ck(cudaMalloc(&d_r2, T*NE*sizeof(float)), "rr2");
        ck(cudaMemcpy(d_x, x.data(), T*H*sizeof(float), cudaMemcpyHostToDevice), "crx");
        ck(cudaMemcpy(d_w, rw.data(), H*sizeof(float), cudaMemcpyHostToDevice), "crw");
        ck(cudaMemcpy(d_b, rb.data(), H*sizeof(float), cudaMemcpyHostToDevice), "crb");
        ck(cudaMemcpy(d_g, gw.data(), NE*H*sizeof(float), cudaMemcpyHostToDevice), "crg");

        const std::size_t shmem = (H + 2) * sizeof(float);
        layer_norm_rows<<<(unsigned)T, NORM_BLOCK, shmem>>>(d_x, d_w, d_b, d_ys, (int)H, cfg::NORM_EPS);
        layer_norm_rows_tree<<<(unsigned)T, NORM_BLOCK>>>(d_x, d_w, d_b, d_yt, (int)H, cfg::NORM_EPS);
        dgemm_nt(d_ys, d_g, d_r1, T, H, NE);
        dgemm_nt(d_yt, d_g, d_r2, T, H, NE);
        ck(cudaDeviceSynchronize(), "real gemm");

        std::vector<float> r1(T*NE), r2(T*NE);
        ck(cudaMemcpy(r1.data(), d_r1, T*NE*sizeof(float), cudaMemcpyDeviceToHost), "g1");
        ck(cudaMemcpy(r2.data(), d_r2, T*NE*sizeof(float), cudaMemcpyDeviceToHost), "g2");

        double maxl = 0, suml = 0, maxd = 0, sumd = 0; std::size_t nd = 0;
        for (std::size_t i = 0; i < T*NE; ++i) {
            maxl = std::max(maxl, (double)std::fabs(r1[i])); suml += std::fabs(r1[i]);
            const double d = std::fabs((double)r1[i] - (double)r2[i]);
            if (d > 0) { ++nd; sumd += d; maxd = std::max(maxd, d); }
        }
        std::printf("REAL  logit |max| %.4f  mean|logit| %.4f  quantum %.0e  tie_eps %.0e\n",
                    maxl, suml/(T*NE), cfg::ROUTER_SCORE_QUANTUM, cfg::ROUTER_TIE_EPS);
        std::printf("REAL  dlogit(tree vs serial) differing %zu/%zu  max %.3e  mean %.3e  "
                    "= %.4f quantum\n", nd, T*NE, maxd, nd?sumd/nd:0.0, maxd/cfg::ROUTER_SCORE_QUANTUM);

        // route() 의 결정 여유(margin): select 가 수행한 모든 비교에서
        // |score[e] - (best + TIE_EPS)| 의 최소값. 이것이 1 quantum 보다 작은 토큰만
        // logit 이 경계를 넘을 때 실제로 expert 가 바뀔 수 있다.
        auto route_margin = [](const float* logits, int& first, int& second) {
            float sc[NE];
            for (std::size_t e = 0; e < NE; ++e) {
                const float s0 = logits[e];
                const float r = std::floor(std::fabs(s0)/cfg::ROUTER_SCORE_QUANTUM + 0.5f)*cfg::ROUTER_SCORE_QUANTUM;
                sc[e] = s0 < 0.0f ? -r : r;
            }
            double margin = std::numeric_limits<double>::infinity();
            auto select = [&](int excluded) {
                int best = -1; float bv = -std::numeric_limits<float>::infinity();
                for (std::size_t e = 0; e < NE; ++e) {
                    if ((int)e == excluded) continue;
                    if (best >= 0) margin = std::min(margin, std::fabs((double)sc[e] - (double)bv - cfg::ROUTER_TIE_EPS));
                    if (best < 0 || sc[e] > bv + cfg::ROUTER_TIE_EPS) { best = (int)e; bv = sc[e]; }
                }
                return best;
            };
            first = select(-1); second = select(first);
            return margin;
        };

        std::size_t set_flip = 0, order_flip = 0, tight = 0, near_boundary = 0, both = 0;
        for (std::size_t t = 0; t < T; ++t) {
            int a1,b1,a2,b2;
            const double m = route_margin(r1.data()+t*NE, a1, b1);
            route_row(r2.data()+t*NE, a2, b2);
            if (!(a1==a2 && b1==b2)) { if (a1==b2 && b1==a2) ++order_flip; else ++set_flip; }
            const bool t_tight = m < cfg::ROUTER_SCORE_QUANTUM;
            bool t_near = false;
            for (std::size_t e = 0; e < NE; ++e) {
                const double u = std::fabs((double)r1[t*NE+e]) / cfg::ROUTER_SCORE_QUANTUM;
                if (std::fabs(u - std::floor(u) - 0.5) * cfg::ROUTER_SCORE_QUANTUM < maxd) { t_near = true; break; }
            }
            tight += t_tight; near_boundary += t_near; both += (t_tight && t_near);
        }
        std::printf("REAL  T=%zu  set_flip %zu  order_flip %zu | margin<1quantum %zu (%.3f%%)  "
                    "logit within dlogit of boundary %zu (%.3f%%)  both %zu\n",
                    T, set_flip, order_flip, tight, 100.0*tight/T,
                    near_boundary, 100.0*near_boundary/T, both);

        // 취약성 곡선: logit 을 ±delta 흔들면 top-2 집합이 바뀌는 토큰 수.
        for (double delta : {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}) {
            std::vector<float> rp(T*NE);
            unsigned s = 2463534242u;
            for (std::size_t i = 0; i < T*NE; ++i) {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                rp[i] = r1[i] + (float)((s & 1u) ? delta : -delta);
            }
            std::size_t sf = 0, of = 0;
            for (std::size_t t = 0; t < T; ++t) {
                int a1,b1,a2,b2;
                route_row(r1.data()+t*NE, a1, b1);
                route_row(rp.data()+t*NE, a2, b2);
                if (!(a1==a2 && b1==b2)) { if (a1==b2 && b1==a2) ++of; else ++sf; }
            }
            std::printf("FRAGILE delta %.0e  set_flip %6zu / %zu (%.5f%%)  order_flip %6zu\n",
                        delta, sf, T, 100.0*sf/T, of);
        }
        cudaFree(d_x); cudaFree(d_ys); cudaFree(d_yt); cudaFree(d_w);
        cudaFree(d_b); cudaFree(d_g); cudaFree(d_r1); cudaFree(d_r2);
    }

    /* ---- 직렬 norm 이 T 에 따라 어떻게 늘어나는가 (block=행, thread 1개) ---- */
    {
        float *d_x, *d_y;
        const std::size_t TM = 19803;
        ck(cudaMalloc(&d_x, TM*H*sizeof(float)), "sx");
        ck(cudaMalloc(&d_y, TM*H*sizeof(float)), "sy");
        ck(cudaMemset(d_x, 0, TM*H*sizeof(float)), "mx");
        const std::size_t shmem = (H + 2) * sizeof(float);
        for (std::size_t T : {1u, 8u, 82u, 328u, 1312u, 5248u, 19803u}) {
            layer_norm_rows<<<(unsigned)T, NORM_BLOCK, shmem>>>(d_x, d_nw, d_nb, d_y, (int)H, cfg::NORM_EPS);
            ck(cudaDeviceSynchronize(), "w");
            const int reps = T <= 1312 ? 100 : 10;
            double s = now();
            for (int r = 0; r < reps; ++r)
                layer_norm_rows<<<(unsigned)T, NORM_BLOCK, shmem>>>(d_x, d_nw, d_nb, d_y, (int)H, cfg::NORM_EPS);
            ck(cudaDeviceSynchronize(), "r");
            const double t1 = (now()-s)/reps;
            s = now();
            for (int r = 0; r < reps; ++r)
                layer_norm_rows_tree<<<(unsigned)T, NORM_BLOCK>>>(d_x, d_nw, d_nb, d_y, (int)H, cfg::NORM_EPS);
            ck(cudaDeviceSynchronize(), "r2");
            const double t2 = (now()-s)/reps;
            std::printf("SCALE rows=%-6zu serial_ms %8.4f  tree_ms %8.4f  ratio %5.2f  "
                        "serial_GBs %6.2f\n", T, t1*1e3, t2*1e3, t1/t2,
                        2.0*T*H*4/t1/1e9);
        }
        cudaFree(d_x); cudaFree(d_y);
    }
    return 0;
}
