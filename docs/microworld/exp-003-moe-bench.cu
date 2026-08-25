// exp-003-moe.html 에 박힌 측정값을 뽑은 마이크로벤치.
//
// 한 레이어의 MoE 블록만 떼어내, 같은 입력·같은 라우팅 결과에 대해
//   BEFORE = layer.cu 의 PhiMoE::forward 경로 (host, tensor_ops, at())
//   AFTER  = EXP-003(39ee87c) 의 DeviceMoE 경로 (gemm_nt_bias + 커널 3개)
// 를 나란히 잰다. 두 경로의 커널/루프는 원본에서 그대로 복사했다.
// 가중치는 난수다 (model.bin 15GB 를 읽지 않는다). 연산량·shape 는 실제와 동일.
//
//   make                     # obj/tensor.o 생성
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 -Xcompiler=-march=native \
//       -Xcompiler=-fopenmp -Iinclude docs/microworld/exp-003-moe-bench.cu obj/tensor.o \
//       -o ~/moebench -L/usr/local/cuda/lib64 -lcudart -lm -lgomp
//   srun --partition aps --gres=gpu:1 --exclusive ~/moebench
//
// 측정 기록: 2026-08-24. 결과는 exp-003-moe.html 의 MEAS 상수.
#include "tensor.h"
#include "config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <vector>

static double now() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static void fill_rand(Tensor& t, unsigned seed){unsigned s=seed;for(std::size_t i=0;i<t.size();++i){s=s*1664525u+1013904223u;t[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}}
static void ck(cudaError_t e, const char* what){ if(e!=cudaSuccess){ std::printf("CUDA FAIL %s: %s\n",what,cudaGetErrorString(e)); std::exit(1);} }

namespace cfg = apss26;
constexpr std::size_t H  = cfg::HIDDEN_SIZE;              // 4096
constexpr std::size_t FF = cfg::EXPERT_INTERMEDIATE_SIZE;  // 448
constexpr std::size_t NE = cfg::NUM_EXPERTS;               // 16

/* ---------------- AFTER: EXP-003 커널 원문 ---------------- */
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
__global__ void gather_rows(const float* __restrict__ src, const int* __restrict__ index,
                            float* __restrict__ dst, int cols) {
    const int row = index[blockIdx.x];
    const float* in = src + static_cast<long long>(row) * cols;
    float* out = dst + static_cast<long long>(blockIdx.x) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = in[c];
}
__global__ void scatter_add_rows(const float* __restrict__ src, const int* __restrict__ index,
                                 float* __restrict__ dst, int cols, float weight) {
    const int row = index[blockIdx.x];
    const float* in = src + static_cast<long long>(blockIdx.x) * cols;
    float* out = dst + static_cast<long long>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] += weight * in[c];
}
__global__ void silu_mul(float* __restrict__ gate, const float* __restrict__ up, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { const float x = gate[i]; gate[i] = (x / (1.0f + expf(-x))) * up[i]; }
}
static void dgemm_nt(const float* x, const float* w, float* y, std::size_t rows,
                     std::size_t in, std::size_t out) {
    const dim3 block(TILE, TILE);
    const dim3 grid((unsigned)((out + TILE - 1) / TILE), (unsigned)((rows + TILE - 1) / TILE));
    gemm_nt_bias<<<grid, block>>>(x, w, nullptr, y, (int)rows, (int)in, (int)out);
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
    /* ---- 한 레이어의 MoE 가중치 (난수, shape 는 실제) ---- */
    Tensor gate_w({NE, H});
    std::vector<Tensor> w1, w2, w3;
    fill_rand(gate_w, 101);
    for (std::size_t e = 0; e < NE; ++e) {
        w1.emplace_back(std::vector<std::size_t>{FF, H});
        w3.emplace_back(std::vector<std::size_t>{FF, H});
        w2.emplace_back(std::vector<std::size_t>{H, FF});
        fill_rand(w1[e], 200u + e); fill_rand(w3[e], 400u + e); fill_rand(w2[e], 600u + e);
    }
    /* ---- 같은 가중치를 device 로 ---- */
    float *d_gate_w, *d_w1[NE], *d_w2[NE], *d_w3[NE];
    ck(cudaMalloc(&d_gate_w, NE*H*sizeof(float)), "malloc gate");
    ck(cudaMemcpy(d_gate_w, gate_w.data(), NE*H*sizeof(float), cudaMemcpyHostToDevice), "cp gate");
    for (std::size_t e = 0; e < NE; ++e) {
        ck(cudaMalloc(&d_w1[e], FF*H*sizeof(float)), "malloc w1");
        ck(cudaMalloc(&d_w3[e], FF*H*sizeof(float)), "malloc w3");
        ck(cudaMalloc(&d_w2[e], H*FF*sizeof(float)), "malloc w2");
        ck(cudaMemcpy(d_w1[e], w1[e].data(), FF*H*sizeof(float), cudaMemcpyHostToDevice), "cp w1");
        ck(cudaMemcpy(d_w3[e], w3[e].data(), FF*H*sizeof(float), cudaMemcpyHostToDevice), "cp w3");
        ck(cudaMemcpy(d_w2[e], w2[e].data(), H*FF*sizeof(float), cudaMemcpyHostToDevice), "cp w2");
    }
    std::printf("weights_per_layer_MB %.1f\n", (NE*(2.0*FF*H + H*FF) + NE*H) * 4.0 / 1048576.0);

    const std::size_t Ts[] = {165, 330, 660, 2640, 9900, 19803};
    const std::size_t HOST_MAX = 660;   // host 경로는 이 이상이면 분 단위라 재지 않는다

    for (std::size_t T : Ts) {
        Tensor post({T, H}); fill_rand(post, 7u + (unsigned)T);

        /* ---- 라우팅은 한 번만 계산해 두 경로가 완전히 같은 행 집합을 돌게 한다 ---- */
        Tensor router({T, NE});
        tensor_ops::matmul_transposed(post, gate_w, router);
        std::vector<std::vector<int>> assign(NE);
        for (std::size_t t = 0; t < T; ++t) {
            int a = 0, b = 0; route_row(router.data() + t*NE, a, b);
            assign[a].push_back((int)t); assign[b].push_back((int)t);
        }
        std::printf("ROWS T=%-6zu", T);
        for (std::size_t e = 0; e < NE; ++e) std::printf(" %zu", assign[e].size());
        std::printf("\n");

        /* ================= BEFORE: host PhiMoE::forward ================= */
        if (T <= HOST_MAX) {
            Tensor y({T, H}); y.zero();
            double t_gate=0, t_route=0, t_gather=0, t_w1=0, t_silu=0, t_w3=0, t_mul=0, t_w2=0, t_scatter=0;
            const double h0 = now();
            double s = now();
            tensor_ops::matmul_transposed(post, gate_w, router);
            t_gate = now() - s;
            // layer.cu: 토큰마다 Tensor({16}) 을 새로 만들고 at() 으로 채운 뒤 route()
            s = now();
            std::vector<std::vector<std::pair<std::size_t,float>>> as2(NE);
            for (std::size_t t = 0; t < T; ++t) {
                Tensor one({NE});
                for (std::size_t e = 0; e < NE; ++e) one[e] = router.at(t, e);
                int a=0,b=0; route_row(one.data(), a, b);
                as2[a].emplace_back(t, 0.5f); as2[b].emplace_back(t, 0.5f);
            }
            t_route = now() - s;
            for (std::size_t e = 0; e < NE; ++e) {
                const std::size_t rows = as2[e].size();
                if (!rows) continue;
                Tensor input({rows, H});
                s = now();
                for (std::size_t i = 0; i < rows; ++i)
                    for (std::size_t j = 0; j < H; ++j) input.at(i,j) = post.at(as2[e][i].first, j);
                t_gather += now() - s;
                Tensor g({rows,FF}), up({rows,FF}), act({rows,FF}), out({rows,H});
                s = now(); tensor_ops::matmul_transposed(input, w1[e], g);   t_w1   += now()-s;
                s = now(); tensor_ops::silu(g, act);                          t_silu += now()-s;
                s = now(); tensor_ops::matmul_transposed(input, w3[e], up);   t_w3   += now()-s;
                s = now(); tensor_ops::mul(act, up, act);                     t_mul  += now()-s;
                s = now(); tensor_ops::matmul_transposed(act, w2[e], out);    t_w2   += now()-s;
                s = now();
                for (std::size_t i = 0; i < rows; ++i)
                    for (std::size_t j = 0; j < H; ++j) y.at(as2[e][i].first, j) += as2[e][i].second * out.at(i,j);
                t_scatter += now() - s;
            }
            const double htot = now() - h0;
            std::printf("HOST T=%-6zu total %8.4f  gate %7.4f route %7.4f gather %7.4f "
                        "w1 %7.4f silu %7.4f w3 %7.4f mul %7.4f w2 %7.4f scatter %7.4f\n",
                        T, htot, t_gate, t_route, t_gather, t_w1, t_silu, t_w3, t_mul, t_w2, t_scatter);
        }

        /* ================= AFTER: EXP-003 DeviceMoE ================= */
        float *d_in, *d_ff, *d_ein, *d_eout, *d_g, *d_u, *d_router; int* d_idx;
        ck(cudaMalloc(&d_in,    T*H*sizeof(float)),  "d_in");
        ck(cudaMalloc(&d_ff,    T*H*sizeof(float)),  "d_ff");
        ck(cudaMalloc(&d_ein,   T*H*sizeof(float)),  "d_ein");
        ck(cudaMalloc(&d_eout,  T*H*sizeof(float)),  "d_eout");
        ck(cudaMalloc(&d_g,     T*FF*sizeof(float)), "d_g");
        ck(cudaMalloc(&d_u,     T*FF*sizeof(float)), "d_u");
        ck(cudaMalloc(&d_router,T*NE*sizeof(float)), "d_router");
        ck(cudaMalloc(&d_idx,   T*sizeof(int)),      "d_idx");
        Tensor ff({T,H});

        // (1) 한 번만 sync 하는 정직한 총시간. model.cu 의 레이어당 MoE 구간과 같은 순서.
        for (int rep = 0; rep < 2; ++rep) {
            ck(cudaDeviceSynchronize(), "pre");
            const double d0 = now();
            ck(cudaMemcpy(d_in, post.data(), T*H*sizeof(float), cudaMemcpyHostToDevice), "h2d");
            dgemm_nt(d_in, d_gate_w, d_router, T, H, NE);
            ck(cudaMemcpy(router.data(), d_router, T*NE*sizeof(float), cudaMemcpyDeviceToHost), "d2h router");
            std::vector<std::vector<int>> a2(NE);
            for (std::size_t t = 0; t < T; ++t) {
                int a=0,b=0; route_row(router.data()+t*NE, a, b);
                a2[a].push_back((int)t); a2[b].push_back((int)t);
            }
            ck(cudaMemset(d_ff, 0, T*H*sizeof(float)), "memset");
            for (std::size_t e = 0; e < NE; ++e) {
                const std::size_t rows = a2[e].size();
                if (!rows) continue;
                ck(cudaMemcpy(d_idx, a2[e].data(), rows*sizeof(int), cudaMemcpyHostToDevice), "idx");
                gather_rows<<<(unsigned)rows,256>>>(d_in, d_idx, d_ein, (int)H);
                dgemm_nt(d_ein, d_w1[e], d_g, rows, H, FF);
                dgemm_nt(d_ein, d_w3[e], d_u, rows, H, FF);
                const long long act = (long long)rows*FF;
                silu_mul<<<(unsigned)((act+255)/256),256>>>(d_g, d_u, act);
                dgemm_nt(d_g, d_w2[e], d_eout, rows, FF, H);
                scatter_add_rows<<<(unsigned)rows,256>>>(d_eout, d_idx, d_ff, (int)H, 0.5f);
            }
            ck(cudaGetLastError(), "kernels");
            ck(cudaMemcpy(ff.data(), d_ff, T*H*sizeof(float), cudaMemcpyDeviceToHost), "d2h ff");
            const double dtot = now() - d0;
            if (rep) std::printf("DEV  T=%-6zu total %8.4f\n", T, dtot);
        }

        // (2) 단계별 분해. 단계마다 sync 하므로 총합은 (1)보다 크다.
        {
            double t_h2d=0,t_gate=0,t_rd2h=0,t_route=0,t_memset=0,t_idx=0,
                   t_gather=0,t_w1=0,t_w3=0,t_silu=0,t_w2=0,t_scatter=0,t_ffd2h=0;
            auto tick=[&](double& acc,double s){ ck(cudaDeviceSynchronize(),"sync"); acc += now()-s; };
            double s = now();
            ck(cudaMemcpy(d_in, post.data(), T*H*sizeof(float), cudaMemcpyHostToDevice), "h2d"); tick(t_h2d,s);
            s = now(); dgemm_nt(d_in, d_gate_w, d_router, T, H, NE); tick(t_gate,s);
            s = now(); ck(cudaMemcpy(router.data(), d_router, T*NE*sizeof(float), cudaMemcpyDeviceToHost),"rd2h"); tick(t_rd2h,s);
            s = now();
            std::vector<std::vector<int>> a2(NE);
            for (std::size_t t = 0; t < T; ++t) { int a=0,b=0; route_row(router.data()+t*NE,a,b); a2[a].push_back((int)t); a2[b].push_back((int)t); }
            t_route += now()-s;
            s = now(); ck(cudaMemset(d_ff,0,T*H*sizeof(float)),"memset"); tick(t_memset,s);
            for (std::size_t e = 0; e < NE; ++e) {
                const std::size_t rows = a2[e].size(); if (!rows) continue;
                s=now(); ck(cudaMemcpy(d_idx,a2[e].data(),rows*sizeof(int),cudaMemcpyHostToDevice),"idx"); tick(t_idx,s);
                s=now(); gather_rows<<<(unsigned)rows,256>>>(d_in,d_idx,d_ein,(int)H); tick(t_gather,s);
                s=now(); dgemm_nt(d_ein,d_w1[e],d_g,rows,H,FF); tick(t_w1,s);
                s=now(); dgemm_nt(d_ein,d_w3[e],d_u,rows,H,FF); tick(t_w3,s);
                const long long act=(long long)rows*FF;
                s=now(); silu_mul<<<(unsigned)((act+255)/256),256>>>(d_g,d_u,act); tick(t_silu,s);
                s=now(); dgemm_nt(d_g,d_w2[e],d_eout,rows,FF,H); tick(t_w2,s);
                s=now(); scatter_add_rows<<<(unsigned)rows,256>>>(d_eout,d_idx,d_ff,(int)H,0.5f); tick(t_scatter,s);
            }
            s=now(); ck(cudaMemcpy(ff.data(),d_ff,T*H*sizeof(float),cudaMemcpyDeviceToHost),"ffd2h"); tick(t_ffd2h,s);
            std::printf("DEVBRK T=%-6zu h2d %7.4f gate %7.4f rd2h %7.4f route %7.4f memset %7.4f idx %7.4f "
                        "gather %7.4f w1 %7.4f w3 %7.4f silu %7.4f w2 %7.4f scatter %7.4f ffd2h %7.4f\n",
                        T,t_h2d,t_gate,t_rd2h,t_route,t_memset,t_idx,t_gather,t_w1,t_w3,t_silu,t_w2,t_scatter,t_ffd2h);
        }
        cudaFree(d_in); cudaFree(d_ff); cudaFree(d_ein); cudaFree(d_eout);
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_router); cudaFree(d_idx);
    }

    /* ---- 행 수가 GEMM 효율을 어떻게 바꾸는가 (TILE=32) ---- */
    {
        const std::size_t Rs[] = {1,2,4,8,16,32,33,48,64,128,256,512,1024,2048,4096};
        float *d_x,*d_y1,*d_y2;
        ck(cudaMalloc(&d_x, 4096*H*sizeof(float)),"x");
        ck(cudaMalloc(&d_y1,4096*FF*sizeof(float)),"y1");
        ck(cudaMalloc(&d_y2,4096*H*sizeof(float)),"y2");
        ck(cudaMemset(d_x,0,4096*H*sizeof(float)),"mx");
        for (std::size_t R : Rs) {
            dgemm_nt(d_x,d_w1[0],d_y1,R,H,FF); ck(cudaDeviceSynchronize(),"warm");
            const int reps = R<=64?200:20;
            double s=now(); for(int r=0;r<reps;++r) dgemm_nt(d_x,d_w1[0],d_y1,R,H,FF);
            ck(cudaDeviceSynchronize(),"s1"); const double t1=(now()-s)/reps;
            s=now(); for(int r=0;r<reps;++r) dgemm_nt(d_y1,d_w2[0],d_y2,R,FF,H);
            ck(cudaDeviceSynchronize(),"s2"); const double t2=(now()-s)/reps;
            std::printf("GEMMROWS R=%-5zu w1_ms %8.4f w1_GFLOPs %8.1f  w2_ms %8.4f w2_GFLOPs %8.1f\n",
                        R,t1*1e3, 2.0*R*H*FF/t1/1e9, t2*1e3, 2.0*R*FF*H/t2/1e9);
        }
        // 커널 3개(gather/silu/scatter)만의 비용
        int* d_idx; ck(cudaMalloc(&d_idx,4096*sizeof(int)),"idx2");
        std::vector<int> idx(4096); for(int i=0;i<4096;++i) idx[i]=i;
        ck(cudaMemcpy(d_idx,idx.data(),4096*sizeof(int),cudaMemcpyHostToDevice),"cpidx");
        for (std::size_t R : {32u,256u,2048u}) {
            const int reps=200;
            double s=now(); for(int r=0;r<reps;++r) gather_rows<<<(unsigned)R,256>>>(d_x,d_idx,d_y2,(int)H);
            ck(cudaDeviceSynchronize(),"g"); const double tg=(now()-s)/reps;
            const long long act=(long long)R*FF;
            s=now(); for(int r=0;r<reps;++r) silu_mul<<<(unsigned)((act+255)/256),256>>>(d_y1,d_y1,act);
            ck(cudaDeviceSynchronize(),"sm"); const double ts=(now()-s)/reps;
            s=now(); for(int r=0;r<reps;++r) scatter_add_rows<<<(unsigned)R,256>>>(d_x,d_idx,d_y2,(int)H,0.5f);
            ck(cudaDeviceSynchronize(),"sc"); const double tc=(now()-s)/reps;
            std::printf("WRAP R=%-5zu gather_ms %8.5f silu_ms %8.5f scatter_ms %8.5f\n",R,tg*1e3,ts*1e3,tc*1e3);
        }
        // 빈 커널 launch 오버헤드 (expert 16개 x 6 launch 의 하한)
        double s=now(); for(int r=0;r<2000;++r) silu_mul<<<1,256>>>(d_y1,d_y1,1);
        ck(cudaDeviceSynchronize(),"lat"); std::printf("LAUNCH us %.3f\n",(now()-s)/2000*1e6);
    }
    return 0;
}
