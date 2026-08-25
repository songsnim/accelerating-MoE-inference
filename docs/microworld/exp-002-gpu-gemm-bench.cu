// exp-002-gpu-gemm.html 에 박힌 측정값을 뽑은 마이크로벤치.
// 무수정 tensor.cu(obj/tensor.o)의 host matmul_transposed 와, EXP-002 가 추가한
// gemm_nt_bias 커널을 같은 shape·같은 바이너리에서 나란히 잰다.
//
//   make                     # obj/tensor.o 생성
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 -Xcompiler=-march=native \
//       -Xcompiler=-fopenmp -Iinclude -c docs/microworld/exp-002-gpu-gemm-bench.cu -o /tmp/b2.o
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 /tmp/b2.o obj/tensor.o \
//       -o /tmp/bench2 -L/usr/local/cuda/lib64 -lcudart -lm -lgomp
//   srun --partition aps --gres=gpu:1 --exclusive /tmp/bench2
//
// 측정 기록: 2026-08-24. 결과는 exp-002-gpu-gemm.html 의 MEAS 상수.
#include "tensor.h"
#include "config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <vector>

static double now() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static void fill_rand(Tensor& t, unsigned seed){unsigned s=seed;for(std::size_t i=0;i<t.size();++i){s=s*1664525u+1013904223u;t[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}}
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){std::printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(1);} }while(0)

// ---- src/model.cu 의 gemm_nt_bias 원문. TILE 만 템플릿으로 뺐다 (원문은 TILE=32). ----
template <int TILE>
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

// ---- 대조군: shared memory 를 쓰지 않는 커널. 같은 결과, 같은 k 순서. ----
__global__ void gemm_nt_bias_naive(const float* __restrict__ a, const float* __restrict__ b,
                                   const float* __restrict__ bias, float* __restrict__ c,
                                   int m, int k, int n) {
    const int row = blockIdx.y * 32 + threadIdx.y;
    const int col = blockIdx.x * 32 + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i)
        acc += a[static_cast<long long>(row) * k + i] * b[static_cast<long long>(col) * k + i];
    if (bias != nullptr) acc += bias[col];
    c[static_cast<long long>(row) * n + col] = acc;
}

template <int TILE>
static double time_tiled(const float* da,const float* db,const float* dbias,float* dc,
                         int M,int K,int N,int reps){
    dim3 blk(TILE,TILE), grd((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    gemm_nt_bias<TILE><<<grd,blk>>>(da,db,dbias,dc,M,K,N); CK(cudaDeviceSynchronize());
    double t0=now();
    for(int r=0;r<reps;++r) gemm_nt_bias<TILE><<<grd,blk>>>(da,db,dbias,dc,M,K,N);
    CK(cudaDeviceSynchronize());
    return (now()-t0)/reps;
}

// ---- 대조군: shared memory 패딩([TILE][TILE])을 뺀 것. 원문은 [TILE][TILE+1]. ----
template <int TILE>
__global__ void gemm_nt_bias_nopad(const float* __restrict__ a, const float* __restrict__ b,
                                   const float* __restrict__ bias, float* __restrict__ c,
                                   int m, int k, int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
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

template <int TILE>
static double time_nopad(const float* da,const float* db,const float* dbias,float* dc,
                         int M,int K,int N,int reps){
    dim3 blk(TILE,TILE), grd((N+TILE-1)/TILE,(M+TILE-1)/TILE);
    gemm_nt_bias_nopad<TILE><<<grd,blk>>>(da,db,dbias,dc,M,K,N); CK(cudaDeviceSynchronize());
    double t0=now();
    for(int r=0;r<reps;++r) gemm_nt_bias_nopad<TILE><<<grd,blk>>>(da,db,dbias,dc,M,K,N);
    CK(cudaDeviceSynchronize());
    return (now()-t0)/reps;
}

struct Shape { const char* tag; std::size_t K, N; };

int main(){
    std::printf("omp_max_threads=%d\n",omp_get_max_threads());
    int dev=0; cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p,dev));
    std::printf("gpu=%s sm=%d.%d SMs=%d clock=%.2fGHz shmemPerBlock=%zu\n",
                p.name,p.major,p.minor,p.multiProcessorCount,p.clockRate/1e6,p.sharedMemPerBlock);

    // EXP-002 가 device 로 옮긴 shape 들. 마지막은 lm_head (M=batch).
    const Shape shapes[] = {
        {"q",   apss26::HIDDEN_SIZE, apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM},
        {"kv",  apss26::HIDDEN_SIZE, apss26::NUM_KV_HEADS*apss26::HEAD_DIM},
        {"o",   apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM, apss26::HIDDEN_SIZE},
        {"lm",  apss26::HIDDEN_SIZE, apss26::VOCAB_SIZE},
    };
    const std::size_t Ms[] = {1,8,32,64,165,330,660,1320};

    for (const Shape& sh : shapes) {
        for (std::size_t M : Ms) {
            const bool big = sh.K*sh.N > 20000000ull;      // lm_head
            if (big && M > 165) continue;                   // host 가 너무 오래 걸린다
            Tensor a({M,sh.K}), b({sh.N,sh.K}), bias({sh.N}), c({M,sh.N});
            fill_rand(a,1); fill_rand(b,2); fill_rand(bias,3);

            // --- before: host matmul_transposed + bias 를 별도 루프로 (Linear::forward 원문) ---
            const int hreps = (double(M)*sh.K*sh.N > 2e9) ? 1 : 2;
            tensor_ops::matmul_transposed(a,b,c);
            double t0=now();
            for(int r=0;r<hreps;++r){
                tensor_ops::matmul_transposed(a,b,c);
                for(std::size_t i=0;i<M;++i) for(std::size_t j=0;j<sh.N;++j) c.at(i,j)+=bias[j];
            }
            const double host=(now()-t0)/hreps;
            Tensor href(c.shape());
            for(std::size_t i=0;i<c.size();++i) href[i]=c[i];

            // --- after: 가중치는 이미 device 에 있다. 커널만 잰다. ---
            float *da,*db,*dbias,*dc;
            CK(cudaMalloc(&da,M*sh.K*4)); CK(cudaMalloc(&db,sh.N*sh.K*4));
            CK(cudaMalloc(&dbias,sh.N*4)); CK(cudaMalloc(&dc,M*sh.N*4));
            CK(cudaMemcpy(da,a.data(),M*sh.K*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(db,b.data(),sh.N*sh.K*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dbias,bias.data(),sh.N*4,cudaMemcpyHostToDevice));

            const int reps = 20;
            const double t32 = time_tiled<32>(da,db,dbias,dc,M,sh.K,sh.N,reps);
            const double t16 = time_tiled<16>(da,db,dbias,dc,M,sh.K,sh.N,reps);
            const double t8  = time_tiled<8>(da,db,dbias,dc,M,sh.K,sh.N,reps);

            dim3 nb(32,32), ng((sh.N+31)/32,(M+31)/32);
            gemm_nt_bias_naive<<<ng,nb>>>(da,db,dbias,dc,M,sh.K,sh.N); CK(cudaDeviceSynchronize());
            double n0=now();
            for(int r=0;r<reps;++r) gemm_nt_bias_naive<<<ng,nb>>>(da,db,dbias,dc,M,sh.K,sh.N);
            CK(cudaDeviceSynchronize());
            const double tnaive=(now()-n0)/reps;

            // TILE=32 결과로 정확도 확인
            time_tiled<32>(da,db,dbias,dc,M,sh.K,sh.N,1);
            Tensor dout(c.shape());
            CK(cudaMemcpy(dout.data(),dc,M*sh.N*4,cudaMemcpyDeviceToHost));
            double maxdiff=0, maxabs=0;
            for(std::size_t i=0;i<dout.size();++i){
                maxdiff=std::max(maxdiff,double(std::fabs(dout[i]-href[i])));
                maxabs =std::max(maxabs, double(std::fabs(href[i])));
            }
            const double mac=double(M)*sh.K*sh.N;
            std::printf("GEMM %-4s K=%-5zu N=%-6zu M=%-5zu host_ms %9.3f  t32_ms %8.4f  t16_ms %8.4f  "
                        "t8_ms %8.4f  naive_ms %8.4f  host_GMACs %6.3f  t32_GMACs %8.2f  "
                        "speedup %8.2f  maxdiff %.3e  maxabs %.3e\n",
                        sh.tag,sh.K,sh.N,M,host*1e3,t32*1e3,t16*1e3,t8*1e3,tnaive*1e3,
                        mac/host/1e9,mac/t32/1e9,host/t32,maxdiff,maxabs);
            cudaFree(da);cudaFree(db);cudaFree(dbias);cudaFree(dc);
        }
    }

    // --- shared memory 패딩 하나만 뺀 대조군 (device only) ---
    for (const Shape& sh : shapes) {
        for (std::size_t M : {165u, 1320u}) {
            if (sh.K*sh.N > 20000000ull && M > 165) continue;
            Tensor a({M,sh.K}), b({sh.N,sh.K}), bias({sh.N});
            fill_rand(a,1); fill_rand(b,2); fill_rand(bias,3);
            float *da,*db,*dbias,*dc;
            CK(cudaMalloc(&da,M*sh.K*4)); CK(cudaMalloc(&db,sh.N*sh.K*4));
            CK(cudaMalloc(&dbias,sh.N*4)); CK(cudaMalloc(&dc,M*sh.N*4));
            CK(cudaMemcpy(da,a.data(),M*sh.K*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(db,b.data(),sh.N*sh.K*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dbias,bias.data(),sh.N*4,cudaMemcpyHostToDevice));
            const double pad32 = time_tiled<32>(da,db,dbias,dc,M,sh.K,sh.N,20);
            const double no32  = time_nopad<32>(da,db,dbias,dc,M,sh.K,sh.N,20);
            const double pad16 = time_tiled<16>(da,db,dbias,dc,M,sh.K,sh.N,20);
            const double no16  = time_nopad<16>(da,db,dbias,dc,M,sh.K,sh.N,20);
            std::printf("PAD  %-4s K=%-5zu N=%-6zu M=%-5zu pad32_ms %8.4f  nopad32_ms %8.4f  ratio %5.2f  "
                        "pad16_ms %8.4f  nopad16_ms %8.4f  ratio %5.2f\n",
                        sh.tag,sh.K,sh.N,M,pad32*1e3,no32*1e3,no32/pad32,pad16*1e3,no16*1e3,no16/pad16);
            cudaFree(da);cudaFree(db);cudaFree(dbias);cudaFree(dc);
        }
    }

    // --- 버스: EXP-002 에서 레이어마다 실제로 건너는 것들 (pageable memcpy) ---
    const std::size_t widths[] = {apss26::HIDDEN_SIZE,
                                  apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM,
                                  apss26::NUM_KV_HEADS*apss26::HEAD_DIM};
    for (std::size_t T : {165u, 660u, 1320u}) {
        for (std::size_t W : widths) {
            Tensor h({T,W}); fill_rand(h,4);
            float* d; CK(cudaMalloc(&d,T*W*4));
            const std::size_t bytes=T*W*4;
            CK(cudaMemcpy(d,h.data(),bytes,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
            double t0=now(); for(int r=0;r<30;++r) CK(cudaMemcpy(d,h.data(),bytes,cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize()); const double h2d=(now()-t0)/30;
            t0=now(); for(int r=0;r<30;++r) CK(cudaMemcpy(h.data(),d,bytes,cudaMemcpyDeviceToHost));
            CK(cudaDeviceSynchronize()); const double d2h=(now()-t0)/30;
            std::printf("BUS T=%-5zu W=%-5zu MB %8.3f  h2d_ms %7.4f  d2h_ms %7.4f  h2d_GBs %6.2f  d2h_GBs %6.2f\n",
                        T,W,bytes/1048576.0,h2d*1e3,d2h*1e3,bytes/h2d/1e9,bytes/d2h/1e9);
            cudaFree(d);
        }
    }

    // --- 생성자에서 한 번 내는 값: 가중치 업로드 ---
    {
        const std::size_t per_layer = apss26::HIDDEN_SIZE*(apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM
                                     + 2*apss26::NUM_KV_HEADS*apss26::HEAD_DIM)
                                     + apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM*apss26::HIDDEN_SIZE;
        const std::size_t lm = apss26::HIDDEN_SIZE*apss26::VOCAB_SIZE;
        const std::size_t total = per_layer*apss26::NUM_LAYERS + lm;
        Tensor chunk({per_layer}); fill_rand(chunk,5);
        float* d; CK(cudaMalloc(&d,per_layer*4));
        CK(cudaMemcpy(d,chunk.data(),per_layer*4,cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
        double t0=now(); for(int r=0;r<10;++r) CK(cudaMemcpy(d,chunk.data(),per_layer*4,cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize()); const double up=(now()-t0)/10;
        std::printf("UPLOAD per_layer_params %zu  per_layer_MB %.2f  total_params %zu  total_MB %.1f  "
                    "per_layer_ms %7.3f  est_total_s %6.3f\n",
                    per_layer,per_layer*4/1048576.0,total,total*4/1048576.0,up*1e3,
                    up*double(total)/per_layer);
        cudaFree(d);
    }
    return 0;
}
