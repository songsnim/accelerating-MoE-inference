// exp-000-baseline.html 에 박힌 측정값을 뽑은 마이크로벤치.
// 무수정 스켈레톤의 tensor.cu 만 링크한다. 모델 가중치는 읽지 않는다.
//
//   make                     # obj/tensor.o 생성
//   g++ -std=c++17 -O3 -march=native -fopenmp -Iinclude \
//       -c docs/microworld/exp-000-baseline-bench.cpp -o /tmp/b.o
//   g++ -std=c++17 -O3 -fopenmp /tmp/b.o obj/tensor.o -o ~/bench \
//       -L/usr/local/cuda/lib64 -lcudart -lm
//   srun --partition aps --gres=gpu:1 --exclusive ~/bench
//
// 측정 기록: 2026-08-24, OMP 스레드 64. 결과는 exp-000-baseline.html 의 MEAS 상수.
#include "tensor.h"
#include "config.h"
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>
#include <utility>
#include <limits>
#include <omp.h>
#include <limits>
#include <algorithm>
static double now() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static void fill_rand(Tensor& t, unsigned seed){unsigned s=seed;for(std::size_t i=0;i<t.size();++i){s=s*1664525u+1013904223u;t[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}}

static inline float accurate_exp(float x){ return static_cast<float>(std::exp(static_cast<double>(x))); }

// verbatim from src/layer.cu PhiAttention::forward (the q.k score section)
static double attention_loop(const Tensor& q, const Tensor& k, const Tensor& v, std::size_t s, Tensor& out) {
    const std::size_t group = apss26::NUM_ATTENTION_HEADS / apss26::NUM_KV_HEADS;
    const double t0 = now();
    for (std::size_t qi = 0; qi < s; ++qi) for (std::size_t qh = 0; qh < apss26::NUM_ATTENTION_HEADS; ++qh) {
        const std::size_t kh = qh / group;
        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t ki = 0; ki <= qi; ++ki) {
            if (qi - ki >= apss26::SLIDING_WINDOW) continue;
            float score = 0.0f;
            for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) score += q.at(qi, qh * apss26::HEAD_DIM + d) * k.at(ki, kh * apss26::HEAD_DIM + d);
            maxv = std::max(maxv, score / std::sqrt(static_cast<float>(apss26::HEAD_DIM)));
        }
        float denom = 0.0f;
        for (std::size_t ki = 0; ki <= qi; ++ki) if (qi - ki < apss26::SLIDING_WINDOW) {
            float score = 0.0f;
            for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) score += q.at(qi, qh * apss26::HEAD_DIM + d) * k.at(ki, kh * apss26::HEAD_DIM + d);
            denom += accurate_exp(score / std::sqrt(static_cast<float>(apss26::HEAD_DIM)) - maxv);
        }
        for (std::size_t d = 0; d < apss26::HEAD_DIM; ++d) {
            float value = 0.0f;
            for (std::size_t ki = 0; ki <= qi; ++ki) if (qi - ki < apss26::SLIDING_WINDOW) {
                float score = 0.0f;
                for (std::size_t z = 0; z < apss26::HEAD_DIM; ++z) score += q.at(qi, qh * apss26::HEAD_DIM + z) * k.at(ki, kh * apss26::HEAD_DIM + z);
                value += accurate_exp(score / std::sqrt(static_cast<float>(apss26::HEAD_DIM)) - maxv) / denom * v.at(ki, kh * apss26::HEAD_DIM + d);
            }
            out.at(qi, qh * apss26::HEAD_DIM + d) = value;
        }
    }
    return now() - t0;
}

static void mm(const char* tag,std::size_t M,std::size_t K,std::size_t N,int reps){
    Tensor a({M,K}),b({N,K}),c({M,N}); fill_rand(a,1); fill_rand(b,2);
    tensor_ops::matmul_transposed(a,b,c);
    double t0=now(); for(int r=0;r<reps;++r) tensor_ops::matmul_transposed(a,b,c);
    double dt=(now()-t0)/reps;
    std::printf("MM %-8s K=%-5zu N=%-6zu M=%-4zu perrow_ms %8.3f  nsPerMAC %6.3f\n",tag,K,N,M,dt*1e3,dt*1e9/(double(K)*N));
}
int main(){
    std::printf("omp_max_threads=%d\n",omp_get_max_threads());
    const std::size_t Ms[]={1,2,4,8,14,19,22,24,28,32,40,46,64};
    for(std::size_t M:Ms) mm("q",M,4096,2048,2);
    for(std::size_t M:Ms) mm("kv",M,4096,512,4);
    for(std::size_t M:Ms) mm("o",M,2048,4096,2);
    for(std::size_t M:Ms) mm("w13",M,4096,448,4);
    for(std::size_t M:Ms) mm("w2",M,448,4096,4);
    for(std::size_t M:{1u,8u,32u}) mm("gate",M,4096,16,20);
    for(std::size_t M:{1u,2u,4u}) mm("lm_head",M,4096,32064,2);

    // elementwise / norm / rope / routing paths, at the real per-layer sizes
    for(std::size_t s:{14u,32u}){
        Tensor x({s,apss26::HIDDEN_SIZE}),y({s,apss26::HIDDEN_SIZE}),w({apss26::HIDDEN_SIZE}),b({apss26::HIDDEN_SIZE});
        fill_rand(x,7); fill_rand(w,8); fill_rand(b,9);
        double t0=now(); for(int r=0;r<20;++r) tensor_ops::layer_norm(x,w,b,apss26::NORM_EPS,y); double ln=(now()-t0)/20;
        t0=now(); for(int r=0;r<20;++r) tensor_ops::add_inplace(y,x); double ad=(now()-t0)/20;
        Tensor g({s,apss26::EXPERT_INTERMEDIATE_SIZE}),g2({s,apss26::EXPERT_INTERMEDIATE_SIZE}); fill_rand(g,10);
        t0=now(); for(int r=0;r<20;++r) tensor_ops::silu(g,g2); double si=(now()-t0)/20;
        t0=now(); for(int r=0;r<20;++r) tensor_ops::mul(g,g2,g2); double mu=(now()-t0)/20;
        Tensor q({s,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM}),k({s,apss26::NUM_KV_HEADS*apss26::HEAD_DIM});
        fill_rand(q,11); fill_rand(k,12);
        t0=now(); for(int r=0;r<5;++r) tensor_ops::apply_rope(q,k,s,apss26::NUM_ATTENTION_HEADS,apss26::NUM_KV_HEADS,apss26::HEAD_DIM,apss26::ROPE_THETA); double rp=(now()-t0)/5;
        // MoE host bookkeeping: per-token Tensor({16}) + gather/scatter via at()
        Tensor router({s,apss26::NUM_EXPERTS}); fill_rand(router,13);
        t0=now();
        for(int r=0;r<20;++r){ double acc=0;
          for(std::size_t t=0;t<s;++t){ Tensor one({apss26::NUM_EXPERTS});
            for(std::size_t e=0;e<apss26::NUM_EXPERTS;++e) one[e]=router.at(t,e); acc+=one[0]; }
          if(acc==1234.5) std::printf(" ");
        }
        double al=(now()-t0)/20;
        Tensor src({s,apss26::HIDDEN_SIZE}),dst({s,apss26::HIDDEN_SIZE}); fill_rand(src,14);
        t0=now();
        for(int r=0;r<20;++r) for(std::size_t i=0;i<s;++i) for(std::size_t j=0;j<apss26::HIDDEN_SIZE;++j) dst.at(i,j)=src.at(i,j);
        double gs=(now()-t0)/20;
        std::printf("OTHER s=%-3zu ln_ms %7.4f add_ms %7.4f silu_ms %7.4f mul_ms %7.4f rope_ms %7.4f alloc16_ms %7.4f gather_ms %7.4f\n",
                    s,ln*1e3,ad*1e3,si*1e3,mu*1e3,rp*1e3,al*1e3,gs*1e3);
    }
    for(std::size_t s:{8u,14u,19u,32u}){
        Tensor q({s,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM}),k({s,apss26::NUM_KV_HEADS*apss26::HEAD_DIM});
        Tensor v({s,apss26::NUM_KV_HEADS*apss26::HEAD_DIM}),o({s,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM});
        fill_rand(q,3); fill_rand(k,4); fill_rand(v,5);
        const double dt=attention_loop(q,k,v,s,o);
        const double mac=double(s)*(s+1)/2*apss26::NUM_ATTENTION_HEADS*130.0*apss26::HEAD_DIM;
        std::printf("ATTN s=%-3zu %8.5f s/layer  nsPerMAC %6.3f  (x32 layers = %7.3f s)\\n",s,dt,dt*1e9/mac,dt*32);
    }
    return 0;
}
