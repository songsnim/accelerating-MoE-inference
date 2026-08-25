// exp-001-batching.html 에 박힌 측정값을 뽑은 마이크로벤치.
// 무수정 스켈레톤의 tensor.cu 만 링크한다. 모델 가중치는 읽지 않는다.
//
//   make                     # obj/tensor.o 생성
//   g++ -std=c++17 -O3 -march=native -fopenmp -Iinclude \
//       -c docs/microworld/exp-001-batching-bench.cpp -o /tmp/b1.o
//   g++ -std=c++17 -O3 -fopenmp /tmp/b1.o obj/tensor.o -o ~/bench1 \
//       -L/usr/local/cuda/lib64 -lcudart -lm
//   srun --partition aps --gres=gpu:1 --exclusive ~/bench1
//
// EXP-001 이 바꾼 것은 딱 하나 -- matmul 에 들어가는 행 수 M.
// before: M = 시퀀스 하나의 토큰 수(평균 19).  after: M = 배치 전체 토큰 수 T.
// 그래서 이 벤치는 M 을 스윕해 "행 하나의 값"이 M 에 따라 어떻게 무너지는지 잰다.
#include "tensor.h"
#include "config.h"
#include <ctime>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include <algorithm>
#include <omp.h>

static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static void fill_rand(Tensor& t, unsigned seed){unsigned s=seed;for(std::size_t i=0;i<t.size();++i){s=s*1664525u+1013904223u;t[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}}
static inline float accurate_exp(float x){ return static_cast<float>(std::exp(static_cast<double>(x))); }

static void mm(const char* tag,std::size_t M,std::size_t K,std::size_t N,int reps){
    Tensor a({M,K}),b({N,K}),c({M,N}); fill_rand(a,1); fill_rand(b,2);
    tensor_ops::matmul_transposed(a,b,c);                       // warm
    double t0=now(); for(int r=0;r<reps;++r) tensor_ops::matmul_transposed(a,b,c);
    double dt=(now()-t0)/reps;
    std::printf("MM %-8s K=%-5zu N=%-6zu M=%-5zu call_ms %9.3f  perrow_ms %8.4f\n",
                tag,K,N,M,dt*1e3,dt*1e3/M);
    std::fflush(stdout);
}

// ---- before: src/layer.cu PhiAttention::forward 원문 (q.k 를 130번 다시 계산) ----
static double attn_before(const Tensor& q,const Tensor& k,const Tensor& v,std::size_t s,Tensor& out){
    const std::size_t group = apss26::NUM_ATTENTION_HEADS / apss26::NUM_KV_HEADS;
    const double t0=now();
    for (std::size_t qi=0; qi<s; ++qi) for (std::size_t qh=0; qh<apss26::NUM_ATTENTION_HEADS; ++qh) {
        const std::size_t kh = qh/group;
        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t ki=0; ki<=qi; ++ki) {
            if (qi-ki >= apss26::SLIDING_WINDOW) continue;
            float score=0.0f;
            for (std::size_t d=0; d<apss26::HEAD_DIM; ++d) score += q.at(qi,qh*apss26::HEAD_DIM+d)*k.at(ki,kh*apss26::HEAD_DIM+d);
            maxv = std::max(maxv, score/std::sqrt(static_cast<float>(apss26::HEAD_DIM)));
        }
        float denom=0.0f;
        for (std::size_t ki=0; ki<=qi; ++ki) if (qi-ki < apss26::SLIDING_WINDOW) {
            float score=0.0f;
            for (std::size_t d=0; d<apss26::HEAD_DIM; ++d) score += q.at(qi,qh*apss26::HEAD_DIM+d)*k.at(ki,kh*apss26::HEAD_DIM+d);
            denom += accurate_exp(score/std::sqrt(static_cast<float>(apss26::HEAD_DIM))-maxv);
        }
        for (std::size_t d=0; d<apss26::HEAD_DIM; ++d) {
            float value=0.0f;
            for (std::size_t ki=0; ki<=qi; ++ki) if (qi-ki < apss26::SLIDING_WINDOW) {
                float score=0.0f;
                for (std::size_t z=0; z<apss26::HEAD_DIM; ++z) score += q.at(qi,qh*apss26::HEAD_DIM+z)*k.at(ki,kh*apss26::HEAD_DIM+z);
                value += accurate_exp(score/std::sqrt(static_cast<float>(apss26::HEAD_DIM))-maxv)/denom*v.at(ki,kh*apss26::HEAD_DIM+d);
            }
            out.at(qi,qh*apss26::HEAD_DIM+d)=value;
        }
    }
    return now()-t0;
}

// ---- after: src/model.cu attend_sequence 원문 (RoPE 부분은 뺀 score/softmax/value) ----
static void attn_after(const Tensor& q,const Tensor& k,const Tensor& v,
                       std::size_t begin,std::size_t len,Tensor& out){
    constexpr std::size_t D = apss26::HEAD_DIM;
    constexpr std::size_t QH = apss26::NUM_ATTENTION_HEADS;
    constexpr std::size_t KVH = apss26::NUM_KV_HEADS;
    const std::size_t group = QH/KVH;
    const float scale = std::sqrt(static_cast<float>(D));
    const float* qbase = q.data() + begin*QH*D;
    const float* kbase = k.data() + begin*KVH*D;
    const float* vbase = v.data() + begin*KVH*D;
    float* obase = out.data() + begin*QH*D;
    std::vector<float> weights(len);
    for (std::size_t qi=0; qi<len; ++qi) for (std::size_t qh=0; qh<QH; ++qh) {
        const std::size_t kh = qh/group;
        const float* qrow = qbase + qi*QH*D + qh*D;
        const std::size_t lo = qi+1 > apss26::SLIDING_WINDOW ? qi+1-apss26::SLIDING_WINDOW : 0;
        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t ki=lo; ki<=qi; ++ki) {
            const float* krow = kbase + ki*KVH*D + kh*D;
            float score=0.0f;
            for (std::size_t d=0; d<D; ++d) score += qrow[d]*krow[d];
            weights[ki] = score/scale;
            maxv = std::max(maxv, weights[ki]);
        }
        float denom=0.0f;
        for (std::size_t ki=lo; ki<=qi; ++ki) { weights[ki]=accurate_exp(weights[ki]-maxv); denom+=weights[ki]; }
        float acc[D]={};
        for (std::size_t ki=lo; ki<=qi; ++ki) {
            const float w = weights[ki]/denom;
            const float* vrow = vbase + ki*KVH*D + kh*D;
            for (std::size_t d=0; d<D; ++d) acc[d] += w*vrow[d];
        }
        float* orow = obase + qi*QH*D + qh*D;
        for (std::size_t d=0; d<D; ++d) orow[d]=acc[d];
    }
}

int main(){
    std::printf("omp_max_threads=%d\n", omp_get_max_threads());

    // A. 스레드 계단: q_proj 를 M=1..40 까지 한 행씩 늘려가며.
    for (std::size_t M=1; M<=40; ++M) mm("qfine",M,4096,2048,1);

    // B. 배치가 실제로 쓰는 M 까지 스윕. n=2 -> T=46, n=4 -> 91, n=8 -> 165.
    const std::size_t Ms[] = {1,2,4,8,14,19,32,46,64,91,128,165,256};
    for (std::size_t M:Ms) mm("q",  M,4096,2048,1);
    for (std::size_t M:Ms) mm("kv", M,4096, 512,1);
    for (std::size_t M:Ms) mm("o",  M,2048,4096,1);
    for (std::size_t M:Ms) mm("w13",M,4096, 448,1);
    for (std::size_t M:Ms) mm("w2", M, 448,4096,1);
    for (std::size_t M:{1u,8u,32u}) mm("lm_head",M,4096,32064,1);

    // C. 행 단위 연산도 M 에 따라 값이 달라지는가 (layer_norm / add_inplace).
    for (std::size_t M:{19u,165u,1024u}) {
        Tensor x({M,apss26::HIDDEN_SIZE}),y({M,apss26::HIDDEN_SIZE}),w({apss26::HIDDEN_SIZE}),b({apss26::HIDDEN_SIZE});
        fill_rand(x,7); fill_rand(w,8); fill_rand(b,9);
        tensor_ops::layer_norm(x,w,b,apss26::NORM_EPS,y);
        double t0=now(); for(int r=0;r<10;++r) tensor_ops::layer_norm(x,w,b,apss26::NORM_EPS,y); double ln=(now()-t0)/10;
        t0=now(); for(int r=0;r<10;++r) tensor_ops::add_inplace(y,x); double ad=(now()-t0)/10;
        std::printf("ROWOP M=%-5zu ln_ms %8.4f (perrow %7.5f)  add_ms %8.4f (perrow %7.5f)\n",
                    M,ln*1e3,ln*1e3/M,ad*1e3,ad*1e3/M);
        std::fflush(stdout);
    }

    // D. 어텐션: 130회 재계산(before) vs 1회(after). 둘 다 스레드 1개.
    for (std::size_t s:{6u,14u,19u,32u}) {
        Tensor q({s,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM}),k({s,apss26::NUM_KV_HEADS*apss26::HEAD_DIM});
        Tensor v({s,apss26::NUM_KV_HEADS*apss26::HEAD_DIM});
        Tensor o1({s,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM}),o2({s,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM});
        fill_rand(q,3); fill_rand(k,4); fill_rand(v,5);
        const double tb = attn_before(q,k,v,s,o1);
        attn_after(q,k,v,0,s,o2);                                  // warm
        double t0=now(); for(int r=0;r<3;++r) attn_after(q,k,v,0,s,o2); const double ta=(now()-t0)/3;
        double maxd=0; for(std::size_t i=0;i<o1.size();++i) maxd=std::max(maxd,(double)std::fabs(o1[i]-o2[i]));
        std::printf("ATTN s=%-3zu before_ms %9.4f  after_ms %8.4f  speedup %6.2fx  maxdiff %.3e\n",
                    s,tb*1e3,ta*1e3,tb/ta,maxd);
        std::fflush(stdout);
    }

    // E. 배치 어텐션은 시퀀스에 대해 omp 로 나뉜다. 직렬 vs 병렬.
    {
        const std::vector<std::size_t> len={14,32,22,23,16,17,30,11};
        std::vector<std::size_t> off(len.size()); std::size_t T=0;
        for(std::size_t i=0;i<len.size();++i){ off[i]=T; T+=len[i]; }
        Tensor q({T,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM}),k({T,apss26::NUM_KV_HEADS*apss26::HEAD_DIM});
        Tensor v({T,apss26::NUM_KV_HEADS*apss26::HEAD_DIM}),o({T,apss26::NUM_ATTENTION_HEADS*apss26::HEAD_DIM});
        fill_rand(q,3); fill_rand(k,4); fill_rand(v,5);
        double t0=now();
        for(int r=0;r<3;++r) for(std::size_t b=0;b<len.size();++b) attn_after(q,k,v,off[b],len[b],o);
        const double ser=(now()-t0)/3;
        t0=now();
        for(int r=0;r<3;++r){
#pragma omp parallel for schedule(dynamic)
            for(long long b=0;b<(long long)len.size();++b) attn_after(q,k,v,off[b],len[b],o);
        }
        const double par=(now()-t0)/3;
        std::printf("ATTNBATCH T=%zu  serial_ms %8.4f  omp_ms %8.4f  speedup %5.2fx\n",T,ser*1e3,par*1e3,ser/par);
    }
    return 0;
}
