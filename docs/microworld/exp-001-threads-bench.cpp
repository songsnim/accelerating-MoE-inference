// "행은 왜 공짜인가" 를 가르는 실험. exp-001-batching.html 의 스레드 계단이 이 출력이다.
//
// 가설 A(처음 쓴 것, 틀림): 호출 1회의 값 = 가중치 33.55 MB 를 훑는 고정비.
//                          -> 그렇다면 스레드 수를 바꿔도 평탄 구간이 그대로여야 한다.
// 가설 B(맞음):            matmul_transposed 는 출력 행(i)에 대해서만 omp 로 나눈다.
//                          행 하나 = 코어 하나. -> 평탄 구간의 끝이 코어 수를 따라 움직여야 한다.
//
// 결과: 코어 1개면 완전한 직선(행당 48.550 ms, 공짜인 행 0개),
//       코어 8개면 8행마다 계단(48.807 -> 97.475 -> 146.532 -> 194.784, 정확히 1·2·3·4배).
//       -> B. 랩노트의 "스레드 포화" 가 메커니즘으로는 맞았다.
//
//   make
//   g++ -std=c++17 -O3 -march=native -fopenmp -Iinclude \
//       -c docs/microworld/exp-001-threads-bench.cpp -o /tmp/thr.o
//   g++ -std=c++17 -O3 -fopenmp /tmp/thr.o obj/tensor.o -o ~/thrbench \
//       -L/usr/local/cuda/lib64 -lcudart -lm
//   srun --partition aps --gres=gpu:1 --exclusive ~/thrbench
//
// 측정 기록: 2026-08-25, aps 계산 노드 1개(--exclusive, omp_max_threads=64).
#include "tensor.h"
#include "config.h"
#include <ctime>
#include <cstdio>
#include <omp.h>
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
static void fill_rand(Tensor& t, unsigned s){for(std::size_t i=0;i<t.size();++i){s=s*1664525u+1013904223u;t[i]=((s>>8)&0xFFFF)/32768.0f-1.0f;}}
int main(){
    const std::size_t K=4096,N=2048;                      // q_proj
    const std::size_t Ms[]={1,2,4,8,16,24,32,40,48,64,80,96};
    const int THR[]={1,8,32,64};
    std::printf("max_threads=%d   shape K=%zu N=%zu (weight %.2f MB)\n",
                omp_get_max_threads(),K,N,K*N*4/1e6);
    for(int t:THR){
        omp_set_num_threads(t);
        for(std::size_t M:Ms){
            if(t==1 && M>32) continue;                    // 1스레드는 오래 걸리니 32행까지만
            Tensor a({M,K}),b({N,K}),c({M,N}); fill_rand(a,1); fill_rand(b,2);
            tensor_ops::matmul_transposed(a,b,c);         // warm
            double t0=now(); tensor_ops::matmul_transposed(a,b,c); double dt=now()-t0;
            std::printf("THR %-3d M %-4zu call_ms %9.3f  perrow_ms %8.3f\n",t,M,dt*1e3,dt*1e3/M);
            std::fflush(stdout);
        }
    }
    return 0;
}
