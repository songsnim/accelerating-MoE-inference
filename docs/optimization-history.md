# Phi-tiny-MoE Prefill 최적화 히스토리

**0.033306 → 640.832 seq/s (19,241x)** · n=1024 · RTX 3090 1장 · fp32 · 전 구간 검증 PASSED

## 0. 출발점 계측

baseline은 GPU 커널이 **0개**다(`Tensor`는 호스트 `std::vector`, `tensor_ops`는 OpenMP).
실효 성능 0.86 GFLOP/s. 입력은 짧다 — 길이 min/median/max 6/19/32, 총 19,803 토큰이라
sliding window(2047)가 발동하지 않고 attention은 병목이 아니다. 연산량은 1.02 GMAC/token,
전체 40 TFLOP. `matmul_transposed`는 **shape와 무관하게 1.2 GMAC/s에 고정**되는데
`Tensor::at()`이 원소마다 `initializer_list`와 bounds check를 만들어 벡터화가 불가능하기
때문이고, 이 경로로는 20.4 TMAC = 1.2시간이다. → 첫 레버는 커널 튜닝이 아니라 GPU 이전이다.

## 1. 구조 이전 (0.033 → 262.9 seq/s)

각 단계는 직전 프로파일이 지목한 **단일 최대 항목**만 건드렸다.

| # | 관찰(측정) | 기법 | before → after |
|---|---|---|---|
| 001 | 시퀀스를 1개씩 처리 → 모든 GEMM이 M≈19. `omp parallel for`가 행에만 걸려 **64코어 중 19개만** 일한다. 또 `layer.cu` attention이 q·k 내적을 **130회 재계산** | 전체 토큰을 하나의 `[T,4096]`으로 연결 + 내적 1회 계산 후 재사용 | n=2 **108.8 → 37.7 s** (2.88x)<br>분해: 배치화 1.59x × 내적 CSE **1.82x** |
| 002 | 남은 벽은 GEMM 하나 | projection/lm_head를 device 상주 + shared-tiled `gemm_nt_bias` | n=8 **62.3 → 36.9 s** (1.69x)<br>가설(2.5x)은 틀렸고, 프로파일이 MoE **98.1%**를 지목 |
| 003 | `PhiMoE::forward`가 토큰마다 `Tensor` 할당 + `at()` gather/scatter + 16 expert × 3 matmul | MoE 전량 device화(gather→w1/w3→silu*mul→w2→scatter), 라우팅만 host | n=8 **36.9 → 1.03 s** (35.9x)<br>**n=1024가 처음 2분 안에: 48.8 s / 21.0 seq/s** |
| 004 | host norm 16.1 s + resid 2.7 + attn 1.3 = 35%, 그로 인한 PCIe 왕복 10.1 s = 18% | norm/residual 커널화 + residual stream을 32레이어 내내 device 상주(double buffer) | **48.8 → 24.1 s** (2.02x)<br>norm 16.144 → **0.350**, d2h 7.50 → 0.85 |
| 005 | GEMM이 87%. thread당 출력 1개라 MAC마다 shared 2회 읽음 → **피크의 6%** | register tiling(64×64, BK=16, 4×4) + k-major 전치 shared + float4 로드 | **23.9 → 6.34 s** (3.77x)<br>projection 효율 **6% → 44%** |
| 006 | attention 관련 2.54 s / 40% (host 1.13 + q/k/v·context 왕복 13 GB) | RoPE·softmax 커널화, q/k/v/context device 상주 | **6.34 → 3.93 s** (1.61x)<br>d2h 0.865 → **0.006** |

이 구간 내내 **출력이 bitwise 동일**하게 유지됐다. 004에서 그 이유가 드러났다: norm을 tree
reduction으로 쓰자 n=1024에서 max abs diff **0.212**로 실패했고, 1024개 중 **4개 시퀀스**만
터졌다. 합의 순서가 1e-7 흔들려 라우터 점수의 양자화 경계에서 **expert 집합이 뒤집힌** 것이다.
이후 `objdump`로 host의 FMA 미발생을 확인하고 `__fadd_rn/__fmul_rn`으로 계약을 막아
**bitwise 동일을 판정 기준**으로 삼았다.

## 2. 일의 제거와 커널 튜닝 (262.9 → 640.8 seq/s)

| # | 관찰 | 기법 | 효과 |
|---|---|---|---|
| 010 | expert w1/w3가 N=448·BN=128 → **12.5% 패딩 폐기**, launch 2개가 wave의 49%만 채움 | w1‖w3를 N=896 한 장으로 융합 | moe 1.414 → **1.155**. 예측의 3배 — 충전율 49→85%가 겹쳤다 |
| 011 | causal + 절대위치 RoPE이므로 **앞선 토큰이 같은 두 토큰은 32레이어 전부에서 같은 hidden state** | prefix trie: 행을 토큰이 아니라 trie 노드로 | 19,803 → **15,583행(-21.3%)**<br>**3.281 → 2.653 s** (1.237x) |
| 012 | w13 wave 충전율 66%, 시간의 53%가 1 wave 미만 launch | (expert, rowtile) work queue로 16 launch → 1 | moe 0.968 → **0.768** |
| 013·014·015 | embed 0.181 s = 255 MB를 CPU 1스레드로 gather / `t_lm` 0.070 중 **0.069가 131 MB 제로필** / host 라우팅 왕복 0.060 | 임베딩 device 상주 · 출력 할당을 별도 스레드로 겹침 · counting-sort 커널로 라우팅 device화 | embed → **0.002**, 각 **-0.068 / -0.071 s**<br>scatter 16 launch → `moe_combine` 1개 |
| 016 | ncu: `attention_heads`만 ld bytes/sector **22.6** + L1 87.6% 포화 | key를 32열 청크로 shared 스테이징(행 통째는 occupancy 12→5로 반증) | attn 0.144 → **0.100** |
| **017** | ncu: LDS 명령당 wavefront **5.00** / conflict **2.00**. 분해로 범인이 `bs` 읽기 하나임을 특정 | `gemm_col_slot`: 레인의 8열을 4열씩 반 타일 떨어뜨려 quarter 8레인이 32뱅크를 1회씩 덮음 | **2.120 → 1.937 s (1.093x)** — 단일 최대 레버<br>conflict → **0**, wavefront 5.00 → 3.00 |
| 019 | ncu: FMA 59.2% / l1tex 62.6%로 둘 다 미포화 → 지연이 한계 | `BK` 8 → 16 (2블록/SM 유지 최대폭) | **-75 ms**. `long_scoreboard` 0.92 → **0.02** |
| 021 | nsys: 에필로그 `cudaFree` 24회가 **측정 구간 안에서 20.8 ms**, GPU idle | 스크래치를 모델 멤버로(grow-only `reserve`) | **-19.7 ms** (상한의 95%) |
| 023 | ncu: `layer_norm_rows` barrier stall **16.67** — bitwise 강제 직렬 누적 탓에 256스레드 중 1개만 일함 | 1행 = 1레인(32행/워프) + 열 청크 shared 전치 + 스테이징 로드 명시적 배칭 | norm 0.116 → **0.082**, -27 ms. DRAM **89%** |
| 024·025 | `add_inplace`·`moe_combine`이 92~95% DRAM = 루프라인. 싸게 만들 길이 없다 | 둘 다 **생산자 에필로그**로 이전(o_proj GEMM / w2 GEMM). 025는 두 조각을 다른 행 서브레인지로 패킹해 launch 2개로 쪼개 순서를 확보 | resid 0.058 → **0**, moe 0.616 → 0.590<br>**-45.1 / -19.8 ms**, 버퍼 511 MB 소멸 |
| 026 | ncu: shared **스토어** STS의 절반이 conflict (로드는 017이 이미 바닥) | 원소 (k,m)을 열 `m+(k>>2)*8`로 시프트(패딩으로는 float4 정렬과 양립 불가) | store conflict → **0**, **-30.8 ms**<br>grouped는 레지스터 3개가 없어 spill → projection만 |
| 028 | 라우터 gate는 out=16인데 BN=128 → **128열 중 112열을 계산해 버림** | `BM=64, BN=16, TM=4` tall-skinny 전용 커널 | gate 1.11 → **0.601 ms**, -15.6 ms |
| 033 | 측정 구간 버스 트래픽 전량이 logits D2H 131 MB. pageable 15.0 vs pinned **5.0 ms** | `tensor.h`에 `Alloc::Pinned` + **생성자에서 256 MB 예약** 후 `std::move` | lm_head 0.029 → **0.020**, -9.1 ms |
| **034·036** | ncu `--set full`: occupancy 98.1%, **Mem Pipes 77.5%** = 명령 수가 벽. `QH/KVH=4`인데 4블록이 같은 k·v를 4번 읽음 | GQA group 4개를 한 블록으로 + shared 패딩을 4의 배수로 맞춰 float4 로드 | attention 3.34 → **0.964 ms**<br>global ld -74%(=1/4), **-70.5 ms** |
| 037 | `layer_norm_rows`가 읽는 양이 **3유닛**(766 MB). 상주는 산술로 불가(레지스터 4096개 필요) | 3번째 패스를 소비자로 이전. `input_norm`은 소비자가 A를 24번 스테이징해 **산술로 기각** | post_norm 1263 → **599 µs**, -17.5 ms |
| 038 | 전수조사가 빠뜨린 `gather_rows`가 레이어당 1.22 ms, 순수 복사 1.02 GB | A 행 주소를 `arowmap[r]`로 간접화해 w13 GEMM의 A 스테이징에 흡수 | 커널 **삭제**, **-23.1 ms** |
| **040** | 입력이 짧으므로 **마지막 레이어는 15,583행 중 lm_head가 읽는 1,024행(6.6%)만 필요** | 꼬리 레이어만 압축 행 경로(k/v/input_norm은 조상 키 때문에 전 행 유지). GEMM은 `rows` 인자만 변경 | 꼬리 58.2 → **13.4 ms**, **-39.7 ms**<br>628.3 → **644.0 seq/s** |
| 042 | `silu_mul`이 DRAM 90% → 삭제만이 길. 막던 것은 gate/up이 3.5타일 떨어진 배치 | 가중치 행을 로드 시점에 인터리브해 쌍이 한 스레드 레지스터에 오게 | 커널 삭제, -5.1 ms |

## 3. 기각 — 메커니즘은 작동했으나 값이 안 나온 것

| # | 기법 | 실측 | 원인 |
|---|---|---|---|
| 007 | 8×8 타일(128×128) 일괄 적용 | 0.99x | wide-N 1.17x인데 expert(N=448)가 **0.81x** — 블록 273→80개로 붕괴(82 SM) |
| 018 | bias 호이스팅 (LDG 68→12) | ±0 | 에필로그는 k타일 512회 뒤 출력당 1회. bias는 L1 상주 |
| 022b | 누산기 64→128 (8×16) | gemm **+75 ms** | shared 압력은 예측대로 -10.7pp 빠졌으나 1블록/SM이 되어 `long_sb` 0.02 → **0.47** 부활 |
| 027 | 죽은 행 그룹 skip | +15 ms | 1088명령 블록을 `if`로 감싼 `BSSY/BSYNC`가 GEMM 전체를 2% 늦춘다 |
| 029 | 글로벌 스트림 4→2 | +54 ms | 명령 -14인데 **섹터 2.006배** — 4레인/행이 32B 섹터를 채우던 배치가 깨진다 |
| 030 | 구간 안 `cudaHostAlloc` | **+105 ms** | D2H는 -9 ms 벌지만 125 ms 페이지락이 **다른 스레드에서도** CUDA 컨텍스트 락을 쥔다 |
| 039 | w2 store 스위즐 | +6.7 ms | conflict -94% 달성. 그런데 125→128 레지스터 + **k 루프 안 8 B spill** |
| 041 | grouped `BM=64/BN=256` | -1.0% | 패딩 -2.9% 회수. 그런데 `L2 ∝ (1/BM+1/BN)`가 BM=BN=128에서 최소 → L2 +22% |
| 043 | 상주 블록 + 타일 간 프롤로그 겹침 | -3.0% | prefetch 자체는 +1.9%로 실재하나 상주가 더 비쌈. **부수: 견적 10.2%가 마이크로벤치 열 드리프트였고 실제 4.3%** |
| 044 | FP16 텐서코어 정밀도 프로브 | max abs **4.34** (임계 3e-3) | GA102는 TF32가 FP32와 동률(38 TF/s)이라 FP16만 이득인데 32레이어 누적이 1448배 초과 |

## 4. 방법론에서 얻은 것

- **노드 클럭이 1.16~1.19x 흔들린다**(빠른 1620 MHz @356 W = 전력 상한, 느린 1395 MHz).
  007에서 "1.04x 이득"이 실은 다른 노드였다 → 이후 모든 A/B를 **한 allocation 안에서 교차 실행**.
- **연산 밀도를 올리면 빠른 노드에서 이론치의 97~98%만 나온다**(측정-B). grouped MoE가 같은
  FLOP을 26% 짧게 밀어넣자 boost가 한 단 내려가 뒤이은 projection이 +2.4%를 물었다.
- **상한을 먼저 재고 순위를 매겼다.** 트리플 버퍼링(0.66%), pinned 단독(0.16%), 루프
  언롤링(95.2%가 이미 전개), coalescing(1.018%)은 **코드를 쓰기 전에** 기각됐다.

## 5. 최종 상태

GEMM이 커널 시간의 **93.9%**, 유효 21.1 TFLOP/s = 실클럭 피크의 **62%**. 측정-K의 ablation이
비-FFMA 33%를 항별로 갈랐고(스토어 12.3 / 프리페치 8.8 / shared 로드 6.7 / barrier 3.6%),
셋은 하한이다 — 전치 스토어는 레지스터 예산을 사는 값(k-inner는 spill 72 B로 2.75배 느림),
`bs`의 4 wavefront는 512 B 고유 데이터의 하드웨어 하한, `as`의 2는 브로드캐스트 하한.
남은 barrier 3.6%는 3블록/SM을 요구하는데 smem 40,960 B와 레지스터 125가 **각각 독립적으로**
2에서 막는다.

남은 후보: w2 combine 기구 잔차 ≈18 ms(rowmap 간접 · `c[o]` RMW · resid 스칼라 LDG 128개),
gate를 DRAM 루프라인까지 -9 ms, 비-GEMM 111 ms를 2 스트림으로 GEMM 뒤에 숨기기(미탐색,
시퀀스 2분할 비용은 +2.3%행으로 실측).
