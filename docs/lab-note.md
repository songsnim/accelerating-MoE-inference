# 조교도 정답을 모르지만 추천한 먼저 해볼만한 것들 목록
- 여러 개의 입력을 묶음으로 처리(batching)
- 각 input의 공통된 토큰(e.g. <|user|>, <|end|>, <|assistanct|>)에 대한 중복 연상 제거
- kernel 최적화
    - matmul 최적화
    - reduction 연산 최적화
    - Kernel Fusion
- 프로파일링: Nsight Systems/Compute 사용
- CPU-GPU 통신 시간을 감안해도, 특정 연산을 CPU에서 처리하는 것이 이득일 수도 있음

---

# EXP-000 (baseline 계측 · 코드 무수정)
## 1. Background
스켈레톤 그대로 제출해 최소 성능 기준선을 확보한다.

## 2. Hypothesis
(없음 — 계측만)

## 3. Design and Implementation
`make` 후 `./submit.sh -n 1`. 소스 무수정.

## 4. Result
| n | Elapsed | Throughput | Validation |
|---|---|---|---|
| 1 | 30.024 s | 0.033306 seq/s | PASSED (max 6.87e-05) |
| 2 | 108.838 s | 0.018376 seq/s | PASSED (max 6.87e-05) |

- 대시보드 등록: 0.0333 seq/s (기본반, n=1)
- 참조 출력 저장: `baseline_n2.bin` (이후 bitwise 비교용)

계측으로 확인한 작업 특성:
- **입력이 짧다**: `max_seq_len=32`, 길이 min/median/max = 6/19/32, 1024개 합 **19,803 토큰**.
  → sliding window(2047)는 절대 발동하지 않음. attention은 병목이 아니다.
- **연산량**: 토큰당 attention proj 21M MAC + MoE(top-2) 11M MAC = 32M MAC/layer,
  ×32 layer = **1.02 G MAC/token**. 전체 1024 seq = **40 TFLOP** + lm_head 0.27 TFLOP.
- **실효 성능 0.86 GFLOP/s** (n=2: 46토큰 / 108.8s). 32-core Xeon E5-2650 피크의 0.4%.
- **baseline은 GPU를 전혀 쓰지 않는다**: `Tensor`는 `std::vector<float>`(호스트), `tensor_ops`는 OpenMP CPU.
  `run.sh`가 GPU를 잡고 `main.cpp`가 device sync를 하지만 커널은 하나도 없다.
- 모델 크기 15.0 GB(fp32, ~3.75B param) — RTX 3090 24GB에 fp32로 **올라간다**.
- 느린 이유 3가지 (코드 리딩):
  1. `generate()`가 시퀀스를 1개씩 처리 → 모든 matmul이 M=19 행. 레이어 가중치 470MB를
     시퀀스마다 다시 스트리밍한다.
  2. `layer.cu`의 attention이 q·k 내적을 **130회 재계산**(max 패스 + denom 패스 + head_dim 128회).
     s=32에서 전체 연산의 ~14%.
  3. `tensor_ops::matmul_transposed`가 `Tensor::at()`을 원소마다 호출 —
     `initializer_list` 생성 + 차원별 bounds check → 벡터화 불가.

## 5. Next action
EXP-001(batching)로 진행. 단, 편집 가능 파일이 `model.h/model.cu/model_loader.*/run.sh`뿐이므로
`layer.cu`의 attention/`tensor.cu`의 matmul은 **고칠 수 없다** → `model.cu`에서 `layer.h`의
공개 클래스(`Linear`, `PhiMoE`)를 직접 조립해 배치 경로를 새로 쓴다.

---

# EXP-001 (batching: 전체 시퀀스를 1개의 [T, H] 활성 행렬로)
## 1. Background
EXP-000에서 `generate()`가 시퀀스를 1개씩 `forward()`에 넣는 것이 확인됐다. 시퀀스 평균 길이가
19토큰이라 모든 matmul이 M≈19의 극단적으로 얇은 GEMM이고, 레이어당 470MB 가중치를
1024번 다시 읽는다.

## 2. Hypothesis
1024개 시퀀스의 토큰 19,803개를 **하나의 [T, 4096] 행렬로 연결**해 32개 레이어를 한 번만
통과시키면, (a) 가중치를 레이어당 1회만 읽고 (b) matmul의 M이 19 → 19,803이 되어
OpenMP 병렬화·캐시 재사용이 살아난다. **연산량은 그대로인데 처리량이 오른다.**

정당성: layer_norm / matmul_transposed / silu / mul / MoE 라우팅은 모두 **행 단위 독립**이라
배치해도 각 행의 부동소수점 연산 순서가 바뀌지 않는다. 시퀀스 간 상호작용은 attention뿐이므로
attention만 시퀀스 구간 [offset, offset+len)에 대해 따로 계산하면 된다.

## 3. Design and Implementation
- `model.h`: `std::vector<PhiDecoderLayer> layers_`를 `Layer{norm 4개 + Linear q/k/v/o + PhiMoE}`로
  교체. `PhiDecoderLayer`/`PhiAttention`은 내부 멤버가 private이라 배치 경로에서 재사용 불가.
  가중치 사본은 여전히 1개(15GB)다.
- `model.cu`: `generate()`가 배치 경로. 레이어당
  `layer_norm → q/k/v proj → (시퀀스별 RoPE+attention) → o proj → residual → layer_norm → PhiMoE → residual`.
  마지막에 final norm → 각 시퀀스 마지막 토큰만 [B, 4096]으로 모아 `lm_head` 1회.
- attention은 `layer.cu`의 산술을 **그대로** 옮기되 q·k 내적을 1회만 계산해 재사용
  (130회 재계산 제거). `exp`는 double로 계산 후 float 캐스팅, 누적 순서(d 오름차순, ki 오름차순)
  모두 동일하게 유지 → **bitwise 동일 출력**이 목표.
- `forward()`는 `generate({ids})`로 위임 → `-d`(decode) 경로도 그대로 동작.
- 가설 1개 원칙: 자체 GEMM/CUDA 커널은 이 실험에 넣지 않는다 (EXP-002 이후).

## 4. Result
`src/model.cu` 전면 교체 + `include/model.h` 멤버 교체. 빌드 경고 없음.

| n | tokens | baseline | EXP-001 | speedup | seq/s | Validation |
|---|---|---|---|---|---|---|
| 1 | 14 | 30.024 s | (미측정) | - | - | - |
| 2 | 46 | 108.838 s | **37.749 s** | **2.88x** | 0.052982 | PASSED |
| 4 | 91 | (>2min) | **46.535 s** | - | 0.085956 | PASSED |
| 8 | 165 | (>2min) | **62.270 s** | - | **0.128473** | PASSED |

- **출력 bitwise 동일**: `cmp baseline_n2.bin exp001_n2.bin` -> IDENTICAL.
  max abs diff 6.86646e-05 / mean 1.20815e-05 로 baseline과 소수점까지 같음.
  -> 배치화가 부동소수점 연산 순서를 전혀 바꾸지 않았음이 증명됨.
- 토큰당 비용: baseline 2.37 s/token -> n=2 0.82 -> n=4 0.51 -> n=8 0.377. n이 커질수록 계속 개선.
  이유: `matmul_transposed`의 `#pragma omp parallel for`가 **행(M)에 대해서만** 병렬화한다.
  baseline은 M=19라 **64코어 중 19개만** 일했다(노드 `omp_max_threads=64`). 배치화로 M=T가 되어 코어가 찬다.
  -> throughput이 n에 의존하므로 리더보드 제출은 2분 제한에 들어가는 최대 n으로 해야 한다.
  n=8(62s)이 여유 있는 최대치. n=12는 250토큰 x 0.37 = 94s + load 10s로 2분 제한에 너무 붙는다.
  이득도 평탄해지는 중(0.086 -> 0.128 -> 추정 0.135)이라 n=8로 제출하고 다음 레버로 넘어간다.
- **제출: 0.128473 seq/s (n=8), baseline 0.033306 대비 3.86x**

#### 정정 (2026-08-25): 두 변경의 분리 측정
이 커밋은 (a) 배치화와 (b) 어텐션 재작성을 **한 번에** 바꿨다(1가설 원칙 위반).
어느 쪽이 레버인지 가르려고 "배치화는 하되 `attend_sequence` 안쪽만 `layer.cu` 원문
(130회 재계산 + `at()`)으로 되돌린" 측정용 빌드를 만들어 같은 입력(n=2)으로 돌렸다.

| 빌드 | 배치화 | 어텐션 | n=2 elapsed |
|---|---|---|---|
| `57ffaeb` baseline | - | 130회 | 108.838 s |
| 가르는 빌드 (측정용) | O | 130회 | **68.640 s** |
| `8f09c7f` EXP-001 | O | 1회 | **37.749 s** |

-> 배치화 **1.59x**, 어텐션 재작성 **1.82x** (1.59 x 1.82 = 2.88 ✓). 셋 다 Validation PASSED.
**어텐션 재작성이 둘 중 큰 레버였다** — 원래 여기 적었던 "13%(1.15x)뿐"은 틀렸다.

메커니즘("스레드 포화")은 맞다. 같은 `q_proj`를 코어 1/8/32/64개로 M 스윕해 확인:
코어 1개면 **완전한 직선**(행당 48.550 ms, 공짜인 행 0개), 코어 8개면 8행마다 계단
(48.807 -> 97.475 -> 146.532 -> 194.784 = 정확히 1:2:3:4배). 평탄 구간의 끝이 코어 수를 따라 움직인다.
행 하나 = 코어 하나이고, 코어를 다 채운 뒤로는 바퀴 수가 정직하게 는다.

### 부수 측정: `tensor_ops::matmul_transposed`의 천장
scratchpad에서 `obj/tensor.o`만 링크해 실제 shape로 측정 (login node, 32 threads):

| 연산 | shape | GMAC/s |
|---|---|---|
| q_proj T=46 | M=46 K=4096 N=2048 | 1.03 |
| q_proj T=736 | M=736 K=4096 N=2048 | 1.23 |
| o_proj T=736 | M=736 K=2048 N=4096 | 1.21 |
| expert w1 r=92 | M=92 K=4096 N=448 | 1.01 |
| expert w2 r=92 | M=92 K=448 N=4096 | 1.09 |
| lm_head B=32 | M=32 K=4096 N=32064 | 1.23 |

**shape와 무관하게 1.2 GMAC/s(2.4 GFLOP/s)에 고정**. E5-2650 x2 피크(~256 GFLOP/s)의 1%.
원인은 `Tensor::at()`: 원소마다 `initializer_list`를 만들고 차원별 bounds check를 하므로
벡터화가 불가능하다. 전체 작업량 20.4 TMAC / 1.2 GMAC/s = **4.7시간** -> 이 경로로는 n=1024 불가.

> 정정 (2026-08-25): 이 표는 login node(32 threads, 공유)라 실제 실행 노드보다 느리다.
> 계산 노드(`--exclusive`, 64 threads)에서 다시 재면 q_proj M=256이 **4.83 GMAC/s**이고,
> 20.4 TMAC / 4.83 = **1.2시간**이다. 여전히 2분 제한의 35배라 결론은 그대로.

## 5. Next action
matmul이 유일한 벽이므로 `tensor_ops::matmul_transposed`에서 벗어나는 것이 다음 레버다.
`tensor.cu`는 수정 금지이므로 `model.cu`에 자체 커널을 쓴다. 작업량 분해(토큰당, 32 layer 합산):

| 부분 | GMAC/token | 비중 | 현재 소유자 |
|---|---|---|---|
| q/k/v/o proj | 0.671 | 66% | `Linear` (model.cu에서 호출 -> 교체 가능) |
| MoE experts (top-2) | 0.344 | 34% | `PhiMoE::forward` 내부 (교체하려면 재구현 필요) |
| lm_head | 0.131 /seq | - | `Linear` (교체 가능) |

-> EXP-002는 교체 가능한 66%만 CUDA로 옮겨 "GEMM이 벽"이라는 가설을 최소 코드로 검증한다.
Amdahl 상한 1/0.34 = 2.9x. 성공하면 EXP-003에서 MoE(34%)를 재구현해 GPU로 옮긴다.

microworld: `docs/microworld/exp-001-batching.html` (+ `-bench.cpp`, `-threads-bench.cpp`).

---

# EXP-002 (q/k/v/o + lm_head GEMM을 CUDA 커널로)
## 1. Background
EXP-001에서 `matmul_transposed`가 shape 무관 1.2 GMAC/s로 고정된 벽임을 측정했다.
RTX 3090은 fp32 35.6 TFLOP/s이고, 모델 15GB는 24GB VRAM에 fp32로 그대로 올라간다.

## 2. Hypothesis
projection 가중치(레이어당 21M param, 32 layer 합 2.7GB)와 lm_head(525MB)를 생성자에서
device에 올려두고, `Linear::forward` 호출을 자체 CUDA GEMM으로 바꾸면 전체 연산의 66%가
사실상 공짜가 되어 **2.5x 이상** 빨라진다.

## 3. Design and Implementation
- `model.h`: `Layer`의 `Linear q/k/v/o`와 `lm_head_`를 `DeviceLinear`로 교체.
  `DeviceLinear`는 생성자에서 `loader.load()` -> `cudaMemcpy` -> **host 사본 즉시 폐기**.
  (host RAM도 2.7GB 절약). 소멸자에서 `cudaFree`. copy 금지 + move ctor로 double-free 차단.
- `model.cu`: shared-memory tiled GEMM 커널 `gemm_nt_bias` 1개.
  `C[M,N] = A[M,K] * B[N,K]^T + bias[N]`, TILE=32, `bs`는 [TILE][TILE+1]로 패딩해 bank conflict 회피.
  k 루프를 오름차순으로 유지해 host `matmul_transposed`와 누적 순서를 맞췄다. bias는 커널에 융합.
- layer_norm / attention / MoE는 host에 남으므로 레이어당 q/k/v와 context가 버스를 1회씩 왕복.
- `APS_PROFILE=1`일 때만 켜지는 stage 타이머 추가 (측정 런에는 sync가 추가되지 않음).

## 4. Result
| n | EXP-001 | EXP-002 | speedup | seq/s | Validation |
|---|---|---|---|---|---|
| 8 | 62.270 s | **36.869 s** | **1.69x** | 0.216985 | PASSED (max 8.77e-05) |

가설(2.5x 이상)은 **틀렸다**. `APS_PROFILE=1`로 원인을 계측:

```
[profile] T=165  embed 0.000  norm 0.140  h2d 0.016  gemm 0.153  d2h 0.040
                 attn 0.094  moe 28.007  resid 0.095  lm_head 0.005
```

- **MoE가 28.007s / 28.55s = 98.1%**. GPU로 옮긴 projection 전체(32 layer x 4 GEMM + lm_head)는
  **0.153s**에 끝난다. 즉 GEMM 커널은 이미 충분히 빠르고, 병목은 전부 MoE로 이동했다.
- Amdahl 추정(MoE 34%)이 틀린 이유: MoE의 비용은 GEMM 연산량(34%)이 아니다.
  `PhiMoE::forward`가 (a) 토큰마다 `Tensor({16})`을 새로 할당하고 (b) expert별 gather/scatter를
  `at()`으로 원소 단위 복사하며 (c) 16 expert x 3 matmul을 전부 느린 `matmul_transposed`로 돈다.
- H2D/D2H는 합쳐서 0.056s로 무시 가능(T=165). host attention도 0.094s로 무시 가능.
- 정확도: max abs diff 6.87e-05 -> 8.77e-05. GPU가 FMA로 계약(contract)한 결과로,
  임계값 3e-3의 3% 수준. 안전하다.

## 5. Next action
MoE만 GPU로 옮기면 된다. 남은 host 작업(norm/attn/resid = 0.33s)은 그 다음 문제다.
-> EXP-003.

---

# EXP-003 (MoE를 GPU로: gather -> expert GEMM -> silu*mul -> scatter)
## 1. Background
EXP-002 프로파일에서 MoE가 98.1%(28.0s/28.55s)로 단일 병목임을 계측했다.
GEMM 커널은 이미 0.153s로 검증됐으므로 같은 커널을 expert에도 쓰면 된다.

## 2. Hypothesis
MoE expert 가중치(레이어당 88M param, 32 layer 합 **11.3GB**)를 device에 올리고
router -> gather -> w1/w3 GEMM -> silu*mul -> w2 GEMM -> weighted scatter를 전부 커널로 돌리면
28.0s가 사실상 사라져 **n=8에서 10x 이상** 빨라진다.
VRAM: projection 2.7GB + MoE 11.3GB + lm_head 0.5GB = 14.5GB < 24GB.

## 3. Design and Implementation
- `model.h`: `Layer::moe`를 `PhiMoE` -> `DeviceMoE`로 교체. `DeviceMoE`는 gate + 16개
  expert의 w1/w2/w3를 전부 `DeviceLinear`로 들고 있다(= EXP-002 커널 재사용). 생성자에서
  device로 올린 뒤 host 사본 폐기.
- `DeviceBuffer`를 `template <typename T>`로 일반화 (expert index 버퍼가 `int`라서 필요).
- 새 커널 3개. 전부 EXP-002의 `gemm_nt_bias`를 그대로 쓰고 그 앞뒤만 붙인다.
  - `gather_rows`: `dst[i,:] = src[index[i],:]` — expert가 담당하는 토큰 행만 모은다.
  - `silu_mul`: `gate = silu(gate) * up`. `expf`(float) 사용 — `tensor.cu`는 double `exp`지만
    상대오차 ~1e-7로 임계값 3e-3 대비 4자리 아래.
  - `scatter_add_rows`: `dst[index[i],:] += 0.5f * src[i,:]`. expert별 개별 launch이고
    한 expert는 같은 행을 최대 1번 담으므로 atomic 불필요.
- 라우팅은 **host에 남긴다**. 토큰당 16개 값이라 연산량이 없고, `layer.cu:47-69`의
  quantise + tie-eps 선택 로직을 그대로 옮겨야 하기 때문(`route_row`). top-2 가중치 0.5/0.5도 동일.
- 레이어당 흐름: `post` H2D -> gate GEMM -> router D2H -> host에서 expert별 토큰 리스트 작성
  -> expert마다 gather -> w1/w3 GEMM -> silu_mul -> w2 GEMM -> scatter_add -> `ff` D2H.
- VRAM: projection 2.7GB + MoE 11.3GB + lm_head 0.5GB + 활성 스크래치 ~2.2GB = **16.7GB / 24GB**.
- norm / attention / residual은 host에 그대로 둔다 (가설 1개 원칙).

## 4. Result
| n | tokens | EXP-002 | EXP-003 | speedup | seq/s | Validation |
|---|---|---|---|---|---|---|
| 8 | 165 | 36.869 s | **1.026 s** | **35.9x** | 7.799838 | PASSED (max 9.16e-05) |
| 1024 | 19,803 | (불가) | **48.765 s** | - | **20.998835** | PASSED (max 1.79e-04) |

가설(n=8에서 10x 이상)은 **맞았다** — 실제 35.9x. MoE 28.0s가 사라졌다.
- **n=1024 전체가 2분 제한 안에 들어왔다** (48.8s + load 5.3s). EXP-001/002는 n=8이 한계였는데,
  이제 throughput이 n에 따라 계속 오르므로(7.8 -> 21.0) 전량 처리가 곧 최고 기록이다.
- **제출: 19.252353 seq/s (n=1024, 리더보드 1위/4). baseline 0.033306 대비 630x, EXP-002 대비 96.8x.**
  (리더보드 값이 로컬 20.999보다 낮은 건 제출 런이 별도 측정이고 런 간 변동이 ±5~8%이기 때문.)
- 정확도: max abs diff 1.79e-04 (임계값 3e-3의 6%). T가 커져 오차 누적이 늘었지만 여유 있다.

`APS_PROFILE=1` 프로파일 (n=1024, sync 추가로 총 57.7s):
```
[profile] T=19803  embed 0.000  norm 16.144  h2d 2.618  gemm 13.164  d2h 7.495
                   attn 1.305   moe 13.449  resid 2.689  lm_head 0.460
```
| 단계 | 초 | 비중 | 위치 |
|---|---|---|---|
| norm | 16.144 | 28% | **host** |
| moe | 13.449 | 23% | device |
| gemm | 13.164 | 23% | device |
| d2h | 7.495 | 13% | 버스 |
| resid | 2.689 | 5% | **host** |
| h2d | 2.618 | 5% | 버스 |
| attn | 1.305 | 2% | **host** |

- 병목이 다시 이동했다: 이제 **host 잔여 연산 20.1s(35%)** + 그 때문에 발생하는
  **PCIe 왕복 10.1s(18%)** = **30.2s / 53%**가 GPU 연산이 아니다.
- `d2h` 7.5s의 대부분은 레이어당 `ff`(324MB) + `context`(162MB) 회수다. 32 레이어 합 15.6GB.
- host `layer_norm`이 단일 최대 항목이 된 이유: `tensor_ops::layer_norm`도 `at()` 경유라
  EXP-001에서 측정한 것과 같은 원소 단위 오버헤드를 문다. 연산량은 GEMM의 1/500인데 시간은 더 크다.

## 5. Next action
활성값을 **레이어 내내 device에 상주**시키는 것이 다음 레버다. norm / residual add를 커널로
옮기면 (a) host 20.1s 중 18.8s가 사라지고 (b) 레이어당 H2D/D2H가 attention과 라우팅용
2회로 줄어 10.1s의 대부분이 사라진다. 상한 30.2s 제거 -> 48.8s에서 **~2.5x**.
-> EXP-004 (layer_norm + residual 커널화, 활성값 device 상주).
그 다음 후보: attention도 커널로 옮겨 D2H를 라우팅용 1회로 축소, GEMM 커널 자체 튜닝
(13.2s는 아직 3090 피크의 수 %다), expert 배치를 grouped GEMM 1회로 묶어 launch 오버헤드 제거.

---

# EXP-004 (layer_norm + residual 커널화, 활성값 device 상주)
## 1. Background
EXP-003 프로파일에서 GPU 연산이 아닌 부분이 53%였다: host norm/resid/attn 20.1s +
그 때문에 발생하는 PCIe 왕복 10.1s.

## 2. Hypothesis
layer_norm과 residual add를 커널로 옮기고 residual stream을 32 레이어 내내 device에
상주시키면, host 잔여 연산 18.8s와 H2D/D2H 대부분이 사라져 **~2.5x**.

## 3. Design and Implementation
- `model.h`: `Layer`의 norm weight/bias Tensor 4개 -> `DeviceNorm input_norm, post_norm`,
  `final_norm_weight_/bias_` -> `DeviceNorm final_norm_`. (`DeviceLinear`와 같은 패턴)
- 커널 2개 추가: `add_inplace_kernel`(elementwise), `layer_norm_rows`(행당 1 block).
- **residual stream double buffer**: `d_stream_a/b`를 raw pointer로 잡고 레이어 끝에서
  `std::swap`. MoE가 다음 레이어 입력을 누적하는 동안 현재 stream은 residual add에 아직 필요.
- 버퍼 재사용: q/k/v를 o_proj가 소비한 뒤 `d_norm`이 post-attention norm(= expert gather
  source)을 받는다. default stream이므로 커널 순서가 보장된다.
- lm_head 입력의 마지막 토큰 추출도 `gather_rows` 재사용 (host 복사 제거).
- 레이어당 버스 통과는 q/k/v D2H + context H2D + router D2H(1.3MB)만 남는다.

### 실패한 첫 시도 — 이게 이 실험의 알짜다
`layer_norm_rows`를 **block tree reduction**으로 쓰면 n=8은 통과하는데 **n=1024가 실패**했다.
```
max abs diff 0.211975  mean abs diff 6.83e-05  first mismatch [344, 6744]
```
EXP-003 출력과 행별 비교: **1024개 중 4개 시퀀스만** 임계값 초과 (344/400/401/1009).
mean은 그대로인데 max만 튄다 -> 정밀도가 아니라 **이산적 결정이 뒤집힌 것**.

원인: tree reduction은 합의 순서를 바꾸므로 router logit이 ~1e-7 흔들린다. `PhiMoE::route`는
score를 `ROUTER_SCORE_QUANTUM`으로 양자화한 뒤 tie-eps로 top-2를 고르므로, 경계에 있는 토큰이
**다른 expert**를 받는다. 그 토큰 출력은 O(0.1) 바뀐다. (634k개 라우팅 결정 중 몇 개면 충분.
top-1/top-2 순서만 바뀌는 flip은 가중치가 둘 다 0.5라 무해하고, **집합**이 바뀔 때만 터진다.)

해결: norm을 host와 bit-identical하게. `objdump obj/tensor.o`로 host 코드를 먼저 확인했다 —
`vaddss` 단일 accumulator 순차 누적, **벡터화도 FMA도 없다**. 그래서
- 행을 shared memory에 올리고 **thread 0이 순차로 두 번 훑는다**
- `__fadd_rn/__fsub_rn/__fmul_rn`으로 nvcc의 FMA contraction을 막는다
  (host가 내지 않는 FMA를 device가 내면 결과가 달라진다)
- `1.0f/sqrtf(...)` — `rsqrtf`는 저정밀 근사라 금지

직렬화 비용은 23.82s -> 24.11s (**1.2%**). host norm 16.1s를 없애는 대가로는 공짜다.

## 4. Result
| n | tokens | EXP-003 | EXP-004 | speedup | seq/s | Validation |
|---|---|---|---|---|---|---|
| 8 | 165 | 1.026 s | **0.727 s** | 1.41x | 11.004965 | PASSED |
| 1024 | 19,803 | 48.765 s | **24.115 s** | **2.02x** | **42.463893** | PASSED (max 1.79e-04) |

가설(~2.5x)은 대체로 맞았다 — 실제 2.02x.
- **출력이 EXP-003과 bitwise 동일**: `cmp exp003_n1024.bin exp004_n1024.bin` -> IDENTICAL.
  max abs diff도 1.79e-04로 소수점까지 같다. bit-exact 목표가 달성됐음이 증명됨.
- **제출: 42.677174 seq/s (n=1024, 리더보드 1위/4). 개인 최고 19.25 -> 42.68 (+121.7%).**
- baseline 0.033306 대비 **1275x**.
- 런 간 변동 ±5% 관측(23.82~24.12s). 실험 간 비교는 같은 세션 연속 측정으로 해야 한다.

프로파일 비교 (n=1024, 둘 다 `APS_PROFILE=1`):
| 단계 | EXP-003 | EXP-004 | 비고 |
|---|---|---|---|
| norm | 16.144 | **0.350** | 46x. 병목에서 노이즈로 |
| gemm | 13.164 | 13.156 | 손대지 않음 — 이제 **56%** |
| moe | 13.449 | **7.276** | EXP-003의 t_moe는 `ff` D2H(10GB)를 포함했다 |
| d2h | 7.495 | **0.850** | |
| resid | 2.689 | **0.075** | |
| h2d | 2.618 | **0.561** | |
| attn | 1.305 | 1.125 | host 잔류 |
| lm_head | 0.460 | 0.148 | |

## 5. Next action
이제 **GEMM 커널 자체가 병목**이다. gemm 13.156s + moe 7.276s = **20.4s / 87%**가 device GEMM.
실효 성능을 계산하면:
- projection+gate+lm_head 13.4 TMAC / 13.156s = **2.04 TFLOP/s**
- MoE expert 6.8 TMAC / 7.276s = **1.87 TFLOP/s**
-> RTX 3090 fp32 피크 35.6 TFLOP/s의 **약 6%**.

`gemm_nt_bias`는 thread당 출력 1개라 내부 루프가 MAC 1회마다 shared를 2번 읽는다
(shared bandwidth bound). thread당 4x4~8x8 출력을 레지스터에 들고 도는 **register tiling**이
정석이며 보통 피크의 30~50%까지 간다. 87%에 걸린 레버이므로 기대값이 가장 크다.
-> EXP-005 (register-tiled GEMM).
그 다음 후보: expert 16개를 grouped GEMM 1회로 묶어 launch/tail 낭비 제거,
host attention(1.1s)과 host 라우팅(t_moe에 포함, 634k회 직렬 `route_row`) 커널화.

---

# EXP-005 (register-tiled GEMM)
## 1. Background
EXP-004 이후 남은 시간의 87%가 device GEMM이었다 (gemm 13.13s + moe 7.27s / 23.90s).
`gemm_nt_bias`는 thread당 출력 1개라 MAC 1회마다 shared를 2번 읽는다. 실효 성능은
피크의 6% 수준(2.0 / 1.9 TFLOP/s vs 35.6)이었고, 이는 FLOP이 아니라 **shared bandwidth**에
막힌 전형적인 신호다.

## 2. Hypothesis
thread당 TMxTN 출력을 레지스터에 들고 돌면 내부 루프의 shared 읽기가 MAC당 2회에서
`(TM+TN)/(TM*TN)` = 4x4일 때 0.5회로 **4배 줄어든다**. shared bound라면 GEMM이 3~5x 빨라지고
전체는 2.5~3x.

정확도: per-output의 k 누적 순서(오름차순)와 `acc += a*b` 식 형태를 그대로 두면 각 출력이 보는
FMA 시퀀스가 바뀌지 않는다 -> **EXP-004와 bitwise 동일**해야 한다. 이게 검증 기준이다.
(host GEMM은 `objdump`로 확인한 결과 `vmulss`+`vaddss`로 FMA를 내지 않는다. 즉 EXP-002 이후
device GEMM은 애초에 host와 bit-exact가 아니고 1.79e-04로 임계값 안에만 있다. 그러므로
비교 기준은 host가 아니라 직전 실험 출력이다.)

## 3. Design and Implementation
`BM=64, BN=64, BK=16, TM=TN=4` -> block 256 thread, block당 64x64 출력.
- shared는 **k-major로 전치** 저장: `as[BK][BM+4]`, `bs[BK][BN+4]`.
  compute 루프가 한 k에 대해 m/n 방향으로 연속 접근하므로 `float4`로 4개를 한 번에 읽는다.
- `+4` 패딩의 의도: row stride를 68 float = 272B로 만들어 **16B 정렬을 유지하면서**(float4
  shared 로드 필수 조건) bare `[BK][64]`의 bank-conflict 패턴을 깬다. `+1`은 정렬이 깨져 못 쓴다.
- global 로드도 `float4`: A/B 타일이 각각 64x16 = 256 float4 = thread 수와 정확히 일치해
  thread당 1회. K가 16의 배수라 k 방향 bound check가 불필요(호출부에서 `in % BK` 검증).
- M/N tail은 로드 시 0 채움 + store 시 bound check. M=19803은 64의 배수가 아니고
  gate는 N=16이라 BN=64의 1/4만 쓰지만, gate는 전체 MAC의 0.3%라 무시.

한 개 커널로 projection/gate/expert/lm_head 전부 처리한다. 별도 경로를 만들지 않았다.

## 4. Result
| n | EXP-004 | EXP-005 | speedup | seq/s | Validation |
|---|---|---|---|---|---|
| 8 | 0.727 s | **0.456 s** | 1.59x | 17.541370 | PASSED (max 9.16e-05) |
| 1024 | 23.901 s | **6.344 s** | **3.77x** | **161.406779** | PASSED (max 1.79e-04) |

가설(2.5~3x)을 **넘겼다** — 실제 3.77x.
- **`cmp exp004_n1024.bin exp005_n1024.bin` -> BITWISE IDENTICAL.** 누적 순서를 건드리지 않았음이
  증명됨. 성능 실험에서 이게 가장 강한 안전망이다.
- **제출: 160.333047 seq/s (n=1024, 리더보드 1위/7). 개인 최고 42.68 -> 160.3 (+275.7%).**
- baseline 0.033306 대비 **4846x**.
- n=1024 두 번 연속 6.3442 / 6.3437s — 변동 0.01%. 이전의 ±5~8%는 host 연산 비중이 컸을 때의 것.

프로파일 (n=1024, `APS_PROFILE=1`):
| 단계 | EXP-004 | EXP-005 | 비중 |
|---|---|---|---|
| gemm | 13.133 | **1.704** | 27% |
| moe | 7.273 | **1.276** | 20% |
| attn | 1.148 | 1.126 | 18% (**host**) |
| d2h | 0.938 | 0.865 | 14% |
| h2d | 0.524 | 0.548 | 9% |
| norm | 0.349 | 0.352 | 6% |
| lm_head | 0.147 | 0.085 | 1% |
| resid | 0.074 | 0.074 | 1% |

실효 성능: projection+gate+lm_head 26.8 TFLOP / 1.704s = **15.7 TFLOP/s (피크의 44%)**,
expert 13.6 TFLOP / 1.276s = **10.7 TFLOP/s (30%)**. 6% -> 44%. shared bound 진단이 맞았다.

## 5. Next action
병목이 평평해졌다. 단일 최대 항목이 더 이상 없다.
- **host attention 1.13s (18%)** 가 이제 최대 단일 비-GEMM 항목이고, `d2h`+`h2d` 1.41s(23%)의
  대부분이 attention 때문에 q/k/v를 내리고 context를 올리는 왕복이다. 둘을 합치면 **2.5s / 40%**.
  -> attention 커널화가 가장 큰 레버. sliding window 2047 + GQA(16Q/4KV)라 flash 스타일이 필요.
- expert GEMM은 M이 작은(평균 T*2/16 ~ 2475행) 16개 GEMM으로 쪼개져 있어 tail/launch 낭비가 있다.
  피크의 30%에 그친 이유. grouped GEMM 1회로 묶는 것이 다음 후보.
- host 라우팅(634k회 직렬 `route_row`)은 `moe`에 포함되어 있어 아직 분리 측정되지 않았다.

---

# EXP-006 (attention을 커널로: q/k/v/context를 device에 상주)
## 1. Background
EXP-005 이후 남은 6.34s 중 attention 관련이 **2.54s / 40%**였다: host `attn` 1.126s와,
q/k/v를 내리고 context를 올리기 위한 버스 왕복(`d2h` 0.865 + `h2d` 0.548의 대부분).
레이어당 243MB↓ + 162MB↑, 32 레이어 합 13GB.

후보 3개 중 attention을 고른 이유 — 이득이 가장 크고(40%), **이후 실험에 주는 제약이 없다**.
host 코드와 버스 왕복을 제거할 뿐 MoE/GEMM/norm 경로를 건드리지 않고, 끝나면 활성값이
device를 떠나는 곳이 router logits 1곳만 남아 다음 실험의 표면이 좁아진다.
(grouped expert GEMM은 이득 ~1.09x인데 나중에 라우팅을 device로 옮기면 다시 짜야 한다.
device routing은 이득이 미측정이고 `route_row`의 양자화+tie-eps가 정확도 지뢰다.)

## 2. Hypothesis
RoPE + softmax attention을 커널로 옮겨 q/k/v/context를 device에 상주시키면 host 1.13s와
버스 왕복 1.2s가 사라져 6.34s -> **~4.0s (1.5~1.6x)**.

연산량은 근거가 아니다 — attention 전체가 30 GMAC(전체의 0.15%)다. 이건 FLOP 실험이 아니라
**host/버스 제거 실험**이고, 그래서 커널이 느려도(스레드당 직렬 dot) 이긴다.

## 3. Design and Implementation
커널 2개만 추가하고 그 자리의 host 코드만 지웠다. GEMM 튜닝/grouped GEMM/device routing은 넣지 않았다.

- `rope_rows`: 행당 1 block, in-place. 스레드 1개가 (j, j+64) 쌍 하나를 담당.
- `attention_heads`: **(시퀀스, query head)당 1 block, HEAD_DIM(128) 스레드**. 스레드 d가 출력 원소 d를 소유.
  shared는 `[q행 128][denom 1][score max_len]`. k/v는 shared에 올리지 않고 L1/L2에 맡긴다
  (한 head의 k 작업집합이 16KB).
- `d_pos[T]`(시퀀스 내 위치), `d_begin/d_len[B]`를 generate() 시작에 1회 업로드.

### 정확도 — 이 실험의 실제 난점
EXP-004에서 배운 것: 누적 순서가 1e-7 흔들리면 `route()`의 양자화 경계에서 expert **집합**이
뒤집혀 O(0.1) 오차가 난다. 그래서 bit-exact로 설계했다.
- **`objdump obj/model.o`로 host를 먼저 확인**: `attend_sequence`에 `vfmadd`가 **0개**.
  GCC가 dot product의 곱만 `vmulps`로 벡터화하고 덧셈은 순서대로 `vaddss`로 낸다.
  -> device도 `__fmul_rn`/`__fadd_rn`으로 FMA 계약을 막아야 한다.
- q·k dot은 한 스레드가 d 오름차순 직렬 (128개 의존 FADD — 느리지만 8%짜리라 무관)
- softmax denom은 thread 0이 ki 오름차순 직렬, `exp`는 double (`accurate_exp`와 동일)
- acc는 스레드 d가 ki 오름차순 직렬
- **RoPE cos/sin 테이블은 host가 `generate()` 안에서 계산**해 8KB만 올린다.
  device `powf/cosf/sinf`는 glibc와 1 ulp 다를 수 있고 그 1e-7이 정확히 라우팅을 뒤집는 크기다.
  측정 구간 안이고, 입력이 아니라 config에서 나오는 상수라 캐싱 금지 조항과 무관하다.

## 4. Result
| n | EXP-005 | EXP-006 | speedup | seq/s | Validation |
|---|---|---|---|---|---|
| 8 | 0.456 s | **0.345 s** | 1.32x | 23.193103 | PASSED (max 9.16e-05) |
| 1024 | 6.344 s | **3.93 s** | **1.61x** | **259.986279** | PASSED (max 1.79e-04) |

가설(1.5~1.6x)이 **맞았다** — 실제 1.61x.
- **`cmp exp005_n1024.bin exp006_n1024.bin` -> BITWISE IDENTICAL.** host 테이블 + `__f*_rn` +
  누적 순서 보존이 전부 통했다. 정확도 실험이 1회로 끝났다.
- 런 간 변동: 10회 중 8회가 [3.889, 3.939], 2회가 4.567s. 이상치는 프로파일에서 gemm 1.717->2.106,
  moe 1.257->1.496, attn 0.303->0.363으로 **모든 커널이 균일하게 1.2x** 느려졌다 -> GPU 클럭.
  **[EXP-007에서 정정] 원인은 노드다.** b5/b6는 gemm 1.72, b1은 gemm 2.11로 노드마다 1.19x 차이가
  고정돼 있다. 여기서 "노드와 무관"이라 적은 것은 틀렸다. **이후 모든 A/B는 srun allocation
  1개 안에서 교차 실행해야 한다.**
- **제출: 262.949606 seq/s (n=1024). 개인 최고 160.3 -> 262.9 (+64.0%). baseline 0.033306 대비 7895x.**
  단 **순위 2위/10** — EXP-005 때는 1위였다. 리더보드가 움직였다.
- 비교 공정성: EXP-005를 `b19dfa4`에서 같은 세션에 재빌드해 재측정(6.318/6.392/6.318s,
  출력 bitwise 동일)했으므로 6.344는 신뢰할 수 있는 기준이다.

프로파일 (n=1024):
| 단계 | EXP-005 | EXP-006 | 비중 |
|---|---|---|---|
| gemm | 1.704 | 1.717 | **44%** |
| moe | 1.276 | 1.257 | **32%** |
| attn | 1.126 (host) | **0.303** (device) | 8% |
| norm | 0.352 | 0.142 | 4% |
| lm_head | 0.085 | 0.085 | 2% |
| resid | 0.074 | 0.074 | 2% |
| h2d | 0.548 | **0.030** | 1% |
| d2h | 0.865 | **0.006** | 0.2% |

- `d2h` 0.865 -> 0.006 (**144x**). 남은 6ms는 router logits 40MB뿐이다.
- 제거한 2.50s(host 1.126 + 버스 1.377) 대비 추가한 커널 0.303s -> 순 -2.20s.
- attention 커널 실효 성능은 60 GFLOP / 0.303s = 198 GFLOP/s (피크의 0.6%). 직렬 dot 때문이지만
  8%짜리라 지금 손댈 이유가 없다.

## 5. Next action
device GEMM이 다시 **76%**(gemm 1.717 + moe 1.257)다. 단, `moe` 1.257s의 내부 구성이
아직 **미측정**이다 — expert GEMM / 16회 launch 오버헤드 / **host 라우팅 634k회 직렬 `route_row`**가
한 타이머에 섞여 있다. 추측하지 말고 먼저 쪼개 재는 것이 EXP-007이다(코드 변경 없음, 타이머만).
그 결과에 따라:
- 라우팅이 크면 -> device routing (router D2H 0.006s까지 같이 사라진다)
- expert GEMM이 크면 -> grouped GEMM 1회로 16개 launch/tail 낭비 제거
- 둘 다 작으면 -> `gemm_nt_bias` 자체 튜닝(현재 피크의 44%)

# EXP-007 (register tile 4x4 -> 8x8: GEMM 커널 하나만 키우기)
## 1. Background
EXP-006 이후 3.93s 중 **76%가 `gemm_nt_bias` 단일 커널**이다(gemm 1.717 + moe 1.257).
shape로 FLOP을 세어 효율을 붙여보니:

| 타이머 | time | FLOP | 달성 | 3090 FP32 peak(35.6 TF/s) 대비 |
|---|---|---|---|---|
| gemm (q/k/v/o/gate) | 1.717 s | 26.66 TFLOP | 15.5 TF/s | **43.6%** |
| moe (expert GEMM + 부대비용) | 1.257 s | 13.96 TFLOP | 11.1 TF/s | **31.2%** |

두 타이머가 서로 다른 문제가 아니라 같은 커널이다. 그래서 예정했던 "moe 타이머 쪼개기"를
철회했다 — 쪼갠 답이 무엇이든 다음 행동이 "공유 GEMM 커널을 고친다"로 같다.
(q_proj 640 FLOP/byte, expert w1 175 FLOP/byte -> 전부 compute-bound. 대역폭 레버가 아니다.)

## 2. Hypothesis
`gemm_nt_bias`는 shared-memory 트래픽/ILP에 묶여 있다. 스레드당 출력을 4x4 -> 8x8
(BM=BN=128, BK=8, 256 threads 유지)로 올리면 inner loop의 shared load 1회당 FMA가 2 -> 4로
두 배가 되어 peak 43.6% -> 60%+로 오르고 전체 1.15~1.3x.

## 3. Design and Implementation
`src/model.cu` 커널 1개, ~30줄.
- `BM=64,BN=64,BK=16,TM=TN=4` -> `BM=128,BN=128,BK=8,TM=TN=8`. thread 수 256 유지,
  A/B 타일 로드도 스레드당 float4 1개씩 그대로(`lr=tid/2` 0..127, `lc=(tid%2)*4`).
- inner loop만 float4 1회 -> `TM/4`회 로드로 확장. shared 8448B, acc 64 reg.
- **비트 동일성은 사전 보장**: k는 여전히 0->K-1 오름차순, 출력당 accumulator 1개,
  `acc += a*b` 표현식(=FMA, host `matmul_transposed`와 동일) 유지. 타일 크기는 누적 순서를
  건드리지 않는다. 순서를 깨는 split-K는 하지 않았다. 모든 K(4096/2048/448)가 8로 나뉜다.
- 끼워넣지 않은 것: double-buffering, gate 전용 커널, grouped expert GEMM, device routing.

## 4. Result
**b1 단일 allocation 안에서 006/007 교차 3회** (각 반복 편차 <0.01s):

| | gemm | moe | elapsed |
|---|---|---|---|
| EXP-006 (64x64, 4x4) | 2.105 | 1.495 | **4.561** |
| EXP-007 (128x128, 8x8) | 1.803 (**1.17x**) | 1.836 (**0.81x**) | **4.599 (0.99x)** |

**가설은 절반만 맞았고 전체로는 반증됐다.**
- wide-N GEMM(q/k/v/o): **1.17x, -0.302s.** 43.6% -> 51% of peak. 레버 자체는 실재한다.
- narrow-N expert GEMM: **0.81x, +0.341s.** 설계 때 적어둔 위험(M/N 패딩 낭비)이 현실화됐고,
  그보다 **블록 수 붕괴**가 크다. w1/w3(M~2475, N=448)이 39x7=273블록 -> 20x4=80블록.
  82 SM에 80블록 = SM당 1블록 = 8 warp(최대 48) -> latency hiding이 죽는다.
  N 패딩 낭비 12.5%만으로는 0.81x(=+23% 시간)를 설명할 수 없으므로 점유율이 주범이다.
- 순효과 **-0.8% 회귀**. 코드를 EXP-006으로 롤백했다.
- 정확도: n=8 / n=1024 모두 `cmp` **BITWISE IDENTICAL**. 타일 크기가 누적 순서를 건드리지
  않는다는 사전 논증이 맞았다 -> 앞으로 타일 shape는 정확도 리스크 없이 자유롭게 바꿔도 된다.
- **방법론 사고를 하나 잡았다**: 처음엔 007=3.78s vs 006=3.93s로 1.04x 이득처럼 보였는데
  **서로 다른 노드**였다. 노드 클럭이 1.19x 고정 차이라 교차 측정 아니면 이 크기의 효과는
  측정 자체가 불가능하다. EXP-006의 "노드와 무관" 기록도 정정했다.

## 5. Next action
반증이 아니라 **분기 지점**이다: 큰 타일은 wide-N에서 -0.302s를 벌고 narrow-N에서 +0.341s를
잃는다. 두 호출 지점이 서로 다른 타일 shape를 원한다는 것이 이 실험의 실측 결론이다.
EXP-008 = **shape별 타일 선택** — 커널을 타일 상수로 템플릿화해 N>=1024는 128x128/8x8,
narrow(expert w1/w3 N=448, gate N=16)는 64x64/4x4 유지. 두 숫자 모두 이미 측정돼 있으므로
예측이 아니라 산수다: b1에서 3.600 -> 3.298, 즉 4.56 -> **4.26s (1.07x)**.
그 다음 후보는 여전히 device routing과 grouped expert GEMM이며, 둘 다 이 커널을 물려받는다.

---

# EXP-008
## 1. Background
EXP-007은 8x8 레지스터 타일(128x128, BK=8)이 wide GEMM에서 1.17x, expert GEMM에서
0.81x로 전체 0.99x라 롤백했다. 그때 wide GEMM은 15.5 TF/s(peak의 43.5%)에 머물렀는데,
8x8의 shared-BW 천장은 100%다. 즉 shared 대역폭이 아닌 다른 것이 묶고 있다.
K 루프가 `LDG -> STS -> sync -> compute -> sync` 구조라 매 반복 글로벌 로드에서 멈춘다.

## 2. Hypothesis
shared를 2중 버퍼로 두어 tile kt+BK의 글로벌 로드를 tile kt의 compute보다 먼저 발행하면
LDG 지연이 FFMA에 가려지고, 버퍼가 교대하므로 반복당 `__syncthreads()`도 2회에서 1회로 준다.
-> GEMM 처리량이 오른다. 누산 순서는 그대로이므로 **출력은 bitwise 동일해야 한다**.

## 3. Design and Implementation
`gemm_nt_bias`만 수정. as/bs를 `[2][BK][...]`로 늘리고 프롤로그에서 tile 0을 적재,
루프는 `LDG(kt+BK) -> compute(cur) -> STS(cur^1) -> sync` 순. 버퍼가 교대하므로 이전
반복의 sync가 write-after-read를 보장한다. 타일은 EXP-007과 동일(128x128, BK=8, 8x8).
대조군을 두 개 둔다: 006(현행 4x4)과 007(8x8, 단일 버퍼). 한 srun 안에서 교차 실행.

## 4. Result
b1, 3회 반복(편차 +-0.003s):

| | elapsed | vs 006 | Validation |
|---|---|---|---|
| 006 (4x4/64x64/BK=16) | 4.566 s | - | PASSED |
| 007 (8x8/128x128/BK=8) | 4.598 s | 0.993x | PASSED |
| **008 (007 + double buffer)** | **4.388 s** | **1.041x** | PASSED |

`cmp outputs/exp008_006.bin outputs/exp008_008.bin` -> **bitwise 동일**. 3자 모두
max abs diff 0.000179291로 EXP-006과 같다. 타일 shape도 double buffering도 정확도 리스크 없음.

b6, 단계별(같은 할당 내):

| | gemm | moe | attn | norm |
|---|---|---|---|---|
| 006 | 1.719 | 1.259 | 0.304 | 0.142 |
| 007 | 1.386 | 1.468 | 0.284 | 0.138 |
| 008 | **1.341** | 1.398 | 0.283 | 0.140 |

- **wide GEMM 1.719 -> 1.341 = 1.28x.** 26.66 TFLOP 기준 15.5 -> **19.9 TF/s (43.5% -> 55.9%)**.
- **가설은 맞았지만 메커니즘은 틀렸다.** SASS를 보면 ptxas가 LDG를 compute 블록 *뒤로*
  sink시켰다 -- 레지스터 127개로 2 blocks/SM 경계(128x256x2=65536)에 붙어 있어서, float4
  두 개를 512 FFMA 구간 내내 살리면 occupancy가 1 block/SM으로 떨어지기 때문. 지연 은닉은
  실패했고, 실제로 작동한 것은 **반복당 BAR.SYNC 2회 -> 1회**뿐이다(k=4096/BK=8 -> 블록당 512회).
- moe는 1.259 -> 1.398로 여전히 0.90x 퇴행. double buffering이 007 퇴행의 절반만 회수했다.
  expert GEMM만 보면 15.0 -> 13.0 TF/s. 원인은 EXP-007과 동일 -- N=448, expert당 M~2475라
  128x128 타일에서 블록이 80개로 붕괴(82 SM).

**부수 결론: feature-major 전환은 보류.** 서브에이전트가 feature-major+DB+큰타일로 측정한
21.8 TF/s에 **feature-major 없이 19.9까지 도달**했다. 남은 간격 1.9 TF/s ~ 0.12s에
layer_norm coalescing ~0.10s를 더해도 0.22s인데, 대가는 전 커널 재인덱싱이다.

### 4-1. ncu 실측 (n=1024, `-s 400` -> 전부 expert GEMM에 착지)

| grid | 커널 | 블록 | sm__throughput | long_scoreboard | sectors/req |
|---|---|---|---|---|---|
| (4,10)/(4,18) | expert w1/w3 (N=448) | 40/72 | 24.4/43.1% | 25.9/26.6% | 15.78 |
| (32,6)/(32,10) | expert w2 (N=4096) | 192/320 | 39.1/49.2% | 14.8/16.9% | 14.47 |

- **feature-major 불필요 확정.** sectors/request 14.5~15.8 (이상값 16의 90~99%),
  dram_throughput 3.7~10.8%. 메모리 경로에 고칠 것이 없다. 현재 로드는 스레드 2개가
  한 행의 32B(정확히 1섹터)를 채우므로 레이아웃을 바꿔도 섹터 수는 그대로다.
- **블록 붕괴 실측.** 2 blocks/SM x 82 SM = 164 슬롯인데 w1/w3는 40~72개(24~44%)만 채운다.
- **새 발견: N 패딩 낭비.** N=448에 BN=128이면 타일 4개=512열 -> **12.5% 버림**.
  BN=64일 때는 448=7x64로 낭비 0이었다. 8x8 타일은 expert GEMM에 블록 붕괴와 N 낭비를
  동시에 입혔고, EXP-007에서 나는 전자만 보고 있었다.
- **cp.async 순위 하향.** long_scoreboard가 블록 적은 w1/w3 26%, 잘 찬 w2 15%다. 지연
  stall의 상당 부분이 워프 부족의 증상이므로, grouping으로 점유율을 올린 뒤 다시 재야 한다.

grouped 융합은 둘 다 동시에 해결한다: w1||w3 -> N=896 = 7x128로 낭비 0, expert를
blockIdx.z로 -> 블록 7x20x16 = 2240.

## 5. Next action
**EXP-009: grouped expert GEMM.** w1/w3를 N=896으로 융합하고 expert를 `blockIdx.z`로 올려
블록 80 -> ~2240. wide GEMM과 같은 ~19.9 TF/s로 가면 expert GEMM 1.07 -> 0.70s (**-0.37s**).
feature-major(0.22s)보다 크고 변경 범위는 MoE 디스패치로 국한된다.

미착수로 남은 것: **글로벌 로드 지연 은닉.** 레지스터 압력이 막고 있으므로 해법은 레지스터를
우회하는 `cp.async.ca.shared.global`(Ampere PTX, 글로벌->shared 직접). EXP-009 이후 후보.

---

# EXP-009: attention softmax의 FP64 exp를 lane에 분산

## 1. Background
EXP-008 이후 stage 분포(b6): gemm 1.341 / moe 1.398 / **attn 0.283** / norm 0.140.
`docs/exploration-003.md`는 2등 참가자의 "Attention Q-head 4-way"(390.6 -> 112.0 ms, 3.49x)를
최우선으로 권했다. **그러나 그 설계는 우리 코드에 이식되지 않는다.** 저쪽은 block이 GQA group
1개를 맡아 thread 0이 q head 4개의 score/softmax를 연달아 계산했고, 그것을 thread 0~3에
나눈 것이 3.49x였다. 우리 커널은 block = (b, qh)라 **q head 4개가 이미 별도 block에서 병렬**이다.
저쪽이 추가한 병렬성을 우리는 이미 갖고 있으므로 배수는 전이되지 않는다.

## 2. Hypothesis
attn 시간의 지배분은 `if (tid == 0)` 직렬 구간이고, 그 안의 **double `exp`**가 사실상 전부다.
`exp`는 elementwise이므로 lane에 분산해도 어떤 합산 순서도 건드리지 않는다 -> **bitwise 동일**.

## 3. Design
직렬 구간을 셋으로 쪼갠다: (a) tid 0이 max 스캔 -> shared 브로드캐스트, (b) **128 lane이
`sw[i] = exp(sw[i]-max)`를 나눠 계산**, (c) tid 0이 denom을 ki 오름차순으로 직렬 누적.
max는 order-independent, denom은 순서 그대로 -> 값이 바뀔 여지가 없다.

## 4. Result
한 srun(b6) 안에서 3변형 interleave x2rep. C는 직렬 구간을 통째로 삭제한 **진단용 하한**(결과 오답).

| 변형 | attn (s) | 비고 |
|---|---|---|
| A 기준(EXP-008) | 0.289 / 0.293 | |
| **B exp 분산** | **0.181 / 0.184** | **-0.109 s, 1.59x** |
| C 직렬 구간 삭제 | 0.178 / 0.178 | 하한 |

- **B가 가용 여유의 96%를 회수**했다(0.109 / 0.113). 남은 max+denom 직렬 비용은 ~0.005 s로,
  이 방향에 더 팔 것이 없다. Q-head 4-way를 완벽히 해내도 상한이 0.113 s다.
- `cmp outputs/exp009.bin` vs EXP-008 출력 -> **bitwise 동일**. Validation PASSED
  (max abs 0.000179291, EXP-008과 같은 값).
- exploration-003의 예측 -0.22 s는 **우리 코드에서 도달 불가**다. 우리 하한이 0.178 s이므로
  저쪽의 0.090 s는 score dot / value 누적까지 재구성해야 나온다.

## 5. Next action
**EXP-010: grouped expert GEMM.** w1/w3를 N=896으로 융합하고 expert를 `blockIdx.z`로 올려
블록 80 -> ~2240. EXP-008 §4-1 ncu가 블록 붕괴(164 슬롯 중 40~72)와 N 패딩 12.5%를 동시에
지목했고, 융합이 둘 다 지운다. 기대 -0.37 s.

남은 attn 0.178 s의 내역(score dot 대 value 누적)은 아직 분해하지 않았다. MoE/embedding보다
작으므로 후순위.

---

# EXP-010: expert W1/W3를 하나의 GEMM으로 융합

## 1. Background
EXP-009 이후 b6: gemm 1.353 / **moe 1.415** / attn 0.183 / norm 0.141. moe가 최대 항목이다.
EXP-008 §4-1 ncu가 expert GEMM의 두 손실을 지목했다 — 블록 붕괴(164 슬롯 중 40~80)와
**N=448에 BN=128이라 4타일=512열을 계산하는 12.5% 패딩 낭비**.

계획서의 "W1/W3 fusion + grouped GEMM"은 가설 2개다. 융합이 N=896=7x128을 만들어
grouping의 전제가 되므로 **융합만 먼저** 잰다(grouping은 EXP-011).

## 2. Hypothesis
w1[e](448x4096)와 w3[e](448x4096)를 행방향으로 쌓아 **896x4096 한 장**으로 만들면
N=896=7x128로 패딩이 0이 되고, launch가 2회에서 1회로 준다.
출력 원소별 내적의 K 순서가 그대로이므로 **bitwise 동일**이어야 한다.

## 3. Design
- `DeviceLinear`에 두 weight를 행방향 concat하는 생성자 추가. `DeviceMoE::{w1,w3}` -> `w13`.
- `silu_mul`이 `[rows, 2F]` 한 버퍼를 읽어 `[rows, F]`로 쓰도록 인덱싱만 변경
  (`(x/(1+expf(-x)))*up` 식은 그대로).
- 버퍼 `d_up(total*F)` -> `d_gate_up(total*2F)`.

## 4. Result
한 srun(b6) 안에서 교차 x2rep, 둘 다 `-v`.

| | moe (s) | elapsed (s) |
|---|---|---|
| EXP-009 | 1.417 / 1.412 | 3.562 / 3.558 |
| **EXP-010** | **1.158 / 1.152** | **3.313 / 3.301** |

**moe -0.259 s (1.23x), elapsed -0.253 s.** `cmp` -> **bitwise 동일**, 양쪽 PASSED.

**예측(-0.09 s)의 3배가 나왔다.** 패딩 12.5%만 계산했는데, 융합이 **블록 붕괴도 절반
해결**했기 때문이다. 이전에는 (4,20)=80 블록짜리 launch 2개가 **직렬**로 각각 164 슬롯의
49%만 채웠는데, 융합 후 (7,20)=140 블록 1개가 85%를 채운다. 두 효과가 겹쳤다.

⇒ **EXP-011 grouping의 남은 여유는 계획서의 -0.28 s보다 작다.** w1/w3는 이미 85%를
채우므로 남은 대상은 w2(N=4096, (32,20)=640 블록 — 이미 충분)와 launch 수 자체다.
grouping 전에 남은 moe 1.155의 내역을 먼저 재야 한다.

## 5. Next action
**EXP-011 착수 전 측정.** moe 1.155가 expert GEMM / gather+scatter / per-expert
`cudaMemcpy` 중 어디에 있는지 분해한다. w1/w3 점유율이 이미 85%라면 grouping의 값은
낭비 제거가 아니라 **launch·H2D 왕복 제거(레이어당 112회)** 쪽으로 옮겨간다.

---

# 측정-A: MoE 및 전체 커널 시간 분해 (nsys, EXP-010 코드, 노드 b1)

EXP-011을 grouping으로 짤지 결정하기 위해 코드 변경 없이 `nsys profile -t cuda` 1회.
b1은 느린 노드이므로 절대값은 b6 대비 약 1.19배. 커널 총합 3.497 s.

| 항목 | 시간 (s) | launch | 비고 |
|---|---|---|---|
| q/o_proj | 0.657 + 0.656 | 64 | 각 10.63 TFLOP -> **16.2 TF/s** |
| k/v_proj | 0.343 | 64 | 15.5 TF/s |
| **expert w13** | **0.800** | 512 | 9.30 TFLOP -> **11.6 TF/s** |
| **expert w2** | **0.379** | 512 | 4.65 TFLOP -> 12.3 TF/s |
| attention | 0.197 | 32 | |
| layer_norm | 0.167 | 65 | |
| scatter/gather/silu | 0.077 / 0.052 / 0.009 | 512 x3 | |
| add_inplace | 0.073 | 64 | |
| router gate / rope / lm_head | 0.050 / 0.016 / 0.020 | | |

## 결론 1: w13의 손실은 **wave 충전율**이 전부다
`gridX=7`(N=896)이라 블록이 `7 x rowtiles`뿐이다.

| | 시간가중 wave 충전율 | 1 wave(164 블록) 미만 launch |
|---|---|---|
| expert w13 | **66%** | **0.423 s = 53%** |
| expert w2 | 89% | 0.007 s = 2% |

11.6 TF/s / 0.66 = 17.6 -> 충전율만 채우면 wide GEMM(16.2)과 같은 대역에 들어온다.
**충전율이 격차를 전부 설명한다.** w2는 `gridX=32`라 이미 89%로 손댈 것이 없다.

## 결론 2: launch 수와 index H2D는 레버가 아니다
per-expert index `cudaMemcpy` 776회 합 **1.1 ms**. grouping의 값은 launch 제거가 아니라
**오직 w13 충전율**이다. CUDA Graph 사망 판정과 같은 결론.

## 결론 3: 순위 재산정
- **w13 grouping**: 0.800 -> ~0.57 = **-0.23 s (b1) ~ -0.19 s (b6)**. work queue로
  (expert, rowtile) 평탄화 필요 — max rows로 잡으면 불균형만큼 빈 블록이 생긴다.
- **prefix trie**: 토큰 19,803 -> 15,583 (-21.3%)은 위 표의 **거의 모든 항목에 비례로**
  꽂힌다(projection 1.656 + expert 1.179 + attn + norm). b1 기준 **-0.7 s** 규모.

⇒ trie가 grouping보다 3배 크고 둘은 독립이다. **EXP-011 = prefix trie**, grouping은 그 뒤.

---

# EXP-011: Prefix trie (공유 prefix 중복 제거)

## 1. Hypothesis
causal attention + 절대위치 RoPE이므로, 앞선 토큰이 전부 같은 두 토큰은 32개 레이어
전부에서 **동일한 hidden state**를 갖는다. 행을 토큰이 아니라 **prefix trie 노드**로
잡으면 공유 prefix를 1번만 계산한다. 정확히 CSE — 산술 변경 0.

입력 실측: 19,803 토큰 -> **15,583 노드 (-21.31%)**.

## 2. Change (`src/model.cu`, `attention_heads` + `generate()`)
- `generate()` 앞에서 `(parent, token) -> node` 해시로 trie 구성.
  노드당 `token/parent/depth`, 시퀀스당 종단 노드.
- 각 노드의 key 집합 = 자기 root path. `anc_off[n]..anc_off[n+1]`에 depth 오름차순으로
  **평탄화**(215,717 int = 863 KB). attention이 parent 체인을 되짚지 않아도 된다.
- attention: `block=(sequence, head)` + 내부 `qi` 루프 -> **`block=(node, head)`, 1 블록
  1 쿼리**. key/value를 `chain[i]` 인덱스로 읽는다. 누산 순서는 전부 그대로.
- position = trie depth. lm_head 행 = 시퀀스 종단 노드.

## 3. Result (노드 b5, 한 allocation 안에서 교차 2rep)

| | T | embed | norm | gemm | attn | moe | resid | elapsed | seq/s |
|---|---|---|---|---|---|---|---|---|---|
| EXP-010 | 19,803 | (미측정) | 0.141 | 1.351 | 0.182 | 1.151 | 0.074 | 3.2807 | 312.1 |
| **EXP-011** | **15,583** | 0.180 | 0.112 | 1.045 | 0.137 | 0.966 | 0.058 | **2.6529** | **386.1** |

**elapsed -0.628 s (1.237x).** `cmp` -> **bitwise 동일**, 양쪽 PASSED. decode(`-d`) PASSED.

EXP-010의 `embed`는 타이머 시작 전이라 0으로 찍혔다. 이번에 타이머를 `generate()` 맨
앞으로 올려 trie 구성 + embedding gather를 실제로 재게 했다. 미계상분으로 역산하면
0.263 -> 0.222 s이므로 **trie 구성 자체는 수 ms**다.

## 4. Analysis
행 수에 비례하는 항목은 전부 -21% 근처로 떨어졌다(gemm -22.6%, moe -16.1%,
norm -20.6%, resid -21.6%). **attn만 -24.7%로 예상(-6.5%)을 크게 넘겼다.**
attention 연산량은 sum(depth+1) 기준 230,759 -> 215,717 = -6.5%에 불과하다 —
나머지는 `qi` 직렬 루프를 없애고 블록을 16,384개(각 최대 32회 반복)에서
249,328개(각 1회)로 편 **병렬성 효과**다. 두 변경이 한 커밋에 섞였다는 뜻이지만,
trie를 넣으면 `qi` 루프는 구조상 유지할 수 없다(노드의 조상은 연속이 아니다).

moe의 -16.1%가 행 감소(-21.3%)에 못 미치는 것은 예상대로다. expert당 행이 줄면
w13 wave 충전율(66%)이 더 나빠진다 — **grouping의 상대 가치는 오히려 올라갔다.**

## 5. Next action
**EXP-012 = expert w13 grouping.** (expert, rowtile) work queue로 평탄화.
행이 21% 줄어 충전율이 더 나빠졌으므로 측정-A의 -0.19 s(b6)보다 커질 가능성이 있다.

---

# EXP-012: expert GEMM grouping (16개 launch -> 1개)

## 1. Hypothesis
측정-A: expert w13의 시간가중 wave 충전율 66%, 시간의 53%가 1 wave(164 블록) 미만
launch. expert당 grid가 `7 x rowtiles`뿐이기 때문이다. EXP-011로 행이 21% 더 줄어
충전율은 더 나빠졌다. 16개 expert의 행을 한 버퍼에 이어 붙이고 **(expert, rowtile)
work queue로 평탄화**하면 한 launch가 그리드를 채운다.

## 2. Change (`src/model.cu`, `include/model.h`)
- `gemm_nt_bias` 본문을 `gemm_nt_body(a, b, bias, c, row0, rows, k, n)`로 빼고,
  `gemm_nt_bias`와 새 `gemm_grouped`가 각각 얇게 감싼다. 타일 경계 판정만 `m` 기준에서
  `row0/rows` 기준으로 바뀌고 누산은 손대지 않았다.
- `tiles[]`: 행 타일당 int 3개 (expert, 패킹 버퍼 내 시작행, 남은 행). expert마다
  부분 타일이 최대 1개. 레이어당 ~250 타일.
- gather를 expert별 16회 -> 패킹 버퍼로 **1회**. expert index H2D도 1회.
- w13/w2를 `gemm_grouped` 각 1회. 디바이스 측 weight 포인터 테이블(`w13_ptrs`,
  `w2_ptrs`)은 모델 로드 시 1회 구성.
- **scatter는 expert별로 유지**: 한 행이 expert 2개에 속하므로 launch를 나누는 것이
  두 add의 경쟁을 막는 방법이다. 이미 expert당 블록 수가 1 wave를 훨씬 넘어 얻을 것도 없다.
- 스크래치 행 수 `total` -> `TOP_K * total` (+650 MB, 총 ~16.3 GB / 24 GB).

## 3. Result (노드 b6, 한 allocation 안에서 교차 2rep)

| | norm | gemm | attn | moe | elapsed | seq/s |
|---|---|---|---|---|---|---|
| EXP-011 | 0.114 | 1.046 | 0.137 | 0.968 | 2.6645 | 384.3 |
| **EXP-012** | 0.113 | 1.069 | 0.141 | **0.768** | **2.4907** | **411.1** |

**moe -0.199 s (1.26x), elapsed -0.171 s.** `cmp` -> **bitwise 동일**, 양쪽 PASSED.

## 4. Analysis
moe는 예측(-0.19 s)과 정확히 맞았다. 충전율 가설이 옳았다.

**gemm이 +0.023 s 퇴행했다(1.046 -> 1.070, 2rep 재현).** gemm 구간은 q/k/v/o proj과
router gate뿐이고 이 실험에서 코드가 바뀌지 않았다. 남는 설명은 스크래치가 650 MB
늘어 버퍼 배치가 달라진 것뿐이다. 실측 이득의 12%를 갉아먹지만 원인은 미확인 —
**추측으로 적지 않고 미해결로 남긴다.** attn도 +0.004로 같은 방향이다.

## 5. Next action
남은 큰 항목은 gemm 1.07(= projection)과 moe 0.77이다. 계획서 순서대로면 Q/O
`cp.async`지만, EXP-012의 gemm 퇴행이 **버퍼 배치가 projection 속도에 영향을 준다**는
신호이므로, 그것부터 확인하는 편이 싸다: `d_expert_in/out`을 `TOP_K * total`에서
실제 최대 행 수로 줄이면 gemm이 1.046으로 돌아오는가.

---

# 측정-B: EXP-012의 gemm 퇴행(+0.023 s) 원인

EXP-012에서 코드가 바뀌지 않은 gemm 구간(q/k/v/o proj + router gate)이 b6에서
1.046 -> 1.070으로 재현성 있게 퇴행했다. 후보를 하나씩 죽였다.

## 1. 버퍼 크기? 아니다
EXP-011에 **할당만** EXP-012와 동일하게(`TOP_K * total`, +650 MB) 키운 `011big`을
만들어 3-way 교차(b6):

| | gemm | moe |
|---|---|---|
| 011 | 1.049 / 1.045 | 0.968 / 0.967 |
| 011big | 1.044 / 1.043 | 0.967 / 0.965 |
| 012 | 1.070 / 1.069 | 0.772 / 0.770 |

**할당을 키워도 gemm은 그대로다.** 계획했던 "버퍼를 줄여 되돌린다"는 실험은 여기서 소멸.

## 2. 리팩터링이 `gemm_nt_bias` 코드 생성을 바꿨나? 바꿨지만 원인은 아니다
`gemm_nt_body`로 본문을 빼면서 경계 판정을 `a_row < m` -> `lr < rows`로 쓴 탓에
SASS가 3918 -> 3966 instr로 달라져 있었다. 행 **개수** 대신 **절대 끝행**을 넘기도록
고쳐 본문 표현을 원본과 문자 단위로 되돌렸고, `gemm_nt_bias`의 SASS가 EXP-011과
**완전히 동일**해졌다(`012b`). 그런데 b0 실측:

| | gemm | attn | moe |
|---|---|---|---|
| 011 | 1.054 / 1.057 | 0.138 | 0.977 |
| 012 (SASS 다름) | 1.081 / 1.082 | 0.142 | 0.776 |
| **012b (SASS 동일)** | **1.080 / 1.079** | 0.142 | 0.775 |

**SASS를 원복해도 퇴행이 그대로다.** 코드가 원인이 아니다.

## 3. 원인: boost 클럭. grouped MoE가 전력을 더 쓴다
b1(느린 노드)에서는 **퇴행이 아예 없다**(011/012/012b 모두 gemm 1.33). 퇴행은
b0·b6에서만 난다. 컴퓨트 phase 구간만 잘라 클럭·전력 샘플링(b0, 2rep):

| | SM clk p50 | power p90 | gemm | attn |
|---|---|---|---|---|
| 011 | **1800 MHz** | 354 W | 1.054 / 1.057 | 0.138 |
| 012 | **1755 MHz** | 363 W | 1.081 / 1.082 | 0.142 |

클럭 **-2.5%**, gemm **+2.4%**, attn **+2.9%**. 반면 norm·resid·h2d·d2h·lm_head는
ms 단위로 불변 — **컴퓨트 바운드 단계만 딱 클럭 낙폭만큼 느려졌다.**

grouped MoE는 같은 FLOP을 0.98 s -> 0.78 s에 밀어넣으므로 전력 밀도가 26% 높다.
GPU가 boost를 한 단 내리고, 그 손해를 뒤이은 projection·attention이 가져간다.
b1은 이미 클럭이 낮아 잃을 것이 없다.

## 결론
**코드로 되돌릴 수 없는 물리적 트레이드오프다.** EXP-012 순이득 -0.17 s는 이 손실을
이미 반영한 값이다. 앞으로 **연산 밀도를 올리는 최적화는 빠른 노드에서 이론치보다
2~3% 덜 나온다**고 보고 기대값을 잡아야 한다.

`012b`는 성능상 중립이지만, 손대지 않은 `gemm_nt_bias`의 SASS가 리팩터링 전과 동일함을
보장하므로 유지한다(누산 순서 불변 논증의 근거).

---

# EXP-013: embedding lookup을 디바이스로

## 1. Hypothesis
계획서 순서는 Q/O `cp.async`였으나 **구조적으로 막혀 있다**: 타일이 `as[BK][BM]`로
전치되어 있고(스레드가 k 연속 float4를 읽어 shared 4개 행에 흩뿌린다) `cp.async`는
global->shared 연속 복사만 된다. 전치를 버리면 내부 루프의 shared 로드가
`LDS.128` 4개 -> `LDS.32` 16개가 되어 손해다. 타일 재설계급이라 보류.

대신 EXP-011에서 처음 계측된 `embed 0.181` + `h2d 0.023`이 남은 최대 단일 항목이고,
**전부 호스트 직렬 코드**다: 15,583행 x 16 KB = 255 MB를 CPU 1스레드로 모으고 다시
255 MB를 올린다. 임베딩 테이블(525 MB)을 다른 가중치와 똑같이 로드 시점에 디바이스로
올리면 이 단계는 커널 1개가 된다. 순수 복사라 산술 위험 0.

## 2. Change (`src/model.cu`, `include/model.h`)
- 생성자에서 `d_embeddings_ = device_copy(embeddings_)` (다른 가중치와 동일 취급).
- 호스트 gather 루프와 `Tensor hidden`, `to_device()` 삭제. 대신 노드 토큰 id를
  올리고 기존 `gather_rows`를 그대로 재사용 — 행이 16 KB 연속이라 완전 coalesced.
- `t_embed`는 이제 **trie 구성만** 잰다.

## 3. Result (노드 b0, 한 allocation 안에서 교차 2rep)

| | embed | h2d | gemm | moe | elapsed | seq/s |
|---|---|---|---|---|---|---|
| EXP-012b | 0.180 / 0.186 | 0.029 / 0.033 | 1.088 / 1.086 | 0.780 | 2.5278 / 2.5494 | ~403 |
| **EXP-013** | **0.002** | **0.004 / 0.003** | 1.086 / 1.090 | 0.782 | **2.3084 / 2.3066** | **~444** |

**elapsed -0.23 s.** `cmp` -> **bitwise 동일**, 양쪽 PASSED.

## 4. Analysis
예측 -0.19 s를 넘겼다(-0.23). 다른 단계는 ms 단위로 불변 — 측정-B의 클럭 효과가
없다. 임베딩 gather는 메모리 바운드라 전력 밀도를 올리지 않기 때문이고, 이는 측정-B의
"컴퓨트 바운드 단계만 클럭 손해를 본다"는 결론과 일관된다.

**trie 구성 자체는 2 ms**로 확정됐다(EXP-011에서 역산했던 "수 ms"의 직접 측정).

## 5. Next action
남은 것은 gemm 1.09(projection)와 moe 0.78뿐이고 둘 다 컴퓨트 바운드다.
- **q/o_proj는 16.2 TF/s / 35.6 peak.** ncu 실측 stall은 barrier 9.4% +
  long_scoreboard 14.8% + short_scoreboard 10.6%. `cp.async`가 막혔으므로 남은 길은
  타일 재설계(register tile 확대 또는 k-major 레이아웃 + mma 스타일 인덱싱)다. 큰 작업.
- 측정-B에 따라 **컴퓨트 밀도를 올리는 최적화의 기대값은 이론치의 97~98%**로 잡는다.

---

# 측정-C: pageable 전송 실태와 pinned memory 상한

`generate()` 안의 호스트<->디바이스 복사는 **전부 pageable**(`std::vector` / `Tensor`).
pinned로 바꿨을 때 얼마나 벌 수 있는지 상한을 재봤다.

## 1. 어디서 얼마나 (b0, elapsed 2.312 s, 프로브 빌드로 구간 분리)

| 복사 | 크기 x 횟수 | 실측 | 비고 |
|---|---|---|---|
| router D2H (`to_host(router)`) | 997 KB x 32 | **0.005 s** | `t_d2h` 전부 |
| expert index+tiles H2D | 125 KB + 3 KB x 32 | **0.001 s** | `t_moe` 안 |
| 셋업 H2D (pos/anc/anc_off/rope/token) | 합 ~1.07 MB x 1 | ~0.001 s | `t_h2d` 안 |
| **logits D2H (`to_host(logits)`)** | **131.3 MB x 1** | **0.015 s** | `t_lm` 안 |

## 2. 대역폭 (같은 노드, 마이크로벤치)

| 크기 | pageable | pinned |
|---|---|---|
| 3 KB | 4.6 us | 5.5 us (**더 느림**) |
| 125 KB | 6.25 GB/s | 10.99 GB/s |
| 997 KB | 6.36 GB/s | 19.87 GB/s |
| 131 MB | 7.95 GB/s | 24.72 GB/s |

## 3. 131 MB logits는 pinned가 답이 아니다
`Tensor`는 `std::vector<float>`이고 `tensor.h`는 수정 금지 -> 할당기를 못 바꾼다. 대안 실측:

| 방법 | 비용 |
|---|---|
| 지금 (pageable D2H) | **15.5 ms** |
| `cudaHostRegister` + D2H + `Unregister` | 13.1 + 5.4 + 5.8 = **24.3 ms** (퇴행) |
| pinned staging + omp memcpy(64T) | 5.6 + 9.2 = **14.8 ms** |
| (참고) `cudaHostAlloc(131 MB)` 자체 | **107 ms** — 측정 구간 안에 두면 불가 |

DMA는 3배 빨라지지만 pinned 버퍼 -> `logits` 호스트 memcpy 9 ms가 이득을 그대로 먹는다.

## 4. 결론
- pinned 단독 상한 = router D2H 3.4 ms + index H2D 0.3 ms + 셋업 0.1 ms
  = **약 3.8 ms / 2312 ms = 0.16%**. 노드 클럭 편차(2~3%)에 묻힌다. **하지 않는다.**
- 진짜 크기의 레버는 pinned가 아니라 그 옆에 있다:
  - **호스트 라우팅 왕복 0.059 s (2.6%)** = router D2H 0.005 + 호스트 top-2/타일 빌드
    `route` **0.054**. 디바이스에서 라우팅하면 복사와 함께 통째로 사라진다.
  - logits D2H 15 ms는 pinned가 아니라 **겹치기**로 지운다. lm_head GEMM이 0.070 s이므로
    행 청크 단위 async D2H(= pinned 필수)를 두 번째 스트림에 태우면 DMA 5.6 ms는 완전히
    숨는다. 단 pinned 버퍼는 생성자에서 잡아야 한다(107 ms).

---

# EXP-014: 출력 Tensor 할당을 GPU와 겹치기

## 1. Hypothesis
측정-C에서 pinned를 검토하다 `t_lm` 0.070 s를 쪼갰더니 GEMM은 0.021 s뿐이고
**`logits = Tensor({1024, 32064})` 호스트 할당+제로필이 0.069 s**였다.

| dlog cudaMalloc | lm_head GEMM | **Tensor 할당** | logits D2H |
|---|---|---|---|
| 0.001 | 0.021 | **0.069** | 0.015 |

131 MB를 `data_.assign(n, 0.0f)`로 페이지 폴트 걸며 채우고(≈1.9 GB/s), 그 제로는
직후 D2H가 전부 덮어쓴다. GPU에 아무것도 의존하지 않는 순수 호스트 작업이므로
32개 레이어 옆에서 별도 스레드로 돌리면 통째로 숨는다.

## 2. Change (`src/model.cu`)
`generate()` 진입부에서 `std::thread`로 `logits = Tensor(...)`를 시작하고 D2H 직전에
join. `cuda_check` 예외 시 `~thread`가 `std::terminate`를 부르지 않도록 `JoinGuard`
RAII 하나 추가. 커널/연산은 한 줄도 안 바뀐다.

## 3. Result (노드 b0, 한 allocation 안에서 교차 2rep)

| | talloc | elapsed |
|---|---|---|
| base | 0.069 / 0.069 | 2.3190 / 2.3169 |
| async | **0.000 / 0.000** | **2.2484 / 2.2530** |

`gemm` 1.085/1.087 -> 1.087/1.086, `moe` 0.725 -> 0.724, `route` 0.054 그대로.
**-0.068 s (-2.9%)**, 다른 구간 퇴행 없음. 출력 `exp013.bin`과 bitwise 동일,
`-v` PASSED, 455.0 seq/s.

## 4. Analysis
69 ms 중 69 ms가 숨었다. 64코어 노드라 페이지 폴트 스레드가 메인 스레드의
호스트 라우팅(0.054 s)과 커널 런치를 밀어내지 않는다.

## 5. 남은 logits D2H 15 ms: 설계 2개 다 재보고 둘 다 기각
"memcpy가 이득을 먹는다"에 대한 답은 "없앤다"가 아니라 "GPU 뒤로 숨긴다"인데,
숨길 값이 작다.

| 설계 | 방법 | 실이득 |
|---|---|---|
| A. register 겹치기 | 헬퍼 스레드에서 `cudaHostRegister`(13 ms, 숨김) -> D2H를 `logits`에 직행 5.6 ms, **memcpy 0** | 15 - 5.6 - **5.8**(unregister는 측정 구간 안) = **~3.6 ms** |
| B. 청크 파이프라인 | lm_head를 행 청크로 쪼개 non-blocking 스트림 async D2H + pinned ring, memcpy를 다음 청크 GEMM 뒤로 숨김 | 마이크로벤치 46.3 -> 39.4 ms (33.5 ms 커널 기준). 실 GEMM 21 ms 환산 **~10~12 ms** |

B가 원리적으로 맞지만 `cudaHostAlloc`이 **0.83 ms/MB**(16 MB에 13.6 ms, 이것도 숨겨야
함) + 청크 GEMM + 이벤트 + 비블로킹 스트림 + memcpy 스레드. 0.5%에 과하다.
근본 제약은 `Tensor`가 `std::vector<float>`이고 `tensor.h`가 수정 금지라 할당기를
못 바꾸는 것. **plan의 "하지 않는 것"으로 이동.**

## 6. Next action
**호스트 라우팅 왕복 0.059 s**(router D2H 0.005 + 호스트 top-2/타일 빌드 0.054)가
남은 최대 비-GEMM 항목 -> EXP-015.

---

# EXP-015: 라우팅을 디바이스로

## 1. Hypothesis
측정-C: MoE 라우팅이 레이어마다 호스트를 왕복한다. router 점수 997 KB D2H(0.005 s)
+ 호스트 top-2/타일 빌드(**0.054 s**) + index/tiles H2D(0.001 s) = **0.060 s (2.6%)**.
전부 디바이스로 옮기면 사라진다.

## 2. Change (`src/model.cu`)
`route_row`(호스트)를 `route_top2`(`__device__`)로 옮기고 counting sort 3-pass:

- `route_count` — 행당 top-2를 **낮은 expert 인덱스 먼저**로 저장(기존 per-expert
  scatter가 오름차순으로 더했으므로), 블록별 expert 카운트 집계
- `route_scan` — 1블록 16스레드. 블록 오프셋 + expert 오프셋 스캔, 타일 리스트 생성
- `route_place` — 각 행의 두 사본을 expert 구간에 **행 오름차순**으로 배치.
  호스트 빌드와 **같은 packed 레이아웃**이 나온다. 각 사본의 slot을 기록

**타일 개수를 호스트로 되돌리지 않으려고** grid를 최악치 `packed_rows/BM + 16`(=259,
실제 ≈256)로 고정하고 `gemm_grouped`에 `if (t[2] == 0) return;`을 넣었다.
낭비 타일 3개.

동반 변경(불가피): 호스트가 expert별 행 수를 모르므로 per-expert `scatter_add_rows`
16 launch를 행 단위 `moe_combine` 1 launch로 융합. `acc = 0; acc += 0.5f*lo;
acc += 0.5f*hi;` — memset + 두 번의 `+=`와 문자 그대로 같은 식이라 라운딩 동일.
덕분에 레이어당 255 MB(총 8.2 GB) `cudaMemset`도 사라진다.

## 3. Result (노드 b0, 한 allocation 안에서 교차 2rep)

| | d2h | moe | gemm | elapsed |
|---|---|---|---|---|
| 014 | 0.005 / 0.005 | 0.779 / 0.780 | 1.082 / 1.085 | 2.2452 / 2.2450 |
| 015 | **0.002 / 0.002** | **0.696 / 0.700** | 1.099 / 1.093 | **2.1759 / 2.1728** |

**-0.071 s (-3.2%)**. 출력 `exp014.bin`과 bitwise 동일, `-v` PASSED, 472.2 seq/s.

## 4. Analysis
moe -0.082(호스트 라우팅 0.054 + memset 0.010 + idxcopy 0.001 + 나머지),
d2h -0.003. 예측 0.060~0.070과 일치.

`gemm` **+0.012 퇴행**은 측정-B의 boost 클럭 현상 재현이다. 호스트 대기가 사라져
GPU가 더 촘촘히 돌면 fast 노드에서 SM 클럭이 내려간다. 코드로 고칠 수 있는 항목이
아니고, 그래도 순이득이 -0.071이다.

## 5. Next action
호스트는 이제 측정 구간 안에서 커널 런치 외에 하는 일이 없다. 남은 것은 전부 GPU:
**gemm 1.09 (50%)** > moe 0.70 > attn 0.144 > norm 0.113. GEMM 타일 재설계.

---

# 측정-E: 트리플 버퍼링(H2D / kernel / D2H 3-stream) 여지 판단

코드 변경 없이 `nsys profile -t cuda ./main -n 1024` 1회 (EXP-015 코드, 노드 b0,
elapsed 2.1612 s / 473.8 seq/s). 측정 구간을 SQLite로 완전 분해했다.

## 1. 측정 구간(generate) 전체 분해 — 합이 elapsed와 일치

| 항목 | 시간 | 비율 | GPU |
|---|---|---|---|
| 호스트 prologue (trie 2.0 + cudaMalloc 2.5 + H2D 0.15) | 4.75 ms | 0.22% | idle |
| **커널 실행 (612 launch)** | **2119.52 ms** | **98.07%** | busy |
| 커널 사이 launch gap 611개 합 | 1.70 ms | 0.08% | idle |
| **logits D2H (131.3 MB, pageable)** | **14.31 ms** | **0.66%** | idle |
| **에필로그 `cudaFree` 24회 (~1.5 GB)** | **20.77 ms** | **0.96%** | idle |
| 합 | 2161.05 | | |

## 2. 측정 구간 안의 버스 트래픽 전량 = 7건, 14.37 ms

| 방향 | 대상 | 크기 | DMA 시간 |
|---|---|---|---|
| H2D | d_pos / d_anc / d_anc_off / d_rope / node_token | 1041 KB | 62.3 µs |
| H2D | last_node (lm_head 직전) | 4 KB | 0.7 µs |
| D2H | logits | 131.3 MB | **14 309.9 µs** |

가중치 15.0 GB / 1.212 s H2D는 **전부 생성자**(측정 구간 밖) 2027건이다.
측정 구간의 H2D는 **0.063 ms = 0.003%**.

## 3. 판단: 트리플 버퍼링은 여지가 없다

1. **파이프라인에 태울 작업 흐름 자체가 없다.** 트리플 버퍼링은 같은 단위가
   H2D→kernel→D2H를 반복 통과할 때 값을 한다. 여기서 H2D는 구간 진입 시 1 MB
   한 번(63 µs)이고 그 1 MB(trie 메타)는 **32개 레이어 전부가 읽으므로 청크로
   쪼갤 수 없다**. D2H는 32개 레이어가 다 끝나야 존재하는 값이라 마지막 1회다.
   중간에 버스를 건너는 것은 아무것도 없다(EXP-013/015가 이미 다 없앴다).
2. **커널 스트림도 채울 틈이 없다.** 커널 윈도우 2121.22 ms 중 idle 1.70 ms
   (0.08%). launch gap은 큐로 이미 완전히 가려져 있어 두 번째 compute 스트림이
   메울 공백이 없다.
3. **겹치기 대상은 꼬리 D2H 하나뿐이고 그 상한이 14.31 ms = 0.66%다.**
   D2H가 공짜가 되어도 473.8 → 477.0 seq/s. 노드 클럭 편차(b0/b5 1.196x)에 묻힌다.
   구현 실이득은 EXP-014 §5에서 이미 실측: 설계 A ~3.6 ms, 설계 B ~10~12 ms.
   D2H는 마지막 커널이 끝난 **뒤에** 시작하므로(gap 0.005 ms) 숨길 상대는
   lm_head GEMM 21 ms뿐이고, pinned→`logits` 호스트 memcpy 9.2 ms까지 같은
   21 ms 안에 겹쳐야 한다. **기각 유지.**
   - D2H 볼륨 축소도 불가: `main.cpp`가 131 MB 전량을 `outputs.bin`에 쓰고
     `answers.bin`과 비교한다.

## 4. 대신 나온 것: 에필로그 `cudaFree` 20.77 ms (0.96%)

D2H 전체보다 **1.45배 크다.** `generate()`의 `DeviceBuffer` 13개 + `d_logits`가
스코프를 벗어나며 24회 `cudaFree`(~1.5 GB)를 측정 구간 안에서 동기 호출한다.
GPU는 그 20.8 ms 동안 아무것도 안 한다. 스트림 문제가 아니라 **할당 수명** 문제다.

- 고칠 방법: 스크래치 버퍼를 모델 멤버로 올려 `~PhiTinyMoEModel`에서 해제.
  `cudaMalloc`(2.5 ms)은 구간 안에 그대로 두므로 **워밍업 캐싱 논점이 없다** —
  free만 구간 밖으로 나간다.
- 규칙 확인: 연산 결과 캐싱이 아니라 메모리 해제 시점 이동이다. 산술 무변경 →
  bitwise 동일이 판정 기준.
- 상한 20.77 ms(0.96%)는 트리플 버퍼링 최선(설계 B ~10~12 ms)의 약 2배이고
  구현이 훨씬 단순하다. **plan에 EXP로 올린다.**

---

# 측정-F: 행렬곱 커널의 더블/트리플 버퍼링 여지

## 1. 현황
`gemm_nt_body`는 **이미 2단 더블 버퍼링**이다 — shared 2버퍼 + 다음 타일을 레지스터에
선행 로드. 모델의 모든 행렬곱이 이 한 함수를 지난다: `gemm_nt_bias`(q/k/v/o_proj·gate·
lm_head)와 `gemm_grouped`(w13·w2). 합이 gemm 1.086 + moe 0.690 = **elapsed의 82%**.
(`attention_heads`는 shared 16 B로 K/V 스테이징 자체가 없다. 깊이 문제가 아니라
staging/coalescing 문제이므로 EXP-017 소관.)

## 2. 방법
`model.cu` 사본 5개를 빌드(obj는 공유), **한 srun 안에서 교차 실행** ×2, n=1024.

## 3. 결과 — 깊이는 레버가 아니다

노드 b5(fast), gemm/moe/elapsed (rep1/rep2):

| 변형 | 레지스터 | smem | gemm | moe | elapsed |
|---|---|---|---|---|---|
| **db = 현재 2단** | 127 | 16.9 KB | 1.086/1.083 | 0.690/0.690 | **2.155/2.160** |
| sb = 버퍼링 제거 | 107/113 | 8.4 KB | 1.127/1.119 | 0.698/0.699 | 2.208/2.193 |
| tb2 = cp.async 2단 | 127 | 16.9 KB | 1.478/1.473 | 0.878/0.880 | 2.762/2.722 |
| tb = cp.async 3단 | 127 | 25.3 KB | 1.468/1.469 | 0.886/0.886 | 2.726/2.725 |
| tb4 = cp.async 4단 | 127 | 33.8 KB | 1.498/1.503 | 0.904/0.904 | 2.779/2.783 |

레지스터 선행 깊이를 2타일로 늘린 rb3(별도 세션 b7, db와 3회 교차):

| 변형 | 레지스터 | gemm | elapsed |
|---|---|---|---|
| db | 127 | 1.330/1.333/1.333 | 2.594/2.592/2.593 |
| rb3 = 선행 깊이 2 | 126, 스필 0 | 1.379/1.379/1.380 | 2.640/2.640/2.640 |

1. **레버 전체 크기가 0.045 s(2.1%)이고 이미 다 먹었다.** 버퍼링을 통째로 제거해도
   gemm +0.041 / moe +0.008뿐이다. 지연 은닉으로 벌 수 있는 총량의 상한이 그 값이다.
2. **더 깊게 가면 손해.** rb3는 **+0.047 s**. 레지스터 126(스필 0, 점유율 2 blocks/SM
   유지)이라 **점유율 문제가 아니다** — 선행 로드 1세트를 더 살려두는 명령어·스케줄
   비용이 은닉 이득보다 크다.
3. **cp.async 경로는 레이아웃이 막고 있다.** shared가 k-major 전치라 한 스레드의 4 float이
   stride(BM+SPAD)로 흩어져 **4 B 복사 8개**가 `LDG.128` 2개 + `STS` 8개를 대체한다:
   +0.34~0.39 s. 그 안에서도 3단이 2단보다 0.7%, 4단은 오히려 나쁘다 → **깊이 자체가
   레버가 아니라는 것이 cp.async 계열 내부에서도 재확인**된다.
4. tb(3단) 출력은 db와 **bitwise 동일**(`-n 128 -v`, max abs 0.000179291 양쪽 일치).
   느린 이유가 버그가 아님을 확인했다.

## 4. ncu — 왜 여지가 없나

`per_issue_active` stall(워프 수), b5:

| 커널 | grid | not_selected | long_sb | mio_thr | short_sb | barrier | FMA % | thru % | warps % |
|---|---|---|---|---|---|---|---|---|---|
| q_proj | (16,122) | **1.61** | 0.81 | 0.76 | 0.74 | 0.56 | 57.9 | 63.4 | 32.8 |
| o_proj | (32,122) | **1.58** | 0.91 | 0.78 | 0.76 | 0.57 | 56.9 | 62.4 | 33.0 |
| w13 | (7,259) | **1.65** | 1.06 | 0.89 | 0.56 | 0.63 | 55.6 | 60.5 | 32.9 |
| w2 | (32,259) | **1.55** | 1.31 | 1.08 | 0.59 | 0.80 | 47.2 | 52.8 | 32.9 |

**최대 stall이 `not_selected`다** — 워프는 발행 준비가 끝났는데 스케줄러가 못 뽑는다.
글로벌 지연(`long_scoreboard`)은 stall 합의 17~25%에 불과하고, 그마저 버퍼를 깊게 해도
줄지 않는다는 것이 §3의 실측이다. 지연 은닉이 아니라 **발행 슬롯**이 한계다.
FMA 파이프는 peak의 47~58%, 점유율은 33%(127 레지스터 → 2 blocks/SM 고정).

## 5. Next action
**버퍼링 깊이는 소진. 새 실험으로 올리지 않는다.**
남은 길은 EXP-016 타일 재설계 — FMA당 shared 로드/발행 명령 수를 줄이는 방향
(레지스터 타일 확대 또는 BK 확대)이다. 근거는 위 표: `not_selected + short_sb +
mio_throttle` 합이 `long_scoreboard`의 약 3배다.

---

# 측정-D: memory coalescing에 남은 여유 (EXP-013 코드에서 측정, `961c1e3`)

EXP-008의 "feature-major 불필요" 판정은 expert GEMM만 본 것이라, trie(011)·
grouping(012)·device embedding(013)으로 접근 패턴이 바뀐 뒤 전 커널을 다시 쟀다.
브랜치 `exp013-coalescing`에서 잰 것을 EXP-016과 함께 main으로 옮긴다
(main의 `측정-C`는 pinned 판단이라 문자 충돌을 피해 개명).

## 1. ncu 전수 (n=128, b6, `--clock-control none --cache-control none`)

| 커널 | ld bytes/sector | dram% | l1% |
|---|---|---|---|
| **attention_heads** | **22.60** | 9.75 | **87.57** |
| rope_rows | 66.22 | 86.42 | 18.87 |
| gemm_nt_bias | 95.02 | 9.85 | 55.18 |
| gemm_grouped | 99.89 | 24.62 | 74.00 |
| gather_rows / scatter_add_rows | 98.65 / 99.32 | 88.81 / 75.63 | 19.97 / 14.25 |
| layer_norm_rows / silu_mul / add_inplace | 100.00 | 28.7 / 85.6 / 90.8 | 29.6 / 15.0 / 14.3 |

- **여유가 있는 커널은 `attention_heads` 하나뿐이다.** 원인은 q·k 내적의
  `for (i = tid; i < nk; i += D)` — 워프 32레인이 서로 다른 key 행을 쳐서 32 B 섹터마다
  4 B만 쓴다(12.5%). value 누적은 100%. 섹터 가중 예상 22.2% vs 실측 22.60%로 일치.
- 그리고 이 커널은 **L1 87.6% 포화**(dram 9.75%)다. 낭비하는 자원이 곧 병목 자원이다.
- rope_rows의 66.22%는 8 KB cos/sin 테이블 stride-2 접근인데 dram 86.42%로 대역폭
  바운드라 섹터를 고쳐도 DRAM 트래픽이 안 줄어든다.

## 2. 천장과 설계 3안 (n=1024, b0, 한 allocation 안에서 교차 2rep)

| 빌드 | attn | elapsed |
|---|---|---|
| main (EXP-013) | 0.143 | 2.300 / 2.302 |
| **D1 진단 하한** (score dot의 global k 로드 삭제, 오답) | **0.069** | 2.231 / 2.229 |
| D2 key 행 통째 스테이징 | 0.154 | 2.307 / 2.316 |
| **D3 청크 스테이징** | **0.096** | **2.260 / 2.260** |

- **천장은 0.074 s.** k 로드를 공짜로 만들어도 그 이상은 없다.
- **D2는 반증됐다.** 행 통째(32 × 129 float = 16.5 KB/block)면 occupancy가 12 → 5
  blocks/SM으로 떨어져 코얼레싱 이득을 통째로 삼킨다. **여유가 있다는 것과 먹을 수 있다는
  것은 다르다.**
- **D3가 답이다.** d를 32열씩 끊으면 shared 4.2 KB로 내려가 occupancy가 유지된다.

## 3. 결론

coalescing에 남은 이론적 여유는 **1.031×**, 실현 가능한 것은 **1.018×**이고 전부
`attention_heads` 한 곳에 있다. gemm 1.09 + moe 0.78(전체의 81%)은 bytes/sector
95.0 / 99.9라 coalescing 문제가 아니다. → D3를 EXP-016으로 채택.

---

# EXP-016: attention 청크 스테이징

## 1. Background
측정-D. `attention_heads`의 q·k 내적만 fetch 바이트의 22.6%를 쓰고, 그 커널은 L1
87.6%로 포화다. 천장 0.074 s, 설계는 D3(청크 스테이징)로 확정됐다. 다만 측정-D는
EXP-013 코드에서 잰 값이고, main은 EXP-015로 호스트 대기가 사라져 boost 클럭이 내려간
상태라 재측정이 필요했다.

## 2. Hypothesis
key 행을 32열씩 shared로 스테이징하면 global 읽기가 행 연속이 되어 attn이 줄고,
shared는 `max_len*33` float(4.9 KB)이라 occupancy는 유지된다. 값도 순서도 안 바뀌므로
출력은 bitwise 동일해야 한다.

## 3. Change (`src/model.cu`)
`attention_heads`의 내적 블록만 교체. `sk = sw + nk`(할당은 `max_len*(CH+1)` 예약),
`CH = ATTN_CHUNK = 32`, `+1` 패딩으로 뱅크 회피(stride 33 → 레인 i는 뱅크 `(i+d)%32`).
스테이징 루프는 128스레드가 `tid/CH`행·`tid%CH`열로 4행×32열씩 채운다.

청크 사이 부분합은 `sw[i]`(shared fp32)에 둔다 — 레지스터에 있었을 때와 같은 fp32라
score마다 `__fadd_rn` 열이 그대로다. 원본의 `nk <= blockDim` 가정은 넣지 않았다:
compute도 `for (i = tid; i < nk; i += D)` 스트라이드라 nk가 커져도 성립한다
(nk ≤ 128인 지금은 동작이 동일). 레지스터 43 → 40.

## 4. Result (노드 b0, 한 allocation 안에서 교차 3rep)

| 빌드 | attn | gemm | moe | elapsed |
|---|---|---|---|---|
| base | 0.144 / 0.144 / 0.144 | 1.099 / 1.093 / 1.094 | 0.697 | 2.176 / 2.174 / 2.171 |
| **D3** | **0.100 / 0.100 / 0.100** | 1.102 / 1.098 / 1.097 | 0.704 / 0.705 / 0.707 | **2.142 / 2.139 / 2.147** |

- **attn −0.044 s** (천장 0.074의 59%), **elapsed −0.031 s = 1.0146×**,
  throughput 471.1 → 477.9.
- 출력 `cmp` 결과 **bitwise 동일**(n=1024, max abs diff 0.000179291, PASSED).
- attn 이득 0.044 중 **0.012가 gemm(+0.004)·moe(+0.008)로 되돌아왔다.** 측정-B의
  "컴퓨트 밀도를 올리면 빠른 노드에서 이론치의 97~98%"와 같은 현상이고, EXP-013 기준
  측정치(−0.041)보다 순이득이 0.010 작아진 이유다. 방향과 bitwise 동일성은 그대로.

## 5. Next action
coalescing 레버는 이걸로 소진(D1 하한까지 남은 0.030 s는 shared 자체 비용이라 접근
불가). 다음은 017 — `bs` 뱅크 충돌 제거, 대상이 gemm 1.10 + moe 0.70 = 82%다.
착수 전에 ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`으로
런타임 충돌을 먼저 확인한다(plan §A).

---

# EXP-017: GEMM `bs` 뱅크 충돌 제거

## 1. Background
plan §로그 대조 A. K 루프의 `&bc[p][tx * TN + j]`는 워프 quarter(8레인)에 워드
stride 8을 준다. `SPAD=4`는 이걸 못 고친다 — 베이스가 `p*132`라 뱅크가 p마다 4씩
회전할 뿐 stride-8 패턴은 그대로다(로그 기각표: 패딩만 늘리기 +1.2% 느림).

## 2. Hypothesis (착수 전 ncu로 확인)
`gemm_nt_bias`, n=128:

| 지표 | 값 | 명령당 |
|---|---|---|
| `inst_executed_op_shared_ld` | 35,651,584 | 1.0 |
| `wavefronts_mem_shared_op_ld` | 178,258,698 | **5.00** |
| `bank_conflicts_..._op_ld` | 71,303,168 | **2.00** |
| `bank_conflicts_..._op_st` | 0 | 0 |

명령 수가 `272 block × 512 ktile × 8p × 4 LDS × 8 warp`와 정확히 일치해 계수 해석이
확정된다. 분해: `as` 2 + 2(브로드캐스트, ty가 quarter 안에서 상수) + `bs` 8 + 8(2중
충돌) = 5.0, 초과분 (8−4)×2 = 8 → 명령당 2.0. 충돌 카운터와 정확히 일치한다.
**정적 분석이 런타임으로 확인됐다: 범인은 `bs` 읽기 하나.**
`gemm_grouped`도 wavefront/conflict = 183.5M/73.4M = 2.5로 동일 비율(같은 device 함수).

## 3. Change (`src/model.cu`)
```cuda
__device__ __forceinline__ int gemm_col_slot(int tx, int j) {
    return tx * 4 + (j >> 2) * (BN / 2) + (j & 3);
}
```
`bs` 읽기와 에필로그 `col` 둘 다 이걸로 간다. 레인이 갖는 8열을 4열씩 반 타일
떨어뜨리면 quarter의 8레인이 32뱅크를 정확히 한 번씩 덮는다. `p*132`가 4의 배수라
`&bc[p][tx*4]`는 여전히 16 B 정렬이다. `as`는 손대지 않았다(충돌 없음).
**스레드↔출력 열의 대응만 바뀐다** — 각 출력은 여전히 같은 k 오름차순 FMA 열을 본다.
레지스터 127, smem 16,896 B 모두 불변.

## 4. Result (노드 b5, 한 allocation 안에서 교차 3rep)

| 빌드 | gemm | moe | attn | elapsed |
|---|---|---|---|---|
| base | 1.087 / 1.085 / 1.087 | 0.698 / 0.698 / 0.696 | 0.099 | 2.120 / 2.122 / 2.116 |
| **bs** | **0.981 / 0.983 / 0.983** | **0.616 / 0.610 / 0.617** | 0.101 | **1.936 / 1.937 / 1.943** |

- **gemm −0.104 s, moe −0.083 s, elapsed −0.181 s = 1.0934×.** 483.2 → 528.1 seq/s.
- 출력 `cmp` **bitwise 동일**(n=1024, PASSED).
- 적용 후 ncu 재측정: 충돌 **2.00 → 0**, wavefront **5.00 → 3.00**/명령
  (178.3M → 107.0M). 예측(`as` 2+2 + `bs` 4+4 = 3.0)과 정확히 일치 —
  **이득이 뱅크 충돌에서 온 것이 맞다.**
- 지금까지 단일 실험 최대 레버다. 측정-F에서 최대 stall이 `not_selected`(발행 슬롯
  포화)로 나온 것과 정합적이다: 충돌은 워프를 지연시키는 게 아니라 LSU 발행을
  2배로 먹고 있었다.

## 5. Next action
- **018 bias 호이스팅**은 이제 새 매핑(`gemm_col_slot`)을 따라 넣는다.
- **GEMM 타일 재설계 재판단**의 전제가 성립했다. 로그의 균형식대로라면 이제
  공유 대역폭과 FFMA가 1:1이므로, K 루프에 더 넣을 여지가 생겼는지 다시 봐야 한다.

---

# EXP-018 — bias 에필로그 호이스팅 (기각)

## 1. Background
SASS 실측으로 미적용이 확정돼 있었다: `gemm_nt_bias` LDG **68** vs `gemm_grouped`
**8**. 차 60은 스레드당 `bias[col]` 64회다. `col`은 `j`에만 의존해 행 루프 불변인데
`if (row >= m) continue;` 때문에 ptxas가 못 뺀다. 대조 로그가 같은 형태에서
행 루프 밖으로 빼 **+1.53%**를 기록했다.

## 2. Hypothesis
`float biasv[TN]` 8개를 행 루프 앞에서 한 번 읽으면 LDG 64회가 사라져 gemm이 줄어든다.
산술 무변경(같은 피가산수, 같은 순서)이라 bitwise 동일해야 한다.

## 3. Design and Implementation
017의 새 매핑을 따라 `col = n0 + gemm_col_slot(tx, j)`로 8개를 선행 로드.
두 형태를 만들었다.

- **hoist**: `if (bias != nullptr)`를 호이스팅 블록에만 씌우고 저장 루프의
  `if (bias != nullptr) v += biasv[j];`는 그대로.
- **hoist2**: null 검사를 `biasv[j] = (bias && col < n) ? bias[col] : 0.0f`로 접고
  저장 루프를 `acc[i][j] + biasv[j]` 단일 경로로. (`acc`는 +0.0f에서 시작해 FFMA로만
  자라므로 −0.0f가 될 수 없고, `+0.0f`는 항등이다.)

`gemm_nt_bias` 정적 SASS (레지스터 127 · smem 16,896 B 세 빌드 모두 불변):

| 빌드 | 총 명령 | LDG | STG |
|---|---|---|---|
| base | 1968 | 68 | 128 (bias/non-bias 이중 에필로그) |
| hoist | 1896 | 12 | 128 |
| hoist2 | **1136** | **12** | **64** |

## 4. Result

**A/B 1 (b5, 한 allocation 교차 3rep) — base vs hoist**

| 빌드 | gemm | elapsed | seq/s |
|---|---|---|---|
| base | 0.978 / 0.984 / 0.984 | 1.925 / 1.932 / 1.930 | 530.8 |
| hoist | 0.997 / 0.999 / 0.997 | 1.950 / 1.947 / 1.949 | 525.5 |

**gemm +0.016, elapsed +0.019 s로 오히려 퇴행**, 3rep 겹침 없음. ncu로 원인 확인:
정적 명령은 72개 줄었는데 `smsp__inst_executed`는 638.9M → 642.2M(**+0.51%**)로 늘었다.
저장 루프에 남은 per-element `if (bias != nullptr)`가 이중 에필로그(STG 128)를
그대로 유지시키고 주소 계산만 늘린 형태(IADD3 +7, ISETP +9).

**A/B 2 (b7, 교차 3rep) — base vs hoist2**

| 빌드 | gemm | moe | elapsed |
|---|---|---|---|
| base | 0.985 / 0.986 / 0.991 | 0.617 / 0.622 / 0.613 | 1.940 / 1.947 / 1.946 |
| hoist2 | 0.989 / 0.987 / 0.988 | 0.616 / 0.614 / 0.616 | 1.945 / 1.940 / 1.944 |

**차이 없음**(gemm 0.988 vs 0.987, elapsed 1.943 vs 1.944 — 노이즈 안).
퇴행은 없앴지만 이득도 없다.

**결론: 기각. `bias[col]` 64회는 비용이 0이다.** bias는 최대 128 KB(lm_head)이고
타일 하나가 읽는 건 128 float뿐이라 전량 L1 상주 + 블록 내 전 스레드 재사용이다.
그리고 에필로그는 K 타일 512회 루프 뒤에 **출력당 1회**만 돈다 — LDG 64회를 12회로
줄여도 커널 시간에 나타나지 않는다. 대조 로그의 +1.53%는 우리와 K 반복 수가
다른 형상에서 나온 값으로 봐야 한다.

부수 확인: hoist2는 `gemm_grouped`(bias=nullptr)에 `+0.0f` FADD 64개를 새로 넣는다.
컴파일러가 IEEE상 `x+0.0f`를 접을 수 없기 때문이다. moe 시간에는 나타나지 않았다.

`src/model.cu`는 **변경 없음**. 두 변형은 `$CLAUDE_JOB_DIR/tmp/exp018/`에만 있다.

## 5. Next action
- 에필로그 최적화 계열은 닫는다. **019 gate tall-skinny도 같은 이유로 상한을
  낮게 잡아야 한다** — 그쪽 이득은 에필로그가 아니라 낭비되는 타일 87.5%에서 오므로
  별개지만, "GEMM 주변부를 다듬어 얻는다"는 기대는 이 실험이 한 번 반증했다.
- 남은 gemm 0.98의 레버는 K 루프 안에 있다. **GEMM 타일 재설계 재판단**이 다음 순번.
# 측정-G: 전송 최적화 + kernel fusion **결합** 시 여지 판단

측정-E는 트리플 버퍼링(전송만)을 기각했다. 여기서는 **전송 최적화와 커널 재구성을
같이 썼을 때** 측정-E가 못 본 여지가 생기는지를 실측으로 확인한다. 결합의 유일한
접점은 꼬리다: `lm_head` GEMM을 행 청크로 쪼개면 각 청크의 D2H를 다음 청크의 GEMM
뒤에 숨길 수 있다. 청크 분할은 베이스 포인터와 행 수만 바뀌므로 산술 무변경이다.

## 1. 측정 구간 안의 버스 트래픽 = 상한

`APS_XFER=-1`(꼬리 분할 계측), 노드 b7, 2회:

| 구간 | 시간 |
|---|---|
| `lm_head` GEMM | 24.91 / 20.32 ms |
| logits D2H (131.3 MB, pageable) | 16.48 / 16.81 ms |
| H2D 전량 (측정-E) | 0.063 ms |

**결합이 건드릴 수 있는 전부가 16.6 ms = 0.64%다.** 측정-E의 0.66%를 독립 재확인.

## 2. PCIe 실측 (`docs/microworld/meas-g-xfer-bw-bench.cu`, b7)

| 항목 | 시간 |
|---|---|
| pageable D2H 131.3 MB | 17.30 ms (7.6 GB/s) |
| **pinned D2H 131.3 MB** | **5.07 ms (25.9 GB/s)** |
| `cudaHostRegister` (GPU idle) | 17.82 ms |
| `cudaHostUnregister` | 6.21 ms |
| `cudaHostAlloc` | 104.38 ms |
| staging(pinned) D2H + host memcpy | 5.40 + 9.52 = **14.92 ms** |

pinned가 pageable보다 **12.2 ms 빠르다.** 여지는 여기 있다. 문제는 값을 치르는 방법이다.

## 3. 구현 A: `cudaHostRegister` + 청크 — **실측 음수**

`logits` Tensor를 alloc 스레드에서 pin(페이지폴트 비용을 이미 치르는 그 스레드)하고,
`lm_head`를 C개 청크로 쪼개 청크마다 copy stream에 async D2H. 노드 b1, 한 allocation
안에서 교차 실행:

| 변형 | elapsed | Δ |
|---|---|---|
| base ×3 | 2.5795 / 2.5790 / 2.5812 | — |
| `APS_XFER=1` | 2.6048 | **+25.0 ms** |
| `APS_XFER=4` ×2 | 2.6070 / 2.6070 | **+27.1 ms** |
| `APS_XFER=8` | 2.6095 | **+29.6 ms** |

출력은 4종 전부 base와 `cmp` **bitwise 동일**(청크 분할이 산술 무변경임을 확인).

내역(`[xfer]`, ms):

| 청크 | register(측면 스레드) | join 대기 | drain | unregister |
|---|---|---|---|---|
| 1 | 2465.97 | 21.94 | 26.35 | 10.81 |
| 4 | 2466.83 | 21.65 | 28.11 | 10.35 |
| 8 | 2467.54 | 21.54 | 29.31 | 10.43 |

**핵심: `cudaHostRegister`가 GPU idle 17.8 ms → 커널 32레이어가 in-flight일 때 2466 ms.
139배.** 드라이버가 바쁜 컨텍스트에 대해 page-lock을 직렬화한다. 즉 **pin 비용은 레이어
뒤에 숨길 수 없고, 숨기려는 시도 자체가 비용이다** — 2.5 s를 줘도 못 끝내서 꼬리에서
21.5~62.7 ms를 더 기다린다(b7 `APS_XFER=2`에서는 join 62.7 ms, elapsed +19.7 ms).

`drain`은 일관적이다: 26.35 ≈ lm_head 21 + pinned D2H 5.1 (§2와 일치).
청크를 늘리면 **drain이 는다**(26.35 → 28.11 → 29.31). 525 MB 가중치를 launch마다
다시 스트리밍하기 때문. **청크 페널티 실측: 4청크 +1.8 ms, 8청크 +3.0 ms.**

## 4. 구현 B의 천장: 구간 밖 pinned staging + 청크 + memcpy 스레드

가드레일상 `tensor.h`를 못 고쳐 `logits`를 pinned로 **할당**할 수 없다. 남는 유일한 설계는
생성자에서 pinned staging을 잡아두고(측정 구간 밖, 104 ms) 청크마다
D2H→staging, staging→Tensor memcpy를 워커 스레드로 넘겨 다음 청크 DMA와 겹치는 것.
`docs/microworld/meas-g-tail-overlap-bench.cu` (GEMM은 20.3 ms spin 커널로 대역, 노드 b5):

| 설계 | 꼬리 시간 |
|---|---|
| baseline (GEMM + pageable D2H) ×6 | 34.09 ~ 35.78 (평균 **34.8**) |
| 결합, 1청크 | 34.31 (**이득 0** — pinned 이득 12.2를 host memcpy 9.5가 먹는다) |
| 결합, 2청크 | 28.87 |
| 결합, 4청크 | 26.34 / 26.32 |
| 결합, 8청크 | 25.14 / 25.12 |

**천장 = 34.8 − 25.13 = 9.7 ms.** 여기서 §3의 실측 청크 페널티를 뺀다:

- 4청크: (34.8 − 26.33) − 1.8 = **6.7 ms**
- 8청크: (34.8 − 25.13) − 3.0 = **6.7 ms**

두 경로가 같은 값으로 수렴한다. **순이득 ≈ 6.7 ms.**
측정 당시 기준(EXP-015, b1 2.58 s)으로 0.26%, EXP-017 기준(b5 1.939 s)으로 **0.35%**다.
D2H 131 MB는 PCIe에 묶여 있어 절대값(6.7 ms)은 코드가 빨라져도 그대로이므로,
총시간이 줄수록 비율만 커진다 — 그래도 아래 5번의 순위는 바뀌지 않는다.

1청크가 이득 0인 것이 이 실험의 요점이다 — **pinned 전환만으로는 아무것도 못 얻고
(host memcpy가 정확히 상쇄), 이득은 전적으로 청크 분할이 만든 겹침에서 나온다.**
그 겹침을 만드는 커널 재구성 자체가 1.8~3.0 ms를 도로 물어낸다.

## 5. 판단: 여지는 있으나 0.35%(EXP-017 기준), 남은 레버 중 최소

1. **상한이 구조적으로 0.64%다.** 측정 구간의 버스 트래픽 전량이 16.6 ms이고
   H2D는 1 MB/0.063 ms라 융합할 것도 겹칠 것도 없다. 커널 fusion은 전송량을
   줄이지 못한다 — 꼬리 D2H가 숨을 GEMM을 만들어줄 뿐이다.
2. **자연스러운 구현(A)은 실측 음수(−25~−30 ms)다.** `cudaHostRegister`의
   139배 팽창이 원인이고 이건 우회로가 없다. `logits`가 `std::vector`이고
   `tensor.h`가 NEVER-EDIT이라 pinned로 **할당**하는 길이 막혀 있다.
3. **유일한 합법 설계(B)의 순이득 6.7 ms.** 대가는 상주 pinned staging 131 MB +
   copy stream + 이벤트 N개 + memcpy 워커 스레드 + 청크 lm_head다.
4. **비교**: EXP-020(`cudaFree` 에필로그, 측정-E) 상한 **20.77 ms**가 3배 크고
   구현은 버퍼 수명 이동뿐이다. EXP-017 기준 1.939 s에서 6.7 ms = 0.35% 대
   20.77 ms = **1.07%**. 이 비(3.1배)는 둘 다 절대 ms라 기준선과 무관하다.
5. **노이즈 이하다.** b7 동일 바이너리 baseline이 2.589~2.632(43 ms) 흔들리고
   노드 편차는 1.196x(≈420 ms)다. 6.7 ms는 단일 노드 재현에도 묻힌다.

**기각.** plan의 「하지 않는 것」에 올린다. 청크 lm_head가 bitwise 동일하다는 사실은
확인해 뒀으므로(§3), 훗날 `tensor.h` 제약이 풀리면 구현 A가 아니라 **B에서** 재개한다.

## 6. Next action
변경 없음. 016~018이 이미 들어갔으므로 꼬리에서 다음으로 값하는 것은 전송이 아니라
**EXP-020**(`cudaFree` 20.77 ms / 1.07%)이다.

---

# EXP-019 — GEMM 타일 재설계 재판단 (BK 8 → 16)

## 1. Background
017이 `bs` 뱅크 충돌을 없애면서 "공유 대역폭과 FFMA가 1:1로 묶여 K 루프에 아무것도
더 넣을 수 없다"는 기각 전제가 깨졌다. 재판단을 위해 먼저 현재 한계를 ncu로 쟀다
(`gemm_nt_bias`, 대표 런치 grid 272):

| 지표 | 값 |
|---|---|
| sm__pipe_fma_cycles_active | **59.2%** |
| l1tex__throughput | 62.6% |
| dram__throughput | 9.7% |
| sm__inst_executed / cycle | 2.62 / 4 |
| smsp__issue_active / cycle | 0.66 / 1.0 |
| sm__warps_active | 28.7% (= 13.8워프/SM, 2블록 상한 16의 86%) |
| stall: not_selected / long_sb / short_sb / barrier / mio | 1.55 / 0.92 / 0.47 / 0.44 / 0.26 |

**FMA도 l1tex도 포화가 아니다 — 지연 은닉이 한계다.** 그리고 타일 형상 축은 산술로
닫힌다: 누산기 64개에서 FFMA/공유로드 = `TM*TN/(TM+TN)`이라 8×8이 최댓값(4.0)이고,
누산기를 늘리면 레지스터가 127을 넘어 2블록/SM이 깨진다. 로그 기각표의
`256×64` 16×4(+93%)가 정확히 그 형태다. **비를 건드리지 않고 K 루프 오버헤드만
줄이는 축은 `BK` 하나뿐이다.**

## 2. Hypothesis
`BK`를 8 → 16으로 넓히면 FFMA/공유로드 비는 그대로인 채 배리어와 글로벌 로드
명령이 절반이 되어, K 루프가 잃던 발행 슬롯을 회수한다. k 오름차순 누산 순서가
그대로라 bitwise 동일해야 한다.

## 3. Design and Implementation (`src/model.cu`)
`BK = 16`. 스테이징만 바뀐다 — 스레드당 float4가 매트릭스당 1개에서 2개가 되므로
각 스레드가 `lr`과 `lr2 = lr + BM/2` 두 행을 올린다. K 루프 본문·에필로그 무변경.

**`BK = 32`는 불가**: 더블 버퍼 두 타일이 블록당 67,584 B라 2블록이면 135 KB로
SM당 100 KB를 넘는다. **BK=16이 2블록/SM을 유지하는 최대폭이다.**

점유율을 통제 변수로 고정하기 위해 두 커널에 `__launch_bounds__(GEMM_THREADS, 2)`를
함께 걸었다. 없으면 ptxas가 `gemm_grouped`에 **141 레지스터**를 줘 1블록/SM으로
떨어진다. 걸었을 때 `gemm_nt_bias` 123 · `gemm_grouped` 125, **spill 0 B**,
smem 16,896 → 33,792 B(2블록 = 67,584 B ≤ 100 KB).

## 4. Result (노드 b1, 한 allocation 안에서 교차 2rep)

| 빌드 | gemm | moe | lm_head | elapsed |
|---|---|---|---|---|
| base (BK=8) | 0.992 / 0.990 | 0.614 / 0.615 | 0.033 | 1.9437 / 1.9415 |
| base + launch_bounds (대조) | 0.994 / 0.997 | 0.593 / 0.595 | 0.033 | 1.9282 / 1.9317 |
| **BK=16 + launch_bounds** | **0.940 / 0.940** | 0.597 / 0.597 | **0.025** | **1.8687 / 1.8664** |

- **elapsed −0.075 s = 1.0402×.** 분해: `launch_bounds` 단독 −0.013(moe −0.021),
  **`BK=16` 자체 −0.062**(gemm −0.055, lm_head −0.008).
- 출력 `cmp` **bitwise 동일**(n=1024, PASSED).
- HEAD 빌드 재확인(b0): gemm 0.947, elapsed 1.884, **543.4 seq/s**.

**적용 후 ncu — 이득의 출처는 예측과 달랐다.**

| 지표 | BK=8 | BK=16 |
|---|---|---|
| fma 파이프 | 59.2% | **67.4%** |
| inst/cycle | 2.62 | **3.00** |
| issue_active | 0.66 | **0.75** |
| **long_scoreboard** | **0.92** | **0.02** |
| short_scoreboard | 0.47 | 0.16 |
| barrier | 0.44 | 0.40 |
| not_selected | 1.55 | 2.02 |

배리어가 줄어서가 아니다(0.44 → 0.40, 거의 무변화). **글로벌 로드 지연이 완전히
숨었다**(long_scoreboard 0.92 → 0.02) — 프리페치와 소비 사이의 컴퓨트 창이 p 8회에서
16회로 두 배가 됐기 때문이다. not_selected가 1.55 → 2.02로 오른 것이 같은 얘기다:
이제 발행 대기 워프가 남아돌고, 커널은 **지연 제약에서 발행 슬롯 제약으로 옮겨갔다**.

*혼선 하나 남긴다*: ncu가 보고한 커널 duration은 1.76 → 2.35 ms로 거꾸로 나왔다.
`--clock-control none` + 16개 메트릭 리플레이라 절대 duration은 비교 대상이 아니라고
보고 비율 지표만 읽었다. 판정은 end-to-end 교차 측정이다.

## 5. Next action
- **타일 형상 축은 닫혔다**(누산기 64개에서 8×8이 비 최적, 늘리면 2블록/SM이 깨짐).
  `BK`도 16이 상한(smem). 이 방향은 소진.
- 남은 stall은 not_selected 2.02 — 발행 슬롯 포화다. 여기서 더 가려면 명령 수 자체를
  줄여야 하는데, FFMA가 K 루프 명령의 94%라 줄일 것이 없다. **gemm 0.94는 이제
  구조적 바닥에 가깝다.**
- 따라서 다음은 gemm 내부가 아니라 **낭비되는 일**로 간다: 020 gate tall-skinny
  (BN=128 타일 중 16열만 유효 → 87.5% 폐기), 021 스크래치 버퍼 수명.

---

# EXP-022a — `gemm_nt_body`를 타일 형상 템플릿으로 (제어 변수 고정, 무변경 확인)

## 1. Background
022는 `gemm_nt_bias`만 `128×256`/`8×16`으로 옮기는 실험인데, 착수해 보니 설계 메모에
없던 장벽이 먼저 있었다: `gemm_nt_body`가 `__device__ __forceinline__`이고 `BM/BN/TM/TN`이
**파일 스코프 상수**라 두 커널이 같은 형상을 공유한다. 한쪽만 바꾸려면 body를 템플릿화해야
하고, 그러면 `gemm_grouped`의 코드젠도 같이 건드린다. 코드 주석 자체가 경고한다 —
*"writing them any other way moves ptxas off the instruction schedule the projections were
measured on"*(`src/model.cu`). 022 본체 측정이 오염되지 않도록 **리팩터가 공짜임을 먼저**
못박는다. 019가 `launch_bounds`를 별도 대조 행으로 잡은 것과 같은 규율.

## 2. Hypothesis
형상 상수를 템플릿 인자로 올리기만 하고 **값을 그대로 두면** ptxas 출력이 불변이어야
한다. 기대 Δ = **0**. 0이 아니면 022b의 이득/손실을 리팩터와 분리할 수 없다.

## 3. Design and Implementation (`src/model.cu`)
- `gemm_nt_body` → `template <int BM, int BN, int TM, int TN>`. `BK`와 블록 크기(256)는
  설계상 두 커널 공통이라 **템플릿 인자로 올리지 않았다**(019가 `BK=16`을 고정했고,
  이번에 움직일 것은 `TM·TN`뿐이라는 가설 1개 규율).
- `gemm_col_slot` → `template <int BN>`. `BN/2` 오프셋이 형상에 의존한다.
- 파일 스코프 `BM,BN,TM,TN` 삭제 → `PROJ_*`(gemm_nt_bias) / `GRP_*`(gemm_grouped) 두 조.
  현재 값은 둘 다 `128,128,8,8`이라 **호스트 grid 계산도 값이 불변**이다.
- static_assert 3종 추가: 타일이 블록을 채우는지(`(BM/TM)*(BN/TN)==256`), 스테이징이
  타일을 정확히 덮는지(`BM/2*(BK/4)==256`, `BN/2*(BK/4)==256`).
  **후자가 022b의 안전장치다** — `BN=256`이면 스레드당 float4가 2개에서 4개로 늘어야
  하는데, 이 assert가 없으면 조용히 절반만 스테이징된 타일을 계산한다.

## 4. Result

**SASS가 완전히 동일하다.** `cuobjdump -sass`를 HEAD와 대조하면 14,965행 중 차이는
번역 단위 이름이 들어간 mangled name 12곳뿐이고 **명령은 0곳**이다(`old.cu` vs `model.cu`).
`ptxas -v`도 019 기록과 일치: `gemm_nt_bias` **123** · `gemm_grouped` **125** 레지스터,
spill 0, smem 33,792 B.

end-to-end (노드 b7, 한 allocation 안에서 교차 2rep):

| 빌드 | gemm | moe | elapsed | seq/s |
|---|---|---|---|---|
| HEAD | 0.951 / 0.951 | 0.600 / 0.600 | 1.8859 / 1.8839 | 542.97 / 543.56 |
| **022a** | 0.944 / 0.955 | 0.606 / 0.598 | **1.8792 / 1.8837** | 544.92 / 543.61 |

**Δ elapsed −0.0035 s (0.19%), 노이즈 이하.** 출력 `cmp` **bitwise 동일**, 양쪽 PASSED
(max abs 0.000179291). SASS가 같으므로 이것은 예측이 아니라 **필연**이다.

부수 관찰: b7이 1.884를 냈다. EXP-015가 b1·b7을 slow(2.58 vs 2.16, 1.196×)로 분류했는데
이번엔 b0 기록(1.884)과 같다. **노드 분류가 그때 그대로인지 미확인** — A/B를 한
allocation 안에서 돌리는 규율은 그대로 유지한다.

## 5. Next action
**022b**: `PROJ_BM×PROJ_BN = 128×256`, `PROJ_TM×PROJ_TN = 8×16`. 이제 한 줄 변경 +
스테이징 확장(스레드당 float4 2 → 4, B만) + `extern __shared__` +
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` + `__launch_bounds__(256,1)`.
`GRP_*`는 손대지 않는다(`N=896`이라 `BN=256`이면 12.5% 패딩이 부활 — EXP-010이 없앤 손해).
반증 조건은 plan 그대로: ncu `long_scoreboard`가 0.02에서 되살아나면 즉시 기각.

---

# EXP-022b — 레지스터 타일 누산기 64 → 128 (기각)

## 1. Hypothesis
K 스텝당 스레드는 shared에서 `TM+TN` float를 읽어 `TM·TN` MAC을 돌리므로
**shared 파이프 수요 / FFMA 발행 수요 = `4(TM+TN)/(TM·TN)`**가 8×8에서 1.00,
8×16에서 0.75다. 017이 두 파이프를 1:1로 묶은 뒤 처음 생기는 여유이므로,
`gemm_nt_bias`를 `128×256`/`8×16`으로 옮기면 gemm이 −0.05~0.10 s 줄어야 한다.
대가는 2블록/SM → 1블록/SM인데, 019가 `not_selected 2.02`·`long_sb 0.02`로
**워프가 남고 숨길 지연이 없다**를 보였으므로 그 대가가 싸다고 봤다.

## 2. Design and Implementation
022a의 템플릿 body에 `WIDE_BM×WIDE_BN=128×256`, `WIDE_TM×WIDE_TN=8×16` 인스턴스를
추가하고 `gemm_nt_bias_wide`로 감쌌다. 부수적으로 필요한 것:
- **B 스테이징 4단계.** B 타일이 `256×16`=4096 float=1024 float4이므로 256스레드가
  스레드당 4개를 올린다(A는 2개 그대로). body의 스테이징을 `A_STEPS`/`B_STEPS`
  루프로 일반화했다.
- **smem 50,176 B > 48 KB 정적 한도** → `extern __shared__` +
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`(측정 구간 안, static local로 1회).
- `__launch_bounds__(GEMM_THREADS, 1)`. ptxas **221 레지스터 · spill 0**(추정 183보다 큼).
- `gemm_col_slot`의 float4 청크 간격을 `BN/2` → `BN/(TN/4)`로 일반화(TN=8에서 동일값).
- **`out >= 256`만 wide로 보낸다.** 라우터 gate는 `out=16`이라 BN=256이면 폐기가
  87.5% → 93.75%로 늘어난다 — 알려진 손해를 측정에 섞지 않기 위해 narrow에 남겼다.
- `GRP_*`는 무변경(N=896에 BN=256이면 EXP-010이 없앤 12.5% 패딩 부활).

## 3. Result — 기각

**3-way A/B (노드 b5, 한 allocation 안에서 교차 2rep).** 리팩터와 가설을 분리하기 위해
`b`(일반화된 스테이징, dispatch만 끔)를 대조로 넣었다:

| 빌드 | gemm | moe | lm_head |
|---|---|---|---|
| **a** = 022a | **1.110 / 1.111** | **0.683 / 0.683** | 0.029 |
| **b** = 일반화만 (전부 128×128) | 1.198 / 1.198 | 0.721 / 0.722 | 0.030 |
| **c** = 8×16 wide dispatch | 1.185 / 1.185 | 0.732 / 0.734 | 0.031 |

- **c − a = gemm +0.075 s.** 같은 주소 계산을 쓰는 b와 비교해도 **−0.013 s(1.1%)**뿐이다.
- 출력은 세 빌드 모두 **bitwise 동일**(`cmp` a=b=c, n=1024). 8×16이 각 출력의 k 오름차순
  FMA 열을 보존한다는 설계 논거는 **맞았다** — 기각 사유는 정확도가 아니라 속도다.

**ncu가 이유를 정확히 말해준다** (`gemm_nt_bias_wide`, grid (8,17), n=128, 019와 동일 조건):

| 지표 | 019 (8×8, BK=16) | 022b (8×16) | |
|---|---|---|---|
| **l1tex** | 62.6% | **51.9%** | ← 예측대로 내려갔다 |
| **long_scoreboard** | **0.02** | **0.47** | ← 반증 조건 발동 |
| warps_active | 28.7% (13.8워프) | 16.7% (8워프 = 1블록/SM) | |
| fma 파이프 | 67.4% | 63.9% | |
| issue_active | 0.75 | 0.69 | |
| not_selected | 2.02 | 0.82 | |
| barrier | 0.40 | 0.09 | |

**메커니즘은 작동했고, 값이 안 나왔다.** shared 파이프 압력은 비 0.75가 예고한 그대로
10.7pp 빠졌다(l1tex 62.6 → 51.9). 그런데 그 여유를 **점유율로 지불했고**, 1블록/SM(8워프)이
되자 019가 완전히 숨겼던 글로벌 지연이 되살아났다(`long_sb` 0.02 → **0.47**). 순효과는
발행이 **오히려 감소**(issue_active 0.75 → 0.69, fma 67.4 → 63.9%). plan에 미리 고정한
반증 조건("`long_scoreboard`가 0.02에서 되살아나면 즉시 기각")이 그대로 발동했다.

**부산물 — K 루프의 주소 형태는 레버다.** b−a = gemm **+0.088** · moe **+0.039**.
스테이징을 `A_STEPS`/`B_STEPS` 루프로 일반화하면서 미리 계산한 두 포인터(`ab`,`ab2`)를
`base + i*step`으로 바꾼 것뿐인데 레지스터가 123/125 → 128/128로 오르고 시간이 6.5%
나빠졌다. `gemm_nt_body` 주석의 경고("writing them any other way moves ptxas off the
instruction schedule")가 실측으로 확인됐다. 022a가 SASS 동일성을 먼저 못박아 둔 덕에
이 0.127 s가 가설의 이득/손실로 오계상되지 않았다.

## 4. 판정과 남는 것
- **코드 되돌림**(HEAD = 022a). 018과 같은 형태의 기각이다.
- 미시도로 남는 변형: wide 커널의 주소 계산을 손으로 두 포인터 형태로 펴는 것. 그러나
  ncu의 손실 1위는 주소가 아니라 **점유율에서 온 `long_sb` 0.47**이고, 주소를 고쳐 b의
  페널티(+0.088)를 전부 회수해도 a와 겨우 동률이다. 레지스터 221에는 지연을 더 숨길
  선행 깊이를 넣을 여유도 없다(측정-F: 선행 깊이 2는 이미 +0.047).
- **`upper-bound.md` §6.3의 후속 후보 1번(shared/LSU 재발행)이 죽었다.** 공유 파이프
  압력은 실제로 내려가지만 `issue_active`는 오르지 않는다 — 이 형상에서 **shared 여유와
  점유율은 맞바꿀 수 없고, 8×8·2블록이 더 좋은 점**이다. 남은 후보는 barrier(0.40)와
  mio(0.26)뿐이고 둘 다 022가 부수 효과로 노렸던 것이라, **FP32에서 gemm 0.94는 구조적
  바닥**이라는 019의 결론이 강화됐다. 57.4% → 80%로 가는 길은 이 코드베이스에 없다.
- 따라서 다음은 plan 그대로 **021 → 023 → 024/025**, 즉 비-GEMM 쪽이다.

---

# EXP-021 — 스크래치 버퍼 수명을 모델로 (채택)

## 1. Background
측정-E: `generate()` 에필로그의 `cudaFree` 24회(~1.5 GB)가 측정 구간 안에서 **20.77 ms
(1.10%)**, 그 동안 GPU는 idle이다. D2H 전체(0.002 s)의 1.45배. 스테이지 타이머 어디에도
안 잡힌다 — 마지막 `tick` 이후 스코프 이탈에서 일어나기 때문이다.

## 2. Hypothesis
24개 `DeviceBuffer`를 모델 멤버로 올려 해제를 `~PhiTinyMoEModel`으로 미루면
elapsed가 **−0.021 s** 준다. `cudaMalloc`은 구간 안에 그대로 둔다(첫 호출이 그대로 지불).
산술 무변경 → **bitwise 동일이 판정 기준**.

## 3. Design and Implementation
- `DeviceBuffer`: 생성자 인자 제거 + **grow-only** `reserve(n)` (`n > capacity_`일 때만
  free+malloc). 소멸자는 그대로 `cudaFree`.
- `PhiTinyMoEModel::Scratch` (model.cu, 익명 namespace **밖**)에 24개를 모아
  `mutable std::unique_ptr<Scratch> scratch_`로 보유. `generate()`는 `const`이므로 `mutable`,
  불완전 타입 멤버는 소멸자가 model.cu에 있어 성립.
- 본문은 `DeviceBuffer<T> d_x(n)` → `T* d_x = sc.x.reserve(n)`, `.get()` 63곳 소멸.
  버퍼가 포인터가 되며 `std::swap(d_hidden, d_next)`는 그대로.
- **재사용 안전성**: 24개 전부 매 호출에서 읽기 전에 전량 기록된다(`route_scan`은
  `max_tiles`까지 0 패딩까지 채운다). 애초에 `cudaMalloc`도 0을 보장하지 않으므로
  이전 호출의 잔여값에 의존하는 경로는 존재할 수 없다.
- **워밍업 논점 없음**: `run.sh`/`submit.sh`는 `-w`를 쓰지 않아 첫 `generate()`가 곧
  측정 구간이다. 즉 malloc은 구간 안에 남는다. (`-w`를 켜면 malloc이 워밍업으로
  빠지지만, 이는 연산 결과 캐싱이 아니라 할당이고 아래 측정은 `-w` 없이 했다.)

## 4. Result
b0, 한 allocation 안에서 a/b 교차 3rep, `APS_PROFILE` 없음:

| 빌드 | rep1 | rep2 | rep3 | 평균 | seq/s |
|---|---|---|---|---|---|
| a = HEAD(022a) | 1.878630 | 1.878076 | 1.875972 | **1.87756** | 545.4 |
| b = 021 | 1.858427 | 1.856389 | 1.858780 | **1.85787** | **551.2** |

**Δ = −0.0197 s (−1.05%, 1.0106×)** — 측정-E의 상한 20.77 ms와 사실상 일치(95%).

프로파일 런에서 스테이지 합은 불변(gemm 0.944/0.945, moe 0.601/0.600, norm 0.117 동일,
resid 0.058 동일)인데 elapsed만 1.882 → 1.862. **이득이 스테이지 밖 = 에필로그 해제에서
왔다는 직접 증거**다.

- n=1024 `-v`: 양쪽 `max abs diff 0.000179291` 동일, `cmp` **bitwise 동일**.
- `-n 2 -d -v` 통과: decode는 `forward()`가 `generate()`를 프리픽스가 자라며 16회 부르므로
  `reserve`의 **성장 경로까지 실측 검증**됐다(prefill 단독 경로에서는 안 밟힌다).

## 4. Next action
- 채택. 남은 malloc 24회는 구간 안에 있으나 측정-E에서 malloc은 free보다 훨씬 싸다
  (가상 주소 예약뿐) — 별도 실험 가치 없음.
- plan 그대로 **020 → 023 → 024/025**.

---

# EXP-023 — `layer_norm_rows` 행↔스레드 매핑 (채택)

## 1. Background
착수 전 ncu(n=1024, `-c 2`):

| 지표 | 값 | |
|---|---|---|
| duration | 1.647 ms × 65 launch | norm 0.117과 일치 |
| **barrier** | **16.67** | 256스레드 중 1스레드만 일하고 7워프가 `__syncthreads` 대기 |
| dram / l1tex | 34.0% / 35.1% | 루프라인이 아니다 |
| `occupancy_limit_shared_mem` | **5** | 행당 16,392 B → 동시 체인 5개/SM |
| issue_active / fma | 0.38 / 26.9% | |

## 2. Hypothesis
행별 j 오름차순 누적을 **그대로 두고** 1행=1레인으로 바꾸면 동시 체인이 5개/SM →
15,583개(전 행)로 늘어 barrier가 사라진다. **bitwise 동일이 판정 기준.**

착수 전 산술 정정: 행 하나가 shared에 16 KB를 잡는 것이 동시성 5의 원인이므로,
동시성을 올리면 **행이 mean/var/출력 세 구간에 걸쳐 상주할 수 없다** → x를 3번 읽는다
(트래픽 2 → 4 유닛). 현재 dram 34%이므로 순이득은 남지만 plan의 −0.045~0.055는
1R+1W를 가정한 값이고 정정하면 **−0.026~0.036**이 상한이다(4 유닛 × 88% dram).

## 3. Design and Implementation
- 32행 = 1워프, 레인 r이 행 r의 체인을 소유. 열을 `NORM_CHUNK`씩 shared로 스테이징:
  글로벌은 행 단위 coalesced, shared는 **전치** 저장(`s[c*33 + r]`)이라 체인이 열을
  따라 내려가며 32개 연속 float(1 wavefront)를 읽는다. 스트라이드 33이 뱅크 충돌을
  양방향으로 없앤다(쓰기 `(t+r)%32`, 읽기 연속).
- mean/inv는 레지스터에 남고 에필로그는 `__shfl_sync`로 브로드캐스트. 에필로그는 체인이
  없으므로 행 하나에 워프 전체를 붙여 coalesced로 되돌린다.
- 산술은 원본과 동일한 `__fadd_rn`/`__fsub_rn`/`__fmul_rn` 순서, `mean`·`inv` 식도 그대로.

**1차 시도는 2.1배 퇴행했다** (norm 0.133 → 0.281, b5). ncu가 원인을 지목:
`long_scoreboard` **24.51**(barrier 16.67을 이걸로 바꿔치기), issue_active **0.05**,
l1tex 8.8%, dram 31%. 스테이징이 행마다 `LDG → STS` 한 쌍이고 `rows_here`가 런타임
값이라 ptxas가 루프를 풀지 못해 **행당 글로벌 지연 1회**를 그대로 지불했다.

여기서 구조적 제약이 드러난다: 체인 수 = 행 수이므로 워프 수는 15,583/32 = **487개로
고정**이고 `waves_per_multiprocessor` 0.59, 즉 **5.9워프/SM이 상한**이다. TLP로는 지연을
가릴 수 없고 **전부 ILP로 가려야 한다.**

두 수정:
1. **스테이징 로드 명시적 배칭** — 행 루프를 컴파일타임 `NORM_ROWS`로 고정하고(범위 밖
   행은 행 0을 읽되 그 열은 되읽히지 않는다) `NORM_GROUP`행씩 **레지스터로 먼저 다 읽고
   나서** shared에 쓴다. 022b의 교훈대로 ptxas 휴리스틱에 맡기지 않고 소스에 의도를 적었다.
   → 0.281 → 0.105 (b7 기준선 0.117)
2. **에필로그 float4** — 순수 스트리밍 구간이라 행당 128트립 → 32트립, 트립마다 128 B
   라인 왕복. → 0.105 → **0.083**

형상 스윕(같은 노드 안에서만 비교):
| (CHUNK, GROUP) | norm |
|---|---|
| pre-float4, b7 (기준선 0.117): (128,4) 0.159 · (128,8) 0.143 · (64,4) 0.105 · (64,8) 0.105 |
| post-float4, b5 (기준선 0.133): (64,8) 0.089 · **(64,16) 0.083** · (32,8) 0.083~0.089 · (32,16) 0.083 |

(64,16) 채택 — (32,16)과 동률이나 레지스터 48 대 128.

## 4. Result
각 6rep, 한 allocation 안 교차, 프로파일 off:

| 노드 | a = HEAD(021) | b = 023 | Δ |
|---|---|---|---|
| **b7 (빠른 노드)** | 1.8606 (1.8580~1.8639) | **1.8340** (1.8314~1.8373) | **−0.0266 (−1.43%)** |
| b5 (느린 노드) | 2.1355 (±0.0002) | 2.0857 (±0.001) | −0.0498 (−2.33%) |

두 집합은 겹치지 않는다. b7 기준 550.9 → **558.9 seq/s**.
스테이지: norm 0.116 → **0.082**, gemm 0.935 → 0.940(+0.005, 측정-B 전력 밀도), 나머지 불변.

**새 커널은 노드에 무관하다**(b7 0.082 / b5 0.083)는데 기준선은 노드에 따라 0.116/0.133이다.
루프라인에 붙은 커널의 서명이고, 이 실험의 메커니즘을 그대로 확인해 준다.

최종 ncu: **dram 89.03%**, duration 1.2588 ms, barrier **0**, l1tex 18.6%.
1,022 MB / 1.2588 ms = 812 GB/s. **트래픽 4유닛으로 루프라인에 도달했다.**
`long_scoreboard`는 18.28로 남아 있으나 이제는 대역폭 포화 커널의 정상 서명이다
(issue 0.07, fma 3.2% — 계산은 아무것도 아니다).

- n=1024 `cmp` **bitwise 동일**, `-v` 양쪽 `0.000179291`. `-n 2 -d -v` 통과.

- **제출: 556.733910 seq/s (n=1024, 리더보드 1위/15). 개인 최고 523.4 → 556.7 (+6.4%),
  baseline 0.033306 대비 16,716x.** 제출 런 elapsed 1.8393 — 노드는 pin하지 않았고 빠른 쪽에
  붙었다(A/B의 b7 1.8340 / b5 2.0857 사이). 직전 최고 523.4는 021·023 이전 상태이므로
  두 실험의 −0.047 s가 그대로 반영된 값이다.

## 4. Next action
- 채택. 커널 쪽 여지는 없다(89% dram). **더 줄이려면 트래픽을 줄여야 한다.**
- **기록해 둘 트레이드오프**: x를 3번 읽는 것은 bitwise 제약이 강제한 것이다. 누적 순서를
  바꿔도 된다면 워프 트리 리덕션으로 1R+1W(2유닛)가 되어 norm 하한이 **~0.045**, 즉 지금보다
  추가로 **−0.037**이다. 이 프로젝트는 MoE top-2 라우팅이 뒤집히는 위험 때문에 bitwise를
  자기 규율로 삼아 왔으므로 채택하지 않았다 — 규율을 완화할지는 판단 사항이다.
- 024(residual → norm/GEMM 에필로그 융합)는 이제 **읽기 1회를 없애는 것**이기도 하다:
  resid 0.058 삭감 외에 norm의 4유닛 중 1유닛이 같이 사라진다.

# EXP-024
## 1. Background
resid 0.058 (2.8%)은 `add_inplace_kernel` 64회(레이어당 2회)다. 계획은 "norm/GEMM
에필로그 융합"으로 적어뒀고 023의 Next action은 **norm 쪽 융합**(읽기 1회 삭감)을 가리켰다.
착수 전 ncu로 어느 쪽이 큰지부터 재측정했다 (n=1024, b7, launch-skip 20):

| 커널 | duration | dram bytes | dram % | l1tex % |
|---|---|---|---|---|
| `add_inplace_kernel` | **905 µs** | 765 MB | **92.7** | 20.2 |
| `moe_combine` | 914 µs | 766 MB | 91.9 | 22.5 |

버퍼는 15,583 × 4096 × 4 B = 255 MB. 765 MB = **3유닛**(read c · read resid · write c)이고
92.7%면 루프라인이다 → 커널을 빠르게 만들 길은 없고 **없애는 것만이 길이다**.
64 × 905 µs = 57.9 ms로 t_resid 0.058과 정확히 일치한다(누락 비용 없음).

## 2. Hypothesis
생산자 에필로그로 옮기면 3유닛이 1유닛(resid read)으로 줄고, 그 1유닛은
**o_proj GEMM의 유휴 DRAM에 숨는다**(o_proj는 레이어당 ~15 ms 계산에 트래픽 0.6 GB =
dram ~4%). 예상 −0.039~0.048.

**소비자(norm) 융합보다 생산자 융합이 산술적으로 우월하다.** norm에 넣으면 add 커널
3유닛을 지우는 대신 norm 1패스에 2유닛(resid read + sum write)이 붙어 순 −302 µs/레이어인데,
에필로그에 넣으면 −905 + (숨는 read) ≈ **−900 µs/레이어**다. 023의 Next action이 가리킨
쪽을 이 산수로 버렸다.

## 3. Design and Implementation
두 add를 각각 자기 생산자의 store 지점으로 옮겼다.
- `gemm_nt_body`에 `bool FUSE_RESID` 템플릿 인자 + `resid` 포인터. 에필로그는
  `v = acc + bias[col]; if (FUSE_RESID) v += resid[o];`. **o_proj만** 쓰는
  `gemm_nt_bias_resid`로 별도 인스턴스화 — 나머지 5개 projection과 `gemm_grouped`는
  `FUSE_RESID=false`로 기존 SASS를 그대로 유지한다(019·022a에서 맞춰놓은 스케줄 보존).
- `moe_combine`은 `out[c] = acc + res[c]`.
- `add_inplace_kernel`/`add_inplace_device` 삭제, 프로파일에서 `resid` 항 삭제.

**bitwise 근거**: 융합 지점의 `v`(또는 `acc`)는 기존 add 커널이 **읽어들이던 그 fp32
비트패턴**이다(에필로그의 마지막 연산이 이미 라운딩을 끝냈다). 뒤에 덧셈 1개가 붙을 뿐이고
곱셈이 인접하지 않아 FMA 축약 여지도 없다. 재결합은 nvcc가 하지 않는다.

**점유율 반증 조건**: `ptxas -v`로 `gemm_nt_bias_resid` **123 레지스터**
(= `gemm_nt_bias` 123), `gemm_grouped` 125 불변, smem 33,792 B 불변 → 2블록/SM 유지.
022b를 죽인 조건이 재발하지 않음을 착수 직후 확인했다.

## 4. Result
각 3rep, 한 allocation 안 교차:

| 노드 | a = HEAD(023) | b = 024 | Δ |
|---|---|---|---|
| **b7 (빠른 노드)** | 1.8434 (1.8422~1.8445) | **1.7983** (1.7958~1.8008) | **−0.0451 (−2.45%)** |
| b5 (느린 노드) | 2.0932 (2.0866~2.0967) | 2.0453 (2.0414~2.0488) | −0.0479 (−2.29%) |

두 집합은 겹치지 않는다. b7 기준 555.5 → **569.5 seq/s**.

스테이지(b5): **resid 0.058 → 0** · gemm 1.112 → 1.114 (**+0.002**) · moe 0.687 → 0.691
(**+0.004**) · 나머지 불변.

- o_proj의 resid read는 **+0.002**로 사실상 무료 — 가설의 "유휴 DRAM에 숨는다"가 그대로 확인.
- `moe_combine`은 루프라인 커널이라 값을 치를 것으로 봤고(예상 +0.0097), 실제 **+0.004**로
  예상의 절반이었다. 3유닛 → 4유닛인데 시간이 4/3배가 안 된 것 = 풋프린트가 커지며 read
  효율이 올라간 쪽으로 본다(미확인, 남은 크기가 4 ms라 추적 안 함).
- n=1024 `cmp` **bitwise 동일**, `-v` `0.000179291`(023과 동일값). `-n 1 -d -v` 통과.

- **제출: 569.907380 seq/s (n=1024, 리더보드 1위/15). 개인 최고 556.7 → 569.9 (+2.4%),
  baseline 대비 17,111x.** 제출 런 elapsed 1.7968 — 빠른 노드에 걸렸다(A/B의 b7 1.7983과
  일치). 556.7 → 569.9의 +13.2 seq/s가 024의 −0.045 s 그대로다.

## 4. Next action
- 채택. **resid 스테이지가 사라졌다** — 남은 분해는 gemm 0.963 · moe 0.614 · norm 0.082 ·
  attn 0.104 · lm_head 0.027 (b7).
- 다음은 025(`moe_combine` → w2 에필로그). 단 **024가 025의 상한을 깎았다**: 이제
  `moe_combine`은 4유닛(read lo · read hi · read resid · write)이고 w2 에필로그로 접으면
  w2가 출력을 안 쓰고 두 조각을 합쳐야 하는데, 두 조각은 서로 다른 블록/타일에 있다.
  착수 전에 이 의존성부터 확인한다.

---

# 측정-H: 커널 루프 언롤링 여지 판단 (부분 전개·occupancy 포함)

EXP-024 코드(`9b76dd5`) 기준, 코드 변경 없이 세 가지 실측. (a) `cuobjdump -sass`로
최내곽 루프 명령 구성, (b) `nsys -t cuda ./main -n 1024` 1회(elapsed 2.0441 s /
501.0 seq/s, 커널 총합 2019.3 ms), (c) `ncu`로 occupancy·SpeedOfLight.
부분 전개는 `#pragma unroll N`만 바꿔 **컴파일만** 반복했다.

## H-1. 시간 점유와 전개 상태

| 커널 | 시간 (ms) | 비율 | 루프 상태 |
|---|---|---|---|
| `gemm_nt_bias` 693.6 / `gemm_grouped` 607.8 / `gemm_nt_bias_resid` 432.8 | 1734.2 | **85.9%** | **전개 완료** |
| `attention_heads` | 105.2 | 5.2% | **전개 완료** |
| `layer_norm_rows` | 82.2 | 4.1% | **전개 완료** |
| `gather_rows` 39.6 / `moe_combine` 37.7 / `silu_mul` 6.6 | 83.9 | 4.2% | 미전개, **대역폭 바운드** |
| `rope_rows` | 12.2 | 0.60% | 미전개 |
| `route_place` 1.10 / `route_scan` 0.36 / `route_count` 0.15 | 1.6 | 0.080% | 미전개 |

**이미 전개된 것이 95.2%**다.

## H-2. SASS: 뜨거운 루프는 전부 펼쳐져 있다

| 커널 | 최내곽 루프 본문 | 구성 |
|---|---|---|
| `gemm_nt_body` (kt 1회, BK=16) | **1147 ins** | FFMA **1024** · LDS 64 · STS 16 · LDG 4 · 정수/분기 ~39 |
| `attention_heads` (청크 1개) | 111 ins | LDS 41 · FMUL 32 · FADD 32 → 32열 청크 전부 전개 |
| `layer_norm_rows` | 725 / 599 ins | FADD 128/64 · LDG 64 · LDS 64 · STS 64 → 64회 전개 |
| `rope_rows` | 37 ins | LEA 11 · IMAD 4 · IADD3 4 · LDG 4 · FMUL 4 |
| `gather_rows` | 14 ins | LDG.E 1 · STG.E 1 · 나머지 주소 계산 |

GEMM은 FFMA 1000개당 비-FFMA가 **120개**다. 루프 제어만 보면 39/1147 = **3.4%**이고,
그것도 FFMA와 다른 파이프다.

## H-3. 부분 전개는 레지스터를 못 내리고 spill을 **만든다**

p 루프 `#pragma unroll N` 스윕 (grouped / nt_bias):

| p 루프 전개 | 레지스터 | spill stores |
|---|---|---|
| **전개 (현재, BK=16 전부)** | **125 / 123** | **0 / 0** |
| `unroll 8` | 127 / 127 | 0 / **8 B** |
| `unroll 4` | 127 / 126 | **16 B** / 0 |
| `unroll 2` | 124 / 126 | 0 / 0 |
| `unroll 1` (전개 없음) | 117 / 115 | 0 / 0 |

`__launch_bounds__(GEMM_THREADS, 2)`가 128에서 자르기 때문에 **부분 전개가 오히려
spill을 유발한다**(8 B, 16 B). 전개를 완전히 끊어도 **115~117**이고, 3블록/SM 문턱인
**80 레지스터**(65536 ÷ 24워프 → 단위 256 내림 2560/워프)에는 근처도 못 간다.
바닥이 구조적이다 — `acc[8][8]` **64개**가 레지스터 타일 자체이고 전개와 무관하다.

## H-4. occupancy 문은 두 번 닫혀 있다 (ncu)

`ncu -k regex:gemm --launch-skip 40 --launch-count 4` (n=64):

| | launch 0 grouped | launch 1 grouped | launch 2 nt_bias | launch 3 nt_bias |
|---|---|---|---|---|
| **Block Limit Registers** | 2 | 2 | 2 | 2 |
| **Block Limit Shared Mem** | **2** | **2** | **2** | **2** |
| Block Limit Warps | 6 | 6 | 6 | 6 |
| Theoretical / Achieved | 33.3 / 29.2% | 33.3 / 32.0% | 33.3 / 27.3% | 33.3 / 16.7% |
| L1/TEX | 65.8% | 65.9% | 66.6% | 63.6% |
| Compute (SM) | 52.7% | 69.1% | 64.6% | 30.5% |
| DRAM | 24.1% | 32.6% | 11.3% | 5.1% |

레지스터가 2블록으로 제한하는 것은 사실이지만, **BK=16의 더블 버퍼 타일 33,792 B가
공유메모리로도 2블록으로 제한한다**. 즉 레지스터를 80까지 내려도 3블록이 되지 않는다.
실제로 `__launch_bounds__(GEMM_THREADS, 3)`을 강제하면 레지스터가 정확히 **80**이 되고
**796 B / 812 B spill stores**가 생긴다 — 1024 FFMA 루프 안에서 누산기를 로컬로 내리는 것이다.

occupancy를 열려면 BK를 8로 되돌려(smem 16,896 B) **동시에** 레지스터를 80 이하로
낮춰야 하는데, 앞은 EXP-019가 BK=16을 이득으로 실측했고 뒤는 EXP-022b가 누산기 변경을
기각했다. 두 문이 이미 각각 측정으로 닫혀 있다.

## H-5. 스트리밍 4.2%는 DRAM 루프라인에 붙어 있다

실측 유효 대역폭: `moe_combine` **867** · `gather_rows` **837** · `silu_mul` **816** GB/s
= 3090 피크 936 GB/s의 **87~93%**. 명령 수를 줄일 이유가 없다.

## 판단: 기각

전개 대상 시간의 **95.2%가 이미 전개돼 있고**, 4.2%는 대역폭 바운드다. 전개 여지가
남은 것은 `rope_rows`(0.60%)와 `route_*`(0.080%) 합 **0.68%**뿐이고, rope는 37 ins 중
19가 주소 계산이라 이득 여지가 가장 크지만 **정수 오버헤드를 전량 제거해도 상한 0.2%**다.

부분 전개도 기각. spill은 현재 0인데 부분 전개가 spill을 만들고, occupancy는 레지스터와
공유메모리에 **이중으로** 막혀 있다. occupancy 제한자는 언롤링이 아니라
**8×8 레지스터 타일 + BK=16 더블 버퍼 타일**이다. `attention_heads`는 40 레지스터라
레지스터 절벽에서 멀어 이 논점이 성립하지 않는다.

---

# EXP-025 — `moe_combine` → w2 에필로그 융합 (채택)
## 1. Background
024가 남긴 마지막 비-GEMM 레버. 024 시점 분해(b7)는 gemm 0.963 · **moe 0.614** · norm
0.082 · attn 0.104 · lm_head 0.027이고, moe 안의 `moe_combine`이 그 대상이다.
024가 이 커널의 상한을 **깎아놓은** 상태(resid add를 여기에 접었으므로 3유닛 → 4유닛)라
착수 전에 재측정했다 (n=1024, launch-skip 20, 3 launch 전부 동일값):

| 커널 | duration | dram bytes | dram % | sm % |
|---|---|---|---|---|
| `moe_combine` | **1.18 ms** | 1.02 GB | **95.1** | 12.5 |
| `gemm_grouped` w13 (7, 259) | 13.3 ms | 1.42~1.63 GB | 11.7~13.7 | 72.5 |
| `gemm_grouped` w2 (32, 259) | **6.76 ms** | 1.60~1.88 GB | **26.0~30.5** | 70.9 |

1.02 GB = **4유닛**(read lo · read hi · read resid · write) × 255 MB, 95.1%면 루프라인이다.
32 × 1.18 = **37.8 ms**가 이 실험의 절대 상한이다.
(ncu는 base 클록으로 돌아 GEMM duration이 boost 대비 ~25% 길다: 커널 합
13.3+6.8+1.18+gather 1.18+silu 0.19 = 22.6 ms/레이어 vs 프로파일 실측 19.2.)

## 2. Hypothesis
w2 에필로그로 옮기면 511 MB expert-output 버퍼를 **쓰지도 읽지도 않는다**. 대신 붙는
읽기는 w2의 **dram 27%**, 즉 유휴 대역폭 쪽에 들어간다(024의 판정 기준: 루프라인
엘리먼트와이즈는 계산 바운드 *생산자*의 에필로그로 접는다).

**계획서가 지적한 의존성이 진짜 문제였다**: 한 토큰의 두 조각은 서로 다른 블록/타일에
있고, atomic으로 합치면 `(0.5lo + 0.5hi) + res`의 3항 그루핑이 깨져 bitwise가 죽는다.
해법은 atomic이 아니라 **순서**다 — 두 조각을 서로 다른 행 서브레인지로 패킹해
launch 2개로 쪼개면 launch 경계가 필요한 순서를 공짜로 준다.

## 3. Design and Implementation
- **패킹을 (expert, 두 expert 중 몇 번째) 32 서브레인지로.** `route_count`가 `cnt[2*lo]`와
  `cnt[2*hi+1]`을 세고, `route_place`의 스레드 32개가 각자 서브레인지 하나(= 토큰의 두 조각
  중 하나)를 맡는다. `slot`은 필요 없어져 삭제.
- **서브레인지 2e와 2e+1을 인접하게** 두어 expert e의 행은 여전히 연속이다 → `route_scan`이
  타일 리스트 3개를 한 버퍼에 쓴다: w13용(expert 16개, 지금과 동일) + w2용 2개(lo 서브레인지
  16개 / hi 16개). **부분 타일 16개 추가 비용을 w2만 부담한다.** 순진하게 "lo 전부 → hi 전부"로
  깔면 w13도 같은 비용을 물어 이득이 반토막 난다(13.2 ms × 3.2% × 32 ≈ 13 ms).
- **에필로그**: `gemm_nt_body`에 `int COMBINE` 템플릿 인자 + `rowmap`(= gather index,
  packed row → token row). `COMBINE=1`은 `c[o] = 0.5f*v`, `COMBINE=2`는
  `s = c[o] + 0.5f*v; c[o] = s + resid[o]`. w2 전용 `gemm_grouped_combine<1|2>`로 별도
  인스턴스화 — projection 5개와 w13은 `COMBINE=0`으로 SASS 그대로.
- `moe_combine`, `d_expert_out`(511 MB), `d_slot` 삭제.

**bitwise 근거**: 기존 `acc = 0.0f; acc += 0.5f*lo; acc += 0.5f*hi; out = acc + res`는
`(0.5lo + 0.5hi) + res`다. 새 경로는 launch 1이 `0.5lo`(0.5 곱은 지수만 건드리므로 정확),
launch 2가 `(0.5lo + 0.5hi) + res` — **같은 두 덧셈, 같은 순서**. `c[o] + 0.5f*v`가 FFMA로
축약돼도 결과가 같다(`0.5*v`가 정확하므로 라운딩 1회가 어디에 있든 동일). 두 조각이 다른
launch에 있으므로 순서는 non-deterministic이 아니다.

**점유율 반증 조건**: `ptxas -v` `gemm_grouped_combine<1>`/`<2>` 모두 **125 레지스터**
(= `gemm_grouped` 125), smem 33,792 B 불변 → 2블록/SM 유지. 022b 조건 재발 없음.

## 4. Result
각 3rep, 한 allocation 안 교차:

| 노드 | a = HEAD(024) | b = 025 | Δ |
|---|---|---|---|
| **b7 (빠른 노드)** | 1.7994 (1.7967~1.8025) | **1.7796** (1.7757~1.7834) | **−0.0198 (−1.10%)** |
| b5 (느린 노드) | 2.0466 (2.0403~2.0520) | 2.0193 (2.0166~2.0213) | −0.0273 (−1.33%) |

두 집합은 겹치지 않는다. b7 기준 569.0 → **575.4 seq/s**.

스테이지(b7): **moe 0.616 → 0.590 (−0.026)** · gemm 0.963 → 0.968(+0.005, b5에서는 0.000이라
전력 밀도 잡음) · 나머지 불변.

**37.8 ms 상한 중 26 ms를 회수했다.** 남은 12 ms의 출처를 ncu로 분해 (base 클록):

| | before | after |
|---|---|---|
| w13 (7, 259) | 13.32 / 13.05 ms · dram 11.7 / 13.7% | 13.22 / 13.04 ms · 11.5 / 12.1% — **불변** |
| w2 | 6.76 ms · 1.74 GB · dram 28% | `<1>` 3.53 + `<2>` 3.73 = **7.26 ms** · 2.06 GB · 23.6 / 38.3% |
| `moe_combine` | 1.18 ms · 1.02 GB · dram 95.1% | **삭제** |

w2가 **+0.50 ms/레이어**다. 산술 귀속(개별 측정은 아님): 부분 타일 16개 추가 ≈ +0.22,
늘어난 트래픽 0.32 GB를 dram 38%에서 삼키는 값 ≈ +0.28. 레이어당 순 −0.68 ms(base 클록
−21.8 ms), boost에서는 GEMM 쪽이 줄어 −26 ms — 측정과 일치.

- 설계 의도 확인: **w13은 duration·dram·waves 전부 불변**이다(서브레인지 인접 배치가 값을 했다).
- DRAM 총량은 레이어당 −0.70 GB. 스크래치도 511 MB 줄었다.
- n=1024 `cmp` **bitwise 동일**, `-v` `0.000179291`(023·024와 동일값). `-n 1 -d -v` 통과
  (max abs 0.000103).

- **제출: 575.888114 seq/s (n=1024, 리더보드 1위/16). 개인 최고 569.9 → 575.9 (+1.0%),
  baseline 대비 17,291x.** 제출 런 elapsed 1.7781 — 빠른 노드(A/B의 b7 1.7796과 일치).

## 4. Next action
- 채택. b7 분해는 **gemm 0.968 · moe 0.590 · norm 0.082 · attn 0.104 · lm_head 0.028**.
- **계획서의 실험 목록은 020(gate tall-skinny, −0.03 이하)만 남았다.**
- moe 0.590의 내역은 이제 w13 GEMM ~0.34 · w2 GEMM ~0.19 · `gather_rows` 0.038 · silu 0.006
  (base 클록 환산)으로, **83%가 GEMM**이다. 남은 엘리먼트와이즈는 `gather_rows` 하나(1.18 ms
  × 32 = 37.8 ms, dram 89%)이고 이것도 생산자(post_norm 에필로그)로 접는 것이 같은 판정
  기준에 걸린다 — 다만 gather는 행을 **복제**하므로(1행 → 2 packed row) 생산자가 자기 출력
  1행을 두 곳에 써야 하고, 그 두 곳은 라우팅 결과에 의존한다. 026 후보.

---

# EXP-026 — 공유메모리 **스토어** bank conflict 제거 (projection만, 채택)

## 1. Background
025로 비-GEMM 레버가 소진됐고(b7 분해: gemm 0.968 · moe 0.590 · norm 0.082 · attn 0.104 ·
lm_head 0.028), 남은 시간의 **83%가 GEMM**이다. 019/022b가 닫아놓은 것은 두 개다 —
occupancy(레지스터·smem 이중 제한)와 레지스터 타일 8×16(shared 압력은 실제로 내려가지만
1블록/SM이 되어 `long_sb` 부활). 그래서 **점유율을 건드리지 않고 shared 파이프의
비-FFMA 작업만 줄이는 것**만 남는데, EXP-017이 본 것은 shared **로드**뿐이었다.

k 루프의 shared 트래픽을 워프·k타일당 wavefront로 실측(ncu, `gemm_nt_bias` grid (4,9)):

| | wavefront / warp / k-tile | 비고 |
|---|---|---|
| 로드 `bs` 2×LDS.128 | 8 | 4씩, EXP-017이 도달시킨 바닥 |
| 로드 `as` 2×LDS.128 | 4 | 2씩 — lane 0-15가 같은 주소라 **브로드캐스트가 실제로 싸다** |
| p 루프 16회 소계 | **192** | |
| **스토어 16×STS.32** | **32** | 이상값 16, **절반이 conflict** |

192 : 256(FFMA 사이클)이므로 shared는 이미 12.5% 여유가 있고 1:1 균형점이 아니다.
`upper-bound.md` §3.1의 비 1.00은 `as`의 브로드캐스트를 세지 않은 값이다.

## 2. Hypothesis
스토어 주소는 `(lc+q)*(BM+4) + lr`, 워프 안에서 `lr∈0..7` · `lc∈{0,4,8,12}`이므로
bank = `(lc*4 + lr) % 32` → **lc=0/8, lc=4/12가 충돌**하고 bank 8-15·24-31은 아예 안 쓴다.
스토어를 conflict-free로 만들면 shared 총 wavefront가 224 → 208(**−7.1%**) 된다.

**패딩으로는 불가능하다.** conflict-free는 `4*stride ≡ 8 (mod 32)` 즉 `stride ≡ 2 (mod 8)`을
요구하고, float4 로드는 `stride ≡ 0 (mod 4)`를 요구한다 — 양립 불가.
그래서 시프트를 **열**에 넣는다: 원소 (k,m)을 열 `m + (k>>2)*8`에 둔다. 네 스테이징
그룹이 서로 다른 bank 옥텟을 잡아 32뱅크를 정확히 한 번씩 덮는다.

## 3. Design and Implementation (`src/model.cu`)
- `SPAD` 4 → **32**(`SWIZ`일 때). 시프트된 열을 담을 폭이 필요하고, 32의 배수라
  **행 스트라이드 자체는 bank 오프셋을 만들지 않는다** → 로드 패턴은 017 그대로.
- **명령이 한 개도 안 는다.** 로드 쪽 `p`는 전개된 내부 루프의 컴파일타임 상수라
  `gemm_kshift(p)`가 `p*(BM+SPAD)`와 같은 리터럴 변위로 접힌다. 스토어 쪽 `lc`는 4의
  배수라 주소가 여전히 `lc*stride + lr` 한 개의 multiply-add다.
  SASS 명령 수 실측: `gemm_nt_bias` 2552 → 2552, `gemm_nt_bias_resid` 3368 → 3368.
- **`SWIZ`를 템플릿 인자로 두고 grouped는 껐다.** 아래 4-2 참조.

## 4. Result
`gemm_nt_bias` ncu (n=64, 같은 런치):

| | before | after |
|---|---|---|
| store bank conflicts | 4,746,502 / 1,179,648 | **0** |
| store wavefronts | 2,359,296 | **1,179,648** (= 이상값) |
| load wavefronts | 14,155,776 | 14,155,776 (불변) |

### 4-1. 전부 켰을 때 — grouped가 손해 (노드 b7, 한 allocation 교차 3rep)
| 스테이지 | a = HEAD | b = 전부 SWIZ | Δ |
|---|---|---|---|
| **gemm** (projection, spill 0) | 0.962 | **0.940** | **−0.022** |
| **moe** (grouped, spill 24 B) | 0.588 | **0.619** | **+0.031** |
| elapsed | 1.7688 | 1.7791 | +0.010 |

가설은 맞았고 **grouped만 손해**였다. 원인은 레지스터다: grouped 3종은 125로
`__launch_bounds__(_, 2)`의 128에 3만 남겨두고 있어서, 시프트를 넣으면 **k 루프 안에서**
24 B가 spill한다(SASS `STL`/`LDL`이 루프 본문 0x4fc0/0x5000·0xd00에 있다).
시도한 네 형태(flat 배열 / 3D 배열 / `sc2 = sc + BM/2` / `lc*2` 형태) 전부 128+spill이었고,
**같은 flat 형태에 패딩만 원래대로 되돌린 대조군도 128+spill**이라 원인이 시프트가 아니라
표현 형태임도 확인했다. `SPAD=32`만 넣고 시프트를 끈 대조군은 125/123·spill 0이다.

### 4-2. projection만 (채택, 노드 b5, 한 allocation 교차 3rep)
| 빌드 | elapsed | gemm | moe |
|---|---|---|---|
| a = HEAD | 2.0194 / 2.0151 / 2.0144 | 1.119 | 0.668 |
| **b = 026** | **1.9886 / 1.9843 / 1.9836** | **1.084** | **0.668** |

**Δ elapsed −0.0308 s (−1.53%)**, 두 집합이 겹치지 않는다. moe는 설계대로 **0.000**이고
grouped 3종의 SASS는 HEAD와 **명령 단위로 완전 동일**(1624/1704/1944 전부 일치)이다.
- n=1024 `cmp` **bitwise 동일** — 산술을 안 건드리는 레이아웃 변경이므로 필연.
  `-v` `0.000179291`(023~025와 같은 값), `-n 1 -d -v` PASSED (max abs 0.000102997).
- 같은 빌드 b7 단독 런 **1.7398 s / 588.6 seq/s**.

## 5. Next action
- 채택. b5 −0.031을 노드비(1.196×)로 환산하면 b7에서 −0.026, 4-1의 gemm −0.022와 일치.
- **남은 것은 grouped의 레지스터 3개다.** 열면 moe 쪽 GEMM에서 −0.012가 더 나온다
  (gemm −0.022 × moe GEMM 0.53/0.96). EXP-027 후보.
- 그 다음은 계획서의 020(gate tall-skinny)과, 아직 안 센 **grouped의 부분 행타일 낭비**.

---

# EXP-027 — 부분 행타일의 죽은 행 그룹 건너뛰기 (기각)

## 1. Background
GEMM 안에 남은 **구조적 낭비**를 셌다. grouped GEMM은 range마다 행을 `GRP_BM=128`
경계까지 패딩하는데, 그 패딩 행의 FFMA는 에필로그의 `row >= m`이 전부 버린다.
실측(n=1024, 라우팅 실제값):

| 런치 | 타일 | 실제 행 | 패딩 후 | **낭비** |
|---|---|---|---|---|
| w13 | 250 | 31,166 | 32,000 | **2.61%** |
| w2 lo | 130 | 15,583 | 16,640 | **6.35%** |
| w2 hi | 129 | 15,583 | 16,512 | **5.63%** |

ncu 커널 시간(base 클록, 레이어당) w13 13.19 ms · w2 3.53+3.73이므로 상한은
레이어당 0.78 ms = **32레이어 25 ms**, boost 환산 약 −0.020 s.
(projection은 m=15583에 마지막 타일이 95행이라 낭비 0.21%로 무시 가능.)

## 2. Hypothesis
스레드는 행 `ty*TM .. ty*TM+TM-1`을 소유하므로 **그 그룹 전체가 타일 끝을 넘으면**
그 스레드의 TM·TN MAC은 전부 버려진다. `live = m0 + ty*TM < m`으로 k 루프의 compute만
건너뛰고 스테이징(`lr`/`lc` 인덱스라 `ty`와 무관)은 남기면 낭비의 대부분이 회수된다.
`ty`는 타일 경계가 걸친 워프 하나를 빼면 워프 유니폼이고 predicate는 별도 파일이라
레지스터 비용이 없어야 한다.

## 3. Design and Implementation
`gemm_nt_body`에 `const bool live = m0 + ty * TM < m;`을 k 루프 앞에 두고 p 루프 전체를
`if (live)`로 감쌌다. 에필로그의 `row >= m`은 그대로라 **산술 무변경**.
`ptxas -v`: 5개 커널 전부 레지스터·spill·smem **불변**(125, spill 0).

## 4. Result — 기각

노드 b6, 한 allocation 교차 3rep:

| 빌드 | elapsed | gemm | moe |
|---|---|---|---|
| a = 026 | 1.7333 / 1.7341 / 1.7319 | 0.931 | 0.585 |
| b = 027 | 1.7537 / 1.7467 / 1.7480 | **0.951** | **0.581** |

**메커니즘은 작동했다** — moe 0.585 → 0.581. 그런데 **gemm이 0.931 → 0.951(+2.1%)**로
그보다 크게 나빠져 순 **+0.015 s**다. projection은 122타일 중 부분 타일이 1개뿐이라
얻을 것이 없는데 분기 비용만 문다. 출력은 `cmp` bitwise 동일.

**비용의 정체는 명령 수가 아니다.** SASS 총 명령은 `gemm_nt_bias` 2552 → 2560,
`gemm_grouped` 1624 → 1632으로 **+8뿐**이고 늘어난 것은 `BSSY`/`BSYNC` 각 +1,
`BRA` +1, `NOP` +4(정렬)다. 즉 1088명령짜리 compute 블록을 **재수렴 구간으로 감싼 것**
자체가 2%다. moe도 같은 2%를 물어 기대 −0.020 중 −0.004만 남았다.

## 5. 판정과 남는 것
- **코드 되돌림.** 018·022b와 같은 형태의 기각 — 메커니즘은 맞고 값이 안 나온다.
- 미시도 변형: `SWIZ`처럼 템플릿으로 **grouped에만** 켜기. gemm은 0.931로 돌아오고
  moe만 −0.004가 남는다. 양수지만 1.3 seq/s라 단독 착수 가치는 낮다.
- 분기를 워프 유니폼으로 만들려면 `ty = tid/16` 매핑을 바꿔야 하는데, 그것은 EXP-017이
  conflict-free로 맞춰놓은 `bs` 읽기 패턴을 깨므로 교환이 성립하지 않는다.
- **부분 타일 낭비 20 ms는 회수되지 않은 채 남는다.** 꼬리 타일을 별도 `BM=64`
  인스턴스로 빼는 길은 B 스테이징 일반화를 요구하고(022b가 +0.088로 실측한 위험)
  낭비도 절반만 준다.

---

# EXP-028 — 라우터 게이트 전용 tall-skinny 커널 (채택, 계획서의 020)

## 1. Background
GEMM 안에 남은 **구조적 낭비** 두 번째. 게이트는 `out=16`인데 출력 타일이 `BN=128`이라
**128열 중 112열을 계산해서 버린다.** 레이어별 런치 실측(ncu, base 클록):

| 커널 | grid | duration | fma% | waves |
|---|---|---|---|---|
| **gate** `gemm_nt_bias` | (1,122) | **1.11 ms** | **70.2** | 0.74 |
| q_proj | (16,122) | 13.59 | 70.5 | 11.90 |
| k/v_proj | (4,122) | 3.46 | 69.3 | 2.98 |
| o_proj resid | (32,122) | 13.37 | 69.8 | 23.80 |
| w13 grouped | (7,259) | 13.19 | 66.7 | 11.05 |

게이트는 **모델에서 fma 효율이 가장 높은 커널**인데 그 87.5%를 아무도 안 읽는 열에 쓴다.
32레이어 × 1.11 = **35.5 ms**.

## 2. Hypothesis
x 원소 하나가 16 MAC에만 쓰이므로 **바이트당 8 flop**, 즉 이 형상은 발행 바운드가 아니라
**DRAM 바운드**다. 레이어당 활성값 255 MB / 800 GB/s = **0.32 ms**로 타일 커널의 1/3.
타일 모양만 `64행 × 16 expert`로 바꾸면 된다.

**bitwise가 여기서 특히 중요하다** — 이 점수는 top-2 argmax로 들어가므로 아주 작은 드리프트도
경계 토큰의 expert를 바꿔 출력이 통째로 달라진다. 그래서 각 출력은 지금처럼 **레지스터 하나에
k 오름차순 `acc += a*b`**로 누산해야 한다(그래서 split-k와 lane별 k 분할은 불가).

## 3. Design and Implementation
**먼저 staging 없는 판을 만들어 봤고, 그 실패가 설계를 결정했다.**
레지스터 전용(스레드 = 4행 × 1 expert, x·w를 글로벌에서 직접):

| | 값 |
|---|---|
| `dram__bytes_read` | **255.7 MB = 이상값 그대로** |
| dram 활용 | 23.1% |
| fma | 6.8% |
| **l1tex** | **70.2%** ← 한계 |
| duration | 1.22 ms (타일판 1.11보다 **느림**) |

대역폭이 아니라 **L1 요청 수**로 죽는다. 레인당 expert가 1개면 워프 32레인 중 16개가 같은
주소를 요청해 **로드 명령 하나가 유효 데이터 32 B만 옮긴다**. GEMM 커널이 shared staging을
쓰는 이유가 정확히 이것이다.

채택판 `gemm_gate`: `BM=64 · BN=16 · TM=4 · TN=1 · BK=16`, 256스레드, 더블 버퍼.
- A 스테이징이 **스레드당 정확히 float4 1개**(64행 × 16k = 256 float4). `as`는 026의
  k그룹 열 시프트를 그대로 쓴다(stride 96 = 32의 배수).
- `bs`는 16×16이라 패딩 없이도 `bs[p][tx]`가 워프당 브로드캐스트 1 wavefront다.
- `DeviceLinear::forward`가 `out == GATE_N && bias == nullptr`일 때만 디스패치.
  다른 5개 projection과 lm_head는 경로가 그대로다.
- 레지스터 40, smem 14,336 B → 6블록/SM. **기존 커널 5종은 무변경.**

## 4. Result

`gemm_gate` ncu: duration **1.11 → 0.601 ms**, dram 23 → **47.1%**, l1tex 64.6%,
`dram__bytes_read` 255.6 MB(여전히 이상값).

노드 b5, 한 allocation 교차 4rep:

| 빌드 | elapsed | gemm |
|---|---|---|
| a = 026 | 1.9848 / 2.0194 / 1.9849 / 1.9841 | 1.083 |
| **b = 028** | **1.9692 / 1.9703 / 1.9692 / 1.9692** | **1.073** |

**Δ −0.0156 s**(중앙값 기준), b7 환산 −0.014. b의 4rep이 1.9692~1.9703으로 거의 무분산이다.
- n=1024 `cmp` **bitwise 동일**, `-v` **0.000179291**(023~026과 동일값),
  `-n 1 -d -v` PASSED (max abs 0.000102997). 라우팅이 한 토큰도 안 바뀌었다는 뜻.

## 5. Next action
- 채택. 게이트는 35.5 → 19.2 ms(base)로 줄었고 **남은 16 ms 중 상당수는 아직 l1tex**다
  (64.6%, DRAM 바닥은 0.28 ms). TN=4·128스레드 형상이면 A의 레지스터 재사용이 생겨
  비가 0.75 → 0.375로 반이 되는데, `BM = 16·TM·TN` 제약 때문에 블록 수가 122로 준다.
  추가 여지 −0.009 정도라 우선순위는 낮다.
- 다음은 **grouped의 레지스터 3개**다. 그것만 열면 026의 SWIZ를 moe 쪽에도 켤 수 있고
  −0.012가 나온다(026 4-1에서 gemm −0.022로 실측된 효과의 moe 지분).

---

# EXP-029 — k 루프의 비-FFMA 명령 줄이기 (3형태 전부 기각)

## 1. Background
026·028이 회수한 것은 둘 다 **일 자체를 없앤 것**이었다(스토어 wavefront 절반, 게이트의
버려지는 열 87.5%). 커널 내부에 그런 것이 더 있는지 SASS로 k 루프를 해부했다
(`gemm_nt_bias`, 역방향 BRA로 루프 경계를 잡음):

```
k 루프 본문 1147 명령 = FFMA 1024 (89.3%) · LDS 64 · STS 16 · LDG 4 · 나머지 39
```

나머지 39의 정체가 새 정보다. **글로벌 로드 4스트림의 주소를 매 반복 처음부터 다시
만든다** — 스트림마다 `IMAD.WIDE a_row, k` + `LEA`/`LEA.HI.X` 쌍, 5명령 × 4 = 20,
거기에 `a_ok2`/`b_ok2` 술어 재계산까지 **26명령**. `ab`/`ab2`/`bb`/`bb2`는 루프 앞에서
한 번 계산되는 `const` 포인터인데, 4개 × 64비트 = **8레지스터**라
`__launch_bounds__(_, 2)`의 128 안에서 125를 쓰는 커널이 들고 있을 수가 없다.

## 2. Hypothesis와 3형태

**(a) 스트림을 4 → 2로.** 스레드가 두 행에서 4k씩 대신 **한 행의 연속 8k**를 float4
2개로 올리면 스트림이 2개가 되고 두 번째 float4는 컴파일타임 +16 B다.
`lr = tid/(BK/8)`(0..127), `lc = (tid%(BK/8))*8`.

**(b) 32비트 원소 오프셋.** 포인터 대신 `ao = a_row*k + lc`를 들고 `a + ao + kt`.
곱셈이 루프 밖으로 나간다. 오프셋 최대는 lm_head의 `n*k` = 131 M로 int에 들어간다.

**(c) grouped에 SWIZ를 켜기 위한 레지스터 확보.** 026이 grouped에 못 켠 이유가 이
레지스터 3개다. `row0`을 포인터에 접어 넣으면(`a + t[1]*k`, `c + t[1]*n`, `rowmap + t[1]`,
`row0=0`, `m=t[2]`) **body는 한 줄도 안 바뀌고** 래퍼만 바뀐다.

## 3. Result — 전부 기각

### (a) 명령은 줄었는데 섹터가 2배 (실측 +0.054 s)
`ptxas -v` 깨끗(projection 127 · grouped 125, spill 0), k 루프 **1147 → 1133**,
FFMA 비율 **89.3 → 90.4%**. 그런데:

| | a = 028 | b = (a) |
|---|---|---|
| elapsed (4rep) | 1.7408 / 1.7385 / 1.7366 / 1.7414 | **1.7933 / 1.7934 / 1.7935 / 1.7928** |
| gemm | 0.935 | 0.967 |
| moe | 0.589 | 0.617 |

**ncu가 원인을 정확히 짚는다**(`gemm_nt_bias`, n=64, 같은 런치):

| | a | b |
|---|---|---|
| global ld **requests** | 304,128 | **304,128** (동일) |
| global ld **sectors** | 4,546,156 | **9,120,848 = 2.006×** |
| fma | 63.9% | **56.5%** |
| duration | 630 us | 702 us |

기존 매핑은 **한 행에 4레인**(`lc ∈ {0,4,8,12}`)이라 LDG.128 하나가 행마다 64 B =
32 B 섹터 2개를 **꽉 채운다**. (a)는 한 행에 2레인이라 같은 명령이 섹터를 2배로 건드리고
각각 절반만 쓴다. **BK=16 · 256스레드에서 float4 2개를 스레드당 올리려면, 섹터를 채우는
배치는 "4레인/행 × 2행"뿐이고 그것이 곧 4스트림이다.** 명령 14개와 섹터 2배의 교환이었다.

### (b) 부호확장이 곱셈보다 싸지 않다 (컴파일 단계 기각)
k 루프 **1147 → 1151**(`SHF` +4). `IMAD.WIDE`는 그대로 남고 `int → 64비트` 부호확장만
붙었다. 레지스터도 그대로. 측정 없이 기각.

### (c) grouped 레지스터는 3개가 안 나온다 (컴파일 단계 기각)

| 조합 | `gemm_grouped` | `gemm_grouped_combine<1,2>` |
|---|---|---|
| HEAD | 125 · spill 0 | 125 · spill 0 |
| **fold만** | **123 · spill 0** ✓ | **128 · spill 12 B** ✗ |
| fold + SWIZ | 128 · spill 24 B ✗ | — |

- fold는 `gemm_grouped`에서 **−2**를 벌지만 combine 2종에서는 **+3 손해**다. 이유가
  분명하다 — `a`는 커널 파라미터라 오프셋이 없으면 **상수뱅크에서 공짜로 다시 읽히는데**,
  `a + t[1]*k`가 되는 순간 레지스터에 들고 있어야 한다. combine은 그 위에 `rowmap`과
  `resid`까지 얹는다.
- `gemm_grouped`만 123까지 내려도 SWIZ가 거기서 **+5**를 먹어(nt_bias에서는 +2였다)
  다시 128 + spill이다. 할당기가 선형이 아니다.
- 026에서 이 spill의 실측 비용은 **moe +0.031 s**로, SWIZ가 줄 이득(−0.012)보다 크다.

## 4. 판정
세 형태 모두 **코드 되돌림**. 종합하면 **k 루프는 이 형상에서 바닥에 있다**:
- FFMA 89.3%는 줄일 명령이 사실상 없다는 뜻이고, 26명령짜리 주소 재생성은 레지스터
  128 상한의 **결과**지 원인이 아니다. 그것을 없애는 유일한 길(스트림 반감)이 섹터를
  2배로 만든다.
- shared 로드도 바닥이다: `bs`의 LDS.128은 워프당 512 B 고유 데이터라 4 wavefront가
  **하드웨어 하한**이고, `as`는 브로드캐스트로 이미 2다. 026이 마지막 여유(스토어)를 먹었다.
- **grouped의 스토어 conflict 16 wavefront/워프/k타일은 회수 불가로 남는다** —
  레지스터 예산 안에 들어가지 않는다.

---

# 판정 — "matmul 커널만으로 600 seq/s" (026~029 세션 결론)

## 출발점과 목표
025 제출 **575.888 seq/s**(elapsed 1.7781, 빠른 노드). 600 seq/s = elapsed **1.7067**이므로
필요량은 **−0.071 s**다. scope를 matmul 안으로 좁히면 GEMM이 1.49/1.77 s이므로
**GEMM을 −4.8%** 해야 한다.

## 도달한 곳
누적 **−2.27%** (노드 b5, 한 allocation 교차 4rep, 025 대비):

| | a = 025 | b = 029(HEAD) |
|---|---|---|
| elapsed | 2.0195 / 2.0165 / 2.0189 / 2.0166 | **1.9744 / 1.9732 / 1.9701 / 1.9706** |
| gemm | 1.114 | **1.069** |
| moe | 0.674 | 0.668 |

`cmp` **bitwise 동일**, `-v` **0.000179291**(023~025와 같은 값), `-n 1 -d -v` PASSED.
575.888의 elapsed에 −2.27%를 적용하면 **≈589 seq/s**. **600 미달이다.**

| 실험 | 결과 |
|---|---|
| 026 공유메모리 **스토어** bank conflict (projection) | 채택 **−0.031 s** |
| 027 부분 행타일 죽은 행 건너뛰기 | 기각 +0.015 |
| 028 gate tall-skinny (= 계획서 020) | 채택 **−0.016 s** |
| 029 k 루프 비-FFMA 명령 줄이기 3형태 | 전부 기각 |

## 왜 600에 못 가는가 (실측 근거)

**k 루프는 이 형상에서 바닥이다.** 세 방향이 각각 하한에 닿아 있다.

1. **shared 로드** — `bs`의 LDS.128 4 wavefront는 워프당 512 B 고유 데이터의
   **하드웨어 하한**이고, `as`는 브로드캐스트라 이미 2다(4가 아니다 — `upper-bound.md`
   §3.1의 비 1.00은 이 브로드캐스트를 안 센 값이라 실제 shared 수요는 FFMA의 81%다).
   남아 있던 유일한 여유가 **스토어**였고 026이 먹었다.
2. **명령 수** — k 루프 1147 중 FFMA가 1024(89.3%)다. 나머지 39의 정체는
   글로벌 4스트림 주소 재생성(26)인데, 이것은 레지스터 128 상한의 **결과**다.
   없애는 유일한 길(스트림 4→2)이 global 섹터를 **2.006배**로 만든다(029-a 실측).
3. **점유율** — 019/022b가 레지스터·smem 이중 제한으로 이미 닫아놨다.

**회수 못 하고 남는 것**(둘 다 상한이 실측돼 있다):

| 남은 레버 | 상한 | 막는 것 |
|---|---|---|
| grouped의 스토어 conflict | −0.012 | 레지스터 3개. 세 형태 전부 128 + **k 루프 안** spill 24 B이고, 그 spill의 실측 비용(moe +0.031)이 이득보다 크다 |
| 부분 행타일 낭비 (w13 2.61% · w2 6.0%) | −0.020 | 1088명령 블록을 `if`로 감싸면 생기는 `BSSY`/`BSYNC`가 GEMM 전체를 2% 늦춘다. 꼬리를 `BM=64` 별도 인스턴스로 빼면 낭비의 절반(−0.010)만 회수되고 B 스테이징 일반화(022b가 +0.088로 실측한 위험)를 요구한다 |
| gate를 DRAM 바닥까지 (0.601 → 0.28 ms/레이어) | −0.006 | `BM = 16·TM·TN` 제약 때문에 TN을 올리면 블록이 122로 준다 |

**둘을 다 회수해도 ≈596이다.** 589 → 600의 −1.8%는 GEMM 안에 없다.

## matmul이 아니라고 확인한 것
- **lm_head**: 14.20 ms · fma **70.7%**로 나머지 GEMM과 동일하게 효율적이다.
  `t_lm` 0.028의 잔여는 **131 MB logits D2H**이고 커널이 아니다.
- 남은 비-GEMM은 norm 0.083 · attn 0.117 · logits D2H. **600은 이쪽에 있다.**

---

# EXP-030 — 출력 `logits`를 pinned memory로 (기각)

## 1. Background
가드레일이 갱신되어 `tensor.h`/`tensor.cu`를 고칠 수 있게 됐다. 이 구간의 버스
트래픽은 **logits D2H 131.3 MB 하나뿐**이고(측정-E: H2D 전량 1 MB/0.063 ms), 대역폭은
실측돼 있다 — pageable 17.3 ms(7.6 GB/s) 대 **pinned 5.07 ms(25.9 GB/s)**(측정-G §2).
지금까지 이 12 ms를 못 먹은 이유는 하나였다: `Tensor`가 `std::vector<float>`이라
**pinned로 할당**할 길이 없어 `cudaHostRegister`(측정-G §3: 구간 안에서 139배 팽창,
실측 −25~−30 ms)나 staging+memcpy(측정-G §4: 1청크 이득 0)밖에 남지 않았다.
제약이 풀렸으므로 직접 할당을 실측한다.

## 2. Hypothesis
`logits`를 `cudaHostAlloc`으로 잡으면 D2H가 25.9 GB/s로 올라 **−12 ms**. 할당 비용
(측정-G: GPU idle에서 104 ms)은 EXP-014가 이미 만들어 둔 alloc 스레드 — 69 ms 제로필을
통째로 숨긴 그 스레드 — 에 얹히므로 숨는다.

**반증 조건**: `cudaHostAlloc`이 `cudaHostRegister`처럼 in-flight 커널과 직렬화되면
그만큼을 어디선가 기다린다. 그래서 `t_alloc`(스레드 안)과 `t_join`(꼬리 대기)을 같이 잰다.

## 3. Design and Implementation
- `tensor.h`: `enum class Alloc { Host, Pinned }` + `Tensor(shape, alloc = Host)`.
  저장소를 `std::vector<float>` → `float* + size_ + alloc_`으로 바꾸고 복사/이동/소멸을
  명시했다(`layer.cu`의 `Tensor flat = x;`가 복사를 요구한다). 기본값이 Host라
  나머지 전 호출지(가중치 2027개, `answers` 131 MB)는 동작 무변경.
- `tensor.cu`: Pinned면 `cudaHostAlloc(..., cudaHostAllocDefault)`, 아니면 `new float[]`.
  해제는 `cudaFreeHost`/`delete[]`. 소멸자는 throw 못 하므로 실패는 stderr 보고.
- `model.cu`: alloc 스레드 한 줄만 `Tensor({batch, VOCAB}, Tensor::Alloc::Pinned)`.
  제로필도 그대로 뒀다(할당자만 바꾸는 변수 1개 실험). 커널·산술 무변경.

## 4. Result
`cmp` **bitwise 동일**, `-v` 0.000179291, `-n 1 -d -v` PASSED. **성능은 +105 ms 손해.**

b7, 한 allocation 교차 3rep (plain):

| | a = HEAD(029) | b = pinned |
|---|---|---|
| elapsed | 1.7232 / 1.7243 / 1.7619 | **1.8329 / 1.8419 / 1.8406** |

b5, 교차 2rep (`APS_PROFILE=1`):

| | h2d | lm_head | alloc | join | elapsed |
|---|---|---|---|---|---|
| a | 0.007 / 0.003 | 0.029 / 0.029 | — | — | 1.9759 / 1.9752 |
| b | **0.119 / 0.117** | **0.020 / 0.020** | 0.125 | **0.000** | 2.0847 / 2.0911 |

가설의 두 항이 둘 다 분리돼 찍혔다.

- **이득 = `lm_head` 0.029 → 0.020 = −9 ms.** pinned D2H는 실제로 붙었다(16.5 → ~7.5 ms).
- **비용 = `t_h2d` 0.003 → 0.118 = +114 ms.** alloc 스레드가 `cudaHostAlloc`을 도는
  125 ms 동안 **메인 스레드의 CUDA 호출이 막힌다.** `t_join` 0.000이므로 꼬리는
  기다리지 않는다 — 숨기기에 실패한 게 아니라 **다른 스레드를 세우는 것이 비용**이다.
- 측정-G의 `cudaHostRegister`와 성격이 다르다. register는 **자기가** 139배
  (17.8 → 2466 ms) 팽창했다. `cudaHostAlloc`은 **자기는 안 팽창한다**
  (idle 104 + 제로필 ~13 ≈ 실측 125). 대신 그 125 ms 동안 컨텍스트 락을 쥐고
  launch/memcpy를 세운다. ⇒ **131 MB 페이지 락 비용은 이 구간 안에서 어느 스레드에
  올려도 숨지 않는다.** 측정-G의 결론이 register의 특성이 아니라 page-lock 자체의
  특성이었음이 확인됐다.

순 −9 + 114 = +105 ms (실측 b7 +108, b5 +109~116). **기각, 코드 원복.**

## 5. 남은 상한 (사용자 판단 필요)
pinned D2H의 값이 **−9 ms**(b7 1.723 s의 0.52%)로 실측 확정됐다. 이걸 먹는 유일한 길은
페이지 락을 측정 구간 **밖**에서 치르는 것 = 모델이 pinned 출력 버퍼를 **생성자에서**
잡아두고 `generate()`에서 `std::move`로 `logits`에 넘기는 것(구간 안 할당 0, D2H만 이득).
대가 2개: (a) 생성자는 `batch`를 모르므로 용량을 **상수로 박아야** 한다(초과 시 Host
폴백) — 벤치마크 크기에 맞춘 상수다. (b) 할당·제로필이 측정 구간 밖으로 나간다 —
연산 결과 캐싱은 아니지만 가드레일 해석 문제다. 상한이 −0.5%이므로 판단은 사용자에게 맡긴다.

## 6. Next action
버스는 이걸로 닫힌다(상한 −9 ms, 그것도 구간 밖 할당 전제). 남은 비-GEMM은
**norm 0.083 · attn 0.117**이다.
# EXP-031·032 — 다른 입력 형상에 대한 방어 (성능 무관)

## 1. Background

평가 입력이 지금 것과 달라져도 되는지 물어 입력 의존 가정을 훑었다. 현재 입력은
1024 seq · 길이 6~32 · 19,803 토큰 → 15,583 노드다. 하드코딩된 상수는 없고 버퍼는
전부 `total`/`batch`에서 동적으로 잡히지만, 두 곳이 이 형상에서만 성립한다.

## 2. 두 지점

- **031** `d_index`/`d_expert_in`은 `packed_rows = 2*total`로 잡는데 마지막 lm_head
  gather는 `batch`행을 쓴다. `total < batch/2`면 오버런. trie가 중복 시퀀스를 접으므로
  `total`은 `batch` 아래로 내려갈 수 있다(현재는 15,583 ≫ 512라 안 걸린다).
- **032** attention의 dynamic shared는 키 개수에 선형이다:
  `(HEAD_DIM+1)*4 + nk*(1+ATTN_CHUNK+1)*4` = `516 + 136*nk` B. `cudaFuncSetAttribute`가
  없어 블록 기본 한도 48 KB에 그대로 걸린다.

## 3. 실측 (sm_86, 128스레드 블록 프로브, 크기당 별도 프로세스)

기본 per-block 49,152 B · opt-in 상한 101,376 B.

| 길이 | shared | opt-in 없음 | opt-in |
|---|---|---|---|
| 32 (현재 입력) | 4,868 B | ok | ok |
| 357 | 49,068 B | ok | ok |
| 358 | 49,204 B | **FAIL** `cudaErrorInvalidValue` | ok |
| 600 | 82,116 B | FAIL | ok |
| 741 | 101,292 B | FAIL | ok |
| 2047 | 278,908 B | FAIL | **FAIL** |

## 4. 조치

- 031: `reserve(max(packed_rows, batch))`. 현재 입력은 할당량 변화 없음.
- 032: 크기 산정을 `nk_max = min(max_len, SLIDING_WINDOW)` 기준으로 바꾸고(윈도우 밖
  키는 커널이 안 읽는다), 48 KB를 넘을 때만 `cudaFuncSetAttribute` opt-in, opt-in 상한도
  넘으면 "최장 741 토큰"을 적어 throw한다. 현재 입력은 4,868 B라 두 CUDA 호출 다 안 탄다.
- **절벽 357 → 741 토큰.** 검증 PASSED · 587.3 seq/s(직전 587.5, 노이즈) ·
  max abs diff 0.000179291로 동일.

## 5. 남는 한계 (조치 안 함)

- 741 토큰 초과는 여전히 못 돈다. shared staging 없는 폴백 커널이 필요한데 누산 순서를
  비트 단위로 보존해야 하고 현재 입력으로는 그걸 검증할 방법이 없다.
- `anc`는 `sum(depth+1)`이라 길이에 quadratic이다. 길이 2048짜리 비공유 시퀀스 1024개면
  `anc_off`의 int32가 넘친다.
- 활성화 ≈ 127 KB/node. 15 GB 가중치 옆이라 대략 6만 노드가 상한(현재의 약 4배).

---

# 탐색 — 새로 열린 4개 파일(`tensor.*`/`layer.*`)에 남은 레버

## 1. `layer.h`/`layer.cu`는 레버가 **아니다** (dead code)

`Linear`/`PhiMLP`/`PhiMoE`/`PhiAttention`/`PhiDecoderLayer`를 참조하는 코드가
저장소에 **하나도 없다**. `tensor_ops::*`를 부르는 곳도 `layer.cu` 자신뿐이다.
model.cu가 `Device*` 구조체로 전부 다시 구현했기 때문이다. 링크는 되지만 측정 구간에서
한 줄도 실행되지 않는다. **기대 speedup 0.**

## 2. `tensor.h`/`tensor.cu`의 접점은 정확히 2곳

| 지점 | 내용 |
|---|---|
| `logits = Tensor({batch, VOCAB})` (alloc 스레드) | 131 MB 할당 + 제로필. 이미 레이어 뒤에 숨겨져 있다 |
| `to_host(logits, d_logits)` | 131 MB pageable D2H — **여기가 전부다** |

`at()`/`reshape()`/`fill()`은 prefill 경로에 없다. 즉 이 두 파일이 새로 여는 것은
**logits D2H 하나**이고, 그 크기는 측정 구간 총 버스 트래픽(측정-G: 16.6 ms)과 같다.

## 3. D2H를 줄이는 4가지 설계 실측 (`docs/microworld/meas-h-d2h-parallel-bench.cu`, b5)

131.3 MB, 목적지는 P를 뺀 전부가 평범한 pageable 호스트 메모리:

| 설계 | 시간 | 필요한 것 |
|---|---|---|
| **A** pageable `cudaMemcpy` 1회 (현재) | **14.96 ~ 15.66 ms** (8.8 GB/s) | — |
| B pageable를 호스트 스레드 2/4/8/16개로 분할 | 16.59 ~ 27.50 ms (**전부 악화**) | 없음 |
| S 고정 pinned staging(16/32/64 MB×2) + 스레드 memcpy | 12.77 ~ 17.99 ms (노이즈, 최선 −2 ms) | 구간 밖 staging |
| **P** 목적지 자체를 pinned | **5.03 ms** (26.1 GB/s) | **tensor.h** |

- **B 기각**: 드라이버의 pageable 경로를 스레드로 쪼개면 오히려 느려진다. staging
  bounce 버퍼 경합이지 호스트 memcpy 대역폭 문제가 아니다.
- **S 기각**: 측정-G §4의 단일 스레드 결론이 멀티스레드에서도 유지된다. pageable
  목적지에 쓰는 비용(write-allocate)이 벽이라 DMA를 아무리 겹쳐도 8~10 GB/s다.
- **P만 유효**: 3배. **호스트 memcpy를 아예 없애는 것**이 유일한 길이고, 그건
  `Tensor`의 저장소를 바꿔야만 된다 = 정확히 이번에 열린 파일.

`cudaHostAlloc` 131 MB = 110.9 ms이고, EXP-030이 이걸 구간 안에서 치르면 **+114 ms**
(다른 스레드에 얹어도 컨텍스트 락으로 메인 스레드의 launch를 막는다)임을 실측했다.
⇒ **구간 밖 할당이 유일한 경로.**

---

# EXP-033 — pinned 출력 버퍼를 생성자에서 예약 (채택)

## 1. Hypothesis
모델이 생성자에서 고정 크기 pinned 버퍼를 잡아두고 `generate()`가 그것을 `logits`로
넘기면, 구간 안 할당은 0이고 D2H만 15.0 → 5.0 ms가 된다. **−10 ms.**

## 2. Design and Implementation
- `tensor.h`/`tensor.cu`: 저장소를 `std::vector<float>` → `float* + size_ + capacity_ +
  alloc_`. `Alloc::{Host,Pinned}`, rule-of-5, `reshape_within_capacity()`(예약 하나가
  임의의 작은 batch를 담도록). 복사는 항상 Host(pinning은 소유자만 요청한다).
- `model.h`: `PINNED_OUT_BYTES = 256 MB`(= 2093행)를 **바이트 예산**으로 고정.
  벤치마크 batch(1024)가 아니라 메모리 예산이고, 넘치면 pageable로 폴백한다.
- `model.cu`: 생성자가 `pinned_out_`을 잡고, `generate()`는 (i) `logits`가 이미 pinned면
  재사용 (ii) 아니면 예약을 `std::move` (iii) 둘 다 아니면 기존 alloc 스레드 경로.
  소유권이 `logits`로 넘어가므로 `model_ref.reset()` 뒤의 `write_outputs`에도 안전하다.
  커널·산술 무변경.

## 3. Result
출력 **bitwise 동일**, `-v` 0.000179291, `-n 1 -d -v` PASSED
(decode는 8192 forward 중 8191개가 폴백 경로를 타므로 폴백도 같이 검증된다).

`[profile]` b5: **`lm_head` 0.029 → 0.020** (−9 ms). 다른 항목 무변화.

교차 실행 (같은 allocation, 3rep):

| 노드 | a = HEAD(032) | b = 033 | Δ |
|---|---|---|---|
| b5 | 1.9740 / 1.9693 / 1.9689 | **1.9628 / 1.9604 / 1.9617** | **−8.3 ms** |
| b7 | (1.7650) / 1.7299 / 1.7274 | **1.7184 / 1.7174 / 1.7217** | **−9.1 ms** |

b7 throughput 592.4 → **595.6 seq/s** (+0.54%). 모델 로드 시간 변화 없음(5.94 → 5.96 s,
노이즈 내). 실측이 예측(−10 ms)의 90%다 — §3의 5.03 ms는 순수 DMA이고 실제로는
`cudaMemcpy` 호출 오버헤드가 붙는다.

## 4. 대가 (사용자 판단 필요)
**256 MB의 페이지 락(약 220 ms)이 측정 구간 밖, 모델 생성자로 나간다.**
- 가드레일 문언상 금지 대상은 "구간 밖 **추론 연산**"과 "연산 결과 캐싱"이다.
  버퍼 할당은 둘 다 아니고, 이미 가중치 1.5 GB `cudaMalloc`과 EXP-020의 scratch
  `cudaFree`가 같은 위치에 있다.
- 다만 **출력 버퍼 용량이 입력과 무관한 상수**가 된다. 벤치마크 크기(1024×VOCAB
  = 131 MB)를 그대로 박지 않고 256 MB 예산으로 잡아 2배 여유를 뒀지만, 상수인 것은
  사실이다. 이 해석을 받아들이지 않으면 이 커밋을 되돌리면 되고, 그 경우
  **`tensor.*`/`layer.*` 4개 파일에서 회수 가능한 것은 0이다**(§1, §3의 B·S 기각).

**판단 확정(사용자, 2026-08-27): 생성자 사전할당 방식으로 진행. 채택 유지.**
HEAD(038)에서 재확인 — b7 `lm_head` 0.018, 1.6342 s / **626.6 seq/s**, 검증 PASSED.

## 5. Next action
버스는 닫혔다(잔여 D2H 5 ms). `layer.*`는 영구히 레버가 없다.
남은 비-GEMM은 **norm 0.083 · attn 0.117**뿐이고 둘 다 model.cu 안이다.

---

# 측정-I: 비-GEMM 커널 루프라인 전수조사 (b5, ncu, n=1024)

033으로 버스를 닫은 뒤 남은 것이 어디인지 확정하기 위해 GEMM 외 커널을 전부 쟀다.
(`TMPDIR`을 개인 디렉터리로 두지 않으면 `/tmp/nsight-compute-lock` 권한 때문에 ncu가
`InterprocessLockFailed`로 죽는다.)

| 커널 | 1회 | 호출 | 총 | DRAM 피크 | FMA 피크 |
|---|---|---|---|---|---|
| `attention_heads` | 3.34 ms | 32 | **107 ms** | **17.4%** | **18.9%** |
| `layer_norm_rows` | 1.27 ms | 65 | **83 ms** | **88.1%** | — |
| `rope_rows` | 0.383 ms | 32 | 12.3 ms | 91.2% | 4.4% |
| `silu_mul` | 0.201 ms | 32 | 6.4 ms | 90.1% | 18.8% |

## 판정

- **`rope_rows`·`silu_mul`은 닫혔다.** 90/91%면 DRAM 바닥이다.
- **`layer_norm_rows`도 88%로 바닥인데, 읽는 양 자체가 3배다.**
  `dram__bytes_read` = **766 MB = 3 × 255 MB**, write 255 MB. L2가 아무것도 못 잡는다
  (487블록 × 32행 × 16 KB = 245 MB 워킹셋 대 L2 6 MB). 즉 커널 튜닝의 여지는 없고,
  **유일한 레버는 mean/var/output 3패스가 행을 재적재하지 않게 만드는 것**이다.
  상한: 1020 MB → 510 MB, **83 → ~45 ms (−38 ms, −1.9%)**.
  제약은 559행 주석의 비트 동일성이다 — 행의 누산은 j 오름차순 직렬이어야 한다
  (트리 리덕션은 라우터 로짓을 1e-7 흔들어 1024개 중 4개 시퀀스가 검증 실패).
  행을 레지스터에 상주시키되(32레인 × 128 float) 레인 간 누산을 순서대로 잇는 설계가
  후보. 128 reg/thread면 16워프/SM은 남는다.
- **`attention_heads`가 프로그램에서 유일하게 두 루프라인 어디에도 안 닿은 커널이다.**
  DRAM 17.4% · FMA 18.9%로 **107 ms(전체의 5.4%)**를 쓴다. 레이턴시/점유율 바운드.
  032가 shared 산정만 고쳤을 뿐 형상은 손대지 않았다. **남은 것 중 가장 큰 여유.**

## Next action
우선순위: **`attention_heads`**(107 ms, 양쪽 루프라인 여유) > `layer_norm_rows`
3패스(−38 ms 상한, 비트 동일성 제약) > 나머지 없음.

---

# EXP-034: attention, kv head를 공유하는 q head 4개를 한 블록으로

## 1. Background
측정-I가 지목한 `attention_heads`(107 ms). 측정-I는 DRAM 17.4% / FMA 18.9%만 보고
"레이턴시/점유율 바운드"로 적었는데, `--set full`로 다시 재니 **틀린 진단이었다**:

| | 값 |
|---|---|
| achieved occupancy | **98.1%** |
| Mem Pipes Busy / SM throughput | **77.5%** |
| Issue Slots Busy | 55.5% |
| 실행 명령 | **819 M** |
| stall(long scoreboard/barrier/wait) | 0.43 / 0.26 / 0.10 |

점유율은 이미 만점이고 stall도 평범하다. 포화된 것은 **L1/LSU 파이프**(77.5%)이고,
곧 **명령 수 자체**가 벽이다. 줄여야 할 것은 메모리 명령이다.

## 2. Hypothesis
`QH/KVH = 4`. q head 4개가 **바이트 단위로 동일한 k·v 행**을 읽는데, 지금은
`blockIdx.y = qh`(16)라서 4개 블록이 같은 스테이징과 같은 v 로드를 4번 반복한다.
이 4개를 한 블록으로 묶으면 k 스테이징·v 로드·chain 로드가 1/4이 된다.

부수 효과: 이 입력은 `max_len = 32`라 `nk ≤ 32`인데 블록은 128스레드다. 점수 루프
`for (i = tid; i < nk; i += 128)`는 **4워프 중 1워프만** 일한다. 묶으면 독립 내적이
`nk` → `4*nk`가 되어 워프가 채워진다.

## 3. Design and Implementation
`src/model.cu` `attention_heads`. grid `(total, 16)` → **`(total, 4)`**, 블록은 128 유지.
- shared: `sq[4][129]`(+1 패드는 g만 다른 레인의 뱅크 충돌 제거), `sdenom[4]`,
  `sw[nk][4]`, `sk[nk][33]`. base 2080 B + 148 B/key → nk=32에 **6.8 KB**(이전 4.9 KB
  × 4블록 = 19.5 KB).
- 점수 인덱스는 `i = ki*4 + g`. `g = i % 4`, `ki = i / 4`가 시프트/마스크로 떨어진다
  (`g*nk + ki`로 잡으면 런타임 `nk` 나눗셈이 붙는다).
- max/denom 직렬 스캔은 `tid < 4`가 head별로 하나씩 → 같은 워프라 4개가 1개 값.
- 값 누산: 스레드가 v 원소를 **1번 읽어 레지스터에서 4 head가 재사용**.

**비트 동일성**: 내적은 여전히 한 스레드 안에서 d 오름차순, denom도 한 레인 안에서
ki 오름차순, 값 누산도 스레드 안에서 ki 오름차순. exp는 원소별. 순서를 바꾼 곳이 없다.

## 4. Result
b5, n=1024, 각 3회(편차 < 0.5 ms).

| | Elapsed | attention_heads 1회 | 명령 | global ld |
|---|---|---|---|---|
| baseline (b6c9fab) | 1.9597 s | 3.34 ms | 819 M | 63.2 M |
| EXP-034 | **1.8979 s** | **1.37 ms** | 377 M | 16.6 M |
| | **−61.9 ms (−3.16%)** | **−59%** | −54% | **−74% (≈1/4)** |

global load가 예측대로 정확히 1/4이 됐다. 커널 총합 107 → 44 ms이고 end-to-end 감소
−62 ms와 일치한다. 검증 `-v` **PASSED, max abs diff 0.000179291 — baseline과 동일 수치**
(라우팅이 한 토큰도 안 바뀜). `-n 1 -d -v`도 PASSED.
점유율 98.1% 유지, SM throughput 77.5% → **86.9%**.

## 5. Next action
SM 86.9%면 이제 파이프 바닥에 가깝다. 남은 것: 값 누산 루프가 스레드마다
`sw[i]/denom`을 **매 key 나눗셈**한다(스레드 전원이 같은 값을 중복 계산).
루프 앞에서 shared에 한 번만 나눠두면 값은 그대로고 나눗셈이 `nk`배 준다 → EXP-035.

---

# EXP-035: 값 누산 루프의 나눗셈을 밖으로

## 1. Background / 2. Hypothesis
034의 값 누산은 `w = sw[i*4+g] / sdenom[g]`를 **key마다, 스레드마다** 한다. 같은
`GROUP*nk`개 몫을 128스레드가 전부 중복 계산한다. 루프 앞에서 shared에 한 번만
나눠두면 같은 피연산자의 같은 IEEE 나눗셈이라 값은 그대로고 나눗셈 수가 `nk`배 준다.

## 3. Design and Implementation
`sdenom` 확정 뒤 `for (i = tid; i < nw; i += D) sw[i] = sw[i] / sdenom[i % GROUP];`
+ `__syncthreads()` 한 번. 누산 루프는 `sw[i*GROUP+g]`를 그대로 곱한다.

## 4. Result
b5, n=1024, 3회.

| | Elapsed |
|---|---|
| EXP-034 | 1.8985 s (1.8998 / 1.8976 / 1.8982) |
| EXP-035 | **1.8954 s** (1.8977 / 1.8945 / 1.8940) |
| | −3.1 ms (−0.16%) |

**노이즈 경계선 수준**이다(런 내 편차 2~4 ms). 방향은 일관되고 일이 순수히 줄어드는
변경이라 유지하지만, 이 크기의 변경으로는 더 가져올 게 없다는 증거로 읽는다.
`-v` PASSED, max abs diff 0.000179291 (동일).

## 5. Next action
034 이후 SM throughput 86.9% = L1/LSU 파이프가 벽. 내적이 스레드당 shared load를
**256회**(`kr[d]` 128 + `qr[d]` 128) 낸다. `float4`로 읽으면 64회가 된다. 패딩을
4의 배수(sk 33→36, sq 129→132)로 바꿔 정렬만 맞추면 덧셈 순서는 그대로다 → EXP-036.

---

# EXP-036: 내적의 shared 로드를 float4로

## 1. Background
034 직후 프로파일: SM throughput 86.9%, FMA 23.5%, DRAM 26%. 포화된 것은 여전히
**L1/LSU 파이프**다. 내적 루프가 스레드당 shared load를 **256회** 낸다
(`kr[d]` 128 + `qr[d]` 128, 청크 4 × 32).

## 2. Hypothesis
청크 32개 float은 shared에서 연속이다. `float4` 4개로 읽으면 같은 값을 같은 d
오름차순으로 쓰면서 로드 명령이 1/4이 된다. 덧셈 순서는 손대지 않는다.

## 3. Design and Implementation
정렬만 맞추면 된다. 패딩을 4의 배수로 바꿨다: `sk` 33 → **`ATTN_KSTRIDE = 36`**,
`sq` 129 → **`QSTRIDE = 132`**. 둘 다 32의 배수가 아니라 뱅크 분산은 유지된다
(`ki*36 % 32 = ki*4`로 워프 안 8개 ki가 전부 다른 뱅크, `g*132 % 32 = g*4`도 동일).
`sk` 시작 오프셋 `4*132 + 4 + 4*nk`는 4의 배수라 float4 정렬이 성립한다.

## 4. Result
b5, n=1024, 3회.

| | Elapsed | attention_heads 1회 | 명령 | shared ld |
|---|---|---|---|---|
| EXP-035 | 1.8954 s | — | — | 40.8 M |
| EXP-036 | **1.8868 s** | **0.964 ms** | 235 M | **15.5 M** |
| | −8.6 ms (−0.45%) | | | −62% |

`-v` PASSED. **출력 131 MB가 034 이전(b6c9fab) 빌드와 바이트 단위로 동일**(`cmp`)하다.

## 5. Next action — attention은 여기서 닫는다
`attention_heads` 누적: **3.34 → 0.964 ms, 32회 합 107 → 30.8 ms**.
end-to-end 누적 **1.9597 → 1.8868 s (−72.9 ms, −3.72%)**.

남은 235 M 명령 중 **~96 M(40%)이 내적의 `__fmul_rn`/`__fadd_rn` 자체**다. 이건
비트 동일성이 요구하는 순서 때문에 줄일 수 없다(d 오름차순 · 스레드 1개). 나머지
global ld 16.5 M도 k 스테이징과 v 로드로, 034에서 이미 1/4로 깎았다. SM 75.4% /
FMA 23.5%에서 더 짜내도 상한이 전체의 1.6%다.

다음 표적은 측정-I가 2순위로 지목한 **`layer_norm_rows` 3패스**(−38 ms 상한)뿐이다.

---

# 측정-J: layer_norm의 3패스는 줄일 수 없다 (산술 증명)

측정-I가 "행을 레지스터에 상주시켜 3패스를 없앤다(−38 ms 상한)"를 후보로 적었다.
**틀렸다.** 상한을 바이트로만 계산하고 **직렬 누산 체인의 비용**을 빼먹었다.

전제(실측): `rows = 15583`, `cols = 4096` → 한 행 16 KB, 전체 255 MB.
1콜 = 읽기 766 MB + 쓰기 255 MB = 1.019 GB, **88.3% DRAM peak, 1.266 ms**.

## 왜 상주가 불가능한가

**레지스터**: 레인 r이 행 r을 j 오름차순으로 누산해야 하므로 행 r은 **레인 r의
레지스터**에 있어야 한다 = 4096개. 하드 상한 255개. **32배 부족.** 끝.

**Shared**: SM당 100 KB → 상주 가능한 행 = **6.25개**. SM당 처리할 행 = 15583/82 = 190.
행 하나의 mean은 종속 `__fadd_rn` 4096개 = 4 cycle × 4096 = **16,384 cycle**이고,
워프는 자기 안에 상주한 행만 커버한다.
  (190 / 6.25) × 16,384 × 2패스 = **996K cycle = 738 µs**
이건 저장하려던 DRAM(510 MB = 619 µs)보다 크다. 게다가 6행 = 96 KB = **SM당 1블록**이라
겹칠 상대가 없어 stage+compute+store가 직렬화된다 → **~1.4 ms, 지금(1.266)보다 나쁘다.**

**L2**: 두 패스가 L2에서 만나려면 동시 워킹셋 ≤ 6 MB인데, 88% DRAM을 내려면
~490 워프가 동시에 돌아야 하고 그 워킹셋은 490 × 32행 × 16 KB = **245 MB**다. 비율 2.4%.

## 지금 커널이 왜 옳은가
워프 하나가 **32개 독립 체인**(레인당 1행)을 돌린다. 같은 16,384 cycle로 32행을
처리하므로 콜당 연산은 ~50 µs, 그래서 DRAM 바운드(88%)다. 상주는 이 32배를 버린다.

## 남은 레버는 하나뿐: 바이트
읽기 3회 중 **3번째(에필로그)를 소비자에게 넘기면** 콜이 절반이 된다 → EXP-037.

---

# EXP-037: post_norm의 3번째 패스를 소비자에게 넘긴다

## 1. Background / 2. Hypothesis
측정-J: 상주는 불가능하고 남은 레버는 바이트뿐이다. 3번째 패스
(`y = (x-m)*inv*w + b`)는 **x를 한 번 더 읽고 y를 한 번 더 쓴다** = 콜당 510 MB,
전체의 절반. 이걸 소비자가 대신 하면 norm은 2패스로 끝난다.

`post_norm`의 소비자는 딱 둘이고 **둘 다 감당할 수 있다**:
- `gemm_gate`: `out = 16`이라 컬럼 타일이 1개 → **A 타일을 정확히 1번만 스테이징**한다.
  즉 원소당 에필로그 1회.
- `gather_rows`(expert): 91.5% DRAM 바운드 → ALU는 공짜.

`input_norm`은 **안 된다**. 소비자가 q/k/v GEMM이고 `PROJ_BN=128`, N=2048/512/512라
A를 **24번** 스테이징한다. 에필로그가 원소당 24회 = 24.5 G op가 20.6 ms/layer짜리
커널들에 얹힌다(+12.5%, ≈+82 ms) — 아끼는 21 ms보다 크다. 산술로 기각.

## 3. Design and Implementation
- `layer_norm_rows<bool STATS_ONLY>`: true면 3번째 패스 대신 `rmean[row]`,
  `rinv[row]`만 쓰고 리턴. `DeviceNorm::forward_stats()`.
- `gather_rows<bool NORM>` / `gemm_gate<bool NORM>`: `(nmean, ninv, nw, nb)`를 받아
  읽어들인 값에 에필로그를 적용. gate는 A 스테이징 지점에서, 범위 밖 행은
  `nmean[a_row]`이 범위를 벗어나므로 `a_ok` 가드.
- `generate()`: `post_norm.forward_stats(d_attn, ...)`, gate와 expert gather는
  `d_norm` 대신 **`d_attn`**을 읽는다. d_norm은 이제 input_norm 전용.

**비트 동일성**: 연산이 `__fsub_rn → __fmul_rn → __fmul_rn → __fadd_rn`으로 동일하고
순서도 동일하다. 바뀐 건 그 fp32를 DRAM에 왕복시키지 않는다는 것뿐(왕복은 항등).

## 4. Result
b5, n=1024, 3회. 커널 계측은 ncu, 1레이어 창.

| | Elapsed |
|---|---|
| EXP-036 | 1.8868 s |
| EXP-037 | **1.8693 s** |
| | **−17.5 ms (−0.93%)** |

| 커널 | before | after |
|---|---|---|
| `layer_norm_rows` (post_norm) | 1262.8 µs / 1.019 GB / 88.5% | **599.2 µs / 0.511 GB / 93.6%** |
| `gemm_gate` | 594.2 µs | 596.5 µs (**+2.3**) |
| `gather_rows` (expert) | 1222 µs | 1224 µs (**+2**) |
| `layer_norm_rows` (input_norm) | 1264.5 µs / 88.4% | 1266.1 µs (변화 없음) |

바이트가 예측대로 정확히 절반이 됐고, 소비자 비용은 사실상 0이었다(gate의 ALU가
+25% 늘 거라 본 예상은 빗나갔다 — 이 커널은 DRAM 47%에 issue 여유가 있었다).
**출력 131 MB가 034 이전(b6c9fab) 빌드와 바이트 단위로 동일**(`cmp`). `-v` PASSED.

## 5. Next action
`layer_norm_rows` 총합 **81 → 60 ms**. 남은 60 ms는 input_norm 32콜(1.266 ms, 88.3%)이고
위 산술로 닫혔다. 다만 2패스판이 **93.6%**를 내는데 3패스판이 88.3%인 차이가 남아
있다(에필로그 패스의 읽기/쓰기 혼합). 그 격차를 다 메워도 −3.3 ms다.

세션 누적(034~037): **1.9597 → 1.8693 s, −90.4 ms (−4.61%)**.

---

# EXP-038: expert gather를 w13 GEMM의 A 스테이징으로 접는다

## 1. Background
037의 전수조사에서 드러난 것: **`gather_rows`(expert)가 레이어당 1.22 ms, 총 39 ms**를
쓴다. 측정-I의 "비-GEMM 전수조사"가 통째로 빠뜨린 커널이다. 하는 일은 순수 복사
(d_attn의 행을 패킹 순서로 d_expert_in에 511 MB 읽고 511 MB 쓰기, 91.5% DRAM).

## 2. Hypothesis
`gemm_grouped`(w13)는 이 버퍼를 A로 읽으면서 **어차피 행 단위로 스테이징**한다.
A의 행 주소를 `arowmap[r]`로 간접화하면 gather 커널 자체가 사라진다. 출력 쪽에는
이미 같은 간접화가 있다(`gemm_grouped_combine`의 `rowmap`). 037의 norm 에필로그도
같은 자리에서 적용하면 된다.

대가: A는 컬럼 타일 수(896/128 = **7회**)만큼 스테이징되므로 에필로그가 원소당 7회
= 3.6 G op. w13의 114 G FMA 대비 +3.2%.

## 3. Design and Implementation
- `gemm_nt_body`에 템플릿 `bool AGATHER = false` + `(arowmap, nmean, ninv, nw, nb)`.
  `a_src = AGATHER && a_ok ? arowmap[a_row] : a_row`로 AGATHER=false면 원래 식으로
  접힌다(투영 5개의 SASS 보존이 목적 — 아래 실측으로 확인).
- `gemm_grouped<bool AGATHER>`가 이를 전달. `generate()`에서 gather 호출 삭제,
  `gemm_grouped<true>(d_attn, ..., d_index, d_nmean, d_ninv, w, b)`.
- `d_expert_in`은 이제 lm_head gather 전용이라 `batch` 행으로 축소(디바이스 메모리 −450 MB).

**비트 동일성**: gather가 쓰던 fp32를 레지스터에서 그대로 만든다. 연산·순서 동일.

## 4. Result
b5, n=1024, 3회.

| | Elapsed | seq/s |
|---|---|---|
| EXP-037 | 1.8693 s | 547.9 |
| EXP-038 | **1.8462 s** | **555.0** |
| | **−23.1 ms (−1.24%)** | |

커널(ncu, 1레이어):

| 커널 | before | after |
|---|---|---|
| `gather_rows` (expert) | 1224 µs / 1.02 GB | **삭제** |
| `gemm_grouped` | 13,015 µs / 1.38 GB | 13,790 µs / 1.98 GB (**+775**) |
| `gemm_nt_bias` ×3, `_resid` | 13506 / 3445 / 3439 / 13658 | 13576 / 3445 / 3451 / 13677 |

투영 4개는 노이즈 범위 안에서 그대로다 → **AGATHER=false 인스턴스는 스케줄을 유지**했다.
`gemm_grouped`의 DRAM이 1.38 → 1.98 GB로 **늘었다**: 읽는 고유 데이터는 오히려 절반
(511 → 255 MB)인데, 행이 흩어져서 7개 컬럼 타일 간 L2 재사용이 나빠졌다. 그래도
삭제한 1.02 GB가 더 크다.

`-v` PASSED, `-n 1 -d -v` PASSED, **출력 131 MB가 034 이전 빌드와 바이트 단위 동일**.

## 5. Next action
세션 누적(034~038): **1.9597 → 1.8462 s, −113.5 ms (−5.79%)**.
b5 **555.0 seq/s**, b7 **626.0 seq/s**.

남은 비-GEMM은 input_norm 40 ms(측정-J로 닫힘)과 attention 30 ms(EXP-036에서 닫힘)뿐.
레이어당 60 ms 중 **55 ms가 GEMM**이고 FMA 피크는 w13 65.7% / 투영 69~70%다.
다음은 GEMM 본체 말고는 없다.

---

# 측정-K: GEMM 비-FFMA 33%의 분해 (ablation) — k 루프 바닥 판정의 재확인과 근거 교체

## 1. Background
"왜 600에 못 가는가"는 k 루프를 **명령 구성비 하나로만** 판정했다(1147 중 FFMA 1024 =
89.3%). 그런데 같은 커널의 실측 FMA 피크는 70.5%다. 두 숫자 사이 19포인트가 어디로
가는지는 기록이 없다. 그 19포인트가 GEMM에 남은 마지막 미측정 구간이다.

(주의: 워크트리를 `origin/main`에서 만들면 HEAD가 EXP-015 시점이다. 그 빌드로 재면
`bs`가 EXP-017 이전 형태라 LDS당 5.0 wf / 2.0 conflict가 나온다 — 아래는 전부
**09dd693**에서 다시 잰 값이다.)

## 2. 항등식: FMA% = 발행률 × FFMA 명령비
`gemm_nt_bias`(q_proj, grid 16×122), ncu, n=1024:

| | |
|---|---|
| `sm__pipe_fma_cycles_active` | **70.55%** |
| `smsp__issue_active` | **78.48%** |
| `inst_executed_pipe_fma / inst_executed` | 4,115,141,120 / 4,597,174,432 = **89.51%** |
| 78.48% × 89.51% | **70.25%** (실측 70.55%) |

**FMA 피크는 발행률과 FFMA 명령비의 곱이고, 노트가 바닥이라 판정한 것은 뒤 항뿐이다.**
앞 항 78.5%(= 발행 공백 21.5%)는 분해된 적이 없고, 이쪽이 1.27배다.
포화된 파이프는 없다: LSU wavefront 66.3%, shared만 57.1%, ALU 3.9%. 즉 처리량이
아니라 레이턴시다. 점유율은 32.8%(2블록 × 8워프).

## 3. Ablation
`bench.cu` = `src/model.cu` 95–358행을 **그대로 복사**(baseline SASS가 출하 커널과 동일:
FFMA 1024 / LDS.128 64 / STS 32 / LDG.E.128 8). 변형마다 텍스트 한 곳만 바꾼다
(결과값은 틀려도 되고 명령 스트림 비교가 목적). q_proj 형상, 20회 평균, 동일 노드:

| 변형 | ms | Δ |
|---|---|---|
| baseline | 11.458 | — |
| shared store 제거 | 10.047 | **−12.3%** |
| global prefetch 제거 | 10.446 | −8.8% |
| shared load를 전부 broadcast로(명령 수 동일) | 10.691 | −6.7% |
| k 루프 barrier 제거 | 11.046 | −3.6% |
| **FFMA만 (shared·global 전부 제거)** | **7.776** | **−32.1%** |

- 네 항의 합 3.602 ms가 전체 차이 3.682 ms와 2% 안에서 일치 → **거의 가산적**이다.
- `ffma_only` 7.776 ms = 4.0936e9 FFMA 워프명령 ÷ (82 SM × 4 스케줄러) = 1.248e7 사이클,
  곧 **1.605 GHz에서 발행률 100%**. 부하 중 실측 클럭(1620 MHz)과 일치한다.
  → **FFMA 스트림 자체에는 여유가 0이고, 33%는 전부 메모리 기구다.**

## 4. 닫은 문 5개 (전부 이번에 직접 실측)
1. **`cp.async`는 전치를 없애 줘도 여전히 느리다.** 측정-F가 이미 기각했지만 그건
   *현재 레이아웃 그대로*의 4 B 복사 8개 형태였다(전치 탓). 여기서는 레이아웃을
   m-major로 바꿔 **16 B 복사 4개**라는 cp.async가 원래 잘하는 형태로 재봤다 —
   스테이징만 떼어내 LDG+STS **0.915** / `cp.async` **1.160**(+27%) /
   전치 없는 STS.128 **0.803** ms. GA102에는 GA100의 우회 경로가 없다.
   즉 측정-F의 기각은 전치 때문만이 아니라 **sm_86에서 cp.async 자체가 이득이 없어서**다.
2. **k-inner(공유를 글로벌과 같은 m-major로 두고 STS.128) 는 2.75배 느리다.**
   31.47 대 11.38 ms. FFMA 1024·LDS.128 64로 명령 수가 같고 STS는 32→8로 **줄었는데도**
   그렇다. 원인은 레지스터다: 피연산자를 k 4개씩 들어야 해 `float4 a4[8]+b4[8]` = 64개
   (현재 16개), 128 상한에서 **spill 72 B**.
   → **전치 스토어는 실수가 아니라 레지스터 예산을 사는 값이다.** 노트가 단언만 하고
   근거를 안 적었던 지점이 이걸로 메워진다.
3. **TM/TN을 16×4로 바꿔도 shared wavefront가 같다.** 워프·p당 12 wf / 64 FFMA로 동일
   (A가 broadcast여도 LDS.128당 2 wf가 하한). 이득 0.
4. **512스레드 8×4(점유율 32.8%→65.6%)는 35% 느리다.** 15.60 대 11.51 ms, 역시 spill
   (64레지스터에 stack 32 B). shared 트래픽 +33%·LDS 명령 +50%가 워프 2배를 잡아먹는다.
5. **L2 스레드블록 스위즐: 이득 없음.** q·o·lm_head 세 형상 모두 +5.9~6.6%인데 그룹
   크기와 무관한 상수 페널티라 스케줄링이 아니라 codegen 교란이다 → 정확히는 "이득의
   증거 없음". 기본 x-major 순서가 이미 동시 상주 164블록을 10×16의 거의 정사각 패치로
   만든다.

## 5. 부수 실측 두 가지
- **b5/b7 격차는 코드가 아니라 클럭이다.** 부하 중 SM 클럭 — 빠른 노드 **1620 MHz @
  356 W**(상한 370 W, 유휴 1905 MHz로 **전력 상한에 닿아 있다**), 느린 노드 **1395 MHz @
  240–280 W**. 1620/1395 = 1.161배 대 실측 처리량 630/556 = 1.133배.
  같은 빌드가 빠른 노드에서 이미 **625.7~629.9 seq/s**를 낸다("600 못 감"은 노드 한정 진술).
- **전문가 타일 패딩은 노트가 맞다.** 그리드 치수로 세면 w13 6.0% / w2 11.1%지만 빈
  타일이 즉시 return하므로 과다계상이다. 실행된 `inst_executed_pipe_fma` 기준으로는
  **w13 3.4% / w2 6.4%**로 노트의 2.61% / 6.0%와 같은 자리다.

## 6. 판정
GEMM은 커널 시간의 **93.3%**다(nsys 무직렬화: GEMM 1,517 ms / 커널 1,626 ms / 벽시계
1,636 ms). 유효 FLOP 21.1 TFLOP/s = 실클럭 피크(1.62 GHz → 34.0 TFLOP/s)의 **62%**.

33%의 네 항 중 셋은 하한에 닿아 있다 — 스토어는 전치의 대가(§4.2), `bs`의 4 wf는
512 B 고유 데이터의 하드웨어 하한, `as`의 2 wf는 broadcast 하한. **남은 여유는 barrier
3.6% 하나이고, 이건 블록/SM을 2→3으로 올려야 먹는데 smem 40,960 B와 레지스터 125가
각각 독립적으로 정확히 2에서 막는다.** 그 이중 구속을 푸는 세 경로(레지스터를 줄이는
k-inner, 워프를 늘리는 512스레드, 스테이징 레지스터를 없애는 cp.async)는 전부 재서
느렸다.

**결론은 노트와 같다 — k 루프에 레버는 없다. 다만 근거가 "명령 구성비 89.3%"에서
"33%의 항별 분해 + 탈출 경로 4개의 실측 기각"으로 교체된다.**

## Next action
GEMM 안에는 없다. 남은 6.7%(비-GEMM)이거나, 128×128/8×8이라는 설계점 자체를 바꾸는
것인데 §4.2/§4.4가 레지스터 예산이 그 설계점을 결정한다는 것을 보였으므로 후자도 닫혔다.
재현: `docs/microworld/meas-k-*`. ablation은 `meas-k-bench-head.cu` + `src/model.cu`
95–358행 + `meas-k-bench-tail.cu`를 이어 붙여 `bench.cu`를 만든 뒤 `meas-k-ablate.py <변형>`
(§3 표의 이름)로 한 곳씩 바꿔 빌드한다. 나머지는 단독 `.cu`다.
(ncu는 `TMPDIR`을 개인 디렉터리로 두지 않으면 `/tmp/nsight-compute-lock` 권한 때문에
`InterprocessLockFailed`로 죽는다 — 측정-I와 같은 함정.)
