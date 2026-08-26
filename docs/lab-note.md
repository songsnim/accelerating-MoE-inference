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
