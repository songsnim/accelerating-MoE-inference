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
  baseline은 M=19라 32스레드 중 19개만 일했다. 배치화로 M=T가 되어 스레드가 포화된다.
  -> throughput이 n에 의존하므로 리더보드 제출은 2분 제한에 들어가는 최대 n으로 해야 한다.
  n=8(62s)이 여유 있는 최대치. n=12는 250토큰 x 0.37 = 94s + load 10s로 2분 제한에 너무 붙는다.
  이득도 평탄해지는 중(0.086 -> 0.128 -> 추정 0.135)이라 n=8로 제출하고 다음 레버로 넘어간다.
- **제출: 0.128473 seq/s (n=8), baseline 0.033306 대비 3.86x**
- 130회 attention 재계산 제거분은 전체의 13%(1.15x)뿐. 2.88x의 주된 원인은 스레드 포화다.

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
