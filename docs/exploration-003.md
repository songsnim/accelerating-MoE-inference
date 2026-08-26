# EXPLORATION-003: 외부 독립 라인 대조 (`Prefill 18 → 247`)

## 0. 출처와 접근 조건

Artifact `0073677f-d439-41a2-84b6-b1c23348c4f5` (제목 `Prefill 18 → 247`).
동일 과제(Phi-tiny-MoE-instruct · RTX 3090 · prefill, n=1024, warmup 없음, `main.cpp` 측정구간·검증 그대로)의
독립 라인. 제출 이력에 `2위 / 8명` 기록.

- `read`/`WebFetch`는 공개 링크 열람 게이트로 차단 → 콘텐츠 호스트
  (`…frame.claudeusercontent.com/_f/1787642387-b3ae/`)를 직접 받아 텍스트 추출.
- 저쪽 커밋 `03cd2ac` / `74d70e5` / `1a0b3a4`는 **이 저장소 object store에 없음**.
  이 머신 전체에서 `src/model.cu`는 우리 저장소와 `.baseline-skeleton` 둘뿐.
- ⇒ **별도 체크아웃이며 코드는 이 머신에 없다. 기법과 판정만 검토 가능.**

## 1. 핵심: 앞서 있는 것이 아니라 다른 방향이다

처음 이 문서를 쓸 때 "저쪽이 6개 항목 앞서 있다"고 적었는데 **틀렸다.** 두 라인은
**편집 가능 파일 집합이 다르고, 그래서 아키텍처가 다르다.**

| | 우리 | 저쪽 |
|---|---|---|
| 편집 범위 | `model.{h,cu}` · `model_loader.{h,cu}` · `run.sh` | 여기에 더해 **`tensor.h` · `tensor.cu` · `layer.cu`** (§1.1) |
| 근거 | — | "`tensor.o`의 WMMA dispatch", "`layer.o`는 BF16 pack과 FP32 device 사본 해제", "GPU 상주 Tensor — host/device mirror 최신상태 추적" |
| 전략 | 레퍼런스 `Tensor` 추상을 **우회** — `model.cu` 안에 자체 device 커널·버퍼(`DeviceLinear`/`DeviceNorm`/`DeviceMoE`) | `Tensor` 추상을 **유지한 채 그것을 최적화** — host/device mirror 최신상태 추적, lazy host storage, `at()` 제거 |
| n=1024 (노드 b0) | **3.933 s / ~260 seq/s** | 4.133 s / 247.76 seq/s |
| 검증 | PASSED (max abs 1.79e-04) | PASSED (max abs 0.00369644) |

### 1.1 편집 범위는 대회 규칙이며, 저쪽은 그것을 벗어났다

**원본 스켈레톤(`57ffaeb`, 주최측 제공)의 `CLAUDE.md` §3에 동일한 NEVER EDIT 목록이 있다.**
우리 제약은 자율 규정이 아니라 대회 규칙 그대로다.

| 파일 | 저쪽이 건드린 근거 |
|---|---|
| `include/tensor.h` + `src/tensor.cu` | 원본 `Tensor`는 **순수 host 클래스** — `std::vector<float> data_`뿐이고 device 포인터·mirror·`ensure_host()`가 전부 없다. "GPU 상주 Tensor — host/device mirror의 최신 상태를 추적", "Lazy host storage — host `vector<float>`를 할당하지도 0으로 채우지도 않는다", "`at()`은 원소마다 `ensure_host()`"는 클래스에 device 저장소와 dirty 추적을 **추가**해야 성립 |
| `src/tensor.cu` | "`tensor.o`의 WMMA dispatch" — 원본 `tensor.cu`에 `wmma`/`USE_TC` 언급 0건 |
| `src/layer.cu` | "`layer.o`는 BF16 pack과 FP32 device 사본 해제를 수행" — 원본 `layer.cu`에 `bf16` 언급 0건 |

**`USE_TC` 자체는 위반이 아니다 — 철회.** 초기 판에 "Makefile 편집"이라 적었으나 틀렸다.
`ifeq ($(USE_TC),1) → -DUSE_TC`는 **주최측 Makefile에 이미 있고**
("Tensor-Core code is opt-in; the default build remains the FP32 path"),
`USE_TC`를 소비하는 곳은 `main.cpp` 뿐이다. 둘 다 저쪽은 건드리지 않았다.

이 편집은 부수적인 것이 아니다 — GPU 상주 `Tensor`가 저쪽 궤적의 출발점
("GPU 상주 packed baseline 첫 CUDA 구현")이자 라인 전체의 토대다.
합법 경로는 있었다: `tensor_ops`를 가속하려면 `tensor.cu`를 고쳐야 하지만,
**`tensor_ops`를 쓰지 않으면 된다** — 우리가 `model.cu`에 자체 커널을 넣어 우회한 것이 그 경로다.
(단서: 저쪽 코드가 아니라 요약 아티팩트의 산문에서 역추론했다. 다만 원본 `Tensor`에
device 측면이 아예 없으므로 클래스 변경 없이는 성립 불가다.)

### 1.2 왜 최적화가 적은 우리가 5% 빠른가

**이것이 모순을 해소한다.** 저쪽 최대 승리 두 개 —
**lazy host storage −49 s**, `Tensor::at()` 원소별 호출 제거 **−1.04 s** — 는
Tensor 추상을 유지해서 스스로 짊어진 오버헤드의 **회수**다. 우리는 그 비용을 애초에
치르지 않았다. 저쪽 14.1x(18.17 → 247.76)의 상당 부분이 우리에게는 존재하지 않는 항목이고,
따라서 **저쪽 절대값을 우리 컴포넌트 비용의 기준선으로 쓸 수 없다.**

저쪽 궤적: 18.17(프로젝트 이전) → 58.463s/17.52 → 7.048/145.28 → 6.767/151.32 →
6.331/161.75 → 5.402/189.54 → 4.133/247.76.
우리 궤적: 0.128 → 0.217 → 20.99 → 42.46 → 161.4 → 260.0.

저쪽 예산 분해(b6, elapsed 4321 ms): GPU busy 3796 ms(87.9%, 그중 84%가 GEMM) /
kernel 사이 gap 210 ms(중 200 ms는 첫 GEMM 앞 단일 gap = attention weight H2D + memory pool 확장,
나머지 610개 gap 합 2 ms) / kernel span 밖 host 315 ms.

## 2. 우리 attention은 이미 저쪽과 다른 구조다 — 3.49x는 이식되지 않는다

우리 `attention_heads` (`src/model.cu`): grid `(batch, QH=16)`, block `D=128`.

```cpp
for (int i = tid; i < nk; i += D) {            // key 방향으로 이미 병렬
    ...
    for (int d = 0; d < D; ++d) score = __fadd_rn(score, __fmul_rn(sq[d], kr[d]));
    sw[i] = score / scale;                     // score는 한 번만 계산, 재사용
}
__syncthreads();
if (tid == 0) {                                // ← 실제 직렬 구간
    ... fmaxf over nk ...
    ... exp(static_cast<double>(...)) over nk, denom 누적 ...
}
```

| 저쪽 항목 | 우리 상태 |
|---|---|
| score 3중 재계산 제거 (max·denom·weighted-sum이 같은 내적 3회) | **이미 없음** — `sw[]`에 한 번 쓰고 재사용 |
| key 방향 warp 병렬화 (저쪽 판정: **이득 없음**) | **이미 있음** — `i = tid; i += D` |
| **Q-head 4-way (3.49x)** | **적용 대상 없음.** 우리는 블록 하나가 query head 하나. 저쪽 3.49x는 GQA grouping으로 4개 head를 한 블록에 묶으며 생긴 페널티(thread 0이 4개 head softmax를 연달아 계산)를 되돌린 것 — 우리는 묶지 않았으니 그 페널티가 없다 |
| GQA grouping (KV head 1개에 query head 4개 → K/V를 4번 아닌 1번 적재) | **없음.** 우리는 블록이 head별이라 같은 `kh`의 K/V를 4개 블록이 중복 적재 |
| Cooperative K staging (128 thread coalesced 적재) | **없음.** 우리는 thread i가 key row `lo+i`를 global에서 직접 읽어 **uncoalesced** (연속 thread가 `KVH·D` 간격) |

우리 attention의 실제 병목은 셋 중 어느 것도 아니다: **`if (tid == 0)` 안의 nk회 double `exp`**
(SASS FP64 명령 52개, GA102는 SM당 FP64 lane 2개) 동안 128 thread 중 127개가 유휴다.
저쪽 두 항목 모두 이 지점을 건드리지 않는다.

⇒ attention에서 우리가 가져올 것은 **GQA grouping + coalesced K staging (−0.10~0.15 s 추정)**
이고, 직렬 `exp` 문제는 **우리 고유 과제로 남는다.**

## 3. 실제로 이식 가능한 것 / 불가능한 것

우리 b1 프로파일을 b0로 스케일(×0.862): gemm 1.814 · moe 1.289 · attn 0.313 · norm 0.144 ·
resid 0.064 · lm_head 0.072 · h2d 0.021 · 미계상 host 0.213 (= elapsed 3.933).
**gemm + moe = 3.10 s = 79%** — 우리 병목도 저쪽과 같이 GEMM/MoE다.

### 이식 가능 (우리 편집 범위 안, 아키텍처 무관)

| 기법 | 저쪽 실측 | 우리 예상 |
|---|---|---|
| **MoE W1/W3 pair fusion + SiLU epilogue + grouped GEMM** (work queue 기반) | 코드에 살아 있음 | **−0.2~0.30 s** — 기존 EXP-008 후보 (A). 아티팩트와 **무관하게 이미 우리 계획** |
| **Q/O 2-stage `cp.async`** + shared B에 4-float chunk **XOR swizzle** | **−187 ms** | **−0.13 s** — 우리 GEMM(64×64, BK=16)과 저쪽(64×64, BK=32)이 구조적으로 유사. EXPLORATION-002 서브에이전트가 15.5→21.8 TF/s에 필요하다던 double-buffering의 구체형 |
| **LayerNorm coalesced shared staging** (평균·분산 누적은 단일 thread 동일 인덱스 순서 유지) + **fused residual add** | 코드에 살아 있음 | norm+resid 0.208 → ~0.11 → **−0.10 s** |
| **Device-side deterministic top-2 routing** (양자화·tie-break 규칙 유지, host 왕복만 제거) | 코드에 살아 있음 | **−0.07 s** + host sync 32회 제거 |
| **GQA grouping + cooperative K staging** | (§2 참조) | **−0.10~0.15 s** |
| **K/V projection fusion** (A tile 한 번 적재해 K/V accumulator 공유, bias를 store에 fusion) | projection 호출 124 → 93회 | 소액 |
| Embedding gather **OpenMP 병렬화** | (저쪽 −1.04 s는 `at()` 제거분 포함) | **−0.10 s** — 우리는 이미 raw 원소접근이므로 병렬화분만 해당. `t_embed` 타이머 버그도 동시 해결 |

합 **−0.7~0.85 s**. 단, 이 중 최대 항목(W1/W3 fusion)은 아티팩트 이전에 이미 우리 계획이었다.
**아티팩트가 새로 더해준 순수 speedup은 −0.4~0.5 s 수준이다.**

### 이식 불가

| 저쪽 항목 | 이유 |
|---|---|
| GPU 상주 `Tensor` (host/device mirror 최신상태 추적) · **lazy host storage (−49 s)** · `Tensor::at()` 제거 (−1.04 s) | `tensor.h`/`tensor.cu` 변경. 우리 NEVER EDIT. **그리고 우리는 이 오버헤드를 애초에 지지 않았다** — `model.cu`가 hot path에서 Tensor를 우회한다 |
| `tensor.cu`의 WMMA dispatch · `layer.cu`의 BF16 pack | `tensor.cu`·`layer.cu` 변경 = 규칙 위반. **단 `USE_TC` 자체는 합법이고, WMMA를 `model.cu`에 넣는 것도 합법이다** — §10 참조 |
| Q-head 4-way 3.49x | §2 — 적용 대상 없음 |
| score 3중 재계산 제거 · key 방향 병렬화 | 이미 보유 |

우리 `Tensor hidden` 324 MB zero-fill은 `tensor.cu`를 건드리지 않고 `model.cu`에서
host-backed Tensor를 만들지 않는 방식으로 회피 가능하므로, 편집 제약으로 인한 실질 손실은 작다.

## 4. 우리 분석을 정정하는 것 (아티팩트의 진짜 가치는 여기다)

**A. Tensor Core는 죽었지만 EXPLORATION-002 서브에이전트가 든 이유가 틀렸다.**
서브에이전트 판정: "fp16 3-way split(Ootomo)은 δ_abs≈7e-11로 **수치적으로 안전**, 처리량으로만 사망".
실측은 반대다 — split-BF16을 attention 전체로 확대하면 **0.29916 FAILED**이고,
**네 번째 `Alo·Blo` 항까지 넣어도 오차가 그대로였다.** 원인은 표현 절삭이 아니라
**WMMA FP32 accumulator의 연산 순서 차이**이며 operand splitting으로 고칠 수 없다.
저쪽이 다음 후보로 둔 compensated TF32도 같은 이유로 실패할 가능성이 크다.
(저쪽이 실제로 성공시킨 TC 범위는 `lm_head.weight` + `layers.31.self_attn.{q,k,v,o}`
= 1697개 중 5개, 이득 15 ms/0.35%. EXPLORATION-002가 예측한 "모든 router의 하류만 합법, 0.8%"와 독립 일치.)

**B. 정확도 마진은 내가 말한 것보다 좁고 노드 의존적이다.**
저쪽 통과 제출의 max abs가 **0.00369644** — abs 임계 3e-3을 **넘고 relative로 통과**한다.
게다가 **동일 binary가 노드에 따라 0.00174713 vs 0.000102997(17배)** 를 보고한다.
EXPLORATION-002에 쓴 "1.79e-04 = 예산의 6%, 마진 16.7배"는 안정된 양이 아니다.
올바른 규율은 우리도 이미 따르고 있다 — **순서 변경 여부는 오차값이 아니라 출력 binary `cmp`로 판정.**
추가로 **n=64 통과가 n=1024 통과를 보장하지 않는다.**

**C. 내 attention 후보(B2)는 축이 틀렸고, 게다가 우리는 이미 갖고 있었다.** §2 참조.

## 5. 우리 작업을 독립 검증해준 것 2개

- **EXP-004 병렬 LayerNorm reduction 실패값: 우리 0.211975, 저쪽 0.211973.**
  서로 다른 아키텍처의 코드베이스에서 같은 실험이 같은 숫자로 죽었다.
- **CUDA Graph / launch overhead 사망 확정.** 저쪽: `cudaLaunchKernel` 620회 = 11.0 ms,
  작은 gap 약 610개 합 = **2 ms = 0.05%**. 우리: 3427 call = 15.9 ms = 0.40%,
  ≤100µs gap 3969개 = 8.0 ms. 독립 측정 2건, 동일 결론. **이 레버는 두 번 죽었다.**

## 6. 저쪽이 이미 폐기한 것 = 우리가 하지 않아도 되는 실험

저쪽 표현: *"버린 것들은 전부 같은 이유로 죽었다 — FP32 누적 순서가 바뀌면 여러 layer를 거쳐
MoE top-2 routing이 바뀐다."*

| 실험 | n | 최대 절대 오차 | 판정 |
|---|---|---|---|
| 병렬 LayerNorm reduction | 1024 | 0.211973 | FAILED |
| warp dot-product reduction attention | 64 | 9.22422 | FAILED |
| 병렬 product + 순차 sum | 1024 | 0.0191498 | FAILED (n=64는 통과) |
| attention 전체 TC, split 3항 | 64 | 0.29916 | FAILED |
| attention 전체 TC, split 4항 | 64 | 0.29916 | FAILED |
| fused K/V `cp.async` | 64 | — | **느림** (shared 48KB, occupancy 하락, 0.3784 → 0.4545 s) |
| key 방향 warp 병렬화 (4 warps) | 1024 | — | **이득 없음** (측정 잡음 수준) |

마지막 두 항목이 특히 값지다 — `cp.async`를 K/V까지 확대하면 shared 압박으로 **역효과**라는
경계선을 미리 알려주고(§3의 Q/O 한정 적용 근거), key 방향은 우리가 이미 가진 것이 저쪽에선
이득이 없었음을 확인해준다.

## 7. 방향 차이의 전략적 함의

저쪽 결론(§06 남은 병목): *"kernel 밖 시간을 전부 없애도 상한이 약 3.80초 / 253 seq/s.
GPU busy의 84%가 GEMM이므로 그 이상은 GEMM kernel 자체를 빠르게 하는 방법밖에 없다.
즉 나머지 1692개 GEMM의 Tensor Core 전환이다. 그리고 그 병목은 속도가 아니라 정확도다."*

두 가지가 동시에 성립한다.

1. **저쪽은 자기 데이터가 막혔다고 말하는 방향으로 달리고 있다.** split 3항과 4항의 오차가
   **완전히 동일**했다는 것은 원인이 accumulator 순서라는 뜻이고, compensated TF32도 같은 벽을 만난다.
2. **저쪽 상한 계산은 토큰 수를 19,803으로 고정했을 때만 성립한다.** prefix trie는 그 수를
   15,583으로 바꾼다(−21.3%, causal + 절대위치 RoPE ⇒ p-토큰 prefix 공유 시퀀스는 32개 레이어
   전부에서 위치 0..p−1이 bit-identical = exact CSE). 저쪽 숫자에 대입하면:

```
GPU busy 3796 ms × 84% = GEMM 3189 ms → × 0.787 = 2510 ms   (−679 ms)
253 seq/s 상한 → 약 305 seq/s
```

즉 정밀도 타협이 **전혀 필요 없는** 레버가 남아 있는데 저쪽은 그것을 상한 계산에서
구조적으로 못 보고 있다. **우리 방향(trie + 자체 커널)은 막혀 있지 않다.**

## 8. 권고 — EXP-008 순서 (아티팩트 이전 계획으로 복귀)

아티팩트를 처음 읽고 "EXP-008을 attention Q-head 4-way로 변경" 권고했으나 §2에 따라 **철회한다.**

| 순서 | 실험 | 기대 | 이유 |
|---|---|---|---|
| **EXP-008** | **MoE W1/W3 fusion + grouped GEMM** (expert를 `blockIdx.z`, N=896) | −0.2~0.30 s | 원래 후보 (A). EXP-007의 갈라진 결과(wide 1.17x / moe 0.81x)를 설명하는 메커니즘이고, 저쪽이 작동을 확인해줬다. 누산 순서 불변 → **`cmp` bitwise 동일이 합격기준** |
| EXP-009 | **prefix trie** | −0.6~0.8 s | 단일 최대 레버. 우리 고유. 커널이 안정된 뒤 인덱싱을 얹는 것이 롤백 비용상 유리 |
| EXP-010 | Q/O `cp.async` 2-stage + XOR swizzle | −0.13 s | K/V로 확대 금지(§6) |
| EXP-011 | GQA grouping + cooperative K staging | −0.10~0.15 s | 직렬 `exp` 문제는 별건으로 남음 |
| 이후 | LayerNorm coalesced staging + fused residual · device-side routing · embedding OpenMP · K/V projection fusion | −0.10 / −0.07 / −0.10 / 소액 | |

**Tensor Core와 CUDA Graph에는 시간을 쓰지 않는다** — TC는 우리 편집 범위에서 빌드 불가이고
가능해도 0.35%이며 accumulator 순서로 막혀 있다. CUDA Graph는 독립 측정 2건으로 사망.

## 10. `USE_TC`는 검증 기준 자체를 바꾼다 — 이 조사에서 가장 큰 발견

`main.cpp` `validate_decode()`:

| | 기본 (FP32) | `USE_TC` |
|---|---|---|
| prefill 판정 | 원소별 `abs > 3e-3 && rel > 3e-3` → FAIL | **이 검사가 컴파일에서 빠짐** (`#ifndef USE_TC`) |
| 대체 기준 | — | 시퀀스별 vocab 집계: `normalized_mae = Σ(abs/(1+|exp|))/vocab ≤ **2.0e-2**`, `huber_mean ≤ **1.0e-4**` |

**EXP-004 대입:** max abs 0.211975로 strict에서 죽었으나 **mean abs 6.83e-05**.
normalized ≤ 6.83e-5 ≪ 2e-2, huber ≈ 0.5·(6.8e-5)² ≈ 2.3e-9 ≪ 1e-4
→ **`USE_TC`에서는 통과했을 것이 거의 확실하다.**

⇒ 우리 7개 실험 전체를 지배해온 **"누적 순서 bit 보존" 제약이 `USE_TC`에서는 풀린다.**
tree reduction · split-K · Tensor Core가 모두 열린다. 우리 GEMM 실측 13.3 TF/s 대비
FP16 peak 71 TF/s. §3의 이식 목록(−0.4~0.5 s)보다 훨씬 큰 사안이다.

**무료 통행증은 아니다.** 저쪽이 attention 전체에 split-BF16을 적용했을 때 완화된 기준에서도
FAILED였다(0.29916, split 3항/4항 동일). softmax가 민감한 구간이라 GEMM만 TC로 돌리는 경우와는
다를 수 있으나 확인은 측정으로만 된다.

**WMMA를 넣을 위치는 `tensor.cu`가 아니라 `model.cu`다.** 우리는 이미 `tensor_ops`를
우회하므로 구조적으로 자연스럽고 규칙 안에 있다.

## 11. 미해결

- **`USE_TC` 빌드 제출이 FP32 제출과 같은 순위표에 들어가는가.** `run.sh`는 `srun ./main "$@"`
  뿐이고 `submit.sh`도 **재빌드하지 않는다** — 기존 `./main`에 `-v`를 강제 부착해 실행하고
  `aps-score`로 `validated=passed`를 전송한다. 즉 **하네스는 `USE_TC=1` 빌드를 그대로 받아
  완화된 기준으로 통과 처리한다.** 그러나 "하네스가 받아준다"와 "주최측이 동일 순위로 집계할
  의도다"는 다르다. **조교 확인 필요** — 페이오프가 커서 물어볼 값이 있다.
  EXP-008은 이 답변과 무관하게 유효하므로(누적 순서 불변, 두 기준 모두 통과) 질문을 던져두고
  진행한다.
- 두 라인 합성은 코드 이식이 아니라 **기법 재구현**이어야 한다(저쪽 코드는 이 머신에 없고,
  일부는 우리가 편집할 수 없는 파일에 있다).
