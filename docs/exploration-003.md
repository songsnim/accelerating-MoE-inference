# EXPLORATION-003: 외부 독립 라인 대조 (`Prefill 18 → 247`)

## 0. 출처와 접근 조건

Artifact `0073677f-d439-41a2-84b6-b1c23348c4f5` (제목 `Prefill 18 → 247`).
동일 과제(Phi-tiny-MoE-instruct · RTX 3090 · prefill, n=1024, warmup 없음, `main.cpp` 측정구간·검증 그대로)의
**완전히 독립된 최적화 라인**. 제출 이력에 `2위 / 8명` 기록.

- `read`/`WebFetch`는 공개 링크 열람 게이트로 차단 → 콘텐츠 호스트
  (`…frame.claudeusercontent.com/_f/1787642387-b3ae/`)를 직접 받아 텍스트 추출.
- 저쪽 커밋 `03cd2ac` / `74d70e5` / `1a0b3a4`는 **이 저장소 object store에 없음**.
  이 머신 전체에서 `src/model.cu`는 우리 저장소와 `.baseline-skeleton` 둘뿐.
- ⇒ **별도 체크아웃이며 코드는 이 머신에 없다. 기법과 판정만 이식 가능.**

## 1. 현재 위치 대조 (둘 다 노드 b0)

| | 우리 | 저쪽 |
|---|---|---|
| n=1024 | **3.933 s / ~260 seq/s** | 4.133 s / 247.76 seq/s |
| 최초 대비 | — | 14.1x (18.17 → 247.76) |
| 검증 | PASSED (max abs 1.79e-04) | PASSED (max abs 0.00369644) |

우리 소스 grep: `cp.async` 0건, `wmma`/`USE_TC` 0건, `route_row` 2건(host 왕복 잔존).
**두 라인의 기법 집합이 거의 서로소다.** 저쪽이 커널 엔지니어링 6개 항목 앞서 있는데도
총시간은 우리가 5% 빠르다 = 곱해질 여지가 크다는 신호.

저쪽 궤적: 18.17(프로젝트 이전) → 58.463s/17.52 → 7.048/145.28 → 6.767/151.32 →
6.331/161.75 → 5.402/189.54 → 4.133/247.76.
우리 궤적: 0.128 → 0.217 → 20.99 → 42.46 → 161.4 → 260.0.

저쪽 예산 분해(b6, elapsed 4321 ms): GPU busy 3796 ms(87.9%, **그중 84%가 GEMM**) /
kernel 사이 gap 210 ms(중 200 ms는 첫 GEMM 앞 단일 gap = attention weight H2D + memory pool 확장,
나머지 610개 gap 합 2 ms) / kernel span 밖 host 315 ms.

## 2. 가져올 기법 6개

우리 b1 프로파일을 b0로 스케일(×0.862): gemm 1.814 · moe 1.289 · attn 0.313 · norm 0.144 ·
resid 0.064 · lm_head 0.072 · h2d 0.021 · 미계상 host 0.213 (= elapsed 3.933).

| # | 기법 | 저쪽 실측 | 우리 예상 | 근거·조건 |
|---|---|---|---|---|
| 1 | **Attention Q-head 4-way** — GQA group의 query head 4개는 value combine 전까지 독립. thread 0이 score/softmax를 4개 연달아 계산하던 것을 thread 0–3에 하나씩 배정하고 `queries`/`scores`를 **head-minor**로 저장해 4 thread가 서로 다른 shared bank에 접근 | 390.6 → **112.0 ms (3.49x)**, 전체 −4.3% | attn 0.313 → 0.090 → **−0.22 s** | 출발점 390.6 ms ≈ 우리 363 ms(b1) → 이식 신뢰도 최상. head별 누적은 `d=0..127` 단일 thread 순차 FP32 FMA 그대로 → **출력 binary bitwise 동일이 합격기준** |
| 2 | **Embedding gather를 row memcpy + OpenMP** + **lazy host storage**(GPU가 전부 덮어쓰는 임시 Tensor는 host `vector<float>` 할당·0채움 생략) | **−1.04 s (−19.5%)**, 출력 IDENTICAL. lazy host storage는 별도로 −49 s | 미계상 host 0.213 → ~0.05 → **−0.16 s** | 저쪽 출발점은 `Tensor::at()` 원소별 호출(약 8500만 원소, 원소마다 `ensure_host()` + rank·범위검사). 우리는 이미 raw 원소접근이라 여지가 그만큼은 아니다. **망가진 `t_embed` 타이머도 같이 고쳐짐** |
| 3 | **MoE W1/W3 pair fusion + SiLU epilogue + GPU work queue 기반 grouped GEMM** | 코드에 살아 있음 | **−0.2~0.30 s** | 우리 EXP-008 후보 (A) 그 자체. 저쪽이 이미 작동 확인 |
| 4 | **Q/O 2-stage `cp.async`** — global load와 현재 K tile 계산 overlap, shared B에 4-float chunk **XOR swizzle** | **−187 ms** | **−0.13 s** | EXPLORATION-002 서브에이전트가 15.5→21.8 TF/s에 필요하다던 double-buffering의 구체형 |
| 5 | **LayerNorm exact-order shared staging** (coalesced 적재 후 평균·분산 누적은 **기존 단일 thread, 동일 인덱스 순서 유지**) + **fused residual add + LayerNorm** | 코드에 살아 있음 | norm+resid 0.208 → ~0.11 → **−0.10 s** | 순서 유지가 조건 — EXP-004가 증명한 제약 |
| 6 | **Device-side deterministic top-2 routing** (양자화·tie-break 규칙 그대로 유지하고 host 왕복만 제거) + `gather_compact`/`combine` kernel | 코드에 살아 있음 | **−0.07 s** + host sync 32회 제거 | |

기타 저쪽 보유·우리 미보유: **K/V projection fusion**(A tile 한 번 적재해 K/V accumulator 공유,
bias를 store에 fusion → FP32 projection 호출 124→93회) · attention **score 3중 재계산 제거**
(max·softmax denom·weighted-sum이 같은 내적을 3번 계산) · **cooperative K staging**(128 thread coalesced) ·
GQA grouping(KV head 1개에 query head 4개를 한 block) · packed RoPE · `add_bias` fusion.

단순 가산 **−0.94 s → 2.99 s / 342 seq/s**. 여기에 prefix trie(×0.787, 토큰비례 작업에만)
적용 시 **~2.4 s / ~430 seq/s**. 합성 손실 + trie 인덱싱 오버헤드 감안한 현실선
**2.4–2.9 s / 350–425 seq/s**.

## 3. 우리 분석을 정정하는 것 3개

**A. Tensor Core는 죽었지만 EXPLORATION-002 서브에이전트가 든 이유가 틀렸다.**
서브에이전트 판정: "fp16 3-way split(Ootomo)은 δ_abs≈7e-11로 **수치적으로 안전**, 처리량으로만 사망".
실측은 반대다 — split-BF16을 attention 전체로 확대하면 **0.29916 FAILED**이고,
**네 번째 `Alo·Blo` 항까지 넣어도 오차가 그대로였다**. 원인은 표현 절삭이 아니라
**WMMA FP32 accumulator의 연산 순서 차이**이며 operand splitting으로 고칠 수 없다.
저쪽이 다음 후보로 둔 compensated TF32도 같은 이유로 실패할 가능성이 크다.
⇒ TC는 **정확도로 사망**, 어떤 split 기법으로도 회생 불가.
(저쪽이 실제로 성공시킨 TC 범위: `lm_head.weight` + `model.layers.31.self_attn.{q,k,v,o}`
= 1697개 중 5개, **이득 약 15 ms / 0.35%**. EXPLORATION-002가 예측한 "모든 router의 하류만 합법,
0.8%"와 독립 일치.)

**B. 제 attention 병렬화 축이 틀렸다.**
제가 랭킹에 올려둔 "B2: attention `exp`를 nk lane에 분산"을 저쪽이 정확히 시도했고
(`key 방향 warp 병렬화, 4 warps`, n=1024) 결과는 **이득 없음 — 측정 잡음 수준**.
작동하는 축은 key가 아니라 **4개 GQA query head**다. 실패 실험 하나를 그대로 절약.

**C. 정확도 마진은 내가 말한 것보다 좁고 노드 의존적이다.**
저쪽 통과 제출의 max abs가 **0.00369644** — abs 임계 3e-3을 **넘고 relative로 통과**한다.
게다가 **동일 binary가 노드에 따라 0.00174713 vs 0.000102997(17배)** 를 보고한다.
내가 EXPLORATION-002에 쓴 "1.79e-04 = 예산의 6%, 마진 16.7배"는 안정된 양이 아니다.
올바른 규율은 저쪽이 명시한 것이고 우리도 이미 따르고 있다 —
**연산 순서 변경 여부는 오차값이 아니라 출력 binary `cmp`로 판정.**
추가로 **n=64 통과가 n=1024 통과를 보장하지 않는다.**

## 4. 우리 작업을 독립 검증해준 것 2개

- **EXP-004 병렬 LayerNorm reduction 실패값: 우리 0.211975, 저쪽 0.211973.**
  서로 다른 코드베이스에서 같은 실험이 같은 숫자로 죽었다. 상호 검증.
- **CUDA Graph / launch overhead 사망 확정.** 저쪽: `cudaLaunchKernel` 620회 = 11.0 ms,
  작은 gap 약 610개 합 = **2 ms = 0.05%** ("Graph가 없앨 수 있는 건 그 2 ms뿐").
  우리: 3427 call = 15.9 ms = 0.40%, ≤100µs gap 3969개 = 8.0 ms.
  독립 측정 2건, 동일 결론. **이 레버는 두 번 죽었다.**

## 5. 저쪽이 이미 폐기한 것 = 우리가 하지 않아도 되는 실험

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

## 6. 우리만 가진 레버 — 저쪽이 선언한 벽을 정확히 부순다

저쪽 결론(§06 남은 병목): *"kernel 밖 시간을 전부 없애도 상한이 약 3.80초 / 253 seq/s.
GPU busy의 84%가 GEMM이므로 그 이상은 GEMM kernel 자체를 빠르게 하는 방법밖에 없다.
즉 나머지 1692개 GEMM의 Tensor Core 전환이다. 그리고 그 병목은 속도가 아니라 정확도다."*

**이 상한 계산은 토큰 수를 19,803으로 고정했을 때만 성립한다.** prefix trie는 그 수를
15,583으로 바꾼다(−21.3%, causal + 절대위치 RoPE ⇒ p-토큰 prefix 공유 시퀀스는 32개 레이어
전부에서 위치 0..p−1이 bit-identical, 즉 exact CSE). 저쪽 자기 숫자에 대입:

```
GPU busy 3796 ms × 84% = GEMM 3189 ms
        → 3189 × 0.787 = 2510 ms      (−679 ms)
253 seq/s 상한 → 약 305 seq/s
```

⇒ **"남은 레버는 TC뿐"은 거짓.** 빠진 레버는 정밀도 타협이 **전혀 필요 없는** 유일한 것이고,
저쪽이 추격 중인 레버(TC)는 자기 데이터가 이미 "split으로 못 고친다"고 말하고 있다.
저쪽 라인에 부족한 것은 이 구조적 항목 하나뿐이다.

## 7. 측정 규율 (저쪽 §07, 우리 EXP-007 결론과 동일)

- **노드 편차가 개선폭보다 크다.** 동일 binary가 n=64에서 0.381 s vs 0.590 s.
  성능 비교는 반드시 **하나의 `srun` 할당 안에서 번갈아** 실행.
- 연산 순서 변경 여부는 **출력 binary `cmp`**로 판정(오차값 아님).
- 최종 판정은 반드시 **n=1024**.

## 8. 권고 — EXP-008 순서 변경

기존 승인 요청은 (A) grouped MoE GEMM이었으나, **attention Q-head 4-way를 먼저** 칠 것을 권한다.

| 순서 | 실험 | 기대 | 이유 |
|---|---|---|---|
| EXP-008 | **Attention Q-head 4-way** | −0.22 s | 3.49x가 **거의 동일한 출발 비용(390.6 vs 363 ms)에서 실측**됨 → 목록 중 이식 신뢰도 최상. 10~20줄. bitwise 동일이 값싸고 명확한 합격기준. 다른 모든 항목과 독립 → 롤백 위험 최소. **줄당 이득 최고** |
| EXP-009 | Embedding row memcpy + OpenMP + lazy host storage | −0.16 s | 순수 host, 수치 위험 0. `t_embed` 타이머 버그 동시 해결 |
| EXP-010 | MoE W1/W3 fusion + grouped GEMM | −0.2~0.30 s | 기존 후보 (A). 저쪽이 작동 확인 |
| EXP-011 | **prefix trie** | −0.6~0.8 s | 단일 최대 레버. 앞 3개로 커널이 안정된 뒤 인덱싱을 얹는 것이 롤백 비용상 유리 |
| 이후 | Q/O `cp.async` + XOR swizzle, device-side routing, K/V projection fusion | −0.13 / −0.07 / ? | |

**Tensor Core와 CUDA Graph에는 시간을 쓰지 않는다** — 각각 독립 측정 2건으로 사망 확정.

## 9. 미해결

- **저쪽 체크아웃 접근 가능 여부.** 이 머신에는 없다. 접근 가능하다면 6개 커널을 우리 쪽에
  재구현하는 것보다 **trie를 저쪽 코드에 이식하는 편이 훨씬 저렴하다** — 저쪽은 이미 6개 항목
  앞서 있고 우리가 추가할 것은 trie 하나뿐이다. 접근 불가면 위 순서대로 우리 쪽에서 재구현.
- 저쪽이 6개 항목 앞서 있는데 총시간은 우리가 5% 빠른 이유가 완전히 해명되지 않았다.
  (저쪽 `USE_TC` 빌드 낭비는 clean 빌드 4133 ms에는 포함되지 않는다고 명시돼 있다.)
  두 라인을 합치면 단순 가산보다 더 큰 이득이 있을 가능성과, 어느 한쪽 측정에 오해가 있을
  가능성이 둘 다 열려 있다.
