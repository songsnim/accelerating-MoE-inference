# EXPLORATION-001 — 독립 서브에이전트 3개의 ≥30% 탐색 결과

- 일자: 2026-08-25 (EXP-007 직후, 3.933s / 262.9 seq/s 시점)
- 목표: 3.933s → ≤2.75s (−30%)
- 방법: 서로 다른 렌즈의 독립 에이전트 3개. GPU 잡 실행 금지(클러스터 경합) → 정적 분석 + 산수만.
  후보마다 (메커니즘 / 산수로 유도한 예상 이득 / 비트 동일성 판정 / 가장 싼 확인 방법 / 구현 규모) 요구.
  - **A. Cold-eyes**: 실험 이력에 얽매이지 않은 레버 탐색
  - **B. Adversarial**: 기존 측정·추론의 오류와 맹점 공격
  - **C. Roofline**: 하드웨어 바닥값 1차원리 계산

---

## 1. 직접 검증한 사실 (에이전트 주장 → 본인 확인)

| 항목 | 확인 방법 | 결과 |
|---|---|---|
| prefix 중복 21.31% | `inputs.bin` 직접 파싱 | **확인.** T 19,803 → distinct prefix **15,583**. 1024개 전부 토큰 32010으로 시작. causal pair 230,759 → 215,717 (93.5%). depth별 1024→1/96/292/549/757/819 |
| 조교가 중복 제거를 권장 | `docs/lab-note.md:3` | **확인.** "각 input의 공통된 토큰(e.g. `<\|user\|>`, `<\|end\|>`, `<\|assistant\|>`)에 대한 중복 연산 제거" → "프로그램 로직 변경 금지" 해석 리스크 해소 |
| attention이 FP64 파이프 바운드 | `cuobjdump -sass obj/model.o` | **확인.** `attention_heads`에 FP64 명령 **52개**(DFMA 43 + DADD 6 + DMUL 3). 에이전트 추론값 ~25보다 많음. 소스상 `exp(double)`은 `if (tid == 0)` 안에만 존재 |
| `gemm_nt_bias` 자원 사용 | `cuobjdump -res-usage` | REG 45, SHARED 8704B → **5 blocks/SM, 83% occupancy, device 전체 410 블록 상주** |

세 에이전트가 dedup 숫자(15583, 21.31%, 230759→215717, depth 분포)를 **독립적으로 동일하게** 유도했다.

---

## 2. 유일한 정면 충돌과 판정 — GEMM은 무엇에 묶여 있는가

| | A (Cold-eyes) | B (Adversarial) | C (Roofline) |
|---|---|---|---|
| 병목 | **HBM 96% 포화** (898 GB/s) | **shared BW 캡 50%** | **shared BW 캡 50%** |
| 총 HBM 트래픽 | **2.539 TB** | — | ~264 GB |

A의 공식 `traffic = M·N·K·(1/BM+1/BN)·4`는 **L2 재사용 0 가정**이므로 트래픽 *상한*이다.
공교롭게 두 모델의 시간 예측이 겹친다: A는 2.539TB/936GB/s → 43% of peak, B·C는 캡 50%, 실측 41%.

**판정: shared-memory 대역폭 캡이 맞다.** 근거 3개.

```
FLOP / shared byte = (TM·TN·2)/((TM+TN)·4) = 32/32 = 1.00   (4×4)
shared 총대역 = 128 B/clk × 1.695 GHz × 82 SM = 17.8 TB/s
→ 상한 17.8 TF/s = FP32 peak의 50%.   8×8 → 2.00 FLOP/B → 100%
```

1. B와 C가 서로 다른 표기(FLOP/byte vs FFMA/byte)로 **독립적으로 같은 50%**에 도달.
2. **외부 앵커**: 3090 cuBLAS SGEMM 실측이 17~19 TF/s — 17.8 TF/s가 실제 벽.
3. EXP-007이 8×8(캡 100%)에서 51%를 기록 — 캡이 풀린 뒤에도 occupancy(2 blocks/SM)가 붙잡았다는 설명과 일치.

**실측 재계산: 41.30 TFLOP / 2.827s = 14.61 TF/s = peak의 41.1% = 자기 천장의 82%.**

> **결론: "런타임의 76%가 GEMM이니 여기가 레버다"는 프레이밍은 과대평가였다.**
> GEMM 잔여 headroom은 1.25~1.3× on 2.827s = **0.55~0.65s**.
> EXP-007의 "43.6% → 60%+" 가설은 4×4에서 **산술적으로 불가능**했다.

미해결: A의 HBM 모델이 완전히 틀렸는지는 `ncu`의 `dram__bytes.sum` 1회로 종결 가능(40배 차이라 즉시 갈림). B4 설계가 이 답에 달려 있으나 현재 2:1 + 외부 앵커로 shared-BW 측이 유리.

---

## 3. 기존 작업에서 발견된 오류 4건 (모두 인정)

**3.1 `t_moe`가 혼합 타이머라 "31.2% of peak"은 아티팩트.**
`t_moe`에는 expert GEMM 외에 scatter 76 + gather 52 + silu 8 + memset ~11 = **147ms의 커널**과 host 라우팅 ~45ms가 섞여 있다. 빼면:
```
expert GEMM ≈ 1.257 − 0.147 − 0.045 − 0.005 ≈ 1.06 s → 14.14 TFLOP/1.06s = 13.3 TF/s = 37.5% of peak
wide GEMM   = 26.91 TFLOP / 1.735 s = 15.5 TF/s = 43.6%
교차검증: 1.06 + 1.735 + 0.085 = 2.88 s vs nsys gemm_nt_bias 2.827 s ✓
```
두 호출 지점은 **같은 천장의 75~87%로 거의 같다**. EXP-007의 전제(31.2% vs 43.6%의 큰 격차)가 무너진다.

**3.2 EXP-007에서 타이머 분리를 철회한 논리는 순환논법이었다.**
"쪼갠 답이 무엇이든 다음 행동은 같다"고 했으나, EXP-007을 정당화한 31.2%라는 숫자 **자체가 타이머를 분리하지 않았기 때문에 존재**했다. 분리했으면 아직 미귀속인 ~190ms(4.8%)가 드러났을 것이다. EXP-006의 next action("먼저 쪼개 재라")이 옳았다.

**3.3 "GPU 96.7% busy"와 "299ms zero GPU work"는 서로 모순.**
96.7%는 **커널 span(3634ms)** 기준, 측정 구간(3933ms) 기준으로는 **92.4%**. 두 숫자를 화해시키지 않고 나란히 보고한 것은 오해를 유발한다.

**3.4 `attention_heads` 오진 — 직렬 dot이 아니라 FP64 파이프.**
GA102는 SM당 FP64 레인 2개. `exp(double)` 호출 수 = 230,759 pair × 16 head × 32 layer = **1.181e8회**, 전부 `if (tid == 0)` 안의 의존 체인. SASS에 FP64 명령 52개.
직렬 q·k dot은 전체 ~20ms로 반올림 오차. EXP-006에 쓴 "직렬 dot 때문이지만 8%짜리라 손댈 이유 없다"는 **메커니즘도 판단도 틀렸다**. 수정은 ~10줄.

**3.5 (부가) 이미 비트 동일성이 깨진 곳이 있다.**
`silu_mul`(src/model.cu:177)이 `expf`를 쓰는데 레퍼런스 `tensor.cu`는 `(float)std::exp((double)x)`.
"1e-7이라 3e-3 임계값보다 4자리 아래"라는 정당화는 **EXP-004가 이미 반증한 논리** — silu 출력은 w2 → scatter → residual → 다음 레이어 router로 흘러가 expert 집합을 뒤집을 수 있다. 이 입력에서 633,696개 결정 중 하나도 안 뒤집힌 것은 운이다. **잠재 지뢰로 기록**(다른 입력에서 실패 가능). 유지 권고 — 단 "비트 동일"이라는 표현은 EXP-004 산출물 대비 자기 회귀 테스트일 뿐, 규칙이 아니다.

---

## 4. 3개 에이전트가 독립 합의한 항목

| 항목 | A | B | C |
|---|---|---|---|
| 임베딩 device 이전 + scratch 사전 할당 | −267ms | −230ms | −290ms |
| `attention_heads`가 자기 바닥에서 크게 벗어남 | −180ms | −230ms | −220ms |
| memory-bound 패스 fusion | −185ms | −120ms | −131ms |
| `layer_norm` 142ms가 유일한 비-대역폭 outlier (바닥 45ms) | ✓ | ✓ | ✓ |
| grouped expert GEMM으로 0.67-wave occupancy 기아 해소 | ✓ | ✓ | ✓ |
| atomicAdd scatter는 비트 동일 | ✓ | ✓ | ✓ |
| device routing은 최소 이득·최대 위험 → 맨 마지막 | ✓ | ✓ | ✓ |

**atomicAdd 비트 동일 논증** (3개 독립 도달): scatter 순서가 임의가 되지만 top-2라 **0에 정확히 2개**만 더해지고, IEEE 덧셈은 **교환법칙 성립**(결합법칙만 깨짐)이므로 `0+a+b = 0+b+a` 비트 동일. addend 3개 이상이면 무너지는 논증.

**occupancy 기아 정량화**: expert w1/w3는 `39×7 = 273 블록 = 0.67 wave`(410 블록 머신). 이것이 1697 launch 중 **1024개, 발행 FLOP의 22.7%(≈642ms)**. **출력 병렬성이 구조적으로 부족해 타일 튜닝으로는 못 고치고, 16 expert를 한 launch로 합치는 것만이 해결책.** EXP-007이 역방향(80 블록, 0.81×)으로 이미 증명.

---

## 5. 랭킹 및 경로

| # | 레버 | Δ | 규모 | 비트 동일 | 가장 싼 확인 |
|---|---|---|---|---|---|
| **B1** | **prefix-trie 중복 제거** (T 19,803→15,583) | **−0.70~0.84s (18~21%)** | ~120줄 | 증명 가능 | 이미 확인됨(inputs.bin) |
| **B2** | **attention `exp`를 nk 레인으로 분산** (292→~40ms) | **−0.23s (5.8%)** | **~10줄** | 자명 | `attn` 타이머 + `cmp` |
| **B3** | 임베딩 테이블 device + arena 할당 | −0.23~0.29s (6~7%) | ~35줄 | memcpy | `t_embed`/`t_h2d` |
| B4 | grouped expert GEMM + shape별 타일 | −0.45s | ~60줄 | EXP-007에서 증명 | `moe` 타이머 교차 A/B |
| B5 | norm 동시성 + fusion + device routing | −0.12~0.19s | ~100줄 | 조건부 | 커널별 microworld |

```
3.933  현재
3.09   B1 후                          1.27x
2.86   B2 후
2.63   B3 후                          1.49x   ← 목표 2.75s 통과 (−33%)
2.18   B4 후
2.06   B5 후                          1.91x
```

**B1+B2+B3만으로 −33%. GEMM 커널을 전혀 건드리지 않는다.** 세 항목 모두 EXP-000~007에서 한 번도 의심하지 않은 것: **행 개수 / attention의 진짜 병목 / GPU 작업이 없는 7.6%**.

### 계산된 하드웨어 바닥

| 구간 | 절대 바닥 | 현실적 hand-written 바닥 | 실측 |
|---|---|---|---|
| `gemm_nt_bias` (41.30 TF 발행) | 1160ms | 1657ms (70% peak) | 2827 |
| `attention_heads` | 14ms | 50ms | 292 |
| norm+resid+gather+scatter+silu+memset | 107ms | 200ms | 351 |
| 임베딩 gather + 324MB H2D + malloc/free | 5ms | 10ms | 299 |
| lm_head + 131MB logits D2H | 25ms | 40ms | 85 |
| **합** | **1311ms (3.0×)** | **1957ms (2.0×)** | **3933** |

목표 2.75s는 현실 바닥의 1.40× 위 — **비트 동일성을 포기하지 않고 도달 가능**. 위 5개 중 비트 동일이 아닌 항목은 없다.

### B1 메커니즘과 위험

causal attention은 자기 시퀀스의 0..i만 읽고 RoPE는 시퀀스 내 위치로만 결정된다. 따라서 길이 p 접두사를 공유하는 두 시퀀스는 **32개 레이어 전부에서 위치 0..p−1의 hidden row가 비트 단위로 동일**하다. `[19803,4096]` → `[15583,4096]`, trie node당 1행.
- 모든 row-wise 연산(norm, 5개 projection, gate, gather/silu/scatter, residual add 2개, rope)은 **무변경**, 행 수만 21.3% 감소.
- attention만 "`[base, base+qi]`에 attend" → "내 조상 경로에 attend"로 변경. `int path[15583][32]` (2MB, host에서 O(T)로 생성, H2D 0.17ms). 조상은 depth 오름차순 = 레퍼런스의 `ki` 오름차순과 동일 순서.
- `lm_head`는 1024개 leaf row를 gather (1024개 시퀀스 전부 distinct → 전부 distinct leaf).
- 이득 배분이 균일하지 않음에 주의: row-wise는 ×0.787, **attention은 ×0.935** (중복 제거되는 위치는 key가 적은 *싼* 행들). dedup을 attention 이득으로 팔지 말 것.
- **미모델링 비용 1개**: 조상의 K/V 행이 더 이상 연속이 아니라 attention의 L1/L2 locality가 나빠진다. 시퀀스당 K+V는 ≤ 32×512×4 = 65KB로 L1에 들어가므로 큰 문제는 아닐 것이나 확인 필요. trie node를 DFS preorder로 배치해 완화.
- 구현을 2단계로 쪼개 위험 분리: ① flat 19,803행 유지 + attention만 `path[][]` 인덱싱 → `cmp` 비트 동일 확인 ② 병합 켜기 → `cmp` 재확인.

### B2 메커니즘

`fmaxf`는 순서 무관(비 NaN)이라 트리 리덕션 가능. 각 `exp`는 `sw[i]`의 독립 함수이므로 `for (i = tid; i < nk; i += D)`로 최대 32 레인에 분산하고, **합산만 thread 0에 남겨 `i` 오름차순 유지**. 같은 FP64 명령 수를 32 레인이 나눠 실행 → 의존 체인 길이가 `nk×chain` → `chain`으로 붕괴. 비트 동일: 동일 입력 → 동일 `exp` 호출, 덧셈 순서 불변.

---

## 6. 근거와 함께 기각된 레버

| 기각 항목 | 근거 |
|---|---|
| **kernel launch 개수 줄이기** | 3427 launch, CPU 15.9ms(0.40%), 커널 간 갭 median 0.83µs / 합 8.0ms. **상한 8ms.** grouped GEMM의 "16회 launch 오버헤드 제거"라는 동기는 근거 없음(진짜 이유는 occupancy) |
| **weight 스트리밍 대역폭** | 전체 weight 14.492 GB / 936 GB/s = **15.5ms = 0.39%**. expert MoE는 AI 142 FLOP/byte(크로스오버 38)로 명확히 compute-bound, headroom 2.8× |
| **타일 패딩 낭비** | 프로그램 전체 **1.02%**. gate GEMM의 N=16 vs BN=64 4× 낭비도 0.25 TFLOP ≈ 17~23ms뿐. N=448은 7×64로 낭비 0 |
| **`add_inplace`/`gather`/`scatter`/`silu`/`rope` 커널 튜닝** | 이미 자기 대역폭 바닥의 **88~94%**. 안을 고쳐서 얻을 것 없음 — 트래픽 자체를 삭제(fusion)해야만 줄어듦 |
| **저정밀도 (TF32/FP16)** | TF32는 GA102에서 FP32와 동일 throughput → **no-op**. FP16은 2×지만 **모든 FP16 GEMM이 어떤 router의 상류**에 있고(`route` 입력은 `post_norm(attn+hidden)`) FP16 상대오차 ~1e-3은 `ROUTER_SCORE_QUANTUM` 자체와 같은 크기. 32개 router 전부의 하류는 레이어 31 expert + `lm_head`뿐 ≈ 125ms → **완전 안전한 FP16은 ≤1.6%**. 부분 적용 구제책 없음 |
| **layer_norm 트리 리덕션** | EXP-004에서 1024개 중 4개 실패, max abs diff 0.212로 실측 반증. 순서를 유지한 채 **동시성만** 6→16 blocks/SM로 올리는 것만 허용 |

---

## 7. 미해결 / 다음 측정

1. **`ncu` 1회로 GEMM 바운드 종결** — `dram__bytes.sum`. shared-BW 모델(264GB) vs HBM 모델(2.539TB)은 40배 차이라 즉시 갈린다. B4 설계가 여기 달려 있다.
2. **(layer, expert)별 실제 row 수 분포** — 정적으로 유도 불가. 16개 expert 전부가 32개 레이어 전부에서 행을 받는다는 것은 launch 수 `32×(5+3×16)+1 = 1697`(nsys 실측과 일치)로 **증명됨**. 평균은 2T/16 = 2475이나 분산은 미지. host `fprintf` 한 줄, GPU 비용 0. B4 사이징의 유일한 미지 입력.
3. **B1의 K/V locality 영향** — trie 조상 경로가 비연속이 되는 비용. `attention_heads` 단독 타이머로 확인.
4. **`t_moe` 타이머 분리** — EXP-007에서 잘못 철회했다. ~190ms가 아직 미귀속.

## 8. 방법론 (모든 후속 실험에 적용)

노드별 GPU 클럭이 **1.19× 고정 차이**(b5/b6 gemm 1.72 vs b1 2.11). B2·B3·B5는 각각 5~6% 효과로 **노드 간 편차보다 작아 allocation을 넘나들면 측정 자체가 불가능**하다. 모든 A/B는 단일 `srun` allocation 안에서 교차 실행할 것. B1은 21%라 이 규율 없이도 측정 가능.
