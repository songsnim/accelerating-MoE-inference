# EXPLORATION-002: 편견 없는 650 seq/s 탐색

목표 650 seq/s = elapsed **1.575 s**. 현재 262.9 seq/s (3.93 s)의 2.47x.
방법: 서브에이전트 1개에게 `git archive 57ffaeb`로 추출한 **미수정 스켈레톤 1,270줄만** 주고
`/home/n8/aps17/project` 접근을 금지(내 구현/실험로그/roofline 분석/후보목록 전부 차단). `inputs.bin`·`model.bin` 헤더는 허용.
에이전트가 직접 작성·실행한 마이크로벤치 4개: `/home/n8/aps17/.mwscratch/{sgemm,g2,g3,g4}.cu`.

에이전트에게 준 crux: `40.9 TFLOP / 1.575 s = 26.0 TF/s = FP32 peak의 73%` — 그것도 GEMM 외
**모든 것이 0초**라는 가정 하에. 손으로 쓴 SGEMM 천장은 50~60%. "좋은 FP32 GEMM으로는 도달 불가.
더 근본적인 무언가가 양보돼야 한다. 해결하라."

---

## 1. 에이전트의 중심 연역 — 그리고 그것이 이미 실측으로 답이 나와 있었다는 점

에이전트의 논증:
- 호스트 CPU는 `Xeon E5-2650 0`(Sandy Bridge-EP) → **FMA 없음**(`/proc/cpuinfo`에 `fma` 플래그 부재).
  Makefile은 `-ffast-math`를 주지 않으므로 gcc는 FP reduction을 재결합하지 않는다.
  레퍼런스 `matmul_transposed`(`tensor.cu:57`)는 `vmulss`+`vaddss`로 **곱과 합을 따로 라운딩**.
- device에서 이를 bit-exact로 흉내내면 MAC당 2개 FP32 명령 → 유효 FLOP 천장이
  82×128×1.74 GHz = **18.3 TF/s** (36.5의 절반).
- 이 문서의 모든 작업량 감소(31.5 TFLOP)를 다 해도 peak 100%에서 1.723 s = **594 seq/s**,
  현실적 60%에서 ~360 seq/s. **그런데 누군가 600+를 냈다.**
- ⇒ **FFMA 수준의 편차는 허용된다.** (추측이 아니라 모순 증명)

**전제는 맞다. 결론은 추론할 필요가 없었다.** 내 커널 SASS:

```
$ cuobjdump -sass obj/model.o   # gemm_nt_bias
  256 FFMA   ← TM·TN·BK = 4·4·16, inner loop 전체
   16 FADD   ← bias/epilogue만
    0 FMUL   ← 단독 곱셈 없음
```

nvcc가 `acc += av4[i]*bv4[j]`를 전부 FFMA로 contract했고 **그 코드가 n=1024 PASSED,
max abs diff 1.79e-04** (임계값 3e-3의 6%, 마진 16.7x). `lab-note.md:199`에 EXP-002 때 이미 기록됨.

즉 40.9 TFLOP의 모든 MAC이 호스트와 다른 라운딩 경로를 타면서 **633,696개 라우팅 결정이 전부 일치**한다.
에이전트가 "프로젝트 최대 리스크 · 실패 0.2~0.6개 예상 · 동전 던지기"라 한 항목은 실측 0개.
에이전트 §6의 리스크 3개(FFMA 안전성, Nk 캘리브레이션, answers.bin lineage)가 여기서 해소된다.

**단, 에이전트를 정정해야 하는 지점:** diff가 `0.0`이 아니므로 `answers.bin`은 이 CPU 빌드
산출물이 아니다. 에이전트의 판정표대로면 "정밀도 예산이 모델보다 훨씬 느슨함"이지만,
EXP-004가 layer-norm tree reduction(δ≈2e-7)으로 **4/1024 실패, max diff 0.212**를 실측했다.
lineage와 무관하게 **reduction order 민감성은 실재**한다. 느슨해진 것은 FFMA 한 칸뿐이고,
에이전트의 결론(`split-K·tree reduction 금지, k 오름차순 단일 누산기`)은 그대로 유효하다.

## 2. 정밀도 사망 선고의 정량화 (신규 · 유용)

router logit 절대오차 기준(상대오차 아님)으로 평가. logit ≈ 0.3 = √4096·|w·h| ⇒ |w·h| ≈ 4.7e-3.
`Nk`(knife-edge 결정 수) ≈ 20,000은 EXP-004의 4-실패 데이터점에 캘리브레이션.

| 경로 | δ_abs on logit | 예상 실패 |
|---|---|---|
| FFMA vs 레퍼런스 mul+add | ~3e-8 | 0.2~0.6 → **실측 0** |
| split-K / tree reduction | ~2e-7 | ~4 → **EXP-004 실측 4** ✓ |
| TF32 / FP16 입력 | ~1.5e-4 | **~3000** |
| fp16 3-way split (Ootomo) | ~7e-11 | ~0 |

- **FP16/TF32는 4자릿수 마진으로 사망.** Nk가 100배 틀려도 여전히 사망.
- **fp16 3-way split은 정확도는 안전하지만 산술로 사망**: 71/3 = 23.7 TF/s *peak*,
  GA102 mma 실효 55~70% → 13~17 TF/s. 평범한 SIMT FFMA 실측 21.8보다 **느리다**.
- **텐서코어가 합법인 유일한 구간** = 32개 router 전부의 하류 = `lm_head` + layer-31 종단
  = 0.28 TFLOP = **13 ms = 0.8%**. 만들지 말라는 정량적 근거.

⇒ 2.47x는 정밀도가 아니라 **작업량**에서 나와야 한다.

## 3. EXP-007의 갈라진 결과를 설명하는 메커니즘 (가장 실행가능한 발견)

EXP-007(8×8 타일)은 wide GEMM 1.17x 개선 / MoE 0.81x 퇴행. 에이전트가 두 절반을 분리 측정:

| GEMM | 형태 | 타일 | 에이전트 실측 | 내 현재 |
|---|---|---|---|---|
| qkv | 15583×3072×4096 | 16×8 | **21.8 TF/s** (60%) | ~13.3 |
| o_proj | 15583×4096×2048 | 8×8 BK=16 | **21.8** | ~13.3 |
| MoE, expert별 개별 launch | 448→512 패딩, 64 block | — | **11.2** | ~13.2 |
| MoE, **grouped**: w1‖w3 융합 N=896, expert를 `blockIdx.z` | 1792 block | 16×8 | **20.6** | — |
| lm_head | 1024×32064×4096 | 8×8 BK=16 | 21.7 | — |

**MoE를 grouped로 만드는 것이 큰 레지스터 타일을 쓸 수 있게 하는 전제조건이다.**
expert별 launch는 block이 부족해 큰 타일에서 82 SM을 못 채우고, 그게 EXP-007에서
moe가 1.495→1.836으로 퇴행한 이유. wide GEMM에 1.17x를 준 동일 변경이 MoE에서는 반대로
작동한 것 — 두 결과는 모순이 아니라 **같은 원인의 양면**이었다.

이 60%는 내 shared-BW 천장 모델과 **충돌하지 않는다**: 8×8은 FLOP/shared-byte = 2.00 → 천장 100%,
실측 60%. 4×4는 1.00 → 천장 50%, 실측 41%(천장의 82%). 같은 모델 위의 두 점이다.
내가 EXP-007에서 얻지 못한 15.5→21.8 구간에는 **feature-major A 레이아웃 + shared double-buffering**이
필요하고, 이건 EXP-007보다 훨씬 큰 변경이다. 에이전트도 "22 TF/s는 이 커널 계열의 plateau"라 봤고
(16×8로 넓혀도 이득 없음 → byte bound 아니라 latency/issue bound; shared +4 패딩도 무효),
그 이상은 SASS 수준 register double-buffering이 필요하다고 명시.

## 4. 기타 신규 항목

- **prefix trie 15,583** — 내가 `inputs.bin`을 직접 파싱해 얻은 숫자와 **정확히 일치**(4자 독립 도출 합치).
  causal + 절대위치 RoPE ⇒ p-토큰 prefix 공유 시퀀스는 32개 레이어 전부에서 위치 0..p-1이 bit-identical.
  깊이별 노드: 1 / 96 / 292 / 549 / 757 / 819 …, 깊이 18에서 1:1 수렴. **1.271x, 수치 리스크 0.**
- **layer-31 dead code** (신규, 깔끔): layer 31 출력은 `model.norm`+`lm_head`가 각 시퀀스 마지막 위치만
  읽는다 ⇒ 비종단 14,559 노드는 input_norm+k_proj+v_proj만 필요, q_proj/attn/o_proj/router/MoE 생략.
  **−0.811 TFLOP ≈ −40 ms.** layer 30으로 전파 안 됨(layer 31의 마지막 위치 attention이 layer 30 전체를 소비).
- 에이전트가 검토 후 기각: 추가 dedup(trie가 이미 exact minimum) · suffix 공유(위치 의존, 무효) ·
  sliding window(2047 ≫ 32, 발동 안 함) · w2의 expert-pair K방향 융합(토큰별 expert 쌍이 달라 dense GEMM 불가).

## 5. 예산 — 여기가 정직하게 위험하다

에이전트 총계 **1569 ms = 653 seq/s**. 내부는 GEMM 1479 + **비GEMM 90**.

| 항목 | 에이전트 추정 | 내 실측 |
|---|---|---|
| 64 layer_norm + 32 residual | 45 ms | **167 + 74 = 241 ms** |
| attention | 25 ms | **363 ms** |
| 호스트 구간(embed gather, 324MB zero-fill, H2D, epilogue) | 미계상 | **299 ms** |
| MoE 비GEMM(scatter/gather/silu/memset/호스트 라우팅) | 20 ms | **~430 ms** |

에이전트도 §6에서 "비GEMM 100 ms는 대역폭에서 유도했을 뿐 측정 안 함, 200 ms로 터지면 590 seq/s"라고
스스로 플래그를 단다. **내 실측은 200이 아니라 ~1.3 s다.** 대부분 고칠 수 있는 것들이고
(attention은 FP64 파이프 52개 명령 + 직렬 dot, 호스트 구간은 device 이전) 목표가 원리적으로
불가능한 건 아니지만, **1569 ms는 낙관 끝단이다. 비GEMM을 400 ms로 착지시키면 1879 ms = 545 seq/s.**

또 trie 없이 같은 커널로는 1569 × 19803/15583 = **1994 ms = 514 seq/s**이고,
trie 없이 650을 치려면 26.0 TF/s = peak의 71%가 필요한데 에이전트 측정으로는 없는 성능이다.
⇒ **trie는 통과와 실패를 가르는 차이 그 자체.**

## 6. 구조적 요구사항과 그 비용

에이전트 설계는 activation을 **feature-major `[4096][15583]`(255 MB)**로 전면 전환할 것을 요구한다.
이유: (a) GEMM A-tile 로드 coalescing, (b) layer_norm coalescing — 레퍼런스는 mean/var를
**4096에 대한 오름차순 순차 fp32 합**(`tensor.cu:97`)으로 계산하므로 tree-reduce 불가
⇒ row당 1 스레드 2패스가 강제되고, row-major면 낭비 sector로 8배(≈280 ms), feature-major면 ≈47 ms.
15,583개 독립 체인 ≫ 126K 스레드 슬롯이라 직렬 의존성은 완전히 은닉된다.

**비용:** attention/RoPE/gather/scatter 전부를 동시에 건드린다. CLAUDE.md의 "1개 가설씩"과
"Touch only what you must"와 정면으로 부딪히는 규모 — 한 실험으로 쪼갤 수 없다.

기타 에이전트가 명시한 비타협 규칙(대부분 이미 내 코드에 반영됨):
`__fmul_rn`/`__fadd_rn` 사용 · `1.0f/sqrtf` (never `rsqrtf`) · RoPE의 `pow`/`sin`/`cos`는 double로
계산 후 float 라운딩(glibc는 correctly rounded, CUDA는 2~8 ulp; 8 ulp면 δ_abs≈1.5e-7 ≈ 3 실패) ·
`accurate_exp`는 double · w2 epilogue는 `[T][2][4096]` 슬롯 버퍼에 쓰고 **expert 오름차순으로 합산**,
`atomicAdd` 금지(비결정적 순서 = EXP-004를 죽인 그 2e-7).

## 7. 결론

- **정밀도 경로는 닫혔다** (FP16/TF32 4자릿수 초과, fp16-split은 처리량 열세, 텐서코어 합법 구간 0.8%).
  대신 FFMA는 실측으로 안전 확정 — 에이전트가 최대 리스크로 본 것이 리스크가 아니다.
- **작업량 감소만 남는다**: trie 1.271x (−0.70~0.84 s) + layer-31 prune (−40 ms).
- **커널 측 남은 이득은 grouped MoE GEMM**: EXP-007이 절반만 성공한 이유가 특정됐고,
  누산 순서를 건드리지 않으므로 **bitwise 동일이 합격 기준**이라 검증이 값싸다.
- 650 seq/s는 에이전트 자신의 예산에서도 마진이 0이며, 내 실측 비GEMM을 대입하면 545~590 대다.
  **달성 가능한 목표로 취급할 수 있지만, 비GEMM 전면 재작성(feature-major 전환) 없이는 불가능하다.**
