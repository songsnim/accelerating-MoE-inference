# 실험 계획

기준: EXP-013 완료. b0 기준 **2.307 s / ~444 seq/s**.
제출 기록은 EXP-012 시점 **408.4 seq/s**.
근거: `exploration-003.md`(2위 라인 대조) + EXP-008 ncu 실측.

## 순서

| # | 실험 | 기대 | 근거 |
|---|---|---|---|
| ~~009~~ | **Attention 직렬 `exp` 분산** (완료 `cd3cdf6`) | **−0.109 s 실측** | attn 0.291 → 0.183. 직렬 구간을 통째로 삭제한 하한이 0.178 s → 가용분의 96% 회수, 이 방향 소진. bitwise 동일 |
| **010** | **MoE W1/W3 fusion + grouped GEMM** — N=896(=7×128) 융합, expert를 `blockIdx.z` | −0.2~0.37 s | ncu 실측: 블록 40~72 / 164 슬롯(24~44% 점유) + N=448에 BN=128이라 **12.5% 패딩 낭비**. EXP-008이 만든 moe 퇴행(1.259→1.398) 상환. 누산 순서 불변 → `cmp` 판정 |
| ~~011~~ | **Prefix trie** (완료 `47aff75`) | **−0.628 s 실측** | 19,803 → 15,583 행. elapsed 3.281 → 2.653 (b5), bitwise 동일. attn은 `qi` 직렬 루프 소멸로 −24.7% |
| **012** | **expert w13 grouping** — (expert, rowtile) work queue | −0.19 s 이상 | 측정-A: w13 시간가중 wave 충전율 66%, 53%가 1 wave 미만 launch. trie로 expert당 행이 21% 줄어 충전율은 더 나빠졌다 |
| ~~013~~ | **embedding lookup 디바이스 이전** (완료) | **−0.23 s 실측** | embed 0.181 + h2d 0.023 → 0.002 + 0.004. bitwise 동일. trie 구성 자체는 2 ms로 확정 |
| ~~014~~ | **출력 Tensor 할당 겹치기** (완료 `9ccf876`) | **−0.068 s 실측** | `Tensor` 생성자가 131 MB를 `assign(0.0f)`로 페이지폴트 채우는 0.069 s. GPU 의존 없음 → 별도 스레드. bitwise 동일 |
| ~~015~~ | **device-side routing** (완료 `0db90c5`) | **−0.071 s 실측** | 호스트 왕복 0.060 + memset 8.2 GB 제거. moe 0.780→0.698, d2h 0.005→0.002. bitwise 동일. gemm은 boost 클럭으로 +0.012 |
| **016** | GEMM 타일 재설계 (projection) | ? | q/o_proj 16.2 TF/s / 35.6 peak. gemm 1.09가 남은 최대 항목 |
| 017 | GQA grouping + coalesced K staging | −0.05 s | 남은 attn 0.143 s. thread i가 key row `chain[i]`를 직접 읽어 **uncoalesced**(연속 thread 간격 `KVH·D`) |
| 이후 | LayerNorm coalesced staging + fused residual −0.10 · K/V projection fusion 소액 | | |

## 하지 않는 것

| 항목 | 사유 |
|---|---|
| **Q/O `cp.async` 2-stage** | 현재 타일이 `as[BK][BM]` **전치** 레이아웃이라 `cp.async`(연속 복사)로 채울 수 없다. 전치를 버리면 내부 루프 shared 로드가 `LDS.128` 4개 → `LDS.32` 16개. 타일 재설계와 묶어야 한다 |
| **Tensor Core (전 범위)** | 정확도 사망. split 3항·4항 모두 0.29916 FAILED이고 **4항을 넣어도 오차 불변** — 원인은 표현 절삭이 아니라 WMMA accumulator 연산 순서라 operand splitting으로 회생 불가. 합법 범위(lm_head + layer 31)는 실측 15 ms(0.35%) |
| **CUDA Graph / launch 수 감소** | 독립 측정 2건 동일 결론. 저쪽 610 gap 합 2 ms(0.05%), 우리 3969 gap 합 8 ms |
| **feature-major 레이아웃 전환** | ncu 실측: sectors/request 14.5~15.8(이상 16), dram_throughput 3.7~10.8%. 메모리 경로에 고칠 것 없음. 대가는 전 커널 재인덱싱 |
| **attention key 방향 warp 병렬화** | 저쪽이 n=1024에서 실측, **이득 없음(잡음 수준)** |
| **attention Q-head 4-way (3.49x)** | 적용 대상 없음. 저쪽 3.49x는 GQA grouping으로 head 4개를 한 블록에 묶으며 생긴 페널티를 되돌린 것 — 우리는 묶지 않아 그 페널티가 없다. exploration-003 §2가 스스로 철회, EXP-009 실측이 독립 확인 |
| **logits D2H를 pinned로** | 측정-C. 131 MB D2H 15 ms를 pinned로 줄이려면 (a) `cudaHostRegister` 13 + 5.8 ms 또는 (b) pinned ring + 청크 파이프라인. 둘 다 실이득 **10~12 ms(0.5%)**인데 `cudaHostAlloc`이 0.83 ms/MB, 청크 GEMM·비블로킹 스트림·이벤트·memcpy 스레드가 붙는다. `Tensor`가 `std::vector`라 할당기를 못 바꾸는 게 근본 제약 |
| 병렬 LayerNorm reduction · warp dot reduction · 병렬 product+순차 sum | 전부 FP32 누적 순서 변경 → MoE top-2 routing 붕괴. EXP-004(0.211975)와 저쪽(0.211973) 상호 검증 |

## 측정 규율

- **A/B는 하나의 `srun` 할당 안에서 교차 실행.** 노드 편차 1.19x가 개선폭보다 크다.
- **연산 순서 변경 여부는 오차값이 아니라 출력 binary `cmp`로 판정.** 저쪽 통과 제출이
  max abs 0.00369644(abs 임계 초과, relative로 통과)이고 동일 binary가 노드에 따라 17배
  차이를 보인다 — 오차값은 안정된 양이 아니다.
- **최종 판정은 n=1024.** n=64 통과가 n=1024 통과를 보장하지 않는다(저쪽 실측).
- 실험 1개 = 가설 1개 = 커밋 1개.

## 미해결

- 저쪽 체크아웃은 이 머신에 없다. 접근 가능해지면 **6개 커널을 우리 쪽에 재구현하는 것보다
  trie를 저쪽에 이식하는 편이 훨씬 싸다** — 저쪽은 6개 앞서 있고 우리가 더할 것은 trie 하나다.
- 남은 attn 0.178 s의 내역(score dot 대 value 누적)은 미분해. MoE/embedding보다 작아 후순위.
- 저쪽이 기법 6개 앞서 있는데 총시간은 우리가 5% 빠른 이유가 미해명. 합성 이득이 단순
  가산보다 클 가능성과 어느 한쪽 측정에 오해가 있을 가능성이 둘 다 열려 있다.
