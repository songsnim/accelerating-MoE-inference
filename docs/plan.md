# 실험 계획

기준: EXP-008 완료 시점 **3.670 s / 279.0 seq/s** (제출 기록, b0급 노드).
근거: `exploration-003.md`(2위 라인 대조) + EXP-008 ncu 실측.

## 순서

| # | 실험 | 기대 | 누적 | 근거 |
|---|---|---|---|---|
| 009 | **Attention Q-head 4-way** — GQA group의 query head 4개를 thread 0–3에 배정, `queries`/`scores`를 head-minor로 저장(bank 분리) | −0.22 s | 3.45 / 297 | 저쪽 390.6→112.0 ms(3.49x)를 **거의 동일한 출발 비용**(우리 363 ms)에서 실측. 이식 신뢰도 최상, ~20줄 |
| 010 | **MoE W1/W3 fusion + grouped GEMM** — N=896(=7×128)으로 융합, expert를 `blockIdx.z` | −0.37 s | 3.08 / 332 | ncu 실측: 블록 40~72 / 164 슬롯(24~44% 점유), N=448에 BN=128이라 **12.5% 패딩 낭비**. EXP-008이 만든 moe 퇴행(1.259→1.398) 상환 |
| 011 | **Embedding row memcpy + OpenMP + lazy host storage** | −0.16 s | 2.92 / 351 | 순수 host, 수치 위험 0. 망가진 `t_embed` 타이머 동시 수정 |
| 012 | **Prefix trie** — 19,803 → 15,583 노드 (−21.3%) | −0.6~0.7 s | ~2.3 / ~440 | 단일 최대 레버, 정밀도 타협 0(exact CSE). 커널이 안정된 뒤 인덱싱을 얹어야 롤백 비용 최소 |
| 이후 | fused residual+LayerNorm(순서 유지) −0.10 · device-side routing −0.07 · Q/O `cp.async`+XOR swizzle −0.13 · K/V projection fusion | | ~2.0 / ~500 | |

## 하지 않는 것

| 항목 | 사유 |
|---|---|
| **Tensor Core (전 범위)** | 정확도 사망. split 3항·4항 모두 0.29916 FAILED이고 **4항을 넣어도 오차 불변** — 원인은 표현 절삭이 아니라 WMMA accumulator 연산 순서라 operand splitting으로 회생 불가. 합법 범위(lm_head + layer 31)는 실측 15 ms(0.35%) |
| **CUDA Graph / launch 수 감소** | 독립 측정 2건 동일 결론. 저쪽 610 gap 합 2 ms(0.05%), 우리 3969 gap 합 8 ms |
| **feature-major 레이아웃 전환** | ncu 실측: sectors/request 14.5~15.8(이상 16), dram_throughput 3.7~10.8%. 메모리 경로에 고칠 것 없음. 대가는 전 커널 재인덱싱 |
| **attention key 방향 warp 병렬화** | 저쪽이 n=1024에서 실측, **이득 없음(잡음 수준)**. 작동하는 축은 key가 아니라 query head — 내가 랭킹에 올렸던 항목이 틀렸음 |
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
- 저쪽이 기법 6개 앞서 있는데 총시간은 우리가 5% 빠른 이유가 미해명. 합성 이득이 단순
  가산보다 클 가능성과 어느 한쪽 측정에 오해가 있을 가능성이 둘 다 열려 있다.
