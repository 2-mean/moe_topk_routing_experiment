# Sparse32 MoE Top-k Routing: 실험 설정, 결과, 베이스라인 통합 보고서

> 이 문서는 `sparse32_experiment_report.md`와 `sparse32_baseline_analysis.md`를 통합하고,
> 실제 수치와 베이스라인 비교를 포함하는 단일 참조 문서다.
> 2026-06-28 기준, fixed-step/same-compute 8-seed 실험 완료 후 작성.

---

## 1. 연구 질문

> Train-time top-k(k_train)와 inference-time top-k(k_infer)의 불일치가
> (a) MoE routing structure의 어느 정도 divergence를 만드는가?
> (b) validation loss에 얼마나, 어느 방향으로 영향을 주는가?
> (c) 이 효과가 training stochasticity를 초과하는 진짜 k 효과인가?

---

## 2. 모델 및 공통 설정

| 항목 | 값 |
|---|---|
| 아키텍처 | TinyMoE Transformer |
| Layers | 8 |
| d_model / n_heads | 192 / 6 |
| n_experts | **32** |
| expert_hidden | 768 |
| sparse_dispatch | true |
| router_aux_loss_coef | 0.01 |
| seq_len | 96 |
| vocab_size | 256 |
| batch_size | 16 |
| learning_rate | 2.5e-4 |
| weight_decay | 0.01 |
| data_seed / order_seed | 1729 / 31415 |
| 데이터 | 7 카테고리 × 2500 samples (합성 템플릿) |
| Seeds | **8** (0..7) |
| Probe samples/category | 48 |
| 서버 GPU | RTX 3080 Ti 12GB |

---

## 3. 실험 설계

### 3.1 Two-budget design

| 설정 | 이름 | k별 steps | 의도 |
|---|---|---|---|
| **Fixed-step** | `sparse32_kgrid_fixed_step_8seed` | 모든 k에 1500 step | step 수 통제 → k 차이만 비교 |
| **Same-compute** | `sparse32_kgrid_same_compute_8seed` | k×step 보정 | active expert invocation 통제 |

Same-compute step schedule:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| steps | 6000 | 3000 | 2000 | **1500** | 1200 | 1000 | 860 | 750 |

k=4가 두 설정의 공통 기준점 (동일 실험).

### 3.2 8-seed 확장 방법

- Seeds 0-2: `sparse32_kgrid_fixed_step_3seed` / `sparse32_kgrid_mechanism_3seed` (원본)
- Seeds 3-7: 추가 run
- Combined run: symlink로 두 raw run 묶기 → **원본 미수정**
- Summary는 source metrics를 직접 병합해 재집계 (raw route 재계산 없음)

### 3.3 Train/Inference k grid

k_train ∈ {1..8}, k_infer ∈ {1..8} → 64 matched runs per budget × 8 seeds = 512 runs

---

## 4. 측정 지표 및 Sanity Gates

| 지표 | 정의 | Random baseline | Oracle |
|---|---|---|---|
| top1_agreement | 두 모델의 top-1 expert 일치율 | **1/32 = 0.0313** | 1.0 |
| nestedness | small set ⊆ large set 비율 | **b/32** | 1.0 |
| spearman | gate logit ranking correlation | **0** | 1.0 |
| mismatch_delta | loss(k_train=a, k_infer=b) − loss(k_train=a, k_infer=a) | **0** | 0 |
| asymmetry | \|delta(hi→lo) − delta(lo→hi)\| | 0 (symmetric) | 0 |

**Sanity gates (모든 실험에서 통과):**

| Gate | 기준 | Fixed-step | Same-compute |
|---|---|---|---|
| logit cutoff nestedness | = 1.0 | ✅ 1.0 | ✅ 1.0 |
| step-0 same-W₀ nestedness | = 1.0 | ✅ 1.0 | ✅ 1.0 |
| step-0 same-W₀ top1 | = 1.0 | ✅ 1.0 | ✅ 1.0 |
| collapsed final runs | = 0 | ✅ 0/64 | ✅ 0/64 |

---

## 5. B5: Training Variability Null (핵심 베이스라인)

**설계**: 같은 k, 다른 seed로 학습한 모델 간 alignment.
within-k alignment = initialization + optimization stochasticity의 합산.

**결과**: within-k top1 ≈ **random baseline (1/32 = 0.031)**에 수렴.

### Fixed-step 8-seed

| k | within-k top1 | within-k spearman | across-k top1 (mean) | across-k spearman | ratio (top1) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0314 | **0.0007** | 0.2297 | 0.5612 | **7.3×** |
| 2 | 0.0316 | 0.0003 | 0.4105 | 0.6988 | **13.0×** |
| 3 | 0.0317 | 0.0019 | 0.4626 | 0.7345 | **14.6×** |
| 4 | 0.0322 | 0.0012 | 0.4907 | 0.7548 | **15.2×** |
| 5 | 0.0316 | 0.0009 | 0.5040 | 0.7649 | **15.9×** |
| 6 | 0.0317 | 0.0009 | 0.5095 | 0.7686 | **16.1×** |
| 7 | 0.0325 | 0.0000 | 0.5063 | 0.7671 | **15.6×** |
| 8 | 0.0331 | 0.0012 | 0.4961 | 0.7602 | **15.0×** |

### Same-compute 8-seed

| k | within-k top1 | within-k spearman | across-k top1 (mean) | ratio (top1) |
|---:|---:|---:|---:|---:|
| 1 | 0.0312 | -0.0006 | 0.1595 | **5.1×** |
| 2 | 0.0310 | 0.0004 | 0.3783 | **12.2×** |
| 4 | 0.0322 | 0.0012 | 0.4767 | **14.8×** |
| 8 | 0.0330 | 0.0005 | 0.5004 | **15.2×** |

**핵심 해석:**

1. **within-k top1 ≈ 1/32**: 같은 k로 학습해도 seed가 달라지면 top-1 expert 선택이 랜덤 수준으로 달라진다. routing path는 initialization에 고정되지 않는다.
2. **within-k spearman ≈ 0**: 랜덤한 전문가 ranking — 두 동일-k 모델이 전문가를 완전히 다른 순서로 선호한다.
3. **across-k는 5–16× 높음**: k 차이가 training stochasticity 자체보다 훨씬 큰 routing divergence를 만든다.
4. **expert frequency 확인**: max_expert_share = 0.037–0.044 (랜덤 기준 1/32 = 0.031에 근접). 인기 expert 편향이 없어서 within-k가 random인 것이 확인된다.

---

## 6. 라우팅 정렬 결과 (Matched Train-k Pairs)

같은 seed, 다른 k로 학습 후 최종 step에서 routing 비교.

### 6.1 Top1 agreement: random baseline vs observed

Random baseline = 1/32 = **0.0313**

#### Fixed-step 8-seed (key pairs)

| pair | rand | obs_top1 | obs/rand | obs_nest | rand_nest | nest_exc |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.031 | 0.228 | **7.3×** | 0.357 | 0.063 | +0.295 |
| 1-4 | 0.031 | 0.229 | **7.3×** | 0.532 | 0.125 | +0.407 |
| 1-8 | 0.031 | 0.231 | **7.4×** | 0.732 | 0.250 | +0.482 |
| 2-4 | 0.031 | 0.457 | **14.6×** | 0.690 | 0.125 | +0.565 |
| 4-8 | 0.031 | 0.526 | **16.8×** | 0.835 | 0.250 | +0.585 |
| 7-8 | 0.031 | 0.645 | **20.6×** | 0.804 | 0.250 | +0.554 |

#### Same-compute 8-seed (key pairs)

| pair | obs_top1 | obs/rand | obs_nest | nest_exc |
|---|---:|---:|---:|---:|
| 1-2 | 0.151 | **4.8×** | 0.251 | +0.188 |
| 1-4 | 0.159 | **5.1×** | 0.419 | +0.294 |
| 1-8 | 0.163 | **5.2×** | 0.635 | +0.385 |
| 4-8 | 0.533 | **17.1×** | 0.844 | +0.594 |
| 7-8 | 0.695 | **22.2×** | 0.838 | +0.588 |

**해석:**
- top1 관점에서 k=1 관련 pair는 random 대비 5–7× 수준 (낮음)
- k 간격이 좁은 pair (7-8, 6-8)는 20× 이상 (높음)
- same-compute k=1 관련이 fixed-step보다 낮음: k=1이 6000 steps로 더 분화됨

### 6.2 Spearman (ranking correlation)

Random baseline = **0.0**

Fixed-step 8-seed 주요 값:
- pair 1-2: mean 0.560, pair 1-8: 0.565, pair 7-8: 0.860
- 인접 k일수록 ranking 상관이 높음 (k 차이 → logit ordering 분기)

**Candidate direction check**: 전체 28 pair × 2 지표 = 56개 모두 8/8 seed에서 < 0.95. 모든 k pair에서 seed 만장일치.

---

## 7. B8: Matched Loss Table (mismatch cost 해석 기준)

mismatch delta는 각 모델 자신의 matched inference를 baseline (δ = 0)으로 정의한다.
이 표는 absolute loss를 맥락화하기 위한 참조값이다.

| k | fixed_step matched loss | fs_std | same_compute matched loss | mc_std | diff(fs-mc) |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.4472** | 0.0115 | **0.3204** | 0.0185 | +0.127 |
| 2 | 0.4196 | 0.0237 | 0.3503 | 0.0229 | +0.069 |
| 3 | 0.4046 | 0.0217 | 0.3779 | 0.0281 | +0.027 |
| **4** | 0.3959 | 0.0216 | 0.3959 | 0.0216 | **0.000** |
| 5 | 0.3904 | 0.0119 | 0.4222 | 0.0375 | -0.032 |
| 6 | 0.3803 | 0.0156 | 0.4373 | 0.0200 | -0.057 |
| 7 | 0.3769 | 0.0084 | 0.4664 | 0.0274 | -0.090 |
| 8 | **0.3839** | 0.0220 | **0.4810** | 0.0396 | -0.097 |

**핵심 관찰:**
- k=4: 두 budget에서 동일 (1500 steps가 교차점)
- Fixed-step: k가 클수록 loss 낮음 (더 많은 experts 활용 이점)
- Same-compute: k=1이 가장 낮음 (6000 steps 충분한 수렴)
- delta(k=8→k=1) fixed = +1.507 → absolute inference loss ≈ 0.384 + 1.507 = **1.891** (matched 대비 392% 증가)

---

## 8. Mismatch Cost 결과

### 8.1 Fixed-step 8-seed (8 seeds 평균)

| train_k | infer_k | mean_delta | direction |
|---:|---:|---:|---|
| 1 | 8 | +0.300 | lo→hi |
| 4 | 1 | +0.890 | hi→lo |
| 5 | 1 | +1.069 | hi→lo |
| 6 | 1 | +1.243 | hi→lo |
| 7 | 1 | +1.377 | hi→lo |
| **8** | **1** | **+1.507** | **hi→lo** |
| 7 | 8 | -0.002 | lo→hi |
| 6 | 7 | -0.000 | lo→hi |

### 8.2 Same-compute 8-seed

| train_k | infer_k | mean_delta |
|---:|---:|---:|
| 1 | 8 | +0.124 |
| 8 | 1 | +1.613 |
| 7 | 1 | +1.518 |
| 6 | 1 | +1.339 |
| 5 | 1 | +1.123 |

---

## 9. B9: Asymmetry by K-gap (95% CI + Linear Extrapolation)

asymmetry = |delta(hi→lo) − delta(lo→hi)|

### Fixed-step 8-seed

| gap | asymm_mean | 95% CI lo | 95% CI hi | linear_pred | ratio | CI > linear? |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0572 | 0.0336 | 0.0809 | 0.0572 | 1.00× | — |
| 2 | 0.1208 | 0.0697 | 0.1720 | 0.1144 | 1.06× | No |
| 3 | 0.2100 | 0.1124 | 0.3077 | 0.1716 | 1.22× | No |
| 4 | 0.2936 | 0.1574 | 0.4299 | 0.2289 | 1.28× | No |
| 5 | 0.4192 | 0.2269 | 0.6114 | 0.2861 | 1.47× | No |
| 6 | 0.6555 | 0.3935 | 0.9176 | 0.3433 | 1.91× | No |
| **7** | **1.2078** | **1.1416** | **1.2739** | **0.4005** | **3.02×** | **YES** |

### Same-compute 8-seed

| gap | asymm_mean | 95% CI lo | 95% CI hi | linear_pred | ratio | CI > linear? |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0553 | 0.0345 | 0.0762 | 0.0553 | 1.00× | — |
| 2 | 0.1351 | 0.0807 | 0.1896 | 0.1107 | 1.22× | No |
| 3 | 0.2467 | 0.1434 | 0.3499 | 0.1660 | 1.49× | No |
| 4 | 0.3709 | 0.2179 | 0.5239 | 0.2213 | 1.68× | No |
| 5 | 0.5579 | 0.3382 | 0.7776 | 0.2766 | 2.02× | No |
| 6 | 0.8748 | 0.5719 | 1.1776 | 0.3319 | 2.64× | No |
| **7** | **1.4887** | **1.4205** | **1.5570** | **0.3873** | **3.84×** | **YES** |

**hi→lo unanimity**: fixed-step 8/8, same-compute 8/8 (gap≥2) — 단 1개 예외 없음.

---

## 10. Noise Floor: Adjacent-k (gap=1) 분리

같은 gap=1이라도 방향에 따라 signal 강도가 전혀 다르다.

### Fixed-step 8-seed, gap=1

| direction | pairs | mean delta | SNR | signal? |
|---|---|---:|---:|---|
| **hi→lo** | 2→1, 3→2, 4→3, 5→4, 6→5, 7→6, 8→7 | 0.005–0.262 | **3–15** | **STRONG (7/7)** |
| lo→hi | 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8 | −0.012–−0.002 | 0.5–1.8 | noise (0/7) |

### Same-compute 8-seed, gap=1

| direction | SNR range | signal? |
|---|---|---|
| **hi→lo** | 2.3–10.7 | **STRONG (7/7)** |
| lo→hi | 0.5–2.8 (most < 2) | 대부분 noise |

**핵심**: "인접 k는 안전하다"는 주장은 **lo→hi 방향에만** 성립한다. hi→lo는 gap=1에서도 명확한 비용이 있다.

---

## 11. Routing-Mismatch Correlation

routing alignment와 |mismatch delta| 간 Spearman 상관 (n=8 seeds, descriptive only).

**Fixed-step 8-seed 상위:**

| pair | metric | Spearman |
|---|---|---:|
| 4-8 | nestedness | −0.881 |
| 4-8 | top1_agreement | −0.857 |
| 3-4 | spearman | −0.810 |
| 4-8 | spearman | −0.810 |
| 2-4 | nestedness | −0.786 |

**Same-compute 8-seed 상위:**

| pair | metric | Spearman |
|---|---|---:|
| 4-5 | spearman | −0.905 |
| 3-8 | spearman | −0.833 |
| 5-8 | top1_agreement | −0.833 |

음의 상관: routing alignment가 낮을수록 mismatch cost가 크다.
n=8이므로 인과 주장은 불가하며, 방향성만 서술 가능.

---

## 12. Claim별 Evidence 강도

| Claim | 표현 | Evidence 강도 | 핵심 수치 |
|---|---|---|---|
| **A** | k가 routing structure를 바꾼다 | **강함** | across-k / within-k ratio 5–16×; within-k ≈ random |
| **B** | 차이가 cardinality만이 아니다 | **중간-강함** | nestedness_excess 0.19–0.59; HCO > 0 |
| **C** | 인기 expert 공유 때문이 아니다 | **중간** | max_expert_share 0.037–0.044 ≈ 1/32; within-k ≈ random이 간접 증거 |
| **D** | hi→lo가 lo→hi보다 비싸다 | **가장 강함** | 8/8 seed unanimity, two budgets 공통 |
| **E** | gap=7에서 비대칭도 선형 외삽 초과 | **강함** | CI lo 1.14/1.42 > linear pred 0.40/0.39 |
| **F** | 두 budget 모두에서 재현 | **강함** | 방향 동일, 크기는 budget 따라 다름 |
| **G** | lo→hi gap=1은 safe | **중간** | SNR < 2, 부호 혼재 |
| **H** | hi→lo gap=1은 safe하지 않음 | **강함** | SNR 3–15, 8/8 positive |

---

## 13. 주장 가능한 범위 및 금지 표현

### 주장 가능 (현재 evidence로)

- "k를 바꾸면 routing이 달라진다: training noise의 5–16배"
- "hi→lo 추론은 gap=1에서도 비용이 있다 (8/8 seed)"
- "gap=7 비대칭은 선형 외삽을 명확히 초과한다"
- "lo→hi gap=1 전환은 noise 수준의 비용"
- "이 패턴은 fixed-step과 same-compute 두 budget에서 일관됨"

### 금지 표현

| 금지 표현 | 이유 |
|---|---|
| "k=1→k=2로 성능이 좋아진다" | delta ≈ −0.005, SNR 0.49, 부호 혼재 (noise) |
| "asymmetry는 power-law로 증가한다" | gap=7에서만 CI로 확인, 함수형 fit 미수행 |
| "routing divergence가 mismatch cost를 유발한다" | 상관 0.71–0.90이지만 n=8, 인과 ablation 없음 |
| "이 결과를 pretrained MoE에 일반화할 수 있다" | 합성 소규모 모델, 별도 검증 필요 |
| "same-compute에서 k=1이 특별히 좋다" | step 효과(6000 steps)와 k 효과 미분리 |

---

## 14. 남은 베이스라인 gap

| 베이스라인 | 우선순위 | 현재 상태 | 비고 |
|---|---|---|---|
| B3 Frequency-empirical null | 낮음 | 간접 커버됨 | max_expert_share ≈ 1/32로 편향 없음 확인 |
| B6 Step-matched within-k | same-compute 쓰면 필요 | 미구현 | k=1@step1500 vs step6000 비교 |
| Routing-mismatch causal ablation | 낮음 (scope 외) | 미구현 | router transplant experiment 필요 |
| HCO for across-k pairs | 낮음 | 구현됨(B5용) | nestedness_excess로 충분 |

---

## 15. 실험 재현

```bash
# Fixed-step full run
python -m moe_topk.scratch_pilot \
  --mode full \
  --config configs/sparse32_kgrid_fixed_step_3seed.json \
  --output-root /tmp/topk_exp --device cuda

# 8-seed combined run 구성
python scripts/build_combined_run.py \
  --source-runs /tmp/.../sparse32_kgrid_fixed_step_3seed/... \
                /tmp/.../sparse32_kgrid_fixed_step_seed3to7/... \
  --out-run /tmp/.../sparse32_kgrid_fixed_step_8seed/...

# Baseline analysis (B5 포함)
python scripts/analyze_training_variability_null.py \
  --run-dir /tmp/.../sparse32_kgrid_fixed_step_8seed/... \
  --out-dir results/baselines/fixed_step_8seed \
  --n-experts 32
```

## 16. 결과 디렉토리

```
results/
├── sparse32_kgrid_fixed_step_3seed_summary/   # 3-seed reference
├── sparse32_kgrid_mechanism_3seed_summary/    # same-compute 3-seed reference
├── sparse32_kgrid_fixed_step_8seed_summary/   # 8-seed main results (fixed)
├── sparse32_kgrid_same_compute_8seed_summary/ # 8-seed main results (same-compute)
└── baselines/
    ├── fixed_step_8seed/
    │   ├── b5_within_k_raw.csv      # 224 within-k pairs, all metrics
    │   ├── b5_summary.csv           # within vs across comparison
    │   ├── b8_matched_loss.csv      # matched loss per k
    │   ├── b9_asymmetry_stats.csv   # asymmetry CI + linear extrap
    │   └── noise_floor.csv          # gap=1 mismatch SNR
    └── same_compute_8seed/
        └── (same structure)
```
