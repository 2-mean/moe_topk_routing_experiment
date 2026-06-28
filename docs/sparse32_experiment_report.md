# sparse32 Top-k Routing Experiment Report

## 1. 연구 목적

Train-time top-k와 inference-time top-k가 다를 때 발생하는 **라우팅 구조 분기**와 **mismatch cost 비대칭** 현상을 정량적으로 측정한다. 특히 다음 세 가지를 분리해서 확인한다.

1. **라우팅 구조 정렬 강도**: 서로 다른 k로 학습한 모델이 같은 입력에 대해 얼마나 비슷한 전문가를 선택하는가.
2. **Mismatch cost**: 학습 k ≠ 추론 k 조건에서 validation loss 증가분.
3. **비대칭성**: `train_k=a → infer_k=b`와 `train_k=b → infer_k=a` 중 어느 방향이 더 큰 mismatch를 만드는가.

---

## 2. 모델 및 공통 설정

| 항목 | 값 |
|---|---|
| 아키텍처 | TinyMoE Transformer (8-layer, d\_model=192, n\_heads=6) |
| 전문가 수 | 32 (`n_experts=32`) |
| Expert hidden | 768 |
| Sparse dispatch | `true` (학습 시 실제 sparse routing) |
| 라우터 보조 손실 | `router_aux_loss_coef=0.01` |
| Seq length | 96 |
| Vocab size | 256 |
| Learning rate | 2.5e-4 |
| Weight decay | 0.01 |
| Dropout | 0 |
| Optimizer | AdamW |
| Data seed | 1729 |
| Data order seed | 31415 |
| 데이터 구조 | 7 카테고리 (general\_ko/en, math\_ko/en, code, reasoning, translation) × 2500 samples |
| Collapse threshold | 0.9 (max\_expert\_share > 0.9이면 collapse 판정) |

---

## 3. 실험 목록 및 설계

### 3.1 sparse32\_kgrid\_fixed\_step (Fixed-step grid)

**목적**: 모든 k 조건에 동일 스텝(1500)을 적용해 step budget 효과를 배제하고 순수 k 차이의 효과만 관찰.

| 항목 | 값 |
|---|---|
| Train k | 1, 2, 3, 4, 5, 6, 7, 8 |
| Inference k | 1, 2, 3, 4, 5, 6, 7, 8 |
| Steps (모든 k 공통) | **1500** |
| Batch size | 16 |
| 총 runs | k × seeds = 8 × seeds |
| 3-seed 버전 | seeds [0,1,2] → 24 runs |
| 8-seed 버전 | seeds [0..7] → 64 runs |
| Config 파일 | `configs/sparse32_kgrid_fixed_step_3seed.json` |

### 3.2 sparse32\_kgrid\_mechanism (Same-compute grid)

**목적**: 각 k에 동일 gradient step 수 × batch\_size 기준 compute를 부여해 k 조건이 공정하게 비교되도록 설정.

| 항목 | 값 |
|---|---|
| Train k | 1, 2, 3, 4, 5, 6, 7, 8 |
| Inference k | 1, 2, 3, 4, 5, 6, 7, 8 |
| Steps (k별 차등) | k=1: 6000, k=2: 3000, k=3: 2000, k=4: 1500, k=5: 1200, k=6: 1000, k=7: 860, k=8: 750 |
| Batch size | 16 |
| 3-seed 버전 | seeds [0,1,2] → 24 runs |
| 8-seed 버전 | seeds [0..7] → 64 runs |
| Config 파일 | `configs/sparse32_kgrid_mechanism_3seed.json` |

### 3.3 8-seed 확장 방법론

3-seed run (seeds 0-2)과 추가 5-seed run (seeds 3-7)의 raw route 파일을 **symlink**로 묶어 combined run dir을 구성. 원본 파일은 복사·수정하지 않음.

- `scripts/build_combined_run.py`: 두 source run의 manifest와 route symlink를 합쳐 새 combined run 생성
- `scripts/summarize_combined_run.py`: combined run의 metrics를 source에서 직접 병합해 summary/plot 재집계 (route raw 재계산 없음)
- `scripts/analyze_full_ranking.py`: full-ranking diagnostics를 combined run에서 생성

---

## 4. 측정 지표

| 지표 | 정의 | 해석 |
|---|---|---|
| `nestedness` | 작은 k의 선택 집합이 큰 k의 집합에 얼마나 포함되는가 (0~1) | 1에 가까울수록 k가 달라도 라우팅 경로가 계층적으로 유사 |
| `top1_agreement` | 두 조건에서 각 토큰의 1위 전문가가 일치하는 비율 (0~1) | 1에 가까울수록 라우터의 top 선택이 k와 무관하게 안정적 |
| `spearman` | 두 조건의 gate logit 전문가 순위 간 Spearman 상관 | 선호 순서 자체의 안정성 |
| `mismatch_delta_loss` | `val_loss(train_k=a, infer_k=b) - val_loss(train_k=a, infer_k=a)` | 양수: k 불일치 시 손실 증가 |
| Asymmetry | `delta(a→b) - delta(b→a)` 절댓값 차이 | 비대칭 방향성: 어느 쪽 전환이 더 비용이 큰가 |

**Sanity gate**:
- `logit cutoff sanity nestedness min ≥ 1.0`: 동일 가중치에서 logit cutoff만 다른 비교가 완전 포함 관계인지 확인
- `step-0 same-W0 same-infer nestedness min = 1.0`: step 0에서 같은 초기화·같은 infer-k면 경로가 동일해야 함
- `collapsed final matched runs = 0`: max\_expert\_share > 0.9인 run이 없어야 함

---

## 5. 실험 결과

### 5.1 Sanity gates

모든 실험에서 sanity gate 통과.

| 실험 | runs | collapsed | logit cutoff nestedness min | step-0 nestedness min |
|---|---:|---:|---:|---:|
| fixed\_step\_3seed | 24/24 | 0 | 1.0 | 1.0 |
| mechanism\_3seed | 24/24 | 0 | 1.0 | 1.0 |
| fixed\_step\_8seed | 64/64 | 0 | 1.0 | 1.0 |
| mechanism\_8seed | 64/64 | 0 | 1.0 | 1.0 |

### 5.2 라우팅 구조 정렬 (Matched train-k pair metrics)

핵심 지표: `nestedness`와 `top1_agreement`. 8-seed 결과 기준.

#### 5.2.1 Fixed-step 8-seed

| pair (train_k) | nestedness | top1_agreement | spearman |
|---|---:|---:|---:|
| 1-2 | 0.3572 | 0.2284 | 0.5602 |
| 1-4 | 0.5324 | 0.2285 | 0.5582 |
| 1-8 | 0.7316 | 0.2313 | 0.5648 |
| 2-4 | 0.6902 | 0.4574 | 0.7313 |
| 2-8 | 0.8371 | 0.4159 | 0.7060 |
| 4-8 | 0.8353 | 0.5262 | 0.7854 |
| 7-8 | 0.8042 | 0.6445 | 0.8600 |

**패턴**: nestedness는 k 차이가 클수록 높아지는 경향 (k=1이 소집합이라 포함 관계 성립 쉬움). top1\_agreement와 spearman은 k 차이가 작을수록 높음. 즉 **가까운 k끼리 1위 전문가 일치율이 높고, k 차이가 클수록 gate 선호 순서도 달라진다**.

#### 5.2.2 Same-compute (mechanism) 8-seed

| pair (train_k) | nestedness | top1_agreement | spearman |
|---|---:|---:|---:|
| 1-2 | 0.2505 | 0.1511 | 0.4126 |
| 1-4 | 0.4190 | 0.1590 | 0.4385 |
| 1-8 | 0.6345 | 0.1633 | 0.4569 |
| 2-4 | 0.6545 | 0.4256 | 0.6892 |
| 2-8 | 0.8213 | 0.4002 | 0.6758 |
| 4-8 | 0.8435 | 0.5333 | 0.7936 |
| 7-8 | 0.8379 | 0.6950 | 0.8916 |

**fixed-step 대비 비교**: same-compute에서 특히 k=1 관련 지표가 낮음. same-compute에서 k=1 모델은 6000 스텝으로 훈련되는 반면 k=8은 750 스텝이라, k=1 라우터가 더 분화(specialization)되어 상대적으로 k=2·4와 정렬이 낮아진다.

### 5.3 Mismatch cost 및 비대칭성

모든 수치는 `mean_delta_loss` (양수 = 불일치 시 손실 증가).

#### 5.3.1 Fixed-step 8-seed 주요 행

| train_k | infer_k | mean_delta |
|---:|---:|---:|
| 1 | 2 | -0.005 |
| 1 | 8 | +0.300 |
| 2 | 1 | +0.262 |
| 4 | 1 | +0.890 |
| 5 | 1 | +1.069 |
| 6 | 1 | +1.243 |
| 7 | 1 | +1.377 |
| 8 | 1 | +1.507 |

**핵심 비대칭**: `k=a → k=1` 방향 손실이 반대 방향보다 항상 크다. 즉 **high-k로 훈련된 모델을 low-k로 추론하면 large mismatch가 발생**한다.

#### 5.3.2 Same-compute (mechanism) 8-seed 주요 행

| train_k | infer_k | mean_delta |
|---:|---:|---:|
| 1 | 8 | +0.124 |
| 5 | 1 | +1.123 |
| 6 | 1 | +1.339 |
| 7 | 1 | +1.518 |
| 8 | 1 | +1.613 |
| 8 | 2 | +0.423 |

fixed-step 대비 `k=1 → higher-k` 방향 mismatch가 더 작음 (k=1 모델이 더 많은 스텝으로 충분히 분화). `higher-k → k=1` 방향은 오히려 더 큼.

### 5.4 Candidate direction check

두 실험 모두 전체 28개 pair × 2 지표 = 56개 combination에서 **모두 `candidate_effect = true`**:
- fixed\_step: 8/8 seeds below 0.95 (min support 5)
- mechanism: 8/8 seeds below 0.95 (min support 5)

즉 어떤 두 k를 비교하더라도, 전체 seed의 과반이 같은 방향(< 0.95)을 가리킨다.

---

## 5.5 비대칭도 k-gap 함수 분석 (8-seed)

`lo→hi`: low-k로 학습 후 high-k로 추론 / `hi→lo`: high-k로 학습 후 low-k로 추론

### Fixed-step 8-seed

| k-gap | lo→hi mean delta | hi→lo mean delta | asymmetry | n_pairs |
|---:|---:|---:|---:|---:|
| 1 | -0.0053 | +0.0520 | 0.0572 | 7 |
| 2 | +0.0062 | +0.1270 | 0.1209 | 6 |
| 3 | +0.0299 | +0.2399 | 0.2100 | 5 |
| 4 | +0.0712 | +0.3649 | 0.2937 | 4 |
| 5 | +0.1329 | +0.5521 | 0.4192 | 3 |
| 6 | +0.2140 | +0.8696 | 0.6555 | 2 |
| 7 | +0.2996 | +1.5073 | 1.2077 | 1 |

### Same-compute (mechanism) 8-seed

| k-gap | lo→hi mean delta | hi→lo mean delta | asymmetry | n_pairs |
|---:|---:|---:|---:|---:|
| 1 | -0.0066 | +0.0487 | 0.0553 | 7 |
| 2 | -0.0029 | +0.1322 | 0.1351 | 6 |
| 3 | +0.0105 | +0.2571 | 0.2466 | 5 |
| 4 | +0.0339 | +0.4047 | 0.3709 | 4 |
| 5 | +0.0638 | +0.6216 | 0.5579 | 3 |
| 6 | +0.0954 | +0.9702 | 0.8748 | 2 |
| 7 | +0.1243 | +1.6131 | 1.4888 | 1 |

**관찰**: 비대칭도가 k-gap에 대해 초선형적으로 증가. gap=1에서 ~0.06이던 비대칭도가 gap=7에서 ~1.2-1.5로 급증.

### Routing alignment by k-gap (fixed_step_8seed)

| k-gap | mean top1_agreement | mean nestedness | mean spearman | n_pairs |
|---:|---:|---:|---:|---:|
| 1 | 0.5312 | 0.6791 | 0.7797 | 7 |
| 2 | 0.4934 | 0.7136 | 0.7559 | 6 |
| 3 | 0.4551 | 0.7384 | 0.7316 | 5 |
| 4 | 0.4181 | 0.7560 | 0.7056 | 4 |
| 5 | 0.3759 | 0.7670 | 0.6751 | 3 |
| 6 | 0.3238 | 0.7663 | 0.6358 | 2 |
| 7 | 0.2313 | 0.7316 | 0.5648 | 1 |

`top1_agreement`와 `spearman`은 k-gap에 단조 감소. `nestedness`는 gap=4-5에서 피크 후 약간 감소 (gap이 클수록 포함 관계 자체가 흔들림).

---

## 6. 핵심 발견

### 발견 1: 라우팅 구조는 k에 따라 체계적으로 분기한다

`top1_agreement`가 k 차이에 비례해 낮아지는 경향이 모든 seed에서 일관됨. k=1과 k=8로 학습한 모델은 동일 입력에서도 top-1 전문가 선택이 크게 달라진다.

### 발견 2: Mismatch cost는 강하게 비대칭이다

`high-k → low-k 추론` 방향의 mismatch cost가 반대보다 항상 큼. 특히 k=8 → k=1 전환 시 val loss가 +1.5 이상 증가. 이는 **high-k 라우팅 표현이 low-k 추론 구조로 쉽게 압축되지 않음**을 시사한다.

### 발견 3: 비대칭 패턴은 step budget 설정과 무관하게 재현된다

fixed-step (공정 스텝)과 same-compute (공정 compute) 두 설정 모두에서 같은 방향의 비대칭이 관찰됨. **비대칭의 원인은 스텝 수 차이가 아니라 k 자체가 만드는 라우팅 표현의 구조적 차이**.

### 발견 4: k 인접 구간은 상호 호환성이 높다

인접한 k (예: k=5 ↔ k=6, k=7 ↔ k=8) 사이 mismatch delta는 ±0.01 수준으로 거의 0. 즉 **추론 시 k를 1 정도 변경하는 것은 실용적으로 안전**하다.

### 발견 5: 비대칭도는 k-gap에 초선형적으로 증가한다

k-gap=1에서 asymmetry ≈ 0.06, k-gap=7에서 ≈ 1.2-1.5. **단순 선형 관계가 아니라 gap이 커질수록 비대칭도가 가속적으로 증가**함. 이는 고-k 라우팅 표현이 저-k 경계 조건으로 갈수록 복구 불가능하게 손실됨을 시사.

### 발견 6: lo→hi mismatch는 음수도 가능하다 (same-compute)

same-compute 설정에서 `k=1 → k=2` 추론 시 mismatch delta ≈ -0.01 (손실 오히려 감소). k=1 모델이 충분히 학습(6000 스텝)되어 라우팅이 전문화된 상태에서, 추론 시 전문가를 1개 더 추가하면 오히려 보완적 처리가 가능해지기 때문으로 해석.

### 발견 7: same-compute에서 hi→lo 방향 비대칭이 더 크다

gap=7 기준: fixed-step 비대칭 1.21 vs same-compute 1.49. same-compute에서 k=1 모델이 더 많은 스텝으로 학습되어 routing 표현이 더 분화됨 → high-k 모델이 k=1로 추론할 때 손실이 더 크게 증가.

---

## 7. 제한 사항

1. **소규모 모델**: d_model=192, n_experts=32로 실제 LLM 스케일과 큰 차이가 있어 정량 수치의 직접 일반화는 보수적으로 다뤄야 함.
2. **합성 데이터**: 실제 텍스트 분포가 아닌 템플릿 기반 합성 데이터 사용. 실제 데이터에서 동일 패턴이 나타나는지 별도 검증 필요.
3. **단일 아키텍처**: MoE layer가 모든 8 레이어에 동일 설정으로 배치됨. Hybrid (일부만 MoE) 설정에서 다를 수 있음.
4. **Aux loss 고정**: `router_aux_loss_coef=0.01`로만 실험. Aux loss 없거나 강한 경우 패턴이 달라질 가능성이 있음 (현재 별도 실험 대기 중).

---

## 8. 실험 재현 방법

### 서버 환경

- 서버: `ssh -p 101 2020110906@cs.dongguk.edu`
- GPU: NVIDIA GeForce RTX 3080 Ti (12 GB)
- CUDA 12.4, Python 3.10, PyTorch
- Conda env: `cas4160`

### 로컬 실행

```bash
# smoke 테스트
python -m moe_topk.scratch_pilot \
  --mode smoke \
  --config configs/sparse32_kgrid_fixed_step_3seed.json \
  --output-root /tmp/topk_exp \
  --device cuda

# full 실험
python -m moe_topk.scratch_pilot \
  --mode full \
  --config configs/sparse32_kgrid_fixed_step_3seed.json \
  --output-root /tmp/topk_exp \
  --device cuda
```

### 8-seed combined run 생성

```bash
# seeds 0-2 run과 seeds 3-7 run을 combined으로 묶기
python scripts/build_combined_run.py \
  --source-runs /tmp/.../sparse32_kgrid_fixed_step_3seed/... \
                /tmp/.../sparse32_kgrid_fixed_step_seed3to7/... \
  --out-run /tmp/.../sparse32_kgrid_fixed_step_8seed/...

# summary/plot 재집계
python scripts/summarize_combined_run.py \
  --run-dir /tmp/.../sparse32_kgrid_fixed_step_8seed/...
```

---

## 9. 결과 디렉토리 구조

```
results/
├── sparse32_kgrid_fixed_step_3seed_summary/    # seeds 0-2, fixed-step
│   ├── summary.md
│   ├── metrics.csv
│   ├── task_metrics.csv
│   ├── plots/
│   └── diagnostics/
├── sparse32_kgrid_mechanism_3seed_summary/     # seeds 0-2, same-compute
├── sparse32_kgrid_fixed_step_8seed_summary/    # seeds 0-7, fixed-step (combined)
│   ├── summary.md
│   ├── source_runs.txt
│   └── ...
└── sparse32_kgrid_same_compute_8seed_summary/  # seeds 0-7, same-compute (combined)
```
