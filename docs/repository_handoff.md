# 저장소 인수인계 및 재현 가이드

> 기준일: 2026-07-02 KST
>
> canonical remote: `https://github.com/2-mean/moe_topk_routing_experiment.git`
>
> canonical branch: `main`

이 문서는 새 PC나 서버에서 이 Git 저장소만 clone한 뒤 개발, 실험, 분석,
발표자료 생성을 이어가기 위한 단일 참고 문서다. 과거 진행 순서보다 현재 코드와
artifact가 무엇을 보장하는지에 초점을 둔다.

## 1. 프로젝트 한눈에 보기

연구 질문은 학습 시 사용한 `k_train`과 추론 시 사용한 `k_infer`가 다를 때
다음 두 값이 어떻게 변하는가이다.

1. expert 선택과 router ranking으로 측정한 routing structure
2. matched inference 대비 validation loss delta

현재 주 결과 패키지는 다음 두 실험을 8개 seed로 비교한다.

| Budget | 설정 | k별 training step | 통제 의미 |
|---|---|---|---|
| Fixed-step | `sparse32_kgrid_fixed_step_8seed` | 모든 k가 1500 | optimizer update 수 동일 |
| Same-compute | `sparse32_kgrid_same_compute_8seed` | `6000,3000,2000,1500,1200,1000,860,750` | active expert invocation proxy를 근사 통제 |

공통 모델은 8-layer, `d_model=192`, 6-head, 32-expert TinyMoE이며,
`k_train,k_infer ∈ {1..8}`을 평가한다. Same-compute는 total compute를 완전히
통제한 설정이 아니며 optimizer update 수는 k에 따라 다르다.

현재 가장 강한 결과는 다음과 같다.

| 관측 | Fixed-step | Same-compute | 해석 범위 |
|---|---:|---:|---|
| `k=8→1` raw mismatch delta | `+1.507` | `+1.613` | high-k 학습 모델의 low-k 추론 비용 |
| `k=8→1` matched loss 대비 변화율 | `+392.67%` | `+335.36%` | scale 보조 설명; primary metric은 raw delta |
| hi→lo 방향 일치 | 8/8 seed | 8/8 seed | 두 budget에서 재현 |
| gap=7 asymmetry CI lower > linear | `1.14 > 0.40` | `1.42 > 0.39` | 선형 외삽보다 큰 extreme-gap 비대칭 |

주장 가능한 범위와 금지 표현은
[`sparse32_full_report.md`](sparse32_full_report.md)의 12–14절을 따른다.
특히 pretrained MoE 일반화와 routing divergence의 인과 효과는 아직 증명하지
않았다.

## 2. 새 환경 준비

### 2.1 공통 clone

```bash
git clone https://github.com/2-mean/moe_topk_routing_experiment.git
cd moe_topk_routing_experiment
git switch main
git pull --ff-only
git status --short --branch
```

정상 상태라면 `main...origin/main`이며 추가 변경이 없어야 한다.

### 2.2 새 Python 환경

Python 3.10 이상이 필요하다. CPU 검증 환경은 다음처럼 만들 수 있다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

`pyproject.toml`의 직접 dependency는 `torch`, `numpy`, `matplotlib`,
`scipy`다. CUDA 장비에서는 generic CPU wheel을 먼저 설치하지 말고 장비의
CUDA/driver에 맞는 PyTorch를 설치한 다음 `python -m pip install -e .`을
실행한다.

### 2.3 학교 서버의 기존 환경

기록된 주 실행 환경:

- login: `ssh -p 101 2020110906@cs.dongguk.edu`
- repo: `/home/2020110906/moe_topk_routing_experiment`
- conda env: `/home/2020110906/miniconda3/envs/cas4160`
- Python: `3.10.19`
- recorded PyTorch: `2.10.0+cu128`
- GPU: RTX 3080 Ti 12GB
- raw root: `/tmp/2020110906_matryo_topk`

```bash
cd ~/moe_topk_routing_experiment
git switch main
git pull --ff-only
source ~/miniconda3/bin/activate cas4160
python -m pip install -e . --no-deps
python - <<'PY'
import torch, numpy, matplotlib, scipy
print("python/torch ready")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Editable install 전 임시 호환 경로로 `export PYTHONPATH="$PWD/src"`를 사용할
수 있지만, 새 환경의 canonical setup은 `pip install -e .`이다.

## 3. clone 직후 acceptance check

아래 네 단계가 모두 통과하면 code/config/compact artifact를 사용할 준비가 된
것이다.

```bash
python -m unittest discover -s tests
python -m moe_topk.scratch_pilot --help
python scripts/build_combined_run.py --help
python scripts/build_visual_assets.py
```

마지막 명령은 committed summary와 baseline CSV로 report figure, manifest,
HTML 발표자료를 재생성한다. 생성 후 의도하지 않은 diff가 없어야 재현성이 맞다.

```bash
git status --short
git diff --check
```

GPU 확인은 별도로 수행한다.

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA torch가 아님"
print(torch.cuda.get_device_name(0))
PY
```

## 4. 코드 실행 흐름

```text
config JSON
  -> scratch_pilot.load_config
  -> synthetic corpus 또는 curated JSONL probe
  -> seed별 공통 W0 생성
  -> train_k별 독립 학습
  -> inference_k grid 평가와 route artifact 저장
  -> metrics.csv / task_metrics.csv / summary.md / plots 생성
```

핵심 모듈:

| 파일 | 책임 |
|---|---|
| `src/moe_topk/data.py` | 7개 category 합성 corpus, JSONL probe, deterministic batch index |
| `src/moe_topk/model.py` | causal TinyMoE Transformer, sparse/dense expert dispatch |
| `src/moe_topk/metrics.py` | top1 agreement, overlap recall, Spearman, coactivation |
| `src/moe_topk/ranking.py` | full-ranking, transition, calibrated overlap metric |
| `src/moe_topk/scratch_pilot.py` | train/eval/artifact/analyze/summarize 전체 runner |

### 실제 inference top-k 구현

`TopKMoE.forward`는 모든 expert에 대한 `gate_logits`를 계산하고
`torch.topk(gate_logits, k=top_k)`로 expert를 고른다. 선택된 logit만 다시
`softmax`해 합이 1인 `top_weights`를 만들고, 선택 밖 expert는 dispatch하지
않는다. 따라서 k를 줄여도 output mass가 단순히 줄어드는 구조가 아니라 선택된
expert 사이에서 weight가 재정규화된다. 이 구현은 보고서의 mismatch 해석에
필수인 전제다.

## 5. 설정 파일 지도

| 그룹 | Config | 목적 |
|---|---|---|
| 초기 pilot | `scratch_pilot.json` | 8 experts, k=1/2/4, fixed-step 3-seed |
| 초기 control | `same_compute_pilot.json` | 8 experts, k별 step 보정 |
| probe control | `curated_probe_pilot.json` | 고정 JSONL probe 사용 |
| robustness | `robust_same_compute_9seed.json` | same-compute 9-seed |
| robustness | `robust_curated_probe_9seed.json` | curated probe 9-seed |
| auxiliary | `robust_same_compute_aux0_9seed.json` | router auxiliary coefficient 0 |
| auxiliary | `robust_same_compute_aux005_9seed.json` | router auxiliary coefficient 0.005 |
| stress | `large_model_same_compute_pilot.json` | 큰 scratch model smoke/stress |
| Sparse32 main | `sparse32_kgrid_fixed_step_3seed.json` | E=32, k=1..8, seed 0..2 fixed-step |
| Sparse32 extension | `sparse32_kgrid_fixed_step_seed3to7.json` | fixed-step seed 3..7 |
| Sparse32 main | `sparse32_kgrid_mechanism_3seed.json` | E=32, k=1..8, seed 0..2 same-compute |
| Sparse32 extension | `sparse32_kgrid_same_compute_seed3to7.json` | same-compute seed 3..7 |
| secondary deep grid | `deep_kgrid_same_compute_10seed.json` | E=8, deeper 10-seed 확인 |

`deep_kgrid_same_compute_10seed`는 E=8 설정이므로 Sparse32 E=32 주 결과와
같은 표에 섞어 해석하지 않는다.

## 6. 실험 실행

### 6.1 빠른 CPU smoke

```bash
python -m moe_topk.scratch_pilot \
  --mode smoke \
  --config configs/scratch_pilot.json \
  --output-root /tmp/moe_topk_smoke \
  --device cpu
```

CPU smoke는 control flow 확인용이다. 논문 수치나 GPU runtime 재현에 사용하지
않는다.

### 6.2 GPU smoke와 full

```bash
python -m moe_topk.scratch_pilot \
  --mode smoke \
  --config configs/sparse32_kgrid_fixed_step_3seed.json \
  --output-root /tmp/2020110906_matryo_topk \
  --device cuda

python -m moe_topk.scratch_pilot \
  --mode full \
  --config configs/sparse32_kgrid_fixed_step_3seed.json \
  --output-root /tmp/2020110906_matryo_topk \
  --device cuda
```

OOM이 발생하면 config의 `oom_fallbacks` 순서대로 batch/model fallback을
시도하며, 실제 선택된 fallback index와 모델 크기는 run directory 이름과
`config.json`에서 확인한다. 서로 다른 fallback 크기의 결과를 같은 실험으로
합치지 않는다.

### 6.3 장시간 background 실행

```bash
bash scripts/launch_background.sh RUN_NAME smoke CONFIG_PATH cuda
bash scripts/check_background.sh RUN_NAME
bash scripts/launch_background.sh RUN_NAME full CONFIG_PATH cuda
bash scripts/check_background.sh RUN_NAME
```

helper는 `tmux` session을 만들고 log와 raw artifact를 `/tmp` 아래에 둔다.
동일 GPU에서 여러 full run을 겹쳐 실행하지 않는다.

## 7. 8-seed combined run과 분석

서로 겹치지 않는 seed raw run을 symlink 기반 combined run으로 합친다. 이
기능은 source raw run이 같은 Linux filesystem에 남아 있어야 한다.

```bash
python scripts/build_combined_run.py \
  --source-run /tmp/.../sparse32_kgrid_fixed_step_3seed/RUN_DIR \
  --source-run /tmp/.../sparse32_kgrid_fixed_step_seed3to7/RUN_DIR \
  --output-run /tmp/.../sparse32_kgrid_fixed_step_8seed/RUN_DIR \
  --experiment-name sparse32_kgrid_fixed_step_8seed

python scripts/summarize_combined_run.py \
  --run-dir /tmp/.../sparse32_kgrid_fixed_step_8seed/RUN_DIR
```

주요 후처리:

```bash
python scripts/analyze_routing_diagnostics.py --help
python scripts/analyze_task_specialization.py --help
python scripts/analyze_task_loss_grid.py --help
python scripts/analyze_full_ranking.py --help
python scripts/analyze_training_variability_null.py --help
python scripts/compare_sparse32_budgets.py --help
```

실제 raw run 경로가 없고 Git clone만 있는 환경에서는 committed
`results/**/metrics.csv`, `task_metrics.csv`, baseline CSV, summary를 입력으로
사용하는 분석만 가능하다.

## 8. 결과 디렉터리 읽는 법

| 경로 | 의미 |
|---|---|
| `results/sparse32_kgrid_fixed_step_8seed_summary/` | Sparse32 fixed-step 주 결과 |
| `results/sparse32_kgrid_same_compute_8seed_summary/` | Sparse32 same-compute 주 결과 |
| `results/baselines/fixed_step_8seed/` | fixed-step B5/B8/B9/noise-floor 표 |
| `results/baselines/same_compute_8seed/` | same-compute baseline 표 |
| `results/report_figures/` | 보고서/발표용 canonical PNG와 numeric JSON |
| `results/deep_kgrid_same_compute_10seed_summary/` | 별도 E=8 deep-grid 결과 |

주 결과 package의 일반 구조:

```text
summary.md
metrics.csv
task_metrics.csv
env.txt
source_runs.txt 또는 raw_run_path.txt
plots/
diagnostics/
  full_ranking/
  task_loss_grid/
  task_specialization/
```

`source_runs.txt`와 `raw_run_path.txt`는 실행 당시 서버 경로의 provenance다.
다른 머신에서 그 절대 경로가 존재한다는 뜻은 아니다.

## 9. 보고서와 시각자료 재생성

```bash
python scripts/build_visual_assets.py
```

이 명령은 다음을 순서대로 수행한다.

1. summary에서 `heatmap_data.json` 추출
2. `01`–`10` comparison PNG와 presentation PNG 생성
3. `docs/sparse32_presentation.html` 재생성
4. `results/report_figures/manifest.json` 갱신

`10_pct_change_comparison.png`와 `pct_change_matrices.json`은
`100 × mismatch_delta / matched_loss(k_train)`를 저장한다. matched loss가 작은
행에서는 비율이 크게 보일 수 있으므로 raw delta와 항상 함께 제시한다.

발표자료 사용법은 [`presentation_export.md`](presentation_export.md)를 따른다.

## 10. Git만으로 가능한 범위

Git에 포함:

- 모든 source, config, test, launcher, 분석 script
- compact summary/CSV/env/plot
- baseline 표와 최종 보고서/발표자료

Git에서 제외:

- `runs/**`
- `checkpoints/**`, `*.pt`, `*.pth`
- raw route `*.npz`
- cache와 Python build artifact

따라서 clone만으로 새 실험을 실행하거나 현재 compact 결과를 검토·가공할 수
있다. 반면 과거 token/layer route를 다시 계산하거나 checkpoint ablation을
수행하려면 서버 raw artifact가 필요하다. 원격 raw root를 정리하기 전에는
필요한 분석을 compact CSV로 내보냈는지 확인한다.

## 11. Git 동기화 절차

작업 시작:

```bash
git switch main
git pull --ff-only
git status --short --branch
```

변경 검증:

```bash
python -m unittest discover -s tests
python scripts/build_visual_assets.py
git diff --check
git status --short
```

커밋과 push:

```bash
git add PATH1 PATH2
git diff --cached --stat
git diff --cached
git commit -m "Describe the validated change"
git push origin main
```

다른 사람이 동시에 `main`을 갱신했다면 강제 push하지 않는다.

```bash
git fetch origin
git rebase origin/main
python -m unittest discover -s tests
git push origin main
```

학교 서버 기존 clone은 fetch URL이 HTTPS, push URL이 전용 SSH alias인 구성이
사용될 수 있다. `git remote -v`에서 둘이 다르게 보이는 것은 의도된 설정이다.
새 머신에서 push하려면 그 머신의 GitHub 인증을 별도로 구성해야 한다.

## 12. 자주 틀리는 지점

- 새 clone에서 과거 `scratch-pilot-v0` branch를 만들지 말고 `main`을 사용한다.
- `build_combined_run.py` 옵션은 `--source-run`을 source마다 반복하고,
  `--output-run`, `--experiment-name`을 사용한다.
- `nestedness`는 complete subset probability가 아니라 per-token overlap recall이다.
- same-compute는 active expert invocation proxy 통제이며 total compute 통제가 아니다.
- k를 바꿀 때 선택된 expert weight는 다시 softmax 정규화된다.
- raw result 절대 경로는 Git으로 이동하지 않는다.
- Windows에서 `.sh` launcher를 직접 실행하기보다 WSL/Linux 서버를 사용한다.
- generated figure를 수정할 때 generator, PNG/JSON, manifest, report reference를
  함께 갱신한다.

## 13. 작업 인수 체크리스트

- [ ] `main`이 `origin/main`과 동기화됨
- [ ] `python -m pip install -e .` 성공
- [ ] unit test 전체 통과
- [ ] `scratch_pilot --help`와 CPU smoke 성공
- [ ] GPU 작업이면 CUDA와 device name 확인
- [ ] 사용할 config의 seed, model size, fallback 확인
- [ ] raw artifact가 필요한지 clone-only 작업인지 구분
- [ ] 결과 해석은 `sparse32_full_report.md`의 claim boundary를 따름
- [ ] figure 변경 시 `build_visual_assets.py` 재실행
- [ ] commit 전 `git diff --check`와 staged diff 확인
