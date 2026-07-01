# MoE Top-k Routing Experiment

학습 시 `top-k`와 추론 시 `top-k`의 불일치가 TinyMoE의 routing 구조와
validation loss에 어떤 영향을 주는지 검증하는 실험 저장소다. 현재 기준
주요 결과는 32-expert, 8-seed, two-budget 실험이며, 모든 수치와 해석 범위는
[`docs/sparse32_full_report.md`](docs/sparse32_full_report.md)에 정리되어 있다.

가장 강한 관측은 high-k로 학습한 모델을 low-k로 추론하는 `hi→lo` 전환의
비용이다. `k=8→1` mismatch delta는 fixed-step에서 `+1.507`,
same-compute에서 `+1.613`이었고 두 budget의 8개 seed에서 같은 방향으로
재현되었다. 이 결과는 합성 TinyMoE 범위의 관측이며 pretrained MoE에 대한
일반화나 인과 주장은 하지 않는다.

## 처음 clone한 뒤

Python 3.10 이상을 사용한다. CUDA 환경에서는 먼저 해당 장비에 맞는
PyTorch를 설치하거나 기존 GPU 환경을 활성화한 뒤 editable install을 한다.

```bash
git clone https://github.com/2-mean/moe_topk_routing_experiment.git
cd moe_topk_routing_experiment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests
python -m moe_topk.scratch_pilot --help
```

Windows PowerShell의 활성화 명령은 다음과 같다.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests
```

학교 서버의 기존 환경에서는 대용량 PyTorch 재설치를 피할 수 있다.

```bash
source ~/miniconda3/bin/activate cas4160
python -m pip install -e . --no-deps
python - <<'PY'
import torch, numpy, matplotlib, scipy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

설치, 디렉터리 역할, 결과 재생성, Git 동기화, 원격 실행의 전체 절차는
[`docs/repository_handoff.md`](docs/repository_handoff.md)를 기준으로 한다.

## 가장 짧은 검증과 실행

CPU smoke는 코드 경로 확인용이다. GPU 실험 결과와 성능을 재현하려면
`--device cuda`와 기록된 설정을 사용한다.

```bash
python -m moe_topk.scratch_pilot \
  --mode smoke \
  --config configs/scratch_pilot.json \
  --output-root /tmp/moe_topk_smoke \
  --device cpu
```

서버에서 현재 주 실험인 Sparse32 fixed-step smoke를 실행하려면:

```bash
python -m moe_topk.scratch_pilot \
  --mode smoke \
  --config configs/sparse32_kgrid_fixed_step_3seed.json \
  --output-root /tmp/2020110906_matryo_topk \
  --device cuda
```

장시간 실행은 `tmux` helper를 사용한다.

```bash
bash scripts/launch_background.sh RUN_NAME smoke CONFIG_PATH cuda
bash scripts/check_background.sh RUN_NAME
bash scripts/launch_background.sh RUN_NAME full CONFIG_PATH cuda
bash scripts/check_background.sh RUN_NAME
```

## 저장소 지도

| 경로 | 역할 |
|---|---|
| `src/moe_topk/` | TinyMoE 모델, 합성/JSONL 데이터, routing metric, 실험 runner |
| `configs/` | pilot, robustness, Sparse32 k-grid 설정 |
| `scripts/` | 원격 실행, combined run, 재분석, baseline, 그림/발표자료 생성 |
| `tests/` | 모델 sparse dispatch, metric, ranking, artifact, combined-run 테스트 |
| `results/` | Git에 보존한 compact CSV/summary/plot과 최종 report figure |
| `docs/` | 실행 runbook, 전체 보고서, 발표자료, 저장소 인수인계 문서 |

## 문서 읽는 순서

1. [`docs/repository_handoff.md`](docs/repository_handoff.md): 다른 환경에서
   clone 후 바로 이어서 작업하는 기준 문서
2. [`docs/sparse32_full_report.md`](docs/sparse32_full_report.md): 최신 핵심 결과,
   수치, claim 가능 범위
3. [`docs/remote_runbook.md`](docs/remote_runbook.md): 학교 서버 운영 절차
4. [`docs/presentation_export.md`](docs/presentation_export.md): PNG/HTML 발표자료 재생성
5. [`docs/feasible_experiment_scope.md`](docs/feasible_experiment_scope.md): scratch
   실험과 pretrained Qwen 범위의 경계

## 결과와 Git 보존 정책

Git에는 source/config/document와 compact 결과만 보존한다. raw route `.npz`,
checkpoint `.pt/.pth`, 전체 `runs/`는 용량 때문에 추적하지 않는다.

- clone만으로 가능한 작업: 테스트, CPU/GPU 새 실행, committed CSV/summary 기반
  분석, 보고서·PNG·HTML 발표자료 재생성
- clone만으로 불가능한 작업: 과거 raw route를 다시 읽는 layer/token-level 분석,
  과거 checkpoint에서 시작하는 ablation

과거 raw artifact가 필요한 작업은 서버의
`/tmp/2020110906_matryo_topk`가 남아 있는지 먼저 확인한다.

## 핵심 sanity gate

- 동일 router logit에서 추출한 top-k cutoff의 overlap recall은 `1.0`
- 동일 seed와 초기 가중치 `W0`의 step-0 동일 inference-k 비교는 `1.0`
- final matched run에서 max expert share가 `0.9`를 넘으면 collapse로 표시
- `nestedness`는 완전한 subset 확률이 아니라 smaller-k expert의 per-token
  overlap recall

현재 결과와 제한 사항은 문서의 snapshot보다 코드와 compact artifact를 우선해
판단한다. 실행 전에는 항상 `git pull --ff-only`와 테스트를 먼저 수행한다.
