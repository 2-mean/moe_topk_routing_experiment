#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
SESSION="${SESSION:-matryo_topk_sparse32_mechanism_analysis}"
FIXED_RUN="${FIXED_RUN:-$OUTPUT_ROOT/runs/sparse32_kgrid_fixed_step_3seed/20260627_165426_full_fb0_b16_d192}"
SAME_RESULT="${SAME_RESULT:-$REPO_DIR/results/sparse32_kgrid_mechanism_3seed_summary}"
FIXED_RESULT="${FIXED_RESULT:-$REPO_DIR/results/sparse32_kgrid_fixed_step_3seed_summary}"
BUDGET_RESULT="${BUDGET_RESULT:-$REPO_DIR/results/sparse32_budget_comparison_3seed}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/sparse32_mechanism_analysis_${STAMP}.log"
RUNNER="$LOG_DIR/sparse32_mechanism_analysis_${STAMP}.runner.sh"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "analysis session already exists: $SESSION" >&2
  exit 2
fi

cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

echo "[analysis-launch] $(date)"
echo "[fixed-run] $FIXED_RUN"
nvidia-smi || true

echo "[budget-compare-start] $(date)"
python scripts/compare_sparse32_budgets.py \
  --same-dir "$SAME_RESULT" \
  --fixed-dir "$FIXED_RESULT" \
  --out-dir "$BUDGET_RESULT"

echo "[task-grid-start] $(date)"
python scripts/analyze_task_loss_grid.py \
  --task-metrics "$FIXED_RUN/task_metrics.csv" \
  --out-dir "$FIXED_RUN/diagnostics/task_loss_grid"

SMOKE_OUT="$OUTPUT_ROOT/mechanism_smoke/$STAMP"
echo "[mechanism-smoke-start] $(date)"
python scripts/analyze_router_representation_mechanism.py \
  --run-dir "$FIXED_RUN" \
  --out-dir "$SMOKE_OUT" \
  --device cuda \
  --seeds 0 \
  --train-ks 1 2 \
  --probe-samples-per-category 1 \
  --probe-batch-size 7 \
  --transplant-samples-per-category 1 \
  --transplant-batch-size 7
test "$(($(wc -l < "$SMOKE_OUT/logit_shapley_decomposition.csv") - 1))" -eq 8
test "$(($(wc -l < "$SMOKE_OUT/router_transplant_loss.csv") - 1))" -eq 36
python - "$SMOKE_OUT/logit_shapley_decomposition.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
residual = max(abs(float(row["mean_abs_decomposition_residual"])) for row in rows)
assert residual < 1e-5, residual
print("[mechanism-smoke-residual]", residual)
PY

FULL_OUT="$FIXED_RUN/diagnostics/router_representation"
echo "[mechanism-full-start] $(date)"
python scripts/analyze_router_representation_mechanism.py \
  --run-dir "$FIXED_RUN" \
  --out-dir "$FULL_OUT" \
  --device cuda \
  --seeds 0 1 2 \
  --train-ks 1 2 3 4 5 6 7 8 \
  --probe-samples-per-category 8 \
  --probe-batch-size 8 \
  --transplant-samples-per-category 2 \
  --transplant-batch-size 14

test "$(($(wc -l < "$FULL_OUT/logit_shapley_decomposition.csv") - 1))" -eq 672
test "$(($(wc -l < "$FULL_OUT/route_component_metrics.csv") - 1))" -eq 10080
test "$(($(wc -l < "$FULL_OUT/router_transplant_loss.csv") - 1))" -eq 1728
python - "$FULL_OUT/logit_shapley_decomposition.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
residual = max(abs(float(row["mean_abs_decomposition_residual"])) for row in rows)
assert residual < 1e-5, residual
print("[mechanism-full-residual]", residual)
PY

bash scripts/collect_compact_results.sh \
  sparse32_kgrid_fixed_step_3seed \
  results/sparse32_kgrid_fixed_step_3seed_summary \
  full
echo "[analysis-done] $(date)"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" env \
  REPO_DIR="$REPO_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ENV_NAME="$ENV_NAME" \
  SESSION="$SESSION" \
  FIXED_RUN="$FIXED_RUN" \
  SAME_RESULT="$SAME_RESULT" \
  FIXED_RESULT="$FIXED_RESULT" \
  BUDGET_RESULT="$BUDGET_RESULT" \
  STAMP="$STAMP" \
  bash -lc "bash '$RUNNER' 2>&1 | tee '$LOG_PATH'"

sleep 1
tmux has-session -t "$SESSION"
echo "started analysis session: $SESSION"
echo "analysis log: $LOG_PATH"
