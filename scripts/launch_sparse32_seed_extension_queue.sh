#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
SESSION="${SESSION:-matryo_topk_sparse32_seed_extension}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/sparse32_seed_extension_${STAMP}.log"
RUNNER="$LOG_DIR/sparse32_seed_extension_${STAMP}.runner.sh"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "seed extension session already exists: $SESSION" >&2
  exit 2
fi

cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

latest_run_dir() {
  local run_name="$1"
  local mode="$2"
  ls -td "$OUTPUT_ROOT/runs/$run_name"/*_"$mode"_* 2>/dev/null | head -1
}

require_summary_gate() {
  local run_name="$1"
  local mode="$2"
  local expected="$3"
  local run_dir
  run_dir="$(latest_run_dir "$run_name" "$mode")"
  test -n "$run_dir"
  test -f "$run_dir/summary.md"
  grep -q "Expected completed runs: $expected" "$run_dir/summary.md"
  grep -q "Completed final matched runs: $expected" "$run_dir/summary.md"
  grep -q "logit cutoff sanity nestedness min: 1.0" "$run_dir/summary.md"
  grep -q "collapsed final matched runs: 0" "$run_dir/summary.md"
  echo "[summary-ok] $run_dir"
}

require_full_artifacts() {
  local run_dir="$1"
  test "$(($(wc -l < "$run_dir/manifest.csv") - 1))" -eq 640
  test "$(($(wc -l < "$run_dir/task_metrics.csv") - 1))" -eq 4480
  test "$(($(wc -l < "$run_dir/checkpoint_manifest.csv") - 1))" -eq 40
  test "$(find "$run_dir/checkpoints" -maxdepth 1 -name 'router_seed*_train*_step*_float32.pt' | wc -l)" -eq 40
  test "$(find "$run_dir/checkpoints" -maxdepth 1 -name 'final_seed*_train*_step*_float16.pt' | wc -l)" -eq 0
}

run_suite() {
  local run_name="$1"
  local config="$2"
  echo "[suite-smoke-start] $run_name $(date)"
  python -m moe_topk.scratch_pilot \
    --mode smoke \
    --config "$config" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda \
    --run-name "$run_name" \
    --timestamp "$STAMP"
  require_summary_gate "$run_name" smoke 1
  local smoke_dir
  smoke_dir="$(latest_run_dir "$run_name" smoke)"
  test "$(($(wc -l < "$smoke_dir/task_metrics.csv") - 1))" -eq 112
  test "$(find "$smoke_dir/checkpoints" -maxdepth 1 -name 'router_seed*_train*_step10_float32.pt' | wc -l)" -eq 1

  echo "[suite-full-start] $run_name $(date)"
  python -m moe_topk.scratch_pilot \
    --mode full \
    --config "$config" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda \
    --run-name "$run_name" \
    --timestamp "$STAMP"
  require_summary_gate "$run_name" full 40
  local full_dir
  full_dir="$(latest_run_dir "$run_name" full)"
  require_full_artifacts "$full_dir"
  python scripts/analyze_routing_diagnostics.py \
    --run-dir "$full_dir" \
    --out-dir "$full_dir/diagnostics" \
    --n-experts 32
  python scripts/analyze_task_specialization.py \
    --run-dir "$full_dir" \
    --out-dir "$full_dir/diagnostics/task_specialization"
  python scripts/analyze_task_loss_grid.py \
    --task-metrics "$full_dir/task_metrics.csv" \
    --out-dir "$full_dir/diagnostics/task_loss_grid"
  bash scripts/collect_compact_results.sh "$run_name" "results/${run_name}_summary" full
  echo "[suite-done] $run_name $(date)"
}

echo "[seed-extension-launch] $(date)"
df -h /tmp "$HOME" || true
nvidia-smi || true
available_kb="$(df --output=avail -k /tmp | tail -1 | tr -d ' ')"
if [ "$available_kb" -lt 31457280 ]; then
  echo "[fail] less than 30 GiB available on /tmp" >&2
  exit 1
fi

run_suite sparse32_kgrid_fixed_step_seed3to7 configs/sparse32_kgrid_fixed_step_seed3to7.json
run_suite sparse32_kgrid_same_compute_seed3to7 configs/sparse32_kgrid_same_compute_seed3to7.json
echo "[seed-extension-done] $(date)"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" env \
  REPO_DIR="$REPO_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ENV_NAME="$ENV_NAME" \
  SESSION="$SESSION" \
  STAMP="$STAMP" \
  bash -lc "bash '$RUNNER' 2>&1 | tee '$LOG_PATH'"

sleep 1
tmux has-session -t "$SESSION"
echo "started seed extension session: $SESSION"
echo "seed extension log: $LOG_PATH"
