#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
SESSION="${SESSION:-matryo_topk_sparse32_8seed_analysis}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/sparse32_8seed_analysis_${STAMP}.log"
RUNNER="$LOG_DIR/sparse32_8seed_analysis_${STAMP}.runner.sh"

FIXED_3="$OUTPUT_ROOT/runs/sparse32_kgrid_fixed_step_3seed/20260627_165426_full_fb0_b16_d192"
FIXED_5="$OUTPUT_ROOT/runs/sparse32_kgrid_fixed_step_seed3to7/20260627_213141_full_fb0_b16_d192"
SAME_3="$OUTPUT_ROOT/runs/sparse32_kgrid_mechanism_3seed/20260626_135134_full_fb0_b16_d192"
SAME_5="$OUTPUT_ROOT/runs/sparse32_kgrid_same_compute_seed3to7/20260627_213141_full_fb0_b16_d192"
FIXED_8="$OUTPUT_ROOT/runs/sparse32_kgrid_fixed_step_8seed/${STAMP}_full_fb0_b16_d192"
SAME_8="$OUTPUT_ROOT/runs/sparse32_kgrid_same_compute_8seed/${STAMP}_full_fb0_b16_d192"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "8-seed analysis session already exists: $SESSION" >&2
  exit 2
fi

cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src:$OUTPUT_ROOT/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

analyze_combined() {
  local run_name="$1"
  local source_a="$2"
  local source_b="$3"
  local run_dir="$4"
  local result_dir="results/${run_name}_summary"

  echo "[8seed-build-start] $run_name $(date)"
  python scripts/build_combined_run.py \
    --source-run "$source_a" \
    --source-run "$source_b" \
    --output-run "$run_dir" \
    --experiment-name "$run_name"
  test "$(($(wc -l < "$run_dir/manifest.csv") - 1))" -eq 1024
  test "$(find "$run_dir/routes" -maxdepth 1 -type l | wc -l)" -eq 1024

  echo "[8seed-standard-start] $run_name $(date)"
  ionice -c3 nice -n 10 python scripts/summarize_combined_run.py --run-dir "$run_dir"
  ionice -c3 nice -n 10 python scripts/analyze_routing_diagnostics.py \
    --run-dir "$run_dir" --out-dir "$run_dir/diagnostics"
  ionice -c3 nice -n 10 python scripts/analyze_task_specialization.py \
    --run-dir "$run_dir" --out-dir "$run_dir/diagnostics/task_specialization"
  ionice -c3 nice -n 10 python scripts/analyze_task_loss_grid.py \
    --task-metrics "$run_dir/task_metrics.csv" \
    --out-dir "$run_dir/diagnostics/task_loss_grid"

  echo "[8seed-ranking-start] $run_name $(date)"
  ionice -c3 nice -n 10 python scripts/analyze_full_ranking.py \
    --run-dir "$run_dir" \
    --out-dir "$run_dir/diagnostics/full_ranking" \
    --seeds 0 1 2 3 4 5 6 7 \
    --train-ks 1 2 3 4 5 6 7 8
  test "$(($(wc -l < "$run_dir/diagnostics/full_ranking/full_ranking_summary_by_seed.csv") - 1))" -eq 2016
  test "$(($(wc -l < "$run_dir/diagnostics/full_ranking/selected_intersection_histogram.csv") - 1))" -eq 8064
  test -f "$run_dir/diagnostics/full_ranking/plots/hypergeometric_calibrated_overlap_pair_layer_heatmap.png"
  test -f "$run_dir/diagnostics/full_ranking/plots/frequency_adjusted_overlap_pair_layer_heatmap.png"

  bash scripts/collect_compact_results.sh "$run_name" "$result_dir" full
  cp "$run_dir/source_runs.txt" "$result_dir/source_runs.txt"
  echo "[8seed-done] $run_name $(date)"
}

echo "[8seed-analysis-launch] $(date)"
analyze_combined sparse32_kgrid_fixed_step_8seed "$FIXED_3" "$FIXED_5" "$FIXED_8"
analyze_combined sparse32_kgrid_same_compute_8seed "$SAME_3" "$SAME_5" "$SAME_8"
echo "[8seed-analysis-all-done] $(date)"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" env \
  REPO_DIR="$REPO_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ENV_NAME="$ENV_NAME" \
  FIXED_3="$FIXED_3" FIXED_5="$FIXED_5" FIXED_8="$FIXED_8" \
  SAME_3="$SAME_3" SAME_5="$SAME_5" SAME_8="$SAME_8" \
  bash -lc "bash '$RUNNER' 2>&1 | tee '$LOG_PATH'"

sleep 1
tmux has-session -t "$SESSION"
echo "started 8-seed analysis session: $SESSION"
echo "8-seed analysis log: $LOG_PATH"
echo "fixed combined run: $FIXED_8"
echo "same combined run: $SAME_8"
