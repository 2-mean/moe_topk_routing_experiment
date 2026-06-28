#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
SESSION="${SESSION:-matryo_topk_full_ranking_analysis}"
FIXED_RUN="${FIXED_RUN:-$OUTPUT_ROOT/runs/sparse32_kgrid_fixed_step_3seed/20260627_165426_full_fb0_b16_d192}"
SAME_RUN="${SAME_RUN:-$OUTPUT_ROOT/runs/sparse32_kgrid_mechanism_3seed/20260626_135134_full_fb0_b16_d192}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/full_ranking_analysis_${STAMP}.log"
RUNNER="$LOG_DIR/full_ranking_analysis_${STAMP}.runner.sh"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "full ranking session already exists: $SESSION" >&2
  exit 2
fi

cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src:$OUTPUT_ROOT/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

run_ranking() {
  local run_dir="$1"
  local run_name="$2"
  local out_dir="$run_dir/diagnostics/full_ranking"
  echo "[ranking-start] $run_name $(date)"
  ionice -c3 nice -n 10 python scripts/analyze_full_ranking.py \
    --run-dir "$run_dir" \
    --out-dir "$out_dir" \
    --seeds 0 1 2 \
    --train-ks 1 2 3 4 5 6 7 8
  test "$(($(wc -l < "$out_dir/rank_transition_matrix.csv") - 1))" -eq 258048
  test "$(($(wc -l < "$out_dir/topm_overlap_curve.csv") - 1))" -eq 24192
  test "$(($(wc -l < "$out_dir/rank_displacement.csv") - 1))" -eq 8064
  test "$(($(wc -l < "$out_dir/full_ranking_summary_by_seed.csv") - 1))" -eq 756
  test "$(($(wc -l < "$out_dir/full_ranking_summary_mean.csv") - 1))" -eq 252
  test "$(($(wc -l < "$out_dir/selected_intersection_histogram.csv") - 1))" -eq 3024
  test -f "$out_dir/rank_transition_seed_counts.npz"
  grep -q "selected_union_kendall_tau" "$out_dir/full_ranking_summary_mean.csv"
  grep -q "hypergeometric_calibrated_overlap" "$out_dir/full_ranking_summary_mean.csv"
  grep -q "frequency_adjusted_overlap" "$out_dir/full_ranking_summary_mean.csv"
  test -f "$out_dir/plots/selected_jaccard_pair_layer_heatmap.png"
  test -f "$out_dir/plots/selected_union_kendall_pair_layer_heatmap.png"
  test -f "$out_dir/plots/common_selected_kendall_pair_layer_heatmap.png"
  test -f "$out_dir/plots/common_pairwise_order_agreement_pair_layer_heatmap.png"
  test -f "$out_dir/plots/common_selected_exact_order_pair_layer_heatmap.png"
  test -f "$out_dir/plots/ordered_containment_pair_layer_heatmap.png"
  test -f "$out_dir/plots/hypergeometric_calibrated_overlap_pair_layer_heatmap.png"
  test -f "$out_dir/plots/frequency_adjusted_overlap_pair_layer_heatmap.png"
  test -f "$out_dir/plots/common_pair_coverage_pair_layer_heatmap.png"
  bash scripts/collect_compact_results.sh "$run_name" "results/${run_name}_summary" full
  echo "[ranking-done] $run_name $(date)"
}

echo "[full-ranking-launch] $(date)"
run_ranking "$FIXED_RUN" sparse32_kgrid_fixed_step_3seed
run_ranking "$SAME_RUN" sparse32_kgrid_mechanism_3seed
echo "[full-ranking-all-done] $(date)"
RUNNER

chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" env \
  REPO_DIR="$REPO_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ENV_NAME="$ENV_NAME" \
  SESSION="$SESSION" \
  FIXED_RUN="$FIXED_RUN" \
  SAME_RUN="$SAME_RUN" \
  STAMP="$STAMP" \
  bash -lc "bash '$RUNNER' 2>&1 | tee '$LOG_PATH'"

sleep 1
tmux has-session -t "$SESSION"
echo "started full ranking session: $SESSION"
echo "full ranking log: $LOG_PATH"
