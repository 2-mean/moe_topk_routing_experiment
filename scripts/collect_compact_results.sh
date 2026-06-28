#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
RUN_NAME="${1:?usage: collect_compact_results.sh RUN_NAME [RESULT_DIR] [MODE]}"
RESULT_DIR="${2:-results/${RUN_NAME}_summary}"
MODE="${3:-full}"

cd "$REPO_DIR"

LATEST="$(ls -td "$OUTPUT_ROOT/runs/$RUN_NAME"/*_"$MODE"_* 2>/dev/null | head -1 || true)"
if [ -z "$LATEST" ]; then
  echo "no run found for $RUN_NAME mode=$MODE" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR/plots"
cp "$LATEST/summary.md" "$RESULT_DIR/summary.md"
cp "$LATEST/metrics.csv" "$RESULT_DIR/metrics.csv"
cp "$LATEST/env.txt" "$RESULT_DIR/env.txt"
cp "$LATEST/task_metrics.csv" "$RESULT_DIR/task_metrics.csv" 2>/dev/null || true
cp "$LATEST/checkpoint_manifest.csv" "$RESULT_DIR/checkpoint_manifest.csv" 2>/dev/null || true
cp "$LATEST"/plots/*.png "$RESULT_DIR/plots/" 2>/dev/null || true
if [ -d "$LATEST/diagnostics" ]; then
  mkdir -p "$RESULT_DIR/diagnostics"
  cp "$LATEST"/diagnostics/*.csv "$RESULT_DIR/diagnostics/" 2>/dev/null || true
  cp "$LATEST"/diagnostics/*.md "$RESULT_DIR/diagnostics/" 2>/dev/null || true
fi
for diagnostic_name in task_specialization task_loss_grid router_representation full_ranking; do
  if [ -d "$LATEST/diagnostics/$diagnostic_name" ]; then
    mkdir -p "$RESULT_DIR/diagnostics/$diagnostic_name/plots"
    if [ "$diagnostic_name" = "full_ranking" ]; then
      rm -f "$RESULT_DIR/diagnostics/$diagnostic_name/plots/empirical_adjusted_overlap_pair_layer_heatmap.png"
    fi
    cp "$LATEST"/diagnostics/"$diagnostic_name"/*.csv "$RESULT_DIR/diagnostics/$diagnostic_name/" 2>/dev/null || true
    cp "$LATEST"/diagnostics/"$diagnostic_name"/*.md "$RESULT_DIR/diagnostics/$diagnostic_name/" 2>/dev/null || true
    cp "$LATEST"/diagnostics/"$diagnostic_name"/plots/*.png "$RESULT_DIR/diagnostics/$diagnostic_name/plots/" 2>/dev/null || true
  fi
done
echo "$LATEST" > "$RESULT_DIR/raw_run_path.txt"

echo "copied compact results from $LATEST to $RESULT_DIR"
