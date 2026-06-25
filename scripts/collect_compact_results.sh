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
cp "$LATEST"/plots/*.png "$RESULT_DIR/plots/" 2>/dev/null || true
echo "$LATEST" > "$RESULT_DIR/raw_run_path.txt"

echo "copied compact results from $LATEST to $RESULT_DIR"

