#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
SOURCE_RUN="${1:-robust_same_compute_9seed}"
NEXT_RUN="${2:-robust_curated_probe_9seed}"
NEXT_CONFIG="${3:-configs/robust_curated_probe_9seed.json}"
DEVICE="${4:-cuda}"
EXPECTED_FULL="${EXPECTED_FULL:-27}"

cd "$REPO_DIR"

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[wait] $session still running at $(date)"
    sleep 60
  done
}

latest_run_dir() {
  local run_name="$1"
  local mode="$2"
  ls -td "$OUTPUT_ROOT/runs/$run_name"/*_"$mode"_* 2>/dev/null | head -1
}

require_summary_gate() {
  local run_name="$1"
  local mode="$2"
  local expected_completed="$3"
  local run_dir
  run_dir="$(latest_run_dir "$run_name" "$mode")"
  if [ -z "$run_dir" ] || [ ! -f "$run_dir/summary.md" ]; then
    echo "[fail] summary not found for $run_name $mode" >&2
    exit 1
  fi
  echo "[summary] $run_dir"
  sed -n '1,80p' "$run_dir/summary.md"
  grep -q "Completed final matched runs: $expected_completed" "$run_dir/summary.md"
  grep -q "logit cutoff sanity nestedness min: 1.0" "$run_dir/summary.md"
  grep -q "collapsed final matched runs: 0" "$run_dir/summary.md"
  if [ "$mode" = "full" ]; then
    grep -q "step-0 same-W0 same-infer nestedness min: 1.0" "$run_dir/summary.md"
  fi
}

echo "[queue] waiting for $SOURCE_RUN full"
wait_for_session "matryo_topk_${SOURCE_RUN}"
require_summary_gate "$SOURCE_RUN" full "$EXPECTED_FULL"
bash scripts/collect_compact_results.sh "$SOURCE_RUN" "results/${SOURCE_RUN}_summary" full

echo "[queue] launching $NEXT_RUN smoke"
bash scripts/launch_background.sh "$NEXT_RUN" smoke "$NEXT_CONFIG" "$DEVICE"
wait_for_session "matryo_topk_${NEXT_RUN}"
require_summary_gate "$NEXT_RUN" smoke 1

echo "[queue] launching $NEXT_RUN full"
bash scripts/launch_background.sh "$NEXT_RUN" full "$NEXT_CONFIG" "$DEVICE"
wait_for_session "matryo_topk_${NEXT_RUN}"
require_summary_gate "$NEXT_RUN" full "$EXPECTED_FULL"
bash scripts/collect_compact_results.sh "$NEXT_RUN" "results/${NEXT_RUN}_summary" full

echo "[queue] done at $(date)"
