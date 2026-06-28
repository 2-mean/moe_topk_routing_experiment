#!/usr/bin/env bash
set -u -o pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"

cd "$REPO_DIR" || exit 1
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

latest_run_dir() {
  local run_name="$1"
  local mode="${2:-full}"
  ls -td "$OUTPUT_ROOT/runs/$run_name"/*_"$mode"_* 2>/dev/null | head -1
}

summary_gate() {
  local run_dir="$1"
  local expected="$2"
  local summary_path="$run_dir/summary.md"

  if [ ! -f "$summary_path" ]; then
    echo "[gate-fail] no summary: $run_dir"
    return 1
  fi
  grep -q "Expected completed runs: $expected" "$summary_path" || return 1
  grep -q "Completed final matched runs: $expected" "$summary_path" || return 1
  grep -q "logit cutoff sanity nestedness min: 1.0" "$summary_path" || return 1
  grep -q "collapsed final matched runs: 0" "$summary_path" || return 1
}

copy_compact_with_diagnostics() {
  local run_name="$1"
  local run_dir="$2"
  local result_dir="results/${run_name}_summary"

  bash scripts/collect_compact_results.sh "$run_name" "$result_dir" full || return 1
  if [ -d "$run_dir/diagnostics" ]; then
    mkdir -p "$result_dir/diagnostics"
    cp "$run_dir"/diagnostics/*.csv "$result_dir/diagnostics/" 2>/dev/null || true
    cp "$run_dir"/diagnostics/*.md "$result_dir/diagnostics/" 2>/dev/null || true
  fi
  echo "$run_dir" > "$result_dir/raw_full_run_path.txt"
}

backfill_one() {
  local run_name="$1"
  local expected="$2"
  local n_experts="$3"
  local reanalyze_first="${4:-0}"
  local run_dir

  run_dir="$(latest_run_dir "$run_name" full)"
  if [ -z "$run_dir" ]; then
    echo "[skip] no full run for $run_name"
    return 0
  fi

  echo "[run] $run_name"
  echo "[dir] $run_dir"

  if [ "$reanalyze_first" = "1" ] || [ ! -f "$run_dir/summary.md" ]; then
    echo "[reanalyze] $run_name"
    python scripts/reanalyze_run.py --run-dir "$run_dir" || return 1
  fi

  if ! summary_gate "$run_dir" "$expected"; then
    echo "[gate-fail] $run_name expected=$expected"
    sed -n '1,80p' "$run_dir/summary.md" 2>/dev/null || true
    return 1
  fi

  if [ ! -f "$run_dir/diagnostics/routing_diagnostics_summary.md" ]; then
    echo "[diagnostics] $run_name"
    python scripts/analyze_routing_diagnostics.py \
      --run-dir "$run_dir" \
      --out-dir "$run_dir/diagnostics" \
      --n-experts "$n_experts" || return 1
  else
    echo "[diagnostics] already exists for $run_name"
  fi

  echo "[collect] $run_name"
  copy_compact_with_diagnostics "$run_name" "$run_dir" || return 1
  echo "[done] $run_name"
}

main() {
  local failures=0

  echo "[analysis-backfill-start] $(date)"
  quota -s || true
  df -h "$HOME" /tmp || true

  backfill_one scratch_pilot 9 8 0 || failures=$((failures + 1))
  backfill_one same_compute_pilot 9 8 0 || failures=$((failures + 1))
  backfill_one curated_probe_pilot 9 8 0 || failures=$((failures + 1))
  backfill_one robust_same_compute_9seed 27 8 1 || failures=$((failures + 1))
  backfill_one robust_curated_probe_9seed 27 8 0 || failures=$((failures + 1))
  backfill_one deep_kgrid_same_compute_10seed 80 8 0 || failures=$((failures + 1))

  echo "[analysis-backfill-done] failures=$failures $(date)"
  return "$failures"
}

main "$@"
