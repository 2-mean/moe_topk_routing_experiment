#!/usr/bin/env bash
set -u -o pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
DEVICE="${DEVICE:-cuda}"
WAIT_SESSION="${WAIT_SESSION:-matryo_topk_deep_kgrid_queue}"
TMP_MIN_GB="${TMP_MIN_GB:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

wait_for_session() {
  local session="$1"
  while has_session "$session"; do
    log "[wait] $session still running"
    sleep 60
  done
}

tmp_free_gb() {
  df -BG /tmp | awk 'NR==2 {gsub(/G/, "", $4); print $4 + 0}'
}

ensure_tmp_space() {
  local free_gb
  free_gb="$(tmp_free_gb)"
  if [ "$free_gb" -lt "$TMP_MIN_GB" ]; then
    log "[warn] /tmp free ${free_gb}G < ${TMP_MIN_GB}G"
    return 1
  fi
  return 0
}

latest_run_dir() {
  local run_name="$1"
  local mode="$2"
  ls -td "$OUTPUT_ROOT/runs/$run_name"/*_"$mode"_* 2>/dev/null | head -1
}

check_summary_gate() {
  local run_name="$1"
  local mode="$2"
  local expected="$3"
  local run_dir
  local summary_path

  run_dir="$(latest_run_dir "$run_name" "$mode")"
  if [ -z "$run_dir" ]; then
    log "[warn] no run dir for $run_name mode=$mode"
    return 1
  fi
  summary_path="$run_dir/summary.md"
  if [ ! -f "$summary_path" ]; then
    log "[warn] no summary for $run_name mode=$mode"
    return 1
  fi

  grep -q "Expected completed runs: $expected" "$summary_path" || return 1
  grep -q "Completed final matched runs: $expected" "$summary_path" || return 1
  grep -q "logit cutoff sanity nestedness min: 1.0" "$summary_path" || return 1
  grep -q "collapsed final matched runs: 0" "$summary_path" || return 1
  if [ "$mode" = "full" ] && grep -q "step-0 same-W0 same-infer nestedness min:" "$summary_path"; then
    grep -q "step-0 same-W0 same-infer nestedness min: 1.0" "$summary_path" || return 1
  fi
  return 0
}

run_diagnostics() {
  local run_name="$1"
  local n_experts="$2"
  local run_dir
  local out_dir

  run_dir="$(latest_run_dir "$run_name" "full")"
  if [ -z "$run_dir" ]; then
    log "[warn] diagnostics skipped, no full run for $run_name"
    return 1
  fi
  out_dir="$run_dir/diagnostics"
  if [ -f "$out_dir/routing_diagnostics_summary.md" ]; then
    log "[diag] already exists for $run_name"
    return 0
  fi

  log "[diag] generating diagnostics for $run_name"
  python scripts/analyze_routing_diagnostics.py \
    --run-dir "$run_dir" \
    --out-dir "$out_dir" \
    --n-experts "$n_experts"
}

collect_with_diagnostics() {
  local run_name="$1"
  local result_dir="results/${run_name}_summary"
  local run_dir

  run_dir="$(latest_run_dir "$run_name" "full")"
  if [ -z "$run_dir" ]; then
    log "[warn] collect skipped, no full run for $run_name"
    return 1
  fi

  log "[collect] compact results for $run_name"
  bash scripts/collect_compact_results.sh "$run_name" "$result_dir" "full" || return 1
  if [ -d "$run_dir/diagnostics" ]; then
    mkdir -p "$result_dir/diagnostics"
    cp "$run_dir"/diagnostics/*.csv "$result_dir/diagnostics/" 2>/dev/null || true
    cp "$run_dir"/diagnostics/*.md "$result_dir/diagnostics/" 2>/dev/null || true
  fi
  echo "$run_dir" > "$result_dir/raw_full_run_path.txt"
}

launch_and_wait() {
  local run_name="$1"
  local mode="$2"
  local config="$3"
  local session="matryo_topk_${run_name}"

  if has_session "$session"; then
    log "[wait] existing session detected: $session"
    wait_for_session "$session"
  fi

  bash scripts/launch_background.sh "$run_name" "$mode" "$config" "$DEVICE" || return 1
  wait_for_session "$session"
  return 0
}

run_pipeline() {
  local run_name="$1"
  local config="$2"
  local expected_full="$3"
  local n_experts="$4"
  local attempt

  log "[pipeline] start $run_name with $config"

  if check_summary_gate "$run_name" "full" "$expected_full"; then
    log "[pipeline] already completed: $run_name"
    run_diagnostics "$run_name" "$n_experts" || true
    collect_with_diagnostics "$run_name" || true
    return 0
  fi

  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    log "[pipeline] attempt $attempt/$MAX_ATTEMPTS for $run_name"
    ensure_tmp_space || log "[warn] low /tmp before $run_name attempt $attempt"

    if ! check_summary_gate "$run_name" "smoke" "1"; then
      launch_and_wait "$run_name" "smoke" "$config" || log "[warn] smoke launch failed for $run_name"
      if ! check_summary_gate "$run_name" "smoke" "1"; then
        log "[warn] smoke gate failed for $run_name on attempt $attempt"
        continue
      fi
    fi

    ensure_tmp_space || log "[warn] low /tmp before full run: $run_name"
    launch_and_wait "$run_name" "full" "$config" || {
      log "[warn] full launch failed for $run_name on attempt $attempt"
      continue
    }
    if check_summary_gate "$run_name" "full" "$expected_full"; then
      run_diagnostics "$run_name" "$n_experts" || log "[warn] diagnostics failed for $run_name"
      collect_with_diagnostics "$run_name" || log "[warn] collect failed for $run_name"
      log "[pipeline] completed $run_name"
      return 0
    fi
    log "[warn] full gate failed for $run_name on attempt $attempt"
  done

  log "[error] exhausted retries for $run_name"
  return 1
}

main() {
  cd "$REPO_DIR"
  source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
  export PYTHONPATH="$PWD/src"

  log "[queue] mechanism overnight queue started"
  quota -s || true
  df -h /tmp "$HOME" || true
  nvidia-smi || true

  if has_session "$WAIT_SESSION"; then
    log "[queue] waiting for prerequisite session: $WAIT_SESSION"
    wait_for_session "$WAIT_SESSION"
  fi

  if check_summary_gate "deep_kgrid_same_compute_10seed" "full" "80"; then
    log "[queue] deep_kgrid full gate passed"
    run_diagnostics "deep_kgrid_same_compute_10seed" "8" || true
    collect_with_diagnostics "deep_kgrid_same_compute_10seed" || true
  else
    log "[warn] deep_kgrid full gate not ready; continuing follow-up queue"
  fi

  run_pipeline "robust_curated_probe_9seed" "configs/robust_curated_probe_9seed.json" "27" "8" || true
  run_pipeline "large_model_same_compute_pilot" "configs/large_model_same_compute_pilot.json" "9" "12" || true
  run_pipeline "robust_same_compute_aux0_9seed" "configs/robust_same_compute_aux0_9seed.json" "27" "8" || true
  run_pipeline "robust_same_compute_aux005_9seed" "configs/robust_same_compute_aux005_9seed.json" "27" "8" || true

  log "[queue] mechanism overnight queue finished"
}

main "$@"
