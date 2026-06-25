#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
RUN_NAME="${RUN_NAME:-deep_kgrid_same_compute_10seed}"
CONFIG="${CONFIG:-configs/deep_kgrid_same_compute_10seed.json}"
DEVICE="${DEVICE:-cuda}"
SESSION="${SESSION:-matryo_topk_deep_kgrid_queue}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/${RUN_NAME}_queue_${STAMP}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "queue session already exists: $SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd '$REPO_DIR'
source '$HOME/miniconda3/bin/activate' '$ENV_NAME'
export PYTHONPATH=\"\$PWD/src\"

latest_run_dir() {
  local run_name=\"\$1\"
  local mode=\"\$2\"
  ls -td '$OUTPUT_ROOT'/runs/\"\$run_name\"/*_\"\$mode\"_* 2>/dev/null | head -1
}

require_summary_gate() {
  local run_name=\"\$1\"
  local mode=\"\$2\"
  local expected=\"\$3\"
  local run_dir
  run_dir=\"\$(latest_run_dir \"\$run_name\" \"\$mode\")\"
  if [ -z \"\$run_dir\" ] || [ ! -f \"\$run_dir/summary.md\" ]; then
    echo \"[fail] summary not found for \$run_name \$mode\" >&2
    exit 1
  fi
  echo \"[summary] \$run_dir\"
  sed -n '1,90p' \"\$run_dir/summary.md\"
  grep -q \"Expected completed runs: \$expected\" \"\$run_dir/summary.md\"
  grep -q \"Completed final matched runs: \$expected\" \"\$run_dir/summary.md\"
  grep -q \"logit cutoff sanity nestedness min: 1.0\" \"\$run_dir/summary.md\"
  grep -q \"collapsed final matched runs: 0\" \"\$run_dir/summary.md\"
  if [ \"\$mode\" = \"full\" ]; then
    grep -q \"step-0 same-W0 same-infer nestedness min: 1.0\" \"\$run_dir/summary.md\"
  fi
}

{
  echo '[queue-launch]' '$STAMP'
  echo '[session]' '$SESSION'
  echo '[repo]' \"\$PWD\"
  echo '[config]' '$CONFIG'
  quota -s || true
  df -h \"\$HOME\" /tmp || true
  nvidia-smi || true

  echo '[smoke-start]' \"\$(date)\"
  python -m moe_topk.scratch_pilot \
    --mode smoke \
    --config '$CONFIG' \
    --output-root '$OUTPUT_ROOT' \
    --device '$DEVICE' \
    --run-name '$RUN_NAME' \
    --timestamp '$STAMP'
  require_summary_gate '$RUN_NAME' smoke 1

  echo '[full-start]' \"\$(date)\"
  python -m moe_topk.scratch_pilot \
    --mode full \
    --config '$CONFIG' \
    --output-root '$OUTPUT_ROOT' \
    --device '$DEVICE' \
    --run-name '$RUN_NAME' \
    --timestamp '$STAMP'

  require_summary_gate '$RUN_NAME' full 80
  LATEST=\"\$(latest_run_dir '$RUN_NAME' full)\"
  echo '[diagnostics-start]' \"\$LATEST\" \"\$(date)\"
  python scripts/analyze_routing_diagnostics.py \
    --run-dir \"\$LATEST\" \
    --out-dir \"\$LATEST/diagnostics\" \
    --n-experts 8
  echo '[queue-done]' \"\$(date)\"
} 2>&1 | tee '$LOG_PATH'
"

sleep 1
tmux has-session -t "$SESSION"
echo "started queue session: $SESSION"
echo "queue log: $LOG_PATH"
