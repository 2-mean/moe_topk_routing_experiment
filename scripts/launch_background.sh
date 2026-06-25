#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
RUN_NAME="${1:?usage: launch_background.sh RUN_NAME MODE CONFIG [DEVICE]}"
MODE="${2:?usage: launch_background.sh RUN_NAME MODE CONFIG [DEVICE]}"
CONFIG="${3:?usage: launch_background.sh RUN_NAME MODE CONFIG [DEVICE]}"
DEVICE="${4:-cuda}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="matryo_topk_${RUN_NAME}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/${RUN_NAME}_${MODE}_${STAMP}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd '$REPO_DIR'
source '$HOME/miniconda3/bin/activate' '$ENV_NAME'
export PYTHONPATH=\"\$PWD/src\"
{
  echo '[launch]' '$STAMP'
  echo '[session]' '$SESSION'
  echo '[repo]' \"\$PWD\"
  echo '[config]' '$CONFIG'
  echo '[mode]' '$MODE'
  quota -s || true
  df -h \"\$HOME\" /tmp || true
  nvidia-smi || true
  python -m moe_topk.scratch_pilot \
    --mode '$MODE' \
    --config '$CONFIG' \
    --output-root '$OUTPUT_ROOT' \
    --device '$DEVICE' \
    --run-name '$RUN_NAME' \
    --timestamp '$STAMP'
} 2>&1 | tee '$LOG_PATH'
"

sleep 1
tmux has-session -t "$SESSION"
echo "started session: $SESSION"
echo "log: $LOG_PATH"

