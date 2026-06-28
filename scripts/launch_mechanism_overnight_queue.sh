#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
SESSION="${SESSION:-matryo_topk_mechanism_overnight_queue}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/mechanism_overnight_queue_${STAMP}.log"

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
bash scripts/run_mechanism_overnight_queue.sh 2>&1 | tee '$LOG_PATH'
"

sleep 1
tmux has-session -t "$SESSION"
echo "started queue session: $SESSION"
echo "queue log: $LOG_PATH"
