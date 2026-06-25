#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
SESSION="${SESSION:-matryo_topk_robust_followup_queue}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/robust_followup_queue_${STAMP}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "queue session already exists: $SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" bash -lc "
set -euo pipefail
cd '$REPO_DIR'
bash scripts/run_robust_followup_queue.sh >> '$LOG_PATH' 2>&1
"

sleep 1
tmux has-session -t "$SESSION"
echo "started queue session: $SESSION"
echo "queue log: $LOG_PATH"
