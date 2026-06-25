#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
RUN_NAME="${1:?usage: check_background.sh RUN_NAME}"
SESSION="matryo_topk_${RUN_NAME}"
LOG_DIR="$OUTPUT_ROOT/logs"

echo "== session =="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "alive: $SESSION"
else
  echo "dead: $SESSION"
fi

echo "== gpu =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo "== quota =="
quota -s || true
df -h /tmp || true
du -sh "$OUTPUT_ROOT" 2>/dev/null || true

echo "== latest log =="
LATEST_LOG="$(ls -t "$LOG_DIR"/"${RUN_NAME}"_*.log 2>/dev/null | head -1 || true)"
if [ -n "$LATEST_LOG" ]; then
  echo "$LATEST_LOG"
  tail -80 "$LATEST_LOG"
else
  echo "no log found"
fi

echo "== latest run =="
LATEST_RUN="$(ls -td "$OUTPUT_ROOT/runs/$RUN_NAME"/* 2>/dev/null | head -1 || true)"
if [ -n "$LATEST_RUN" ]; then
  echo "$LATEST_RUN"
  if [ -f "$LATEST_RUN/summary.md" ]; then
    sed -n '1,120p' "$LATEST_RUN/summary.md"
  else
    echo "summary.md not present yet"
  fi
  if [ -f "$LATEST_RUN/failure.txt" ]; then
    echo "failure.txt:"
    tail -80 "$LATEST_RUN/failure.txt"
  fi
else
  echo "no run directory found"
fi

