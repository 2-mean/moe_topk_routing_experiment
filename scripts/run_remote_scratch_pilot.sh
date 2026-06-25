#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
MODE="${1:-smoke}"
CONFIG="${2:-configs/scratch_pilot.json}"
RUN_NAME="${3:-scratch_pilot}"

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"

python -m moe_topk.scratch_pilot \
  --mode "$MODE" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --run-name "$RUN_NAME"
