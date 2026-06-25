#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
MODE="${1:-smoke}"

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"

python -m moe_topk.scratch_pilot \
  --mode "$MODE" \
  --config configs/scratch_pilot.json \
  --output-root "$OUTPUT_ROOT" \
  --device cuda

