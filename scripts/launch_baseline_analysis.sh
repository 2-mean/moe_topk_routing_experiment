#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
BASELINE_DIR="$OUTPUT_ROOT/baselines"

mkdir -p "$BASELINE_DIR/fixed_step_8seed" "$BASELINE_DIR/same_compute_8seed"

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"

FIXED_RUN="$OUTPUT_ROOT/runs/sparse32_kgrid_fixed_step_8seed/20260628_142529_full_fb0_b16_d192"
MECH_RUN="$OUTPUT_ROOT/runs/sparse32_kgrid_same_compute_8seed/20260628_142529_full_fb0_b16_d192"

echo "[fixed-step] starting..."
python scripts/analyze_training_variability_null.py \
    --run-dir "$FIXED_RUN" \
    --out-dir "$BASELINE_DIR/fixed_step_8seed" \
    --n-experts 32 \
    2>&1 | tee "$BASELINE_DIR/fixed_step_8seed_run.log"

echo "[same-compute] starting..."
python scripts/analyze_training_variability_null.py \
    --run-dir "$MECH_RUN" \
    --out-dir "$BASELINE_DIR/same_compute_8seed" \
    --n-experts 32 \
    2>&1 | tee "$BASELINE_DIR/same_compute_8seed_run.log"

echo "[done] all baseline analyses complete"
