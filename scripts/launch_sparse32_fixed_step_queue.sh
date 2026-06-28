#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
RUN_NAME="${RUN_NAME:-sparse32_kgrid_fixed_step_3seed}"
CONFIG="${CONFIG:-configs/sparse32_kgrid_fixed_step_3seed.json}"
DEVICE="${DEVICE:-cuda}"
SESSION="${SESSION:-matryo_topk_sparse32_fixed_step_queue}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_PATH="$LOG_DIR/${RUN_NAME}_queue_${STAMP}.log"
RUNNER="$LOG_DIR/${RUN_NAME}_queue_${STAMP}.runner.sh"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "queue session already exists: $SESSION" >&2
  exit 2
fi

cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"
source "$HOME/miniconda3/bin/activate" "$ENV_NAME"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

latest_run_dir() {
  local run_name="$1"
  local mode="$2"
  ls -td "$OUTPUT_ROOT/runs/$run_name"/*_"$mode"_* 2>/dev/null | head -1
}

require_summary_gate() {
  local run_name="$1"
  local mode="$2"
  local expected="$3"
  local run_dir
  run_dir="$(latest_run_dir "$run_name" "$mode")"
  if [ -z "$run_dir" ] || [ ! -f "$run_dir/summary.md" ]; then
    echo "[fail] summary not found for $run_name $mode" >&2
    exit 1
  fi
  echo "[summary] $run_dir"
  sed -n '1,45p' "$run_dir/summary.md"
  grep -q "Expected completed runs: $expected" "$run_dir/summary.md"
  grep -q "Completed final matched runs: $expected" "$run_dir/summary.md"
  grep -q "logit cutoff sanity nestedness min: 1.0" "$run_dir/summary.md"
  grep -q "collapsed final matched runs: 0" "$run_dir/summary.md"
}

require_full_artifacts() {
  local run_dir="$1"
  test -f "$run_dir/task_metrics.csv"
  test -f "$run_dir/checkpoint_manifest.csv"
  test "$(find "$run_dir/checkpoints" -maxdepth 1 -name 'final_seed*_train*_step1500_float16.pt' | wc -l)" -eq 24
  test "$(find "$run_dir/checkpoints" -maxdepth 1 -name 'router_seed*_train*_step1500_float32.pt' | wc -l)" -eq 24
  test "$(($(wc -l < "$run_dir/task_metrics.csv") - 1))" -eq 2688
  test "$(($(wc -l < "$run_dir/checkpoint_manifest.csv") - 1))" -eq 48
}

require_smoke_artifacts() {
  local run_dir="$1"
  test -f "$run_dir/task_metrics.csv"
  test -f "$run_dir/checkpoint_manifest.csv"
  test "$(find "$run_dir/checkpoints" -maxdepth 1 -name 'final_seed0_train1_step10_float16.pt' | wc -l)" -eq 1
  test "$(find "$run_dir/checkpoints" -maxdepth 1 -name 'router_seed0_train1_step10_float32.pt' | wc -l)" -eq 1
  test "$(($(wc -l < "$run_dir/task_metrics.csv") - 1))" -eq 112
  test "$(($(wc -l < "$run_dir/checkpoint_manifest.csv") - 1))" -eq 2
  python - "$run_dir" <<'PY'
import sys
from pathlib import Path

import torch

run_dir = Path(sys.argv[1])
full = torch.load(next((run_dir / "checkpoints").glob("final_*.pt")), map_location="cpu", weights_only=True)
router = torch.load(next((run_dir / "checkpoints").glob("router_*.pt")), map_location="cpu", weights_only=True)
assert full["state_dict"]
assert router["state_dict"]
assert all(value.dtype == torch.float16 for value in full["state_dict"].values() if value.is_floating_point())
assert all(".moe.router." in f".{key}" for key in router["state_dict"])
print("[smoke-checkpoint-load-ok]", len(full["state_dict"]), len(router["state_dict"]))
PY
}

echo "[queue-launch] $(date)"
echo "[session] $SESSION"
echo "[repo] $PWD"
echo "[config] $CONFIG"
quota -s || true
df -h "$HOME" /tmp || true
nvidia-smi || true
python - <<'PY'
from moe_topk.model import TopKMoE
from moe_topk.scratch_pilot import task_loss_rows
print("fixed-step queue import ok", TopKMoE.__name__, task_loss_rows({"a": 2.0}, {"a": 2}, {"a": 0}))
PY

echo "[smoke-start] $(date)"
python -m moe_topk.scratch_pilot \
  --mode smoke \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --run-name "$RUN_NAME" \
  --timestamp "$STAMP"
require_summary_gate "$RUN_NAME" smoke 1
SMOKE_LATEST="$(latest_run_dir "$RUN_NAME" smoke)"
require_smoke_artifacts "$SMOKE_LATEST"

echo "[full-start] $(date)"
python -m moe_topk.scratch_pilot \
  --mode full \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --run-name "$RUN_NAME" \
  --timestamp "$STAMP"
require_summary_gate "$RUN_NAME" full 24

LATEST="$(latest_run_dir "$RUN_NAME" full)"
require_full_artifacts "$LATEST"
echo "[diagnostics-start] $LATEST $(date)"
python scripts/analyze_routing_diagnostics.py \
  --run-dir "$LATEST" \
  --out-dir "$LATEST/diagnostics" \
  --n-experts 32
python scripts/analyze_task_specialization.py \
  --run-dir "$LATEST" \
  --out-dir "$LATEST/diagnostics/task_specialization"
bash scripts/collect_compact_results.sh "$RUN_NAME" "results/${RUN_NAME}_summary" full
echo "[queue-done] $(date)"
RUNNER

chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" env \
  REPO_DIR="$REPO_DIR" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ENV_NAME="$ENV_NAME" \
  RUN_NAME="$RUN_NAME" \
  CONFIG="$CONFIG" \
  DEVICE="$DEVICE" \
  SESSION="$SESSION" \
  STAMP="$STAMP" \
  bash -lc "bash '$RUNNER' 2>&1 | tee '$LOG_PATH'"

sleep 1
tmux has-session -t "$SESSION"
echo "started queue session: $SESSION"
echo "queue log: $LOG_PATH"
