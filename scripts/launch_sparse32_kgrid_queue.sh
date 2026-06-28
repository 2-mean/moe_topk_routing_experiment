#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/moe_topk_routing_experiment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/2020110906_matryo_topk}"
ENV_NAME="${ENV_NAME:-cas4160}"
RUN_NAME="${RUN_NAME:-sparse32_kgrid_mechanism_3seed}"
CONFIG="${CONFIG:-configs/sparse32_kgrid_mechanism_3seed.json}"
DEVICE="${DEVICE:-cuda}"
SESSION="${SESSION:-matryo_topk_sparse32_kgrid_queue}"
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
  sed -n '1,100p' "$run_dir/summary.md"
  grep -q "Expected completed runs: $expected" "$run_dir/summary.md"
  grep -q "Completed final matched runs: $expected" "$run_dir/summary.md"
  grep -q "logit cutoff sanity nestedness min: 1.0" "$run_dir/summary.md"
  grep -q "collapsed final matched runs: 0" "$run_dir/summary.md"
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
print("sparse32 queue import ok", TopKMoE.__name__)
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
echo "[diagnostics-start] $LATEST $(date)"
python scripts/analyze_routing_diagnostics.py \
  --run-dir "$LATEST" \
  --out-dir "$LATEST/diagnostics" \
  --n-experts 32
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
