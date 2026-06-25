# Remote Runbook

## 1. Confirm Space

```bash
quota -s
df -h "$HOME" /tmp
du -sh ~/.cache/whisper ~/.cache/pip ~/.cache/huggingface 2>/dev/null || true
```

If quota is tight, move user-owned caches to `/tmp`:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
STASH=/tmp/2020110906_quota_stash/$STAMP
mkdir -p "$STASH"
for d in "$HOME/.cache/whisper" "$HOME/.cache/pip" "$HOME/.cache/huggingface"; do
  if [ -e "$d" ]; then
    mkdir -p "$STASH/$(dirname "${d#$HOME/}")"
    mv "$d" "$STASH/${d#$HOME/}"
  fi
done
quota -s
```

## 2. Clone And Prepare

```bash
git clone https://github.com/2-mean/moe_topk_routing_experiment.git ~/moe_topk_routing_experiment
cd ~/moe_topk_routing_experiment
git checkout -b scratch-pilot-v0
source ~/miniconda3/bin/activate cas4160
export PYTHONPATH="$PWD/src"
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

## 3. Smoke

```bash
python -m moe_topk.scratch_pilot --mode smoke --config configs/scratch_pilot.json --output-root /tmp/2020110906_matryo_topk --device cuda
```

Pass criteria:

- command exits 0
- `summary.md` exists
- logit cutoff sanity nestedness min is 1.0 or effectively 1.0
- step-0 same-W0 same-infer nestedness min is 1.0 or effectively 1.0

## 4. Full Pilot

```bash
tmux new -s matryo_topk
cd ~/moe_topk_routing_experiment
source ~/miniconda3/bin/activate cas4160
export PYTHONPATH="$PWD/src"
python -m moe_topk.scratch_pilot --mode full --config configs/scratch_pilot.json --output-root /tmp/2020110906_matryo_topk --device cuda
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t matryo_topk
```

## 5. Copy Compact Results Into Repo

```bash
LATEST=$(ls -td /tmp/2020110906_matryo_topk/runs/scratch_pilot/*_full_* | head -1)
mkdir -p results/scratch_pilot_summary
cp "$LATEST"/summary.md results/scratch_pilot_summary/summary.md
cp "$LATEST"/metrics.csv results/scratch_pilot_summary/metrics.csv
cp "$LATEST"/env.txt results/scratch_pilot_summary/env.txt
mkdir -p results/scratch_pilot_summary/plots
cp "$LATEST"/plots/*.png results/scratch_pilot_summary/plots/ 2>/dev/null || true
```

Do not commit `routes/*.npz` or checkpoints.
