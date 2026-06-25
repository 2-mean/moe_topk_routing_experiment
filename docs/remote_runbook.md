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

## 5. Detached Background Runs

Use this route for stable unattended execution. It creates a detached `tmux`
session, writes logs under `/tmp/2020110906_matryo_topk/logs`, and keeps raw
artifacts under `/tmp/2020110906_matryo_topk/runs`.

```bash
cd ~/moe_topk_routing_experiment
bash scripts/launch_background.sh scratch_pilot smoke configs/scratch_pilot.json cuda
bash scripts/check_background.sh scratch_pilot
bash scripts/launch_background.sh scratch_pilot full configs/scratch_pilot.json cuda
bash scripts/check_background.sh scratch_pilot
```

Follow-up experiments should be run one at a time:

```bash
bash scripts/launch_background.sh same_compute_pilot smoke configs/same_compute_pilot.json cuda
bash scripts/check_background.sh same_compute_pilot
bash scripts/launch_background.sh same_compute_pilot full configs/same_compute_pilot.json cuda

bash scripts/launch_background.sh curated_probe_pilot smoke configs/curated_probe_pilot.json cuda
bash scripts/check_background.sh curated_probe_pilot
bash scripts/launch_background.sh curated_probe_pilot full configs/curated_probe_pilot.json cuda
```

If a run fails, inspect the latest log reported by `check_background.sh` before
starting another run.

## 6. Larger Robustness Runs

The first larger run expands the same-compute pilot from three seeds to nine
seeds while keeping model size, data, and train-k schedule unchanged:

```bash
bash scripts/launch_background.sh robust_same_compute_9seed smoke configs/robust_same_compute_9seed.json cuda
bash scripts/check_background.sh robust_same_compute_9seed
bash scripts/launch_background.sh robust_same_compute_9seed full configs/robust_same_compute_9seed.json cuda
bash scripts/check_background.sh robust_same_compute_9seed
```

Run the curated nine-seed follow-up only after the same-compute nine-seed run
has completed:

```bash
bash scripts/launch_background.sh robust_curated_probe_9seed smoke configs/robust_curated_probe_9seed.json cuda
bash scripts/check_background.sh robust_curated_probe_9seed
bash scripts/launch_background.sh robust_curated_probe_9seed full configs/robust_curated_probe_9seed.json cuda
```

Or queue it automatically after the same-compute nine-seed full run passes its
summary gates:

```bash
bash scripts/launch_robust_followup_queue.sh
```

The larger model config is a stress test. Do not launch its full run until its
smoke run exits successfully:

```bash
bash scripts/launch_background.sh large_model_same_compute_pilot smoke configs/large_model_same_compute_pilot.json cuda
bash scripts/check_background.sh large_model_same_compute_pilot
```

## 7. Copy Compact Results Into Repo

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

The helper script performs the same compact copy for any named run:

```bash
bash scripts/collect_compact_results.sh scratch_pilot results/scratch_pilot_summary full
bash scripts/collect_compact_results.sh same_compute_pilot results/same_compute_summary full
bash scripts/collect_compact_results.sh curated_probe_pilot results/curated_probe_summary full
bash scripts/collect_compact_results.sh robust_same_compute_9seed results/robust_same_compute_9seed_summary full
bash scripts/collect_compact_results.sh robust_curated_probe_9seed results/robust_curated_probe_9seed_summary full
```
