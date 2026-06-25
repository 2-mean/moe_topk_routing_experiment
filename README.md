# MoE Top-k Routing Experiment

This repository runs a small, self-contained MoE pilot for checking whether
training-time top-k changes routing patterns beyond a simple inference-time
cutoff.

The first executable target is a scratch GPT-style MoE pilot. It is deliberately
small enough for the Dongguk server GPU:

- server: `ssh -p 101 2020110906@cs.dongguk.edu`
- preferred environment: `~/miniconda3/envs/cas4160`
- GPU observed during planning: RTX 3080 Ti 12GB
- raw outputs: `/tmp/2020110906_matryo_topk`
- committed outputs only: compact summaries under `results/`

For the current boundary between feasible runs and out-of-scope Qwen work, see
`docs/feasible_experiment_scope.md`.

For completed settings, gate results, metrics, and objective interpretation so
far, see `docs/experiment_results_so_far.md`.

## Quick Start On Server

```bash
git clone https://github.com/2-mean/moe_topk_routing_experiment.git ~/moe_topk_routing_experiment
cd ~/moe_topk_routing_experiment
git checkout -b scratch-pilot-v0
tmux new -s matryo_topk
```

Inside tmux:

```bash
source ~/miniconda3/bin/activate cas4160
export PYTHONPATH="$PWD/src"
python -m moe_topk.scratch_pilot --mode smoke --config configs/scratch_pilot.json --output-root /tmp/2020110906_matryo_topk --device cuda
python -m moe_topk.scratch_pilot --mode full --config configs/scratch_pilot.json --output-root /tmp/2020110906_matryo_topk --device cuda
```

For detached background execution, prefer the helper scripts:

```bash
bash scripts/launch_background.sh scratch_pilot smoke configs/scratch_pilot.json cuda
bash scripts/check_background.sh scratch_pilot
bash scripts/launch_background.sh scratch_pilot full configs/scratch_pilot.json cuda
bash scripts/check_background.sh scratch_pilot
bash scripts/collect_compact_results.sh scratch_pilot results/scratch_pilot_summary full
```

The full pilot runs `train_k={1,2,4}` x `seed={0,1,2}`. It writes raw route
tables as `.npz` files under `/tmp`, not into git.

## Follow-up Pilots

The initial pilot uses equal training steps. Two follow-up configs are included
to test whether the result survives stricter checks:

- `configs/same_compute_pilot.json`: approximates equal compute by running
  `train_k=1` longer than `train_k=2`, and `train_k=2` longer than `train_k=4`.
- `configs/curated_probe_pilot.json`: uses the same compute schedule but
  evaluates routing on fixed JSONL prompts in `probes/curated_probe.jsonl`.

Run them one at a time on the single GPU:

```bash
bash scripts/launch_background.sh same_compute_pilot smoke configs/same_compute_pilot.json cuda
bash scripts/check_background.sh same_compute_pilot
bash scripts/launch_background.sh same_compute_pilot full configs/same_compute_pilot.json cuda

bash scripts/launch_background.sh curated_probe_pilot smoke configs/curated_probe_pilot.json cuda
bash scripts/check_background.sh curated_probe_pilot
bash scripts/launch_background.sh curated_probe_pilot full configs/curated_probe_pilot.json cuda
```

## Larger Runs

The next robustness tier keeps the model and compute schedule fixed, but expands
from three seeds to nine seeds:

```bash
bash scripts/launch_background.sh robust_same_compute_9seed smoke configs/robust_same_compute_9seed.json cuda
bash scripts/check_background.sh robust_same_compute_9seed
bash scripts/launch_background.sh robust_same_compute_9seed full configs/robust_same_compute_9seed.json cuda
```

After it completes, collect only compact artifacts:

```bash
bash scripts/collect_compact_results.sh robust_same_compute_9seed results/robust_same_compute_9seed_summary full
```

`configs/robust_curated_probe_9seed.json` repeats the nine-seed check on the
fixed JSONL probe. `configs/large_model_same_compute_pilot.json` is a larger
model stress test; run smoke first before committing GPU time to a full run.

To queue the curated nine-seed follow-up after the same-compute nine-seed run
passes its gates:

```bash
bash scripts/launch_robust_followup_queue.sh
```

## What Gets Checked

- same-W0 gate: all `train_k` models for the same seed start from the same
  initialization.
- logit cutoff gate: top-1/top-2/top-4 derived from the same router logits must
  be nested.
- same-W0 gate: models with the same seed and the same inference_k must match at
  step 0.
- same-weight different-inference-k paths are reported but are not a strict
  sanity gate, because earlier MoE layer outputs can change later router inputs.
- collapse gate: final matched runs with max expert share over `0.9` are marked
  collapsed.
- interpretation guardrail: call an effect a `candidate effect` only when at
  least two of three seeds move in the same direction.

## Outputs

Each run directory contains:

- `env.txt`
- `config.json`
- `manifest.csv`
- `train_log.csv`
- `metrics.csv`
- `summary.md`
- `plots/*.png`
- `routes/*.npz`

Only compact files should be copied into `results/scratch_pilot_summary/`.
