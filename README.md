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

The full pilot runs `train_k={1,2,4}` x `seed={0,1,2}`. It writes raw route
tables as `.npz` files under `/tmp`, not into git.

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
