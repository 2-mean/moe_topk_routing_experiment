# Feasible Experiment Scope

Status date: 2026-06-26 KST

This note records only experiments that are feasible on the current Dongguk
server path:

- server: `ssh -p 101 2020110906@cs.dongguk.edu`
- environment: `~/miniconda3/envs/cas4160`
- GPU: RTX 3080 Ti 12GB
- raw output root: `/tmp/2020110906_matryo_topk`
- home quota: about 50GB and nearly full
- `/tmp` free space observed during planning: about 75GB

The goal is to avoid writing down experiments that sound desirable but are not
credible under the current hardware and software constraints.

## Feasible Now

These use the existing scratch MoE code and do not require new dependencies.

| Run | Status | Feasible action |
|---|---|---|
| `scratch_pilot` | completed | Keep compact summary in `results/scratch_pilot_summary/`. |
| `same_compute_pilot` | completed | Keep compact summary in `results/same_compute_summary/`. |
| `curated_probe_pilot` | completed | Keep compact summary in `results/curated_probe_summary/`. |
| `robust_same_compute_9seed` | running | Wait for completion, gate-check, then collect compact summary. |
| `robust_curated_probe_9seed` | queued | Run only after `robust_same_compute_9seed` passes gates. |
| `large_model_same_compute_pilot` | smoke passed | Full run is feasible as a scratch stress test, but lower priority than the 9-seed robustness runs. |

Acceptance gates for these runs:

- completed matched final runs equal expected count
- logit cutoff sanity nestedness min is `1.0`
- step-0 same-W0 same-infer nestedness min is `1.0` for full runs
- collapsed final matched runs is `0`
- no raw `.npz` routes or checkpoints are committed

## Feasible With Small Setup

These are feasible only after installing lightweight Hugging Face dependencies
and keeping all model/cache files under `/tmp`, not home.

### Qwen1.5-MoE Inference-Only Routing Pilot

Candidate model:

- `Qwen/Qwen1.5-MoE-A2.7B`
- Hugging Face page: `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B`
- Model-card facts checked on 2026-06-26: total parameters about `14.3B`,
  activated parameters about `2.7B`, BF16 weights.

Feasible scope:

- load with quantization/offload, not plain BF16
- run a small fixed probe set
- capture router logits or selected expert ids where the model implementation
  exposes them
- compare inference-time `k={1,2,4}` at router-logit level
- save compact CSV/summary only

Required setup:

- install or stage `transformers`, `accelerate`, `safetensors`,
  `huggingface_hub`
- likely install `bitsandbytes` if 4-bit or 8-bit loading is used
- set `HF_HOME=/tmp/2020110906_hf_cache`
- run a load-only smoke before any routing extraction

Minimum success criteria:

- model/tokenizer load succeeds without using home quota
- one tiny probe batch runs without OOM
- routing hooks return non-empty expert scores or expert ids
- output includes model name, dtype/quantization mode, device map, and peak GPU
  memory

This is a feasibility pilot, not a fine-tuning experiment.

## Feasible Later, Not First

### Large Scratch Full

`configs/large_model_same_compute_pilot.json` passed smoke on the current GPU.
The full run is feasible if the 9-seed robustness queue finishes cleanly and GPU
time is still available.

Use it only as a scratch scaling/stress check. It does not replace the Qwen
pretrained setting.

## Current Scope Exclusions

These are not recorded as executable next steps for the current server:

- full Qwen1.5-MoE BF16 loading on the 12GB GPU
- Qwen1.5-MoE `train_k={1,2,4}` fine-tuning as a main claim experiment
- Qwen3-30B-A3B full inference or fine-tuning on this single 12GB GPU
- any Qwen fine-tuning result that would be used as a strong research claim
  without a larger GPU or a much more careful low-rank/quantized protocol

Reason:

- `Qwen/Qwen1.5-MoE-A2.7B` is already too large for plain BF16 on 12GB.
- `Qwen/Qwen3-30B-A3B` is much larger: its model card reports about `30.5B`
  total parameters and `3.3B` activated parameters.
- The current conda environment does not include the Hugging Face stack needed
  for Qwen loading.
- Home quota is nearly full, so any model-cache experiment must be isolated to
  `/tmp`.

## Practical Order

1. Finish `robust_same_compute_9seed`.
2. Let the queued `robust_curated_probe_9seed` run only if gates pass.
3. Collect and push compact summaries for both robust runs.
4. If still useful, run `large_model_same_compute_pilot` full.
5. Separately implement a Qwen1.5-MoE inference-only load/routing smoke under
   `/tmp` cache.

Do not start Qwen fine-tuning on this server unless the hardware or protocol
changes.
