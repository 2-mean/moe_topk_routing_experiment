# Experiment Results So Far

Status date: 2026-06-26 KST

This document records completed and currently running experiments with objective
settings, gate outcomes, and metric observations. It intentionally avoids causal
claims that are not directly supported by the completed runs.

## Environment

- Server: `ssh -p 101 2020110906@cs.dongguk.edu`
- Environment: `~/miniconda3/envs/cas4160`
- GPU: RTX 3080 Ti 12GB
- Python: 3.10.19 in recorded run env files
- PyTorch: 2.10.0+cu128 in recorded run env files
- Raw output root: `/tmp/2020110906_matryo_topk`
- Committed output policy: compact summaries, CSV metrics, env files, and small
  plots only; raw route `.npz` files and checkpoints stay outside git.

## Metric Notes

- `logit cutoff sanity nestedness`: computed from the same router logits. This
  is expected to be `1.0` when top-1/top-2/top-4 are pure cutoffs of one ranking.
- `step-0 same-W0 same-infer nestedness`: checks that same-seed models that
  share the same initial weights agree before training, when inference_k is the
  same.
- `same-weight different-infer-path nestedness`: reported but not used as a
  strict gate because earlier MoE layers can change later hidden states when
  inference_k changes.
- `matched train-k pair metrics`: compare final matched runs, such as
  train_k=1 infer_k=1 against train_k=2 infer_k=2.
- `mismatch cost`: validation loss delta under train/inference k mismatch. It is
  reported as a measurement; sign and magnitude should not be overinterpreted
  without a larger protocol.

## Completed Runs

| Run | Config | Seeds | Train k | Steps | Probe | Raw run |
|---|---|---:|---|---|---|---|
| scratch pilot | `configs/scratch_pilot.json` | 3 | 1,2,4 | same token, 1500 each | synthetic | `/tmp/2020110906_matryo_topk/runs/scratch_pilot/20260625_183427_full_fb0_b32_d128` |
| same-compute pilot | `configs/same_compute_pilot.json` | 3 | 1,2,4 | k1=6000, k2=3000, k4=1500 | synthetic | `/tmp/2020110906_matryo_topk/runs/same_compute_pilot/20260625_192104_full_fb0_b32_d128` |
| curated-probe pilot | `configs/curated_probe_pilot.json` | 3 | 1,2,4 | k1=6000, k2=3000, k4=1500 | fixed JSONL probe | `/tmp/2020110906_matryo_topk/runs/curated_probe_pilot/20260625_230713_full_fb0_b32_d128` |

Shared model settings for the three completed pilot families:

- GPT-style scratch MoE
- `n_layers=4`
- `d_model=128`
- `n_heads=4`
- `n_experts=8`
- `expert_hidden=512`
- `seq_len=64`
- `batch_size=32`
- `eval_batch_size=64`
- `router_aux_loss_coef=0.01`
- `dropout=0.0`

## Gate Outcomes

| Run | Completed matched final runs | Logit cutoff sanity min | Step-0 same-W0 same-infer min | Collapsed final matched runs |
|---|---:|---:|---:|---:|
| scratch pilot | 9/9 | 1.0 | 1.0 | 0 |
| same-compute pilot | 9/9 | 1.0 | 1.0 | 0 |
| curated-probe pilot | 9/9 | 1.0 | 1.0 | 0 |

Objective gate interpretation:

- All completed full runs passed the router-logit cutoff sanity gate.
- All completed full runs passed the same-W0 step-0 same-inference-k gate.
- No completed full run crossed the configured expert collapse threshold.
- These gates support treating the recorded metrics as valid measurements for
  these scratch runs. They do not by themselves imply that the effect
  generalizes to pretrained Qwen-MoE models.

## Final Matched Pair Metrics

### Nestedness

| Pair | scratch pilot | same-compute pilot | curated-probe pilot |
|---|---:|---:|---:|
| 1-2 | 0.6225 | 0.5180 | 0.5548 |
| 1-4 | 0.8327 | 0.7570 | 0.7901 |
| 2-4 | 0.9086 | 0.8998 | 0.9031 |

### Top-1 Agreement

| Pair | scratch pilot | same-compute pilot | curated-probe pilot |
|---|---:|---:|---:|
| 1-2 | 0.4089 | 0.3165 | 0.3534 |
| 1-4 | 0.3919 | 0.3055 | 0.3426 |
| 2-4 | 0.6514 | 0.6416 | 0.6056 |

### Spearman

| Pair | scratch pilot | same-compute pilot | curated-probe pilot |
|---|---:|---:|---:|
| 1-2 | 0.5278 | 0.3992 | 0.4567 |
| 1-4 | 0.5072 | 0.3846 | 0.4413 |
| 2-4 | 0.7301 | 0.7130 | 0.7264 |

Objective metric observations:

- In all three completed runs, final matched pair nestedness is below `1.0` for
  every train-k pair.
- In all three completed runs, pair `2-4` has higher nestedness, top-1
  agreement, and Spearman than pairs involving `1`.
- Same-compute and curated-probe pilots show similar ordering across pairs:
  `2-4` highest, `1-4` middle for nestedness, and `1-2` lowest for nestedness.
- The completed runs measure scratch-model behavior only. They do not test a
  pretrained Qwen router.

## Final Mismatch Cost

The scratch pilot summary predates the final mismatch-cost table. Mismatch cost
is available for same-compute and curated-probe pilots.

| Train k | Inference k | same-compute mean delta loss | curated-probe mean delta loss |
|---:|---:|---:|---:|
| 1 | 2 | 0.0315 | 0.0705 |
| 1 | 4 | 0.1743 | -0.0258 |
| 2 | 1 | 0.3387 | 0.2246 |
| 2 | 4 | 0.0337 | -0.3336 |
| 4 | 1 | 1.4513 | 0.6754 |
| 4 | 2 | 0.1466 | 0.3798 |

Objective mismatch observations:

- For same-compute, all reported mismatch deltas are positive.
- For curated-probe, two up-scaling deltas are negative: train_k=1 to infer_k=4
  and train_k=2 to infer_k=4.
- Down-scaling from train_k=4 to infer_k=1 has the largest positive delta in
  both same-compute and curated-probe pilots.
- These values describe validation loss deltas on the configured probes only.
  They are not a general quality benchmark.

## Running Or Queued Runs

| Run | Status at latest check | Objective status |
|---|---|---|
| `robust_same_compute_9seed` | running | Latest observed progress: seed=7 train_k=1, step about 2900/6000, routes=369, GPU about 794MiB used, no summary yet. |
| `robust_curated_probe_9seed` | queued | `matryo_topk_robust_followup_queue` waits for robust same-compute gates before launching it. |
| `large_model_same_compute_pilot` | smoke completed | Smoke passed at batch 16, d_model 192. Full run has not been executed. |

No objective result interpretation should be made for running or queued runs
until a `summary.md` exists and gates have been checked.

## Scope Boundaries

- The completed results are scratch MoE results.
- Qwen1.5-MoE inference-only routing extraction has not been run.
- Qwen-MoE fine-tuning has not been run.
- No claim about pretrained Qwen routing behavior is supported by the completed
  results above.
- No claim about downstream task quality improvement is supported by these
  routing metrics alone.

## Current Evidence Statement

Within the completed scratch MoE pilots, changing training-time top-k is
associated with final routing tables that are not identical cutoff views of one
shared ranking across separately trained models. This statement is limited to
the configured scratch models, seeds, probes, and metrics above.

The stronger 9-seed robustness runs are still needed before increasing the
confidence of this statement.
