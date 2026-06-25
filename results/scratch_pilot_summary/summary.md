# Scratch Pilot Summary

Run directory: `/tmp/2020110906_matryo_topk/runs/scratch_pilot/20260625_183427_full_fb0_b32_d128`
Expected completed runs: 9
Completed final matched runs: 9
Final checkpoint step: 1500

## Gates

- logit cutoff sanity nestedness min: 1.0
- step-0 same-W0 same-infer nestedness min: 1.0
- same-weight different-infer-path nestedness min: 0.9096205357142857
- collapse threshold: 0.9
- collapsed final matched runs: 0

## Final Matched Train-k Pair Metrics

| pair | metric | mean | values |
|---|---:|---:|---|
| 1-2 | nestedness | 0.6225 | 0.6241, 0.6237, 0.6198 |
| 1-2 | spearman | 0.5278 | 0.5358, 0.5183, 0.5292 |
| 1-2 | top1_agreement | 0.4089 | 0.4083, 0.4004, 0.4179 |
| 1-4 | nestedness | 0.8327 | 0.8253, 0.8311, 0.8417 |
| 1-4 | spearman | 0.5072 | 0.5129, 0.4888, 0.5198 |
| 1-4 | top1_agreement | 0.3919 | 0.3917, 0.3802, 0.4039 |
| 2-4 | nestedness | 0.9086 | 0.9066, 0.9103, 0.9091 |
| 2-4 | spearman | 0.7301 | 0.7200, 0.7313, 0.7391 |
| 2-4 | top1_agreement | 0.6514 | 0.6456, 0.6395, 0.6692 |

## Interpretation Guardrail

Use `candidate effect` only when at least two of three seeds move in the same direction.
The same-weight different-infer-path row is not a sanity gate; it can fall below 1 because earlier MoE layers change hidden states when inference_k changes.
If a gate fails, treat the run as an execution/measurement failure before making research claims.
