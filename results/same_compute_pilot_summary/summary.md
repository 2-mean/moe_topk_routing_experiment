# Same Compute Pilot Summary

Run directory: `/tmp/2020110906_matryo_topk/runs/same_compute_pilot/20260625_192104_full_fb0_b32_d128`
Expected completed runs: 9
Completed final matched runs: 9
Final checkpoint step: 6000

## Gates

- logit cutoff sanity nestedness min: 1.0
- step-0 same-W0 same-infer nestedness min: 1.0
- same-weight different-infer-path nestedness min: 0.9096205357142857
- collapse threshold: 0.9
- collapsed final matched runs: 0

## Final Matched Train-k Pair Metrics

| pair | metric | mean | values |
|---|---:|---:|---|
| 1-2 | nestedness | 0.5180 | 0.5223, 0.5140, 0.5177 |
| 1-2 | spearman | 0.3992 | 0.4056, 0.3918, 0.4004 |
| 1-2 | top1_agreement | 0.3165 | 0.3169, 0.3062, 0.3265 |
| 1-4 | nestedness | 0.7570 | 0.7455, 0.7600, 0.7656 |
| 1-4 | spearman | 0.3846 | 0.3768, 0.3862, 0.3907 |
| 1-4 | top1_agreement | 0.3055 | 0.2948, 0.2973, 0.3244 |
| 2-4 | nestedness | 0.8998 | 0.8956, 0.8992, 0.9046 |
| 2-4 | spearman | 0.7130 | 0.6967, 0.7172, 0.7250 |
| 2-4 | top1_agreement | 0.6416 | 0.6346, 0.6330, 0.6573 |

## Final Mismatch Cost

| train_k | inference_k | mean_delta_loss | values |
|---:|---:|---:|---|
| 1 | 2 | 0.0315 | 0.0214, 0.0223, 0.0509 |
| 1 | 4 | 0.1743 | 0.1393, 0.1890, 0.1946 |
| 2 | 1 | 0.3387 | 0.3259, 0.3547, 0.3355 |
| 2 | 4 | 0.0337 | 0.0403, 0.0306, 0.0302 |
| 4 | 1 | 1.4513 | 1.4172, 1.5375, 1.3992 |
| 4 | 2 | 0.1466 | 0.1588, 0.1531, 0.1279 |

## Candidate Direction Check

| pair | metric | seeds_below_0.95 | candidate_effect |
|---|---|---:|---|
| 1-2 | nestedness | 3/3 | true |
| 1-2 | top1_agreement | 3/3 | true |
| 1-4 | nestedness | 3/3 | true |
| 1-4 | top1_agreement | 3/3 | true |
| 2-4 | nestedness | 3/3 | true |
| 2-4 | top1_agreement | 3/3 | true |

## Interpretation Guardrail

Use `candidate effect` only when at least two of three seeds move in the same direction.
The same-weight different-infer-path row is not a sanity gate; it can fall below 1 because earlier MoE layers change hidden states when inference_k changes.
If a gate fails, treat the run as an execution/measurement failure before making research claims.
