# Curated Probe Pilot Summary

Run directory: `/tmp/2020110906_matryo_topk/runs/curated_probe_pilot/20260625_230713_full_fb0_b32_d128`
Expected completed runs: 9
Completed final matched runs: 9
Final checkpoint step: 6000

## Gates

- logit cutoff sanity nestedness min: 1.0
- step-0 same-W0 same-infer nestedness min: 1.0
- same-weight different-infer-path nestedness min: 0.9170386904761905
- collapse threshold: 0.9
- collapsed final matched runs: 0

## Final Matched Train-k Pair Metrics

| pair | metric | mean | values |
|---|---:|---:|---|
| 1-2 | nestedness | 0.5548 | 0.5547, 0.5466, 0.5631 |
| 1-2 | spearman | 0.4567 | 0.4571, 0.4397, 0.4734 |
| 1-2 | top1_agreement | 0.3534 | 0.3610, 0.3414, 0.3579 |
| 1-4 | nestedness | 0.7901 | 0.7940, 0.7844, 0.7919 |
| 1-4 | spearman | 0.4413 | 0.4367, 0.4327, 0.4546 |
| 1-4 | top1_agreement | 0.3426 | 0.3391, 0.3316, 0.3570 |
| 2-4 | nestedness | 0.9031 | 0.8984, 0.9007, 0.9103 |
| 2-4 | spearman | 0.7264 | 0.7132, 0.7272, 0.7388 |
| 2-4 | top1_agreement | 0.6056 | 0.5959, 0.6060, 0.6149 |

## Final Mismatch Cost

| train_k | inference_k | mean_delta_loss | values |
|---:|---:|---:|---|
| 1 | 2 | 0.0705 | 0.0483, 0.1058, 0.0575 |
| 1 | 4 | -0.0258 | -0.0911, 0.0459, -0.0321 |
| 2 | 1 | 0.2246 | 0.2843, 0.1907, 0.1989 |
| 2 | 4 | -0.3336 | -0.3155, -0.3514, -0.3340 |
| 4 | 1 | 0.6754 | 0.8906, 0.5913, 0.5444 |
| 4 | 2 | 0.3798 | 0.4605, 0.3118, 0.3670 |

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
