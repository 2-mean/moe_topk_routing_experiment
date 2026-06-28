# Robust Same Compute 9Seed Summary

Run directory: `/tmp/2020110906_matryo_topk/runs/robust_same_compute_9seed/20260625_234205_full_fb0_b32_d128`
Expected completed runs: 27
Completed final matched runs: 27
Final checkpoint step: 6000

## Gates

- logit cutoff sanity nestedness min: 1.0
- step-0 same-W0 same-infer nestedness min: 1.0
- same-weight different-infer-path nestedness min: 0.9038560267857143
- collapse threshold: 0.9
- collapsed final matched runs: 0

## Final Matched Train-k Pair Metrics

| pair | metric | mean | values |
|---|---:|---:|---|
| 1-2 | nestedness | 0.5119 | 0.5223, 0.5140, 0.5177, 0.5001, 0.5090, 0.5060, 0.5099, 0.5135, 0.5144 |
| 1-2 | spearman | 0.4022 | 0.4056, 0.3918, 0.4004, 0.3976, 0.4137, 0.3987, 0.3981, 0.4067, 0.4072 |
| 1-2 | top1_agreement | 0.3098 | 0.3169, 0.3062, 0.3265, 0.3012, 0.3094, 0.3053, 0.3165, 0.3060, 0.3006 |
| 1-4 | nestedness | 0.7562 | 0.7455, 0.7600, 0.7656, 0.7489, 0.7575, 0.7536, 0.7553, 0.7660, 0.7530 |
| 1-4 | spearman | 0.3897 | 0.3768, 0.3862, 0.3907, 0.3845, 0.3947, 0.3855, 0.3865, 0.4038, 0.3986 |
| 1-4 | top1_agreement | 0.2994 | 0.2948, 0.2973, 0.3244, 0.2923, 0.2929, 0.2924, 0.2975, 0.3012, 0.3020 |
| 2-4 | nestedness | 0.9004 | 0.8956, 0.8992, 0.9046, 0.9106, 0.8799, 0.8967, 0.9019, 0.9186, 0.8968 |
| 2-4 | spearman | 0.7133 | 0.6967, 0.7172, 0.7250, 0.7250, 0.6853, 0.7114, 0.7184, 0.7402, 0.7004 |
| 2-4 | top1_agreement | 0.6494 | 0.6346, 0.6330, 0.6573, 0.6695, 0.6318, 0.6566, 0.6585, 0.6767, 0.6263 |

## Final Mismatch Cost

| train_k | inference_k | mean_delta_loss | values |
|---:|---:|---:|---|
| 1 | 2 | 0.0363 | 0.0214, 0.0223, 0.0509, 0.0292, 0.0482, 0.0235, 0.0517, 0.0431, 0.0369 |
| 1 | 4 | 0.1844 | 0.1393, 0.1890, 0.1946, 0.1638, 0.2189, 0.1454, 0.2371, 0.1688, 0.2026 |
| 2 | 1 | 0.2911 | 0.3259, 0.3547, 0.3355, 0.2757, 0.2502, 0.3080, 0.3412, 0.2335, 0.1954 |
| 2 | 4 | 0.0246 | 0.0403, 0.0306, 0.0302, 0.0191, 0.0161, 0.0304, 0.0234, 0.0173, 0.0143 |
| 4 | 1 | 1.4204 | 1.4172, 1.5375, 1.3992, 1.5149, 1.3434, 1.6501, 1.4104, 1.1543, 1.3564 |
| 4 | 2 | 0.1425 | 0.1588, 0.1531, 0.1279, 0.1285, 0.1238, 0.2065, 0.1387, 0.0956, 0.1494 |

## Candidate Direction Check

| pair | metric | seeds_below_0.95 | candidate_effect |
|---|---|---:|---|
| 1-2 | nestedness | 9/9 | true |
| 1-2 | top1_agreement | 9/9 | true |
| 1-4 | nestedness | 9/9 | true |
| 1-4 | top1_agreement | 9/9 | true |
| 2-4 | nestedness | 9/9 | true |
| 2-4 | top1_agreement | 9/9 | true |

## Interpretation Guardrail

Use `candidate effect` only when a strict seed majority moves in the same direction.
For this run, the minimum seed support is 5.
The same-weight different-infer-path row is not a sanity gate; it can fall below 1 because earlier MoE layers change hidden states when inference_k changes.
If a gate fails, treat the run as an execution/measurement failure before making research claims.
