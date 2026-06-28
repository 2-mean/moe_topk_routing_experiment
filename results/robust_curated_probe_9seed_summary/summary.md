# Robust Curated Probe 9Seed Summary

Run directory: `/tmp/2020110906_matryo_topk/runs/robust_curated_probe_9seed/20260626_125741_full_fb0_b32_d128`
Expected completed runs: 27
Completed final matched runs: 27
Final checkpoint step: 6000

## Gates

- logit cutoff sanity nestedness min: 1.0
- step-0 same-W0 same-infer nestedness min: 1.0
- same-weight different-infer-path nestedness min: 0.9162016369047619
- collapse threshold: 0.9
- collapsed final matched runs: 0

## Final Matched Train-k Pair Metrics

| pair | metric | mean | values |
|---|---:|---:|---|
| 1-2 | nestedness | 0.5533 | 0.5547, 0.5466, 0.5631, 0.5512, 0.5629, 0.5455, 0.5530, 0.5568, 0.5459 |
| 1-2 | spearman | 0.4538 | 0.4571, 0.4397, 0.4734, 0.4466, 0.4679, 0.4432, 0.4461, 0.4608, 0.4492 |
| 1-2 | top1_agreement | 0.3452 | 0.3610, 0.3414, 0.3579, 0.3327, 0.3459, 0.3461, 0.3424, 0.3472, 0.3321 |
| 1-4 | nestedness | 0.7913 | 0.7940, 0.7844, 0.7919, 0.7877, 0.7975, 0.7829, 0.7873, 0.8081, 0.7883 |
| 1-4 | spearman | 0.4415 | 0.4367, 0.4327, 0.4546, 0.4310, 0.4486, 0.4311, 0.4380, 0.4625, 0.4388 |
| 1-4 | top1_agreement | 0.3373 | 0.3391, 0.3316, 0.3570, 0.3280, 0.3366, 0.3291, 0.3290, 0.3490, 0.3360 |
| 2-4 | nestedness | 0.9025 | 0.8984, 0.9007, 0.9103, 0.9095, 0.8904, 0.8977, 0.9049, 0.9146, 0.8956 |
| 2-4 | spearman | 0.7245 | 0.7132, 0.7272, 0.7388, 0.7315, 0.7007, 0.7185, 0.7328, 0.7447, 0.7131 |
| 2-4 | top1_agreement | 0.6070 | 0.5959, 0.6060, 0.6149, 0.5956, 0.5906, 0.6107, 0.6097, 0.6355, 0.6046 |

## Final Mismatch Cost

| train_k | inference_k | mean_delta_loss | values |
|---:|---:|---:|---|
| 1 | 2 | 0.0637 | 0.0483, 0.1058, 0.0575, -0.0184, 0.0667, 0.0908, 0.0113, 0.1267, 0.0842 |
| 1 | 4 | -0.0148 | -0.0911, 0.0459, -0.0321, -0.1276, -0.0337, 0.0662, -0.0765, 0.0858, 0.0295 |
| 2 | 1 | 0.1827 | 0.2843, 0.1907, 0.1989, 0.2053, 0.2791, 0.0285, 0.2114, 0.2458, 0.0008 |
| 2 | 4 | -0.3285 | -0.3155, -0.3514, -0.3340, -0.2933, -0.3768, -0.2807, -0.4202, -0.2705, -0.3137 |
| 4 | 1 | 0.7000 | 0.8906, 0.5913, 0.5444, 0.8120, 0.7665, 0.5622, 0.8052, 0.6526, 0.6747 |
| 4 | 2 | 0.3890 | 0.4605, 0.3118, 0.3670, 0.4620, 0.4396, 0.3143, 0.4601, 0.3305, 0.3548 |

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
