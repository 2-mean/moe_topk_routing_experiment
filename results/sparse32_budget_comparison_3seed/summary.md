# Sparse32 Budget Comparison

All deltas are paired by seed and defined as fixed-step minus same-compute.

## Key Routing Deltas

| pair | metric | same_mean | fixed_mean | delta | seeds_same_sign |
|---|---|---:|---:|---:|---:|
| 1-2 | nestedness | 0.2524 | 0.3607 | +0.1083 | 3/3 |
| 1-2 | top1_agreement | 0.1540 | 0.2323 | +0.0783 | 3/3 |
| 1-2 | spearman | 0.4081 | 0.5589 | +0.1508 | 3/3 |
| 1-4 | nestedness | 0.4199 | 0.5308 | +0.1109 | 3/3 |
| 1-4 | top1_agreement | 0.1619 | 0.2317 | +0.0698 | 3/3 |
| 1-4 | spearman | 0.4354 | 0.5554 | +0.1201 | 3/3 |
| 1-8 | nestedness | 0.6357 | 0.7308 | +0.0951 | 3/3 |
| 1-8 | top1_agreement | 0.1637 | 0.2345 | +0.0708 | 3/3 |
| 1-8 | spearman | 0.4546 | 0.5648 | +0.1102 | 3/3 |
| 2-4 | nestedness | 0.6476 | 0.6846 | +0.0370 | 3/3 |
| 2-4 | top1_agreement | 0.4207 | 0.4521 | +0.0314 | 3/3 |
| 2-4 | spearman | 0.6854 | 0.7298 | +0.0445 | 3/3 |
| 2-8 | nestedness | 0.8199 | 0.8327 | +0.0127 | 3/3 |
| 2-8 | top1_agreement | 0.4008 | 0.4153 | +0.0145 | 3/3 |
| 2-8 | spearman | 0.6739 | 0.7050 | +0.0311 | 3/3 |
| 4-8 | nestedness | 0.8388 | 0.8326 | -0.0062 | 3/3 |
| 4-8 | top1_agreement | 0.5253 | 0.5203 | -0.0050 | 3/3 |
| 4-8 | spearman | 0.7889 | 0.7837 | -0.0052 | 3/3 |
| 7-8 | nestedness | 0.8346 | 0.8042 | -0.0303 | 3/3 |
| 7-8 | top1_agreement | 0.6902 | 0.6408 | -0.0494 | 3/3 |
| 7-8 | spearman | 0.8890 | 0.8606 | -0.0284 | 3/3 |

## Key Mismatch Deltas

| direction | same_mean | fixed_mean | delta | seeds_same_sign |
|---|---:|---:|---:|---:|
| 1->4 | 0.0129 | 0.0937 | +0.0808 | 3/3 |
| 4->1 | 0.9146 | 0.9146 | -0.0000 | 2/3 |
| 1->8 | 0.1260 | 0.3126 | +0.1866 | 3/3 |
| 8->1 | 1.6318 | 1.4762 | -0.1557 | 3/3 |
| 2->8 | 0.0869 | 0.2057 | +0.1188 | 3/3 |
| 8->2 | 0.4358 | 0.3382 | -0.0977 | 3/3 |
| 4->8 | 0.0282 | 0.0282 | -0.0000 | 2/3 |
| 8->4 | 0.0802 | 0.0543 | -0.0259 | 3/3 |

## Specialization Deltas

| train_k | normalized_mi_delta | purity_delta | task_js_delta |
|---:|---:|---:|---:|
| 1 | +0.0014 | +0.0039 | +0.0021 |
| 2 | +0.0013 | +0.0018 | +0.0018 |
| 3 | -0.0001 | -0.0005 | -0.0004 |
| 4 | +0.0000 | +0.0000 | +0.0000 |
| 5 | +0.0015 | +0.0027 | +0.0021 |
| 6 | +0.0012 | +0.0019 | +0.0019 |
| 7 | +0.0013 | +0.0037 | +0.0019 |
| 8 | +0.0016 | +0.0041 | +0.0024 |

## Guardrails

- This is a paired three-seed comparison, not an inferential significance test.
- Fixed-step equalizes optimizer updates and token exposure; active expert compute still grows with k.
- Positive routing deltas mean the fixed-step models route more similarly than the same-compute models.
