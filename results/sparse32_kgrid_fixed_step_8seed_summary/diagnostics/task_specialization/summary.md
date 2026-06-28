# Task Specialization Diagnostics

Values are averaged over seeds and layers; task-conditioned gate values are also averaged over tasks.

| train_k | normalized_mi | expert_purity | task_js | gate_entropy | logit_margin | selected_mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0137 | 0.2004 | 0.0206 | 0.9722 | 0.1803 | 0.0720 |
| 2 | 0.0384 | 0.2444 | 0.0554 | 0.9162 | 0.5261 | 0.2261 |
| 3 | 0.0343 | 0.2379 | 0.0498 | 0.9074 | 0.5322 | 0.3026 |
| 4 | 0.0275 | 0.2258 | 0.0404 | 0.9043 | 0.5246 | 0.3623 |
| 5 | 0.0233 | 0.2178 | 0.0345 | 0.9025 | 0.5169 | 0.4145 |
| 6 | 0.0199 | 0.2120 | 0.0296 | 0.9002 | 0.5141 | 0.4625 |
| 7 | 0.0165 | 0.2046 | 0.0246 | 0.8976 | 0.5183 | 0.5065 |
| 8 | 0.0147 | 0.2007 | 0.0221 | 0.8962 | 0.5153 | 0.5467 |

## Guardrails

- Expert purity has a balanced-task baseline of 1 / number_of_tasks.
- Mutual information and JS divergence describe routing specialization, not task quality.
- Expert identities are only compared within a seed; cross-seed expert labels are permutation-sensitive.
