# Task Specialization Diagnostics

Values are averaged over seeds and layers; task-conditioned gate values are also averaged over tasks.

| train_k | normalized_mi | expert_purity | task_js | gate_entropy | logit_margin | selected_mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0123 | 0.1966 | 0.0185 | 0.9861 | 0.1294 | 0.0573 |
| 2 | 0.0366 | 0.2421 | 0.0531 | 0.9081 | 0.5499 | 0.2372 |
| 3 | 0.0347 | 0.2387 | 0.0506 | 0.9036 | 0.5357 | 0.3084 |
| 4 | 0.0284 | 0.2269 | 0.0419 | 0.9050 | 0.5156 | 0.3613 |
| 5 | 0.0235 | 0.2187 | 0.0348 | 0.9041 | 0.5182 | 0.4120 |
| 6 | 0.0198 | 0.2119 | 0.0293 | 0.9025 | 0.5139 | 0.4590 |
| 7 | 0.0161 | 0.2036 | 0.0240 | 0.9019 | 0.5096 | 0.5006 |
| 8 | 0.0142 | 0.1992 | 0.0213 | 0.8984 | 0.5232 | 0.5421 |

## Guardrails

- Expert purity has a balanced-task baseline of 1 / number_of_tasks.
- Mutual information and JS divergence describe routing specialization, not task quality.
- Expert identities are only compared within a seed; cross-seed expert labels are permutation-sensitive.
