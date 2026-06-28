# Task Specialization Diagnostics

Values are averaged over seeds and layers; task-conditioned gate values are also averaged over tasks.

| train_k | normalized_mi | expert_purity | task_js | gate_entropy | logit_margin | selected_mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0123 | 0.1967 | 0.0186 | 0.9860 | 0.1286 | 0.0573 |
| 2 | 0.0381 | 0.2453 | 0.0551 | 0.9082 | 0.5477 | 0.2371 |
| 3 | 0.0336 | 0.2374 | 0.0489 | 0.9034 | 0.5440 | 0.3087 |
| 4 | 0.0275 | 0.2258 | 0.0404 | 0.9043 | 0.5246 | 0.3623 |
| 5 | 0.0230 | 0.2172 | 0.0341 | 0.9038 | 0.5160 | 0.4123 |
| 6 | 0.0194 | 0.2108 | 0.0288 | 0.9024 | 0.5157 | 0.4588 |
| 7 | 0.0157 | 0.2026 | 0.0235 | 0.9019 | 0.5109 | 0.5004 |
| 8 | 0.0140 | 0.1985 | 0.0209 | 0.8989 | 0.5217 | 0.5412 |

## Guardrails

- Expert purity has a balanced-task baseline of 1 / number_of_tasks.
- Mutual information and JS divergence describe routing specialization, not task quality.
- Expert identities are only compared within a seed; cross-seed expert labels are permutation-sensitive.
