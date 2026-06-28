# Task Specialization Diagnostics

Values are averaged over seeds and layers; task-conditioned gate values are also averaged over tasks.

| train_k | normalized_mi | expert_purity | task_js | gate_entropy | logit_margin | selected_mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0136 | 0.2004 | 0.0206 | 0.9721 | 0.1806 | 0.0721 |
| 2 | 0.0379 | 0.2439 | 0.0549 | 0.9167 | 0.5232 | 0.2253 |
| 3 | 0.0346 | 0.2382 | 0.0503 | 0.9078 | 0.5305 | 0.3019 |
| 4 | 0.0284 | 0.2269 | 0.0419 | 0.9050 | 0.5156 | 0.3613 |
| 5 | 0.0249 | 0.2214 | 0.0369 | 0.9034 | 0.5098 | 0.4138 |
| 6 | 0.0210 | 0.2138 | 0.0313 | 0.9016 | 0.5028 | 0.4616 |
| 7 | 0.0174 | 0.2073 | 0.0260 | 0.8985 | 0.5129 | 0.5057 |
| 8 | 0.0158 | 0.2033 | 0.0237 | 0.8961 | 0.5125 | 0.5469 |

## Guardrails

- Expert purity has a balanced-task baseline of 1 / number_of_tasks.
- Mutual information and JS divergence describe routing specialization, not task quality.
- Expert identities are only compared within a seed; cross-seed expert labels are permutation-sensitive.
