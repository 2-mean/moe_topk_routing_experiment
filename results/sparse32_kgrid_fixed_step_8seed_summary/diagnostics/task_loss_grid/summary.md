# Fixed-step Task Loss Grid

Loss deltas are relative to each model's matched inference k and averaged over three seeds.

| task | matched_k1 | matched_k8 | 1->8 delta | 8->1 delta | asymmetry ratio |
|---|---:|---:|---:|---:|---:|
| general_ko | 0.4757 | 0.4096 | +0.2700 | +1.4648 | 5.42 |
| general_en | 0.4577 | 0.4077 | +0.2367 | +1.5604 | 6.59 |
| math_ko | 0.3726 | 0.3356 | +0.3033 | +1.4525 | 4.79 |
| math_en | 0.4464 | 0.3788 | +0.3061 | +1.5349 | 5.01 |
| code | 0.5046 | 0.4079 | +0.3451 | +1.5419 | 4.47 |
| reasoning | 0.4137 | 0.3644 | +0.3157 | +1.4056 | 4.45 |
| translation | 0.4595 | 0.3831 | +0.3202 | +1.5913 | 4.97 |

## Guardrails

- Tasks are synthetic corpus categories, not external benchmark domains.
- Per-task losses share the same trained models and are not independent observations.
- A task heatmap localizes mismatch cost but does not establish task-specific expert causality.
