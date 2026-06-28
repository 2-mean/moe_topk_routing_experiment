# Fixed-step Task Loss Grid

Loss deltas are relative to each model's matched inference k and averaged over three seeds.

| task | matched_k1 | matched_k8 | 1->8 delta | 8->1 delta | asymmetry ratio |
|---|---:|---:|---:|---:|---:|
| general_ko | 0.3173 | 0.5061 | +0.1361 | +1.5890 | 11.68 |
| general_en | 0.2947 | 0.4755 | +0.0894 | +1.7251 | 19.30 |
| math_ko | 0.2932 | 0.4426 | +0.0856 | +1.5315 | 17.89 |
| math_en | 0.3302 | 0.4867 | +0.0986 | +1.5766 | 15.99 |
| code | 0.3427 | 0.4593 | +0.1824 | +1.4391 | 7.89 |
| reasoning | 0.3072 | 0.4692 | +0.1400 | +1.5837 | 11.32 |
| translation | 0.3172 | 0.4881 | +0.1313 | +1.7675 | 13.47 |

## Guardrails

- Tasks are synthetic corpus categories, not external benchmark domains.
- Per-task losses share the same trained models and are not independent observations.
- A task heatmap localizes mismatch cost but does not establish task-specific expert causality.
