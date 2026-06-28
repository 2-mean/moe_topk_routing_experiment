# Fixed-step Task Loss Grid

Loss deltas are relative to each model's matched inference k and averaged over three seeds.

| task | matched_k1 | matched_k8 | 1->8 delta | 8->1 delta | asymmetry ratio |
|---|---:|---:|---:|---:|---:|
| general_ko | 0.4634 | 0.3933 | +0.2600 | +1.5170 | 5.83 |
| general_en | 0.4146 | 0.4001 | +0.2722 | +1.4856 | 5.46 |
| math_ko | 0.3748 | 0.3492 | +0.2911 | +1.3147 | 4.52 |
| math_en | 0.4985 | 0.4133 | +0.3141 | +1.5836 | 5.04 |
| code | 0.5154 | 0.3805 | +0.3954 | +1.4752 | 3.73 |
| reasoning | 0.3738 | 0.3375 | +0.3373 | +1.3865 | 4.11 |
| translation | 0.4725 | 0.3711 | +0.3182 | +1.5705 | 4.94 |

## Guardrails

- Tasks are synthetic corpus categories, not external benchmark domains.
- Per-task losses share the same trained models and are not independent observations.
- A task heatmap localizes mismatch cost but does not establish task-specific expert causality.
