# Router and Representation Mechanism Analysis

The logit decomposition uses the exact identity `total = router_shapley + representation_shapley` for the diagonal change from model A to B.

## Key Logit Decomposition

| pair | total_mse | router_share | representation_share | interaction_mse |
|---|---:|---:|---:|---:|
| 1-2 | 0.338902 | 0.4387 | 0.5613 | 0.058828 |
| 1-4 | 0.378084 | 0.4379 | 0.5621 | 0.064962 |
| 1-8 | 0.393786 | 0.4703 | 0.5297 | 0.069588 |
| 2-4 | 0.276621 | 0.2784 | 0.7216 | 0.012610 |
| 2-8 | 0.316772 | 0.3132 | 0.6868 | 0.018298 |
| 4-8 | 0.236634 | 0.2966 | 0.7034 | 0.008501 |
| 7-8 | 0.148983 | 0.2603 | 0.7397 | 0.002427 |

## Key All-layer Router Transplants

| host_k | donor_k | mean_delta_loss |
|---:|---:|---:|
| 1 | 4 | +0.3978 |
| 4 | 1 | +0.1480 |
| 1 | 8 | +0.4075 |
| 8 | 1 | +0.0947 |
| 2 | 8 | +0.1560 |
| 8 | 2 | +0.0000 |
| 4 | 8 | -0.0038 |
| 8 | 4 | -0.0103 |

## Guardrails

- Router and representation contributions are Shapley path averages in aligned hidden coordinates, not independent causal variables.
- Router shares can fall outside [0, 1] when router and representation changes oppose each other.
- Router transplant loss includes router-expert co-adaptation and downstream hidden-state changes.
- The mechanism probe is a deterministic subset of the synthetic validation corpus.
