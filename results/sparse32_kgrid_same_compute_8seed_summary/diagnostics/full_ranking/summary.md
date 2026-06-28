# Full Expert Ranking Analysis

Every row uses all 32 expert ranks. Layer `-1` aggregates all eight MoE layers.

| pair | RBO p=0.90 | Kendall tau | Footrule similarity | mean abs displacement | top1 | top4 | top8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.4553 | 0.2956 | 0.5124 | 7.8012 | 0.1511 | 0.3112 | 0.4451 |
| 1-4 | 0.4670 | 0.3157 | 0.5254 | 7.5938 | 0.1590 | 0.3272 | 0.4596 |
| 1-8 | 0.4746 | 0.3294 | 0.5344 | 7.4502 | 0.1633 | 0.3371 | 0.4692 |
| 2-4 | 0.6268 | 0.5286 | 0.6641 | 5.3739 | 0.4256 | 0.5239 | 0.6197 |
| 2-8 | 0.6165 | 0.5159 | 0.6556 | 5.5101 | 0.4002 | 0.5137 | 0.6110 |
| 4-8 | 0.7053 | 0.6360 | 0.7366 | 4.2139 | 0.5333 | 0.6270 | 0.7000 |
| 7-8 | 0.8019 | 0.7574 | 0.8193 | 2.8916 | 0.6950 | 0.7504 | 0.7962 |

## Selected Experts Only

| pair | small containment | full containment | ordered containment | common exact order | pairwise precedence | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.2505 | 0.2505 | nan | nan | nan | nan |
| 1-4 | 0.4189 | 0.4189 | nan | nan | nan | nan |
| 1-8 | 0.6345 | 0.6345 | nan | nan | nan | nan |
| 2-4 | 0.6545 | 0.4209 | 0.2820 | 0.6699 | 0.6699 | 0.4209 |
| 2-8 | 0.8213 | 0.6774 | 0.4496 | 0.6637 | 0.6637 | 0.6774 |
| 4-8 | 0.8435 | 0.5272 | 0.1039 | 0.3138 | 0.7165 | 0.7159 |
| 7-8 | 0.8379 | 0.3009 | 0.0191 | 0.0934 | 0.7983 | 0.7033 |

## Cardinality-Calibrated Selected Overlap

| pair | hypergeometric calibrated | frequency adjusted | ceiling utilization | empirical null | empirical ceiling | headroom |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.2005 | 0.1014 | 0.9963 | 0.1659 | 0.2508 | 0.0849 |
| 1-4 | 0.3359 | 0.1741 | 0.9991 | 0.2965 | 0.4190 | 0.1226 |
| 1-8 | 0.5127 | 0.2868 | 0.9997 | 0.4876 | 0.6346 | 0.1470 |
| 2-4 | 0.7485 | 0.4528 | 1.0000 | 0.3685 | 0.6545 | 0.2859 |
| 2-8 | 0.8135 | 0.6108 | 1.0000 | 0.5407 | 0.8213 | 0.2805 |
| 4-8 | 0.9348 | 0.6562 | 1.0000 | 0.5449 | 0.8435 | 0.2986 |
| 7-8 | 0.9886 | 0.6571 | 1.0000 | 0.5274 | 0.8379 | 0.3105 |

## Guardrails

- RBO weights the top of the ranking more heavily but uses all 32 ranks.
- Kendall tau measures pairwise ordering agreement across all 496 expert pairs and is estimated on an evenly spaced record sample per layer.
- Normalized Spearman footrule is one minus mean absolute rank displacement divided by its permutation maximum.
- Top-m overlap compares equal cutoffs in the two separately trained models; it is different from unequal-k nestedness.
- Selected-set containment and Jaccard use the models' matched operational cutoffs, not all 32 experts.
- Hypergeometric calibration uses the complete random-intersection distribution for the pair's exact set sizes; random maps to zero and perfect nesting to one.
- Frequency adjustment matches sample blocks within task. A no-fixed-point random derangement maps to zero and theoretical perfect containment maps to one.
- Ceiling utilization separately reports how much of the empirically attainable optimal-matching headroom is realized; it is not used as the main k-comparison metric.
- Ceiling utilization is unstable when empirical alignment headroom is near zero, so headroom is always reported.
- Union Kendall compares only experts selected by at least one model; common-selected Kendall compares only experts selected by both.
- Common-selected Kendall is pair-weighted and must be read with common-pair coverage. It is undefined when the smaller cutoff is one.
- Common exact order requires every comparable common-expert pair to preserve precedence; pairwise precedence reports the fraction preserved.
- Ordered containment requires the entire smaller selected set to appear in the larger set in the same relative order.
- Transition matrices aggregate aligned expert IDs within each seed. Cross-seed expert labels are not treated as independently aligned models.
- Token records are repeated observations from the same trained models, not independent statistical samples.
