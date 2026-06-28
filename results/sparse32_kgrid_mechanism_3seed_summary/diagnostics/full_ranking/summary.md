# Full Expert Ranking Analysis

Every row uses all 32 expert ranks. Layer `-1` aggregates all eight MoE layers.

| pair | RBO p=0.90 | Kendall tau | Footrule similarity | mean abs displacement | top1 | top4 | top8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.4540 | 0.2917 | 0.5101 | 7.8377 | 0.1539 | 0.3091 | 0.4430 |
| 1-4 | 0.4663 | 0.3126 | 0.5237 | 7.6205 | 0.1619 | 0.3263 | 0.4580 |
| 1-8 | 0.4735 | 0.3270 | 0.5331 | 7.4708 | 0.1637 | 0.3359 | 0.4676 |
| 2-4 | 0.6233 | 0.5253 | 0.6616 | 5.4140 | 0.4207 | 0.5189 | 0.6161 |
| 2-8 | 0.6152 | 0.5136 | 0.6543 | 5.5304 | 0.4007 | 0.5116 | 0.6091 |
| 4-8 | 0.7016 | 0.6312 | 0.7333 | 4.2673 | 0.5253 | 0.6228 | 0.6960 |
| 7-8 | 0.7996 | 0.7538 | 0.8169 | 2.9294 | 0.6903 | 0.7491 | 0.7927 |

## Selected Experts Only

| pair | small containment | full containment | ordered containment | common exact order | pairwise precedence | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.2524 | 0.2524 | nan | nan | nan | nan |
| 1-4 | 0.4199 | 0.4199 | nan | nan | nan | nan |
| 1-8 | 0.6357 | 0.6357 | nan | nan | nan | nan |
| 2-4 | 0.6476 | 0.4120 | 0.2752 | 0.6679 | 0.6679 | 0.4120 |
| 2-8 | 0.8199 | 0.6752 | 0.4478 | 0.6633 | 0.6633 | 0.6752 |
| 4-8 | 0.8388 | 0.5170 | 0.0987 | 0.3121 | 0.7129 | 0.7081 |
| 7-8 | 0.8346 | 0.2927 | 0.0184 | 0.0934 | 0.7975 | 0.6978 |

## Cardinality-Calibrated Selected Overlap

| pair | hypergeometric calibrated | frequency adjusted | ceiling utilization | empirical null | empirical ceiling | headroom |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.2026 | 0.1005 | 0.9958 | 0.1689 | 0.2528 | 0.0839 |
| 1-4 | 0.3370 | 0.1726 | 0.9989 | 0.2989 | 0.4200 | 0.1211 |
| 1-8 | 0.5143 | 0.2844 | 0.9996 | 0.4910 | 0.6358 | 0.1447 |
| 2-4 | 0.7417 | 0.4445 | 1.0000 | 0.3657 | 0.6476 | 0.2819 |
| 2-8 | 0.8120 | 0.6083 | 1.0000 | 0.5404 | 0.8199 | 0.2796 |
| 4-8 | 0.9314 | 0.6475 | 1.0000 | 0.5427 | 0.8388 | 0.2961 |
| 7-8 | 0.9876 | 0.6504 | 1.0000 | 0.5268 | 0.8346 | 0.3078 |

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
