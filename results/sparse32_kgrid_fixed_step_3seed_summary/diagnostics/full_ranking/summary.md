# Full Expert Ranking Analysis

Every row uses all 32 expert ranks. Layer `-1` aggregates all eight MoE layers.

| pair | RBO p=0.90 | Kendall tau | Footrule similarity | mean abs displacement | top1 | top4 | top8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.5271 | 0.4132 | 0.5879 | 6.5935 | 0.2323 | 0.4005 | 0.5272 |
| 1-4 | 0.5276 | 0.4112 | 0.5864 | 6.6178 | 0.2317 | 0.4032 | 0.5277 |
| 1-8 | 0.5326 | 0.4181 | 0.5909 | 6.5464 | 0.2345 | 0.4117 | 0.5340 |
| 2-4 | 0.6507 | 0.5693 | 0.6916 | 4.9338 | 0.4521 | 0.5519 | 0.6477 |
| 2-8 | 0.6327 | 0.5442 | 0.6752 | 5.1968 | 0.4153 | 0.5330 | 0.6300 |
| 4-8 | 0.6975 | 0.6247 | 0.7289 | 4.3380 | 0.5203 | 0.6181 | 0.6925 |
| 7-8 | 0.7699 | 0.7139 | 0.7894 | 3.3702 | 0.6409 | 0.7121 | 0.7639 |

## Selected Experts Only

| pair | small containment | full containment | ordered containment | common exact order | pairwise precedence | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.3607 | 0.3607 | nan | nan | nan | nan |
| 1-4 | 0.5308 | 0.5308 | nan | nan | nan | nan |
| 1-8 | 0.7308 | 0.7308 | nan | nan | nan | nan |
| 2-4 | 0.6846 | 0.4654 | 0.3176 | 0.6823 | 0.6823 | 0.4654 |
| 2-8 | 0.8327 | 0.6974 | 0.4649 | 0.6665 | 0.6665 | 0.6974 |
| 4-8 | 0.8326 | 0.5016 | 0.0948 | 0.3166 | 0.7095 | 0.6975 |
| 7-8 | 0.8042 | 0.2199 | 0.0092 | 0.0865 | 0.7709 | 0.6471 |

## Cardinality-Calibrated Selected Overlap

| pair | hypergeometric calibrated | frequency adjusted | ceiling utilization | empirical null | empirical ceiling | headroom |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.3181 | 0.1897 | 1.0000 | 0.2110 | 0.3607 | 0.1497 |
| 1-4 | 0.4637 | 0.2882 | 1.0000 | 0.3408 | 0.5308 | 0.1899 |
| 1-8 | 0.6411 | 0.4313 | 1.0000 | 0.5267 | 0.7308 | 0.2041 |
| 2-4 | 0.7751 | 0.4894 | 1.0000 | 0.3823 | 0.6846 | 0.3023 |
| 2-8 | 0.8257 | 0.6301 | 1.0000 | 0.5478 | 0.8327 | 0.2848 |
| 4-8 | 0.9281 | 0.6364 | 1.0000 | 0.5395 | 0.8326 | 0.2931 |
| 7-8 | 0.9819 | 0.5973 | 1.0000 | 0.5139 | 0.8042 | 0.2903 |

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
