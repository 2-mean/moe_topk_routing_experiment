# Full Expert Ranking Analysis

Every row uses all 32 expert ranks. Layer `-1` aggregates all eight MoE layers.

| pair | RBO p=0.90 | Kendall tau | Footrule similarity | mean abs displacement | top1 | top4 | top8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.5266 | 0.4147 | 0.5887 | 6.5812 | 0.2284 | 0.3999 | 0.5277 |
| 1-4 | 0.5280 | 0.4136 | 0.5881 | 6.5901 | 0.2285 | 0.4038 | 0.5291 |
| 1-8 | 0.5322 | 0.4184 | 0.5913 | 6.5396 | 0.2313 | 0.4110 | 0.5348 |
| 2-4 | 0.6532 | 0.5710 | 0.6926 | 4.9184 | 0.4574 | 0.5568 | 0.6492 |
| 2-8 | 0.6339 | 0.5462 | 0.6759 | 5.1851 | 0.4159 | 0.5357 | 0.6317 |
| 4-8 | 0.6993 | 0.6269 | 0.7302 | 4.3170 | 0.5263 | 0.6192 | 0.6942 |
| 7-8 | 0.7698 | 0.7134 | 0.7888 | 3.3793 | 0.6444 | 0.7102 | 0.7639 |

## Selected Experts Only

| pair | small containment | full containment | ordered containment | common exact order | pairwise precedence | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.3572 | 0.3572 | nan | nan | nan | nan |
| 1-4 | 0.5324 | 0.5324 | nan | nan | nan | nan |
| 1-8 | 0.7316 | 0.7316 | nan | nan | nan | nan |
| 2-4 | 0.6902 | 0.4734 | 0.3239 | 0.6841 | 0.6841 | 0.4734 |
| 2-8 | 0.8371 | 0.7055 | 0.4701 | 0.6663 | 0.6663 | 0.7055 |
| 4-8 | 0.8353 | 0.5079 | 0.0979 | 0.3162 | 0.7118 | 0.7022 |
| 7-8 | 0.8042 | 0.2203 | 0.0090 | 0.0864 | 0.7704 | 0.6470 |

## Cardinality-Calibrated Selected Overlap

| pair | hypergeometric calibrated | frequency adjusted | ceiling utilization | empirical null | empirical ceiling | headroom |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 | 0.3144 | 0.1888 | 1.0000 | 0.2076 | 0.3572 | 0.1496 |
| 1-4 | 0.4656 | 0.2923 | 1.0000 | 0.3393 | 0.5324 | 0.1931 |
| 1-8 | 0.6422 | 0.4345 | 1.0000 | 0.5254 | 0.7316 | 0.2062 |
| 2-4 | 0.7802 | 0.4963 | 1.0000 | 0.3849 | 0.6902 | 0.3053 |
| 2-8 | 0.8303 | 0.6384 | 1.0000 | 0.5496 | 0.8371 | 0.2875 |
| 4-8 | 0.9292 | 0.6422 | 1.0000 | 0.5396 | 0.8353 | 0.2957 |
| 7-8 | 0.9821 | 0.5978 | 1.0000 | 0.5133 | 0.8042 | 0.2909 |

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
