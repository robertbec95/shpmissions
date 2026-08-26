# Stability test: level3_baseline

20 independent Stage 2 clustering runs on the same 2,364,042-basket population (category_level=3, k=6, sample_size=6000), each with a different random seed - varying which 6,000 baskets get CLARA-sampled for the k-medoids fit, and the k-medoids++ initialization. Pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) computed between every pair of the resulting full-population labelings (190 pairs). Both are permutation-invariant, so cluster-ID relabeling between runs doesn't affect the score - only genuine agreement/disagreement about which baskets group together does.

## Agreement between runs

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.2338 | 0.0708 | 0.0905 | 0.4471 |
| NMI | 0.2985 | 0.0656 | 0.1420 | 0.4541 |

Rule of thumb: ARI/NMI > 0.9 = highly stable, 0.7-0.9 = reasonably stable, 0.5-0.7 = moderate instability worth investigating, < 0.5 = the clustering is largely re-deriving noise from sample to sample, not a repeatable structure.

## Cluster size variability across runs (% share of baskets)

|           |   mean |   std |   min |   max |
|:----------|-------:|------:|------:|------:|
| cluster_0 |  26.47 |  3.47 | 17.98 | 34.15 |
| cluster_1 |  16.82 |  5.81 |  5.75 | 24.78 |
| cluster_2 |  16.25 |  4.77 |  8.06 | 26.83 |
| cluster_3 |  14.37 |  7.53 |  3.5  | 28.85 |
| cluster_4 |  11.78 |  4.95 |  2.88 | 21.2  |
| cluster_5 |  14.31 |  5.81 |  5.98 | 24.52 |

A cluster whose share swings widely (high std, or min/max far apart) across runs is one whose existence/size depends heavily on which particular sample was drawn - not something to treat as a fixed, known percentage of the business.

Reproduce with: `PYTHONPATH=src python3 scripts/run_stability_test.py level3_baseline --n-runs 20`