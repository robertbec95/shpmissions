# Stability test: level3_8day

20 independent Stage 2 clustering runs on the same 6,694,099-basket population (category_level=3, k=6, sample_size=6000), each with a different random seed - varying which 6,000 baskets get CLARA-sampled for the k-medoids fit, and the k-medoids++ initialization. Pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) computed between every pair of the resulting full-population labelings (190 pairs). Both are permutation-invariant, so cluster-ID relabeling between runs doesn't affect the score - only genuine agreement/disagreement about which baskets group together does.

## Agreement between runs

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.2102 | 0.0753 | 0.0547 | 0.4312 |
| NMI | 0.2763 | 0.0683 | 0.1236 | 0.5050 |

Rule of thumb: ARI/NMI > 0.9 = highly stable, 0.7-0.9 = reasonably stable, 0.5-0.7 = moderate instability worth investigating, < 0.5 = the clustering is largely re-deriving noise from sample to sample, not a repeatable structure.

## Cluster size variability across runs (% share of baskets)

|           |   mean |   std |   min |   max |
|:----------|-------:|------:|------:|------:|
| cluster_0 |  30.08 |  6.43 | 17.02 | 42.64 |
| cluster_1 |  16.59 |  6.19 |  2.39 | 27.05 |
| cluster_2 |  16.32 |  6.59 |  0.82 | 29.46 |
| cluster_3 |  13.86 |  3.85 |  4.66 | 19.5  |
| cluster_4 |  12.27 |  6.94 |  2.68 | 29.12 |
| cluster_5 |  10.88 |  5.71 |  2.17 | 23.56 |

A cluster whose share swings widely (high std, or min/max far apart) across runs is one whose existence/size depends heavily on which particular sample was drawn - not something to treat as a fixed, known percentage of the business.

Reproduce with: `PYTHONPATH=src python3 scripts/run_stability_test.py level3_8day --n-runs 20`