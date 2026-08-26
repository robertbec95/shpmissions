# Stability test: level3_value_distribution

20 independent Bray-Curtis CLARA runs per k, same population and methodology as `docs/stability/level3_baseline.md` (which tests the binary/Jaccard baseline) - only the distance metric and input matrix differ (spend-distribution Bray-Curtis vs. category-presence Jaccard). Tests whether Bray-Curtis's much sharper per-run category lifts (see the scenario report) are also more reproducible run-to-run, or just a better fit to whichever CLARA sample happened to get drawn.

| k | ARI mean | ARI std | NMI mean | NMI std |
|---|---|---|---|---|
| 2 | 0.1957 | 0.1893 | 0.1795 | 0.1607 |
| 10 | 0.3500 | 0.0556 | 0.3994 | 0.0428 |

For reference: binary Jaccard CLARA (k=6, `docs/stability/level3_baseline.md`) scores ARI mean=0.2338, NMI mean=0.2985 on this same population. Graph community detection (a different, non-CLARA method, `docs/graph_missions/level3_baseline.md`) scores ARI mean=0.5565 - still the most stable method tested in this project.

Rule of thumb (same as other stability reports): ARI/NMI > 0.9 = highly stable, 0.7-0.9 = reasonably stable, 0.5-0.7 = moderate instability worth investigating, < 0.5 = largely re-deriving noise from sample to sample.

## Cluster size variability across runs (% share of baskets)

### k=2

|           |   mean |   std |   min |   max |
|:----------|-------:|------:|------:|------:|
| cluster_0 |  55.87 | 10.84 | 39.27 | 80.8  |
| cluster_1 |  44.13 | 10.84 | 19.2  | 60.73 |

### k=10

|           |   mean |   std |   min |   max |
|:----------|-------:|------:|------:|------:|
| cluster_0 |  14.75 |  4.44 |  8.37 | 23.07 |
| cluster_1 |  10.84 |  4.47 |  2.39 | 21.36 |
| cluster_2 |   8.87 |  4.29 |  1.51 | 16.14 |
| cluster_3 |  11.11 |  3.34 |  5.14 | 16.09 |
| cluster_4 |  10.18 |  4.97 |  1.6  | 21.24 |
| cluster_5 |   9.61 |  2.88 |  3.78 | 15.63 |
| cluster_6 |   8.08 |  3.66 |  0.71 | 13.55 |
| cluster_7 |   8.31 |  4.32 |  0.58 | 16.97 |
| cluster_8 |   7.18 |  4.46 |  1.51 | 17.13 |
| cluster_9 |  11.07 |  3.91 |  2.66 | 17.35 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_distribution_scenario.py`