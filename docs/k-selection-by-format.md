# Gap statistic, re-run on variety>=2 with real store format

Two changes from `docs/k-selection.md`: (1) input is now `outputs/clustering_dataset.parquet` - negative-quantity/return baskets dropped and, critically, **single-category (variety=1) baskets excluded from the variety-clustering input itself** (previously ~36% of the population and likely the reason the original run picked k=1); (2) real store format is now available from `data/locations-export (3).csv` (`store_format`, joined on `Store ID` / `location_external_id` - matches all 1,740 stores seen in the transaction data) in place of the basket-volume-tertile proxy, so results are also stratified by format instead of only pooled.

Stage 1 sample size/B reduced from the original run (50k/B=10 to 30k/B=8) to keep 4 strata tractable in one pass; still a stable sample for 1-D k-means. Stage 2 keeps its original 1,500/B=10.

Elbow method added alongside gap/silhouette: the point of maximum distance from the chord connecting the first and last points on the raw within-cluster-dispersion-vs-k curve (`cluster_selection.find_elbow_k`) - a third, independent read (no reference datasets, unlike gap; no cluster-quality metric, unlike silhouette) of the same underlying dispersion values gap statistic already computes.

## Stage 1 - variety clusters

| Stratum | n baskets | gap-selected k | silhouette-preferred k | elbow k |
|---|---:|---:|---:|---:|
| ALL (pooled, variety>=2) | 2,365,670 | 1 | 7 | 3 |
| Supermarket | 1,081,420 | 1 | 7 | 3 |
| Mini Market | 625,703 | 1 | 7 | 3 |
| Convenience Store | 650,449 | 1 | 7 | 3 |

**Gap statistic still picks k=1 in every stratum — pooled and all three formats.** This is the important negative result: removing single-category baskets and isolating a single store format were the two leading suspects for why the original run (`docs/k-selection.md`) found no variety structure, and *neither* changes the answer. **Elbow method now agrees with itself across every stratum too — k=3 everywhere** - and, notably, that's exactly the k the pipeline already uses by fixed convention (`variety_stage.fit_variety_segments`). That's a nice corroboration of the existing three-band choice, but it's still a much weaker signal than it sounds: elbow only asks "where does the dispersion curve bend the most," which any smoothly decaying curve has *some* answer to, whether or not real cluster structure exists - it doesn't test against a null the way gap statistic does. Silhouette's k=7 remains the least trustworthy of the three here, for the ordered-1-D-skewed-data reason given in `docs/k-selection.md`.

## Stage 2 - mission clusters (Stage 1 fixed at k=3 per stratum)

| Stratum | relatively-concrete n | gap-selected k | silhouette-preferred k | elbow k |
|---|---:|---:|---:|---:|
| ALL (pooled, variety>=2) | 525,910 | 3 | 3 | 8 |
| Supermarket | 369,748 | 2 | 2 | 7 |
| Mini Market | 172,679 | 3 | 7 | 7 |
| Convenience Store | 175,883 | 2 | 2 | 6 |

**The three statistics now genuinely disagree, and that disagreement is itself informative.** Gap stays low (2-3, barely moved from the original run's 2). Silhouette mostly agrees with gap (2-3) except Mini Market, where it jumps to 7 - a red flag of instability, not evidence of richer structure, given gap says 3 for the same stratum. But **elbow method lands at 6-8 in every stratum** - much closer to the paper's k=6-7 than either of the other two statistics suggests. That's a real tension worth sitting with rather than picking a winner: elbow is detecting *some* bend around 6-8 clusters worth of dispersion reduction, but gap statistic - the only one of the three that tests against a structureless null - doesn't think that bend beats what a random reference would produce at the same k. The honest reading: there's a visible knee in the raw dispersion curve around k=6-8 (consistent with the paper's choice), but it's not statistically distinguishable from noise at this sample size/window - which argues for more data (longer window, so the gap-statistic's reference comparison gets more power) rather than for picking whichever of the three numbers is most convenient.

## Raw values

### ALL (pooled, variety>=2)

**Stage 1 (variety)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   | silhouette   |
|----:|---------:|-------:|-------:|:---------------------|:-------------|
|   1 |  12.0295 | 2.5305 | 0.0041 | True                 | n/a          |
|   2 |  10.9571 | 2.2142 | 0.0067 | True                 | 0.7058       |
|   3 |  10.2905 | 2.0711 | 0.0042 | True                 | 0.6798       |
|   4 |   9.7489 | 2.037  | 0.0076 | True                 | 0.6911       |
|   5 |   9.3637 | 1.9745 | 0.0045 | False                | 0.7473       |
|   6 |   8.9275 | 2.0467 | 0.0053 | False                | 0.8343       |
|   7 |   8.5272 | 2.1381 | 0.0066 | False                | 0.8411       |

**Stage 2 (missions)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.4579 | 0.004  | 0.0037 | False                |       0.0384 |
|   3 |   6.4376 | 0.0119 | 0.0037 | True                 |       0.042  |
|   4 |   6.4246 | 0.0165 | 0.005  | True                 |       0.0364 |
|   5 |   6.4158 | 0.0168 | 0.0045 | True                 |       0.0346 |
|   6 |   6.4082 | 0.0169 | 0.0042 | True                 |       0.0335 |
|   7 |   6.398  | 0.0206 | 0.0045 | True                 |       0.0319 |
|   8 |   6.3887 | 0.0256 | 0.0051 | True                 |       0.0341 |
|   9 |   6.3855 | 0.0244 | 0.0061 | True                 |       0.0328 |
|  10 |   6.3834 | 0.0212 | 0.0057 | True                 |       0.0272 |
|  11 |   6.3789 | 0.0217 | 0.0071 | False                |       0.026  |

### Supermarket

**Stage 1 (variety)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   | silhouette   |
|----:|---------:|-------:|-------:|:---------------------|:-------------|
|   1 |  12.0372 | 2.4526 | 0.0041 | True                 | n/a          |
|   2 |  11.036  | 2.0651 | 0.0067 | True                 | 0.6822       |
|   3 |  10.3426 | 1.9489 | 0.0042 | True                 | 0.6806       |
|   4 |   9.8474 | 1.8683 | 0.0076 | True                 | 0.6851       |
|   5 |   9.423  | 1.845  | 0.0045 | True                 | 0.7432       |
|   6 |   9.1497 | 1.7543 | 0.0053 | False                | 0.8238       |
|   7 |   8.6323 | 1.9629 | 0.0066 | False                | 0.8882       |

**Stage 2 (missions)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.4584 | 0.0198 | 0.003  | True                 |       0.0696 |
|   3 |   6.4442 | 0.0187 | 0.0037 | True                 |       0.0544 |
|   4 |   6.4322 | 0.0192 | 0.0027 | True                 |       0.0407 |
|   5 |   6.4296 | 0.0111 | 0.0029 | True                 |       0.0276 |
|   6 |   6.4183 | 0.0153 | 0.0056 | False                |       0.0262 |
|   7 |   6.4057 | 0.0207 | 0.0053 | True                 |       0.0218 |
|   8 |   6.4033 | 0.0172 | 0.006  | True                 |       0.0251 |
|   9 |   6.3913 | 0.0229 | 0.0074 | True                 |       0.0305 |
|  10 |   6.388  | 0.02   | 0.006  | True                 |       0.03   |
|  11 |   6.3809 | 0.0221 | 0.0066 | False                |       0.0302 |

### Mini Market

**Stage 1 (variety)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   | silhouette   |
|----:|---------:|-------:|-------:|:---------------------|:-------------|
|   1 |  12.3025 | 2.3908 | 0.0041 | True                 | n/a          |
|   2 |  11.2332 | 2.0715 | 0.0067 | True                 | 0.6894       |
|   3 |  10.5321 | 1.9629 | 0.0042 | True                 | 0.6712       |
|   4 |  10.0313 | 1.8879 | 0.0076 | True                 | 0.6771       |
|   5 |   9.5868 | 1.8848 | 0.0045 | True                 | 0.6899       |
|   6 |   9.2768 | 1.8308 | 0.0053 | False                | 0.8099       |
|   7 |   8.9215 | 1.8772 | 0.0066 | False                | 0.8666       |

**Stage 2 (missions)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.4412 | 0.0092 | 0.0033 | False                |       0.0307 |
|   3 |   6.4238 | 0.0153 | 0.0033 | True                 |       0.0315 |
|   4 |   6.4133 | 0.0146 | 0.0048 | True                 |       0.0292 |
|   5 |   6.4091 | 0.0117 | 0.0043 | True                 |       0.0215 |
|   6 |   6.3975 | 0.0166 | 0.0053 | True                 |       0.029  |
|   7 |   6.3878 | 0.0199 | 0.005  | True                 |       0.0316 |
|   8 |   6.3871 | 0.0137 | 0.0038 | True                 |       0.0283 |
|   9 |   6.3839 | 0.0124 | 0.0057 | True                 |       0.0312 |
|  10 |   6.3837 | 0.0093 | 0.0049 | True                 |       0.0279 |
|  11 |   6.3807 | 0.0079 | 0.0062 | False                |       0.0274 |

### Convenience Store

**Stage 1 (variety)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   | silhouette   |
|----:|---------:|-------:|-------:|:---------------------|:-------------|
|   1 |  11.6192 | 2.5623 | 0.0041 | True                 | n/a          |
|   2 |  10.4971 | 2.2957 | 0.0067 | True                 | 0.7133       |
|   3 |   9.747  | 2.2361 | 0.0042 | True                 | 0.7086       |
|   4 |   9.2661 | 2.1413 | 0.0076 | False                | 0.7103       |
|   5 |   8.7069 | 2.2528 | 0.0045 | False                | 0.8638       |
|   6 |   8.2091 | 2.3866 | 0.0053 | False                | 0.9168       |
|   7 |   7.731  | 2.5558 | 0.0066 | False                | 0.9493       |

**Stage 2 (missions)**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.4614 | 0.0235 | 0.003  | True                 |       0.0693 |
|   3 |   6.4465 | 0.0217 | 0.0033 | True                 |       0.0618 |
|   4 |   6.4341 | 0.0218 | 0.004  | True                 |       0.0491 |
|   5 |   6.4238 | 0.0215 | 0.0036 | False                |       0.0379 |
|   6 |   6.4076 | 0.0287 | 0.005  | True                 |       0.0391 |
|   7 |   6.4013 | 0.0278 | 0.004  | True                 |       0.0362 |
|   8 |   6.3973 | 0.0243 | 0.0059 | False                |       0.0352 |
|   9 |   6.3881 | 0.0284 | 0.0039 | True                 |       0.0328 |
|  10 |   6.3787 | 0.0322 | 0.0047 | True                 |       0.0332 |
|  11 |   6.3743 | 0.0314 | 0.0049 | False                |       0.034  |

Reproduce with: `PYTHONPATH=src python3 scripts/select_k_by_format.py` (requires `outputs/clustering_dataset.parquet` - run `scripts/build_clustering_dataset.py` first if missing).