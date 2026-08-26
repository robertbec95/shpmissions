# Faithful reproduction: Sarantopoulos et al. (2015)

Every step reproduced as specified: Stage 1 variety segmentation actually gates Stage 2's input (only relatively-concrete baskets), a 60/40 train/test split is used for Stage 2's k-selection and fit, gap statistic derives k on the training set, Jaccard distance + k-centroid (k-medoids/PAM), unweighted binary category presence. Dates: 20260509,20260510,20260511 (3-day window), category level 3 (99 categories - the paper's own taxonomy derived 75; this is a property of each retailer's product hierarchy, not a modeling choice). Multi-store chain, run chain-wide (not per-store, matching every other scenario in this project - the paper's case study was one pilot store). Run in 157.1s total.

## Stage 1: variety segmentation

| segment             |                n |       value |   n_share |   value_share |
|:--------------------|-----------------:|------------:|----------:|--------------:|
| concrete            |      2.16823e+06 | 3.54772e+07 |     58.97 |         31.04 |
| relatively_concrete |      1.22237e+06 | 4.91549e+07 |     33.25 |         43    |
| abstract            | 286121           | 2.96763e+07 |      7.78 |         25.96 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 6), 'abstract': (7, None)}`

Gap-statistic check on Stage 1's own k (fixed at k=3 regardless, matching the paper's three-band framing): gap_k=1.

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   1 | 2.5467 | 0.0033 | True                 |     nan      |
|   2 | 2.223  | 0.0051 | True                 |       0.7111 |
|   3 | 2.0318 | 0.0044 | True                 |       0.6813 |
|   4 | 2.027  | 0.0047 | True                 |       0.6905 |
|   5 | 1.9965 | 0.0043 | True                 |       0.7477 |
|   6 | 1.8931 | 0.0053 | False                |       0.7242 |
|   7 | 2.1456 | 0.0038 | False                |       0.8448 |

## Stage 2: k-selection (gap statistic, on the 60% training set)

1,222,374 relatively-concrete baskets -> 733,424 train (60%) / 488,950 test (40%). gap_k=3 (decision method), elbow_k=7, silhouette_k=3 (comparison only, per this project's established convention).

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.004  | 0.0036 | False                |       0.0442 |
|   3 | 0.0228 | 0.0048 | True                 |       0.0648 |
|   4 | 0.0103 | 0.0036 | True                 |       0.0578 |
|   5 | 0.0113 | 0.0067 | True                 |       0.0585 |
|   6 | 0.0154 | 0.0064 | False                |       0.0564 |
|   7 | 0.0267 | 0.0069 | True                 |       0.0535 |
|   8 | 0.0182 | 0.0073 | True                 |       0.0427 |
|   9 | 0.0197 | 0.0083 | True                 |       0.0442 |
|  10 | 0.0176 | 0.009  | True                 |       0.0416 |
|  11 | 0.0159 | 0.0094 | False                |       0.0426 |

## Stage 2: mission clusters (k=3, fit on train, assigned to full relatively-concrete population)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | breakfast (conf 0.25) | 59.99% | 37.22 | VEGETABLES (x1.6), SUPPLIED BAKERY (x1.5), FRUITS (x1.4), HOUSEHOLD PRODUCTS (x1.2) |
| 1 | snack (conf 0.5) | 27.64% | 48.39 | TOBACCO (x2.6), CARBONATED DRINKS (x2.6), BEER (x2.5), SALTY SNACKS (x1.5) |
| 2 | snack (conf 1.0) | 12.37% | 36.47 | ICE CREAM AND DESSERTS (x5.6), SALTY SNACKS (x3.6), SWEET SNACKS (x1.5) |

## Train vs. test validation

Cluster shares computed separately within the 60% training baskets and the 40% test baskets (both assigned to the same train-fit centroids) - close agreement means the clusters found on training data generalize to held-out baskets, not just noise in that particular 60%.

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 60.04% | 59.92% | 0.12 pp |
| 1 | 27.59% | 27.72% | 0.13 pp |
| 2 | 12.37% | 12.37% | 0.0 pp |

## Cross-validation by day-part

| day_part      |   C0 (breakfast) |   C1 (snack) |   C2 (snack) |
|:--------------|-----------------:|-------------:|-------------:|
| night         |             35   |         49.7 |         15.3 |
| early_morning |             70.8 |         21.9 |          7.4 |
| morning       |             74.2 |         18.6 |          7.2 |
| afternoon     |             61.3 |         26   |         12.7 |
| evening       |             51   |         32.8 |         16.2 |
| late_evening  |             41.2 |         42.3 |         16.5 |

## Cross-validation by county (top 10 by volume)

| county    |   C0 (breakfast) |   C1 (snack) |   C2 (snack) |
|:----------|-----------------:|-------------:|-------------:|
| Timis     |             53.5 |         34.2 |         12.3 |
| Cluj      |             55.7 |         31.2 |         13   |
| Bucharest |             58.8 |         29   |         12.3 |
| Dolj      |             66.1 |         22.3 |         11.6 |
| Constanta |             61.2 |         26.9 |         11.9 |
| Bihor     |             57.3 |         30.6 |         12.1 |
| Suceava   |             58.7 |         26.3 |         15   |
| Bacau     |             61.1 |         26.6 |         12.3 |
| Iasi      |             61.2 |         24.6 |         14.2 |
| Arges     |             63.4 |         25   |         11.6 |

## Bonus: does concrete/abstract fail to produce interpretable missions here too?

The paper: "the results for the baskets of 'concrete' and 'abstract' goals do not produce any interpretable outcomes." Same gap-statistic-selected Stage 2 fit applied to each segment on its own (no train/test split - this is a confirmatory check, not the primary result).

### Concrete (2,168,229 baskets)

gap_k=2, 1 of 2 clusters have no category clearing the lift/support threshold (uninterpretable).

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | unlabeled (conf 0.0) | 83.99% | _none clear threshold_ |
| 1 | unlabeled (conf 0.0) | 16.01% | TOBACCO (x6.2) |

### Abstract (286,121 baskets)

gap_k=3, 0 of 3 clusters have no category clearing the lift/support threshold (uninterpretable).

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | snack (conf 0.33) | 28.41% | TOBACCO (x1.9), CHOCOLATE (x1.8), BEER (x1.8), CARBONATED DRINKS (x1.8) |
| 1 | light_meal (conf 0.5) | 38.44% | VEGETABLES (x1.6), CHEESE PACKED (x1.3), PACKED SAUSAGES (x1.2), FRUITS (x1.2) |
| 2 | breakfast (conf 0.17) | 33.14% | RICE / BEANS / PASTA (x1.9), MILK / CREAM (x1.8), WATER (x1.7), SWEET SNACKS (x1.4) |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful.py`