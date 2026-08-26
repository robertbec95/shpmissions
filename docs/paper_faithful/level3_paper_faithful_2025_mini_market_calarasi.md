# Faithful reproduction (2025 data): Mini Market — Calarasi

Store `65587f43fae8eed7bbc91c04`, Mini Market, Calarasi. Full calendar-2025 purchase history. Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, 60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store alone. Run in 5.4s.

## Stage 1: variety segmentation

| segment             |     n |   value |   n_share |   value_share |
|:--------------------|------:|--------:|----------:|--------------:|
| concrete            | 94902 |  472193 |     64.71 |         33.65 |
| relatively_concrete | 41057 |  579394 |     28    |         41.29 |
| abstract            | 10691 |  351602 |      7.29 |         25.06 |

Bounds: `{'concrete': (1, 3), 'relatively_concrete': (4, 8), 'abstract': (9, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

41,057 relatively-concrete baskets -> 24,634 train / 16,423 test. gap_k=3 (decision), elbow_k=6, silhouette_k=2 (comparison).

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.0108 | 0.0026 | False                |       0.0411 |
|   3 | 0.0197 | 0.0033 | True                 |       0.041  |
|   4 | 0.0141 | 0.0034 | True                 |       0.0361 |
|   5 | 0.0144 | 0.0041 | False                |       0.0329 |
|   6 | 0.0189 | 0.004  | True                 |       0.0316 |
|   7 | 0.0208 | 0.0035 | True                 |       0.03   |
|   8 | 0.02   | 0.0037 | True                 |       0.029  |
|   9 | 0.017  | 0.0038 | True                 |       0.0275 |
|  10 | 0.0153 | 0.0045 | True                 |       0.0271 |
|  11 | 0.0194 | 0.0044 | False                |       0.027  |

## Mission clusters (k=3)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | snack (conf 0.75) | 45.46% | 13.72 | SWEET SNACKS (x1.9), SALTY SNACKS (x1.9), BEER (x1.5), CHOCOLATE (x1.3) |
| 1 | light_meal (conf 0.5) | 19.24% | 15.83 | PACKED SAUSAGES (x2.5), YOGURT  / SANA / KEFIR, DESSERT (x2.3), CARBONATED DRINKS (x2.2), CHEESE PACKED (x1.4) |
| 2 | breakfast (conf 0.33) | 35.29% | 13.68 | VEGETABLES (x1.9), MILK / CREAM (x1.8), FRUITS (x1.8), WATER (x1.5), SUPPLIED BAKERY (x1.2) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 45.66% | 45.17% | 0.48 pp |
| 1 | 19.14% | 19.4% | 0.26 pp |
| 2 | 35.2% | 35.43% | 0.22 pp |

## Cross-validation by day-part

| day_part      |   C0 (snack) |   C1 (light_meal) |   C2 (breakfast) |
|:--------------|-------------:|------------------:|-----------------:|
| night         |         57   |              21.9 |             21.2 |
| early_morning |        100   |               0   |              0   |
| morning       |         42.8 |              17.9 |             39.3 |
| afternoon     |         41.7 |              18.9 |             39.5 |
| evening       |         46.9 |              19.5 |             33.6 |
| late_evening  |         54.7 |              20.7 |             24.6 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`