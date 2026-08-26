# Faithful reproduction (2025 data): Convenience Store — Brasov

Store `65587f43fae8eed7bbc91cd5`, Convenience Store, Brasov. Full calendar-2025 purchase history. Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, 60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store alone. Run in 4.8s.

## Stage 1: variety segmentation

| segment             |     n |   value |   n_share |   value_share |
|:--------------------|------:|--------:|----------:|--------------:|
| concrete            | 82139 |  293676 |     69.26 |         44.38 |
| relatively_concrete | 29488 |  242318 |     24.86 |         36.62 |
| abstract            |  6966 |  125765 |      5.87 |         19    |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 5), 'abstract': (6, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

29,488 relatively-concrete baskets -> 17,692 train / 11,796 test. gap_k=3 (decision), elbow_k=5, silhouette_k=3 (comparison).

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.0184 | 0.0043 | False                |       0.0605 |
|   3 | 0.0252 | 0.0039 | True                 |       0.0646 |
|   4 | 0.0248 | 0.0086 | True                 |       0.0545 |
|   5 | 0.0237 | 0.0068 | True                 |       0.0579 |
|   6 | 0.0189 | 0.0072 | True                 |       0.0583 |
|   7 | 0.0225 | 0.0084 | True                 |       0.0571 |
|   8 | 0.0246 | 0.0087 | True                 |       0.0555 |
|   9 | 0.0149 | 0.0072 | True                 |       0.0549 |
|  10 | 0.0162 | 0.0064 | True                 |       0.053  |
|  11 | 0.0158 | 0.0068 | False                |       0.0512 |

## Mission clusters (k=3)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | snack (conf 0.6) | 39.59% | 9.13 | CARBONATED DRINKS (x1.8), BEER (x1.8), CHOCOLATE (x1.7), SALTY SNACKS (x1.6), TOBACCO (x1.3) |
| 1 | unlabeled (conf 0.0) | 27.22% | 8.58 | YOGURT  / SANA / KEFIR, DESSERT (x3.1), WATER (x2.9), TOBACCO (x1.7) |
| 2 | meal (conf 0.25) | 33.19% | 6.84 | VEGETABLES (x2.5), FRUITS (x2.4), HOUSEHOLD PRODUCTS (x2.1), SWEET SNACKS (x1.7) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 39.45% | 39.78% | 0.33 pp |
| 1 | 27.08% | 27.44% | 0.36 pp |
| 2 | 33.47% | 32.77% | 0.69 pp |

## Cross-validation by day-part

| day_part      |   C0 (snack) |   C1 (unlabeled) |   C2 (meal) |
|:--------------|-------------:|-----------------:|------------:|
| night         |         52.5 |             23.9 |        23.6 |
| early_morning |          0   |              0   |       100   |
| morning       |         27.9 |             38.5 |        33.5 |
| afternoon     |         31.8 |             28.5 |        39.7 |
| evening       |         43.9 |             24.8 |        31.3 |
| late_evening  |         47.7 |             24.9 |        27.4 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`