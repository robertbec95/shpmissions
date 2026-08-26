# Faithful reproduction (2025 data): Convenience Store — Bucharest

Store `65587f43fae8eed7bbc91c43`, Convenience Store, Bucharest. Full calendar-2025 purchase history. Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, 60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store alone. Run in 6.4s.

## Stage 1: variety segmentation

| segment             |      n |            value |   n_share |   value_share |
|:--------------------|-------:|-----------------:|----------:|--------------:|
| concrete            | 235494 | 814006           |     57.8  |         30.57 |
| relatively_concrete | 128518 |      1.07762e+06 |     31.54 |         40.47 |
| abstract            |  43448 | 771313           |     10.66 |         28.96 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 5), 'abstract': (6, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

128,518 relatively-concrete baskets -> 77,110 train / 51,408 test. gap_k=2 (decision), elbow_k=6, silhouette_k=2 (comparison).

|   k |     gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|--------:|-------:|:---------------------|-------------:|
|   2 |  0.0102 | 0.0048 | True                 |       0.0494 |
|   3 |  0.0107 | 0.0048 | False                |       0.0419 |
|   4 |  0.0188 | 0.0065 | True                 |       0.0451 |
|   5 |  0.0188 | 0.007  | True                 |       0.0473 |
|   6 |  0.0212 | 0.007  | True                 |       0.0407 |
|   7 |  0.0139 | 0.0093 | True                 |       0.0387 |
|   8 |  0.0022 | 0.0099 | True                 |       0.0303 |
|   9 | -0.0006 | 0.0076 | True                 |       0.025  |
|  10 | -0.0055 | 0.0089 | True                 |       0.0252 |
|  11 | -0.0042 | 0.0101 | False                |       0.0238 |

## Mission clusters (k=2)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | snack (conf 0.75) | 53.51% | 8.32 | SWEET SNACKS (x1.5), SALTY SNACKS (x1.4), CARBONATED DRINKS (x1.3), MILK / CREAM (x1.2) |
| 1 | breakfast (conf 0.25) | 46.49% | 8.46 | VEGETABLES (x2.0), BEER (x2.0), SUPPLIED BAKERY (x1.9), HOUSEHOLD PRODUCTS (x1.9) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 53.42% | 53.64% | 0.22 pp |
| 1 | 46.58% | 46.36% | 0.22 pp |

## Cross-validation by day-part

| day_part      |   C0 (snack) |   C1 (breakfast) |
|:--------------|-------------:|-----------------:|
| night         |         55.3 |             44.7 |
| early_morning |        100   |              0   |
| morning       |         61.1 |             38.9 |
| afternoon     |         51.1 |             48.9 |
| evening       |         52.7 |             47.3 |
| late_evening  |         54.6 |             45.4 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`