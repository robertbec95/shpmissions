# Faithful reproduction (2025 data): Supermarket — Bucharest

Store `65587f43fae8eed7bbc917e4`, Supermarket, Bucharest. Full calendar-2025 purchase history. Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, 60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store alone. Run in 7.2s.

## Stage 1: variety segmentation

| segment             |      n |            value |   n_share |   value_share |
|:--------------------|-------:|-----------------:|----------:|--------------:|
| concrete            | 216388 | 843221           |     51.3  |         24.84 |
| relatively_concrete | 163748 |      1.54863e+06 |     38.82 |         45.61 |
| abstract            |  41634 |      1.00327e+06 |      9.87 |         29.55 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 6), 'abstract': (7, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

163,748 relatively-concrete baskets -> 98,248 train / 65,500 test. gap_k=3 (decision), elbow_k=6, silhouette_k=4 (comparison).

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.0125 | 0.0025 | False                |       0.0475 |
|   3 | 0.018  | 0.0044 | True                 |       0.0464 |
|   4 | 0.0191 | 0.0045 | True                 |       0.0479 |
|   5 | 0.0211 | 0.0051 | True                 |       0.0474 |
|   6 | 0.0224 | 0.0044 | True                 |       0.0466 |
|   7 | 0.0199 | 0.0054 | True                 |       0.04   |
|   8 | 0.0209 | 0.0053 | True                 |       0.0423 |
|   9 | 0.0194 | 0.0061 | True                 |       0.0369 |
|  10 | 0.0189 | 0.0064 | True                 |       0.0352 |
|  11 | 0.0212 | 0.0058 | False                |       0.0319 |

## Mission clusters (k=3)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | snack (conf 0.25) | 31.26% | 9.79 | WATER (x1.8), OTHERS FIIGO (x1.8), TOBACCO (x1.8), SALTY SNACKS (x1.7) |
| 1 | breakfast (conf 0.25) | 39.32% | 9.75 | VEGETABLES (x2.2), HOUSEHOLD PRODUCTS (x2.0), SUPPLIED BAKERY (x1.9), FRUITS (x1.6) |
| 2 | snack (conf 0.67) | 29.42% | 8.72 | CARBONATED DRINKS (x2.6), SWEET SNACKS (x2.6), YOGURT  / SANA / KEFIR, DESSERT (x2.4) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 31.62% | 30.72% | 0.9 pp |
| 1 | 39.21% | 39.5% | 0.29 pp |
| 2 | 29.17% | 29.78% | 0.6 pp |

## Cross-validation by day-part

| day_part      |   C0 (snack) |   C1 (breakfast) |   C2 (snack) |
|:--------------|-------------:|-----------------:|-------------:|
| night         |         37.9 |             24   |         38.1 |
| early_morning |         37.8 |             24.9 |         37.3 |
| morning       |         33.2 |             35.7 |         31.1 |
| afternoon     |         29.6 |             45.2 |         25.1 |
| evening       |         28.5 |             43.3 |         28.2 |
| late_evening  |         31.2 |             37.5 |         31.3 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`