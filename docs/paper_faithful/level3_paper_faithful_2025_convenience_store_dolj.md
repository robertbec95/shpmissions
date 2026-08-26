# Faithful reproduction (2025 data): Convenience Store — Dolj

Store `65587f43fae8eed7bbc91c35`, Convenience Store, Dolj. Full calendar-2025 purchase history. Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, 60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store alone. Run in 5.4s.

## Stage 1: variety segmentation

| segment             |      n |   value |   n_share |   value_share |
|:--------------------|-------:|--------:|----------:|--------------:|
| concrete            | 137860 |  459413 |     65.16 |         38.2  |
| relatively_concrete |  57091 |  441280 |     26.98 |         36.69 |
| abstract            |  16636 |  302048 |      7.86 |         25.11 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 5), 'abstract': (6, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

57,091 relatively-concrete baskets -> 34,254 train / 22,837 test. gap_k=3 (decision), elbow_k=6, silhouette_k=6 (comparison).

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.01   | 0.0022 | False                |       0.0383 |
|   3 | 0.02   | 0.0045 | True                 |       0.0503 |
|   4 | 0.0216 | 0.0057 | True                 |       0.0502 |
|   5 | 0.0262 | 0.007  | True                 |       0.0537 |
|   6 | 0.0288 | 0.0077 | True                 |       0.0552 |
|   7 | 0.023  | 0.0079 | True                 |       0.0437 |
|   8 | 0.0216 | 0.0071 | True                 |       0.0454 |
|   9 | 0.0205 | 0.0066 | True                 |       0.0428 |
|  10 | 0.0158 | 0.009  | True                 |       0.043  |
|  11 | 0.0112 | 0.0079 | False                |       0.0428 |

## Mission clusters (k=3)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | snack (conf 0.4) | 37.62% | 8.55 | BEER (x2.0), SALTY SNACKS (x1.8), CARBONATED DRINKS (x1.7), SUPPLIED BAKERY (x1.3), TOBACCO (x1.2) |
| 1 | breakfast (conf 0.33) | 38.01% | 7.23 | CHOCOLATE (x2.2), BAKE OFF (x2.2), WATER (x2.2) |
| 2 | breakfast (conf 0.25) | 24.37% | 7.24 | VEGETABLES (x3.1), FRUITS (x2.7), HOUSEHOLD PRODUCTS (x2.6), SUPPLIED BAKERY (x1.4) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 37.45% | 37.87% | 0.42 pp |
| 1 | 37.99% | 38.04% | 0.05 pp |
| 2 | 24.56% | 24.09% | 0.47 pp |

## Cross-validation by day-part

| day_part      |   C0 (snack) |   C1 (breakfast) |   C2 (breakfast) |
|:--------------|-------------:|-----------------:|-----------------:|
| night         |         46.8 |             34.8 |             18.4 |
| early_morning |         50   |             25   |             25   |
| morning       |         24.1 |             50.7 |             25.1 |
| afternoon     |         33.1 |             40.9 |             26   |
| evening       |         40.3 |             33.6 |             26   |
| late_evening  |         45.8 |             33.4 |             20.8 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`