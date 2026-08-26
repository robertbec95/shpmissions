# Faithful reproduction v3 (2025 data): Convenience Store — Dolj

Store `65587f43fae8eed7bbc91c35`, Convenience Store, Dolj, full calendar-2025 purchase history (211,587 baskets). Baseline method: unweighted binary category presence, Jaccard distance, k-centroid (k-medoids), gap statistic for k at both stages, 60/40 train/test split, 10 k-medoids initializations kept-best-by-dispersion. Run in 19.9s.

## Stage 1: variety segmentation

Gap statistic on basket variety selected **gap_k=1**. Since gap_k<3, there aren't enough raw clusters to form 3 non-empty concrete/relatively-concrete/abstract bands - **fell back to the paper's fixed k=3 framing** (`variety_stage.fit_variety_segments`), rather than silently pretending the gap statistic chose 3.

| segment             |      n |   value |   n_share |   value_share |
|:--------------------|-------:|--------:|----------:|--------------:|
| concrete            | 137860 |  459413 |     65.16 |         38.2  |
| relatively_concrete |  57091 |  441280 |     26.98 |         36.69 |
| abstract            |  16636 |  302048 |      7.86 |         25.11 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 5), 'abstract': (6, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

57,091 relatively-concrete baskets -> 34,254 train / 22,837 test. gap_k=3 (decision), elbow_k=6, silhouette_k=6 (comparison only).

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

Multiple initializations: 10 independent k-medoids++ restarts on the same training CLARA sample, kept the lowest total within-cluster Jaccard dispersion. Dispersion across attempts ranged 5448372.94 to 6316257.66 (best: seed 0) - the spread is the direct evidence for why a single initialization isn't reliable: different restarts converge to different local optima.

## Mission clusters (k=3)

| Cluster | Name | Weak? | Share | Avg value | Category participation (support, lift) |
|---|---|---|---|---|---|
| 0 | Beer + Salty Snacks + Carbonated Drinks | no | 37.62% | 8.55 | BEER (23%, x2.0), SALTY SNACKS (29%, x1.8), CARBONATED DRINKS (31%, x1.7), SUPPLIED BAKERY (28%, x1.3), TOBACCO (20%, x1.2) |
| 1 | Chocolate + Bake Off + Water | no | 38.01% | 7.23 | CHOCOLATE (28%, x2.2), BAKE OFF (40%, x2.2), WATER (51%, x2.2) |
| 2 | Vegetables + Fruits + Household Products | no | 24.37% | 7.24 | VEGETABLES (44%, x3.1), FRUITS (55%, x2.7), HOUSEHOLD PRODUCTS (58%, x2.6), SUPPLIED BAKERY (30%, x1.4) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 37.45% | 37.87% | 0.42 pp |
| 1 | 37.99% | 38.04% | 0.05 pp |
| 2 | 24.56% | 24.09% | 0.47 pp |

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |
|:--------------|-----:|-----:|-----:|
| night         | 46.8 | 34.8 | 18.4 |
| early_morning | 50   | 25   | 25   |
| morning       | 24.1 | 50.7 | 25.1 |
| afternoon     | 33.1 | 40.9 | 26   |
| evening       | 40.3 | 33.6 | 26   |
| late_evening  | 45.8 | 33.4 | 20.8 |

- C0: Beer + Salty Snacks + Carbonated Drinks
- C1: Chocolate + Bake Off + Water
- C2: Vegetables + Fruits + Household Products

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`