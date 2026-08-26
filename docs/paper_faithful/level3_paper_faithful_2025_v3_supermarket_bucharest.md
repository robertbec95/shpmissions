# Faithful reproduction v3 (2025 data): Supermarket — Bucharest

Store `65587f43fae8eed7bbc917e4`, Supermarket, Bucharest, full calendar-2025 purchase history (421,770 baskets). Baseline method: unweighted binary category presence, Jaccard distance, k-centroid (k-medoids), gap statistic for k at both stages, 60/40 train/test split, 10 k-medoids initializations kept-best-by-dispersion. Run in 21.2s.

## Stage 1: variety segmentation

Gap statistic on basket variety selected **gap_k=1**. Since gap_k<3, there aren't enough raw clusters to form 3 non-empty concrete/relatively-concrete/abstract bands - **fell back to the paper's fixed k=3 framing** (`variety_stage.fit_variety_segments`), rather than silently pretending the gap statistic chose 3.

| segment             |      n |            value |   n_share |   value_share |
|:--------------------|-------:|-----------------:|----------:|--------------:|
| concrete            | 216388 | 843221           |     51.3  |         24.84 |
| relatively_concrete | 163748 |      1.54863e+06 |     38.82 |         45.61 |
| abstract            |  41634 |      1.00327e+06 |      9.87 |         29.55 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 6), 'abstract': (7, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

163,748 relatively-concrete baskets -> 98,248 train / 65,500 test. gap_k=3 (decision), elbow_k=6, silhouette_k=4 (comparison only).

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

Multiple initializations: 10 independent k-medoids++ restarts on the same training CLARA sample, kept the lowest total within-cluster Jaccard dispersion. Dispersion across attempts ranged 5310742.75 to 6450326.98 (best: seed 9) - the spread is the direct evidence for why a single initialization isn't reliable: different restarts converge to different local optima.

## Mission clusters (k=3)

| Cluster | Name | Weak? | Share | Avg value | Category participation (support, lift) |
|---|---|---|---|---|---|
| 0 | Supplied Bakery + Salty Snacks | no | 30.42% | 9.22 | SUPPLIED BAKERY (46%, x2.0), SALTY SNACKS (30%, x1.9) |
| 1 | Fruits + Vegetables + Household Products | no | 33.78% | 9.67 | FRUITS (63%, x2.7), VEGETABLES (43%, x2.0), HOUSEHOLD PRODUCTS (40%, x1.9) |
| 2 | Tobacco + Water + Bake Off | no | 35.8% | 9.46 | TOBACCO (40%, x2.1), WATER (43%, x1.9), BAKE OFF (31%, x1.8), CARBONATED DRINKS (34%, x1.8) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 30.37% | 30.5% | 0.12 pp |
| 1 | 33.58% | 34.08% | 0.5 pp |
| 2 | 36.05% | 35.43% | 0.62 pp |

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |
|:--------------|-----:|-----:|-----:|
| night         | 33.8 | 20.9 | 45.3 |
| early_morning | 31.6 | 23.7 | 44.7 |
| morning       | 25.9 | 33.3 | 40.8 |
| afternoon     | 27.8 | 37.5 | 34.7 |
| evening       | 31.4 | 37.4 | 31.3 |
| late_evening  | 35.1 | 32.6 | 32.3 |

- C0: Supplied Bakery + Salty Snacks
- C1: Fruits + Vegetables + Household Products
- C2: Tobacco + Water + Bake Off

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`