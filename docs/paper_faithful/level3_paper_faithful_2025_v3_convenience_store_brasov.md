# Faithful reproduction v3 (2025 data): Convenience Store — Brasov

Store `65587f43fae8eed7bbc91cd5`, Convenience Store, Brasov, full calendar-2025 purchase history (118,593 baskets). Baseline method: unweighted binary category presence, Jaccard distance, k-centroid (k-medoids), gap statistic for k at both stages, 60/40 train/test split, 10 k-medoids initializations kept-best-by-dispersion. Run in 18.1s.

## Stage 1: variety segmentation

Gap statistic on basket variety selected **gap_k=1**. Since gap_k<3, there aren't enough raw clusters to form 3 non-empty concrete/relatively-concrete/abstract bands - **fell back to the paper's fixed k=3 framing** (`variety_stage.fit_variety_segments`), rather than silently pretending the gap statistic chose 3.

| segment             |     n |   value |   n_share |   value_share |
|:--------------------|------:|--------:|----------:|--------------:|
| concrete            | 82139 |  293676 |     69.26 |         44.38 |
| relatively_concrete | 29488 |  242318 |     24.86 |         36.62 |
| abstract            |  6966 |  125765 |      5.87 |         19    |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 5), 'abstract': (6, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

29,488 relatively-concrete baskets -> 17,692 train / 11,796 test. gap_k=3 (decision), elbow_k=5, silhouette_k=3 (comparison only).

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

Multiple initializations: 10 independent k-medoids++ restarts on the same training CLARA sample, kept the lowest total within-cluster Jaccard dispersion. Dispersion across attempts ranged 5251062.25 to 6593013.61 (best: seed 1) - the spread is the direct evidence for why a single initialization isn't reliable: different restarts converge to different local optima.

## Mission clusters (k=3)

| Cluster | Name | Weak? | Share | Avg value | Category participation (support, lift) |
|---|---|---|---|---|---|
| 0 | Vegetables + Fruits + Household Products | no | 35.31% | 6.76 | VEGETABLES (36%, x2.2), FRUITS (40%, x2.0), HOUSEHOLD PRODUCTS (31%, x1.7), SUPPLIED BAKERY (31%, x1.7), MILK / CREAM (17%, x1.5), PACKED SAUSAGES (15%, x1.5) |
| 1 | Sweet Snacks + Chocolate + Salty Snacks | no | 35.35% | 7.58 | SWEET SNACKS (44%, x2.5), CHOCOLATE (37%, x2.5), SALTY SNACKS (50%, x2.4) |
| 2 | Tobacco + Beer + Carbonated Drinks | no | 29.34% | 10.74 | TOBACCO (58%, x2.5), BEER (46%, x2.4), CARBONATED DRINKS (40%, x2.1) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 35.38% | 35.21% | 0.17 pp |
| 1 | 35.29% | 35.45% | 0.17 pp |
| 2 | 29.34% | 29.34% | 0.01 pp |

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |
|:--------------|-----:|-----:|-----:|
| night         | 22.2 | 43.4 | 34.4 |
| early_morning | 33.3 | 66.7 |  0   |
| morning       | 40.7 | 32.8 | 26.5 |
| afternoon     | 44.7 | 30.6 | 24.7 |
| evening       | 32.7 | 35   | 32.3 |
| late_evening  | 24.9 | 42.7 | 32.4 |

- C0: Vegetables + Fruits + Household Products
- C1: Sweet Snacks + Chocolate + Salty Snacks
- C2: Tobacco + Beer + Carbonated Drinks

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`