# Faithful reproduction v3 (2025 data): Supermarket — Neamt

Store `65587f43fae8eed7bbc917ff`, Supermarket, Neamt, full calendar-2025 purchase history (587,487 baskets). Baseline method: unweighted binary category presence, Jaccard distance, k-centroid (k-medoids), gap statistic for k at both stages, 60/40 train/test split, 10 k-medoids initializations kept-best-by-dispersion. Run in 22.1s.

## Stage 1: variety segmentation

Gap statistic on basket variety selected **gap_k=1**. Since gap_k<3, there aren't enough raw clusters to form 3 non-empty concrete/relatively-concrete/abstract bands - **fell back to the paper's fixed k=3 framing** (`variety_stage.fit_variety_segments`), rather than silently pretending the gap statistic chose 3.

| segment             |      n |            value |   n_share |   value_share |
|:--------------------|-------:|-----------------:|----------:|--------------:|
| concrete            | 328527 |      1.1183e+06  |     55.92 |         29.18 |
| relatively_concrete | 215038 |      1.81736e+06 |     36.6  |         47.43 |
| abstract            |  43922 | 896187           |      7.48 |         23.39 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 6), 'abstract': (7, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

215,038 relatively-concrete baskets -> 129,022 train / 86,016 test. gap_k=2 (decision), elbow_k=6, silhouette_k=2 (comparison only).

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.0176 | 0.0043 | True                 |       0.0575 |
|   3 | 0.0042 | 0.0049 | True                 |       0.0541 |
|   4 | 0.0061 | 0.0065 | True                 |       0.0497 |
|   5 | 0.0099 | 0.0061 | True                 |       0.0528 |
|   6 | 0.013  | 0.0054 | True                 |       0.047  |
|   7 | 0.0096 | 0.0039 | True                 |       0.0434 |
|   8 | 0.0066 | 0.0056 | True                 |       0.043  |
|   9 | 0.0101 | 0.0056 | True                 |       0.0425 |
|  10 | 0.0149 | 0.0055 | True                 |       0.0406 |
|  11 | 0.0117 | 0.007  | False                |       0.0452 |

Multiple initializations: 10 independent k-medoids++ restarts on the same training CLARA sample, kept the lowest total within-cluster Jaccard dispersion. Dispersion across attempts ranged 8134213.46 to 9483346.64 (best: seed 1) - the spread is the direct evidence for why a single initialization isn't reliable: different restarts converge to different local optima.

## Mission clusters (k=2)

| Cluster | Name | Weak? | Share | Avg value | Category participation (support, lift) |
|---|---|---|---|---|---|
| 0 | Carbonated Drinks + Salty Snacks + Sweet Snacks | no | 55.49% | 8.6 | CARBONATED DRINKS (25%, x1.6), SALTY SNACKS (32%, x1.6), SWEET SNACKS (30%, x1.6), WATER (25%, x1.5) |
| 1 | Vegetables + Fruits + Household Products | no | 44.51% | 8.27 | VEGETABLES (41%, x2.0), FRUITS (45%, x1.8), HOUSEHOLD PRODUCTS (46%, x1.7), SUPPLIED BAKERY (47%, x1.7) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 55.5% | 55.49% | 0.01 pp |
| 1 | 44.5% | 44.51% | 0.01 pp |

## Cross-validation by day-part

| day_part      |   C0 |   C1 |
|:--------------|-----:|-----:|
| night         | 78.1 | 21.9 |
| early_morning | 71.5 | 28.5 |
| morning       | 50.9 | 49.1 |
| afternoon     | 45.1 | 54.9 |
| evening       | 53.5 | 46.5 |
| late_evening  | 65.2 | 34.8 |

- C0: Carbonated Drinks + Salty Snacks + Sweet Snacks
- C1: Vegetables + Fruits + Household Products

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`