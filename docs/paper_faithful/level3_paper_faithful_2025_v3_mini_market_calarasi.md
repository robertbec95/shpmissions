# Faithful reproduction v3 (2025 data): Mini Market — Calarasi

Store `65587f43fae8eed7bbc91c04`, Mini Market, Calarasi, full calendar-2025 purchase history (146,650 baskets). Baseline method: unweighted binary category presence, Jaccard distance, k-centroid (k-medoids), gap statistic for k at both stages, 60/40 train/test split, 10 k-medoids initializations kept-best-by-dispersion. Run in 19.3s.

## Stage 1: variety segmentation

Gap statistic on basket variety selected **gap_k=1**. Since gap_k<3, there aren't enough raw clusters to form 3 non-empty concrete/relatively-concrete/abstract bands - **fell back to the paper's fixed k=3 framing** (`variety_stage.fit_variety_segments`), rather than silently pretending the gap statistic chose 3.

| segment             |     n |   value |   n_share |   value_share |
|:--------------------|------:|--------:|----------:|--------------:|
| concrete            | 94902 |  472193 |     64.71 |         33.65 |
| relatively_concrete | 41057 |  579394 |     28    |         41.29 |
| abstract            | 10691 |  351602 |      7.29 |         25.06 |

Bounds: `{'concrete': (1, 3), 'relatively_concrete': (4, 8), 'abstract': (9, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

41,057 relatively-concrete baskets -> 24,634 train / 16,423 test. gap_k=3 (decision), elbow_k=6, silhouette_k=2 (comparison only).

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

Multiple initializations: 10 independent k-medoids++ restarts on the same training CLARA sample, kept the lowest total within-cluster Jaccard dispersion. Dispersion across attempts ranged 5275184.33 to 6041039.70 (best: seed 5) - the spread is the direct evidence for why a single initialization isn't reliable: different restarts converge to different local optima.

## Mission clusters (k=3)

| Cluster | Name | Weak? | Share | Avg value | Category participation (support, lift) |
|---|---|---|---|---|---|
| 0 | Vegetables + Fruits + Water | no | 34.15% | 13.46 | VEGETABLES (44%, x1.6), FRUITS (43%, x1.5), WATER (36%, x1.5), HOUSEHOLD PRODUCTS (55%, x1.4), RICE / BEANS / PASTA (16%, x1.4), CHICKEN MEAT (17%, x1.3) |
| 1 | Beer + Sweet Snacks + Carbonated Drinks | no | 27.81% | 13.82 | BEER (43%, x2.1), SWEET SNACKS (45%, x2.0), CARBONATED DRINKS (42%, x1.7), WATER (35%, x1.5), SUPPLIED BAKERY (43%, x1.5) |
| 2 | Yogurt  / Sana / Kefir, Dessert + Tobacco + Salty Snacks | no | 38.04% | 14.91 | YOGURT  / SANA / KEFIR, DESSERT (33%, x2.0), TOBACCO (23%, x1.9), SALTY SNACKS (44%, x1.7), SUPPLIED BAKERY (38%, x1.3), CARBONATED DRINKS (30%, x1.2) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 33.84% | 34.62% | 0.77 pp |
| 1 | 27.89% | 27.68% | 0.21 pp |
| 2 | 38.26% | 37.7% | 0.56 pp |

## Cross-validation by day-part

| day_part      |   C0 |   C1 |    C2 |
|:--------------|-----:|-----:|------:|
| night         | 20.6 | 34.2 |  45.2 |
| early_morning |  0   |  0   | 100   |
| morning       | 38.3 | 27.1 |  34.6 |
| afternoon     | 37.2 | 25.8 |  36.9 |
| evening       | 32.5 | 28.6 |  38.9 |
| late_evening  | 26.7 | 31.9 |  41.4 |

- C0: Vegetables + Fruits + Water
- C1: Beer + Sweet Snacks + Carbonated Drinks
- C2: Yogurt  / Sana / Kefir, Dessert + Tobacco + Salty Snacks

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`