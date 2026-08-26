# Faithful reproduction (2025 data): Supermarket — Neamt

Store `65587f43fae8eed7bbc917ff`, Supermarket, Neamt. Full calendar-2025 purchase history. Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, 60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store alone. Run in 7.4s.

## Stage 1: variety segmentation

| segment             |      n |            value |   n_share |   value_share |
|:--------------------|-------:|-----------------:|----------:|--------------:|
| concrete            | 328527 |      1.1183e+06  |     55.92 |         29.18 |
| relatively_concrete | 215038 |      1.81736e+06 |     36.6  |         47.43 |
| abstract            |  43922 | 896187           |      7.48 |         23.39 |

Bounds: `{'concrete': (1, 2), 'relatively_concrete': (3, 6), 'abstract': (7, None)}`

## Stage 2: k-selection (gap statistic, on the 60% training set)

215,038 relatively-concrete baskets -> 129,022 train / 86,016 test. gap_k=2 (decision), elbow_k=6, silhouette_k=2 (comparison).

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

## Mission clusters (k=2)

| Cluster | Label | Share | Avg value | Top categories (lift) |
|---|---|---|---|---|
| 0 | breakfast (conf 0.33) | 56.42% | 8.49 | VEGETABLES (x1.5), SUPPLIED BAKERY (x1.4), BEER (x1.3) |
| 1 | snack (conf 0.67) | 43.58% | 8.4 | SWEET SNACKS (x2.2), SALTY SNACKS (x2.2), FRUITS (x1.2) |

## Train vs. test validation

| Cluster | Train share | Test share | Abs. diff |
|---|---|---|---|
| 0 | 56.27% | 56.64% | 0.37 pp |
| 1 | 43.73% | 43.36% | 0.37 pp |

## Cross-validation by day-part

| day_part      |   C0 (breakfast) |   C1 (snack) |
|:--------------|-----------------:|-------------:|
| night         |             44.2 |         55.8 |
| early_morning |             45.9 |         54.1 |
| morning       |             59   |         41   |
| afternoon     |             62.8 |         37.2 |
| evening       |             57.2 |         42.8 |
| late_evening  |             50.1 |         49.9 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`