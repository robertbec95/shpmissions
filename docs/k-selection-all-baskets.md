# Gap statistic / silhouette / elbow - Stage 2 on ALL baskets (no variety floor)

Same diagnostic as `docs/k-selection-variety-ge2.md`, run on the full 3,676,724-basket population used in `docs/table6-comparison-all-baskets.md` - every basket, single-category included. Sample size 1,500, B=10.

**Gap statistic selects k=2. Silhouette is maximized at k=11. Elbow method picks k=5.**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.5025 | 0.0317 | 0.0094 | True                 |       0.0864 |
|   3 |   6.4803 | 0.0296 | 0.0113 | True                 |       0.0728 |
|   4 |   6.4566 | 0.0293 | 0.0115 | True                 |       0.0815 |
|   5 |   6.4304 | 0.0375 | 0.0106 | True                 |       0.0976 |
|   6 |   6.4234 | 0.0297 | 0.0107 | True                 |       0.0997 |
|   7 |   6.4197 | 0.0182 | 0.0094 | False                |       0.0934 |
|   8 |   6.379  | 0.0442 | 0.007  | True                 |       0.1252 |
|   9 |   6.3771 | 0.0347 | 0.0128 | True                 |       0.1197 |
|  10 |   6.3569 | 0.0459 | 0.0127 | False                |       0.1333 |
|  11 |   6.3206 | 0.0731 | 0.0142 | False                |       0.1502 |

Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_all_baskets.py`