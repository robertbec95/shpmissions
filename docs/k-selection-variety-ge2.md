# Gap statistic / silhouette / elbow - Stage 2 on ALL variety>=2 baskets

Same diagnostic as `docs/k-selection.md` and `docs/k-selection-by-format.md`, run on the population used in `docs/table6-comparison-variety-ge2.md` - all 2,364,042 variety>=2 baskets, Stage 1's relatively-concrete filter not applied. Sample size 1,500, B=10.

**Gap statistic selects k=4. Silhouette is maximized at k=8. Elbow method picks k=7.**

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.5115 | 0.0021 | 0.0026 | False                |       0.0413 |
|   3 |   6.4802 | 0.0156 | 0.0031 | False                |       0.0538 |
|   4 |   6.4598 | 0.0247 | 0.0068 | True                 |       0.0585 |
|   5 |   6.4446 | 0.0265 | 0.0069 | True                 |       0.063  |
|   6 |   6.4316 | 0.0295 | 0.0088 | False                |       0.054  |
|   7 |   6.4092 | 0.0434 | 0.0111 | True                 |       0.0709 |
|   8 |   6.3969 | 0.0468 | 0.0126 | True                 |       0.072  |
|   9 |   6.3876 | 0.0475 | 0.0123 | True                 |       0.0677 |
|  10 |   6.388  | 0.0432 | 0.0151 | True                 |       0.0564 |
|  11 |   6.3787 | 0.0446 | 0.014  | False                |       0.0566 |

Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_variety_ge2.py`