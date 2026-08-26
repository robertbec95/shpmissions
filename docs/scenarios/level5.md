# Scenario: level5

Even finer granularity (~1,210 categories possible). Tests whether this is too sparse for basket-level Jaccard co-occurrence to find anything - each basket touching only a handful of very specific categories, diluting overlap between baskets that would agree at a coarser level.

**Config:** category_level=5 | dates=20260509,20260510,20260511 | min_variety=2 | k=6 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 2,455,448 baskets, 899 categories seen, run in 38.8s
. k-selection at this population: gap_k=2, silhouette_k=11, elbow_k=5, silhouette@k=6=0.0218

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | unlabeled (conf 0.0) | 45.93% | 29.77 | 3.14 |  | no |
| 1 | unlabeled (conf 0.0) | 32.25% | 49.87 | 6.0 | BANANAS (31%), SHOPPING BAGS (89%) | no |
| 2 | unlabeled (conf 0.0) | 3.3% | 39.9 | 4.49 | PANDISPAN CAKES (MINI CAKES) (44%), 3.5% FAT (55%) | no |
| 3 | unlabeled (conf 0.0) | 10.62% | 57.65 | 4.48 | BEER CAN (43%), FULL FLAVOR (65%) | no |
| 4 | unlabeled (conf 0.0) | 3.09% | 55.55 | 6.38 | 1.5% FAT (100%), SHOPPING BAGS (41%) | **YES** |
| 5 | unlabeled (conf 0.0) | 4.82% | 27.44 | 3.55 | CAN (62%), BAR WITH INGREDIENTS (42%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (unlabeled) |   C1 (unlabeled) |   C2 (unlabeled) |   C3 (unlabeled) |   C4 (unlabeled) |   C5 (unlabeled) |
|:----------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Timis     |             48.9 |             26.2 |              3.2 |             13   |              3   |              5.7 |
| Cluj      |             49.7 |             24.6 |              4.3 |             12   |              4.2 |              5.2 |
| Bucharest |             48.6 |             28.8 |              3.7 |             10.6 |              3.2 |              5.1 |
| Dolj      |             41.6 |             39.9 |              2.8 |              8   |              3.2 |              4.3 |
| Bihor     |             51.2 |             25.9 |              3.2 |             10.7 |              3.7 |              5.2 |
| Constanta |             46.6 |             32.2 |              3.3 |             10.2 |              2.8 |              4.9 |
| Suceava   |             46.5 |             33.6 |              3   |              9.3 |              2.2 |              5.3 |
| Bacau     |             47   |             32   |              3.3 |             10.6 |              2.7 |              4.4 |
| Iasi      |             46.8 |             33.3 |              3.3 |              8.6 |              3   |              5   |
| Arges     |             41.8 |             37.2 |              3.3 |             10.5 |              2.9 |              4.3 |

## Cluster mix by day-part

| day_part      |   C0 (unlabeled) |   C1 (unlabeled) |   C2 (unlabeled) |   C3 (unlabeled) |   C4 (unlabeled) |   C5 (unlabeled) |
|:--------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| night         |             55.2 |             16.3 |              3.2 |             15   |              1.5 |              8.7 |
| early_morning |             49.1 |             31   |              3.8 |              9.9 |              2.8 |              3.4 |
| morning       |             41.1 |             40.5 |              3.3 |              8.7 |              3.7 |              2.6 |
| afternoon     |             45.5 |             33.7 |              3   |             10.1 |              3   |              4.8 |
| evening       |             47.2 |             28.7 |              3.4 |             12   |              3   |              5.8 |
| late_evening  |             50.7 |             22.1 |              3.8 |             13   |              2.7 |              7.7 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level5`