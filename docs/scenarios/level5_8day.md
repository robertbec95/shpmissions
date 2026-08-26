# Scenario: level5_8day

Level 5 re-tested on the 8-day window, completing the granularity x window-length comparison.

**Config:** category_level=5 | dates=20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511 | min_variety=2 | k=6 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 6,948,669 baskets, 914 categories seen, run in 78.2s
. k-selection at this population: gap_k=2, silhouette_k=6, elbow_k=7, silhouette@k=6=0.037

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | unlabeled (conf 0.0) | 43.95% | 28.71 | 3.19 |  | no |
| 1 | unlabeled (conf 0.0) | 30.81% | 44.57 | 5.52 | BANANAS (32%), PACKED BREAD (23%), SHOPPING BAGS (79%) | no |
| 2 | unlabeled (conf 0.0) | 11.63% | 50.13 | 3.93 | PET (34%), FULL FLAVOR (69%) | no |
| 3 | breakfast (conf 1.0) | 6.07% | 29.26 | 4.19 | SWEET BAKE OFF FINAL PRODUCTION (37%), SALTED BAKE OFF FINAL PRODUCTION (75%) | no |
| 4 | unlabeled (conf 0.0) | 2.89% | 48.24 | 4.07 | HEATED TOBACCO WITH TOBACCO (59%), #N/A (42%) | no |
| 5 | unlabeled (conf 0.0) | 4.66% | 63.96 | 7.33 | CUCUMBERS (55%), TOMATOES (76%), ONION (15%), SHOPPING BAGS (73%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (unlabeled) |   C1 (unlabeled) |   C2 (unlabeled) |   C3 (breakfast) |   C4 (unlabeled) |   C5 (unlabeled) |
|:----------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| Timis     |             45.7 |             24.1 |             14.5 |              6.8 |              4.2 |              4.7 |
| Cluj      |             47.6 |             24.1 |             13.4 |              6.3 |              4.1 |              4.5 |
| Bucharest |             44.4 |             25.6 |             12.5 |              7   |              6.1 |              4.3 |
| Dolj      |             39.8 |             36.8 |              9.1 |              7.2 |              1.9 |              5.2 |
| Bihor     |             45.7 |             27.4 |             12.1 |              6.8 |              3.1 |              4.8 |
| Constanta |             44.5 |             28.8 |             11.2 |              6.6 |              4   |              5   |
| Suceava   |             45.9 |             32.4 |             10   |              5.5 |              1.8 |              4.4 |
| Bacau     |             46.3 |             31.4 |             10.9 |              4.7 |              2.5 |              4.2 |
| Iasi      |             45.1 |             32.6 |              9.3 |              5.1 |              3.6 |              4.3 |
| Arges     |             39.9 |             36.3 |             11.2 |              6.3 |              2.1 |              4.3 |

## Cluster mix by day-part

| day_part      |   C0 (unlabeled) |   C1 (unlabeled) |   C2 (unlabeled) |   C3 (breakfast) |   C4 (unlabeled) |   C5 (unlabeled) |
|:--------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| night         |             52.1 |             15.7 |             18.9 |              4.6 |              6.5 |              2.2 |
| early_morning |             42   |             26.3 |             11.9 |             14.6 |              2.3 |              3   |
| morning       |             38.2 |             37.8 |              8.9 |              8   |              1.9 |              5.2 |
| afternoon     |             43.5 |             32   |             11.3 |              5.5 |              2.6 |              5.1 |
| evening       |             46.5 |             29.4 |             12.5 |              3.5 |              3.3 |              4.8 |
| late_evening  |             51.4 |             22.5 |             15.1 |              2.5 |              4.9 |              3.6 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level5_8day`