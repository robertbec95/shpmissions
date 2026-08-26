# Scenario: level4_8day

Level 4 re-tested on the 8-day window, for the same reason as level3_8day - more data may change which granularity wins, not just whether k=6 is well-supported.

**Config:** category_level=4 | dates=20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511 | min_variety=2 | k=6 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 6,865,810 baskets, 361 categories seen, run in 76.8s
. k-selection at this population: gap_k=2, silhouette_k=8, elbow_k=7, silhouette@k=6=0.0467

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | unlabeled (conf 0.0) | 39.65% | 29.69 | 2.92 | CIGARETTES (18%) | no |
| 1 | unlabeled (conf 0.0) | 15.56% | 55.17 | 5.35 | WHITE BREAD (64%), PALE ALE BEER (37%), CIGARETTES (32%), HOME CONSUMABLES (58%) | no |
| 2 | unlabeled (conf 0.0) | 27.21% | 45.43 | 5.37 | BANANAS (40%), HOME CONSUMABLES (85%), STEM (18%) | no |
| 3 | unlabeled (conf 0.0) | 7.2% | 34.28 | 3.52 | CHIPS (40%), PALE ALE BEER (66%) | no |
| 4 | breakfast (conf 1.0) | 6.19% | 33.32 | 4.3 | BAKE OFF FINAL PRODUCTION (100%) | **YES** |
| 5 | coffee (conf 0.5) | 4.2% | 30.25 | 3.79 | BAR - CHOCOLATE (70%), WHITE YOGURT (33%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (unlabeled) |   C1 (unlabeled) |   C2 (unlabeled) |   C3 (unlabeled) |   C4 (breakfast) |   C5 (coffee) |
|:----------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|--------------:|
| Timis     |             44.9 |             13.7 |             22.3 |              7.9 |              7   |           4.2 |
| Cluj      |             46.1 |             12   |             22.8 |              8.1 |              6.4 |           4.6 |
| Bucharest |             43.3 |             12   |             26   |              7   |              7.1 |           4.6 |
| Dolj      |             33.3 |             20.2 |             29.8 |              5.5 |              7.4 |           3.9 |
| Bihor     |             44.7 |             12.6 |             24.3 |              7.2 |              7.1 |           4.1 |
| Constanta |             39.2 |             15.5 |             27.3 |              6.6 |              6.8 |           4.6 |
| Suceava   |             38.9 |             15.2 |             27.4 |              8.4 |              5.6 |           4.4 |
| Bacau     |             39.5 |             18.1 |             25.7 |              8.2 |              4.6 |           3.8 |
| Iasi      |             40.1 |             14.6 |             27.7 |              7.9 |              5   |           4.7 |
| Arges     |             34.5 |             18.5 |             30.1 |              6.3 |              6.3 |           4.2 |

## Cluster mix by day-part

| day_part      |   C0 (unlabeled) |   C1 (unlabeled) |   C2 (unlabeled) |   C3 (unlabeled) |   C4 (breakfast) |   C5 (coffee) |
|:--------------|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|--------------:|
| night         |             50.7 |             13.2 |             14.1 |             12.5 |              4.6 |           4.9 |
| early_morning |             40.9 |             13.4 |             23.7 |              3.5 |             14.3 |           4.1 |
| morning       |             33.9 |             16.8 |             33.4 |              4.2 |              8.2 |           3.5 |
| afternoon     |             39.7 |             15.2 |             28.8 |              6.3 |              5.8 |           4.1 |
| evening       |             40.8 |             16.2 |             25.4 |              9.6 |              3.6 |           4.5 |
| late_evening  |             45.6 |             14.9 |             19   |             13   |              2.5 |           5   |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level4_8day`