# Scenario: level4

Finer category granularity (422 categories possible, ~351 seen on a single day, vs. 99-103 at level 3). Tests whether splitting catch-all categories like BREAKFAST PRODUCTS into JAM/COMFITURE/SPREADABLE CREAM/HONEY sharpens cluster definitions rather than just adding noise.

**Config:** category_level=4 | dates=20260509,20260510,20260511 | min_variety=2 | k=6 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 2,425,521 baskets, 355 categories seen, run in 33.4s
. k-selection at this population: gap_k=2, silhouette_k=8, elbow_k=9, silhouette@k=6=0.0489

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | unlabeled (conf 0.0) | 28.95% | 27.68 | 3.03 |  | no |
| 1 | breakfast (conf 0.5) | 5.81% | 28.85 | 3.97 | BAKE OFF FINAL PRODUCTION (59%), BANANAS (52%) | no |
| 2 | unlabeled (conf 0.0) | 21.13% | 53.36 | 4.03 | CIGARETTES (61%), PALE ALE BEER (51%) | no |
| 3 | snack (conf 0.5) | 10.64% | 33.59 | 3.97 | IMPULSE (55%), ICE CREAM (52%) | no |
| 4 | breakfast (conf 0.2) | 7.41% | 56.92 | 6.44 | ROOT VEGETABLES (42%), FRESH MILK (65%), ONION (15%), STEM (23%) | no |
| 5 | unlabeled (conf 0.0) | 26.06% | 44.76 | 5.21 | WHITE BREAD (37%), HOME CONSUMABLES (85%), STEM (17%), BANANAS (17%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (unlabeled) |   C1 (breakfast) |   C2 (unlabeled) |   C3 (snack) |   C4 (breakfast) |   C5 (unlabeled) |
|:----------|-----------------:|-----------------:|-----------------:|-------------:|-----------------:|-----------------:|
| Timis     |             31.2 |              5.8 |             25.4 |         11.6 |              6.5 |             19.4 |
| Cluj      |             33.2 |              5.7 |             23.3 |         10.7 |              9.1 |             18   |
| Bucharest |             32.4 |              6.7 |             21   |         11.2 |              7.4 |             21.3 |
| Dolj      |             23.9 |              6.4 |             16.4 |         10.8 |              7.3 |             35.2 |
| Bihor     |             32.6 |              5   |             21.4 |         11.7 |              7.7 |             21.7 |
| Constanta |             30.3 |              6.1 |             19.4 |         10.5 |              7.1 |             26.6 |
| Suceava   |             29.8 |              6.3 |             19.2 |         11.5 |              5.7 |             27.5 |
| Bacau     |             28.4 |              6.5 |             23.1 |          9.6 |              6   |             26.4 |
| Iasi      |             30.2 |              7.2 |             19.2 |         11.2 |              6.5 |             25.7 |
| Arges     |             25.6 |              7   |             19.1 |         10.1 |              7.6 |             30.5 |

## Cluster mix by day-part

| day_part      |   C0 (unlabeled) |   C1 (breakfast) |   C2 (unlabeled) |   C3 (snack) |   C4 (breakfast) |   C5 (unlabeled) |
|:--------------|-----------------:|-----------------:|-----------------:|-------------:|-----------------:|-----------------:|
| night         |             37.9 |              3.6 |             32   |         11.7 |              2.3 |             12.4 |
| early_morning |             29.8 |             12.2 |             18.8 |          5.2 |              6.9 |             27.1 |
| morning       |             24.9 |              8   |             16.5 |          6.2 |             10   |             34.3 |
| afternoon     |             28   |              5.4 |             20   |         12.3 |              7.6 |             26.7 |
| evening       |             30.4 |              3.8 |             24.2 |         13.1 |              6.4 |             22.1 |
| late_evening  |             35.3 |              2.7 |             27.3 |         12.4 |              4.8 |             17.5 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level4`