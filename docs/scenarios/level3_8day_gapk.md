# Scenario: level3_8day_gapk

level3_8day re-clustered at k=2 - its own gap-statistic-selected k post-HOUSEHOLD PRODUCTS exclusion (silhouette@k=6=0.058, 2 degenerate clusters at k=6). Same purpose as level3_baseline_gapk: test the data-suggested k directly rather than assume k=6 is right just because the paper used it.

**Config:** category_level=3 | dates=20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511 | min_variety=2 | k=2 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 6,694,099 baskets, 99 categories seen, run in 67.0s
. k-selection at this population: gap_k=3, silhouette_k=6, elbow_k=6, silhouette@k=2=0.0358

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | meal (conf 0.33) | 70.96% | 40.33 | 3.81 | VEGETABLES (26%), FRUITS (25%), HOUSEHOLD PRODUCTS (43%) | no |
| 1 | snack (conf 0.5) | 29.04% | 35.63 | 3.76 | WATER (51%), SWEET SNACKS (42%), YOGURT  / SANA / KEFIR, DESSERT (27%), SALTY SNACKS (20%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (meal) |   C1 (snack) |
|:----------|------------:|-------------:|
| Timis     |        68.3 |         31.7 |
| Cluj      |        68.9 |         31.1 |
| Bucharest |        68.8 |         31.2 |
| Dolj      |        73.4 |         26.6 |
| Bihor     |        70.3 |         29.7 |
| Constanta |        70.7 |         29.3 |
| Suceava   |        70.2 |         29.8 |
| Bacau     |        70.2 |         29.8 |
| Iasi      |        69.5 |         30.5 |
| Arges     |        72.2 |         27.8 |

## Cluster mix by day-part

| day_part      |   C0 (meal) |   C1 (snack) |
|:--------------|------------:|-------------:|
| night         |        61.4 |         38.6 |
| early_morning |        64.7 |         35.3 |
| morning       |        75.8 |         24.2 |
| afternoon     |        72.9 |         27.1 |
| evening       |        69.8 |         30.2 |
| late_evening  |        64.5 |         35.5 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level3_8day_gapk`