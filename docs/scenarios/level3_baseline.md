# Scenario: level3_baseline

Anchor/reproduction of docs/table6-comparison-variety-ge2.md - the best-known result prior to this scenario framework - re-run through the configurable pipeline so it's directly comparable to level4/level5 on identical machinery.

**Config:** category_level=3 | dates=20260509,20260510,20260511 | min_variety=2 | k=6 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 2,364,042 baskets, 99 categories seen, run in 36.1s
. k-selection at this population: gap_k=4, silhouette_k=8, elbow_k=7, silhouette@k=6=0.054

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | meal (conf 0.33) | 30.84% | 41.33 | 3.44 | BEER (29%), TOBACCO (31%), VEGETABLES (30%) | no |
| 1 | snack (conf 1.0) | 21.49% | 35.65 | 3.52 | SALTY SNACKS (57%), CARBONATED DRINKS (61%), SWEET SNACKS (20%) | no |
| 2 | breakfast (conf 0.5) | 16.99% | 44.4 | 4.44 | SUPPLIED BAKERY (82%), HOUSEHOLD PRODUCTS (61%) | no |
| 3 | meal (conf 0.5) | 3.5% | 63.26 | 4.99 | CHICKEN MEAT (100%), HOUSEHOLD PRODUCTS (70%), VEGETABLES (35%), MILK / CREAM (16%) | **YES** |
| 4 | cleaning (conf 0.33) | 21.2% | 37.46 | 4.07 | FRUITS (62%), WATER (53%), HOUSEHOLD PRODUCTS (51%) | no |
| 5 | breakfast (conf 0.5) | 5.98% | 46.63 | 4.35 | MILK / CREAM (82%), SUPPLIED BAKERY (48%), CARBONATED DRINKS (37%), YOGURT  / SANA / KEFIR, DESSERT (15%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (meal) |   C1 (snack) |   C2 (breakfast) |   C3 (meal) |   C4 (cleaning) |   C5 (breakfast) |
|:----------|------------:|-------------:|-----------------:|------------:|----------------:|-----------------:|
| Timis     |        33.7 |         24.2 |             11.8 |         3   |            21.4 |              5.9 |
| Cluj      |        34.9 |         22.9 |             11.3 |         2.2 |            21   |              7.7 |
| Bucharest |        31.1 |         21.4 |             15.1 |         2.3 |            23.8 |              6.3 |
| Dolj      |        27.3 |         18.5 |             24   |         4.2 |            20.2 |              5.8 |
| Bihor     |        32.7 |         23.1 |             13.6 |         3.3 |            20.3 |              7   |
| Constanta |        29.8 |         21.6 |             18.6 |         3.5 |            21   |              5.4 |
| Suceava   |        29.4 |         23.5 |             18.4 |         2.9 |            20.7 |              5.1 |
| Bacau     |        31   |         20.4 |             18.1 |         3.1 |            21.8 |              5.6 |
| Iasi      |        30.1 |         20.8 |             17   |         2.3 |            24.2 |              5.7 |
| Arges     |        26.8 |         20.1 |             20.9 |         4.5 |            21.2 |              6.5 |

## Cluster mix by day-part

| day_part      |   C0 (meal) |   C1 (snack) |   C2 (breakfast) |   C3 (meal) |   C4 (cleaning) |   C5 (breakfast) |
|:--------------|------------:|-------------:|-----------------:|------------:|----------------:|-----------------:|
| night         |        35.3 |         35.2 |              9.2 |         0.6 |            16.4 |              3.4 |
| early_morning |        31.7 |         14.8 |             20   |         3.4 |            24.3 |              5.8 |
| morning       |        29.5 |         13.1 |             22.4 |         5.2 |            23   |              6.9 |
| afternoon     |        31   |         20.5 |             17   |         3.9 |            21.5 |              6.1 |
| evening       |        31.3 |         26.4 |             14.1 |         2.6 |            20   |              5.7 |
| late_evening  |        30.4 |         34.2 |             11.6 |         1.2 |            17.4 |              5.1 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level3_baseline`