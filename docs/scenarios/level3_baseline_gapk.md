# Scenario: level3_baseline_gapk

The baseline/most-proven scenario: paper-faithful Stage 2 (Jaccard k-medoids) on baskets with variety>=2, category level 3, 3-day window - with k chosen by the gap statistic (the paper's own k-selection method, Tibshirani/Walther/Hastie 2001) rather than a fixed k=6 copied from the paper's pilot store. Re-verified 2026-07-21: on the current population (post HOUSEHOLD PRODUCTS revert) gap_k=4, with elbow_k=7 and silhouette_k=8 as comparison/support diagnostics, not decision criteria - a previous run of this scenario used a stale k=3 computed before that revert.

**Config:** category_level=3 | dates=20260509,20260510,20260511 | min_variety=2 | k=4 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 2,364,042 baskets, 99 categories seen, run in 25.7s
. k-selection at this population: gap_k=4, silhouette_k=8, elbow_k=7, silhouette@k=4=0.0585

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | meal (conf 0.33) | 42.83% | 40.46 | 3.47 | BEER (24%), TOBACCO (27%), VEGETABLES (27%) | no |
| 1 | snack (conf 1.0) | 24.36% | 38.18 | 3.74 | CARBONATED DRINKS (64%), SALTY SNACKS (54%), SWEET SNACKS (20%) | no |
| 2 | breakfast (conf 0.67) | 20.21% | 45.92 | 4.56 | SUPPLIED BAKERY (78%), HOUSEHOLD PRODUCTS (61%), MILK / CREAM (15%) | no |
| 3 | meal (conf 0.25) | 12.61% | 39.56 | 4.35 | FRUITS (100%), HOUSEHOLD PRODUCTS (69%), YOGURT  / SANA / KEFIR, DESSERT (15%), VEGETABLES (26%) | **YES** |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (meal) |   C1 (snack) |   C2 (breakfast) |   C3 (meal) |
|:----------|------------:|-------------:|-----------------:|------------:|
| Timis     |        47.7 |         27.1 |             14.3 |        10.9 |
| Cluj      |        49   |         25.6 |             13.8 |        11.6 |
| Bucharest |        44.9 |         24.4 |             18.1 |        12.6 |
| Dolj      |        37.9 |         21.4 |             28.1 |        12.6 |
| Bihor     |        46.7 |         26.2 |             16.6 |        10.5 |
| Constanta |        41.5 |         24.4 |             21.9 |        12.1 |
| Suceava   |        39.7 |         26.3 |             21.6 |        12.4 |
| Bacau     |        42.4 |         23   |             21.3 |        13.3 |
| Iasi      |        41.6 |         23.3 |             20.1 |        15.1 |
| Arges     |        37.7 |         23.7 |             24.8 |        13.8 |

## Cluster mix by day-part

| day_part      |   C0 (meal) |   C1 (snack) |   C2 (breakfast) |   C3 (meal) |
|:--------------|------------:|-------------:|-----------------:|------------:|
| night         |        46.8 |         37.6 |             10.5 |         5.1 |
| early_morning |        46.3 |         16.8 |             23.4 |        13.4 |
| morning       |        41.5 |         15.8 |             26.7 |        16.1 |
| afternoon     |        43   |         23.5 |             20.3 |        13.2 |
| evening       |        42.8 |         29.4 |             16.8 |        11   |
| late_evening  |        41.6 |         37.3 |             13.6 |         7.5 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level3_baseline_gapk`