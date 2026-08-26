# Scenario: level3_8day

Same config as level3_baseline (the winner among the 3-day scenarios), extended to 8 days (2026-05-04 through 2026-05-11, up from 3) once more data became available. Directly tests the leading remaining explanation for weak k-selection structure flagged throughout docs/k-selection*.md: too short a time window. SKU/store join coverage verified at 100% for all 5 new days against the existing reference products/locations files before running.

**Config:** category_level=3 | dates=20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511 | min_variety=2 | k=6 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 6,694,099 baskets, 99 categories seen, run in 67.3s
. k-selection at this population: gap_k=3, silhouette_k=6, elbow_k=6, silhouette@k=6=0.0697

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | breakfast (conf 0.33) | 42.64% | 38.69 | 3.86 | SUPPLIED BAKERY (38%), CARBONATED DRINKS (35%), BEER (17%) | no |
| 1 | snack (conf 0.5) | 16.23% | 28.93 | 3.45 | WATER (54%), SWEET SNACKS (46%), YOGURT  / SANA / KEFIR, DESSERT (32%), SALTY SNACKS (20%) | no |
| 2 | meal (conf 0.5) | 0.82% | 38.0 | 4.05 | GARNISHES (68%), COOKED MEALS (78%), COFFEE CORNER (26%), BAKE OFF (25%) | no |
| 3 | snack (conf 0.5) | 17.43% | 45.58 | 3.17 | TOBACCO (72%), ICE CREAM AND DESSERTS (33%) | no |
| 4 | light_meal (conf 0.33) | 3.27% | 33.06 | 3.27 | OTHERS FIIGO (33%), NONCARBONATED DRINKS (36%), CHEESE PACKED (45%) | no |
| 5 | meal (conf 0.33) | 19.6% | 43.03 | 4.58 | VEGETABLES (65%), FRUITS (65%), HOUSEHOLD PRODUCTS (77%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (breakfast) |   C1 (snack) |   C2 (meal) |   C3 (snack) |   C4 (light_meal) |   C5 (meal) |
|:----------|-----------------:|-------------:|------------:|-------------:|------------------:|------------:|
| Timis     |             39.4 |         17.5 |         0.8 |         21.7 |               3.7 |        16.9 |
| Cluj      |             36.3 |         18   |         0.8 |         22   |               4   |        19   |
| Bucharest |             39.5 |         17.4 |         0.6 |         19.8 |               4.2 |        18.6 |
| Dolj      |             48.2 |         14.3 |         1.4 |         13.5 |               2.9 |        19.6 |
| Bihor     |             41.6 |         16.6 |         0.9 |         18.7 |               3.4 |        18.8 |
| Constanta |             44   |         16.3 |         1.2 |         15.8 |               3.7 |        18.9 |
| Suceava   |             45.7 |         17   |         0.6 |         16.4 |               2.9 |        17.3 |
| Bacau     |             42.2 |         17.3 |         0.6 |         18.1 |               3.2 |        18.7 |
| Iasi      |             40.3 |         18.1 |         0.6 |         18.1 |               3.3 |        19.6 |
| Arges     |             46.2 |         14.9 |         0.8 |         15.2 |               3   |        19.9 |

## Cluster mix by day-part

| day_part      |   C0 (breakfast) |   C1 (snack) |   C2 (meal) |   C3 (snack) |   C4 (light_meal) |   C5 (meal) |
|:--------------|-----------------:|-------------:|------------:|-------------:|------------------:|------------:|
| night         |             41.8 |         21.3 |         0.2 |         26.3 |               4.1 |         6.4 |
| early_morning |             40   |         24   |         0.9 |         14.8 |               3.3 |        17   |
| morning       |             43.5 |         14.3 |         1   |         11.7 |               3.1 |        26.4 |
| afternoon     |             42.1 |         14.6 |         1.1 |         17.8 |               3.2 |        21.2 |
| evening       |             42.9 |         15.9 |         0.5 |         20.2 |               3.3 |        17.2 |
| late_evening  |             45.1 |         18.6 |         0.2 |         22   |               3.6 |        10.5 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level3_8day`