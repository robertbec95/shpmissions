# Scenario: level3_baseline_gapk_no_household

level3_baseline_gapk with HOUSEHOLD PRODUCTS (404) additionally excluded from clustering - same population/method otherwise. This re-tests the 2026-07-21 HOUSEHOLD PRODUCTS exclusion (previously reverted as net-negative against a fixed k=6) under the new gap-statistic-selected-k baseline, which is a materially different comparison point. Excluding it drops variety>=2 baskets from 2,364,042 to 2,257,930 (baskets whose only 2nd category was HOUSEHOLD PRODUCTS fall below the variety filter). gap_k=3 on this population (elbow_k=7, silhouette_k=6 as comparison), vs. gap_k=4 for the baseline - not a like-for-like k, by design, since each scenario's k is chosen independently by the same gap-statistic rule.

**Config:** category_level=3 | dates=20260509,20260510,20260511 | min_variety=2 | k=3 | sample_size=6000 | county_crosstab=True | daypart_crosstab=True

**Result:** 2,257,930 baskets, 98 categories seen, run in 30.7s
. k-selection at this population: gap_k=3, silhouette_k=6, elbow_k=7, silhouette@k=3=0.0568

## Cluster profiles

| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | breakfast (conf 0.25) | 40.65% | 40.12 | 3.21 | VEGETABLES (34%), TOBACCO (27%), FRUITS (25%), SUPPLIED BAKERY (27%) | no |
| 1 | breakfast (conf 0.33) | 27.66% | 50.12 | 4.62 | CHEESE PACKED (24%), PACKED SAUSAGES (25%), YOGURT  / SANA / KEFIR, DESSERT (32%), MILK / CREAM (32%) | no |
| 2 | snack (conf 0.67) | 31.69% | 35.63 | 3.27 | CARBONATED DRINKS (53%), SALTY SNACKS (43%), WATER (34%) | no |

## Cluster mix by county (top 10 counties by basket volume)

| county    |   C0 (breakfast) |   C1 (breakfast) |   C2 (snack) |
|:----------|-----------------:|-----------------:|-------------:|
| Timis     |             39.1 |             24.7 |         36.1 |
| Cluj      |             39.9 |             27.5 |         32.7 |
| Bucharest |             39.9 |             27.2 |         32.9 |
| Dolj      |             41.1 |             30.3 |         28.6 |
| Bihor     |             39.8 |             25.9 |         34.2 |
| Constanta |             41.3 |             26.9 |         31.8 |
| Suceava   |             39   |             28.2 |         32.8 |
| Bacau     |             42.1 |             28.1 |         29.8 |
| Iasi      |             40.1 |             29.9 |         30   |
| Arges     |             40.5 |             30.1 |         29.5 |

## Cluster mix by day-part

| day_part      |   C0 (breakfast) |   C1 (breakfast) |   C2 (snack) |
|:--------------|-----------------:|-----------------:|-------------:|
| night         |             31.1 |             22.6 |         46.3 |
| early_morning |             44.7 |             30.7 |         24.6 |
| morning       |             48.1 |             30.2 |         21.7 |
| afternoon     |             42   |             26.8 |         31.1 |
| evening       |             36.1 |             26.8 |         37.1 |
| late_evening  |             29.8 |             25.5 |         44.6 |

*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "breakfast" cluster should show a bump in morning/early_morning here.*

Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py level3_baseline_gapk_no_household`