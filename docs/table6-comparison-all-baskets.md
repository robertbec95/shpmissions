# Table 6 analog - Stage 2 run on ALL baskets (no variety floor at all)

Most permissive of the three Stage-2 population variants: every basket goes in, including single-category (variety=1) ones - only negative-quantity/return baskets are excluded (that's a data-quality fix from `ingest.py`, unrelated to variety). Compare against `docs/table6-comparison.md` (Stage 1's relatively_concrete band only) and `docs/table6-comparison-variety-ge2.md` (variety>=2).

**Scope:** 3,676,724 baskets (vs. 2,364,042 in the variety>=2 run, vs. 1,222,439 relatively-concrete-only). k=6.

## Basket descriptives

| | All | Cluster 1 (unlabeled) | Cluster 2 (breakfast) | Cluster 3 (snack) | Cluster 4 (meal) | Cluster 5 (unlabeled) | Cluster 6 (unlabeled) |
|---|---|---|---|---|---|---|---|
| Average basket variety (categories) | 2.84 | 2.04 | 3.39 | 2.76 | 4.62 | 1.93 | 2.04 |
| Average basket size (items) | 4.85 | 3.42 | 5.80 | 4.29 | 8.19 | 3.01 | 3.92 |
| Average basket value (RON) | 31.09 | 19.87 | 34.27 | 29.78 | 46.14 | 41.66 | 23.39 |
| Total share of baskets (%) | 100.0% | 39.17% | 9.75% | 9.95% | 23.65% | 10.97% | 6.51% |
| Number of baskets | 3,676,724 | 1,440,348 | 358,508 | 365,770 | 869,449 | 403,243 | 239,406 |

## Category participation probabilities (top 25 categories)

`All (baskets)` is the raw count of baskets (out of this population) containing the category; cluster columns are the % of that cluster's baskets containing it.

| Category | All (baskets) | Cluster 1 (unlabeled) | Cluster 2 (breakfast) | Cluster 3 (snack) | Cluster 4 (meal) | Cluster 5 (unlabeled) | Cluster 6 (unlabeled) |
|---|---|---|---|---|---|---|---|
| HOUSEHOLD PRODUCTS | 893,994 | 4.0% | 19.4% | 12.7% | 76.7% | 5.6% | 12.9% |
| TOBACCO | 614,778 | 3.7% | 12.0% | 15.8% | 6.6% | 100.0% | 0.0% |
| SUPPLIED BAKERY | 560,869 | 3.5% | 100.0% | 0.0% | 17.5% | 0.0% | 0.0% |
| CARBONATED DRINKS | 555,197 | 4.7% | 12.5% | 100.0% | 8.9% | 0.0% | 0.0% |
| WATER | 519,654 | 30.6% | 0.0% | 0.0% | 9.0% | 0.0% | 0.0% |
| VEGETABLES | 505,230 | 0.9% | 6.8% | 2.3% | 51.0% | 2.1% | 3.0% |
| FRUITS | 502,189 | 1.5% | 6.0% | 2.5% | 50.1% | 2.0% | 2.7% |
| BEER | 478,844 | 2.9% | 12.0% | 10.6% | 7.8% | 11.7% | 100.0% |
| SALTY SNACKS | 453,493 | 12.8% | 11.0% | 16.9% | 13.5% | 5.8% | 11.0% |
| SWEET SNACKS | 394,018 | 11.3% | 10.8% | 13.1% | 12.7% | 5.0% | 5.5% |
| MILK / CREAM | 303,625 | 7.1% | 10.9% | 4.4% | 14.3% | 3.3% | 3.7% |
| YOGURT  / SANA / KEFIR, DESSERT | 281,419 | 6.7% | 9.2% | 4.9% | 13.3% | 2.8% | 3.1% |
| ICE CREAM AND DESSERTS | 244,958 | 8.5% | 5.1% | 7.1% | 6.7% | 2.6% | 3.9% |
| CHOCOLATE | 237,216 | 7.2% | 5.5% | 8.2% | 7.2% | 3.3% | 3.6% |
| COFFEE CORNER | 225,815 | 10.7% | 2.3% | 6.2% | 2.4% | 3.8% | 1.9% |
| PACKED SAUSAGES | 213,710 | 3.9% | 11.7% | 3.1% | 10.4% | 1.8% | 2.9% |
| CHEESE PACKED | 198,283 | 3.8% | 8.7% | 2.7% | 10.5% | 1.6% | 2.1% |
| BAKE OFF | 197,961 | 6.8% | 4.3% | 4.3% | 6.2% | 2.6% | 2.2% |
| OTHER DRINKS | 186,147 | 7.2% | 2.8% | 5.8% | 2.5% | 5.9% | 2.4% |
| DESSERT DRINKS | 177,220 | 4.6% | 5.8% | 3.3% | 6.8% | 2.9% | 2.7% |
| CHICKEN MEAT | 155,976 | 2.1% | 5.3% | 1.9% | 10.4% | 1.1% | 2.1% |
| RICE / BEANS / PASTA | 143,324 | 2.8% | 4.3% | 1.9% | 8.2% | 1.1% | 1.9% |
| NONCARBONATED DRINKS | 139,946 | 4.2% | 3.3% | 4.5% | 4.2% | 2.2% | 2.7% |
| CHEWING GUM | 110,848 | 3.9% | 1.9% | 3.7% | 2.4% | 2.6% | 1.4% |
| SUGAR CONFECTIONERY | 110,297 | 3.3% | 2.5% | 4.1% | 3.4% | 1.5% | 1.6% |

Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_all_baskets.py --k 6`