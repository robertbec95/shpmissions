# Table 6 analog - Stage 2 run on ALL variety>=2 baskets (Stage 1 bypassed)

Same Jaccard k-medoids clustering as `docs/table6-comparison.md`, but the input population is different: instead of restricting to Stage 1's k-means-selected "relatively_concrete" band, every basket with variety>=2 is included (only single-category baskets excluded - the rule already applied when building `outputs/clustering_dataset.parquet`). Stage 1's concrete/relatively-concrete/abstract clustering plays no role in selecting this input.

**Scope:** 2,364,042 baskets (vs. 1,222,439 relatively-concrete baskets in the original `docs/table6-comparison.md` run - 1.93x the population, since former-"abstract" baskets are back in). k=6.

## Basket descriptives

| | All | Cluster 1 (meal) | Cluster 2 (snack) | Cluster 3 (breakfast) | Cluster 4 (meal) | Cluster 5 (cleaning) | Cluster 6 (breakfast) |
|---|---|---|---|---|---|---|---|
| Average basket variety (categories) | 3.87 | 3.44 | 3.52 | 4.44 | 4.99 | 4.07 | 4.35 |
| Average basket size (items) | 6.60 | 5.89 | 5.74 | 7.78 | 8.81 | 7.01 | 7.17 |
| Average basket value (RON) | 40.90 | 41.33 | 35.65 | 44.40 | 63.26 | 37.46 | 46.63 |
| Total share of baskets (%) | 100.0% | 30.84% | 21.49% | 16.99% | 3.5% | 21.2% | 5.98% |
| Number of baskets | 2,364,042 | 728,974 | 508,062 | 401,747 | 82,676 | 501,136 | 141,447 |

## Category participation probabilities (top 25 categories)

`All (baskets)` is the raw count of baskets (out of this population) containing the category; cluster columns are the % of that cluster's baskets containing it.

| Category | All (baskets) | Cluster 1 (meal) | Cluster 2 (snack) | Cluster 3 (breakfast) | Cluster 4 (meal) | Cluster 5 (cleaning) | Cluster 6 (breakfast) |
|---|---|---|---|---|---|---|---|
| HOUSEHOLD PRODUCTS | 864,267 | 31.5% | 12.9% | 60.6% | 69.7% | 51.1% | 8.4% |
| SUPPLIED BAKERY | 485,579 | 4.8% | 6.7% | 82.4% | 0.0% | 3.6% | 47.5% |
| VEGETABLES | 475,630 | 30.4% | 4.8% | 20.3% | 34.7% | 19.5% | 15.3% |
| FRUITS | 468,469 | 3.5% | 6.3% | 18.7% | 21.1% | 61.5% | 7.1% |
| CARBONATED DRINKS | 464,826 | 5.5% | 61.3% | 5.0% | 7.7% | 6.8% | 37.3% |
| WATER | 415,903 | 2.4% | 15.6% | 9.7% | 7.4% | 52.8% | 6.9% |
| SALTY SNACKS | 410,575 | 5.6% | 57.2% | 6.0% | 6.7% | 8.4% | 5.0% |
| TOBACCO | 402,798 | 31.1% | 14.3% | 7.7% | 4.9% | 9.9% | 13.4% |
| SWEET SNACKS | 364,008 | 14.2% | 20.3% | 13.6% | 10.0% | 15.0% | 13.7% |
| BEER | 361,113 | 28.7% | 10.8% | 8.0% | 7.4% | 8.7% | 10.4% |
| MILK CREAM | 276,264 | 4.4% | 3.9% | 9.3% | 16.4% | 11.6% | 81.7% |
| YOGURT  SANA KEFIR DESSERT | 264,674 | 10.0% | 8.4% | 13.2% | 12.7% | 12.9% | 15.1% |
| CHOCOLATE | 215,909 | 9.5% | 11.8% | 7.3% | 5.9% | 8.4% | 7.6% |
| PACKED SAUSAGES | 205,254 | 7.9% | 5.2% | 14.8% | 11.7% | 7.0% | 12.4% |
| CHEESE PACKED | 190,660 | 7.3% | 4.8% | 12.0% | 10.5% | 7.7% | 12.2% |
| ICE CREAM AND DESSERTS | 190,546 | 8.5% | 9.4% | 7.1% | 5.9% | 7.5% | 7.1% |
| BAKE OFF | 164,824 | 8.9% | 5.4% | 6.1% | 5.9% | 7.1% | 5.6% |
| DESSERT DRINKS | 154,638 | 7.1% | 4.8% | 7.6% | 7.0% | 6.1% | 8.1% |
| CHICKEN MEAT | 147,291 | 2.0% | 1.9% | 7.4% | 100.0% | 1.0% | 3.8% |
| OTHER DRINKS | 142,686 | 9.1% | 7.1% | 3.3% | 2.6% | 4.0% | 3.2% |
| RICE BEANS PASTA | 134,525 | 5.7% | 2.7% | 7.1% | 12.2% | 5.5% | 8.8% |
| COFFEE CORNER | 134,302 | 7.9% | 6.5% | 3.2% | 2.3% | 4.9% | 2.6% |
| NONCARBONATED DRINKS | 126,569 | 5.8% | 6.8% | 4.4% | 3.8% | 4.6% | 4.2% |
| SUGAR CONFECTIONERY | 101,453 | 4.3% | 6.3% | 3.4% | 2.9% | 3.5% | 3.4% |
| BULK SAUSAGES | 99,912 | 4.0% | 2.0% | 7.7% | 7.6% | 3.3% | 4.8% |

Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_variety_ge2.py --k 6`