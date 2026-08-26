# Table 6 analog — shopping-mission cluster solution (our data)

Reproduces the structure of **Table 6** in Sarantopoulos et al. (2015) — the final cluster solution for relatively-concrete baskets, with basket descriptives and category participation probabilities — computed on our own dataset instead of their supermarket-pilot store.

**Scope:** 20260509, 20260510, 20260511 · offline channel · all 3,677,678 baskets pooled across all stores (no store-format label is available in this dataset — see `docs/mvp-plan.md` §2 — so this is the whole-chain analog of one of the paper's per-store tables, not a single supermarket). k=6.

## Basket descriptives

| | All | Cluster 1 (breakfast) | Cluster 2 (cleaning) | Cluster 3 (light_meal) | Cluster 4 (snack) | Cluster 5 (meal) | Cluster 6 (meal) |
|---|---|---|---|---|---|---|---|
| Average basket variety (categories) | 3.99 | 3.72 | 3.98 | 3.93 | 3.87 | 4.08 | 4.47 |
| Average basket size (items) | 6.57 | 5.77 | 6.91 | 6.54 | 6.17 | 6.74 | 7.16 |
| Average basket value (RON) | 40.21 | 40.70 | 50.12 | 34.99 | 34.54 | 33.88 | 39.91 |
| Total share of baskets (%) | 100.0% | 16.2% | 27.1% | 11.9% | 14.2% | 20.7% | 9.8% |
| Number of baskets | 1,222,439 | 197,817 | 331,517 | 145,996 | 173,328 | 253,608 | 120,173 |

## Category participation probabilities (top 25 categories)

`All (baskets)` is the raw count of relatively-concrete baskets containing the category (matches the paper's format); cluster columns are the % of that cluster's baskets containing the category (participation probability).

| Category | All (baskets) | Cluster 1 (breakfast) | Cluster 2 (cleaning) | Cluster 3 (light_meal) | Cluster 4 (snack) | Cluster 5 (meal) | Cluster 6 (meal) |
|---|---|---|---|---|---|---|---|
| HOUSEHOLD PRODUCTS | 525,676 | 8.4% | 59.3% | 7.8% | 14.3% | 80.9% | 59.1% |
| FRUITS | 273,005 | 3.9% | 8.9% | 13.5% | 9.2% | 62.4% | 34.7% |
| SUPPLIED BAKERY | 272,357 | 42.9% | 16.7% | 22.7% | 9.5% | 24.6% | 16.7% |
| VEGETABLES | 271,502 | 3.9% | 10.1% | 9.5% | 5.6% | 62.3% | 40.6% |
| CARBONATED DRINKS | 246,459 | 34.4% | 12.5% | 6.8% | 59.5% | 6.9% | 5.4% |
| SALTY SNACKS | 227,148 | 4.1% | 16.3% | 11.8% | 70.0% | 7.3% | 6.6% |
| WATER | 224,086 | 47.7% | 11.2% | 16.3% | 15.7% | 13.0% | 7.0% |
| TOBACCO | 209,839 | 31.5% | 37.6% | 1.9% | 6.9% | 1.3% | 3.9% |
| SWEET SNACKS | 199,060 | 3.5% | 9.6% | 46.5% | 40.6% | 5.9% | 5.9% |
| BEER | 193,803 | 4.4% | 49.6% | 1.4% | 5.0% | 1.7% | 4.7% |
| MILK / CREAM | 145,943 | 8.6% | 4.7% | 11.6% | 6.1% | 0.0% | 71.3% |
| YOGURT  / SANA / KEFIR, DESSERT | 139,441 | 3.8% | 7.4% | 49.4% | 1.9% | 8.0% | 9.9% |
| CHOCOLATE | 118,490 | 11.0% | 8.6% | 12.8% | 16.8% | 5.7% | 5.1% |
| PACKED SAUSAGES | 102,113 | 4.1% | 6.5% | 34.4% | 0.9% | 5.4% | 5.7% |
| ICE CREAM AND DESSERTS | 99,228 | 9.6% | 8.0% | 7.6% | 11.2% | 6.4% | 5.7% |
| CHEESE PACKED | 92,893 | 7.0% | 5.4% | 16.4% | 3.8% | 7.5% | 9.6% |
| BAKE OFF | 88,755 | 9.4% | 6.2% | 10.3% | 6.5% | 7.0% | 4.7% |
| DESSERT DRINKS | 77,121 | 7.7% | 6.5% | 7.0% | 5.5% | 5.4% | 5.8% |
| CHICKEN MEAT | 76,464 | 4.0% | 7.8% | 3.8% | 2.1% | 9.0% | 8.9% |
| OTHER DRINKS | 69,994 | 8.9% | 6.9% | 5.3% | 8.5% | 1.9% | 1.7% |
| NONCARBONATED DRINKS | 67,154 | 6.6% | 6.1% | 5.7% | 8.8% | 2.9% | 2.5% |
| RICE / BEANS / PASTA | 64,737 | 2.8% | 1.6% | 3.1% | 2.0% | 0.0% | 38.1% |
| COFFEE CORNER | 63,179 | 9.2% | 4.7% | 5.8% | 7.6% | 2.4% | 1.5% |
| SUGAR CONFECTIONERY | 53,919 | 5.0% | 4.0% | 5.2% | 8.8% | 2.1% | 2.0% |
| CHEWING GUM | 51,328 | 6.8% | 3.7% | 4.6% | 6.4% | 2.2% | 1.8% |

## How this differs from the paper's Table 6

| | Paper (Table 6, supermarket pilot) | This report |
|---|---|---|
| Scope | 1 store (supermarket pilot, 1,000 m²) | All ~1,740 stores pooled — no store-format label exists in this dataset yet (see `docs/mvp-plan.md` §2), so store- or format-level cluster solutions aren't possible yet |
| Time window | 17 months (Jan 2012–May 2013) | 3 days (2026-05-09 to 2026-05-11) — enough to sanity-check the pipeline, not to claim stable missions |
| Categories | 75, fixed taxonomy level chosen by 2 researchers + retailer round-table | 99 `Catalog Name` values, used as-is (see `docs/mvp-plan.md` §2.2) |
| Relatively-concrete baskets | 309,591 of 412,786 (variety 3–16) | 1,222,439 of 3,677,678 (variety 3–6, bounds fit per-run by k-means, see `docs/mvp-plan.md` Step D) |
| k | 7 (gap statistic) | 6, fixed — chosen to mirror the paper's overall 6-mission finding, **not** statistically derived from this data. Gap statistic + silhouette are now both implemented (`cluster_selection.py`) and actually disagree with k=6 here, preferring k=2 — see `docs/k-selection.md` for the full diagnostic and why that's likely (pooled stores/formats, 3 days vs. 17 months) rather than a pipeline bug |
| Cluster labels | Assigned by the authors reading category patterns | Auto-suggested by keyword heuristic (`profiling.py`), confidence-scored — **treat as a draft, not ground truth**. Cluster 2 above is a beer/tobacco/water "convenience run" mislabeled "cleaning" purely because `HOUSEHOLD PRODUCTS` (bundled produce bags) is both a cleaning keyword and ubiquitous — a known heuristic weakness already flagged when the pipeline was built |
| Currency | € | RON (Romanian leu) |

Reproduce with: `PYTHONPATH=src python3 scripts/generate_table6.py --dates 20260509,20260510,20260511`

**See also:** `docs/table6-comparison-k2.md` — the same table at k=2, the value gap statistic and silhouette both actually prefer for this data (`docs/k-selection.md`). At k=2 the six missions collapse into two starkly separated, easy-to-name clusters: a fresh-food/everyday-basics run (64%) and a convenience/vice run — tobacco, beer, carbonated drinks, salty snacks (36%). Worth reading alongside this table before treating the six-mission breakdown as confirmed rather than assumed.
