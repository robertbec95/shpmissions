# v6: folding in granular POI counts + nearest-competitor distance — same verdict as v5

v5 added catchment population, unemployment, and GDP - none beat v4. This round adds the remaining 9 on-demand `catchment.py` attributes that were computed during v5's build but never saved as columns: 8 POI category counts (schools, kindergartens, places of worship, pharmacies, banks, competitor supermarkets/convenience, transit stops - `poi_competitor_grocery_count` dropped, zero variance across all 6 stores) plus `nearest_competitor_km`.

Kept as 2 new groups, separate from the existing `poi` group (PC1/PC2, still the old live-Overpass pull, unchanged) - merging raw counts into the same group as their own PCA summary would double-count the same underlying signal. Same disciplined grid-search pattern as v5: every previously-established weight held fixed (v4's 4 groups at 0.5/2.0/4.0/0.5, v5's tied-best catchment/economic at 1.0/0.25), only the 2 new groups' weights searched, 25 combinations, against leave-one-out k=3 propagation accuracy.

## Headline: same result as v5 - ties, never beats

| Configuration | L1 | Cosine | Spearman |
|---|---|---|---|
| v4/v5 baseline | 18.11 | 0.9750 | 0.851 |
| Best v6 (poi_detail=0.25, competition=0.25) | 18.11 | 0.9750 | 0.851 |
| v6, poi_detail weight=1.0 | 19.00–19.55 | 0.970–0.971 | 0.838–0.843 |
| v6, poi_detail weight=4.0 | 19.41–20.24 | 0.968–0.972 | 0.828–0.838 |

7 of 25 combinations tie the baseline (all at low weight on the new groups); every combination that gives the new groups real weight is worse, monotonically, same shape as v5's result.

## Why

The 8 POI counts are drawn from the same catchment (1km around each store) that PC1/PC2 already summarize - they're not an independent signal, they're a more granular decomposition of a signal already captured. At n=6, more granular columns just mean more ways for a single store's idiosyncratic count (e.g. a store that happens to sit near an unusually large cluster of schools) to add noise without a large enough sample to separate "real catchment signal" from "one store's local quirk." `nearest_competitor_km` has the same problem in miniature - it's a single number per store, easily dominated by hyperlocal siting (literally which side of the street the nearest shop happens to be on) rather than catchment quality.

## Verdict

**v4 weighted Gower + k=3 k-NN remains the recommendation, confirmed a second time.** Two independent attempts to fold in new real-data groups (v5: catchment population/unemployment/GDP; v6: granular POI counts/nearest-competitor distance) both land on the identical pattern - ties at low weight, degrades at real weight. This is now a repeated, not a one-off, finding: at n=6, additional feature groups beyond the original 4 don't have room to prove themselves useful, regardless of how real or well-sourced the underlying data is. The ceiling is sample size, not data quality - re-test if the analyzed-store set ever grows meaningfully past 6.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v6_validation.py`
