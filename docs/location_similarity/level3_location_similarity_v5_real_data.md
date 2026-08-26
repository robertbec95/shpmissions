# v5: folding in real unemployment, GDP, and catchment population — does it help?

v4 (9 columns, weighted Gower, k=3 k-NN) was the standing recommendation. This session pulled three genuinely new real-data sources at national scale - WorldPop population raster, the OSM/Geofabrik POI extract, INS unemployment, Eurostat GDP - and this test asks the direct question: does folding them into the propagation pipeline actually improve leave-one-out accuracy, or just add noise?

v4's 4 group weights (primary, poi, population, age) were held **fixed** at their already-validated values (0.5/2.0/4.0/0.5), not re-searched - re-optimizing all 6 groups at once over 6 stores would be close to pure noise. Only the 2 new groups (catchment population, economic) were grid-searched, 25 combinations, against leave-one-out k=3 L1/cosine/spearman using the joint NMF's cached ground truth.

## Headline: none of the three new attributes beat v4

| Configuration | L1 | Cosine | Spearman |
|---|---|---|---|
| v4 baseline (unchanged) | 18.11 | 0.9750 | 0.851 |
| Best v5 (catchment=2.0, economic=0.25) | 18.11 | 0.9750 | 0.851 |
| v5, moderate weight on both new groups | 19.12 | 0.9734 | 0.846 |
| v5, economic weight=4.0 | 20.30–20.85 | 0.968–0.971 | 0.822–0.829 |

The best possible v5 configuration **ties** v4 exactly - it never beats it. Every combination that gives the new "economic" group (unemployment + GDP) meaningful weight makes accuracy monotonically **worse**. Catchment population (1km-radius WorldPop figure) is neutral at best.

## Why this isn't surprising, on reflection

At n=6 stores across 5 counties, unemployment and GDP per capita are each **one value per county** - Bucharest's two stores share an identical value on both. That's almost perfectly collinear with information the existing `population`/`primary` groups already encode indirectly (bigger, richer Romanian cities already have systematically different population scale and age structure). Adding two more columns that don't carry meaningfully independent information, at real weight, just dilutes the already-tuned signal - the same mechanism that made the v3 age-feature regression fail before it was fixed by weighting, and the same small-n ceiling the ridge-regression propagation test hit.

Catchment population (1km around the exact store, not the whole city) is conceptually the most promising of the three - a genuinely different signal from whole-city population - but is also the most exposed to store-specific siting noise (a store on a wide commercial strip vs. deep in a residential block reads very differently even within the same city), which likely explains why it doesn't clearly help either at only 6 data points.

## Verdict

**v4 remains the recommended feature set for propagation.** The three new real-data sources are still independently valuable - national POI/population coverage is a new capability this project didn't have before (see the catchment-profile validation), and unemployment/GDP are legitimate site-selection context - but folding them into THIS SPECIFIC propagation pipeline at n=6 doesn't pay off, and shouldn't be presented as an improvement. Worth re-testing once the analyzed-store set grows past 6, same caveat attached to every other small-n finding in this thread.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v5_validation.py`
