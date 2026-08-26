# v7: folding in every remaining real attribute — worse than v5/v6, not just neutral

v7 adds the last 14 real, non-synthetic columns pulled this session that weren't already in v6: population growth trend, unemployment trend (distinct from the level already in the table), school enrollment %, tourism overnight-stays per capita, county-level car ownership, and 9 banner-level competitor counts (Mega Image, Carrefour, Penny, etc.). Ethnicity/religion/mother-tongue from the 2021 census was **explicitly excluded** — the user was asked directly whether to make it a weighted input to the site-scoring algorithm versus keeping it as reference-only data, and chose reference-only. That data lives in `census2021_demographics.json`, untouched by this table.

Same disciplined grid-search pattern as v5/v6: all 8 previously-established group weights held fixed (v4's original 4 at 0.5/2.0/4.0/0.5, v5's catchment/economic at 1.0/0.25, v6's poi_detail/competition at 0.25/0.25). Only the 3 new groups were searched — `growth`, `footfall_activity`, `banner` — 125 combinations.

## Headline: for the first time, not even the low-weight case ties baseline

| Configuration | L1 | Cosine | Spearman |
|---|---|---|---|
| v4/v5/v6 baseline | 18.11 | 0.9750 | 0.851 |
| **Best possible v7** (all 3 new groups at weight 0.25) | **18.95** | 0.9739 | 0.848 |
| v7, heaviest weight on all 3 new groups | 21.19 | 0.9667 | 0.818 |

**Zero of 125 combinations tie or beat the baseline** - a first for this project. Every prior addition (v5, v6) always had at least one low-weight configuration that exactly matched 18.11; this time the floor itself is 0.84 worse.

## Why this one is worse, not just neutral

v5 and v6 each added a handful of columns per new group (2-3 for v5's catchment/economic, 1-8 for v6's competition/poi_detail). v7 adds **14 columns across 3 groups**, 9 of them in a single sparse `banner` group where most stores show a nonzero count in only 2-4 of the 9 possible banners. Gower's per-column weighting means a group's *total* contribution to the distance scales with how many columns are in it, not just the group weight - a `banner` weight of 0.25 still means 9 separate columns each pulling at that weight, collectively outweighing a smaller group at the same nominal setting. More sparse columns at n=6 doesn't dilute gracefully into "does nothing" the way one or two extra columns did before - it actively degrades the signal even at the lowest weight tested.

## Verdict

**v4 weighted Gower + k=3 k-NN remains the recommendation - now three separate attempts (v5, v6, v7) have all failed to beat it, and v7's failure mode is worse than the first two.** This isn't a reason to distrust the new data itself (population/unemployment trends, tourism intensity, banner mix are all real, correctly-extracted signals) - it's a reason to distrust adding many more feature dimensions to a 6-point Gower distance without either far more analyzed stores or a principled dimensionality-reduction step (e.g. PCA-summarizing the banner columns the way POI counts were summarized into PC1/PC2, rather than feeding 9 raw sparse columns in directly).

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v7_validation.py`
