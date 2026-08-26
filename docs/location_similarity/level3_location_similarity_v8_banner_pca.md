# v8: PCA-summarizing the banner columns fixes v7's regression

v7 added 9 raw banner-level competitor columns and, for the first time in this project, made propagation accuracy actively worse even at the lowest tested weight (best L1=18.95 vs. the 18.11 baseline, 0/125 combinations tied or beat it). The diagnosis was that Gower's per-column weighting makes a group's total pull scale with its column count, and 9 sparse columns (most stores nonzero in only 2-4 of them) doesn't dilute gracefully.

v8 tests the fix directly: same 9 raw banner columns, run through the exact same StandardScaler + PCA(n_components=2) methodology already used for the original POI PC1/PC2 (2 components explain 80.3% of variance: 65.5% + 14.8%). Everything else is identical to v7 - same 8 previously-established groups held fixed, same 125-combination grid search over growth/footfall_activity/banner.

## Result: back to neutral, matching v4/v5/v6 exactly

| Version | Banner representation | Best L1 | Combos tying/beating baseline |
|---|---|---|---|
| v7 | 9 raw columns | 18.95 (worse) | 0 / 125 |
| **v8** | **2 PCA components** | **18.11 (exact tie)** | **10 / 125** |

The fix worked precisely as hypothesized - PCA-summarizing brought the banner group back to the same "ties but doesn't beat" pattern every other addition in this project has shown (v5's catchment/economic, v6's poi_detail/competition), rather than v7's unique regression. The underlying banner-mix signal isn't harmful, it just needed to be compressed to 2 dimensions like every other high-cardinality feature group in this pipeline (the original POI counts went through identical treatment before ever reaching the table) - v7's mistake was skipping that step for a new group instead of assuming raw columns would behave the same as everywhere else.

## Verdict

**Still doesn't beat v4 - ties it exactly, same as every other addition tried this project.** But this closes the loop on the specific hypothesis from v7's writeup: the failure wasn't about the banner data being useless, it was a preventable methodology gap (skipping PCA-summarization for a new high-cardinality group). Worth remembering as a standing rule for any FUTURE new attribute group with more than ~3-4 raw columns: summarize first, don't feed raw sparse columns into the weighted Gower distance and hope the group weight alone protects against dilution.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v8_validation.py`
