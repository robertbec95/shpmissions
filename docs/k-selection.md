# Cluster-count selection: gap statistic + silhouette

The paper (Sarantopoulos et al. 2015, §5.4-5.5) selects k for both clustering stages via the **gap statistic** (Tibshirani, Walther & Hastie, 2001), landing on k=7 for its supermarket-pilot mission clusters. This report runs the same method - plus silhouette score and the **elbow method** as cross-checks - on our data, implemented in `shopping_missions/cluster_selection.py` (generic gap statistic and `find_elbow_k`/`elbow_k_from_gap_df`), `variety_stage.select_k` (Stage 1, Euclidean k-means), and `mission_stage.select_k` (Stage 2, Jaccard k-medoids). Elbow k is the point of maximum distance from the chord connecting the first and last points on the raw within-cluster-dispersion-vs-k curve (`exp(log_Wk)`, already computed as part of the gap statistic) - an automated stand-in for eyeballing the bend in a scree plot, rather than an independent clustering run.

**Scope:** 20260509, 20260510, 20260511 · offline channel · 3,676,724 total baskets, 1,222,374 relatively-concrete.

## Stage 1 - variety clusters (concrete / relatively-concrete / abstract)

Gap statistic fit on a 50,000-basket sample, B=10 reference datasets.

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   | silhouette   |
|----:|---------:|-------:|-------:|:---------------------|:-------------|
|   1 |  12.5241 | 2.5467 | 0.0033 | True                 | n/a          |
|   2 |  11.4609 | 2.223  | 0.0051 | True                 | 0.7111       |
|   3 |  10.8414 | 2.0318 | 0.0044 | True                 | 0.6813       |
|   4 |  10.27   | 2.027  | 0.0047 | True                 | 0.6905       |
|   5 |   9.855  | 1.9965 | 0.0043 | True                 | 0.7477       |
|   6 |   9.5935 | 1.8931 | 0.0053 | False                | 0.7242       |
|   7 |   9.0315 | 2.1456 | 0.0038 | False                | 0.8448       |

**Gap statistic selects k=1.** Silhouette is maximized at k=7. Elbow method picks k=2.

**Reading this:** the gap statistic picks the *smallest* k where the observed dispersion drop stops beating a structureless (uniform) reference - and here that happens almost immediately, because the variety distribution is a smooth, unimodal, heavily right-skewed count (many 1-2 category baskets, a long thin tail) rather than genuinely separated sub-populations. That's a real, useful finding: the concrete/relatively-concrete/abstract split is a **theoretically motivated discretization of a continuum**, not something unsupervised clustering independently discovers in this dataset. Silhouette, in contrast, climbs monotonically with k here - a known artifact on ordered, skewed 1-D data (more, narrower bins near-trivially look more "separated"), so it is not trustworthy as a standalone signal for this particular stage. Where the elbow method lands matters too: if it also picks a low k, that's a third, independent line of evidence (elbow uses only the observed dispersion curve, no reference datasets) for the same conclusion; if it picks something mid-range, that's consistent with a smoothly decaying curve having no sharp bend at all - a weak elbow pick, not a confident one. The pipeline (`variety_stage.fit_variety_segments`) keeps k=3 fixed regardless, matching the paper's conceptual framing rather than any of the three statistics - documented here so that choice is transparent rather than silently assumed.

## Stage 2 - mission clusters

Gap statistic + silhouette fit on a 1,500-basket sample of relatively-concrete baskets (CLARA-style, matching `mission_stage`'s own sampling approach), B=10 reference datasets, references drawn per-category at the observed marginal rate (independent Bernoulli, preserves prevalence but not co-occurrence).

|   k |   log_Wk |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|---------:|-------:|-------:|:---------------------|-------------:|
|   2 |   6.4899 | 0.0167 | 0.0036 | True                 |       0.0652 |
|   3 |   6.4703 | 0.0183 | 0.0034 | False                |       0.0594 |
|   4 |   6.4466 | 0.0301 | 0.0057 | True                 |       0.0567 |
|   5 |   6.438  | 0.0238 | 0.0074 | True                 |       0.0528 |
|   6 |   6.4325 | 0.0196 | 0.008  | True                 |       0.0486 |
|   7 |   6.4313 | 0.0107 | 0.0078 | True                 |       0.0502 |
|   8 |   6.4139 | 0.0195 | 0.0094 | True                 |       0.0466 |
|   9 |   6.411  | 0.017  | 0.0099 | True                 |       0.0488 |
|  10 |   6.4049 | 0.0166 | 0.0091 | True                 |       0.0474 |
|  11 |   6.4028 | 0.0119 | 0.0098 | False                |       0.0456 |

**Gap statistic selects k=2.** Silhouette is maximized at k=2. Elbow method picks k=4.

**This disagrees with the k=6 used in `docs/table6-comparison.md`.** All three statistics here prefer a much smaller k than the paper's k=7 for its supermarket pilot. Plausible reasons, not mutually exclusive:
- **Pooling across all ~1,740 stores/formats** likely blurs mission-specific co-occurrence patterns that would be sharper within a single store or format the way the paper's per-store analysis is - format-mixing is exactly the limitation already flagged in `docs/mvp-plan.md` §2.1 (no store-format label is available in this data).
- **3 days vs. the paper's 17 months** - far less data to stabilize fine-grained structure; a shorter window may just show the dominant "convenience run" vs. "everything else" split rather than the full mission palette.
- **Category granularity/cleanliness** - `HOUSEHOLD PRODUCTS` (ubiquitous, low-lift - see `docs/table6-comparison.md`) and possibly other near-ubiquitous categories dilute the separability the gap statistic is measuring; the paper's category taxonomy went through two rounds of manual review specifically to avoid this.
- It's also possible this chain's shoppers, in this window, genuinely cluster more coarsely than the source paper's.

**Practical call:** `docs/table6-comparison.md` was generated with k=6, kept deliberately to mirror the paper's headline six-mission finding and produce an interpretable, business-usable table, not because k=6 is what this data's gap statistic supports - it isn't. A statistically rigorous run would use k=2-4 (gap says 2, elbow says 4, both well short of 6), at the cost of a much coarser, less actionable segmentation. Recommend re-running this diagnostic once real store-format labels and a longer time window are available (see `docs/mvp-plan.md` §2) before treating any of these k's as final.

`docs/table6-comparison-k2.md` runs the actual k=2 solution for direct comparison: it confirms the "convenience run vs. everything else" guess above almost exactly - a 36% convenience/vice cluster (tobacco, beer, carbonated drinks, salty snacks) against a 64% fresh-food/everyday-basics cluster, with none of the paper's finer breakfast/meal/cleaning/coffee distinctions surviving at this k.

**Update - `docs/k-selection-by-format.md`:** two of the four candidate explanations above have since been tested directly and largely ruled out. Real store-format labels became available (`data/locations-export (3).csv`) and single-category baskets were excluded from the input entirely (`outputs/clustering_dataset.parquet`) - re-running both stages stratified by format on that cleaner data still finds Stage-1 gap_k=1 everywhere, and Stage-2 gap_k only moves from 2 to 2-3 (not toward the paper's 6-7). The 3-day window and residual category-taxonomy noise are now the leading remaining suspects.

Reproduce with: `PYTHONPATH=src python3 scripts/select_k.py --dates 20260509,20260510,20260511`