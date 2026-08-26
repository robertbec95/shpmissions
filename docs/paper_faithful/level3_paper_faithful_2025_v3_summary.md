# Faithful reproduction v3 (2025 data): assumptions, results, limitations

6 Profi stores, full calendar-2025 purchase history each, Sarantopoulos et al. (2015)'s method applied per store as literally as this project has managed - gap statistic chosen k at both stages (not fixed), multiple k-medoids initializations at Stage 2, 60/40 train/test split, unweighted binary category presence. Total run time 120.6s.

## Assumptions and thresholds

| Decision | Value | Rationale |
|---|---|---|
| Basket definition | `transaction_id` | One checkout = one basket; verified globally unique across stores in this export |
| SKU-to-category mapping | Direct from source (`category_name` per line) | Source data already carries one consistent category per SKU line - no separate remapping needed or performed |
| Excluded categories | ['ALTELE', 'CAMPANII MARKETING', 'SERVICES'] | Non-product/administrative (services, marketing campaigns, catch-all "other") - matches this project's original-dataset exclusion list, by name since this export's category IDs are a different (hash) encoding |
| Negative-quantity lines | Dropped (`quantity_unit >= 0`) | Defensive - none actually found in this export, unlike the original dataset's raw files |
| Category weighting (Stage 2) | None - binary presence only | The paper's explicit baseline: "does not weigh the variables" |
| Distance measure (Stage 2) | Unweighted Jaccard | Paper's stated measure |
| Stage-2 k-medoids | k-centroid / PAM, 10 initializations, keep lowest dispersion | Custom Python implementation - no R/flexclust environment available; multi-init addresses this project's own instability findings |
| Train/test split | 60% / 40%, fixed random_state | Paper's stated split |
| k-selection | Gap statistic (Tibshirani et al. 2001), both stages | Paper's stated method; elbow + silhouette reported as comparison only, never the decision rule |
| Category participation threshold | support >= 15%, lift >= 1.2 | A category must appear in a meaningful share of the cluster's baskets *and* be more common than baseline to count toward a mission's profile |
| Degenerate-cluster flag | one category at >= 90% support | Flags a near-single-category split, arguably not a genuine multi-category "mission" |

## Stage 1: what the gap statistic actually says

Applied literally (not assumed to be 3), the gap statistic selected **gap_k=1 at every one of the 6 stores**

| Store | Format | gap_k (Stage 1) | Fell back to fixed k=3? |
|---|---|---|---|
| Bucharest | Supermarket | 1 | yes |
| Neamt | Supermarket | 1 | yes |
| Calarasi | Mini Market | 1 | yes |
| Dolj | Convenience Store | 1 | yes |
| Bucharest | Convenience Store | 1 | yes |
| Brasov | Convenience Store | 1 | yes |

**This is a real, honestly-reported limitation, not smoothed over**: basket variety follows a smoothly, roughly-monotonically decreasing distribution at every store (most baskets touch very few categories, progressively fewer baskets touch more) - exactly the shape that gives a formal gap-statistic decision of "1 cluster is already sufficient," since there's no genuine multi-modal break in the distribution. The paper's concrete/relatively-concrete/abstract framing is retained here as a theoretically-motivated assumption (basket variety plausibly still tracks shopping-goal abstractness even without gap-statistic support for exactly 3 natural bands), not as something this dataset's variety distribution empirically discovers on its own.

## Overview by store

| Store | Format | Total baskets | Relatively-concrete | Stage-2 gap_k | Weak/degenerate clusters |
|---|---|---|---|---|---|
| Bucharest | Supermarket | 421,770 | 163,748 | 3 | 0 of 3 |
| Neamt | Supermarket | 587,487 | 215,038 | 2 | 0 of 2 |
| Calarasi | Mini Market | 146,650 | 41,057 | 3 | 0 of 3 |
| Dolj | Convenience Store | 211,587 | 57,091 | 3 | 0 of 3 |
| Bucharest | Convenience Store | 407,460 | 128,518 | 2 | 0 of 2 |
| Brasov | Convenience Store | 118,593 | 29,488 | 3 | 0 of 3 |

## Full mission breakdown by store

### Supermarket — Bucharest

| Cluster | Name | Weak? | Share | Category participation (support, lift) |
|---|---|---|---|---|
| 0 | Supplied Bakery + Salty Snacks | no | 30.42% | SUPPLIED BAKERY (46%, x2.0), SALTY SNACKS (30%, x1.9) |
| 1 | Fruits + Vegetables + Household Products | no | 33.78% | FRUITS (63%, x2.7), VEGETABLES (43%, x2.0), HOUSEHOLD PRODUCTS (40%, x1.9) |
| 2 | Tobacco + Water + Bake Off | no | 35.8% | TOBACCO (40%, x2.1), WATER (43%, x1.9), BAKE OFF (31%, x1.8), CARBONATED DRINKS (34%, x1.8) |

### Supermarket — Neamt

| Cluster | Name | Weak? | Share | Category participation (support, lift) |
|---|---|---|---|---|
| 0 | Carbonated Drinks + Salty Snacks + Sweet Snacks | no | 55.49% | CARBONATED DRINKS (25%, x1.6), SALTY SNACKS (32%, x1.6), SWEET SNACKS (30%, x1.6), WATER (25%, x1.5) |
| 1 | Vegetables + Fruits + Household Products | no | 44.51% | VEGETABLES (41%, x2.0), FRUITS (45%, x1.8), HOUSEHOLD PRODUCTS (46%, x1.7), SUPPLIED BAKERY (47%, x1.7) |

### Mini Market — Calarasi

| Cluster | Name | Weak? | Share | Category participation (support, lift) |
|---|---|---|---|---|
| 0 | Vegetables + Fruits + Water | no | 34.15% | VEGETABLES (44%, x1.6), FRUITS (43%, x1.5), WATER (36%, x1.5), HOUSEHOLD PRODUCTS (55%, x1.4) |
| 1 | Beer + Sweet Snacks + Carbonated Drinks | no | 27.81% | BEER (43%, x2.1), SWEET SNACKS (45%, x2.0), CARBONATED DRINKS (42%, x1.7), WATER (35%, x1.5) |
| 2 | Yogurt  / Sana / Kefir, Dessert + Tobacco + Salty Snacks | no | 38.04% | YOGURT  / SANA / KEFIR, DESSERT (33%, x2.0), TOBACCO (23%, x1.9), SALTY SNACKS (44%, x1.7), SUPPLIED BAKERY (38%, x1.3) |

### Convenience Store — Dolj

| Cluster | Name | Weak? | Share | Category participation (support, lift) |
|---|---|---|---|---|
| 0 | Beer + Salty Snacks + Carbonated Drinks | no | 37.62% | BEER (23%, x2.0), SALTY SNACKS (29%, x1.8), CARBONATED DRINKS (31%, x1.7), SUPPLIED BAKERY (28%, x1.3) |
| 1 | Chocolate + Bake Off + Water | no | 38.01% | CHOCOLATE (28%, x2.2), BAKE OFF (40%, x2.2), WATER (51%, x2.2) |
| 2 | Vegetables + Fruits + Household Products | no | 24.37% | VEGETABLES (44%, x3.1), FRUITS (55%, x2.7), HOUSEHOLD PRODUCTS (58%, x2.6), SUPPLIED BAKERY (30%, x1.4) |

### Convenience Store — Bucharest

| Cluster | Name | Weak? | Share | Category participation (support, lift) |
|---|---|---|---|---|
| 0 | Sweet Snacks + Salty Snacks + Carbonated Drinks | no | 53.51% | SWEET SNACKS (24%, x1.5), SALTY SNACKS (27%, x1.4), CARBONATED DRINKS (23%, x1.3), MILK / CREAM (22%, x1.2) |
| 1 | Vegetables + Beer + Supplied Bakery | no | 46.49% | VEGETABLES (23%, x2.0), BEER (24%, x2.0), SUPPLIED BAKERY (52%, x1.9), HOUSEHOLD PRODUCTS (33%, x1.9) |

### Convenience Store — Brasov

| Cluster | Name | Weak? | Share | Category participation (support, lift) |
|---|---|---|---|---|
| 0 | Vegetables + Fruits + Household Products | no | 35.31% | VEGETABLES (36%, x2.2), FRUITS (40%, x2.0), HOUSEHOLD PRODUCTS (31%, x1.7), SUPPLIED BAKERY (31%, x1.7) |
| 1 | Sweet Snacks + Chocolate + Salty Snacks | no | 35.35% | SWEET SNACKS (44%, x2.5), CHOCOLATE (37%, x2.5), SALTY SNACKS (50%, x2.4) |
| 2 | Tobacco + Beer + Carbonated Drinks | no | 29.34% | TOBACCO (58%, x2.5), BEER (46%, x2.4), CARBONATED DRINKS (40%, x2.1) |

## Limitations

- **Stage 1's gap statistic doesn't support the 3-band structure on its own** (see above) - the concrete/relatively-concrete/abstract split is retained as a theoretical assumption from the paper, not something this data's variety distribution independently confirms.
- **Category granularity differs from the paper's own taxonomy** (paper: 75 derived categories; this export: 99, after excluding non-product categories) - a property of each retailer's product hierarchy, not a modeling choice, but it does mean category-level results aren't numerically comparable to the paper's reported figures.
- **Store sizes vary substantially** (66,847 to 587,487 total baskets across the 6 stores), which affects the statistical power and reliability of gap statistic and cluster stability at the smaller stores - treat results from Brasov (smallest) with more caution than Neamt (largest).
- **This reproduction is per-store**, matching the paper's own framing, but does not further control for factors the paper's single pilot store didn't need to (seasonality pooled across the full year, regional/demographic differences between counties, store-format-specific assortment differences) - these are folded into one full-year fit per store, not modeled separately.
- **Multiple initializations only address one source of instability.** This project's stability investigation (`docs/stability/*.md`) found CLARA's single-*sample* variance (which baskets get drawn into the training CLARA sample) is at least as large a factor as single-*initialization* variance (which local optimum a given sample converges to) - this run fixes the latter, not the former. A full multi-sample CLARA (repeat the whole draw-sample-then-fit procedure, not just the k-medoids restart) was not implemented here.

## Modifications, labeled separately from the paper's unweighted baseline

None of the following are part of this report's numbers - they are separate, already-published explorations elsewhere in this project, listed here only so they aren't confused with the baseline above:
- **Value-weighted graph community detection** (spend-distribution based, no basket sampling) - this project's best-performing method by stability, but a deliberate deviation from the paper's unweighted method on every axis (what's clustered, algorithm, k-selection). Applied to this same 2025 per-store data separately - see `docs/graph_missions/level3_value_distribution_graph_2025_summary.md`.
- **Item2Vec-style category embeddings** - a further deviation layered on the graph method above, not applied to the 2025 data as of this report.

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`