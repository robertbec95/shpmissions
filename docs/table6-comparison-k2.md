# Table 6 analog — shopping-mission cluster solution (our data)

Reproduces the structure of **Table 6** in Sarantopoulos et al. (2015) — the final cluster solution for relatively-concrete baskets, with basket descriptives and category participation probabilities — computed on our own dataset instead of their supermarket-pilot store.

**Scope:** 20260509, 20260510, 20260511 · offline channel · all 3,677,678 baskets pooled across all stores (no store-format label is available in this dataset — see `docs/mvp-plan.md` §2 — so this is the whole-chain analog of one of the paper's per-store tables, not a single supermarket). k=2.

## Basket descriptives

| | All | Cluster 1 (breakfast) | Cluster 2 (snack) |
|---|---|---|---|
| Average basket variety (categories) | 3.99 | 4.04 | 3.90 |
| Average basket size (items) | 6.57 | 6.62 | 6.46 |
| Average basket value (RON) | 40.21 | 37.15 | 45.67 |
| Total share of baskets (%) | 100.0% | 64.1% | 35.9% |
| Number of baskets | 1,222,439 | 783,979 | 438,460 |

## Category participation probabilities (top 25 categories)

`All (baskets)` is the raw count of relatively-concrete baskets containing the category (matches the paper's format); cluster columns are the % of that cluster's baskets containing the category (participation probability).

| Category | All (baskets) | Cluster 1 (breakfast) | Cluster 2 (snack) |
|---|---|---|---|
| HOUSEHOLD PRODUCTS | 525,676 | 51.6% | 27.7% |
| FRUITS | 273,005 | 33.2% | 3.0% |
| SUPPLIED BAKERY | 272,357 | 31.9% | 5.1% |
| VEGETABLES | 271,502 | 33.2% | 2.6% |
| CARBONATED DRINKS | 246,459 | 6.7% | 44.2% |
| SALTY SNACKS | 227,148 | 6.9% | 39.4% |
| WATER | 224,086 | 16.8% | 21.0% |
| TOBACCO | 209,839 | 6.3% | 36.6% |
| SWEET SNACKS | 199,060 | 14.1% | 20.2% |
| BEER | 193,803 | 6.8% | 32.0% |
| MILK / CREAM | 145,943 | 14.3% | 7.8% |
| YOGURT  / SANA / KEFIR, DESSERT | 139,441 | 13.1% | 8.4% |
| CHOCOLATE | 118,490 | 8.3% | 12.3% |
| PACKED SAUSAGES | 102,113 | 10.1% | 5.2% |
| ICE CREAM AND DESSERTS | 99,228 | 7.4% | 9.4% |
| CHEESE PACKED | 92,893 | 9.4% | 4.3% |
| BAKE OFF | 88,755 | 7.8% | 6.2% |
| DESSERT DRINKS | 77,121 | 6.6% | 5.8% |
| CHICKEN MEAT | 76,464 | 8.0% | 3.1% |
| OTHER DRINKS | 69,994 | 3.8% | 9.1% |
| NONCARBONATED DRINKS | 67,154 | 4.3% | 7.6% |
| RICE / BEANS / PASTA | 64,737 | 6.7% | 2.8% |
| COFFEE CORNER | 63,179 | 4.3% | 6.8% |
| SUGAR CONFECTIONERY | 53,919 | 3.5% | 6.1% |
| CHEWING GUM | 51,328 | 3.5% | 5.5% |

## Reading this result

This is the k the gap statistic and silhouette both actually prefer for this data (see `docs/k-selection.md`) — as opposed to the k=6 in `docs/table6-comparison.md`, which was a deliberate choice to mirror the paper's headline finding rather than a statistically-derived one.

The two clusters are cleanly separated and, unlike several of the k=6 clusters, don't need a confidence caveat on the label — the split is stark enough to name by eye:

- **Cluster 1 (64.1% of relatively-concrete baskets)** — dominated by `FRUITS` (33.2%), `VEGETABLES` (33.2%), `SUPPLIED BAKERY` (31.9%), `HOUSEHOLD PRODUCTS` (51.6%, produce bags). This is a **fresh-food / everyday-basics run**, not specifically "breakfast" — the auto-labeler's "breakfast" tag here is a miss (its keyword match fired on `SUPPLIED BAKERY`/`FRUITS` without the milk/cereal/eggs signal that would make "breakfast" a good name). "Fresh basics" is the more honest label.
- **Cluster 2 (35.9% of relatively-concrete baskets)** — dominated by `CARBONATED DRINKS` (44.2%), `TOBACCO` (36.6%), `SALTY SNACKS` (39.4%), `BEER` (32.0%). This is a **convenience / vice run** (tobacco+beer+carbonated+snacks), not "snack" in the paper's sense (chocolate/confectionery/carbonated) — again the auto-label undersells what's actually there.

So at the k the data statistically supports, the story isn't "six missions" but **two dominant behavioral modes**: a fresh-food grocery trip vs. a convenience/vice trip — plausibly the strongest, most stable signal in this pooled, all-format, 3-day dataset, with the paper's finer missions (breakfast vs. meal vs. cleaning, etc.) either not present at this pooling level or too weak to survive gap-statistic scrutiny here. That's consistent with the `docs/k-selection.md` hypothesis that store-format pooling and the short time window wash out finer structure that a single-store, longer-window analysis (like the paper's) would resolve.

**Practical read:** k=6 (`docs/table6-comparison.md`) is more useful for shopper-marketing purposes if the finer clusters hold up under more data/format-segmentation — but right now that's an assumption carried over from the paper, not something this data independently confirms. k=2 is the more defensible statistical baseline today; treat the k=6 breakdown as a hypothesis to validate once store-format labels and a longer window are available, not as a confirmed replication of the paper's six missions.

Reproduce with: `PYTHONPATH=src python3 scripts/generate_table6.py --dates 20260509,20260510,20260511 --k 2 --output docs/table6-comparison-k2.md`
