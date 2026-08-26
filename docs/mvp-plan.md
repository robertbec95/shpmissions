# Shopping Missions MVP — Build Plan

> **Status: implemented.** See `src/shopping_missions/` for the pipeline and `outputs/` for a run on 2026-05-09. Run with `PYTHONPATH=src python3 -m shopping_missions.pipeline --dates 20260509[,20260510,...]` from a venv with `requirements.txt` installed. Key results and deviations from this plan are noted inline below.

Based on Sarantopoulos et al. (2015/2016), *"Analysing Shopping Trip Purpose..."*, Journal of Business Research. Goal: reimplement the paper's basket-variety + Jaccard/k-centroid two-stage clustering method on our own POS data (Profi Romania) to discover recurring "shopping missions" without using loyalty/survey data.

## 1. What data we actually have

Located in `data/`, three days of exports (`20260509`, `20260510`, `20260511`):

| Folder | File(s) | Grain | Size |
|---|---|---|---|
| `transaction_lines/` | `retail_media_sales_data_YYYYMMDD.txt` | one row per **line item** | ~1.3GB/day, ~4.9M rows/day, ~1.19M baskets/day |
| `products/` | `retail_media_products_YYYYMMDD.txt` | one row per SKU, with 8-level category hierarchy | ~65MB/day (mostly static across days) |
| `stores/` | `retail_media_store_details_YYYYMMDD.txt` | one row per store, opening hours only | ~79KB/day |
| `loyalty_customers/` | `retail_media_loyalty_members_YYYYMMDD.txt` | one row per loyalty member | ~250KB/day |

### Key fields in `transaction_lines` (tab-separated)

`Transaction ID, Store Name, Store ID, Site Address, Sales Channel, Client ID, Date, Timestamp, Brand, Brand ID, Product Name, Product Id, SKU ID, SKU Name, Sales Barcode, Catalog Name, Catalog ID, Quantity, Price, Discount, Payment mode, GROSS_RETAIL_AMT, GROSS_TRANS_AMT, NET_RETAIL_AMT, NET_TRANS_AMT`

Mapping to the paper's requirements:

| Paper concept | Our field | Notes |
|---|---|---|
| Basket ID | `Transaction ID` | one basket = one Transaction ID at one Store ID |
| Product category | `Catalog Name` / `Catalog ID` | **102 distinct catalogs** in a single day — close to the paper's 75, usable directly, or can be re-rolled up via `products/` hierarchy (see §4) |
| Basket value | `NET_TRANS_AMT` / `GROSS_TRANS_AMT` | for commercial-share reporting (paper's ~78%/50% figures) |
| Date/time | `Date`, `Timestamp` | for optional daypart segmentation (breakfast/dinner sanity checks) |
| Store | `Store ID`, `Store Name` | **1,740 distinct stores** in one day |
| Store format | `store_format` in `data/locations-export (3).csv`, joined on `Store ID` / `location_external_id` | **resolved** — Supermarket / Mini Market / Convenience Store / a handful of Specialty Store & blank; matches all 1,740 stores seen in the transaction data. `store_formats.py` |
| Channel | `Sales Channel` | `Offline` / `Online` — online baskets likely behave differently (planned, high-variety); consider excluding or analyzing separately |

`products/retail_media_products_*.txt` gives a full 8-level category tree (`article_level_01` … `article_level_08`, RO + EN names) per SKU — useful if `Catalog Name` (102 buckets) turns out too coarse or too noisy, since it lets us pick a specific hierarchy level to match the paper's ~75-category granularity.

`stores/` only has opening hours, not format/size — but a separate export, `data/locations-export (3).csv`, does carry a real `store_format` label (Supermarket / Mini Market / Convenience Store, plus a few Specialty Store and blank), joinable on `Store ID`. See `store_formats.py` and `docs/k-selection-by-format.md`.

`loyalty_customers/` is **not used** by the modeling pipeline (the whole point of the method is to avoid needing it), but `Client ID` on transaction lines could later support an optional validation cross-check (e.g., "do frequent loyalty shoppers show different mission mixes?") — out of scope for the MVP.

## 2. Open decisions before coding (data gaps)

1. ~~**Store format.**~~ **Resolved** — `data/locations-export (3).csv` carries a real `store_format` (Supermarket / Mini Market / Convenience Store) for all 1,740 stores in the transaction data, joined via `store_formats.py`. `validation.py`'s basket-volume tertile proxy is now superseded for anything that needs real format; re-running Stage 1/2 gap statistic stratified by this label (`docs/k-selection-by-format.md`) shows it explains only a small part of the earlier weak-structure finding, not most of it — see that doc for the update to the pooling hypothesis.
2. **Category granularity.** Use `Catalog Name` (102 categories) as-is for the MVP, or roll up/down via `article_level_0X` in `products/`. Recommendation: start with `Catalog Name` since it's already close to the paper's scale and requires no join.
3. **Time window.** Only 3 days of data available now. The paper uses much longer windows. For the MVP, 3 days is enough to validate the pipeline mechanically (does it run, do clusters look sane) but not enough to claim stable "missions" — flag this explicitly in any output.
4. **Online transactions.** Decide whether to include `Sales Channel = Online` baskets or restrict to `Offline` to match the paper's brick-and-mortar setting. Recommendation: restrict to `Offline` for the MVP, revisit later.

## 3. Pipeline overview

```
raw transaction_lines (line items)
        │
        ▼
[Step A] Ingest & clean
        │
        ▼
[Step B] Define category vocabulary
        │
        ▼
[Step C] Build basket × category matrix (binary)
        │
        ▼
[Step D] Stage 1 — variety clustering (concrete / relatively concrete / abstract)
        │            (keep only "relatively concrete" baskets)
        ▼
[Step E] Stage 2 — Jaccard distance + k-centroid clustering → missions
        │
        ▼
[Step F] Label & profile each mission cluster
        │
        ▼
[Step G] Validate (per store, per format if available, commercial share)
        │
        ▼
[Step H] Output artifacts
```

### Step A — Ingest & clean

- Read `transaction_lines` in chunks (files are ~1.3GB — do not load a full day into memory as a single `pd.read_csv` without `usecols`/`dtype`/chunking; use `chunksize` or a columnar engine like `pyarrow`/`duckdb` to keep the MVP responsive on a laptop).
- Load only the columns needed: `Transaction ID, Store ID, Catalog Name, Quantity, NET_TRANS_AMT, Sales Channel, Date, Timestamp`.
- Filter `Sales Channel == 'Offline'` (per decision in §2.4).
- Drop non-product lines if any show up as noise categories (e.g., `SERVICES` for bottle deposits/`GARANTIE AMB SGR`, `CAMPANII MARKETING`) — these aren't "categories" a shopper chose, they're transactional artifacts, and including them will pollute the co-occurrence signal. Build an explicit exclude-list, reviewed by eye against the 102 catalog names.
- Deduplicate the basket key as `(Transaction ID, Store ID, Date)` — Transaction ID may not be globally unique across stores/days.

### Step B — Define category vocabulary

- Take the cleaned, deduplicated set of `Catalog Name` values (~90-100 after excluding non-product noise categories from Step A).
- Produce a fixed, ordered category list — this becomes the column space for the binary basket vectors in Step C. Persist it (e.g., `category_vocab.json`) so every run uses the same dimensionality.

### Step C — Build basket × category matrix

- Group line items by basket key from Step A.
- For each basket, compute:
  - `variety` = count of distinct categories present.
  - a binary vector over the Step B vocabulary (1 if category present in basket, 0 otherwise) — quantity/spend is intentionally discarded here, matching the paper's binary-flag approach.
- Store as a sparse matrix (scipy `csr_matrix`) keyed by basket ID — with ~1.19M baskets/day × ~100 categories, a dense matrix is wasteful; sparse keeps this tractable.

### Step D — Stage 1: variety clustering

- Cluster baskets on the single `variety` feature into 3 groups (concrete / relatively concrete / abstract). Two viable approaches for the MVP:
  - 1-D k-means (`k=3`) on `variety`, or
  - fixed empirical thresholds derived by inspecting the variety histogram (e.g., look for the natural bimodal/trimodal breakpoints, similar to how the paper describes its regimes) — simpler and more interpretable for an MVP, revisit with k-means once real thresholds are validated.
- Keep only the "relatively concrete" segment for Stage 2. Log the size and commercial value (sum of `NET_TRANS_AMT`) of each of the 3 segments — this reproduces the paper's "X% of value comes from relatively concrete trips" statistic.

### Step E — Stage 2: Jaccard distance + k-centroid clustering

- Restrict to relatively-concrete baskets' binary vectors from Step C.
- Compute pairwise Jaccard distance. At basket-population scale (likely hundreds of thousands of baskets even after filtering), a full O(n²) pairwise distance matrix is infeasible — for the MVP:
  - Either subsample to a manageable number of baskets (e.g., 20k-50k) per run for prototyping, or
  - Cluster on **unique basket signatures** instead of raw baskets (many baskets share the exact same category combination once quantity is discarded) and weight each signature by its frequency — this is both faster and closer to what actually drives the paper's category-affinity signal.
- Run k-centroid (k-medoids) clustering (e.g., `sklearn_extra.cluster.KMedoids`, or a manual PAM/CLARA implementation if that dependency is unavailable) with a chosen `k` (start with `k=6` to mirror the paper, but also sweep `k=4..10` and compare via silhouette score on the Jaccard distance matrix to justify the choice rather than assuming 6 will replicate on this retailer/category set).

**Implementation note:** unique-signature dedup was tested against real data and rejected — at 99 categories, 275,629 relatively-concrete baskets only collapse to 225,301 unique signatures (18% reduction), nowhere near enough. Implemented CLARA instead (`mission_stage.py`): fit k-medoids (custom PAM, no `sklearn_extra` dependency) on a random sample of 6,000 baskets via a vectorized dense Jaccard distance matrix, then assign *all* relatively-concrete baskets to their nearest sampled medoid via one vectorized sparse-dense matmul (O(n·k), handles millions of baskets in seconds). Kept `k=6` as the default to mirror the paper's headline six-mission finding.

**Update:** the paper actually picks k via the **gap statistic** (Tibshirani, Walther & Hastie, 2001), not silhouette. Both are now implemented (`cluster_selection.gap_statistic`, `variety_stage.select_k`, `mission_stage.select_k`) and run on the real data in `docs/k-selection.md` — short version: the gap statistic disagrees with k=6, preferring k=2 for missions and k=1 for the variety stage (i.e., it finds materially weaker cluster structure in this pooled, 3-day, all-format dataset than the paper found in its single-store, 17-month one). `k=6` is kept as a deliberate, documented business/interpretability choice, not a statistically-derived one — see `docs/k-selection.md` for the full reasoning and the candidate explanations for the gap.

### Step F — Label & profile each mission cluster

- For each cluster, compute category "lift" or simple frequency-within-cluster vs. frequency-overall to identify the categories that define it (e.g., a cluster dominated by `BREAKFAST PRODUCTS`, `CEREALS`, `BUTTER / MARGARINE` → "breakfast" mission).
- Manually assign a human-readable label per cluster based on top categories (mirrors the paper's breakfast/meal/snack/cleaning/coffee/light-meal naming).
- Identify "bridge" categories — those with meaningfully high presence across multiple clusters (paper's milk example) — as a byproduct for cross-sell analysis.

**Implementation note:** the first pass surfaced a real data quirk worth flagging — `HOUSEHOLD PRODUCTS` (catalog 404, which includes biodegradable produce bags) showed up as a top category in almost every cluster with support 0.45–0.74 but lift close to 1 (0.78–1.3): it's just ubiquitous whenever fresh produce is bought, not mission-defining. Added a `MIN_LIFT = 1.2` filter alongside the support filter in `profiling.py` so ambient/bundled items like this don't crowd out the categories that actually distinguish a mission. The auto-label heuristic (`LABEL_KEYWORDS` in `profiling.py`) is a starting point, not a final labeling — confidences of 0.25–0.5 on several clusters (see `outputs/summary.md`) mean a human should relabel using the full `top_categories` list, not just trust the string.

### Step G — Validate

- Recompute Steps D-F split by `Store ID` (or store-size proxy from §2.1 once defined) to check whether the mission mix varies sensibly by store type, and whether the same ~6 clusters re-emerge when run on different store subsets (paper's pilot/control replication check).
- Report, per store group: % of baskets and % of basket value that fall into the "relatively concrete" segment, and the share belonging to each mission.

**Implementation note:** implemented with the store-size tertile proxy from §2.1 (`validation.py`, bucketed by daily basket volume per store). On 2026-05-09, mission mix was strikingly stable across small/medium/large tiers (e.g. the largest mission cluster sat at 38.9%/40.4%/40.4% of relatively-concrete baskets across the three tiers). That's a good sign the clusters reflect a real population-level pattern rather than store-specific noise — but it also suggests the volume-based size proxy isn't capturing the same axis of variation a true convenience/supermarket/hypermarket format label would (the paper's format comparison found real mix differences, e.g. convenience skewing snack/breakfast). Getting the real format label remains the highest-value data addition.

### Step H — Output artifacts

- `basket_missions.parquet`: basket ID → variety segment, mission cluster ID/label.
- `mission_profiles.json`: per-mission top categories, size, avg basket value.
- `category_bridges.json`: categories with cross-mission presence.
- A short markdown/notebook summary reproducing the paper's key charts (mission mix by store, commercial share by segment).

## 4. Proposed project structure

```
shpmissions/
├── data/                     # existing raw exports (untouched)
├── docs/
│   └── mvp-plan.md           # this file
├── src/
│   └── shopping_missions/
│       ├── ingest.py         # Step A
│       ├── categories.py     # Step B
│       ├── baskets.py        # Step C
│       ├── variety_stage.py  # Step D
│       ├── mission_stage.py  # Step E
│       ├── profiling.py      # Step F
│       ├── validation.py     # Step G
│       └── pipeline.py       # orchestrates A→H, CLI entry point
├── notebooks/
│   └── exploration.ipynb     # ad hoc EDA, cluster-count sweeps, sanity plots
├── outputs/                  # generated artifacts from Step H (gitignored)
└── pyproject.toml / requirements.txt
```

## 5. Tech stack

- **Python** with `pandas` for orchestration, `pyarrow` or `duckdb` for efficient reading of the large daily text files, `scipy.sparse` for the basket×category matrix, `scikit-learn` (k-means for Stage 1) and `sklearn_extra` (k-medoids for Stage 2) or a lightweight custom PAM implementation if that package proves awkward to install.
- No loyalty/survey inputs used in the modeling path, consistent with the paper's premise.

## 6. Suggested build order

1. Steps A-C on a single day (`20260509`), offline channel only — get a clean basket × category sparse matrix and a variety histogram.
2. Step D — pick and justify the variety thresholds/clusters, report segment sizes and commercial share.
3. Step E on a subsample or on unique basket signatures — get a first pass at k=6 clusters, sanity-check cluster sizes aren't wildly imbalanced.
4. Step F — label clusters by eye, confirm they resemble sensible missions (not guaranteed on the first pass; category exclude-list from Step A will likely need iteration here).
5. Step G — extend to all 3 available days and compare across stores, once a store-format or store-size proxy exists.
6. Step H — package outputs, write the summary.
