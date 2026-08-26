"""Shared constants for the shopping-missions pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRANSACTION_LINES_DIR = DATA_DIR / "transaction_lines"
PRODUCTS_DIR = DATA_DIR / "products"
LOCATIONS_FILE = DATA_DIR / "locations-export (3).csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCENARIOS_DIR = OUTPUTS_DIR / "scenarios"

# Location-similarity / catchment pipeline (see catchment.py,
# location_similarity.py). LOCATION_FEATURES_DIR is where every pulled
# national/per-store data file lives (INS, Eurostat, WorldPop, OSM,
# census2021, and the store_feature_table_v*.json versions).
LOCATION_FEATURES_DIR = DATA_DIR / "location_features"

# The catchment radius used throughout this project's real-data pulls and
# validations (v5-v8, the candidate-site scoring scripts) - was duplicated
# as a literal `radius_km=1.0` across ~4 scripts before being centralized
# here. Not a default baked into `catchment.catchment_profile` itself
# (that function keeps radius_km required/explicit by design) - callers
# import this constant instead of repeating the literal.
CATCHMENT_RADIUS_KM = 1.0

# Catalog IDs excluded from basket-affinity clustering - transactional
# artifacts, not shopper-chosen product categories:
#   500 SERVICES            - bottle-deposit returns, bag fees, etc.
#   900 CAMPANII MARKETING  - marketing/promo campaign line items
#   813 ALTELE               - Romanian "OTHERS" catch-all (local-producer taxonomy)
#   828 ALTELE               - duplicate "OTHERS" catch-all under a different ID
EXCLUDE_CATALOG_IDS = {500, 900, 813, 828}

# HOUSEHOLD PRODUCTS (404) was tried as an addition to this list on
# 2026-07-21, on the theory that its 36.6% overall prevalence and near-
# uniform presence across clusters (lift capped ~1.9) was diluting mission
# definitions. Tested empirically across all 6 tracked scenarios: it helped
# the 3-day-window scenarios (silhouette@k=6 +13-29%) but hurt the 8-day
# ones (silhouette -12-27%, and 8day's degenerate-cluster count went 0->2) -
# net negative for the leading scenario, so reverted same day. Left as a
# documented, already-tested dead end rather than silently forgotten - don't
# re-try this exact change without new evidence.

# Stage 1 (variety) segment boundaries, inclusive, in number of distinct
# categories per basket. Derived empirically from a 1-D k-means (k=3) fit
# on 2026-05-09 offline baskets - see variety_stage.fit_variety_segments
# for the automatic version; these are the fallback/default values.
DEFAULT_VARIETY_BOUNDS = {
    "concrete": (1, 3),
    "relatively_concrete": (4, 7),
    "abstract": (8, None),  # None = unbounded
}

# Stage 2 (mission) defaults
DEFAULT_K = 6
DEFAULT_SAMPLE_SIZE = 6000
RANDOM_STATE = 0

# --- Configurable feature toggles (scenarios.py composes these) ---

# Category granularity: which level of the products/ article_level_0X tree to
# cluster on. 3 = our original vocabulary (Catalog Name/ID, ~99-103 categories,
# what transaction_lines ships natively). 4/5 require joining transaction_lines
# to products/ on SKU ID (verified: 100% join coverage, 100% valid level-4/5
# data for joined offline lines, and level-4/5 codes are globally unique - they
# encode their full ancestry, e.g. level-4 "20402" always rolls up to level-3
# "204", never to a different parent).
CATEGORY_LEVEL_CHOICES = (3, 4, 5)
DEFAULT_CATEGORY_LEVEL = 3

# Minimum basket variety to include in Stage 2 clustering. Settled at 2
# (exclude single-category baskets only) after empirically confirming that
# including them degenerates several clusters into trivial single-category
# splits - see docs/table6-comparison-all-baskets.md. Do not lower this to 1
# in a scenario; it has already been tested and ruled out.
MIN_VARIETY = 2

# Day-part buckets, derived from Timestamp's hour component. Boundaries follow
# the actual observed hourly basket-volume curve (peaks ~9-13h and ~17-20h,
# trough overnight) rather than arbitrary round numbers - see the 2026-07-20
# hourly-volume analysis. (lo, hi) inclusive.
DAYPART_BOUNDS = [
    (0, 5, "night"),
    (6, 8, "early_morning"),
    (9, 11, "morning"),
    (12, 16, "afternoon"),
    (17, 20, "evening"),
    (21, 23, "late_evening"),
]
