"""Named, trackable clustering scenarios - combinations of the configurable
pipeline features (category level, county, day-part) - so results are
comparable across runs instead of one-off scripts each time.

Every scenario builds on settled findings from this project's prior analysis,
not re-litigated per run:
  - min_variety=2 always. Single-category baskets are excluded from Stage 2
    - empirically confirmed necessary, not just theoretically motivated: with
    them included, 3 of 6 clusters degenerated into trivial single-category
    splits (TOBACCO/BEER/SUPPLIED BAKERY at ~100% support each) instead of
    real multi-category missions. See docs/table6-comparison-all-baskets.md.
  - Stage 1's relatively_concrete band is NOT used as a Stage-2 input filter.
    Bypassing it (variety>=2, no upper bound) produced better-separated
    clusters than restricting to Stage 1's k-means-selected middle band -
    see docs/table6-comparison-variety-ge2.md (gap_k moved from 2-3 to 4,
    silhouette from ~0.03-0.05 to ~0.05-0.07).
  - Negative-quantity/return baskets always excluded (ingest.py default).
  - Offline channel only.

Add new scenarios to SCENARIOS below; `scripts/run_scenarios.py` runs
whichever are named on its command line (or all of them) and appends one row
per scenario to outputs/scenarios/scenario_results.csv, so results accumulate
and stay comparable across separate invocations rather than overwriting.
"""

from dataclasses import dataclass

from .config import EXCLUDE_CATALOG_IDS


@dataclass
class Scenario:
    name: str
    category_level: int = 3          # 3, 4, or 5 - see config.CATEGORY_LEVEL_CHOICES
    dates: tuple = ("20260509", "20260510", "20260511")
    min_variety: int = 2             # do not lower - see module docstring
    k: int = 6
    sample_size: int = 6000
    include_county: bool = True      # adds a per-cluster county cross-tab (validation only, not a clustering feature)
    include_daypart: bool = True     # adds a per-cluster day-part cross-tab (validation only, not a clustering feature)
    run_k_selection: bool = True     # gap statistic + silhouette + elbow diagnostic for this scenario's k
    exclude_category_ids: frozenset | None = None  # None = config.EXCLUDE_CATALOG_IDS default; set to add scenario-specific exclusions (e.g. testing HOUSEHOLD PRODUCTS removal) without touching the shared default
    notes: str = ""


DATES_3DAY = ("20260509", "20260510", "20260511")
DATES_8DAY = ("20260504", "20260505", "20260506", "20260507", "20260508", "20260509", "20260510", "20260511")

SCENARIOS: list[Scenario] = [
    Scenario(
        name="level3_baseline",
        category_level=3,
        dates=DATES_3DAY,
        notes=(
            "Anchor/reproduction of docs/table6-comparison-variety-ge2.md - the best-known "
            "result prior to this scenario framework - re-run through the configurable "
            "pipeline so it's directly comparable to level4/level5 on identical machinery."
        ),
    ),
    Scenario(
        name="level4",
        category_level=4,
        dates=DATES_3DAY,
        notes=(
            "Finer category granularity (422 categories possible, ~351 seen on a single "
            "day, vs. 99-103 at level 3). Tests whether splitting catch-all categories like "
            "BREAKFAST PRODUCTS into JAM/COMFITURE/SPREADABLE CREAM/HONEY sharpens cluster "
            "definitions rather than just adding noise."
        ),
    ),
    Scenario(
        name="level5",
        category_level=5,
        dates=DATES_3DAY,
        notes=(
            "Even finer granularity (~1,210 categories possible). Tests whether this is too "
            "sparse for basket-level Jaccard co-occurrence to find anything - each basket "
            "touching only a handful of very specific categories, diluting overlap between "
            "baskets that would agree at a coarser level."
        ),
    ),
    Scenario(
        name="level3_8day",
        category_level=3,
        dates=DATES_8DAY,
        notes=(
            "Same config as level3_baseline (the winner among the 3-day scenarios), extended "
            "to 8 days (2026-05-04 through 2026-05-11, up from 3) once more data became "
            "available. Directly tests the leading remaining explanation for weak k-selection "
            "structure flagged throughout docs/k-selection*.md: too short a time window. SKU/"
            "store join coverage verified at 100% for all 5 new days against the existing "
            "reference products/locations files before running."
        ),
    ),
    Scenario(
        name="level4_8day",
        category_level=4,
        dates=DATES_8DAY,
        notes="Level 4 re-tested on the 8-day window, for the same reason as level3_8day - "
              "more data may change which granularity wins, not just whether k=6 is well-supported.",
    ),
    Scenario(
        name="level5_8day",
        category_level=5,
        dates=DATES_8DAY,
        notes="Level 5 re-tested on the 8-day window, completing the granularity x window-length comparison.",
    ),
    Scenario(
        name="level3_baseline_gapk",
        category_level=3,
        dates=DATES_3DAY,
        k=4,
        notes=(
            "FROZEN 2026-07-21 - this is the reference baseline scenario. Do not change its "
            "config (category_level, dates, min_variety, k, exclude_category_ids) without an "
            "explicit new instruction; new ideas get their own scenario that is compared against "
            "this one, not a mutation of it. Paper-faithful Stage 2 (Jaccard k-medoids) on baskets "
            "with variety>=2, category level 3, 3-day window - with k chosen by the gap statistic "
            "(the paper's own k-selection method, Tibshirani/Walther/Hastie 2001) rather than a "
            "fixed k=6 copied from the paper's pilot store. On the current population (post "
            "HOUSEHOLD PRODUCTS revert) gap_k=4, with elbow_k=7 and silhouette_k=8 as comparison/ "
            "support diagnostics, not decision criteria - a previous run of this scenario used a "
            "stale k=3 computed before that revert. Report: "
            "https://claude.ai/code/artifact/415144c7-b2b1-4095-b4b8-d53a90e6975e"
        ),
    ),
    Scenario(
        name="level3_baseline_gapk_no_household",
        category_level=3,
        dates=DATES_3DAY,
        k=3,
        exclude_category_ids=EXCLUDE_CATALOG_IDS | {404},
        notes=(
            "level3_baseline_gapk with HOUSEHOLD PRODUCTS (404) additionally excluded from "
            "clustering - same population/method otherwise. This re-tests the 2026-07-21 "
            "HOUSEHOLD PRODUCTS exclusion (previously reverted as net-negative against a fixed "
            "k=6) under the new gap-statistic-selected-k baseline, which is a materially "
            "different comparison point. Excluding it drops variety>=2 baskets from 2,364,042 to "
            "2,257,930 (baskets whose only 2nd category was HOUSEHOLD PRODUCTS fall below the "
            "variety filter). gap_k=3 on this population (elbow_k=7, silhouette_k=6 as "
            "comparison), vs. gap_k=4 for the baseline - not a like-for-like k, by design, since "
            "each scenario's k is chosen independently by the same gap-statistic rule."
        ),
    ),
    Scenario(
        name="level3_8day_gapk",
        category_level=3,
        dates=DATES_8DAY,
        k=2,
        notes=(
            "level3_8day re-clustered at k=2 - its own gap-statistic-selected k. Same purpose as "
            "level3_baseline_gapk: test the data-suggested k directly rather than assume k=6 is "
            "right just because the paper used it."
        ),
    ),
]


def get_scenario(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"no scenario named {name!r} - available: {[s.name for s in SCENARIOS]}")
