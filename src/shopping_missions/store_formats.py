"""Real store-format labels, from the Profi locations export.

Replaces the basket-volume tertile proxy used in `validation.py` (documented
as a stand-in in docs/mvp-plan.md Sec. 2.1) now that a genuine format label
is available: `data/locations-export (3).csv` carries `store_format` per
store, keyed by `location_external_id` - confirmed to match `Store ID` in
the transaction-line exports for all 1,740 stores seen in the 3-day window
(zero unmatched).
"""

from pathlib import Path

import pandas as pd

from .config import LOCATIONS_FILE

# Formats with a meaningful sample in the 3-day window (Specialty Store: 2,854
# baskets, Unknown/blank: 5,244 - both excluded from per-format analyses as
# too small/not a real retail segment, kept as "Other" if ever surfaced).
MAIN_FORMATS = ["Supermarket", "Mini Market", "Convenience Store"]


def load_store_formats(path: Path = LOCATIONS_FILE) -> pd.DataFrame:
    """store_id (str) -> store_format + a few descriptive fields.

    2 of 1,839 rows (store_id 3891, 3892) are duplicated: an older record
    with blank county/stale format alongside a newer, more complete one
    (different `location_id`, same `location_external_id`/store_id). Kept
    naively, this fans out any join keyed on store_id. Dedup by preferring
    the record with a populated county - the more complete/current one.
    """
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns={"location_external_id": "store_id"})
    has_county = df["location_county"].fillna("").str.strip() != ""
    df = df.loc[has_county.sort_values(ascending=False).index].drop_duplicates(subset="store_id", keep="first")
    df["store_format"] = df["store_format"].replace("", "Unknown").fillna("Unknown")
    df["location_county"] = df["location_county"].replace("", "Unknown").fillna("Unknown")
    return df[["store_id", "store_format", "location_name", "location_county", "location_city"]]


def join_store_format(baskets: pd.DataFrame, path: Path = LOCATIONS_FILE) -> pd.DataFrame:
    """Left-joins `store_format` onto a baskets/fact-table DataFrame with a
    `store_id` column. Unmatched stores get 'Unknown' rather than being
    silently dropped."""
    formats = load_store_formats(path)
    out = baskets.copy()
    out["store_id"] = out["store_id"].astype(str)
    out = out.merge(formats[["store_id", "store_format"]], on="store_id", how="left")
    out["store_format"] = out["store_format"].fillna("Unknown")
    return out


def join_county(baskets: pd.DataFrame, path: Path = LOCATIONS_FILE) -> pd.DataFrame:
    """Left-joins `county` onto a baskets/fact-table DataFrame with a
    `store_id` column. Independent of `join_store_format` so county can be
    toggled on/off without pulling in format, and vice versa - 43 distinct
    counties, 1.3% of stores unmatched/blank (-> 'Unknown')."""
    formats = load_store_formats(path)
    out = baskets.copy()
    out["store_id"] = out["store_id"].astype(str)
    out = out.merge(
        formats[["store_id", "location_county"]].rename(columns={"location_county": "county"}),
        on="store_id", how="left",
    )
    out["county"] = out["county"].fillna("Unknown")
    return out
