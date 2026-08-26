"""Day-part bucketing, derived from a basket's Timestamp.

Timestamp is transaction-level (constant across every line item in one
basket - confirmed in the raw data), so bucketing happens once per basket,
not per line. Like `store_formats.join_store_format`/`join_county`, this is
a stratification/validation dimension - used to cross-tab against mission
labels (does a "breakfast" cluster actually skew morning?), not fed into the
Jaccard clustering matrix itself.
"""

import pandas as pd

from .config import DAYPART_BOUNDS


def add_daypart(baskets: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Adds a `day_part` column bucketed from `timestamp_col`'s hour.
    Requires baskets to have been built with `include_timestamp=True`
    (`baskets.build_basket_data`) from a `lines` table that itself was
    ingested with `include_daypart=True` (`ingest.connect_lines`)."""
    if timestamp_col not in baskets.columns:
        raise ValueError(
            f"'{timestamp_col}' column not found - build basket data with "
            f"include_timestamp=True (and ingest with include_daypart=True) first"
        )
    out = baskets.copy()

    def bucket(t):
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return "Unknown"
        hour = t.hour
        for lo, hi, name in DAYPART_BOUNDS:
            if lo <= hour <= hi:
                return name
        return "Unknown"

    out["day_part"] = out[timestamp_col].apply(bucket)
    return out
