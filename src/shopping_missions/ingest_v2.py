"""Step A (v2) - Ingest the new 2025 per-store parquet exports (6 Profi
stores spanning mini market / convenience store / supermarket formats), a
different data source from the original `transaction_lines/` CSV exports
`ingest.py` reads.

Same underlying Profi category taxonomy as the original pipeline (verified:
identical ~103 category names - SUPPLIED BAKERY, OTHERS FIIGO, CAMPANII
MARKETING, the same ALTELE-appears-twice quirk, etc.), but a different ID
encoding: MongoDB-style hash strings here (e.g.
`658edf01142da233f5707424`) instead of the old pipeline's small integer
Catalog IDs. Re-derived as small sequential integers (alphabetical rank of
category_name, fixed across all 6 stores so the same category always gets
the same ID regardless of which store's data is loaded) for compatibility
with the rest of this project's code, which assumes integer category IDs
throughout (`CategoryVocab.ids: list[int]` and every JSON/graph export that
casts category IDs with `int(...)`).

Each parquet file is one store's full calendar-2025 transaction history
(~120K-590K transactions per store, ~7M line items total across all 6) -
small enough to query directly with DuckDB, no need for `multiday.py`'s
incremental per-date processing (that existed to bound memory/disk for a
single chain-wide multi-day CSV union; these files are already small,
pre-partitioned per store).
"""

from pathlib import Path

import duckdb
import pandas as pd

NEW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "new data" / "OneDrive_1_8-3-2026"
EXCLUDE_CATEGORY_NAMES = {"SERVICES", "CAMPANII MARKETING", "ALTELE"}


def list_store_files() -> dict[str, dict]:
    """{location_id: {"path": Path, "name": str, "format": str, "county": str}}
    for every store parquet found, parsed from each file's own name
    (`store_id=<location_id>-<format>-<county>.parquet`)."""
    out = {}
    for path in sorted(NEW_DATA_DIR.glob("store_id=*.parquet")):
        stem = path.stem  # e.g. store_id=65587f...-Supermarket-Bucharest
        location_id = stem.split("=", 1)[1].split("-", 1)[0]
        rest = stem.split("-", 1)[1] if "-" in stem else ""
        parts = rest.rsplit("-", 1)
        store_format = parts[0] if len(parts) == 2 else rest
        county = parts[1] if len(parts) == 2 else ""
        out[location_id] = {"path": path, "format": store_format, "county": county}
    return out


def build_global_category_vocab() -> pd.DataFrame:
    """Distinct (category_id, category) pairs, fixed once across ALL 6
    stores (alphabetical-rank integer IDs) - every store's `connect_store_lines`
    call is given this same mapping, so category N always means the same
    thing regardless of which store's data is being analyzed."""
    con = duckdb.connect()
    glob = str(NEW_DATA_DIR / "*.parquet")
    df = con.sql(f"""
        SELECT DISTINCT category_name AS category
        FROM read_parquet('{glob}')
        WHERE category_name NOT IN ({",".join(f"'{n}'" for n in EXCLUDE_CATEGORY_NAMES)})
        ORDER BY category_name
    """).df()
    df["category_id"] = range(1, len(df) + 1)
    return df[["category_id", "category"]]


def connect_store_lines(
    store_path: Path,
    category_vocab: pd.DataFrame,
    exclude_category_names: set[str] = EXCLUDE_CATEGORY_NAMES,
    include_daypart: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with a materialized `lines` table for one
    store's full parquet export - same shape/grain as `ingest.connect_lines`'s
    output (one row per raw line item: basket_id, store_id, store_name,
    date, quantity, net_trans_amt, category, category_id[, timestamp]), so
    `baskets.py`/`categories.py` work completely unchanged.

    No negative-quantity rows were found in this export (unlike the old
    pipeline's raw daily files, which had a handful of return/correction
    lines) - the `quantity_unit >= 0` filter is kept anyway, for the same
    reason `ingest.connect_lines`'s `drop_negative_quantity` default is on:
    a defensive invariant, not a fix for an observed problem here."""
    con = duckdb.connect()
    con.register("category_vocab", category_vocab)
    exclude_sql = ",".join(f"'{n}'" for n in exclude_category_names) or "''"
    timestamp_sql = ", (l.date::TIMESTAMP + (l.hour_of_day * INTERVAL '1 hour')) AS timestamp" if include_daypart else ""

    con.sql(f"""
        CREATE TABLE lines AS
        SELECT
            l.transaction_id::VARCHAR AS basket_id,
            l.location_id AS store_id,
            l.location_name AS store_name,
            l.date AS date,
            l.quantity_unit AS quantity,
            l.sales_value_ct AS net_trans_amt,
            l.category_name AS category,
            v.category_id AS category_id
            {timestamp_sql}
        FROM read_parquet('{store_path}') l
        JOIN category_vocab v ON l.category_name = v.category
        WHERE l.category_name NOT IN ({exclude_sql})
          AND l.quantity_unit >= 0
    """)
    return con
