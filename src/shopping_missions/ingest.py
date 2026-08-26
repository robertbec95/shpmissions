"""Step A - Ingest & clean raw transaction-line exports.

Uses DuckDB to read the (large, tab-separated) daily export files directly
off disk and push filtering/column-selection down to the query engine, so we
never materialize a full day's raw file as a pandas DataFrame.
"""

from pathlib import Path

import duckdb

from .config import CATEGORY_LEVEL_CHOICES, EXCLUDE_CATALOG_IDS, PRODUCTS_DIR, TRANSACTION_LINES_DIR

# category_level=3 columns come straight off transaction_lines (Catalog
# Name/ID) - no join needed, this is the cheap/default path.
BASE_COLUMNS = """
    "Date" || '_' || t."Store ID"::VARCHAR || '_' || t."Transaction ID"::VARCHAR AS basket_id,
    t."Store ID" AS store_id,
    t."Store Name" AS store_name,
    t."Date" AS date,
    TRY_CAST(t."Quantity" AS DOUBLE) AS quantity,
    TRY_CAST(t."NET_TRANS_AMT" AS DOUBLE) AS net_trans_amt
"""


def resolve_paths(dates: list[str]) -> list[str]:
    """Map YYYYMMDD date strings to their transaction-line file paths."""
    paths = []
    for date in dates:
        path = TRANSACTION_LINES_DIR / f"retail_media_sales_data_{date}.txt"
        if not path.exists():
            raise FileNotFoundError(f"No transaction-line export for date {date}: {path}")
        paths.append(str(path))
    return paths


def resolve_product_reference_path(dates: list[str]) -> str:
    """The products master is near-static day to day (verified: a single
    day's file covers 100% of SKUs sold across every transaction day tested
    so far, including the 5 dates - 2026-05-04..08 - added on 2026-07-21
    that have no products/ export of their own). Use whichever requested
    date actually has a products export as the reference catalog, rather
    than unioning multiple days' product files - the latter risks a
    duplicate-SKU fan-out in the join if a product's category assignment
    ever drifted between daily exports, for no coverage benefit given the
    verified 100% overlap."""
    for date in dates:
        path = PRODUCTS_DIR / f"retail_media_products_{date}.txt"
        if path.exists():
            return str(path)
    available = sorted(p.stem.rsplit("_", 1)[-1] for p in PRODUCTS_DIR.glob("retail_media_products_*.txt"))
    raise FileNotFoundError(
        f"No products export found for any of {dates} - available dates: {available}"
    )


def connect_lines(
    dates: list[str],
    channel: str = "Offline",
    exclude_category_ids: set[int] = EXCLUDE_CATALOG_IDS,
    drop_negative_quantity: bool = True,
    category_level: int = 3,
    include_daypart: bool = False,
    product_reference_dates: list[str] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with a materialized `lines` table.

    `lines` holds one row per (basket, category-bearing line item), already
    filtered to the requested sales channel and stripped of non-product
    "noise" categories (deposits, marketing campaigns, catch-all buckets).
    Exclusion is always evaluated against the level-3 `Catalog ID` regardless
    of `category_level`, since those exclusions (deposits, marketing lines)
    are channel/transaction artifacts, not a property of category depth.

    `drop_negative_quantity` excludes negative-`Quantity` lines (returns/
    corrections - on 2026-05-09, 601 such lines across the whole file, and
    every one of them belongs to a basket where *every* line is negative -
    339 pure-return "baskets" with no genuine purchase in them, not returns
    mixed into an otherwise normal trip. Dropping the lines is therefore
    equivalent to, and implemented as, dropping those baskets outright).
    This mirrors the paper's own sales-data-preparation step (Sec. 5.1),
    which similarly cleaned the unit-quantity field before aggregating to
    the basket level.

    `category_level` selects which level of the products/ article_level_0X
    hierarchy to use as the clustering category:
      - 3 (default): `Catalog Name`/`Catalog ID` straight off transaction_lines,
        ~99-103 categories, no join needed.
      - 4 or 5: joins to the matching-date products/ export on `SKU ID`
        (verified 100% join coverage and 100% valid level-4/5 data for
        offline lines; level-4/5 codes are globally unique - they encode
        their full ancestry, so no cross-parent collisions). 422 categories
        at level 4, ~1,210 at level 5.

    `include_daypart` carries the transaction `Timestamp` through to `lines`
    (constant across all line items in one basket) for later bucketing via
    `temporal.add_daypart` - not computed here, since day-part is a basket-
    level, not line-level, concept.

    `product_reference_dates` overrides which dates are searched for a
    products/ export when resolving the level-4/5 reference catalog
    (`resolve_product_reference_path`). Needed by `multiday.py`, which calls
    this once per individual date - a single date like `20260504` has no
    products/ export of its own, so the caller passes the *full* requested
    date range here to find one that does (e.g. `20260509`), while `dates`
    itself still names only the one transaction file being read.
    """
    if category_level not in CATEGORY_LEVEL_CHOICES:
        raise ValueError(f"category_level must be one of {CATEGORY_LEVEL_CHOICES}, got {category_level}")

    tx_paths = resolve_paths(dates)
    con = duckdb.connect()
    tx_files_sql = "[" + ",".join(f"'{p}'" for p in tx_paths) + "]"
    exclude_sql = ",".join(str(c) for c in exclude_category_ids) or "-1"
    quantity_sql = 'AND TRY_CAST(t."Quantity" AS DOUBLE) >= 0' if drop_negative_quantity else ""
    timestamp_sql = ', t."Timestamp" AS timestamp' if include_daypart else ""

    if category_level == 3:
        con.sql(f"""
            CREATE TABLE lines AS
            SELECT {BASE_COLUMNS}, t."Catalog Name" AS category, t."Catalog ID" AS category_id {timestamp_sql}
            FROM read_csv({tx_files_sql}, delim='\t', header=true, quote='', union_by_name=true) t
            WHERE t."Sales Channel" = '{channel}'
              AND t."Catalog ID" NOT IN ({exclude_sql})
              {quantity_sql}
        """)
        return con

    prod_path = resolve_product_reference_path(product_reference_dates or dates)
    level_col = f"article_level_{category_level:02d}"
    level_name_col = f"{level_col}_name_en"

    con.sql(f"""
        CREATE TABLE lines AS
        SELECT {BASE_COLUMNS}, p.{level_name_col} AS category, TRY_CAST(p.{level_col} AS BIGINT) AS category_id {timestamp_sql}
        FROM read_csv({tx_files_sql}, delim='\t', header=true, quote='', union_by_name=true) t
        JOIN (
            SELECT "SKU ID"::VARCHAR AS sku_id, {level_col}, {level_name_col}
            FROM read_csv('{prod_path}', delim='\t', header=true, quote='')
            WHERE {level_col} != 'N/A'
        ) p ON t."SKU ID"::VARCHAR = p.sku_id
        WHERE t."Sales Channel" = '{channel}'
          AND t."Catalog ID" NOT IN ({exclude_sql})
          {quantity_sql}
    """)
    return con


def list_categories(
    dates: list[str],
    channel: str = "Offline",
    exclude_category_ids: set[int] = EXCLUDE_CATALOG_IDS,
    drop_negative_quantity: bool = True,
    category_level: int = 3,
):
    """Distinct (category_id, category) pairs across `dates`, via a
    streaming DISTINCT aggregation rather than materializing the full
    line-item table for every date at once.

    Needed once the requested date range is large enough that
    `connect_lines`' single `CREATE TABLE lines AS SELECT ...` over the
    union of all files stops fitting in available memory/disk (hit exactly
    this on an 8-day/~40M-row run with ~3GB free disk and 8GB RAM) - the
    fix is per-date incremental processing (see `multiday.build_basket_data_multi_date`),
    which needs a *stable* category vocabulary decided up front, before any
    per-date basket matrix is built, so every date's matrix uses the same
    column indexing. This function produces that vocabulary cheaply: the
    query still reads every requested file, but only ever materializes the
    tiny distinct-category result (order-of-100s of rows), not the full
    filtered line-item set.
    """
    if category_level not in CATEGORY_LEVEL_CHOICES:
        raise ValueError(f"category_level must be one of {CATEGORY_LEVEL_CHOICES}, got {category_level}")

    tx_paths = resolve_paths(dates)
    con = duckdb.connect()
    tx_files_sql = "[" + ",".join(f"'{p}'" for p in tx_paths) + "]"
    exclude_sql = ",".join(str(c) for c in exclude_category_ids) or "-1"
    quantity_sql = 'AND TRY_CAST(t."Quantity" AS DOUBLE) >= 0' if drop_negative_quantity else ""

    if category_level == 3:
        df = con.sql(f"""
            SELECT DISTINCT t."Catalog ID" AS category_id, t."Catalog Name" AS category
            FROM read_csv({tx_files_sql}, delim='\t', header=true, quote='', union_by_name=true) t
            WHERE t."Sales Channel" = '{channel}'
              AND t."Catalog ID" NOT IN ({exclude_sql})
              {quantity_sql}
        """).df()
        return df

    prod_path = resolve_product_reference_path(dates)
    level_col = f"article_level_{category_level:02d}"
    level_name_col = f"{level_col}_name_en"

    df = con.sql(f"""
        SELECT DISTINCT TRY_CAST(p.{level_col} AS BIGINT) AS category_id, p.{level_name_col} AS category
        FROM read_csv({tx_files_sql}, delim='\t', header=true, quote='', union_by_name=true) t
        JOIN (
            SELECT "SKU ID"::VARCHAR AS sku_id, {level_col}, {level_name_col}
            FROM read_csv('{prod_path}', delim='\t', header=true, quote='')
            WHERE {level_col} != 'N/A'
        ) p ON t."SKU ID"::VARCHAR = p.sku_id
        WHERE t."Sales Channel" = '{channel}'
          AND t."Catalog ID" NOT IN ({exclude_sql})
          {quantity_sql}
    """).df()
    return df
