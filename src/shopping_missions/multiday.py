"""Process multiple dates incrementally - one date at a time, immediately
reduced to basket-level - instead of unioning every requested day's raw
line items into one materialized table at once.

Why this exists: `ingest.connect_lines(dates)` unions all requested dates'
files in a single query and materializes the filtered result as a `lines`
table. That's fine at 3 days (~15M rows) but exhausted both disk (~3GB free)
and RAM (8GB total) on an 8-day/~40M-row run on 2026-07-21. Basket-level
data is far smaller than line-item data (~1 row per basket vs. ~4 per
basket), so processing date-by-date and concatenating basket-level results
bounds peak memory/disk to roughly one day's raw file regardless of how many
dates are requested - the same property `read_csv` alone gave us for a
single day, just extended across the date loop instead of relying on
DuckDB's query engine to do it for a multi-file union.
"""

import pandas as pd
from scipy.sparse import vstack

from . import baskets as baskets_mod
from . import categories, ingest
from .baskets import BasketData
from .categories import CategoryVocab
from .config import EXCLUDE_CATALOG_IDS


def build_basket_data_multi_date(
    dates: list[str],
    channel: str = "Offline",
    exclude_category_ids: set[int] = EXCLUDE_CATALOG_IDS,
    drop_negative_quantity: bool = True,
    category_level: int = 3,
    include_daypart: bool = False,
    include_value_matrix: bool = False,
) -> BasketData:
    """Same result shape as `baskets.build_basket_data`, built incrementally.

    1. Fix a category vocabulary across ALL requested dates up front
       (`ingest.list_categories` - a cheap distinct-only scan), so every
       date's basket matrix uses identical column indexing.
    2. For each date: ingest just that date, build its basket-level table
       and matrix, then close the connection before moving on - at no point
       does more than one date's raw line items exist in memory/on disk at
       once.
    3. Concatenate the (small) basket-level DataFrames and vertically stack
       the sparse matrices.
    """
    vocab_pairs = ingest.list_categories(
        dates, channel=channel, exclude_category_ids=exclude_category_ids,
        drop_negative_quantity=drop_negative_quantity, category_level=category_level,
    )
    vocab: CategoryVocab = categories.build_vocab_from_pairs(vocab_pairs)

    all_baskets = []
    matrices = []
    value_matrices = []
    for date in dates:
        con = ingest.connect_lines(
            [date], channel=channel, exclude_category_ids=exclude_category_ids,
            drop_negative_quantity=drop_negative_quantity, category_level=category_level,
            include_daypart=include_daypart, product_reference_dates=dates,
        )
        data = baskets_mod.build_basket_data(
            con, vocab, include_timestamp=include_daypart, include_value_matrix=include_value_matrix,
        )
        con.close()
        all_baskets.append(data.baskets)
        matrices.append(data.matrix)
        if include_value_matrix:
            value_matrices.append(data.value_matrix)

    baskets = pd.concat(all_baskets, ignore_index=True)
    matrix = vstack(matrices, format="csr")
    value_matrix = vstack(value_matrices, format="csr") if include_value_matrix else None
    return BasketData(baskets=baskets, matrix=matrix, vocab=vocab, value_matrix=value_matrix)
