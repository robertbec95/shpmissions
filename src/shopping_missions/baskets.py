"""Step C - Build the basket-level table and basket x category binary matrix."""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .categories import CategoryVocab


@dataclass
class BasketData:
    baskets: pd.DataFrame  # basket_id, store_id, variety, basket_volume, basket_value
    matrix: csr_matrix  # rows aligned to `baskets`, cols aligned to vocab
    vocab: CategoryVocab
    value_matrix: csr_matrix | None = None  # basket x category, cell = SUM(net_trans_amt) - only populated when include_value_matrix=True
    quantity_matrix: csr_matrix | None = None  # basket x category, cell = SUM(quantity) - only populated when include_quantity_matrix=True


def build_basket_data(
    con: duckdb.DuckDBPyConnection, vocab: CategoryVocab, include_timestamp: bool = False,
    include_value_matrix: bool = False, include_quantity_matrix: bool = False,
) -> BasketData:
    """`include_timestamp` carries `ANY_VALUE(timestamp)` through from a
    `lines` table ingested with `include_daypart=True` (`ingest.connect_lines`)
    - required before calling `temporal.add_daypart` on the result.

    `include_value_matrix` additionally builds a basket x category matrix of
    summed RON spend per category (vs. `matrix`'s binary presence/absence) -
    for clustering on a basket's spend *distribution* across categories
    rather than just which categories it touched (see mission_stage's
    Bray-Curtis functions).

    `include_quantity_matrix` additionally builds a basket x category matrix
    of summed item quantity per category - genuine integer-like counts,
    needed for discrete bag-of-words topic models (`dmr_missions`) where a
    category is represented as a token repeated by how many units of it
    were bought, rather than a continuous value share."""
    ts_select = ", ANY_VALUE(timestamp) AS timestamp" if include_timestamp else ""
    baskets = con.sql(f"""
        SELECT
            basket_id,
            ANY_VALUE(store_id) AS store_id,
            COUNT(DISTINCT category_id) AS variety,
            SUM(quantity) AS basket_volume,
            SUM(net_trans_amt) AS basket_value
            {ts_select}
        FROM lines
        GROUP BY basket_id
    """).df()
    baskets = baskets.sort_values("basket_id").reset_index(drop=True)

    pairs = con.sql("SELECT DISTINCT basket_id, category_id FROM lines").df()

    basket_row = {bid: i for i, bid in enumerate(baskets["basket_id"])}
    rows = pairs["basket_id"].map(basket_row).to_numpy()
    cols = pairs["category_id"].map(vocab.index_of).to_numpy()
    data = np.ones(len(pairs), dtype=np.uint8)

    matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(baskets), vocab.size), dtype=np.uint8
    )

    value_matrix = None
    if include_value_matrix:
        value_pairs = con.sql("""
            SELECT basket_id, category_id, SUM(net_trans_amt) AS value
            FROM lines
            GROUP BY basket_id, category_id
        """).df()
        vrows = value_pairs["basket_id"].map(basket_row).to_numpy()
        vcols = value_pairs["category_id"].map(vocab.index_of).to_numpy()
        vdata = value_pairs["value"].to_numpy(dtype=np.float64)
        value_matrix = csr_matrix(
            (vdata, (vrows, vcols)), shape=(len(baskets), vocab.size), dtype=np.float64
        )

    quantity_matrix = None
    if include_quantity_matrix:
        qty_pairs = con.sql("""
            SELECT basket_id, category_id, SUM(quantity) AS qty
            FROM lines
            GROUP BY basket_id, category_id
        """).df()
        qrows = qty_pairs["basket_id"].map(basket_row).to_numpy()
        qcols = qty_pairs["category_id"].map(vocab.index_of).to_numpy()
        qdata = qty_pairs["qty"].to_numpy(dtype=np.float64)
        quantity_matrix = csr_matrix(
            (qdata, (qrows, qcols)), shape=(len(baskets), vocab.size), dtype=np.float64
        )

    return BasketData(baskets=baskets, matrix=matrix, vocab=vocab, value_matrix=value_matrix, quantity_matrix=quantity_matrix)
