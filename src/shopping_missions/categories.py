"""Step B - Define the fixed category vocabulary used for basket vectors."""

from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass
class CategoryVocab:
    """Fixed, ordered mapping between category IDs and vector positions."""

    ids: list[int]
    names: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.ids)

    def index_of(self, category_id: int) -> int:
        return self._index[category_id]

    def __post_init__(self):
        self._index = {cid: i for i, cid in enumerate(self.ids)}


def build_vocab(con: duckdb.DuckDBPyConnection) -> CategoryVocab:
    """Derive the category vocabulary from the (already-cleaned) `lines` table.

    Ordering is by category_id ascending, purely for reproducibility - the
    order has no modeling meaning since basket vectors are treated as sets.
    """
    df: pd.DataFrame = con.sql("""
        SELECT DISTINCT category_id, category
        FROM lines
        ORDER BY category_id
    """).df()
    return build_vocab_from_pairs(df)


def build_vocab_from_pairs(pairs: pd.DataFrame) -> CategoryVocab:
    """Same as `build_vocab`, but from a plain (category_id, category)
    DataFrame - e.g. `ingest.list_categories`'s output - rather than a live
    `lines` table. Used by `multiday.build_basket_data_multi_date` to fix a
    stable vocabulary across dates before any per-date basket matrix is built."""
    df = pairs.drop_duplicates(subset="category_id").sort_values("category_id")
    ids = df["category_id"].tolist()
    names = dict(zip(df["category_id"], df["category"]))
    return CategoryVocab(ids=ids, names=names)
