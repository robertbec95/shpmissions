"""Materialize a clustering-ready fact table: one row per basket, matching
the paper's participation-matrix structure (Table 3) - basket descriptives
plus one binary column per category.

Drops single-category ("very concrete") baskets before returning, since per
the paper (Sec. 2.3, 5.4-5.5) these carry no discoverable mission signal -
the need state is self-evident (the category itself), and empirically
clustering this segment produces no interpretable output. Negative-quantity
("return") baskets are already excluded upstream, in `ingest.connect_lines`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baskets import BasketData

BASKET_COLUMNS = ["basket_id", "store_id", "basket_volume", "basket_variety", "basket_value"]


@dataclass
class FactTable:
    df: pd.DataFrame  # basket_id, store_id, basket_volume, basket_variety, basket_value, <category columns>
    category_columns: list[str]
    n_dropped_single_category: int


def build_fact_table(data: BasketData, min_variety: int = 2) -> FactTable:
    baskets = data.baskets
    keep_mask = (baskets["variety"] >= min_variety).to_numpy()
    n_dropped = int((~keep_mask).sum())

    kept_baskets = baskets.loc[keep_mask].reset_index(drop=True)
    kept_matrix = data.matrix[keep_mask]

    category_columns = [f"cat_{cid}_{_slug(data.vocab.names[cid])}" for cid in data.vocab.ids]
    cat_df = pd.DataFrame(np.asarray(kept_matrix.todense()), columns=category_columns)

    df = pd.concat(
        [kept_baskets.rename(columns={"variety": "basket_variety"}).reset_index(drop=True), cat_df],
        axis=1,
    )
    df = df[BASKET_COLUMNS + category_columns]

    return FactTable(df=df, category_columns=category_columns, n_dropped_single_category=n_dropped)


def _slug(name: str) -> str:
    return (
        name.strip().upper()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace("  ", " ")
        .replace(" ", "_")
    )
