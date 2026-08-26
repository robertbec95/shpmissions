"""Materialize a cleaned, clustering-ready basket fact table (parquet).

Cleaning applied, beyond the pipeline's existing offline-channel and
noise-category filters:
  - negative-quantity ("return") baskets dropped (`ingest.connect_lines`;
    on 2026-05-09 this is 339 baskets, each entirely made of return lines -
    no mixed baskets, see `ingest.py` docstring)
  - single-category ("very concrete") baskets dropped (`fact_table.py`) -
    self-evident need state, no mission signal to discover
    (Sarantopoulos et al. 2015, Sec. 2.3/5.4-5.5)

Output is one row per basket: basket_id, store_id, basket_volume (items),
basket_variety (categories), basket_value, and one binary column per
category - directly loadable for clustering experiments without re-running
the ingestion/vocab/matrix-building steps.

Usage:
    PYTHONPATH=src python3 scripts/build_clustering_dataset.py --dates 20260509,20260510,20260511
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_missions import ingest, categories, baskets as baskets_mod, fact_table
from shopping_missions.config import EXCLUDE_CATALOG_IDS, OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument(
        "--min-variety", type=int, default=2,
        help="drop baskets with fewer distinct categories than this (default 2, i.e. drop single-category baskets)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "clustering_dataset.parquet")
    args = parser.parse_args()
    dates = [d.strip() for d in args.dates.split(",")]

    print(f"Ingesting {dates} (offline channel; noise categories and negative-quantity/return baskets excluded)...")
    con = ingest.connect_lines(dates)
    vocab = categories.build_vocab(con)
    data = baskets_mod.build_basket_data(con, vocab)
    n_after_qty_clean = len(data.baskets)
    print(f"{n_after_qty_clean:,} baskets after negative-quantity cleaning, {vocab.size} categories in vocabulary")

    ft = fact_table.build_fact_table(data, min_variety=args.min_variety)
    print(f"Dropped {ft.n_dropped_single_category:,} baskets with < {args.min_variety} categories "
          f"({ft.n_dropped_single_category / n_after_qty_clean * 100:.1f}% of the cleaned population)")
    print(f"Final clustering dataset: {len(ft.df):,} baskets x {len(ft.category_columns)} category columns "
          f"(+ {len(ft.df.columns) - len(ft.category_columns)} descriptive columns)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ft.df.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}")

    manifest = {
        "dates": dates,
        "channel": "Offline",
        "excluded_catalog_ids": sorted(EXCLUDE_CATALOG_IDS),
        "negative_quantity_dropped": True,
        "min_variety": args.min_variety,
        "n_baskets_after_negative_quantity_cleaning": n_after_qty_clean,
        "n_baskets_dropped_single_category": ft.n_dropped_single_category,
        "n_baskets_final": len(ft.df),
        "n_category_columns": len(ft.category_columns),
        "category_columns": ft.category_columns,
        "descriptive_columns": fact_table.BASKET_COLUMNS,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
