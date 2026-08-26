"""Generate a Table-6-style report (per Sarantopoulos et al. 2015) for our
own data: cluster solution + basket descriptives + category participation
probabilities for the top-25 categories.

Reuses the same pipeline stages/params as `shopping_missions.pipeline`
(k=6, sample_size=6000, random_state=0) so results are consistent with
`outputs/all3days`, and additionally pulls basket size (unit quantity summed
per basket) since the paper's Table 6 reports it and our core pipeline
doesn't need it for clustering.

Usage:
    PYTHONPATH=src python3 scripts/generate_table6.py --dates 20260509,20260510,20260511
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from shopping_missions import ingest, categories, baskets as baskets_mod, variety_stage, mission_stage, profiling
from shopping_missions.config import DEFAULT_K, DEFAULT_SAMPLE_SIZE, RANDOM_STATE

TOP_N_CATEGORIES = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output", type=Path, default=Path("docs/table6-comparison.md"))
    args = parser.parse_args()
    dates = [d.strip() for d in args.dates.split(",")]

    print(f"Ingesting {dates}...")
    con = ingest.connect_lines(dates)
    vocab = categories.build_vocab(con)
    data = baskets_mod.build_basket_data(con, vocab)  # includes basket_volume (paper's "basket size (items)")

    seg = variety_stage.fit_variety_segments(data.baskets)
    rel_mask = (seg.baskets["segment"] == "relatively_concrete").to_numpy()
    rel_baskets = seg.baskets.loc[rel_mask].reset_index(drop=True)
    rel_matrix = data.matrix[rel_mask]

    print(f"Clustering {len(rel_baskets)} relatively-concrete baskets (k={args.k})...")
    clustering = mission_stage.fit_and_assign_missions(
        rel_matrix, k=args.k, sample_size=args.sample_size, random_state=RANDOM_STATE
    )
    profiles = profiling.profile_clusters(rel_matrix, clustering.labels, vocab, rel_baskets)
    label_by_cluster = {p.cluster_id: p.suggested_label for p in profiles}

    dense = np.asarray(rel_matrix.todense())
    labels = clustering.labels
    k = args.k

    # --- Basket descriptives block ---
    def describe(mask):
        n = int(mask.sum())
        return {
            "variety": rel_baskets.loc[mask, "variety"].mean(),
            "basket_size": rel_baskets.loc[mask, "basket_volume"].mean(),
            "basket_value": rel_baskets.loc[mask, "basket_value"].mean(),
            "share_pct": n / len(rel_baskets) * 100,
            "n_baskets": n,
        }

    all_mask = np.ones(len(rel_baskets), dtype=bool)
    desc_cols = {"All": describe(all_mask)}
    for c in range(k):
        desc_cols[f"Cluster {c+1} ({label_by_cluster[c]})"] = describe(labels == c)

    # --- Top-N categories by overall popularity within relatively-concrete baskets ---
    overall_counts = dense.sum(axis=0).astype(np.int64)  # dense is uint8/uint64; negating unsigned wraps around
    top_idx = np.argsort(-overall_counts)[:TOP_N_CATEGORIES]

    cat_rows = []
    for i in top_idx:
        category_name = vocab.names[vocab.ids[i]]
        row = {"category": category_name, "All (baskets)": int(overall_counts[i])}
        for c in range(k):
            mask = labels == c
            support_pct = dense[mask, i].mean() * 100
            row[f"Cluster {c+1} ({label_by_cluster[c]})"] = round(support_pct, 1)
        cat_rows.append(row)

    _write_markdown(args.output, dates, args.k, len(data.baskets), len(rel_baskets), desc_cols, cat_rows)
    print(f"Wrote {args.output}")


def _write_markdown(path, dates, k, n_all_baskets, n_rel_baskets, desc_cols, cat_rows):
    cluster_headers = list(desc_cols.keys())

    lines = []
    lines.append("# Table 6 analog — shopping-mission cluster solution (our data)")
    lines.append("")
    lines.append(
        f"Reproduces the structure of **Table 6** in Sarantopoulos et al. (2015) — "
        f"the final cluster solution for relatively-concrete baskets, with basket "
        f"descriptives and category participation probabilities — computed on our "
        f"own dataset instead of their supermarket-pilot store."
    )
    lines.append("")
    lines.append(
        f"**Scope:** {', '.join(dates)} · offline channel · all {n_all_baskets:,} baskets pooled "
        f"across all stores (no store-format label is available in this dataset — see "
        f"`docs/mvp-plan.md` §2 — so this is the whole-chain analog of one of the paper's "
        f"per-store tables, not a single supermarket). k={k}."
    )
    lines.append("")

    lines.append("## Basket descriptives")
    lines.append("")
    header = "| | " + " | ".join(cluster_headers) + " |"
    sep = "|---|" + "---|" * len(cluster_headers)
    lines.append(header)
    lines.append(sep)

    def fmt_row(label, key, fmt):
        cells = [fmt(desc_cols[c][key]) for c in cluster_headers]
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(fmt_row("Average basket variety (categories)", "variety", lambda v: f"{v:.2f}"))
    lines.append(fmt_row("Average basket size (items)", "basket_size", lambda v: f"{v:.2f}"))
    lines.append(fmt_row("Average basket value (RON)", "basket_value", lambda v: f"{v:.2f}"))
    lines.append(fmt_row("Total share of baskets (%)", "share_pct", lambda v: f"{v:.1f}%"))
    lines.append(fmt_row("Number of baskets", "n_baskets", lambda v: f"{v:,}"))
    lines.append("")

    lines.append(f"## Category participation probabilities (top {len(cat_rows)} categories)")
    lines.append("")
    lines.append(
        "`All (baskets)` is the raw count of relatively-concrete baskets containing the "
        "category (matches the paper's format); cluster columns are the % of that cluster's "
        "baskets containing the category (participation probability)."
    )
    lines.append("")
    cat_header = "| Category | All (baskets) | " + " | ".join(cluster_headers[1:]) + " |"
    cat_sep = "|---|---|" + "---|" * (len(cluster_headers) - 1)
    lines.append(cat_header)
    lines.append(cat_sep)
    for row in cat_rows:
        cells = [row["category"], f"{row['All (baskets)']:,}"] + [
            f"{row[c]:.1f}%" for c in cluster_headers[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
