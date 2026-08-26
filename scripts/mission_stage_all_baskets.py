"""Stage 2 (mission) clustering with NO variety floor at all: every basket
(variety>=1) goes into the Jaccard k-medoids clustering, including
single-category baskets. Negative-quantity/return baskets are still
excluded (that's a data-quality fix from `ingest.py`, not a variety filter).

This is the most permissive of the three Stage-2 population variants now
run for comparison:
  - docs/table6-comparison.md            - Stage 1's relatively_concrete band only
  - docs/table6-comparison-variety-ge2.md - variety>=2 (single-category dropped)
  - docs/table6-comparison-all-baskets.md - this one: every basket, no floor

Single-category baskets are the ones the paper argues carry no discoverable
mission signal (the category itself is the self-evident answer) - this run
tests that claim directly: if it holds, these baskets should mostly land in
whichever cluster their one category has the highest lift in, without
distorting the other clusters' definitions.

Usage:
    PYTHONPATH=src python3 scripts/mission_stage_all_baskets.py --dates 20260509,20260510,20260511 --k 6
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from shopping_missions import ingest, categories, baskets as baskets_mod, mission_stage
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import RANDOM_STATE

TOP_N_CATEGORIES = 25
LABEL_KEYWORDS = {
    "breakfast": {"BREAKFAST", "CEREALS", "BUTTER", "EGGS", "MILK", "BAKE OFF", "SUPPLIED BAKERY"},
    "coffee": {"COFFEE", "SUGAR", "CHOCOLATE"},
    "meal": {"MEAT", "CHICKEN", "PORK", "FISH", "VEGETABLES", "RICE", "PASTA", "GARNISHES", "COOKED MEALS", "SAUCES"},
    "light_meal": {"CHEESE", "COLD CUTS", "PREPARED MEAT", "SALADS", "BAKE OFF", "PACKED SAUSAGES"},
    "snack": {"SALTY SNACKS", "SWEET SNACKS", "CHEWING GUM", "CHOCOLATE", "CARBONATED DRINKS", "ICE CREAM"},
    "cleaning": {"DETERGENTS", "BLEACHES", "HOME CARE", "HOUSHOLD PAPER", "HOUSEHOLD PRODUCTS"},
}


def suggest_label(top_categories):
    names = {c["category"].upper() for c in top_categories}
    best_label, best_score = "unlabeled", 0.0
    for label, keywords in LABEL_KEYWORDS.items():
        hits = sum(1 for name in names if any(kw in name for kw in keywords))
        score = hits / max(len(top_categories), 1)
        if score > best_score:
            best_label, best_score = label, score
    return best_label, round(best_score, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", default="20260509,20260510,20260511")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--run-k-selection", action="store_true", default=True)
    parser.add_argument("--no-k-selection", dest="run_k_selection", action="store_false")
    parser.add_argument("--table-output", type=Path, default=Path("docs/table6-comparison-all-baskets.md"))
    parser.add_argument("--kselect-output", type=Path, default=Path("docs/k-selection-all-baskets.md"))
    args = parser.parse_args()
    dates = [d.strip() for d in args.dates.split(",")]

    print(f"Ingesting {dates} (offline channel; noise categories and negative-quantity/return baskets excluded; NO variety floor)...")
    con = ingest.connect_lines(dates)
    vocab = categories.build_vocab(con)
    data = baskets_mod.build_basket_data(con, vocab)
    n = len(data.baskets)
    print(f"{n:,} baskets, {vocab.size} categories - variety distribution:")
    print(data.baskets["variety"].value_counts().sort_index().head(10).to_string())

    print(f"\nClustering (k={args.k}, sample_size={args.sample_size})...")
    t0 = time.time()
    clustering = mission_stage.fit_and_assign_missions(
        data.matrix, k=args.k, sample_size=args.sample_size, random_state=RANDOM_STATE
    )
    print(f"  done in {time.time() - t0:.1f}s")

    profiles = _profile_clusters(data.baskets, data.matrix, clustering.labels, vocab, args.k)
    for p in profiles:
        top = ", ".join(c["category"] for c in p["top_categories"][:4])
        print(f"  cluster {p['cluster_id']}: {p['suggested_label']} (conf={p['label_confidence']}) "
              f"n={p['n_baskets']:,} ({p['share_pct']}%) avg_variety={p['avg_basket_variety']} top=[{top}]")

    _write_table6(args.table_output, data.baskets, data.matrix, clustering.labels, profiles, vocab, args.k, n)
    print(f"Wrote {args.table_output}")

    if args.run_k_selection:
        print("\nRunning gap statistic + silhouette + elbow on this population...")
        _run_and_write_k_selection(data, args.kselect_output, n)
        print(f"Wrote {args.kselect_output}")


def _profile_clusters(baskets, matrix, labels, vocab, k):
    n = len(labels)
    overall_support = np.asarray(matrix.sum(axis=0)).ravel() / n

    profiles = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        nc = int(mask.sum())
        sub = matrix[mask]
        cluster_support = np.asarray(sub.sum(axis=0)).ravel() / nc if nc else np.zeros(vocab.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_support > 0, cluster_support / overall_support, 0.0)

        candidates = [
            {"category": vocab.names[vocab.ids[i]], "support": round(float(cluster_support[i]), 4),
             "lift": round(float(lift[i]), 3)}
            for i in range(vocab.size)
            if cluster_support[i] >= 0.15 and lift[i] >= 1.2
        ]
        candidates.sort(key=lambda c: c["lift"], reverse=True)
        top_categories = candidates[:6]
        label, confidence = suggest_label(top_categories)

        profiles.append({
            "cluster_id": cluster_id,
            "n_baskets": nc,
            "share_pct": round(nc / n * 100, 2),
            "avg_basket_value": round(float(baskets.loc[mask, "basket_value"].mean()), 2) if nc else None,
            "avg_basket_volume": round(float(baskets.loc[mask, "basket_volume"].mean()), 2) if nc else None,
            "avg_basket_variety": round(float(baskets.loc[mask, "variety"].mean()), 2) if nc else None,
            "top_categories": top_categories,
            "suggested_label": label,
            "label_confidence": confidence,
        })
    return profiles


def _write_table6(path, baskets, matrix, labels, profiles, vocab, k, n):
    lines = []
    lines.append("# Table 6 analog - Stage 2 run on ALL baskets (no variety floor at all)")
    lines.append("")
    lines.append(
        "Most permissive of the three Stage-2 population variants: every basket goes in, including "
        "single-category (variety=1) ones - only negative-quantity/return baskets are excluded "
        "(that's a data-quality fix from `ingest.py`, unrelated to variety). Compare against "
        "`docs/table6-comparison.md` (Stage 1's relatively_concrete band only) and "
        "`docs/table6-comparison-variety-ge2.md` (variety>=2)."
    )
    lines.append("")
    lines.append(f"**Scope:** {n:,} baskets (vs. 2,364,042 in the variety>=2 run, vs. 1,222,439 relatively-concrete-only). k={k}.")
    lines.append("")

    lines.append("## Basket descriptives")
    lines.append("")
    headers = ["All"] + [f"Cluster {p['cluster_id']+1} ({p['suggested_label']})" for p in profiles]
    lines.append("| | " + " | ".join(headers) + " |")
    lines.append("|---|" + "---|" * len(headers))

    def row(label, all_val, fn):
        vals = [fn(p) for p in profiles]
        return f"| {label} | {all_val} | " + " | ".join(vals) + " |"

    lines.append(row("Average basket variety (categories)", f"{baskets['variety'].mean():.2f}",
                      lambda p: f"{p['avg_basket_variety']:.2f}"))
    lines.append(row("Average basket size (items)", f"{baskets['basket_volume'].mean():.2f}",
                      lambda p: f"{p['avg_basket_volume']:.2f}"))
    lines.append(row("Average basket value (RON)", f"{baskets['basket_value'].mean():.2f}",
                      lambda p: f"{p['avg_basket_value']:.2f}"))
    lines.append(row("Total share of baskets (%)", "100.0%", lambda p: f"{p['share_pct']}%"))
    lines.append(row("Number of baskets", f"{n:,}", lambda p: f"{p['n_baskets']:,}"))
    lines.append("")

    lines.append(f"## Category participation probabilities (top {TOP_N_CATEGORIES} categories)")
    lines.append("")
    lines.append(
        "`All (baskets)` is the raw count of baskets (out of this population) containing the "
        "category; cluster columns are the % of that cluster's baskets containing it."
    )
    lines.append("")

    overall_counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    top_idx = np.argsort(-overall_counts)[:TOP_N_CATEGORIES]

    cat_header = "| Category | All (baskets) | " + " | ".join(headers[1:]) + " |"
    lines.append(cat_header)
    lines.append("|---|---|" + "---|" * len(profiles))
    for i in top_idx:
        cat_name = vocab.names[vocab.ids[i]]
        row_cells = [cat_name, f"{overall_counts[i]:,}"]
        for p in profiles:
            mask = labels == p["cluster_id"]
            nc = int(mask.sum())
            pct = (np.asarray(matrix[mask, i].sum()).ravel()[0] / nc * 100) if nc else 0.0
            row_cells.append(f"{pct:.1f}%")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append(
        "Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_all_baskets.py --k " + str(k) + "`"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _run_and_write_k_selection(data, path, n):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(1500, n)
    idx = rng.choice(n, size=sample_size, replace=False)
    X_sample = np.asarray(data.matrix[idx].todense(), dtype=np.float64)

    gdf = mission_stage.select_k(X_sample, k_range=range(2, 12), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    elbow_k = elbow_k_from_gap_df(gdf)
    print(f"  gap_k={gap_k}  sil_k={sil_k}  elbow_k={elbow_k}")

    lines = []
    lines.append("# Gap statistic / silhouette / elbow - Stage 2 on ALL baskets (no variety floor)")
    lines.append("")
    lines.append(
        f"Same diagnostic as `docs/k-selection-variety-ge2.md`, run on the full {n:,}-basket "
        f"population used in `docs/table6-comparison-all-baskets.md` - every basket, single-category "
        f"included. Sample size {sample_size:,}, B=10."
    )
    lines.append("")
    lines.append(f"**Gap statistic selects k={gap_k}. Silhouette is maximized at k={sil_k}. Elbow method picks k={elbow_k}.**")
    lines.append("")

    gdf_display = gdf[["k", "log_Wk", "gap", "sk", "gap_rule_satisfied", "silhouette"]].copy()
    for col in ["log_Wk", "gap", "sk", "silhouette"]:
        gdf_display[col] = gdf_display[col].map(lambda v: f"{v:.4f}" if v == v else "n/a")
    lines.append(gdf_display.to_markdown(index=False))
    lines.append("")
    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_all_baskets.py`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
