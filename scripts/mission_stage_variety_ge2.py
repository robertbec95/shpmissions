"""Stage 2 (mission) clustering with Stage 1 bypassed entirely: instead of
restricting to the "relatively_concrete" band that Stage 1's k-means fit
would select, this runs Jaccard k-medoids directly on every basket with
variety >= 2 (i.e., only single-category baskets are excluded - the simple
rule already baked into outputs/clustering_dataset.parquet, see
fact_table.py / build_clustering_dataset.py).

Motivated by the Stage 1 gap-statistic finding (docs/k-selection.md,
docs/k-selection-by-format.md): the concrete/relatively-concrete/abstract
split has no independent support in this data's variety distribution (gap
statistic prefers k=1 there, in every stratum tried), so restricting Stage 2
to Stage 1's middle band may be filtering out baskets for no data-driven
reason. This tests what Stage 2 finds if that filter is dropped and only
the single-category exclusion remains.

Produces two outputs:
  - docs/table6-comparison-variety-ge2.md - a Table-6-style cluster profile
    (mirrors docs/table6-comparison.md and docs/table6-comparison-k2.md)
  - docs/k-selection-variety-ge2.md - gap statistic + silhouette + elbow on
    this same population, to check whether k=6 (kept for comparability with
    the earlier tables) is actually supported here or just inherited

Usage:
    PYTHONPATH=src python3 scripts/mission_stage_variety_ge2.py --k 6
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from shopping_missions import mission_stage
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import OUTPUTS_DIR, RANDOM_STATE

DATASET_PATH = OUTPUTS_DIR / "clustering_dataset.parquet"
TOP_N_CATEGORIES = 25
LABEL_KEYWORDS = {
    "breakfast": {"BREAKFAST", "CEREALS", "BUTTER", "EGGS", "MILK", "BAKE OFF", "SUPPLIED BAKERY"},
    "coffee": {"COFFEE", "SUGAR", "CHOCOLATE"},
    "meal": {"MEAT", "CHICKEN", "PORK", "FISH", "VEGETABLES", "RICE", "PASTA", "GARNISHES", "COOKED MEALS", "SAUCES"},
    "light_meal": {"CHEESE", "COLD CUTS", "PREPARED MEAT", "SALADS", "BAKE OFF", "PACKED SAUSAGES"},
    "snack": {"SALTY SNACKS", "SWEET SNACKS", "CHEWING GUM", "CHOCOLATE", "CARBONATED DRINKS", "ICE CREAM"},
    "cleaning": {"DETERGENTS", "BLEACHES", "HOME CARE", "HOUSHOLD PAPER", "HOUSEHOLD PRODUCTS"},
}


def load_data():
    df = pd.read_parquet(DATASET_PATH)
    category_columns = [c for c in df.columns if c.startswith("cat_")]
    category_names = {c: c[len("cat_"):].split("_", 1)[1].replace("_", " ") for c in category_columns}
    return df, category_columns, category_names


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
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--run-k-selection", action="store_true", default=True)
    parser.add_argument("--no-k-selection", dest="run_k_selection", action="store_false")
    parser.add_argument("--table-output", type=Path, default=Path("docs/table6-comparison-variety-ge2.md"))
    parser.add_argument("--kselect-output", type=Path, default=Path("docs/k-selection-variety-ge2.md"))
    args = parser.parse_args()

    print("Loading outputs/clustering_dataset.parquet (variety>=2, negative-quantity already excluded)...")
    df, category_columns, category_names = load_data()
    print(f"{len(df):,} baskets, {len(category_columns)} categories - Stage 1 bypassed entirely")

    matrix = csr_matrix(df[category_columns].to_numpy(dtype=np.uint8))

    print(f"Clustering (k={args.k}, sample_size={args.sample_size})...")
    t0 = time.time()
    clustering = mission_stage.fit_and_assign_missions(
        matrix, k=args.k, sample_size=args.sample_size, random_state=RANDOM_STATE
    )
    print(f"  done in {time.time() - t0:.1f}s")

    profiles = _profile_clusters(df, matrix, clustering.labels, category_columns, category_names, args.k)
    for p in profiles:
        top = ", ".join(c["category"] for c in p["top_categories"][:4])
        print(f"  cluster {p['cluster_id']}: {p['suggested_label']} (conf={p['label_confidence']}) "
              f"n={p['n_baskets']} ({p['share_pct']}%) top=[{top}]")

    _write_table6(args.table_output, df, clustering.labels, profiles, category_columns, category_names, args.k)
    print(f"Wrote {args.table_output}")

    if args.run_k_selection:
        print("Running gap statistic + silhouette + elbow on this population...")
        _run_and_write_k_selection(df, category_columns, args.kselect_output)
        print(f"Wrote {args.kselect_output}")


def _profile_clusters(df, matrix, labels, category_columns, category_names, k):
    dense = matrix.toarray().astype(float)
    overall_support = dense.mean(axis=0)
    n = len(labels)

    profiles = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        n_baskets = int(mask.sum())
        cluster_support = dense[mask].mean(axis=0) if n_baskets else np.zeros(len(category_columns))
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_support > 0, cluster_support / overall_support, 0.0)

        candidates = [
            {"category": category_names[category_columns[i]], "support": round(float(cluster_support[i]), 4),
             "lift": round(float(lift[i]), 3)}
            for i in range(len(category_columns))
            if cluster_support[i] >= 0.15 and lift[i] >= 1.2
        ]
        candidates.sort(key=lambda c: c["lift"], reverse=True)
        top_categories = candidates[:6]
        label, confidence = suggest_label(top_categories)

        profiles.append({
            "cluster_id": cluster_id,
            "n_baskets": n_baskets,
            "share_pct": round(n_baskets / n * 100, 2),
            "avg_basket_value": round(float(df.loc[mask, "basket_value"].mean()), 2) if n_baskets else None,
            "avg_basket_volume": round(float(df.loc[mask, "basket_volume"].mean()), 2) if n_baskets else None,
            "avg_basket_variety": round(float(df.loc[mask, "basket_variety"].mean()), 2) if n_baskets else None,
            "top_categories": top_categories,
            "suggested_label": label,
            "label_confidence": confidence,
        })
    return profiles


def _write_table6(path, df, labels, profiles, category_columns, category_names, k):
    n = len(df)
    dense = None  # computed lazily below only for the popularity ranking

    lines = []
    lines.append("# Table 6 analog - Stage 2 run on ALL variety>=2 baskets (Stage 1 bypassed)")
    lines.append("")
    lines.append(
        "Same Jaccard k-medoids clustering as `docs/table6-comparison.md`, but the input population "
        "is different: instead of restricting to Stage 1's k-means-selected \"relatively_concrete\" "
        "band, every basket with variety>=2 is included (only single-category baskets excluded - "
        "the rule already applied when building `outputs/clustering_dataset.parquet`). Stage 1's "
        "concrete/relatively-concrete/abstract clustering plays no role in selecting this input."
    )
    lines.append("")
    lines.append(
        f"**Scope:** {n:,} baskets (vs. 1,222,439 relatively-concrete baskets in the original "
        f"`docs/table6-comparison.md` run - {n/1222439:.2f}x the population, since former-\"abstract\" "
        f"baskets are back in). k={k}."
    )
    lines.append("")

    lines.append("## Basket descriptives")
    lines.append("")
    headers = ["All"] + [f"Cluster {p['cluster_id']+1} ({p['suggested_label']})" for p in profiles]
    lines.append("| | " + " | ".join(headers) + " |")
    lines.append("|---|" + "---|" * len(headers))

    def row(label, all_val, fn):
        vals = [fn(p) for p in profiles]
        return f"| {label} | {all_val} | " + " | ".join(vals) + " |"

    lines.append(row("Average basket variety (categories)", f"{df['basket_variety'].mean():.2f}",
                      lambda p: f"{p['avg_basket_variety']:.2f}"))
    lines.append(row("Average basket size (items)", f"{df['basket_volume'].mean():.2f}",
                      lambda p: f"{p['avg_basket_volume']:.2f}"))
    lines.append(row("Average basket value (RON)", f"{df['basket_value'].mean():.2f}",
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

    dense_full = csr_matrix(df[category_columns].to_numpy(dtype=np.uint8))
    overall_counts = np.asarray(dense_full.sum(axis=0)).ravel().astype(np.int64)
    top_idx = np.argsort(-overall_counts)[:TOP_N_CATEGORIES]
    dense = dense_full.toarray()

    cat_header = "| Category | All (baskets) | " + " | ".join(headers[1:]) + " |"
    lines.append(cat_header)
    lines.append("|---|---|" + "---|" * len(profiles))
    for i in top_idx:
        cat_name = category_names[category_columns[i]]
        row_cells = [cat_name, f"{overall_counts[i]:,}"]
        for p in profiles:
            mask = labels == p["cluster_id"]
            pct = dense[mask, i].mean() * 100 if mask.sum() else 0.0
            row_cells.append(f"{pct:.1f}%")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append(
        "Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_variety_ge2.py --k " + str(k) + "`"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _run_and_write_k_selection(df, category_columns, path):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(1500, len(df))
    idx = rng.choice(len(df), size=sample_size, replace=False)
    X_sample = df.iloc[idx][category_columns].to_numpy(dtype=np.float64)

    gdf = mission_stage.select_k(X_sample, k_range=range(2, 12), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    elbow_k = elbow_k_from_gap_df(gdf)
    print(f"  gap_k={gap_k}  sil_k={sil_k}  elbow_k={elbow_k}")

    lines = []
    lines.append("# Gap statistic / silhouette / elbow - Stage 2 on ALL variety>=2 baskets")
    lines.append("")
    lines.append(
        f"Same diagnostic as `docs/k-selection.md` and `docs/k-selection-by-format.md`, run on the "
        f"population used in `docs/table6-comparison-variety-ge2.md` - all {len(df):,} variety>=2 "
        f"baskets, Stage 1's relatively-concrete filter not applied. Sample size {sample_size:,}, B=10."
    )
    lines.append("")
    lines.append(f"**Gap statistic selects k={gap_k}. Silhouette is maximized at k={sil_k}. Elbow method picks k={elbow_k}.**")
    lines.append("")

    gdf_display = gdf[["k", "log_Wk", "gap", "sk", "gap_rule_satisfied", "silhouette"]].copy()
    for col in ["log_Wk", "gap", "sk", "silhouette"]:
        gdf_display[col] = gdf_display[col].map(lambda v: f"{v:.4f}" if v == v else "n/a")
    lines.append(gdf_display.to_markdown(index=False))
    lines.append("")
    lines.append(
        "Reproduce with: `PYTHONPATH=src python3 scripts/mission_stage_variety_ge2.py`"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
