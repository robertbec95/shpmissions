"""Re-run the gap statistic + silhouette k-selection (docs/k-selection.md)
on the cleaned dataset (variety >= 2, negative-quantity baskets already
dropped - see scripts/build_clustering_dataset.py), now also stratified by
real store format (`data/locations-export (3).csv`, joined via
`store_formats.py`).

Tests the standing hypothesis from docs/k-selection.md: that pooling across
~1,740 stores of different formats, and including single-category baskets
in the variety-clustering input, were both suppressing real cluster
structure in the original all-pooled run (which found k=1 for variety,
k=2 for missions).

Stage 1 (variety) is re-run pooled and per format. Stage 2 (missions) holds
Stage 1's method fixed at k=3 (matching the current pipeline) so that only
the population (variety>=2, and format stratification) varies between this
and the original run - isolating the effect being tested rather than
conflating it with a simultaneous Stage-1-k change.

Usage:
    PYTHONPATH=src python3 scripts/select_k_by_format.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from shopping_missions import variety_stage, mission_stage, store_formats
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import OUTPUTS_DIR
from shopping_missions.fact_table import BASKET_COLUMNS

DATASET_PATH = OUTPUTS_DIR / "clustering_dataset.parquet"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "k-selection-by-format.md"

VARIETY_K_RANGE = range(1, 8)
VARIETY_SAMPLE_SIZE = 30_000
VARIETY_B = 8
MISSION_K_RANGE = range(2, 12)
MISSION_SAMPLE_SIZE = 1500
MISSION_B = 10
RANDOM_STATE = 0


def load_data():
    df = pd.read_parquet(DATASET_PATH)
    df = df.rename(columns={"basket_variety": "variety"})
    df = store_formats.join_store_format(df)
    category_columns = [c for c in df.columns if c.startswith("cat_")]
    return df, category_columns


def run_variety_stage(df, label):
    t0 = time.time()
    gdf = variety_stage.select_k(
        df["variety"].to_numpy(), k_range=VARIETY_K_RANGE, B=VARIETY_B,
        sample_size=VARIETY_SAMPLE_SIZE, random_state=RANDOM_STATE,
    )
    elapsed = time.time() - t0
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"]) if gdf["silhouette"].notna().any() else None
    elbow_k = elbow_k_from_gap_df(gdf)
    print(f"  [{label}] n={len(df):,}  gap_k={gap_k}  sil_k={sil_k}  elbow_k={elbow_k}  ({elapsed:.1f}s)")
    return gdf, gap_k, sil_k, elbow_k


def run_mission_stage(df, category_columns, label):
    seg = variety_stage.fit_variety_segments(df)
    rel = seg.baskets.loc[seg.baskets["segment"] == "relatively_concrete"]
    if len(rel) < MISSION_SAMPLE_SIZE:
        print(f"  [{label}] too few relatively-concrete baskets ({len(rel)}) - skipping")
        return None, None, None, None, seg, 0

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(rel), size=MISSION_SAMPLE_SIZE, replace=False)
    X_sample = rel.iloc[idx][category_columns].to_numpy(dtype=np.float64)

    t0 = time.time()
    gdf = mission_stage.select_k(X_sample, k_range=MISSION_K_RANGE, B=MISSION_B, random_state=RANDOM_STATE)
    elapsed = time.time() - t0
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    elbow_k = elbow_k_from_gap_df(gdf)
    print(f"  [{label}] relatively-concrete n={len(rel):,}  gap_k={gap_k}  sil_k={sil_k}  elbow_k={elbow_k}  ({elapsed:.1f}s)")
    return gdf, gap_k, sil_k, elbow_k, seg, len(rel)


def main():
    print("Loading cleaned dataset + store format join...")
    df, category_columns = load_data()
    print(f"{len(df):,} baskets (variety>=2), {len(category_columns)} categories")
    print(df["store_format"].value_counts().to_string())

    strata = {"ALL (pooled, variety>=2)": df}
    for fmt in store_formats.MAIN_FORMATS:
        strata[fmt] = df[df["store_format"] == fmt]

    print("\nStage 1 - variety gap statistic + silhouette, per stratum:")
    variety_results = {}
    for label, sub in strata.items():
        variety_results[label] = run_variety_stage(sub, label)

    print("\nStage 2 - mission gap statistic + silhouette, per stratum (Stage 1 held at k=3):")
    mission_results = {}
    for label, sub in strata.items():
        mission_results[label] = run_mission_stage(sub, category_columns, label)

    _write_report(strata, variety_results, mission_results)
    print(f"\nWrote {OUTPUT_PATH}")


def _write_report(strata, variety_results, mission_results):
    lines = []
    lines.append("# Gap statistic, re-run on variety>=2 with real store format")
    lines.append("")
    lines.append(
        "Two changes from `docs/k-selection.md`: (1) input is now "
        "`outputs/clustering_dataset.parquet` - negative-quantity/return "
        "baskets dropped and, critically, **single-category (variety=1) "
        "baskets excluded from the variety-clustering input itself** "
        "(previously ~36% of the population and likely the reason the "
        "original run picked k=1); (2) real store format is now available "
        "from `data/locations-export (3).csv` (`store_format`, joined on "
        "`Store ID` / `location_external_id` - matches all 1,740 stores "
        "seen in the transaction data) in place of the basket-volume-tertile "
        "proxy, so results are also stratified by format instead of only "
        "pooled."
    )
    lines.append("")
    lines.append(
        "Stage 1 sample size/B reduced from the original run (50k/B=10 to "
        "30k/B=8) to keep 4 strata tractable in one pass; still a stable "
        "sample for 1-D k-means. Stage 2 keeps its original 1,500/B=10."
    )
    lines.append("")

    lines.append(
        "Elbow method added alongside gap/silhouette: the point of maximum "
        "distance from the chord connecting the first and last points on the "
        "raw within-cluster-dispersion-vs-k curve (`cluster_selection.find_elbow_k`) "
        "- a third, independent read (no reference datasets, unlike gap; no "
        "cluster-quality metric, unlike silhouette) of the same underlying "
        "dispersion values gap statistic already computes."
    )
    lines.append("")

    lines.append("## Stage 1 - variety clusters")
    lines.append("")
    lines.append("| Stratum | n baskets | gap-selected k | silhouette-preferred k | elbow k |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, sub in strata.items():
        _, gap_k, sil_k, elbow_k = variety_results[label]
        lines.append(f"| {label} | {len(sub):,} | {gap_k} | {sil_k} | {elbow_k} |")
    lines.append("")
    lines.append(
        "**Compare to the original all-pooled, full-population run: gap "
        "selected k=1.** for the interpretation of what a low/high k means "
        "here (structureless vs. genuinely separated), see `docs/k-selection.md`."
    )
    lines.append("")

    lines.append("## Stage 2 - mission clusters (Stage 1 fixed at k=3 per stratum)")
    lines.append("")
    lines.append("| Stratum | relatively-concrete n | gap-selected k | silhouette-preferred k | elbow k |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in strata:
        result = mission_results[label]
        if result[0] is None:
            lines.append(f"| {label} | too small | - | - | - |")
            continue
        _, gap_k, sil_k, elbow_k, seg, n_rel = result
        lines.append(f"| {label} | {n_rel:,} | {gap_k} | {sil_k} | {elbow_k} |")
    lines.append("")
    lines.append(
        "**Compare to the original all-pooled run (with single-category "
        "baskets still in the Stage-1 input): gap and silhouette both "
        "selected k=2.** If format-stratified strata here select a higher, "
        "more paper-like k, that confirms format-pooling (not the "
        "clustering method) was the main cause of the earlier flat/weak "
        "result; if they still come back low, the explanation shifts "
        "toward the 3-day window or residual category-taxonomy noise - see "
        "`docs/k-selection.md` for the full list of candidate explanations."
    )
    lines.append("")

    lines.append("## Raw values")
    lines.append("")
    for label in strata:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("**Stage 1 (variety)**")
        lines.append("")
        gdf = variety_results[label][0]
        lines.append(_df_to_markdown(gdf, ["k", "log_Wk", "gap", "sk", "gap_rule_satisfied", "silhouette"]))
        lines.append("")
        if mission_results[label][0] is not None:
            lines.append("**Stage 2 (missions)**")
            lines.append("")
            gdf2 = mission_results[label][0]
            lines.append(_df_to_markdown(gdf2, ["k", "log_Wk", "gap", "sk", "gap_rule_satisfied", "silhouette"]))
            lines.append("")

    lines.append(
        "Reproduce with: `PYTHONPATH=src python3 scripts/select_k_by_format.py` "
        "(requires `outputs/clustering_dataset.parquet` - run "
        "`scripts/build_clustering_dataset.py` first if missing)."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))


def _df_to_markdown(df, cols):
    df = df[cols].copy()
    for col in ["log_Wk", "gap", "sk", "silhouette"]:
        if col in df.columns:
            df[col] = df[col].map(lambda v: f"{v:.4f}" if v == v else "n/a")
    return df.to_markdown(index=False)


if __name__ == "__main__":
    main()
