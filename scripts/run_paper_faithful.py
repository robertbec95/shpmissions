"""Faithful reproduction of Sarantopoulos et al. (2015)'s two-stage method,
respecting each step as specified - including two elements every other
scenario/report in this project has deliberately deviated from, for
empirical-quality reasons unrelated to faithfulness:

  1. Stage 1 (variety-based k=3 k-means segmentation into concrete/
     relatively-concrete/abstract) actually GATES which baskets reach
     Stage 2 here - only "relatively concrete" baskets are clustered,
     exactly as the paper describes ("the results for the baskets of
     concrete and abstract shopping goals do not produce any interpretable
     outcomes"). Every scenario in scenarios.py bypasses this (variety>=2
     directly, no Stage-1 filter) because that was found to produce
     better-separated clusters - see scenarios.py's module docstring - but
     that was a deliberate choice for a different goal (best empirical
     result), not a faithful reproduction. This script exists to answer a
     different question: what does the paper's own procedure find here.
  2. A 60%/40% train/test split, applied to Stage 2 clustering (the paper:
     "training and testing sets with 60% and 40% of observations can be
     selected, and the gap statistic procedure derives the optimal number
     of clusters" - stated once but framed as applying "similar to the
     shopping goal segmentation phases", i.e. to both stages). k-selection
     (gap statistic) and the k-medoids centroid fit both happen on the
     TRAINING 60% only; the full relatively-concrete population (train+test)
     is then assigned to those centroids, and train-vs-test cluster-share
     agreement is reported as the paper's implied validation check - no
     scenario elsewhere in this project has ever used a train/test split;
     CLARA's sample-then-assign-to-everyone pattern is a different thing
     (a computational shortcut, not a held-out validation set).

Also faithfully unweighted (raw binary category presence/absence, no
category weighting - "does not weigh the variables, because the product
taxonomy preparation ensures that all derived product categories are at the
same level of detail"), Jaccard distance, k-centroid (k-medoids/PAM) fit -
matching the paper's R `kcca` (flexclust package) call in spirit,
implemented in pure Python since R/flexclust isn't available here.

Acknowledged, unavoidable data-context difference: the paper's method is
per-store (a single pilot store); this dataset is a multi-store chain, so
this reproduction runs chain-wide rather than fitting Stage 1/2 separately
per store - matching every other scenario in this project, not a new
deviation introduced here. Category granularity also differs (75 derived
categories in the paper vs ~99 at this project's level-3 taxonomy) - a
property of each retailer's own product hierarchy, not a modeling choice.

As a bonus check on point 4 of the paper's description (concrete/abstract
segments don't yield interpretable results), this script also runs the
same gap-statistic-selected Stage 2 fit on the concrete and abstract
segments and reports whether that holds here too.

Usage:
    PYTHONPATH=src python3 scripts/run_paper_faithful.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from shopping_missions import mission_stage, multiday, profiling, store_formats, temporal, variety_stage
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE
from shopping_missions.scenarios import DATES_3DAY

DOCS_DIR = PROJECT_ROOT / "docs" / "paper_faithful"
SCENARIO_NAME = "level3_paper_faithful"
CATEGORY_LEVEL = 3
TRAIN_FRACTION = 0.6
K_SELECT_SAMPLE = 1500
CLARA_SAMPLE_SIZE = 6000


def main():
    print(f"=== {SCENARIO_NAME}: faithful reproduction of Sarantopoulos et al. (2015) ===")
    t0 = time.time()

    data = multiday.build_basket_data_multi_date(
        list(DATES_3DAY), category_level=CATEGORY_LEVEL, include_daypart=True,
    )
    vocab = data.vocab
    print(f"  ingested: {len(data.baskets):,} baskets (ALL varieties, no pre-filter), "
          f"{vocab.size} categories ({time.time()-t0:.1f}s)")

    # ---- Stage 1: variety segmentation (k=3 k-means, gates Stage 2 input) ----
    t1 = time.time()
    seg = variety_stage.fit_variety_segments(data.baskets, random_state=RANDOM_STATE)
    print(f"  Stage 1 ({time.time()-t1:.1f}s): bounds={seg.bounds}")
    print(seg.summary.to_string())

    stage1_gap_df = variety_stage.select_k(data.baskets["variety"].to_numpy(), k_range=range(1, 8), random_state=RANDOM_STATE)
    stage1_gap_k = best_k_by_gap(stage1_gap_df)
    print(f"  Stage 1 gap-statistic check: gap_k={stage1_gap_k} (fixed k=3 used regardless, matching the "
          f"paper's three-band framing - this just checks whether 3 is well-supported here too)")

    baskets = store_formats.join_county(seg.baskets)
    baskets = temporal.add_daypart(baskets)
    matrix = data.matrix

    rel_mask = (baskets["segment"] == "relatively_concrete").to_numpy()
    rel_baskets = baskets.loc[rel_mask].reset_index(drop=True)
    rel_matrix = matrix[rel_mask]
    n_rel = len(rel_baskets)
    print(f"  {n_rel:,} relatively-concrete baskets carried into Stage 2 "
          f"(variety {seg.bounds['relatively_concrete'][0]}-{seg.bounds['relatively_concrete'][1]})")

    # ---- 60/40 train/test split ----
    all_idx = np.arange(n_rel)
    train_idx, test_idx = train_test_split(all_idx, train_size=TRAIN_FRACTION, random_state=RANDOM_STATE)
    train_matrix = rel_matrix[train_idx]
    print(f"  train/test split: {len(train_idx):,} train ({TRAIN_FRACTION*100:.0f}%), "
          f"{len(test_idx):,} test ({(1-TRAIN_FRACTION)*100:.0f}%)")

    # ---- Stage 2 k-selection on TRAINING set only ----
    t2 = time.time()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(train_matrix.shape[0], size=min(K_SELECT_SAMPLE, train_matrix.shape[0]), replace=False)
    X_sample = np.asarray(train_matrix[sample_idx].todense(), dtype=np.float64)
    gdf = mission_stage.select_k(X_sample, k_range=range(2, 12), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    elbow_k = elbow_k_from_gap_df(gdf)
    print(f"  Stage 2 k-selection on training set ({time.time()-t2:.1f}s): "
          f"gap_k={gap_k} (decision), elbow_k={elbow_k}, silhouette_k={sil_k} (comparison only)")

    # ---- Stage 2 fit on TRAINING set, assign full relatively-concrete population ----
    t3 = time.time()
    clustering = mission_stage.fit_and_assign_missions(
        train_matrix, k=gap_k, sample_size=min(CLARA_SAMPLE_SIZE, train_matrix.shape[0]), random_state=RANDOM_STATE,
    )
    full_labels, full_distances = mission_stage._assign_to_medoids(rel_matrix, clustering.medoid_vectors)
    print(f"  Stage 2 fit on train, assigned to full relatively-concrete population ({time.time()-t3:.1f}s)")

    profiles = profiling.profile_clusters(rel_matrix, full_labels, vocab, rel_baskets)
    for p in profiles:
        top = ", ".join(c["category"] for c in p.top_categories[:4])
        print(f"  cluster {p.cluster_id}: {p.suggested_label} (conf={p.label_confidence}) "
              f"n={p.n_baskets:,} ({p.share_of_relatively_concrete}%) top=[{top}]")

    # ---- Train vs test validation ----
    is_train = np.zeros(n_rel, dtype=bool)
    is_train[train_idx] = True
    train_test_shares = _train_test_comparison(full_labels, is_train, gap_k)

    # ---- Cross-validation: county, day-part (never fed into clustering) ----
    county_xtab = _crosstab(rel_baskets, full_labels, "county")
    daypart_xtab = _crosstab(rel_baskets, full_labels, "day_part")

    # ---- Bonus: confirm concrete/abstract don't yield interpretable results ----
    bonus_results = {}
    for segment_name in ["concrete", "abstract"]:
        bonus_results[segment_name] = _run_segment_check(baskets, matrix, vocab, segment_name)

    _write_report(
        n_rel, vocab, seg, stage1_gap_df, stage1_gap_k, gdf, gap_k, elbow_k, sil_k,
        len(train_idx), len(test_idx), profiles, train_test_shares, county_xtab, daypart_xtab,
        bonus_results, time.time() - t0,
    )
    print(f"\nWrote {DOCS_DIR / (SCENARIO_NAME + '.md')}")


def _run_segment_check(baskets, matrix, vocab, segment_name):
    """Lightweight Stage 2 fit on the concrete or abstract segment (no
    train/test split needed - this is a confirmatory check, not the primary
    reported result), to test the paper's claim that these two segments
    don't produce interpretable missions."""
    mask = (baskets["segment"] == segment_name).to_numpy()
    seg_baskets = baskets.loc[mask].reset_index(drop=True)
    seg_matrix = matrix[mask]
    n = len(seg_baskets)
    if n < 50:
        return {"n": n, "gap_k": None, "profiles": [], "note": "too few baskets to cluster"}

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(n, size=min(K_SELECT_SAMPLE, n), replace=False)
    X_sample = np.asarray(seg_matrix[sample_idx].todense(), dtype=np.float64)
    gdf = mission_stage.select_k(X_sample, k_range=range(2, 12), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)

    clustering = mission_stage.fit_and_assign_missions(
        seg_matrix, k=gap_k, sample_size=min(CLARA_SAMPLE_SIZE, n), random_state=RANDOM_STATE,
    )
    profiles = profiling.profile_clusters(seg_matrix, clustering.labels, vocab, seg_baskets)
    n_uninterpretable = sum(1 for p in profiles if not p.top_categories)
    return {"n": n, "gap_k": gap_k, "profiles": profiles, "n_uninterpretable": n_uninterpretable}


def _train_test_comparison(labels, is_train, k):
    rows = []
    train_labels = labels[is_train]
    test_labels = labels[~is_train]
    for c in range(k):
        train_share = float((train_labels == c).mean() * 100) if len(train_labels) else 0.0
        test_share = float((test_labels == c).mean() * 100) if len(test_labels) else 0.0
        rows.append({"cluster_id": c, "train_share": round(train_share, 2), "test_share": round(test_share, 2),
                      "abs_diff": round(abs(train_share - test_share), 2)})
    return rows


def _crosstab(baskets, labels, dim_col):
    df = baskets.copy()
    df["cluster"] = labels
    xtab = df.groupby([dim_col, "cluster"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_report(
    n_rel, vocab, seg, stage1_gap_df, stage1_gap_k, gdf, gap_k, elbow_k, sil_k,
    n_train, n_test, profiles, train_test_shares, county_xtab, daypart_xtab, bonus_results, elapsed,
):
    lines = []
    lines.append("# Faithful reproduction: Sarantopoulos et al. (2015)")
    lines.append("")
    lines.append(
        f"Every step reproduced as specified: Stage 1 variety segmentation actually gates Stage 2's input "
        f"(only relatively-concrete baskets), a 60/40 train/test split is used for Stage 2's k-selection and "
        f"fit, gap statistic derives k on the training set, Jaccard distance + k-centroid (k-medoids/PAM), "
        f"unweighted binary category presence. Dates: {','.join(DATES_3DAY)} (3-day window), category level 3 "
        f"({vocab.size} categories - the paper's own taxonomy derived 75; this is a property of each "
        f"retailer's product hierarchy, not a modeling choice). Multi-store chain, run chain-wide (not "
        f"per-store, matching every other scenario in this project - the paper's case study was one pilot "
        f"store). Run in {elapsed:.1f}s total."
    )
    lines.append("")

    lines.append("## Stage 1: variety segmentation")
    lines.append("")
    lines.append(seg.summary.round(2).to_markdown())
    lines.append("")
    lines.append(f"Bounds: `{seg.bounds}`")
    lines.append("")
    lines.append(
        f"Gap-statistic check on Stage 1's own k (fixed at k=3 regardless, matching the paper's three-band "
        f"framing): gap_k={stage1_gap_k}."
    )
    lines.append("")
    lines.append(stage1_gap_df[["k", "gap", "sk", "gap_rule_satisfied", "silhouette"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## Stage 2: k-selection (gap statistic, on the 60% training set)")
    lines.append("")
    lines.append(
        f"{n_rel:,} relatively-concrete baskets -> {n_train:,} train (60%) / {n_test:,} test (40%). "
        f"gap_k={gap_k} (decision method), elbow_k={elbow_k}, silhouette_k={sil_k} (comparison only, per "
        f"this project's established convention)."
    )
    lines.append("")
    lines.append(gdf[["k", "gap", "sk", "gap_rule_satisfied", "silhouette"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append(f"## Stage 2: mission clusters (k={gap_k}, fit on train, assigned to full relatively-concrete population)")
    lines.append("")
    lines.append("| Cluster | Label | Share | Avg value | Top categories (lift) |")
    lines.append("|---|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p.top_categories[:5])
        lines.append(f"| {p.cluster_id} | {p.suggested_label} (conf {p.label_confidence}) | {p.share_of_relatively_concrete}% | {p.avg_basket_value} | {top} |")
    lines.append("")

    lines.append("## Train vs. test validation")
    lines.append("")
    lines.append(
        "Cluster shares computed separately within the 60% training baskets and the 40% test baskets "
        "(both assigned to the same train-fit centroids) - close agreement means the clusters found on "
        "training data generalize to held-out baskets, not just noise in that particular 60%."
    )
    lines.append("")
    lines.append("| Cluster | Train share | Test share | Abs. diff |")
    lines.append("|---|---|---|---|")
    for row in train_test_shares:
        lines.append(f"| {row['cluster_id']} | {row['train_share']}% | {row['test_share']}% | {row['abs_diff']} pp |")
    lines.append("")

    lines.append("## Cross-validation by day-part")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    pivot = daypart_xtab.pivot(index="day_part", columns="cluster", values="share_within_dim").fillna(0)
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    cluster_labels = {p.cluster_id: p.suggested_label for p in profiles}
    pivot.columns = [f"C{c} ({cluster_labels.get(c,'?')})" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")

    lines.append("## Cross-validation by county (top 10 by volume)")
    lines.append("")
    top_counties = county_xtab.groupby("county", observed=True)["n"].sum().sort_values(ascending=False).head(10).index
    pivot_c = county_xtab[county_xtab["county"].isin(top_counties)].pivot(index="county", columns="cluster", values="share_within_dim").fillna(0)
    pivot_c = pivot_c.loc[top_counties]
    pivot_c.columns = [f"C{c} ({cluster_labels.get(c,'?')})" for c in pivot_c.columns]
    lines.append(pivot_c.round(1).to_markdown())
    lines.append("")

    lines.append("## Bonus: does concrete/abstract fail to produce interpretable missions here too?")
    lines.append("")
    lines.append(
        "The paper: \"the results for the baskets of 'concrete' and 'abstract' goals do not produce any "
        "interpretable outcomes.\" Same gap-statistic-selected Stage 2 fit applied to each segment on its own "
        "(no train/test split - this is a confirmatory check, not the primary result)."
    )
    lines.append("")
    for segment_name, result in bonus_results.items():
        lines.append(f"### {segment_name.capitalize()} ({result['n']:,} baskets)")
        lines.append("")
        if result["gap_k"] is None:
            lines.append(f"_{result['note']}_")
            lines.append("")
            continue
        lines.append(f"gap_k={result['gap_k']}, {result['n_uninterpretable']} of {len(result['profiles'])} clusters have no category clearing the lift/support threshold (uninterpretable).")
        lines.append("")
        lines.append("| Cluster | Label | Share | Top categories (lift) |")
        lines.append("|---|---|---|---|")
        for p in result["profiles"]:
            top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p.top_categories[:4]) or "_none clear threshold_"
            lines.append(f"| {p.cluster_id} | {p.suggested_label} (conf {p.label_confidence}) | {p.share_of_relatively_concrete}% | {top} |")
        lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{SCENARIO_NAME}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
