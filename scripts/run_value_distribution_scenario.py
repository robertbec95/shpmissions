"""Formal run of the value-distribution (Bray-Curtis) clustering approach:
Stage 2 on each basket's normalized *spend distribution* across categories
(what fraction of the basket's RON value went to each category) instead of
category presence/absence (Jaccard). Same population as the frozen baseline
(`level3_baseline_gapk`) so results are directly comparable - variety>=2,
category level 3, 3-day window - with one additional filter: baskets with
basket_value<=0 are dropped (924 of 2,364,042, ~0.04%, all exactly 0 -
promotional/data edge cases with no meaningful spend distribution to cluster
on).

Runs k-selection (gap statistic + elbow + silhouette), then profiles the
full clustering at BOTH the gap-statistic k and a second k chosen from
elbow/silhouette convergence when the gap rule is satisfied almost
everywhere (a weak-signal pattern already seen with level-4 categories -
see docs/scenarios/level4.md) - exactly the same "report the mechanical
answer, but don't pretend it's the only honest one" approach used
throughout this project. Then stability-tests both k values (20 seeds each)
to check whether Bray-Curtis's much sharper category lifts are also more
reproducible, or just a better fit to whichever CLARA sample got drawn.

Usage:
    PYTHONPATH=src python3 scripts/run_value_distribution_scenario.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import mission_stage, multiday, stability, store_formats, temporal
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE
from shopping_missions.scenarios import get_scenario

DOCS_SCENARIOS_DIR = PROJECT_ROOT / "docs" / "scenarios"
DOCS_STABILITY_DIR = PROJECT_ROOT / "docs" / "stability"
SCENARIO_NAME = "level3_value_distribution"
FEATURED_K = 10  # elbow-selected, matches the silhouette peak region (11) without k=11's early fragmentation signs
N_STABILITY_RUNS = 20


def main():
    scenario = get_scenario("level3_baseline_gapk")
    print(f"=== {SCENARIO_NAME}: Bray-Curtis on spend distribution ===")
    t0 = time.time()

    data = multiday.build_basket_data_multi_date(
        list(scenario.dates), category_level=scenario.category_level, include_daypart=True,
        include_value_matrix=True,
    )
    keep_mask = ((data.baskets["variety"] >= scenario.min_variety) & (data.baskets["basket_value"] > 0)).to_numpy()
    baskets = data.baskets.loc[keep_mask].reset_index(drop=True)
    value_matrix = data.value_matrix[keep_mask]
    binary_matrix = data.matrix[keep_mask]
    vocab = data.vocab
    n = len(baskets)
    n_dropped_zero_value = int((data.baskets["variety"] >= scenario.min_variety).sum()) - n
    print(f"  ingested: {n:,} baskets (variety>={scenario.min_variety}, basket_value>0; "
          f"{n_dropped_zero_value} zero-value baskets dropped) ({time.time()-t0:.1f}s)")

    norm = mission_stage.normalize_value_matrix(value_matrix)

    t1 = time.time()
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n, size=min(1500, n), replace=False)
    X_sample = np.asarray(norm[idx].todense(), dtype=np.float64)
    gdf = mission_stage.select_k_bray_curtis(X_sample, k_range=range(2, 21), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    elbow_k = elbow_k_from_gap_df(gdf)
    print(f"  k-selection ({time.time()-t1:.1f}s): gap_k={gap_k} sil_k={sil_k} elbow_k={elbow_k}")
    print(f"  featured k={FEATURED_K} (elbow-selected; gap rule is satisfied at nearly every k here, "
          f"a weak-signal pattern, not a strong preference for gap_k={gap_k})")

    baskets = store_formats.join_county(baskets)
    baskets = temporal.add_daypart(baskets)

    profiles_by_k = {}
    for k in sorted({gap_k, FEATURED_K}):
        t2 = time.time()
        clustering = mission_stage.fit_and_assign_missions_bray_curtis(
            norm, k=k, sample_size=scenario.sample_size, random_state=RANDOM_STATE
        )
        profiles = _profile_clusters(baskets, value_matrix, binary_matrix, clustering.labels, vocab, k)
        profiles_by_k[k] = {
            "profiles": profiles,
            "county_xtab": _crosstab(baskets, clustering.labels, "county", k),
            "daypart_xtab": _crosstab(baskets, clustering.labels, "day_part", k),
        }
        print(f"  fit+profile k={k} ({time.time()-t2:.1f}s)")
        for p in profiles:
            top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p["top_categories"][:3])
            print(f"    c{p['cluster_id']}: n={p['n_baskets']:,} ({p['share_pct']}%) top=[{top}]")

    stability_by_k = {}
    for k in sorted({gap_k, FEATURED_K}):
        t3 = time.time()
        result = stability.run_stability_test_bray_curtis(
            norm, k=k, sample_size=scenario.sample_size, n_runs=N_STABILITY_RUNS, base_seed=RANDOM_STATE
        )
        stability_by_k[k] = result
        ari, nmi = result.off_diagonal_ari, result.off_diagonal_nmi
        print(f"  stability k={k} ({N_STABILITY_RUNS} runs, {time.time()-t3:.1f}s): "
              f"ARI mean={ari.mean():.4f} std={ari.std():.4f} | NMI mean={nmi.mean():.4f} std={nmi.std():.4f}")

    _write_scenario_report(n, n_dropped_zero_value, vocab, gdf, gap_k, sil_k, elbow_k, profiles_by_k, time.time() - t0)
    _write_stability_report(stability_by_k)
    print(f"\nWrote {DOCS_SCENARIOS_DIR / (SCENARIO_NAME + '.md')} and "
          f"{DOCS_STABILITY_DIR / (SCENARIO_NAME + '.md')}")


def _profile_clusters(baskets, value_matrix, binary_matrix, labels, vocab, k):
    n = len(labels)
    overall_value = np.asarray(value_matrix.sum(axis=0)).ravel()
    overall_value_share = overall_value / overall_value.sum()

    profiles = []
    for c in range(k):
        mask = labels == c
        nc = int(mask.sum())
        if nc == 0:
            profiles.append({"cluster_id": c, "n_baskets": 0, "share_pct": 0.0, "top_categories": [],
                              "avg_value": None, "avg_variety": None, "avg_n_categories_touched": None})
            continue
        sub_val = value_matrix[mask]
        cshare = np.asarray(sub_val.sum(axis=0)).ravel()
        cshare = cshare / cshare.sum() if cshare.sum() > 0 else cshare
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_value_share > 0, cshare / overall_value_share, 0.0)
        order = np.argsort(-cshare)[:8]
        top_categories = [
            {"category": vocab.names[vocab.ids[i]], "value_share": round(float(cshare[i]), 4), "lift": round(float(lift[i]), 3)}
            for i in order
        ]
        profiles.append({
            "cluster_id": c,
            "n_baskets": nc,
            "share_pct": round(nc / n * 100, 2),
            "avg_value": round(float(baskets.loc[mask, "basket_value"].mean()), 2),
            "avg_variety": round(float(baskets.loc[mask, "variety"].mean()), 2),
            "top_categories": top_categories,
        })
    return profiles


def _crosstab(baskets, labels, dim_col, k):
    df = baskets.copy()
    df["cluster"] = labels
    xtab = df.groupby([dim_col, "cluster"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_scenario_report(n, n_dropped_zero_value, vocab, gdf, gap_k, sil_k, elbow_k, profiles_by_k, elapsed):
    lines = []
    lines.append(f"# Scenario: {SCENARIO_NAME}")
    lines.append("")
    lines.append(
        "Stage 2 clustering on each basket's *spend distribution* across categories (Bray-Curtis "
        "dissimilarity on row-normalized RON value per category) instead of category presence/absence "
        "(Jaccard on the binary matrix, used everywhere else in this project). Same population as the "
        f"frozen baseline `level3_baseline_gapk` (variety>=2, category level 3, 3-day window), minus "
        f"{n_dropped_zero_value} baskets with basket_value<=0 (all exactly 0 - promotional/data edge "
        "cases with no spend distribution to speak of)."
    )
    lines.append("")
    lines.append(f"**Result:** {n:,} baskets, {vocab.size} categories, run in {elapsed:.1f}s.")
    lines.append(
        f" k-selection: gap_k={gap_k}, silhouette_k={sil_k}, elbow_k={elbow_k}. The gap rule is satisfied "
        f"at nearly every k from 4 upward here (only k=3 fails it) - the same weak-signal pattern flagged "
        f"for level-4 categories (docs/scenarios/level4.md), where the mechanical \"smallest satisfying k\" "
        f"answer doesn't reflect a strong preference. Silhouette has a genuine peak (not just monotonic "
        f"drift) around k=10-11 - 0.166-0.174, 2-3x higher than any silhouette value seen anywhere in this "
        f"project under Jaccard (ceiling ~0.07). Both k={gap_k} (gap statistic, mechanical) and "
        f"k={FEATURED_K} (elbow; the featured result - see below) are profiled in full."
    )
    lines.append("")
    lines.append("## k-selection curve")
    lines.append("")
    lines.append(gdf[["k", "gap", "sk", "gap_rule_satisfied", "silhouette"]].round(4).to_markdown(index=False))
    lines.append("")

    for k in sorted(profiles_by_k.keys()):
        featured = " (FEATURED)" if k == FEATURED_K else " (gap statistic, mechanical answer)" if k == gap_k else ""
        lines.append(f"## Cluster profiles at k={k}{featured}")
        lines.append("")
        lines.append("| Cluster | Share | Avg value | Avg variety | Top categories by value share (lift) |")
        lines.append("|---|---|---|---|---|")
        for p in profiles_by_k[k]["profiles"]:
            top = ", ".join(f"{c['category']} ({c['value_share']*100:.0f}%, x{c['lift']:.1f})" for c in p["top_categories"][:5])
            lines.append(f"| {p['cluster_id']} | {p['share_pct']}% | {p['avg_value']} | {p['avg_variety']} | {top} |")
        lines.append("")

        lines.append(f"### k={k}: cluster mix by day-part")
        lines.append("")
        order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening", "Unknown"]
        pivot = profiles_by_k[k]["daypart_xtab"].pivot(index="day_part", columns="cluster", values="share_within_dim").fillna(0)
        pivot = pivot.reindex([o for o in order if o in pivot.index])
        pivot.columns = [f"C{c}" for c in pivot.columns]
        lines.append(pivot.round(1).to_markdown())
        lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_distribution_scenario.py`")

    DOCS_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_SCENARIOS_DIR / f"{SCENARIO_NAME}.md").write_text("\n".join(lines))


def _write_stability_report(stability_by_k):
    lines = []
    lines.append(f"# Stability test: {SCENARIO_NAME}")
    lines.append("")
    lines.append(
        f"{N_STABILITY_RUNS} independent Bray-Curtis CLARA runs per k, same population and methodology "
        "as `docs/stability/level3_baseline.md` (which tests the binary/Jaccard baseline) - only the "
        "distance metric and input matrix differ (spend-distribution Bray-Curtis vs. category-presence "
        "Jaccard). Tests whether Bray-Curtis's much sharper per-run category lifts (see the scenario "
        "report) are also more reproducible run-to-run, or just a better fit to whichever CLARA sample "
        "happened to get drawn."
    )
    lines.append("")
    lines.append("| k | ARI mean | ARI std | NMI mean | NMI std |")
    lines.append("|---|---|---|---|---|")
    for k in sorted(stability_by_k.keys()):
        result = stability_by_k[k]
        ari, nmi = result.off_diagonal_ari, result.off_diagonal_nmi
        lines.append(f"| {k} | {ari.mean():.4f} | {ari.std():.4f} | {nmi.mean():.4f} | {nmi.std():.4f} |")
    lines.append("")
    lines.append(
        "For reference: binary Jaccard CLARA (k=6, `docs/stability/level3_baseline.md`) scores ARI "
        "mean=0.2338, NMI mean=0.2985 on this same population. Graph community detection (a different, "
        "non-CLARA method, `docs/graph_missions/level3_baseline.md`) scores ARI mean=0.5565 - still the "
        "most stable method tested in this project."
    )
    lines.append("")
    lines.append(
        "Rule of thumb (same as other stability reports): ARI/NMI > 0.9 = highly stable, 0.7-0.9 = "
        "reasonably stable, 0.5-0.7 = moderate instability worth investigating, < 0.5 = largely "
        "re-deriving noise from sample to sample."
    )
    lines.append("")
    lines.append("## Cluster size variability across runs (% share of baskets)")
    for k in sorted(stability_by_k.keys()):
        lines.append("")
        lines.append(f"### k={k}")
        lines.append("")
        size_stats = stability_by_k[k].cluster_size_shares.agg(["mean", "std", "min", "max"]).T
        lines.append(size_stats.round(2).to_markdown())
    lines.append("")
    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_distribution_scenario.py`")

    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / f"{SCENARIO_NAME}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
