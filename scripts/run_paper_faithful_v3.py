"""Faithful reproduction of Sarantopoulos et al. (2015), v3: closes two gaps
the v2 script (run_paper_faithful_v2.py) still had relative to a literal
reading of the paper, on the same 2025 per-store dataset.

  1. Stage 1's k is now chosen by the gap statistic, not fixed at 3. If the
     gap statistic selects k>=3 raw clusters, they're grouped into the
     paper's concrete/relatively-concrete/abstract bands by splitting the
     clusters (sorted by ascending mean variety) into 3 contiguous groups.
     If gap_k<3 (as it turns out to be, at every single store here - see
     the "Stage 1: what the gap statistic actually says" section of the
     combined report), there aren't enough raw clusters to band, and the
     fallback is the fixed k=3 fit - flagged explicitly, not silently
     substituted. See `variety_stage.fit_variety_segments_gap_k`.
  2. Stage 2's k-medoids fit now uses multiple initializations (n_init=10
     independent k-medoids++ restarts on the same 60%-training CLARA
     sample, keeping the lowest-dispersion result) instead of a single
     initialization - mirroring scikit-learn's `KMeans(n_init=...)` pattern
     and addressing part of the instability this project's stability
     investigation (docs/stability/*.md) attributed to CLARA's single-shot
     approximation. See `mission_stage.fit_and_assign_missions_multi_init`.

Everything else matches v2 and the paper's description: sales data
already consolidated into one category per SKU by the source export (see
`ingest_v2.py`), unweighted binary category presence for Stage 2 (no
value-weighting - that is a *separate*, already-published deviation, not
part of this baseline), Jaccard distance + k-centroid/k-medoids, 60/40
train/test split, gap statistic (training set only) selecting k, category
participation (support) + lift for profiling, descriptive names built
directly from each mission's most distinctive categories.

Usage:
    PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from shopping_missions import categories, ingest_v2, mission_stage, profiling, temporal, variety_stage
from shopping_missions import baskets as baskets_mod
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "paper_faithful"
TRAIN_FRACTION = 0.6
K_SELECT_SAMPLE = 1500
CLARA_SAMPLE_SIZE = 6000
N_INIT = 10
STAGE1_GAP_SAMPLE = 15000  # sklearn's default n_init=5-per-fit KMeans got prohibitively slow in this
STAGE1_GAP_B = 10          # environment at the select_k default of sample_size=50_000 (many minutes per
                            # store) - 15,000 was confirmed to give the same gap_k as 50,000 (see report)
MIN_SUPPORT = profiling.MIN_SUPPORT  # 0.15 - category must appear in >=15% of a cluster's baskets to "participate"
MIN_LIFT = profiling.MIN_LIFT        # 1.2 - and be >=20% more common in-cluster than baseline
DEGENERATE_THRESHOLD = 0.90          # a single category at >=90% support flags the cluster as near-single-category


def main():
    print("=== Paper-faithful reproduction v3, per store (2025 data) ===")
    print(f"Thresholds: category participation support>={MIN_SUPPORT}, lift>={MIN_LIFT}; "
          f"degenerate flag at single-category support>={DEGENERATE_THRESHOLD}; "
          f"train/test={TRAIN_FRACTION}/{1-TRAIN_FRACTION}; Stage-2 n_init={N_INIT}")
    t_all = time.time()

    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)
    print(f"Global category vocabulary: {vocab.size} categories (fixed across all stores, "
          f"excluding {sorted(ingest_v2.EXCLUDE_CATEGORY_NAMES)} as non-product/administrative)")

    store_results = {}
    for loc_id, info in stores.items():
        print(f"\n--- {info['format']} / {info['county']} ({loc_id}) ---")
        store_results[loc_id] = run_for_store(loc_id, info, vocab, vocab_df)

    _write_combined_report(stores, store_results, vocab, time.time() - t_all)
    print(f"\nWrote {DOCS_DIR / 'level3_paper_faithful_2025_v3_summary.md'} and one report per store.")


def run_for_store(loc_id, info, vocab, vocab_df):
    t0 = time.time()
    con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
    data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True)
    con.close()
    baskets_all = temporal.add_daypart(data.baskets)
    matrix_all = data.matrix
    n_total = len(baskets_all)
    print(f"  ingested: {n_total:,} baskets ({time.time()-t0:.1f}s)")

    # ---- Stage 1: variety segmentation, k chosen by gap statistic ----
    t1 = time.time()
    seg = variety_stage.fit_variety_segments_gap_k(
        baskets_all, k_range=range(1, 8), B=STAGE1_GAP_B, sample_size=STAGE1_GAP_SAMPLE, random_state=RANDOM_STATE,
    )
    print(f"  Stage 1 ({time.time()-t1:.1f}s): gap_k={seg.gap_k}"
          + (" -> gap_k<3, fell back to fixed k=3" if seg.fallback_to_fixed_k3 else f" -> banded into 3 groups: {seg.cluster_to_segment}"))
    print(f"    bounds={seg.bounds}")
    baskets_all = seg.baskets

    rel_mask = (baskets_all["segment"] == "relatively_concrete").to_numpy()
    rel_baskets = baskets_all.loc[rel_mask].reset_index(drop=True)
    rel_matrix = matrix_all[rel_mask]
    n_rel = len(rel_baskets)
    print(f"  {n_rel:,} relatively-concrete baskets ({n_rel/n_total*100:.1f}% of total)")

    if n_rel < 200:
        print(f"  too few relatively-concrete baskets ({n_rel}) to cluster meaningfully - skipping Stage 2")
        return {"info": info, "n_total": n_total, "seg": seg, "n_rel": n_rel, "skipped": True}

    # ---- 60/40 split ----
    all_idx = np.arange(n_rel)
    train_idx, test_idx = train_test_split(all_idx, train_size=TRAIN_FRACTION, random_state=RANDOM_STATE)
    train_matrix = rel_matrix[train_idx]

    # ---- Stage 2 k-selection on the training set ----
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(train_matrix.shape[0], size=min(K_SELECT_SAMPLE, train_matrix.shape[0]), replace=False)
    X_sample = np.asarray(train_matrix[sample_idx].todense(), dtype=np.float64)
    gdf = mission_stage.select_k(X_sample, k_range=range(2, 12), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    elbow_k = elbow_k_from_gap_df(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    print(f"  Stage 2 k-selection: gap_k={gap_k} (decision), elbow_k={elbow_k}, silhouette_k={sil_k} (comparison)")

    # ---- Stage 2 fit: multiple initializations on the training sample ----
    t2 = time.time()
    clustering = mission_stage.fit_and_assign_missions_multi_init(
        train_matrix, k=gap_k, sample_size=min(CLARA_SAMPLE_SIZE, train_matrix.shape[0]),
        n_init=N_INIT, random_state=RANDOM_STATE,
    )
    print(f"  Stage 2 fit ({N_INIT} inits, {time.time()-t2:.1f}s): best init seed={clustering.best_init_seed}, "
          f"dispersion range [{min(clustering.init_costs):.1f}, {max(clustering.init_costs):.1f}] "
          f"across {N_INIT} attempts")
    full_labels, _ = mission_stage._assign_to_medoids(rel_matrix, clustering.medoid_vectors)

    profiles = profiling.profile_clusters(rel_matrix, full_labels, vocab, rel_baskets)
    named_profiles = [_build_distinctive_name(p) for p in profiles]
    for p, name_info in zip(profiles, named_profiles):
        flag = " [WEAK/DEGENERATE]" if name_info["is_weak"] else ""
        print(f"    cluster {p.cluster_id}: \"{name_info['name']}\" n={p.n_baskets:,} "
              f"({p.share_of_relatively_concrete}%){flag}")

    is_train = np.zeros(n_rel, dtype=bool)
    is_train[train_idx] = True
    train_test_shares = _train_test_comparison(full_labels, is_train, gap_k)

    daypart_xtab = _crosstab(rel_baskets, full_labels, "day_part")

    _write_store_report(
        loc_id, info, n_total, seg, gdf, gap_k, elbow_k, sil_k, n_rel, len(train_idx), len(test_idx),
        clustering, profiles, named_profiles, train_test_shares, daypart_xtab, time.time() - t0,
    )

    return {
        "info": info, "n_total": n_total, "seg": seg, "n_rel": n_rel, "skipped": False,
        "gap_k": gap_k, "profiles": profiles, "named_profiles": named_profiles,
        "train_test": train_test_shares, "clustering": clustering,
    }


def _build_distinctive_name(profile) -> dict:
    """Descriptive name built directly from the mission's most distinctive
    categories (ranked by lift, same participation thresholds as
    `profiling.profile_clusters`), rather than the coarse fixed-vocabulary
    keyword labels (`profiling.LABEL_KEYWORDS`) used elsewhere in this
    project - a more literal reading of "assign descriptive names based on
    the most distinctive category combinations". Flags a mission as
    weak/degenerate if either no category clears the participation
    threshold at all, or one category alone accounts for
    `DEGENERATE_THRESHOLD` of the cluster's baskets (arguably a
    single-category split, not a genuine multi-category "mission")."""
    cats = profile.top_categories
    if not cats:
        return {"name": "Unlabeled (no category clears participation threshold)", "is_weak": True}
    name = " + ".join(c["category"].title() for c in cats[:3])
    is_degenerate = cats[0]["support"] >= DEGENERATE_THRESHOLD
    return {"name": name, "is_weak": is_degenerate}


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


def _write_store_report(loc_id, info, n_total, seg, gdf, gap_k, elbow_k, sil_k, n_rel, n_train, n_test,
                         clustering, profiles, named_profiles, train_test_shares, daypart_xtab, elapsed):
    name = f"level3_paper_faithful_2025_v3_{info['format'].lower().replace(' ', '_')}_{info['county'].lower()}"
    lines = []
    lines.append(f"# Faithful reproduction v3 (2025 data): {info['format']} — {info['county']}")
    lines.append("")
    lines.append(
        f"Store `{loc_id}`, {info['format']}, {info['county']}, full calendar-2025 purchase history "
        f"({n_total:,} baskets). Baseline method: unweighted binary category presence, Jaccard distance, "
        f"k-centroid (k-medoids), gap statistic for k at both stages, 60/40 train/test split, "
        f"{clustering.n_init} k-medoids initializations kept-best-by-dispersion. Run in {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Stage 1: variety segmentation")
    lines.append("")
    lines.append(
        f"Gap statistic on basket variety selected **gap_k={seg.gap_k}**."
        + (
            f" Since gap_k<3, there aren't enough raw clusters to form 3 non-empty concrete/"
            f"relatively-concrete/abstract bands - **fell back to the paper's fixed k=3 framing** "
            f"(`variety_stage.fit_variety_segments`), rather than silently pretending the gap "
            f"statistic chose 3."
            if seg.fallback_to_fixed_k3 else
            f" Raw clusters grouped into 3 bands by ascending mean variety (`numpy.array_split`): "
            f"{seg.cluster_to_segment}."
        )
    )
    lines.append("")
    lines.append(seg.summary.round(2).to_markdown())
    lines.append("")
    lines.append(f"Bounds: `{seg.bounds}`")
    lines.append("")

    lines.append("## Stage 2: k-selection (gap statistic, on the 60% training set)")
    lines.append("")
    lines.append(f"{n_rel:,} relatively-concrete baskets -> {n_train:,} train / {n_test:,} test. "
                  f"gap_k={gap_k} (decision), elbow_k={elbow_k}, silhouette_k={sil_k} (comparison only).")
    lines.append("")
    lines.append(gdf[["k", "gap", "sk", "gap_rule_satisfied", "silhouette"]].round(4).to_markdown(index=False))
    lines.append("")
    lines.append(
        f"Multiple initializations: {clustering.n_init} independent k-medoids++ restarts on the same "
        f"training CLARA sample, kept the lowest total within-cluster Jaccard dispersion. Dispersion "
        f"across attempts ranged {min(clustering.init_costs):.2f} to {max(clustering.init_costs):.2f} "
        f"(best: seed {clustering.best_init_seed}) - the spread is the direct evidence for why a single "
        f"initialization isn't reliable: different restarts converge to different local optima."
    )
    lines.append("")

    lines.append(f"## Mission clusters (k={gap_k})")
    lines.append("")
    lines.append("| Cluster | Name | Weak? | Share | Avg value | Category participation (support, lift) |")
    lines.append("|---|---|---|---|---|---|")
    for p, ninfo in zip(profiles, named_profiles):
        top = ", ".join(f"{c['category']} ({c['support']*100:.0f}%, x{c['lift']:.1f})" for c in p.top_categories)
        flag = "**yes**" if ninfo["is_weak"] else "no"
        lines.append(f"| {p.cluster_id} | {ninfo['name']} | {flag} | {p.share_of_relatively_concrete}% | {p.avg_basket_value} | {top or '_none clear threshold_'} |")
    lines.append("")

    lines.append("## Train vs. test validation")
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
    name_map = {p.cluster_id: named_profiles[i]["name"] for i, p in enumerate(profiles)}
    pivot.columns = [f"C{c}" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")
    for c, nm in name_map.items():
        lines.append(f"- C{c}: {nm}")
    lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{name}.md").write_text("\n".join(lines))


def _write_combined_report(stores, store_results, vocab, elapsed):
    lines = []
    lines.append("# Faithful reproduction v3 (2025 data): assumptions, results, limitations")
    lines.append("")
    lines.append(
        f"{len(stores)} Profi stores, full calendar-2025 purchase history each, Sarantopoulos et al. "
        f"(2015)'s method applied per store as literally as this project has managed - gap statistic "
        f"chosen k at both stages (not fixed), multiple k-medoids initializations at Stage 2, 60/40 "
        f"train/test split, unweighted binary category presence. Total run time {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Assumptions and thresholds")
    lines.append("")
    lines.append("| Decision | Value | Rationale |")
    lines.append("|---|---|---|")
    lines.append("| Basket definition | `transaction_id` | One checkout = one basket; verified globally unique across stores in this export |")
    lines.append("| SKU-to-category mapping | Direct from source (`category_name` per line) | Source data already carries one consistent category per SKU line - no separate remapping needed or performed |")
    lines.append(f"| Excluded categories | {sorted(ingest_v2.EXCLUDE_CATEGORY_NAMES)} | Non-product/administrative (services, marketing campaigns, catch-all \"other\") - matches this project's original-dataset exclusion list, by name since this export's category IDs are a different (hash) encoding |")
    lines.append("| Negative-quantity lines | Dropped (`quantity_unit >= 0`) | Defensive - none actually found in this export, unlike the original dataset's raw files |")
    lines.append("| Category weighting (Stage 2) | None - binary presence only | The paper's explicit baseline: \"does not weigh the variables\" |")
    lines.append("| Distance measure (Stage 2) | Unweighted Jaccard | Paper's stated measure |")
    lines.append(f"| Stage-2 k-medoids | k-centroid / PAM, {N_INIT} initializations, keep lowest dispersion | Custom Python implementation - no R/flexclust environment available; multi-init addresses this project's own instability findings |")
    lines.append("| Train/test split | 60% / 40%, fixed random_state | Paper's stated split |")
    lines.append("| k-selection | Gap statistic (Tibshirani et al. 2001), both stages | Paper's stated method; elbow + silhouette reported as comparison only, never the decision rule |")
    lines.append(f"| Category participation threshold | support >= {MIN_SUPPORT*100:.0f}%, lift >= {MIN_LIFT} | A category must appear in a meaningful share of the cluster's baskets *and* be more common than baseline to count toward a mission's profile |")
    lines.append(f"| Degenerate-cluster flag | one category at >= {DEGENERATE_THRESHOLD*100:.0f}% support | Flags a near-single-category split, arguably not a genuine multi-category \"mission\" |")
    lines.append("")

    lines.append("## Stage 1: what the gap statistic actually says")
    lines.append("")
    all_gap1 = all(store_results[loc_id]["seg"].gap_k < 3 for loc_id in stores if not store_results[loc_id].get("skipped"))
    lines.append(
        f"Applied literally (not assumed to be 3), the gap statistic selected "
        f"**gap_k={store_results[list(stores.keys())[0]]['seg'].gap_k} at every one of the {len(stores)} stores**"
        if all_gap1 else "Gap statistic results varied by store (see table below)."
    )
    lines.append("")
    lines.append("| Store | Format | gap_k (Stage 1) | Fell back to fixed k=3? |")
    lines.append("|---|---|---|---|")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        if r.get("skipped"):
            lines.append(f"| {info['county']} | {info['format']} | - | too few baskets |")
            continue
        lines.append(f"| {info['county']} | {info['format']} | {r['seg'].gap_k} | {'yes' if r['seg'].fallback_to_fixed_k3 else 'no'} |")
    lines.append("")
    lines.append(
        "**This is a real, honestly-reported limitation, not smoothed over**: basket variety follows a "
        "smoothly, roughly-monotonically decreasing distribution at every store (most baskets touch very "
        "few categories, progressively fewer baskets touch more) - exactly the shape that gives a formal "
        "gap-statistic decision of \"1 cluster is already sufficient,\" since there's no genuine multi-modal "
        "break in the distribution. The paper's concrete/relatively-concrete/abstract framing is retained "
        "here as a theoretically-motivated assumption (basket variety plausibly still tracks shopping-goal "
        "abstractness even without gap-statistic support for exactly 3 natural bands), not as something "
        "this dataset's variety distribution empirically discovers on its own."
    )
    lines.append("")

    lines.append("## Overview by store")
    lines.append("")
    lines.append("| Store | Format | Total baskets | Relatively-concrete | Stage-2 gap_k | Weak/degenerate clusters |")
    lines.append("|---|---|---|---|---|---|")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        if r.get("skipped"):
            lines.append(f"| {info['county']} | {info['format']} | {r['n_total']:,} | {r['n_rel']:,} | - | too few baskets |")
            continue
        n_weak = sum(1 for np_ in r["named_profiles"] if np_["is_weak"])
        lines.append(f"| {info['county']} | {info['format']} | {r['n_total']:,} | {r['n_rel']:,} | {r['gap_k']} | {n_weak} of {len(r['profiles'])} |")
    lines.append("")

    lines.append("## Full mission breakdown by store")
    lines.append("")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        lines.append(f"### {info['format']} — {info['county']}")
        lines.append("")
        if r.get("skipped"):
            lines.append("_Too few relatively-concrete baskets to cluster._")
            lines.append("")
            continue
        lines.append("| Cluster | Name | Weak? | Share | Category participation (support, lift) |")
        lines.append("|---|---|---|---|---|")
        for p, ninfo in zip(r["profiles"], r["named_profiles"]):
            top = ", ".join(f"{c['category']} ({c['support']*100:.0f}%, x{c['lift']:.1f})" for c in p.top_categories[:4])
            flag = "**yes**" if ninfo["is_weak"] else "no"
            lines.append(f"| {p.cluster_id} | {ninfo['name']} | {flag} | {p.share_of_relatively_concrete}% | {top or '_none_'} |")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Stage 1's gap statistic doesn't support the 3-band structure on its own** (see above) - the "
        "concrete/relatively-concrete/abstract split is retained as a theoretical assumption from the "
        "paper, not something this data's variety distribution independently confirms.\n"
        "- **Category granularity differs from the paper's own taxonomy** (paper: 75 derived categories; "
        "this export: 99, after excluding non-product categories) - a property of each retailer's product "
        "hierarchy, not a modeling choice, but it does mean category-level results aren't numerically "
        "comparable to the paper's reported figures.\n"
        "- **Store sizes vary substantially** (66,847 to 587,487 total baskets across the 6 stores), which "
        "affects the statistical power and reliability of gap statistic and cluster stability at the "
        "smaller stores - treat results from Brasov (smallest) with more caution than Neamt (largest).\n"
        "- **This reproduction is per-store**, matching the paper's own framing, but does not further "
        "control for factors the paper's single pilot store didn't need to (seasonality pooled across "
        "the full year, regional/demographic differences between counties, store-format-specific "
        "assortment differences) - these are folded into one full-year fit per store, not modeled "
        "separately.\n"
        "- **Multiple initializations only address one source of instability.** This project's stability "
        "investigation (`docs/stability/*.md`) found CLARA's single-*sample* variance (which baskets get "
        "drawn into the training CLARA sample) is at least as large a factor as single-*initialization* "
        "variance (which local optimum a given sample converges to) - this run fixes the latter, not the "
        "former. A full multi-sample CLARA (repeat the whole draw-sample-then-fit procedure, not just the "
        "k-medoids restart) was not implemented here."
    )
    lines.append("")

    lines.append("## Modifications, labeled separately from the paper's unweighted baseline")
    lines.append("")
    lines.append(
        "None of the following are part of this report's numbers - they are separate, already-published "
        "explorations elsewhere in this project, listed here only so they aren't confused with the "
        "baseline above:\n"
        "- **Value-weighted graph community detection** (spend-distribution based, no basket sampling) - "
        "this project's best-performing method by stability, but a deliberate deviation from the paper's "
        "unweighted method on every axis (what's clustered, algorithm, k-selection). Applied to this same "
        "2025 per-store data separately - see `docs/graph_missions/level3_value_distribution_graph_2025_summary.md`.\n"
        "- **Item2Vec-style category embeddings** - a further deviation layered on the graph method above, "
        "not applied to the 2025 data as of this report."
    )
    lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v3.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_paper_faithful_2025_v3_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
