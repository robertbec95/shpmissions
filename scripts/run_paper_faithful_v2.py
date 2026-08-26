"""Faithful reproduction of Sarantopoulos et al. (2015), run on the NEW
2025 per-store dataset (6 Profi stores - 2 Supermarket, 3 Convenience
Store, 1 Mini Market - full calendar-2025 purchase history each), run
PER STORE as the paper specifies ("For each store, a second clustering
analysis pertains to each of the three shopping goal abstractness
levels") - the first time this project has had genuine per-store data to
do that with; every earlier scenario ran chain-wide across many stores.

Same faithful procedure as `run_paper_faithful.py` (real Stage 1 gating,
60/40 train/test split, gap statistic on the training set, Jaccard
k-centroid/k-medoids, unweighted binary category presence), applied
independently to each store, plus a cross-store comparison since the
paper's own framing implies missions may differ by store/format.

Data source: `ingest_v2.py` (new parquet exports, different schema/ID
encoding from the original `transaction_lines/` CSVs, same underlying
Profi category taxonomy - see that module's docstring).

Usage:
    PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py
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


def main():
    print("=== Paper-faithful reproduction, per store (2025 full-year data) ===")
    t_all = time.time()

    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)
    print(f"Global category vocabulary: {vocab.size} categories (fixed across all stores)")
    print(f"Stores: {len(stores)}")
    for loc_id, info in stores.items():
        print(f"  {loc_id}  {info['format']:<18} {info['county']}")

    store_results = {}
    for loc_id, info in stores.items():
        print(f"\n--- {info['format']} / {info['county']} ({loc_id}) ---")
        store_results[loc_id] = run_for_store(loc_id, info, vocab, vocab_df)

    _write_combined_summary(stores, store_results, vocab, time.time() - t_all)
    print(f"\nWrote {DOCS_DIR / 'level3_paper_faithful_2025_summary.md'} and one report per store.")


def run_for_store(loc_id, info, vocab, vocab_df):
    t0 = time.time()
    con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
    data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True)
    con.close()
    baskets_all = temporal.add_daypart(data.baskets)
    matrix_all = data.matrix
    print(f"  ingested: {len(baskets_all):,} baskets ({time.time()-t0:.1f}s)")

    seg = variety_stage.fit_variety_segments(baskets_all, random_state=RANDOM_STATE)
    print(f"  Stage 1: bounds={seg.bounds}")
    baskets_all = seg.baskets  # already carries day_part - fit_variety_segments copies its input, doesn't drop columns

    rel_mask = (baskets_all["segment"] == "relatively_concrete").to_numpy()
    rel_baskets = baskets_all.loc[rel_mask].reset_index(drop=True)
    rel_matrix = matrix_all[rel_mask]
    n_rel = len(rel_baskets)
    print(f"  {n_rel:,} relatively-concrete baskets")

    if n_rel < 200:
        print(f"  too few relatively-concrete baskets ({n_rel}) to cluster meaningfully - skipping Stage 2")
        return {"info": info, "n_total": len(baskets_all), "seg": seg, "n_rel": n_rel, "skipped": True}

    all_idx = np.arange(n_rel)
    train_idx, test_idx = train_test_split(all_idx, train_size=TRAIN_FRACTION, random_state=RANDOM_STATE)
    train_matrix = rel_matrix[train_idx]

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(train_matrix.shape[0], size=min(K_SELECT_SAMPLE, train_matrix.shape[0]), replace=False)
    X_sample = np.asarray(train_matrix[sample_idx].todense(), dtype=np.float64)
    gdf = mission_stage.select_k(X_sample, k_range=range(2, 12), B=10, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    elbow_k = elbow_k_from_gap_df(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    print(f"  Stage 2 k-selection ({time.time()-t0:.1f}s total so far): gap_k={gap_k}, elbow_k={elbow_k}, silhouette_k={sil_k}")

    clustering = mission_stage.fit_and_assign_missions(
        train_matrix, k=gap_k, sample_size=min(CLARA_SAMPLE_SIZE, train_matrix.shape[0]), random_state=RANDOM_STATE,
    )
    full_labels, _ = mission_stage._assign_to_medoids(rel_matrix, clustering.medoid_vectors)

    profiles = profiling.profile_clusters(rel_matrix, full_labels, vocab, rel_baskets)
    for p in profiles:
        top = ", ".join(c["category"] for c in p.top_categories[:4])
        print(f"    cluster {p.cluster_id}: {p.suggested_label} n={p.n_baskets:,} ({p.share_of_relatively_concrete}%) top=[{top}]")

    is_train = np.zeros(n_rel, dtype=bool)
    is_train[train_idx] = True
    train_test_shares = _train_test_comparison(full_labels, is_train, gap_k)

    daypart_xtab = _crosstab(rel_baskets, full_labels, "day_part")

    _write_store_report(loc_id, info, seg, gdf, gap_k, elbow_k, sil_k, n_rel, len(train_idx), len(test_idx),
                         profiles, train_test_shares, daypart_xtab, time.time() - t0)

    return {
        "info": info, "n_total": len(baskets_all), "seg": seg, "n_rel": n_rel, "skipped": False,
        "gap_k": gap_k, "profiles": profiles, "train_test": train_test_shares,
    }


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


def _write_store_report(loc_id, info, seg, gdf, gap_k, elbow_k, sil_k, n_rel, n_train, n_test,
                         profiles, train_test_shares, daypart_xtab, elapsed):
    name = f"level3_paper_faithful_2025_{info['format'].lower().replace(' ', '_')}_{info['county'].lower()}"
    lines = []
    lines.append(f"# Faithful reproduction (2025 data): {info['format']} — {info['county']}")
    lines.append("")
    lines.append(
        f"Store `{loc_id}`, {info['format']}, {info['county']}. Full calendar-2025 purchase history. "
        f"Same faithful procedure as `docs/paper_faithful/level3_paper_faithful.md` (real Stage 1 gating, "
        f"60/40 train/test split, gap statistic on the training set, Jaccard k-medoids), run for this store "
        f"alone. Run in {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Stage 1: variety segmentation")
    lines.append("")
    lines.append(seg.summary.round(2).to_markdown())
    lines.append("")
    lines.append(f"Bounds: `{seg.bounds}`")
    lines.append("")

    lines.append("## Stage 2: k-selection (gap statistic, on the 60% training set)")
    lines.append("")
    lines.append(f"{n_rel:,} relatively-concrete baskets -> {n_train:,} train / {n_test:,} test. "
                  f"gap_k={gap_k} (decision), elbow_k={elbow_k}, silhouette_k={sil_k} (comparison).")
    lines.append("")
    lines.append(gdf[["k", "gap", "sk", "gap_rule_satisfied", "silhouette"]].round(4).to_markdown(index=False))
    lines.append("")

    lines.append(f"## Mission clusters (k={gap_k})")
    lines.append("")
    lines.append("| Cluster | Label | Share | Avg value | Top categories (lift) |")
    lines.append("|---|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p.top_categories[:5])
        lines.append(f"| {p.cluster_id} | {p.suggested_label} (conf {p.label_confidence}) | {p.share_of_relatively_concrete}% | {p.avg_basket_value} | {top} |")
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
    cluster_labels = {p.cluster_id: p.suggested_label for p in profiles}
    pivot.columns = [f"C{c} ({cluster_labels.get(c,'?')})" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{name}.md").write_text("\n".join(lines))


def _write_combined_summary(stores, store_results, vocab, elapsed):
    lines = []
    lines.append("# Faithful reproduction (2025 data): cross-store summary")
    lines.append("")
    lines.append(
        f"{len(stores)} Profi stores, full calendar-2025 purchase history each, same faithful procedure "
        f"(real Stage 1 gating, 60/40 train/test split, gap statistic on train, Jaccard k-medoids) run "
        f"independently per store - this is the paper's own framing (\"For each store, a second clustering "
        f"analysis...\"), the first time this project has had genuine per-store data to honor that with. "
        f"Total run time {elapsed:.1f}s."
    )
    lines.append("")
    lines.append("## Stage 1 + Stage 2 overview by store")
    lines.append("")
    lines.append("| Store | Format | County | Total baskets | Relatively-concrete | gap_k | Largest mission | Top categories |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        if r.get("skipped"):
            lines.append(f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n_total']:,} | {r['n_rel']:,} | - | - | too few baskets |")
            continue
        top_profile = max(r["profiles"], key=lambda p: p.n_baskets)
        top_cats = ", ".join(c["category"] for c in top_profile.top_categories[:3])
        lines.append(
            f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n_total']:,} | {r['n_rel']:,} | "
            f"{r['gap_k']} | {top_profile.share_of_relatively_concrete}% ({top_profile.suggested_label}) | {top_cats} |"
        )
    lines.append("")

    lines.append("## Full mission breakdown by store")
    lines.append("")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        lines.append(f"### {info['format']} — {info['county']} (`{loc_id}`)")
        lines.append("")
        if r.get("skipped"):
            lines.append("_Too few relatively-concrete baskets to cluster._")
            lines.append("")
            continue
        lines.append("| Cluster | Label | Share | Top categories (lift) |")
        lines.append("|---|---|---|---|")
        for p in r["profiles"]:
            top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p.top_categories[:4])
            lines.append(f"| {p.cluster_id} | {p.suggested_label} | {p.share_of_relatively_concrete}% | {top} |")
        lines.append("")

    lines.append("## Does mission structure differ by store format?")
    lines.append("")
    by_format = {}
    for loc_id, info in stores.items():
        by_format.setdefault(info["format"], []).append((loc_id, store_results[loc_id]))
    for fmt, entries in by_format.items():
        lines.append(f"**{fmt}** ({len(entries)} store{'s' if len(entries)>1 else ''}):")
        for loc_id, r in entries:
            if r.get("skipped"):
                lines.append(f"- {stores[loc_id]['county']}: too few baskets")
                continue
            lines.append(f"- {stores[loc_id]['county']}: gap_k={r['gap_k']}, largest mission "
                          f"{max(p.share_of_relatively_concrete for p in r['profiles'])}%")
        lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_paper_faithful_2025_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
