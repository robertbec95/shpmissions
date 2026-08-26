"""NMF (non-negative matrix factorization) topic model for missions,
applied to the 2025 per-store dataset, one store at a time - the soft/
probabilistic alternative to every hard-label method tried so far in this
project (Jaccard k-medoids, HDBSCAN, value-weighted graph). Same population
convention as the value-weighted graph runs (variety>=2, basket_value>0, no
Stage-1 gating; this method is a deliberate deviation from the paper, not
tied to its relatively-concrete restriction).

For each store: picks a topic count via the reconstruction-error elbow (NMF
has no equivalent of the gap statistic), fits the reference model
(deterministic nndsvda init), profiles each topic two ways (native H-row
loadings, and value-share/lift on the hard argmax assignment - for
comparability against the graph method's reports), stability-tests solution
variance under non-convexity (20 random inits), and cross-validates by
day-part.

Usage:
    PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import categories, cluster_selection, ingest_v2, mission_stage, stability, temporal, topic_missions
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "topic_missions"
DOCS_STABILITY_DIR = PROJECT_ROOT / "docs" / "stability"
K_RANGE = range(3, 10)
N_STABILITY_RUNS = 20


def main():
    print("=== NMF topic model for missions, per store (2025 data) ===")
    t_all = time.time()

    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)
    print(f"Global category vocabulary: {vocab.size} categories (fixed across all stores)")

    store_results = {}
    for loc_id, info in stores.items():
        print(f"\n--- {info['format']} / {info['county']} ({loc_id}) ---")
        store_results[loc_id] = run_for_store(loc_id, info, vocab, vocab_df)

    _write_combined_summary(stores, store_results, time.time() - t_all)
    print(f"\nWrote {DOCS_DIR / 'level3_nmf_missions_2025_summary.md'} and one report per store.")


def run_for_store(loc_id, info, vocab, vocab_df):
    t0 = time.time()
    con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
    data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True, include_value_matrix=True)
    con.close()

    keep_mask = ((data.baskets["variety"] >= 2) & (data.baskets["basket_value"] > 0)).to_numpy()
    baskets = data.baskets.loc[keep_mask].reset_index(drop=True)
    baskets = temporal.add_daypart(baskets)
    value_matrix = data.value_matrix[keep_mask]
    n = len(baskets)
    print(f"  ingested: {n:,} baskets (variety>=2, basket_value>0) ({time.time()-t0:.1f}s)")

    if n < 500:
        print(f"  too few baskets ({n}) - skipping")
        return {"info": info, "n": n, "skipped": True}

    norm = mission_stage.normalize_value_matrix(value_matrix)

    t1 = time.time()
    scree = topic_missions.select_n_topics(norm, k_range=K_RANGE, random_state=RANDOM_STATE)
    k = cluster_selection.find_elbow_k(scree["k"], scree["reconstruction_err"])
    print(f"  topic-count scree: {dict(zip(scree['k'], scree['reconstruction_err'].round(2)))} -> elbow k={k} ({time.time()-t1:.1f}s)")

    t2 = time.time()
    fit = topic_missions.fit_nmf_missions(norm, k, random_state=RANDOM_STATE)
    profiles = topic_missions.profile_topics(norm, fit, vocab, baskets)
    n_degenerate = sum(1 for p in profiles if p["is_degenerate"])
    print(f"  fit: {k} topics, reconstruction_err={fit.reconstruction_err:.3f}, {n_degenerate} degenerate ({time.time()-t2:.1f}s)")
    for p in profiles:
        top = ", ".join(f"{c['category']} (w={c['loading']:.2f})" for c in p["native_top_categories"][:3])
        flag = " [DEGENERATE]" if p["is_degenerate"] else ""
        print(f"    topic {p['topic_id']}{flag}: hard_share={p['hard_share_pct']}% avg_soft_weight={p['avg_soft_weight_pct']}% top=[{top}]")

    t3 = time.time()
    stab = stability.run_stability_test_nmf(norm, k, n_runs=N_STABILITY_RUNS, base_seed=RANDOM_STATE)
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    print(f"  stability ({N_STABILITY_RUNS} random inits, {time.time()-t3:.1f}s): ARI mean={ari.mean():.4f} | NMI mean={nmi.mean():.4f}")

    daypart_xtab = _crosstab(baskets, fit.hard_labels, "day_part")

    _write_store_report(loc_id, info, n, k, scree, fit, profiles, stab, daypart_xtab, time.time() - t0)

    return {
        "info": info, "n": n, "skipped": False, "n_topics": k, "reconstruction_err": fit.reconstruction_err,
        "profiles": profiles, "stability": {"ari_mean": float(ari.mean()), "nmi_mean": float(nmi.mean())},
    }


def _crosstab(baskets, labels, dim_col):
    df = baskets.copy()
    df["topic"] = labels
    xtab = df.groupby([dim_col, "topic"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_store_report(loc_id, info, n, k, scree, fit, profiles, stab, daypart_xtab, elapsed):
    name = f"level3_nmf_missions_2025_{info['format'].lower().replace(' ', '_')}_{info['county'].lower()}"
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi

    lines = []
    lines.append(f"# NMF topic model for missions (2025 data): {info['format']} — {info['county']}")
    lines.append("")
    lines.append(
        f"Store `{loc_id}`, {info['format']}, {info['county']}. Full calendar-2025 purchase history, "
        f"variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), "
        f"{k} topics chosen by reconstruction-error elbow, reconstruction_err={fit.reconstruction_err:.3f}. "
        f"Run in {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Topic-count selection (reconstruction-error scree)")
    lines.append("")
    lines.append("| k | Reconstruction error |")
    lines.append("|---|---|")
    for _, row in scree.iterrows():
        marker = " <- elbow" if int(row["k"]) == k else ""
        lines.append(f"| {int(row['k'])} | {row['reconstruction_err']:.3f}{marker} |")
    lines.append("")

    lines.append("## Topic profiles")
    lines.append("")
    lines.append("| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |")
    lines.append("|---|---|---|---|---|---|")
    for p in profiles:
        native = ", ".join(f"{c['category']} ({c['loading']*100:.0f}%)" for c in p["native_top_categories"][:5])
        lines.append(
            f"| {p['topic_id']} | {'yes' if p['is_degenerate'] else 'no'} | {p['hard_share_pct']}% | "
            f"{p['avg_soft_weight_pct']}% | {p['avg_value']} | {native} |"
        )
    lines.append("")

    lines.append("## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)")
    lines.append("")
    lines.append("| Topic | Top categories by value share (lift) |")
    lines.append("|---|---|")
    for p in profiles:
        hard = ", ".join(f"{c['category']} ({c['value_share']*100:.0f}%, x{c['lift']:.1f})" for c in p["hard_top_categories"][:5])
        lines.append(f"| {p['topic_id']} | {hard} |")
    lines.append("")

    lines.append("## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    lines.append("")

    lines.append("## Cross-validation by day-part (hard assignment)")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    pivot = daypart_xtab.pivot(index="day_part", columns="topic", values="share_within_dim").fillna(0)
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    pivot.columns = [f"T{c}" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{name}.md").write_text("\n".join(lines))

    stab_lines = [
        f"# Stability test: {name}", "",
        f"{stab.n_runs} independent NMF fits ({stab.n_topics} topics each, `init=\"random\"`) on the identical "
        f"full value matrix for this store - no basket sampling, isolates non-convex solution variance alone "
        f"(see `docs/topic_missions/{name}.md` for the reference fit and full topic profiles).",
        "", "| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|",
        f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |",
        f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |",
        "", f"Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`",
    ]
    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / f"{name}.md").write_text("\n".join(stab_lines))


def _write_combined_summary(stores, store_results, elapsed):
    lines = []
    lines.append("# NMF topic model for missions (2025 data): cross-store summary")
    lines.append("")
    lines.append(
        f"{len(stores)} Profi stores, full calendar-2025 purchase history each, NMF (non-negative matrix "
        f"factorization) on the row-normalized value matrix, run independently per store - the soft/"
        f"probabilistic alternative to this project's hard-label methods (Jaccard k-medoids, value-weighted "
        f"graph). Total run time {elapsed:.1f}s."
    )
    lines.append("")
    lines.append("## Overview by store")
    lines.append("")
    lines.append("| Store | Format | County | Baskets | Topics | Degenerate | ARI | NMI | Largest topic (hard share) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        if r.get("skipped"):
            lines.append(f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n']:,} | - | - | - | - | too few baskets |")
            continue
        n_degenerate = sum(1 for p in r["profiles"] if p["is_degenerate"])
        top_profile = max(r["profiles"], key=lambda p: p["hard_share_pct"])
        lines.append(
            f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n']:,} | {r['n_topics']} | "
            f"{n_degenerate} | {r['stability']['ari_mean']:.3f} | {r['stability']['nmi_mean']:.3f} | "
            f"{top_profile['hard_share_pct']}% |"
        )
    lines.append("")

    lines.append("## Full topic breakdown by store")
    lines.append("")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        lines.append(f"### {info['format']} — {info['county']} (`{loc_id}`)")
        lines.append("")
        if r.get("skipped"):
            lines.append("_Too few baskets._")
            lines.append("")
            continue
        lines.append("| Topic | Degenerate? | Hard share | Native top categories (loading) |")
        lines.append("|---|---|---|---|")
        for p in sorted(r["profiles"], key=lambda p: -p["hard_share_pct"]):
            native = ", ".join(f"{c['category']} ({c['loading']*100:.0f}%)" for c in p["native_top_categories"][:4])
            lines.append(f"| {p['topic_id']} | {'yes' if p['is_degenerate'] else 'no'} | {p['hard_share_pct']}% | {native} |")
        lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_nmf_missions_2025_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
