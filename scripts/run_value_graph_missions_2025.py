"""Value-weighted graph community detection (this project's best-performing
method - see docs/graph_missions/level3_value_distribution_graph.md) applied
to the NEW 2025 per-store dataset, one store at a time - the same population
definition used everywhere else this method has been run (variety>=2,
basket_value>0, no Stage-1 gating; this method is a deliberate deviation
from the paper, not tied to the paper's relatively-concrete restriction).

Builds the category graph from value-weighted co-occurrence over each
store's *exact* full population (no CLARA sampling), detects communities
(Louvain), assigns every basket, profiles the results, and stability-tests
each store (20 seeds, Louvain-only randomness - the graph itself never
changes) exactly like every other run of this method.

Usage:
    PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import categories, graph_missions, ingest_v2, mission_stage, stability, temporal
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "graph_missions"
DOCS_STABILITY_DIR = PROJECT_ROOT / "docs" / "stability"
MIN_LIFT = 1.0
MIN_PREVALENCE = 1e-4
RESOLUTION = 2.0
N_STABILITY_RUNS = 20


def main():
    print("=== Value-weighted graph community detection, per store (2025 data) ===")
    t_all = time.time()

    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)
    print(f"Global category vocabulary: {vocab.size} categories (fixed across all stores)")

    store_results = {}
    for loc_id, info in stores.items():
        print(f"\n--- {info['format']} / {info['county']} ({loc_id}) ---")
        store_results[loc_id] = run_for_store(loc_id, info, vocab, vocab_df)

    _write_combined_summary(stores, store_results, vocab, time.time() - t_all)
    print(f"\nWrote {DOCS_DIR / 'level3_value_distribution_graph_2025_summary.md'} and one report per store.")


def run_for_store(loc_id, info, vocab, vocab_df):
    t0 = time.time()
    con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
    data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True, include_value_matrix=True)
    con.close()

    keep_mask = ((data.baskets["variety"] >= 2) & (data.baskets["basket_value"] > 0)).to_numpy()
    n_dropped = int((data.baskets["variety"] >= 2).sum()) - int(keep_mask.sum())
    baskets = data.baskets.loc[keep_mask].reset_index(drop=True)
    baskets = temporal.add_daypart(baskets)
    value_matrix = data.value_matrix[keep_mask]
    n = len(baskets)
    print(f"  ingested: {n:,} baskets (variety>=2, basket_value>0; {n_dropped} zero-value dropped) ({time.time()-t0:.1f}s)")

    if n < 500:
        print(f"  too few baskets ({n}) to build a meaningful graph - skipping")
        return {"info": info, "n": n, "skipped": True}

    norm = mission_stage.normalize_value_matrix(value_matrix)

    t1 = time.time()
    G = graph_missions.compute_value_pmi_graph(norm, vocab, min_lift=MIN_LIFT, min_prevalence=MIN_PREVALENCE)
    community_map = graph_missions.detect_communities(G, resolution=RESOLUTION, seed=RANDOM_STATE)
    assignment = graph_missions.assign_baskets_to_missions(norm, vocab, community_map)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{assignment.n_communities} communities ({time.time()-t1:.1f}s)")

    profiles = _profile_communities(baskets, value_matrix, assignment, vocab, community_map)
    for p in profiles:
        top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p["top_categories"][:3])
        print(f"    community {p['community_id']}: n={p['n_baskets']:,} ({p['share_pct']}%) top=[{top}]")

    t2 = time.time()
    stab = stability.run_stability_test_value_graph(
        norm, vocab, min_lift=MIN_LIFT, min_prevalence=MIN_PREVALENCE, resolution=RESOLUTION,
        n_runs=N_STABILITY_RUNS, base_seed=RANDOM_STATE,
    )
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    print(f"  stability ({N_STABILITY_RUNS} runs, {time.time()-t2:.1f}s): ARI mean={ari.mean():.4f} | NMI mean={nmi.mean():.4f}")

    daypart_xtab = _crosstab(baskets, assignment.hard_labels, "day_part")

    _write_store_report(loc_id, info, n, vocab, G, assignment, profiles, stab, daypart_xtab, time.time() - t0)

    return {
        "info": info, "n": n, "skipped": False, "n_communities": assignment.n_communities,
        "profiles": profiles, "stability": {"ari_mean": float(ari.mean()), "nmi_mean": float(nmi.mean())},
    }


def _profile_communities(baskets, value_matrix, assignment, vocab, community_map):
    n = len(assignment.hard_labels)
    overall_value = np.asarray(value_matrix.sum(axis=0)).ravel()
    overall_share = overall_value / overall_value.sum() if overall_value.sum() > 0 else overall_value
    id_to_idx = {cid: i for i, cid in enumerate(vocab.ids)}

    profiles = []
    for comm_id in range(assignment.n_communities):
        mask = assignment.hard_labels == comm_id
        nc = int(mask.sum())
        sub = value_matrix[mask]
        cshare = np.asarray(sub.sum(axis=0)).ravel()
        cshare = cshare / cshare.sum() if cshare.sum() > 0 else cshare
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_share > 0, cshare / overall_share, 0.0)
        member_categories = sorted(
            (cid for cid, c in community_map.items() if c == comm_id),
            key=lambda cid: -cshare[id_to_idx[cid]] if cid in id_to_idx else 0,
        )
        top_categories = [
            {"category": vocab.names[cid], "value_share": round(float(cshare[id_to_idx[cid]]), 4), "lift": round(float(lift[id_to_idx[cid]]), 3)}
            for cid in member_categories if cid in id_to_idx
        ]
        profiles.append({
            "community_id": comm_id,
            "n_baskets": nc,
            "share_pct": round(nc / n * 100, 2) if n else 0,
            "n_member_categories": len(member_categories),
            "avg_value": round(float(baskets.loc[mask, "basket_value"].mean()), 2) if nc else None,
            "avg_variety": round(float(baskets.loc[mask, "variety"].mean()), 2) if nc else None,
            "top_categories": top_categories,
        })
    return profiles


def _crosstab(baskets, labels, dim_col):
    df = baskets.copy()
    df["community"] = labels
    xtab = df.groupby([dim_col, "community"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_store_report(loc_id, info, n, vocab, G, assignment, profiles, stab, daypart_xtab, elapsed):
    name = f"level3_value_distribution_graph_2025_{info['format'].lower().replace(' ', '_')}_{info['county'].lower()}"
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi

    lines = []
    lines.append(f"# Value-weighted graph community detection (2025 data): {info['format']} — {info['county']}")
    lines.append("")
    lines.append(
        f"Store `{loc_id}`, {info['format']}, {info['county']}. Full calendar-2025 purchase history, "
        f"variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` "
        f"(value-weighted PMI graph, min_lift={MIN_LIFT}, min_prevalence={MIN_PREVALENCE}, Louvain "
        f"resolution={RESOLUTION}), run for this store alone. {n:,} baskets -> {G.number_of_nodes()} graph "
        f"nodes, {G.number_of_edges()} edges -> {assignment.n_communities} communities. Run in {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Community profiles")
    lines.append("")
    lines.append("| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |")
    lines.append("|---|---|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} ({c['value_share']*100:.0f}%, x{c['lift']:.1f})" for c in p["top_categories"][:5])
        lines.append(f"| {p['community_id']} | {p['share_pct']}% | {p['avg_value']} | {p['avg_variety']} | {p['n_member_categories']} | {top} |")
    lines.append("")

    lines.append("## Stability across seeds (Louvain-only randomness - graph itself is fixed)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    lines.append("")
    lines.append(f"Communities found per run: {stab.n_communities_per_run.tolist()}")
    lines.append("")

    lines.append("## Cross-validation by day-part")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    pivot = daypart_xtab.pivot(index="day_part", columns="community", values="share_within_dim").fillna(0)
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    pivot.columns = [f"C{c}" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{name}.md").write_text("\n".join(lines))

    stab_lines = [
        f"# Stability test: {name}", "",
        f"{stab.n_runs} Louvain runs on the fixed value-weighted PMI graph for this store "
        f"(see `docs/graph_missions/{name}.md` for the graph itself and full community profiles).",
        "", "| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|",
        f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |",
        f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |",
        "", f"Communities found per run: {stab.n_communities_per_run.tolist()}", "",
        f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`",
    ]
    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / f"{name}.md").write_text("\n".join(stab_lines))


def _write_combined_summary(stores, store_results, vocab, elapsed):
    lines = []
    lines.append("# Value-weighted graph community detection (2025 data): cross-store summary")
    lines.append("")
    lines.append(
        f"{len(stores)} Profi stores, full calendar-2025 purchase history each, this project's "
        f"best-performing method (value-weighted graph community detection, no basket sampling) run "
        f"independently per store. Total run time {elapsed:.1f}s."
    )
    lines.append("")
    lines.append("## Overview by store")
    lines.append("")
    lines.append("| Store | Format | County | Baskets | Communities | ARI | NMI | Largest mission |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        if r.get("skipped"):
            lines.append(f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n']:,} | - | - | - | too few baskets |")
            continue
        top_profile = max(r["profiles"], key=lambda p: p["n_baskets"])
        lines.append(
            f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n']:,} | {r['n_communities']} | "
            f"{r['stability']['ari_mean']:.3f} | {r['stability']['nmi_mean']:.3f} | "
            f"{top_profile['share_pct']}% |"
        )
    lines.append("")

    lines.append("## Full mission breakdown by store")
    lines.append("")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        lines.append(f"### {info['format']} — {info['county']} (`{loc_id}`)")
        lines.append("")
        if r.get("skipped"):
            lines.append("_Too few baskets._")
            lines.append("")
            continue
        lines.append("| Community | Share | Top categories (lift) |")
        lines.append("|---|---|---|")
        for p in sorted(r["profiles"], key=lambda p: -p["share_pct"]):
            top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p["top_categories"][:4])
            lines.append(f"| {p['community_id']} | {p['share_pct']}% | {top} |")
        lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_value_distribution_graph_2025_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
