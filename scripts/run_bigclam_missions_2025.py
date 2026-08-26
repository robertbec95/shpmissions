"""BigCLAM (Yang & Leskovec 2013) overlapping community detection, applied
to the 2025 per-store dataset - the genuinely-overlapping alternative to
this project's Louvain-based value-weighted graph method (hard partition)
and to CPM (clique-based overlap, which - see project history - found real
overlap but structurally dropped standalone single-category missions like
TOBACCO/BEER/WINE, since a node only joins a CPM community by sitting
inside a dense k-clique).

Builds the exact same value-weighted PMI category graph as
`run_value_graph_missions_2025.py` (so results are directly comparable to
this project's best-performing method), then fits BigCLAM instead of
Louvain. Sweeps the number of communities via log-likelihood elbow,
extracts overlapping community membership, assigns every basket (a
basket's affinity to a community is the sum of its categories' continuous
membership strengths - see `bigclam.assign_baskets_to_missions`), profiles
each community, and stability-tests (independent BigCLAM refits, same
fixed graph).

Usage:
    PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import bigclam, categories, cluster_selection, graph_missions, ingest_v2, mission_stage, stability, temporal
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "bigclam"
DOCS_STABILITY_DIR = PROJECT_ROOT / "docs" / "stability"
MIN_LIFT = 1.0
MIN_PREVALENCE = 1e-4
C_RANGE = range(4, 9)
# 0.3 (the threshold validated on a single store during development) turned
# out not to transfer: on a sparser/differently-structured store it let 51%
# of categories "overlap" (up to 5 communities each) - noise, not signal,
# traced to relative-threshold-vs-absolute-floor interaction (see
# `bigclam.extract_communities` for the isolated-node bug this surfaced).
# 0.5 is a documented compromise across the two stores checked, not a
# per-store-tuned optimum - true per-store tuning would need its own
# threshold sweep per store, not done here given the added complexity of a
# 2D (C, threshold) search per store.
REL_THRESHOLD = 0.5
N_STABILITY_RUNS = 10


def main():
    print("=== BigCLAM overlapping community detection, per store (2025 data) ===")
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
    print(f"\nWrote {DOCS_DIR / 'level3_bigclam_2025_summary.md'} and one report per store.")


def run_for_store(loc_id, info, vocab, vocab_df):
    t0 = time.time()
    con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
    data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True, include_value_matrix=True)
    con.close()

    keep_mask = ((data.baskets["variety"] >= 2) & (data.baskets["basket_value"] > 0)).to_numpy()
    baskets = temporal.add_daypart(data.baskets.loc[keep_mask].reset_index(drop=True))
    value_matrix = data.value_matrix[keep_mask]
    n = len(baskets)
    print(f"  ingested: {n:,} baskets (variety>=2, basket_value>0) ({time.time()-t0:.1f}s)")

    if n < 500:
        print(f"  too few baskets ({n}) - skipping")
        return {"info": info, "n": n, "skipped": True}

    norm = mission_stage.normalize_value_matrix(value_matrix)

    t1 = time.time()
    G = graph_missions.compute_value_pmi_graph(norm, vocab, min_lift=MIN_LIFT, min_prevalence=MIN_PREVALENCE)
    scree = bigclam.select_n_communities(G, c_range=C_RANGE, seed=RANDOM_STATE)
    C = cluster_selection.find_elbow_k(scree["n_communities"], -scree["log_likelihood"])
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"C sweep {dict(zip(scree['n_communities'], scree['log_likelihood'].round(1)))} -> elbow C={C} ({time.time()-t1:.1f}s)")

    t2 = time.time()
    fit = bigclam.fit_bigclam(G, C, seed=RANDOM_STATE)
    communities = bigclam.extract_communities(fit, rel_threshold=REL_THRESHOLD)
    assignment = bigclam.assign_baskets_to_missions(norm, vocab, fit)
    covered = {cid for members in communities.values() for cid in members}
    print(f"  fit: {C} communities, log_likelihood={fit.log_likelihood:.1f}, "
          f"coverage={len(covered)}/{G.number_of_nodes()} ({time.time()-t2:.1f}s)")

    profiles = _profile_communities(baskets, value_matrix, assignment, vocab, communities)
    overlap = _find_overlap(communities, vocab)
    for p in profiles:
        top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p["top_categories"][:3])
        print(f"    community {p['community_id']}: n={p['n_baskets']:,} ({p['share_pct']}%) top=[{top}]")
    overlap_preview = ", ".join(f"{o['category']}({o['communities']})" for o in overlap[:6])
    print(f"  {len(overlap)} categories in multiple communities: {overlap_preview}{' ...' if len(overlap) > 6 else ''}")

    t3 = time.time()
    stab = stability.run_stability_test_bigclam(G, norm, vocab, C, n_runs=N_STABILITY_RUNS, base_seed=RANDOM_STATE)
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    print(f"  stability ({N_STABILITY_RUNS} runs, {time.time()-t3:.1f}s): ARI mean={ari.mean():.4f} | NMI mean={nmi.mean():.4f}")

    daypart_xtab = _crosstab(baskets, assignment.hard_labels, "day_part")

    _write_store_report(loc_id, info, n, vocab, G, C, scree, fit, communities, assignment, profiles, overlap, stab, daypart_xtab, time.time() - t0)

    return {
        "info": info, "n": n, "skipped": False, "n_communities": C, "coverage": len(covered), "n_nodes": G.number_of_nodes(),
        "profiles": profiles, "overlap": overlap,
        "stability": {"ari_mean": float(ari.mean()), "nmi_mean": float(nmi.mean())},
    }


def _profile_communities(baskets, value_matrix, assignment, vocab, communities):
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
            communities.get(comm_id, []),
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


def _find_overlap(communities, vocab):
    from collections import defaultdict
    membership = defaultdict(list)
    for comm_id, members in communities.items():
        for cid in members:
            membership[cid].append(comm_id)
    overlap = [
        {"category": vocab.names[cid], "communities": sorted(comms)}
        for cid, comms in membership.items() if len(comms) > 1
    ]
    overlap.sort(key=lambda o: -len(o["communities"]))
    return overlap


def _crosstab(baskets, labels, dim_col):
    df = baskets.copy()
    df["community"] = labels
    xtab = df.groupby([dim_col, "community"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_store_report(loc_id, info, n, vocab, G, C, scree, fit, communities, assignment, profiles, overlap, stab, daypart_xtab, elapsed):
    name = f"level3_bigclam_2025_{info['format'].lower().replace(' ', '_')}_{info['county'].lower()}"
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    covered = {cid for members in communities.values() for cid in members}

    lines = []
    lines.append(f"# BigCLAM overlapping community detection (2025 data): {info['format']} — {info['county']}")
    lines.append("")
    lines.append(
        f"Store `{loc_id}`, {info['format']}, {info['county']}. Full calendar-2025 purchase history, "
        f"variety>=2, basket_value>0. Same value-weighted PMI graph as "
        f"`docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift={MIN_LIFT}, "
        f"min_prevalence={MIN_PREVALENCE}), fit with BigCLAM instead of Louvain - {C} communities "
        f"(log-likelihood elbow), relative threshold={REL_THRESHOLD}. {n:,} baskets -> {G.number_of_nodes()} "
        f"graph nodes, {G.number_of_edges()} edges -> {C} communities covering {len(covered)}/{G.number_of_nodes()} "
        f"categories. Run in {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Community-count selection (log-likelihood scree)")
    lines.append("")
    lines.append("| C | Log-likelihood |")
    lines.append("|---|---|")
    for _, row in scree.iterrows():
        marker = " <- elbow" if int(row["n_communities"]) == C else ""
        lines.append(f"| {int(row['n_communities'])} | {row['log_likelihood']:.1f}{marker} |")
    lines.append("")

    lines.append("## Community profiles")
    lines.append("")
    lines.append("| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |")
    lines.append("|---|---|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} ({c['value_share']*100:.0f}%, x{c['lift']:.1f})" for c in p["top_categories"][:5])
        lines.append(f"| {p['community_id']} | {p['share_pct']}% | {p['avg_value']} | {p['avg_variety']} | {p['n_member_categories']} | {top} |")
    lines.append("")

    lines.append("## Categories in multiple communities (the overlap CPM/Louvain cannot represent)")
    lines.append("")
    if overlap:
        lines.append("| Category | Communities |")
        lines.append("|---|---|")
        for o in overlap:
            lines.append(f"| {o['category']} | {', '.join(str(c) for c in o['communities'])} |")
    else:
        lines.append("_None at this threshold - every category resolved to a single dominant community._")
    lines.append("")

    lines.append("## Stability across independent refits (fixed graph, different random init)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    lines.append("")

    lines.append("## Cross-validation by day-part (hard assignment)")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    pivot = daypart_xtab.pivot(index="day_part", columns="community", values="share_within_dim").fillna(0)
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    pivot.columns = [f"C{c}" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{name}.md").write_text("\n".join(lines))

    stab_lines = [
        f"# Stability test: {name}", "",
        f"{stab.n_runs} independent BigCLAM refits ({C} communities each, fresh random init) on the "
        f"identical fixed value-weighted PMI graph for this store (see `docs/bigclam/{name}.md` for the "
        f"graph itself and full community profiles).",
        "", "| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|",
        f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |",
        f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |",
        "", "Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`",
    ]
    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / f"{name}.md").write_text("\n".join(stab_lines))


def _write_combined_summary(stores, store_results, elapsed):
    lines = []
    lines.append("# BigCLAM overlapping community detection (2025 data): cross-store summary")
    lines.append("")
    lines.append(
        f"{len(stores)} Profi stores, full calendar-2025 purchase history each, BigCLAM (Yang & Leskovec 2013) "
        f"fit on the same value-weighted PMI graph as this project's Louvain-based method, per store. "
        f"Total run time {elapsed:.1f}s."
    )
    lines.append("")
    lines.append("## Overview by store")
    lines.append("")
    lines.append("| Store | Format | County | Baskets | Communities | Coverage | Overlapping categories | ARI | NMI |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for loc_id, info in stores.items():
        r = store_results[loc_id]
        if r.get("skipped"):
            lines.append(f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n']:,} | - | - | - | - | too few baskets |")
            continue
        lines.append(
            f"| {loc_id[-6:]} | {info['format']} | {info['county']} | {r['n']:,} | {r['n_communities']} | "
            f"{r['coverage']}/{r['n_nodes']} | {len(r['overlap'])} | "
            f"{r['stability']['ari_mean']:.3f} | {r['stability']['nmi_mean']:.3f} |"
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
        if r["overlap"]:
            over_str = ", ".join(f"{o['category']} ({','.join(str(c) for c in o['communities'])})" for o in r["overlap"])
            lines.append(f"Overlapping categories: {over_str}")
            lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_bigclam_2025_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
