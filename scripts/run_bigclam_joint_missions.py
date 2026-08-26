"""Joint (pooled) BigCLAM overlapping community detection, fit ONCE across
all 6 stores' pooled value-weighted category graph - the graph-based
analogue of `run_nmf_joint_missions.py`. Every per-store BigCLAM run so far
(`run_bigclam_missions_2025.py`) produced its own independent community
space - Community 3 at Neamt isn't Community 3 at Bucharest, so those
results can't be compared store-to-store or used as one shared ground
truth. This script builds ONE category graph from co-occurrence pooled
across all 6 stores, fits BigCLAM once, and reports per-store PREVALENCE
within that single shared community space - directly analogous to how the
joint NMF's topics are one shared space with per-store prevalence.

Built specifically so its output can be validated the same way the joint
NMF was: through `location_similarity`'s weighted-Gower leave-one-out
propagation pipeline (see `run_location_similarity_propagation.py`) - two
different transaction-based mission-detection methods, checked against the
identical downstream task, for a real head-to-head comparison rather than
comparing methods on different, incompatible criteria.

Population convention matches the per-store BigCLAM/value-graph family
(variety>=2, basket_value>0), NOT the joint NMF's variety>=1 - this
extends the graph-based method lineage, not the topic-model lineage.

Usage:
    PYTHONPATH=src python3 scripts/run_bigclam_joint_missions.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.sparse import vstack

from shopping_missions import bigclam, categories, cluster_selection, graph_missions, ingest_v2, mission_stage, temporal
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "bigclam"
MIN_LIFT = 1.0
MIN_PREVALENCE = 1e-4
C_RANGE = range(4, 11)
REL_THRESHOLD = 0.5  # matches the per-store convention - see bigclam module + memory for why 0.3 didn't transfer


def main():
    print("=== Joint (pooled) BigCLAM overlapping community detection, all 6 stores ===")
    t_all = time.time()

    print("\n--- Ingest + pool ---")
    t0 = time.time()
    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)

    value_parts, meta_parts = [], []
    for loc_id in sorted(stores):
        info = stores[loc_id]
        con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
        data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True, include_value_matrix=True)
        con.close()
        keep_mask = ((data.baskets["variety"] >= 2) & (data.baskets["basket_value"] > 0)).to_numpy()
        baskets = temporal.add_daypart(data.baskets.loc[keep_mask].reset_index(drop=True))
        baskets["loc_id"] = loc_id
        value_parts.append(data.value_matrix[keep_mask])
        meta_parts.append(baskets[["basket_id", "loc_id", "basket_value"]])
        print(f"  {info['format']} / {info['county']}: {int(keep_mask.sum()):,} baskets (variety>=2)")

    value_matrix = vstack(value_parts).tocsr()
    meta = pd.concat(meta_parts, ignore_index=True)
    n = value_matrix.shape[0]
    print(f"pooled: {n:,} baskets, {vocab.size} categories ({time.time()-t0:.1f}s)")

    norm = mission_stage.normalize_value_matrix(value_matrix)

    print("\n--- Pooled value-weighted PMI graph ---")
    t1 = time.time()
    G = graph_missions.compute_value_pmi_graph(norm, vocab, min_lift=MIN_LIFT, min_prevalence=MIN_PREVALENCE)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time()-t1:.1f}s)")

    print(f"\n--- Community-count sweep: C in {list(C_RANGE)} ---")
    t2 = time.time()
    scree = bigclam.select_n_communities(G, c_range=C_RANGE, seed=RANDOM_STATE)
    C = cluster_selection.find_elbow_k(scree["n_communities"], -scree["log_likelihood"])
    print(scree.to_string(index=False))
    print(f"elbow C={C} ({time.time()-t2:.1f}s)")

    print(f"\n--- Reference fit: C={C} ---")
    t3 = time.time()
    fit = bigclam.fit_bigclam(G, C, seed=RANDOM_STATE)
    communities = bigclam.extract_communities(fit, rel_threshold=REL_THRESHOLD)
    assignment = bigclam.assign_baskets_to_missions(norm, vocab, fit)
    covered = {cid for members in communities.values() for cid in members}
    print(f"  {C} communities, log_likelihood={fit.log_likelihood:.1f}, coverage={len(covered)}/{G.number_of_nodes()} ({time.time()-t3:.1f}s)")

    profiles = _profile_communities(meta, value_matrix, assignment, vocab, communities)
    for p in profiles:
        top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p["top_categories"][:4])
        print(f"    community {p['community_id']}: n={p['n_baskets']:,} ({p['share_pct']}%) top=[{top}]")
    overlap = _find_overlap(communities, vocab)
    print(f"  {len(overlap)} categories in multiple communities")

    print("\n--- Per-store prevalence (post-hoc, shared community space) ---")
    store_prevalence = _store_soft_prevalence(meta, assignment, C)

    _write_report(vocab, G, C, scree, fit, profiles, overlap, store_prevalence, meta, time.time() - t_all)
    print(f"\nTotal runtime {time.time()-t_all:.1f}s")
    print(f"Wrote {DOCS_DIR / 'level3_bigclam_joint_2025.md'}")

    return profiles, store_prevalence


def _profile_communities(meta, value_matrix, assignment, vocab, communities):
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
            "community_id": comm_id, "n_baskets": nc, "share_pct": round(nc / n * 100, 2) if n else 0,
            "n_member_categories": len(member_categories), "top_categories": top_categories,
        })
    return profiles


def _find_overlap(communities, vocab):
    from collections import defaultdict
    membership = defaultdict(list)
    for comm_id, members in communities.items():
        for cid in members:
            membership[cid].append(comm_id)
    overlap = [{"category": vocab.names[cid], "communities": sorted(comms)} for cid, comms in membership.items() if len(comms) > 1]
    overlap.sort(key=lambda o: -len(o["communities"]))
    return overlap


def _store_soft_prevalence(meta, assignment, C):
    meta = meta.copy()
    meta["soft"] = list(assignment.soft_scores)
    out = {}
    for loc_id, grp in meta.groupby("loc_id"):
        avg = np.mean(np.stack(grp["soft"].to_numpy()), axis=0)
        out[loc_id] = (avg * 100).tolist()
    return out


def _write_report(vocab, G, C, scree, fit, profiles, overlap, store_prevalence, meta, elapsed):
    stores = ingest_v2.list_store_files()
    store_names = {loc_id: f"{stores[loc_id]['format']} | {stores[loc_id]['county']}" for loc_id in store_prevalence}

    lines = []
    lines.append("# Joint (pooled) BigCLAM overlapping community detection (2025 data, all 6 stores)")
    lines.append("")
    lines.append(
        f"{len(meta):,} baskets across all 6 stores (variety>=2, basket_value>0), pooled into ONE value-weighted "
        f"PMI graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) and fit with BigCLAM once - "
        f"{C} communities (log-likelihood elbow), relative threshold={REL_THRESHOLD}. Unlike the per-store "
        f"BigCLAM runs (`docs/bigclam/level3_bigclam_2025_summary.md`), this is ONE shared community space, so "
        f"community IDs are directly comparable across stores. Total run time {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Community-count selection")
    lines.append("")
    lines.append("| C | Log-likelihood |")
    lines.append("|---|---|")
    for _, row in scree.iterrows():
        marker = " <- elbow" if int(row["n_communities"]) == C else ""
        lines.append(f"| {int(row['n_communities'])} | {row['log_likelihood']:.1f}{marker} |")
    lines.append("")

    lines.append("## Community profiles")
    lines.append("")
    lines.append("| Community | Share | # categories | Top categories by value share (lift) |")
    lines.append("|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} ({c['value_share']*100:.0f}%, x{c['lift']:.1f})" for c in p["top_categories"][:5])
        lines.append(f"| {p['community_id']} | {p['share_pct']}% | {p['n_member_categories']} | {top} |")
    lines.append("")

    lines.append("## Categories in multiple communities")
    lines.append("")
    if overlap:
        lines.append("| Category | Communities |")
        lines.append("|---|---|")
        for o in overlap:
            lines.append(f"| {o['category']} | {', '.join(str(c) for c in o['communities'])} |")
    else:
        lines.append("_None at this threshold._")
    lines.append("")

    lines.append("## Per-store prevalence (shared community space - directly comparable across stores)")
    lines.append("")
    header = "| Store | " + " | ".join(f"C{c}" for c in range(C)) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * C)
    for loc_id, prev in store_prevalence.items():
        row = " | ".join(f"{v:.1f}%" for v in prev)
        lines.append(f"| {store_names[loc_id]} | {row} |")
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_joint_missions.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_bigclam_joint_2025.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
