"""Run graph community detection (src/shopping_missions/graph_missions.py)
against a named scenario's population, profile the resulting missions, and
stability-test it the same way CLARA k-medoids and HDBSCAN were tested (see
docs/stability/*.md) - the whole point of this approach is to check whether
building the category graph from the exact full population (no basket
sampling) actually produces more repeatable missions.

Usage:
    PYTHONPATH=src python3 scripts/run_graph_missions.py level3_8day
    PYTHONPATH=src python3 scripts/run_graph_missions.py level3_baseline level3_8day --n-runs 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import graph_missions, multiday, stability
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE
from shopping_missions.scenarios import get_scenario

DOCS_DIR = PROJECT_ROOT / "docs" / "graph_missions"
DEGENERATE_SUPPORT_THRESHOLD = 0.90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="scenario name(s) from scenarios.py")
    parser.add_argument(
        "--min-lift", type=float, default=2.0,
        help="edge threshold on lift - 2.0 chosen empirically (see docs/graph_missions/*.md): "
             "1.0 leaves the graph too densely connected (~87%% of all category pairs have some "
             "positive lift) and Louvain collapses everything into 1-2 giant communities; 2.0 "
             "gives a reasonable number of communities without fragmenting into singletons.",
    )
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution - higher = more, smaller communities")
    parser.add_argument("--n-runs", type=int, default=20, help="stability test run count")
    args = parser.parse_args()

    for name in args.scenarios:
        run_for_scenario(name, args.min_lift, args.resolution, args.n_runs)


def run_for_scenario(name: str, min_lift: float, resolution: float, n_runs: int):
    scenario = get_scenario(name)
    print(f"\n=== Graph missions: {name} ===")
    t0 = time.time()

    data = multiday.build_basket_data_multi_date(
        list(scenario.dates), category_level=scenario.category_level, include_daypart=False
    )
    vocab = data.vocab
    keep_mask = (data.baskets["variety"] >= scenario.min_variety).to_numpy()
    baskets = data.baskets.loc[keep_mask].reset_index(drop=True)
    matrix = data.matrix[keep_mask]
    n = len(baskets)
    print(f"  ingested + filtered: {n:,} baskets, {vocab.size} categories ({time.time()-t0:.1f}s)")

    t1 = time.time()
    G = graph_missions.compute_pmi_graph(matrix, vocab, min_lift=min_lift)
    community_map = graph_missions.detect_communities(G, resolution=resolution, seed=RANDOM_STATE)
    assignment = graph_missions.assign_baskets_to_missions(matrix, vocab, community_map)
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{assignment.n_communities} communities found ({time.time()-t1:.1f}s)")

    profiles = _profile_communities(baskets, matrix, assignment, vocab, G, community_map)
    for p in profiles:
        top = ", ".join(c["category"] for c in p["top_categories"][:5])
        flag = " [DEGENERATE]" if p["is_degenerate"] else ""
        print(f"  community {p['community_id']}: n={p['n_baskets']:,} ({p['share_pct']}%) "
              f"categories=[{top}]{flag}")

    t2 = time.time()
    stab = stability.run_stability_test_graph(
        matrix, vocab, min_lift=min_lift, resolution=resolution, n_runs=n_runs, base_seed=RANDOM_STATE
    )
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    print(f"  stability ({n_runs} runs, {time.time()-t2:.1f}s): "
          f"ARI mean={ari.mean():.4f} std={ari.std():.4f} | NMI mean={nmi.mean():.4f} std={nmi.std():.4f}")
    print(f"  communities found per run: {stab.n_communities_per_run.tolist()}")

    _write_report(name, scenario, n, vocab, G, assignment.n_communities, profiles, stab, min_lift, resolution, time.time() - t0)
    print(f"  wrote {DOCS_DIR / (name + '.md')}")


def _profile_communities(baskets, matrix, assignment, vocab, G, community_map):
    n = len(assignment.hard_labels)
    overall_support = np.asarray(matrix.sum(axis=0)).ravel() / n
    id_to_idx = {cid: i for i, cid in enumerate(vocab.ids)}

    profiles = []
    for comm_id in range(assignment.n_communities):
        mask = assignment.hard_labels == comm_id
        nc = int(mask.sum())
        sub = matrix[mask]
        cluster_support = np.asarray(sub.sum(axis=0)).ravel() / nc if nc else np.zeros(vocab.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_support > 0, cluster_support / overall_support, 0.0)

        # Categories whose *community membership* is this one (the graph-native definition of the mission)
        member_categories = sorted(
            (cid for cid, c in community_map.items() if c == comm_id),
            key=lambda cid: -cluster_support[id_to_idx[cid]] if cid in id_to_idx else 0,
        )
        top_categories = [
            {"category": vocab.names[cid], "support": round(float(cluster_support[id_to_idx[cid]]), 4),
             "lift": round(float(lift[id_to_idx[cid]]), 3)}
            for cid in member_categories if cid in id_to_idx
        ]
        is_degenerate = bool(len(top_categories) and top_categories[0]["support"] >= DEGENERATE_SUPPORT_THRESHOLD)

        profiles.append({
            "community_id": comm_id,
            "n_baskets": nc,
            "share_pct": round(nc / n * 100, 2) if n else 0,
            "n_member_categories": len(member_categories),
            "avg_basket_value": round(float(baskets.loc[mask, "basket_value"].mean()), 2) if nc else None,
            "avg_basket_variety": round(float(baskets.loc[mask, "variety"].mean()), 2) if nc else None,
            "top_categories": top_categories,
            "is_degenerate": is_degenerate,
        })
    return profiles


def _write_report(name, scenario, n, vocab, G, n_communities, profiles, stab, min_lift, resolution, elapsed):
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi

    lines = []
    lines.append(f"# Graph community detection: {name}")
    lines.append("")
    lines.append(
        f"Category co-occurrence graph (PMI edges, min_lift={min_lift}) built from the exact "
        f"full {n:,}-basket population ({vocab.size} categories -> {G.number_of_nodes()} graph "
        f"nodes, {G.number_of_edges()} positive-lift edges). Louvain community detection "
        f"(resolution={resolution}) found {n_communities} communities at the reference seed; "
        f"every basket assigned by which community's categories it contains the most of. Run in "
        f"{elapsed:.1f}s total."
    )
    lines.append("")

    lines.append("## Community profiles")
    lines.append("")
    lines.append("| Community | Share | Avg value | Avg variety | # categories | Top categories (by support) | Degenerate? |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} ({c['support']*100:.0f}%)" for c in p["top_categories"][:5])
        flag = "**YES**" if p["is_degenerate"] else "no"
        lines.append(
            f"| {p['community_id']} | {p['share_pct']}% | {p['avg_basket_value']} | {p['avg_basket_variety']} | "
            f"{p['n_member_categories']} | {top} | {flag} |"
        )
    lines.append("")

    lines.append("## Stability across seeds (Louvain-only randomness - graph itself is fixed)")
    lines.append("")
    lines.append(
        f"{stab.n_runs} independent community-detection runs on the *same* PMI graph (built once "
        f"from the full population), each with a different Louvain seed. Unlike the CLARA "
        f"k-medoids and HDBSCAN stability tests (docs/stability/*.md), no basket sampling happens "
        f"here at all - this isolates how much Louvain's own internal randomness (traversal order, "
        f"tie-breaking) affects the result, as opposed to which baskets happened to get sampled."
    )
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    lines.append("")
    lines.append(f"Communities found per run: {stab.n_communities_per_run.tolist()}")
    lines.append("")
    lines.append(
        "Rule of thumb (same as the other stability reports): ARI/NMI > 0.9 = highly stable, "
        "0.7-0.9 = reasonably stable, 0.5-0.7 = moderate instability worth investigating, "
        "< 0.5 = largely re-deriving noise."
    )
    lines.append("")
    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_graph_missions.py {name} --n-runs {stab.n_runs}`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{name}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
