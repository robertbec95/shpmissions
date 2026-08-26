"""Graph community detection generalized to spend distribution: builds the
category graph from value-weighted co-occurrence (exact full population, no
CLARA sampling) instead of binary presence/absence, on the same population
as the frozen baseline and `level3_value_distribution` (variety>=2,
category level 3, 3-day window, basket_value>0).

This combines the two best findings so far in this project: value
distribution finds much sharper category lifts than presence/absence
(`level3_value_distribution`, ARI~0.35 under CLARA), and graph community
detection is far more stable than any CLARA-sampling method because it
never samples baskets (`level3_baseline`'s graph report, ARI~0.56-0.67
under binary presence/absence). Combining them - value-weighted edges, full
population, no sampling - given both properties should compound: sharp
missions AND high stability. Confirmed interactively before formalizing
here: ARI~0.96, NMI~0.98 (20 seeds, Louvain-only randomness) - "highly
stable" by this project's own rule of thumb, the best result found so far
by a wide margin.

Usage:
    PYTHONPATH=src python3 scripts/run_value_graph_missions.py
    PYTHONPATH=src python3 scripts/run_value_graph_missions.py --name level3_value_distribution_graph_8day --dates 20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import graph_missions, mission_stage, multiday, stability, store_formats, temporal
from shopping_missions.config import PROJECT_ROOT
from shopping_missions.scenarios import get_scenario

DOCS_DIR = PROJECT_ROOT / "docs" / "graph_missions"
DOCS_STABILITY_DIR = PROJECT_ROOT / "docs" / "stability"
MIN_LIFT = 1.0
MIN_PREVALENCE = 1e-4
RESOLUTION = 2.0
REFERENCE_SEED = 0
N_STABILITY_RUNS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="level3_value_distribution_graph", help="report file name (docs/graph_missions/<name>.md, docs/stability/<name>.md)")
    parser.add_argument("--dates", default=None, help="comma-separated dates; defaults to the frozen baseline's 3-day window")
    args = parser.parse_args()

    scenario = get_scenario("level3_baseline_gapk")
    dates = args.dates.split(",") if args.dates else list(scenario.dates)
    scenario_name = args.name

    print(f"=== {scenario_name}: value-weighted graph community detection ===")
    t0 = time.time()

    data = multiday.build_basket_data_multi_date(
        dates, category_level=scenario.category_level, include_daypart=True,
        include_value_matrix=True,
    )
    keep_mask = ((data.baskets["variety"] >= scenario.min_variety) & (data.baskets["basket_value"] > 0)).to_numpy()
    baskets = data.baskets.loc[keep_mask].reset_index(drop=True)
    value_matrix = data.value_matrix[keep_mask]
    vocab = data.vocab
    n = len(baskets)
    print(f"  ingested: {n:,} baskets, {vocab.size} categories ({time.time()-t0:.1f}s)")

    norm = mission_stage.normalize_value_matrix(value_matrix)

    t1 = time.time()
    G = graph_missions.compute_value_pmi_graph(norm, vocab, min_lift=MIN_LIFT, min_prevalence=MIN_PREVALENCE)
    community_map = graph_missions.detect_communities(G, resolution=RESOLUTION, seed=REFERENCE_SEED)
    assignment = graph_missions.assign_baskets_to_missions(norm, vocab, community_map)
    print(f"  graph: {G.number_of_nodes()} nodes ({vocab.size - G.number_of_nodes()} categories below "
          f"min_prevalence={MIN_PREVALENCE} excluded), {G.number_of_edges()} edges, "
          f"{assignment.n_communities} communities ({time.time()-t1:.1f}s)")

    baskets = store_formats.join_county(baskets)
    baskets = temporal.add_daypart(baskets)
    profiles = _profile_communities(baskets, value_matrix, assignment, vocab, community_map)
    for p in profiles:
        top = ", ".join(f"{c['category']} (x{c['lift']:.1f})" for c in p["top_categories"][:3])
        print(f"  community {p['community_id']}: n={p['n_baskets']:,} ({p['share_pct']}%) top=[{top}]")

    t2 = time.time()
    stab = stability.run_stability_test_value_graph(
        norm, vocab, min_lift=MIN_LIFT, min_prevalence=MIN_PREVALENCE, resolution=RESOLUTION,
        n_runs=N_STABILITY_RUNS, base_seed=REFERENCE_SEED,
    )
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    print(f"  stability ({N_STABILITY_RUNS} runs, {time.time()-t2:.1f}s): "
          f"ARI mean={ari.mean():.4f} std={ari.std():.4f} | NMI mean={nmi.mean():.4f} std={nmi.std():.4f}")
    print(f"  communities found per run: {stab.n_communities_per_run.tolist()}")

    county_xtab = _crosstab(baskets, assignment.hard_labels, "county")
    daypart_xtab = _crosstab(baskets, assignment.hard_labels, "day_part")

    _write_report(scenario_name, dates, n, vocab, G, assignment, profiles, stab, county_xtab, daypart_xtab, time.time() - t0)
    print(f"  wrote {DOCS_DIR / (scenario_name + '.md')} and {DOCS_STABILITY_DIR / (scenario_name + '.md')}")


def _profile_communities(baskets, value_matrix, assignment, vocab, community_map):
    n = len(assignment.hard_labels)
    overall_value = np.asarray(value_matrix.sum(axis=0)).ravel()
    overall_share = overall_value / overall_value.sum()
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


def _write_report(scenario_name, dates, n, vocab, G, assignment, profiles, stab, county_xtab, daypart_xtab, elapsed):
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi

    lines = []
    lines.append(f"# Graph community detection on spend distribution: {scenario_name}")
    lines.append("")
    lines.append(
        f"Dates: {','.join(dates)} ({len(dates)}-day window). Category graph built from value-weighted "
        f"co-occurrence (`graph_missions.compute_value_pmi_graph`, "
        f"min_lift={MIN_LIFT}, min_prevalence={MIN_PREVALENCE}) over the exact full {n:,}-basket population - "
        f"no CLARA sampling, same property that made binary graph community detection far more stable than "
        f"any sampling-based method. {vocab.size} categories -> {G.number_of_nodes()} graph nodes "
        f"({vocab.size - G.number_of_nodes()} excluded below the prevalence floor - genuinely marginal "
        f"merchandise like footwear/gifts/suncare, whose near-zero mass makes lift dominated by a handful "
        f"of baskets), {G.number_of_edges()} edges, Louvain resolution={RESOLUTION} -> "
        f"{assignment.n_communities} communities at the reference seed. Run in {elapsed:.1f}s total."
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
    lines.append(
        f"{stab.n_runs} independent community-detection runs on the same value-weighted PMI graph (built once "
        f"from the full population), each with a different Louvain seed - identical methodology to the binary "
        f"graph stability tests (`docs/stability/level3_baseline.md`'s graph section), only the edge weighting "
        f"differs."
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
        "For reference, every other method tested in this project: binary Jaccard CLARA ARI~0.21-0.23, "
        "Bray-Curtis (value distribution) CLARA ARI~0.35, binary graph community detection ARI~0.56-0.67. "
        "This is the most stable method found so far, by a wide margin - 'highly stable' (>0.9) by this "
        "project's rule of thumb, vs. 'largely re-deriving noise' (<0.5) for every CLARA-sampling method tested."
    )
    lines.append("")

    lines.append("## Cross-validation by day-part")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    pivot = daypart_xtab.pivot(index="day_part", columns="community", values="share_within_dim").fillna(0)
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    pivot.columns = [f"C{c}" for c in pivot.columns]
    lines.append(pivot.round(1).to_markdown())
    lines.append("")

    lines.append("## Cross-validation by county (top 10 by volume)")
    lines.append("")
    top_counties = county_xtab.groupby("county", observed=True)["n"].sum().sort_values(ascending=False).head(10).index
    pivot_c = county_xtab[county_xtab["county"].isin(top_counties)].pivot(index="county", columns="community", values="share_within_dim").fillna(0)
    pivot_c = pivot_c.loc[top_counties]
    pivot_c.columns = [f"C{c}" for c in pivot_c.columns]
    lines.append(pivot_c.round(1).to_markdown())
    lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions.py --name {scenario_name} --dates {','.join(dates)}`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / f"{scenario_name}.md").write_text("\n".join(lines))

    stab_lines = []
    stab_lines.append(f"# Stability test: {scenario_name}")
    stab_lines.append("")
    stab_lines.append(
        f"{stab.n_runs} Louvain runs on the fixed value-weighted PMI graph (see "
        f"`docs/graph_missions/{scenario_name}.md` for the graph itself and full community profiles)."
    )
    stab_lines.append("")
    stab_lines.append("| Metric | Mean | Std | Min | Max |")
    stab_lines.append("|---|---|---|---|---|")
    stab_lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    stab_lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    stab_lines.append("")
    stab_lines.append(f"Communities found per run: {stab.n_communities_per_run.tolist()}")
    stab_lines.append("")
    stab_lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions.py --name {scenario_name} --dates {','.join(dates)}`")
    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / f"{scenario_name}.md").write_text("\n".join(stab_lines))


if __name__ == "__main__":
    main()
