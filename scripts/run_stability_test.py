"""Run cluster stability testing against a named scenario's population:
repeat Stage 2 clustering N times with different random seeds, measure
pairwise ARI/NMI between labelings, and report cluster-size variability
across runs.

Usage:
    PYTHONPATH=src python3 scripts/run_stability_test.py level3_8day --n-runs 20
    PYTHONPATH=src python3 scripts/run_stability_test.py level3_baseline level3_8day
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import multiday, stability
from shopping_missions.scenarios import get_scenario

DOCS_STABILITY_DIR = Path(__file__).resolve().parents[1] / "docs" / "stability"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="scenario name(s) from scenarios.py to stability-test")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    args = parser.parse_args()

    for name in args.scenarios:
        run_for_scenario(name, args.n_runs, args.base_seed)


def run_for_scenario(name: str, n_runs: int, base_seed: int):
    scenario = get_scenario(name)
    print(f"\n=== Stability test: {name} (k={scenario.k}, n_runs={n_runs}) ===")
    t0 = time.time()

    data = multiday.build_basket_data_multi_date(
        list(scenario.dates), category_level=scenario.category_level, include_daypart=False
    )
    keep_mask = (data.baskets["variety"] >= scenario.min_variety).to_numpy()
    matrix = data.matrix[keep_mask]
    n_baskets = matrix.shape[0]
    print(f"  ingested + filtered: {n_baskets:,} baskets ({time.time()-t0:.1f}s)")

    t1 = time.time()
    result = stability.run_stability_test(
        matrix, k=scenario.k, sample_size=scenario.sample_size, n_runs=n_runs, base_seed=base_seed
    )
    print(f"  {n_runs} clustering runs completed ({time.time()-t1:.1f}s)")

    ari = result.off_diagonal_ari
    nmi = result.off_diagonal_nmi
    print(f"  ARI: mean={ari.mean():.4f} min={ari.min():.4f} max={ari.max():.4f} std={ari.std():.4f}")
    print(f"  NMI: mean={nmi.mean():.4f} min={nmi.min():.4f} max={nmi.max():.4f} std={nmi.std():.4f}")

    size_stats = result.cluster_size_shares.agg(["mean", "std", "min", "max"]).T
    print(size_stats.round(2).to_string())

    _write_report(name, scenario, n_baskets, result)
    print(f"  wrote {DOCS_STABILITY_DIR / (name + '.md')}")


def _write_report(name, scenario, n_baskets, result: stability.StabilityResult):
    ari = result.off_diagonal_ari
    nmi = result.off_diagonal_nmi
    size_stats = result.cluster_size_shares.agg(["mean", "std", "min", "max"]).T

    lines = []
    lines.append(f"# Stability test: {name}")
    lines.append("")
    lines.append(
        f"{result.n_runs} independent Stage 2 clustering runs on the same {n_baskets:,}-basket "
        f"population (category_level={scenario.category_level}, k={result.k}, "
        f"sample_size={result.sample_size}), each with a different random seed - varying which "
        f"6,000 baskets get CLARA-sampled for the k-medoids fit, and the k-medoids++ "
        f"initialization. Pairwise Adjusted Rand Index (ARI) and Normalized Mutual Information "
        f"(NMI) computed between every pair of the resulting full-population labelings "
        f"({result.n_runs * (result.n_runs - 1) // 2} pairs). Both are permutation-invariant, "
        f"so cluster-ID relabeling between runs doesn't affect the score - only genuine "
        f"agreement/disagreement about which baskets group together does."
    )
    lines.append("")
    lines.append("## Agreement between runs")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    lines.append("")
    lines.append(
        "Rule of thumb: ARI/NMI > 0.9 = highly stable, 0.7-0.9 = reasonably stable, "
        "0.5-0.7 = moderate instability worth investigating, < 0.5 = the clustering is largely "
        "re-deriving noise from sample to sample, not a repeatable structure."
    )
    lines.append("")
    lines.append("## Cluster size variability across runs (% share of baskets)")
    lines.append("")
    lines.append(size_stats.round(2).to_markdown())
    lines.append("")
    lines.append(
        "A cluster whose share swings widely (high std, or min/max far apart) across runs is "
        "one whose existence/size depends heavily on which particular sample was drawn - not "
        "something to treat as a fixed, known percentage of the business."
    )
    lines.append("")
    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_stability_test.py {name} --n-runs {result.n_runs}`")

    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / f"{name}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
