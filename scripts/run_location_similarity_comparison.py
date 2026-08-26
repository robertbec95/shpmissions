"""Head-to-head comparison: how well does weighted-Gower location-similarity
propagation reconstruct each store's mission profile, using TWO different
transaction-based mission-detection methods as ground truth - the joint
NMF (`run_nmf_joint_missions.py`, soft topic membership) and the joint
BigCLAM (`run_bigclam_joint_missions.py`, overlapping community affinity)?

Both methods produce ONE shared, per-store-comparable profile across all 6
stores (unlike per-store-only methods, which can't serve as propagation
ground truth at all - see `location_similarity` module docstring). Running
the identical propagation pipeline (same weighted-Gower distance, same
k-sweep) against both lets the two ground truths be compared on the exact
same downstream task, rather than on different, method-specific metrics
that aren't comparable to each other.

Usage:
    PYTHONPATH=src python3 scripts/run_location_similarity_comparison.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bigclam_joint_missions
import run_location_similarity_propagation as nmf_prop

from shopping_missions import ingest_v2, location_similarity as loc_sim
from shopping_missions.config import PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "docs" / "location_similarity"
K_SWEEP = [1, 2, 3, 4, 5]
REPORT_K = 3


def main():
    print("=== Location-similarity propagation: joint NMF vs. joint BigCLAM ===")
    t_all = time.time()

    print("\n--- Refitting joint NMF (ground truth A) ---")
    t0 = time.time()
    _, nmf_meta, nmf_fit, nmf_profiles = nmf_prop._refit_joint_nmf()
    store_ids = sorted(nmf_meta["loc_id"].unique())
    nmf_ground_truth = nmf_prop._store_soft_prevalence_by_loc(nmf_meta, nmf_fit, store_ids)
    print(f"  {nmf_prop.N_TOPICS} topics ({time.time()-t0:.1f}s)")

    print("\n--- Refitting joint BigCLAM (ground truth B) ---")
    t1 = time.time()
    bigclam_profiles, bigclam_ground_truth = run_bigclam_joint_missions.main()
    n_communities = len(bigclam_profiles)
    print(f"  {n_communities} communities ({time.time()-t1:.1f}s)")

    print("\n--- Weighted-Gower distance (shared across both) ---")
    feature_table = pd.read_json(nmf_prop.FEATURE_TABLE_PATH, orient="index").loc[store_ids]
    distance = loc_sim.weighted_gower_distance(feature_table)

    store_names = nmf_prop._store_display_names(nmf_meta, store_ids)

    print(f"\n--- k sweep, both ground truths ---")
    sweep_rows = []
    for k in K_SWEEP:
        nmf_errs = _evaluate_k(distance, nmf_ground_truth, k)
        bigclam_errs = _evaluate_k(distance, bigclam_ground_truth, k)
        sweep_rows.append({"k": k, "method": "joint NMF", **nmf_errs})
        sweep_rows.append({"k": k, "method": "joint BigCLAM", **bigclam_errs})
        print(f"  k={k}  NMF: L1={nmf_errs['l1_mean']:.2f} cos={nmf_errs['cos_mean']:.4f} sp={nmf_errs['spearman_mean']:.3f}"
              f"   |  BigCLAM: L1={bigclam_errs['l1_mean']:.2f} cos={bigclam_errs['cos_mean']:.4f} sp={bigclam_errs['spearman_mean']:.3f}")
    sweep = pd.DataFrame(sweep_rows)

    print(f"\n--- Per-store detail at k={REPORT_K} ---")
    per_store = []
    for s in store_ids:
        nmf_result = loc_sim.propagate_with_detail(distance, nmf_ground_truth, s, k=REPORT_K)
        bigclam_result = loc_sim.propagate_with_detail(distance, bigclam_ground_truth, s, k=REPORT_K)
        nmf_l1 = float(np.abs(np.array(nmf_ground_truth[s]) - nmf_result.predicted_profile).sum())
        bigclam_l1 = float(np.abs(np.array(bigclam_ground_truth[s]) - bigclam_result.predicted_profile).sum())
        print(f"  {store_names[s]:28s} NMF L1={nmf_l1:6.2f}   BigCLAM L1={bigclam_l1:6.2f}   {'<- BigCLAM better' if bigclam_l1 < nmf_l1 else '<- NMF better'}")
        per_store.append({"loc_id": s, "name": store_names[s], "nmf_l1": nmf_l1, "bigclam_l1": bigclam_l1})

    _write_report(sweep, per_store, nmf_profiles, bigclam_profiles, time.time() - t_all)
    print(f"\nTotal runtime {time.time()-t_all:.1f}s")
    print(f"Wrote {DOCS_DIR / 'level3_location_similarity_comparison.md'}")


def _evaluate_k(distance, ground_truth, k):
    l1s, coss, sps = [], [], []
    for s in ground_truth:
        pred = loc_sim.propagate_profile(distance, ground_truth, s, k=k)
        actual = np.array(ground_truth[s])
        l1s.append(float(np.abs(actual - pred).sum()))
        coss.append(float(actual @ pred / (np.linalg.norm(actual) * np.linalg.norm(pred))))
        sps.append(spearmanr(actual, pred).correlation)
    return {"l1_mean": float(np.mean(l1s)), "cos_mean": float(np.mean(coss)), "spearman_mean": float(np.mean(sps))}


def _write_report(sweep, per_store, nmf_profiles, bigclam_profiles, elapsed):
    lines = []
    lines.append("# Location-similarity propagation: joint NMF vs. joint BigCLAM ground truth")
    lines.append("")
    lines.append(
        f"The identical weighted-Gower k-NN propagation pipeline (`location_similarity`), run against two "
        f"different transaction-based mission-detection methods as ground truth: the joint NMF "
        f"({len(nmf_profiles)} soft topics) and the joint BigCLAM ({len(bigclam_profiles)} overlapping "
        f"communities), both fit once across all 6 stores' pooled data so both give one shared, per-store-"
        f"comparable profile. Same distance matrix, same k values, both cases - a genuine head-to-head on the "
        f"same downstream task. Total run time {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## k sweep, side by side")
    lines.append("")
    lines.append("| k | NMF L1 | NMF cosine | NMF spearman | BigCLAM L1 | BigCLAM cosine | BigCLAM spearman |")
    lines.append("|---|---|---|---|---|---|---|")
    nmf_rows = {r["k"]: r for r in sweep.to_dict("records") if r["method"] == "joint NMF"}
    bigclam_rows = {r["k"]: r for r in sweep.to_dict("records") if r["method"] == "joint BigCLAM"}
    for k in sorted(nmf_rows):
        n, b = nmf_rows[k], bigclam_rows[k]
        marker = " (report k)" if k == REPORT_K else ""
        lines.append(
            f"| {k}{marker} | {n['l1_mean']:.2f} | {n['cos_mean']:.4f} | {n['spearman_mean']:.3f} | "
            f"{b['l1_mean']:.2f} | {b['cos_mean']:.4f} | {b['spearman_mean']:.3f} |"
        )
    lines.append("")

    lines.append(f"## Per-store L1 error at k={REPORT_K}")
    lines.append("")
    lines.append("| Store | NMF L1 | BigCLAM L1 | Better |")
    lines.append("|---|---|---|---|")
    for s in per_store:
        better = "BigCLAM" if s["bigclam_l1"] < s["nmf_l1"] else "NMF"
        lines.append(f"| {s['name']} | {s['nmf_l1']:.2f} | {s['bigclam_l1']:.2f} | {better} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A lower propagation error under one ground truth doesn't necessarily mean that method's missions are "
        "\"more correct\" - it means that method's per-store profile varies more smoothly with catchment "
        "similarity (population, POI density, urban/rural, age structure) and less with idiosyncratic per-"
        "store noise. That smoothness is exactly the property this project's propagation pipeline depends on, "
        "so it is a meaningful criterion for choosing a production ground truth even though it is not the "
        "same thing as detection accuracy."
    )
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_comparison.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_location_similarity_comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
