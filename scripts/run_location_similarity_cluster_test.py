"""Tests a coarser propagation strategy against the validated k-NN default:
instead of averaging each store's individually-ranked k=3 nearest catchment
analogs (`location_similarity.propagate_profile`), pre-cluster ALL stores
via Gower distance + k-medoids (`location_similarity.cluster_stores`) and
predict each store's profile as its cluster-mates' average
(`propagate_via_cluster`). Simpler to communicate to a retailer ("this
store is in catchment-cluster 2") at the cost of losing k-NN's per-store
ranking - this script checks whether that simplicity actually costs
accuracy, and how much, against both ground truths already established
(joint NMF, joint BigCLAM) so the comparison isn't tied to one detection
method's quirks.

Usage:
    PYTHONPATH=src python3 scripts/run_location_similarity_cluster_test.py
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

from shopping_missions import location_similarity as loc_sim
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "location_similarity"
CLUSTER_K_RANGE = [2, 3, 4, 5]
KNN_K = 3  # the established k-NN baseline to compare against


def main():
    print("=== Cluster-based (Gower + k-medoids) vs. k-NN propagation ===")
    t_all = time.time()

    print("\n--- Ground truths ---")
    t0 = time.time()
    _, nmf_meta, nmf_fit, _ = nmf_prop._refit_joint_nmf()
    store_ids = sorted(nmf_meta["loc_id"].unique())
    nmf_gt = nmf_prop._store_soft_prevalence_by_loc(nmf_meta, nmf_fit, store_ids)
    print(f"  joint NMF refit ({time.time()-t0:.1f}s)")

    t1 = time.time()
    _, bigclam_gt = run_bigclam_joint_missions.main()
    print(f"  joint BigCLAM refit ({time.time()-t1:.1f}s)")

    feature_table = pd.read_json(nmf_prop.FEATURE_TABLE_PATH, orient="index").loc[store_ids]
    distance = loc_sim.weighted_gower_distance(feature_table)
    store_names = nmf_prop._store_display_names(nmf_meta, store_ids)

    print(f"\n--- k-NN baseline (k={KNN_K}, already validated) ---")
    knn_nmf = _evaluate_knn(distance, nmf_gt, KNN_K)
    knn_bigclam = _evaluate_knn(distance, bigclam_gt, KNN_K)
    print(f"  NMF:     L1={knn_nmf['l1_mean']:.2f}  cos={knn_nmf['cos_mean']:.4f}  sp={knn_nmf['spearman_mean']:.3f}")
    print(f"  BigCLAM: L1={knn_bigclam['l1_mean']:.2f}  cos={knn_bigclam['cos_mean']:.4f}  sp={knn_bigclam['spearman_mean']:.3f}")

    print(f"\n--- Cluster-based sweep, C in {CLUSTER_K_RANGE} ---")
    cluster_rows = []
    cluster_assignments = {}
    for c in CLUSTER_K_RANGE:
        labels = loc_sim.cluster_stores(distance, c, random_state=RANDOM_STATE)
        cluster_assignments[c] = labels
        nmf_errs = _evaluate_cluster(distance, nmf_gt, labels)
        bigclam_errs = _evaluate_cluster(distance, bigclam_gt, labels)
        cluster_rows.append({"c": c, "nmf": nmf_errs, "bigclam": bigclam_errs})
        print(f"  C={c}  NMF: L1={nmf_errs['l1_mean']:.2f} cos={nmf_errs['cos_mean']:.4f} sp={nmf_errs['spearman_mean']:.3f}"
              f"   |  BigCLAM: L1={bigclam_errs['l1_mean']:.2f} cos={bigclam_errs['cos_mean']:.4f} sp={bigclam_errs['spearman_mean']:.3f}")

    best_nmf = min(cluster_rows, key=lambda r: r["nmf"]["l1_mean"])
    best_bigclam = min(cluster_rows, key=lambda r: r["bigclam"]["l1_mean"])
    print(f"\nBest cluster-based C for NMF: C={best_nmf['c']} (L1={best_nmf['nmf']['l1_mean']:.2f} vs k-NN {knn_nmf['l1_mean']:.2f})")
    print(f"Best cluster-based C for BigCLAM: C={best_bigclam['c']} (L1={best_bigclam['bigclam']['l1_mean']:.2f} vs k-NN {knn_bigclam['l1_mean']:.2f})")

    _write_report(store_names, knn_nmf, knn_bigclam, cluster_rows, cluster_assignments, time.time() - t_all)
    print(f"\nTotal runtime {time.time()-t_all:.1f}s")
    print(f"Wrote {DOCS_DIR / 'level3_location_similarity_cluster_test.md'}")


def _evaluate_knn(distance, ground_truth, k):
    l1s, coss, sps = [], [], []
    for s in ground_truth:
        pred = loc_sim.propagate_profile(distance, ground_truth, s, k=k)
        actual = np.array(ground_truth[s])
        l1s.append(float(np.abs(actual - pred).sum()))
        coss.append(float(actual @ pred / (np.linalg.norm(actual) * np.linalg.norm(pred))))
        sps.append(spearmanr(actual, pred).correlation)
    return {"l1_mean": float(np.mean(l1s)), "cos_mean": float(np.mean(coss)), "spearman_mean": float(np.mean(sps))}


def _evaluate_cluster(distance, ground_truth, cluster_labels):
    l1s, coss, sps = [], [], []
    for s in ground_truth:
        pred = loc_sim.propagate_via_cluster(distance, ground_truth, cluster_labels, s)
        actual = np.array(ground_truth[s])
        l1s.append(float(np.abs(actual - pred).sum()))
        coss.append(float(actual @ pred / (np.linalg.norm(actual) * np.linalg.norm(pred))))
        sps.append(spearmanr(actual, pred).correlation)
    return {"l1_mean": float(np.mean(l1s)), "cos_mean": float(np.mean(coss)), "spearman_mean": float(np.mean(sps))}


def _write_report(store_names, knn_nmf, knn_bigclam, cluster_rows, cluster_assignments, elapsed):
    lines = []
    lines.append("# Cluster-based (Gower + k-medoids) vs. k-NN propagation")
    lines.append("")
    lines.append(
        f"Tests whether pre-clustering all 6 stores via weighted Gower distance + k-medoids "
        f"(`location_similarity.cluster_stores`, reusing `mission_stage.kmedoids_fit` unchanged - it already "
        f"works on any precomputed distance matrix) and predicting each store's profile as its cluster-mates' "
        f"average is competitive with the validated k=3 nearest-neighbor default. Plain k-means could not be "
        f"used for this - it requires a centroid in the original feature space, which does not exist for mixed "
        f"categorical/continuous Gower-distance data; k-medoids picks an actual data point as each cluster's "
        f"center instead. Both ground truths (joint NMF, joint BigCLAM) tested identically. Total run time {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## k-NN baseline (k=3, already validated)")
    lines.append("")
    lines.append("| Ground truth | L1 | Cosine | Spearman |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Joint NMF | {knn_nmf['l1_mean']:.2f} | {knn_nmf['cos_mean']:.4f} | {knn_nmf['spearman_mean']:.3f} |")
    lines.append(f"| Joint BigCLAM | {knn_bigclam['l1_mean']:.2f} | {knn_bigclam['cos_mean']:.4f} | {knn_bigclam['spearman_mean']:.3f} |")
    lines.append("")

    lines.append("## Cluster-based sweep")
    lines.append("")
    lines.append("| Clusters (C) | NMF L1 | NMF cosine | NMF spearman | BigCLAM L1 | BigCLAM cosine | BigCLAM spearman |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in cluster_rows:
        n, b = r["nmf"], r["bigclam"]
        lines.append(f"| {r['c']} | {n['l1_mean']:.2f} | {n['cos_mean']:.4f} | {n['spearman_mean']:.3f} | {b['l1_mean']:.2f} | {b['cos_mean']:.4f} | {b['spearman_mean']:.3f} |")
    lines.append("")

    lines.append("## Cluster assignments by C")
    lines.append("")
    for c, labels in cluster_assignments.items():
        lines.append(f"**C={c}:**")
        groups = {}
        for s, cl in labels.items():
            groups.setdefault(cl, []).append(store_names[s])
        for cl, members in sorted(groups.items()):
            lines.append(f"- Cluster {cl}: {', '.join(members)}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    best_nmf = min(cluster_rows, key=lambda r: r["nmf"]["l1_mean"])
    best_bigclam = min(cluster_rows, key=lambda r: r["bigclam"]["l1_mean"])
    nmf_verdict = "beats" if best_nmf["nmf"]["l1_mean"] < knn_nmf["l1_mean"] else "does not beat"
    bigclam_verdict = "beats" if best_bigclam["bigclam"]["l1_mean"] < knn_bigclam["l1_mean"] else "does not beat"
    lines.append(
        f"Best cluster-based result for NMF ground truth is C={best_nmf['c']} (L1={best_nmf['nmf']['l1_mean']:.2f}), "
        f"which {nmf_verdict} the k-NN baseline (L1={knn_nmf['l1_mean']:.2f})."
    )
    lines.append(
        f"Best cluster-based result for BigCLAM ground truth is C={best_bigclam['c']} (L1={best_bigclam['bigclam']['l1_mean']:.2f}), "
        f"which {bigclam_verdict} the k-NN baseline (L1={knn_bigclam['l1_mean']:.2f})."
    )
    lines.append(
        "Cluster-based propagation forces every member of a group to share one predicted profile, discarding "
        "the fine-grained ranking within a group that k-NN preserves - at n=6 with clusters of 1-3 members, "
        "that's a real loss of resolution the k-NN approach doesn't have. The trade-off is interpretability: "
        "\"this store is in catchment-cluster 2\" is a much easier sentence for a retailer to act on than a "
        "specific nearest-3 weighted average, even where it costs some accuracy."
    )
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_cluster_test.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_location_similarity_cluster_test.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
