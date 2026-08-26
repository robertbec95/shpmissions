"""Location-similarity propagation applied to the joint NMF's topic
definitions (see `shopping_missions.location_similarity` and
`docs/topic_missions/level3_nmf_joint_all_baskets_2025.md`): predicts each
store's shopping-mission profile (topic prevalence) from its weighted-Gower
nearest analogs among the OTHER 5 analyzed stores, using the *same* fixed
topic definitions the joint NMF learned - then compares the prediction
against that store's own real profile.

This is explicitly a leave-one-out validation, not a demonstration on a
genuinely new/unanalyzed store: this project currently has real transaction
data for exactly 6 stores, all of which are already "analyzed" (fit into
the joint NMF). Framed honestly, this script answers "if this store's own
transaction data were withheld, how well would the location-similarity
pipeline reconstruct its mission mix from the other 5 stores' patterns" -
the appropriate confidence check before trusting this pipeline on a real
future store that has no transaction data to fit at all. For a genuinely
new store, the same propagation call (`location_similarity.propagate_profile`)
would run with `exclude=set()` (no held-out data to exclude) against the
full 6-store known set, once that new store's own catchment features have
been geocoded/pulled the same way as the 6 in `data/location_features/`.

Refits the joint NMF fresh (not loaded from a cached file) so the topic
definitions and store prevalence used here are guaranteed current -
`fit_nmf_missions` is deterministic (nndsvda init, fixed random_state),
confirmed to reproduce byte-identical results across reruns earlier in
this project.

Usage:
    PYTHONPATH=src python3 scripts/run_location_similarity_propagation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from scipy.stats import spearmanr

from shopping_missions import categories, ingest_v2, location_similarity as loc_sim, temporal, topic_missions
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "location_similarity"
FEATURE_TABLE_PATH = PROJECT_ROOT / "data" / "location_features" / "store_feature_table_v3.json"
N_TOPICS = 12  # the held-out-reconstruction-error elbow k established in run_nmf_joint_missions.py
K_SWEEP = [1, 2, 3, 4, 5]
REPORT_K = 3  # the validated default (see location_similarity module docstring)


def main():
    print("=== Location-similarity propagation of the joint NMF's mission profiles ===")
    t_all = time.time()

    print("\n--- Refitting the joint NMF (topic definitions + ground-truth store prevalence) ---")
    t0 = time.time()
    vocab, meta, fit, profiles = _refit_joint_nmf()
    print(f"  {N_TOPICS} topics, {len(meta):,} baskets ({time.time()-t0:.1f}s)")

    store_ids = sorted(meta["loc_id"].unique())
    ground_truth = _store_soft_prevalence_by_loc(meta, fit, store_ids)

    print("\n--- Loading location features + building weighted Gower distance ---")
    feature_table = pd.read_json(FEATURE_TABLE_PATH, orient="index")
    feature_table = feature_table.loc[store_ids]  # align order to ground_truth's key order
    distance = loc_sim.weighted_gower_distance(feature_table)
    print(f"  {len(store_ids)} stores, weights={loc_sim.DEFAULT_WEIGHTS}")

    print(f"\n--- k sweep (leave-one-out, all {len(store_ids)} stores) ---")
    sweep_rows = []
    for k in K_SWEEP:
        errs = _evaluate_k(distance, ground_truth, k)
        sweep_rows.append({"k": k, **errs})
        print(f"  k={k}: L1={errs['l1_mean']:.2f}  cosine={errs['cos_mean']:.4f}  spearman={errs['spearman_mean']:.3f}")
    sweep = pd.DataFrame(sweep_rows)

    print(f"\n--- Per-store propagation detail (k={REPORT_K}) ---")
    store_names = _store_display_names(meta, store_ids)
    per_store = []
    for s in store_ids:
        result = loc_sim.propagate_with_detail(distance, ground_truth, s, k=REPORT_K)
        actual = ground_truth[s]
        l1 = float(np.abs(np.array(actual) - result.predicted_profile).sum())
        cos = float(np.array(actual) @ result.predicted_profile / (
            np.linalg.norm(actual) * np.linalg.norm(result.predicted_profile)
        ))
        sp = spearmanr(actual, result.predicted_profile).correlation
        analog_names = [store_names[a] for a in result.nearest_analogs]
        print(f"  {store_names[s]:28s} <- {', '.join(analog_names)}  (L1={l1:.2f}, cosine={cos:.4f})")
        per_store.append({
            "loc_id": s, "name": store_names[s], "analogs": analog_names, "distances": result.distances,
            "actual": actual, "predicted": result.predicted_profile.tolist(),
            "l1": l1, "cosine": cos, "spearman": sp,
        })

    _write_report(profiles, store_names, sweep, per_store, time.time() - t_all)
    print(f"\nTotal runtime {time.time()-t_all:.1f}s")
    print(f"Wrote {DOCS_DIR / 'level3_location_similarity_propagation.md'}")


def _refit_joint_nmf():
    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)

    qty_parts, presence_parts, meta_parts = [], [], []
    for loc_id in sorted(stores):
        info = stores[loc_id]
        con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
        data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True, include_quantity_matrix=True)
        con.close()
        keep_mask = ((data.baskets["variety"] >= 1) & (data.baskets["basket_value"] > 0)).to_numpy()
        baskets = temporal.add_daypart(data.baskets.loc[keep_mask].reset_index(drop=True))
        baskets["loc_id"] = loc_id
        qty_parts.append(data.quantity_matrix[keep_mask])
        presence_parts.append(data.matrix[keep_mask])
        meta_parts.append(baskets[["basket_id", "loc_id", "basket_value"]])

    quantity_matrix = vstack(qty_parts).tocsr()
    presence_matrix = vstack(presence_parts).tocsr()
    meta = pd.concat(meta_parts, ignore_index=True)

    weighted = topic_missions.build_idf_weighted_matrix(quantity_matrix, presence_matrix)
    fit = topic_missions.fit_nmf_missions(weighted, N_TOPICS, random_state=RANDOM_STATE)
    profiles = topic_missions.profile_topics(weighted, fit, vocab, meta, top_n=10)
    return vocab, meta, fit, profiles


def _store_soft_prevalence_by_loc(meta: pd.DataFrame, fit, store_ids: list[str]) -> dict[str, list[float]]:
    meta = meta.copy()
    meta["soft"] = list(fit.soft_membership)
    out = {}
    for loc_id in store_ids:
        grp = meta.loc[meta["loc_id"] == loc_id, "soft"]
        out[loc_id] = (np.mean(np.stack(grp.to_numpy()), axis=0) * 100).tolist()
    return out


def _store_display_names(meta: pd.DataFrame, store_ids: list[str]) -> dict[str, str]:
    stores = ingest_v2.list_store_files()
    return {loc_id: f"{stores[loc_id]['format']} | {stores[loc_id]['county']}" for loc_id in store_ids}


def _evaluate_k(distance: pd.DataFrame, ground_truth: dict[str, list[float]], k: int) -> dict:
    l1s, coss, sps = [], [], []
    for s in ground_truth:
        pred = loc_sim.propagate_profile(distance, ground_truth, s, k=k)
        actual = np.array(ground_truth[s])
        l1s.append(float(np.abs(actual - pred).sum()))
        coss.append(float(actual @ pred / (np.linalg.norm(actual) * np.linalg.norm(pred))))
        sps.append(spearmanr(actual, pred).correlation)
    return {"l1_mean": float(np.mean(l1s)), "cos_mean": float(np.mean(coss)), "spearman_mean": float(np.mean(sps))}


def _topic_label(profile: dict, n: int = 3) -> str:
    cats = [c["category"].title() for c in profile["native_top_categories"][:n]]
    return " + ".join(cats) if cats else "(no category clears threshold)"


def _write_report(profiles, store_names, sweep, per_store, elapsed):
    lines = []
    lines.append("# Location-similarity propagation of the joint NMF's mission profiles")
    lines.append("")
    lines.append(
        f"Leave-one-out validation: for each of the 6 analyzed stores, predict its shopping-mission profile "
        f"(the joint NMF's {N_TOPICS}-topic prevalence vector - see `docs/topic_missions/level3_nmf_joint_all_baskets_2025.md`) "
        f"from the weighted-Gower nearest {REPORT_K} analogs among the OTHER 5 stores, then compare to that "
        f"store's own real profile. Distance weights: `{{'primary': 0.5, 'poi': 2.0, 'population': 4.0, 'age': 0.5}}` "
        f"(see `shopping_missions.location_similarity` module docstring for how these were chosen and the "
        f"flat-grid-search caveat). Total run time {elapsed:.1f}s."
    )
    lines.append("")
    lines.append(
        "**This is not a test on a genuinely new/unanalyzed store** - all 6 stores here already have real "
        "transaction data and were fit into the joint NMF. It answers a narrower, still load-bearing question: "
        "if a store's own data were withheld, how well does its catchment similarity to the other stores "
        "reconstruct its actual mission mix? That is the honest confidence check before trusting this pipeline "
        "on a store with no transaction data at all."
    )
    lines.append("")

    lines.append("## Topic definitions (reference)")
    lines.append("")
    lines.append("| Topic | Label (top 3 categories) | Overall hard share |")
    lines.append("|---|---|---|")
    for p in profiles:
        lines.append(f"| T{p['topic_id']} | {_topic_label(p)} | {p['hard_share_pct']}% |")
    lines.append("")

    lines.append(f"## k sweep (leave-one-out, averaged across all 6 stores)")
    lines.append("")
    lines.append("| k | L1 error | Cosine similarity | Spearman rho |")
    lines.append("|---|---|---|---|")
    for _, row in sweep.iterrows():
        marker = f" (report default)" if int(row["k"]) == REPORT_K else ""
        lines.append(f"| {int(row['k'])}{marker} | {row['l1_mean']:.2f} | {row['cos_mean']:.4f} | {row['spearman_mean']:.3f} |")
    lines.append("")

    lines.append(f"## Per-store propagation detail (k={REPORT_K})")
    lines.append("")
    for s in per_store:
        lines.append(f"### {s['name']}")
        lines.append("")
        analog_str = ", ".join(f"{a} (dist={d:.3f})" for a, d in zip(s["analogs"], s["distances"]))
        lines.append(f"Propagated from: {analog_str}")
        lines.append("")
        lines.append(f"Accuracy: L1={s['l1']:.2f}, cosine={s['cosine']:.4f}, spearman={s['spearman']:.3f}")
        lines.append("")
        lines.append("| Topic | Actual % | Predicted % | Diff |")
        lines.append("|---|---|---|---|")
        for t, p in enumerate(profiles):
            a, pr = s["actual"][t], s["predicted"][t]
            lines.append(f"| T{t} ({_topic_label(p, n=2)}) | {a:.2f}% | {pr:.2f}% | {pr-a:+.2f} |")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **n=6.** The k=3 default and the specific distance weights were both chosen at this sample size - "
        "re-validate if the analyzed-store set grows."
    )
    lines.append(
        "- **Leave-one-out, not a genuinely new store.** Every store used as a \"target\" here already has its "
        "own real transaction data; the test measures reconstruction quality against ground truth that exists, "
        "not blind extrapolation."
    )
    lines.append(
        "- **Distance weights came from a grid search with a flat likelihood surface** (many tied-best weight "
        "combinations) - the *direction* (population >> POI > primary attributes/age structure) is well-supported; "
        "the exact multipliers are one reasonable choice among several statistically indistinguishable ones."
    )
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_propagation.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_location_similarity_propagation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
