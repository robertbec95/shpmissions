"""Re-validates the location-similarity propagation pipeline on v5 of the
store feature table - v4's 9 columns (unchanged) plus 3 new real-data
columns from this session's national pulls: `log_catchment_population`
(WorldPop), `unemployment_share_pct` (INS), `gdp_per_capita_eur`
(Eurostat).

Grid search scope, deliberately narrow: v4's 4 group weights (primary,
poi, population, age) are HELD FIXED at their already-validated values
(0.5 / 2.0 / 4.0 / 0.5) rather than re-searched - re-optimizing all 6
groups at once over just 6 stores would be almost pure noise (the v4
search was already flagged as a flat/tied landscape with only 4 groups;
adding 2 more free parameters at the same fixed n=6 makes that guaranteed
worse, not better). Only the 2 NEW groups' weights are grid-searched
(catchment, economic), each over {0.25, 0.5, 1, 2, 4} - 25 combinations,
against leave-one-out k=3 L1 error using the joint NMF's cached
topic-prevalence ground truth (`store_topic_prevalence_ground_truth.json`
- reused directly rather than refit, since the fit is confirmed
deterministic/byte-identical across reruns).

Headline question this answers: does adding real unemployment, GDP, and
fine-grained catchment population data actually IMPROVE propagation
accuracy over the v4 baseline (L1=18.11, cosine=0.9750, spearman=0.851),
or does it just add noise? Reported plainly either way.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v5_validation.py`
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shopping_missions import location_similarity as loc_sim

V5_FILE = "data/location_features/store_feature_table_v5.json"
GROUND_TRUTH_FILE = "data/location_features/store_topic_prevalence_ground_truth.json"

V4_WEIGHTS = {"primary": 0.5, "poi": 2.0, "population": 4.0, "age": 0.5}
NEW_GROUPS = ["catchment", "economic"]
CANDIDATE_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]

FEATURE_GROUPS_V5 = {
    **loc_sim.FEATURE_GROUPS,
    "log_catchment_population": "catchment",
    "unemployment_share_pct": "economic",
    "gdp_per_capita_eur": "economic",
}
CATEGORICAL_COLUMNS_V5 = loc_sim.CATEGORICAL_COLUMNS  # no new categorical columns in v5


def leave_one_out_eval(X: pd.DataFrame, group_weights: dict, ground_truth: dict, k: int = 3) -> dict:
    distance = loc_sim.weighted_gower_distance(
        X, feature_groups=FEATURE_GROUPS_V5, group_weights=group_weights, categorical_columns=CATEGORICAL_COLUMNS_V5,
    )
    l1_errors, cosines, spearmans = [], [], []
    for store in X.index:
        actual = np.array(ground_truth[store])
        predicted = loc_sim.propagate_profile(distance, ground_truth, store, k=k)
        l1_errors.append(np.abs(actual - predicted).sum())
        cosine_sim = np.dot(actual, predicted) / (np.linalg.norm(actual) * np.linalg.norm(predicted) + 1e-12)
        cosines.append(cosine_sim)
        spearmans.append(spearmanr(actual, predicted).correlation)
    return {"L1": float(np.mean(l1_errors)), "cosine": float(np.mean(cosines)), "spearman": float(np.mean(spearmans))}


def main():
    v5 = json.load(open(V5_FILE))
    ground_truth = {k: v for k, v in json.load(open(GROUND_TRUTH_FILE)).items()}
    X = pd.DataFrame(v5).T
    for col in ["PC1", "PC2", "log_population", "pct_female", "pct_children_0_14", "pct_seniors_65plus",
                "log_catchment_population", "unemployment_share_pct", "gdp_per_capita_eur"]:
        X[col] = X[col].astype(float)

    print("=== v4 baseline (for reference, unchanged) ===")
    print("  L1=18.11  cosine=0.9750  spearman=0.851\n")

    print("=== Grid search: catchment weight x economic weight (v4 groups held fixed) ===")
    results = []
    for w_catch in CANDIDATE_WEIGHTS:
        for w_econ in CANDIDATE_WEIGHTS:
            weights = {**V4_WEIGHTS, "catchment": w_catch, "economic": w_econ}
            metrics = leave_one_out_eval(X, weights, ground_truth, k=3)
            results.append({"w_catchment": w_catch, "w_economic": w_econ, **metrics})

    df = pd.DataFrame(results).sort_values("L1")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest by L1: catchment={best.w_catchment}, economic={best.w_economic} "
          f"-> L1={best.L1:.2f}, cosine={best.cosine:.4f}, spearman={best.spearman:.3f}")

    n_tied = (df["L1"] <= best.L1 + 0.01).sum()
    print(f"Combinations tied (within 0.01) at the best L1: {n_tied} / {len(df)}")

    # sanity check: v5 with the two new groups effectively zeroed out should reproduce v4 exactly
    zeroed = leave_one_out_eval(X, {**V4_WEIGHTS, "catchment": 1e-9, "economic": 1e-9}, ground_truth, k=3)
    print(f"\nSanity check (new groups ~zeroed): L1={zeroed['L1']:.2f} (should match v4's 18.11 closely)")


if __name__ == "__main__":
    main()
