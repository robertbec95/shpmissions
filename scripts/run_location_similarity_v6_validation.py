"""Re-validates propagation on v6 (v5's 12 columns + the 9 on-demand
`catchment.py` attributes - 8 POI category counts and nearest-competitor
distance, minus the zero-variance grocery column).

Same disciplined pattern as the v5 validation: everything already
established stays FIXED (v4's 4 groups at 0.5/2.0/4.0/0.5, plus v5's
tied-best catchment/economic weights at 1.0/0.25 - the combo that tied
v4's baseline without beating or hurting it). Only the 2 NEW groups
("poi_detail": 8 counts, "competition": nearest_competitor_km) are
grid-searched, 25 combinations, against leave-one-out k=3 accuracy on the
joint NMF's cached ground truth - same reasoning as v5: searching all 8
groups' weights at once over 6 stores would be far more free parameters
than data points.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v6_validation.py`
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shopping_missions import location_similarity as loc_sim

V6_FILE = "data/location_features/store_feature_table_v6.json"
GROUND_TRUTH_FILE = "data/location_features/store_topic_prevalence_ground_truth.json"

FIXED_WEIGHTS = {"primary": 0.5, "poi": 2.0, "population": 4.0, "age": 0.5, "catchment": 1.0, "economic": 0.25}
CANDIDATE_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]

POI_DETAIL_COLS = [
    "poi_school_count", "poi_kindergarten_count", "poi_place_of_worship_count", "poi_pharmacy_count",
    "poi_bank_count", "poi_competitor_supermarket_count", "poi_competitor_convenience_count",
    "poi_transit_stop_count",
]

FEATURE_GROUPS_V6 = {
    **loc_sim.FEATURE_GROUPS,
    "log_catchment_population": "catchment",
    "unemployment_share_pct": "economic",
    "gdp_per_capita_eur": "economic",
    "nearest_competitor_km": "competition",
    **{col: "poi_detail" for col in POI_DETAIL_COLS},
}
CATEGORICAL_COLUMNS_V6 = loc_sim.CATEGORICAL_COLUMNS


def leave_one_out_eval(X: pd.DataFrame, group_weights: dict, ground_truth: dict, k: int = 3) -> dict:
    distance = loc_sim.weighted_gower_distance(
        X, feature_groups=FEATURE_GROUPS_V6, group_weights=group_weights, categorical_columns=CATEGORICAL_COLUMNS_V6,
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
    v6 = json.load(open(V6_FILE))
    ground_truth = json.load(open(GROUND_TRUTH_FILE))
    X = pd.DataFrame(v6).T
    numeric_cols = [c for c, g in FEATURE_GROUPS_V6.items() if c not in CATEGORICAL_COLUMNS_V6]
    for col in numeric_cols:
        X[col] = X[col].astype(float)

    print("=== v4/v5 baseline (for reference) ===")
    print("  L1=18.11  cosine=0.9750  spearman=0.851\n")

    print("=== Grid search: poi_detail weight x competition weight (all else fixed) ===")
    results = []
    for w_poi in CANDIDATE_WEIGHTS:
        for w_comp in CANDIDATE_WEIGHTS:
            weights = {**FIXED_WEIGHTS, "poi_detail": w_poi, "competition": w_comp}
            metrics = leave_one_out_eval(X, weights, ground_truth, k=3)
            results.append({"w_poi_detail": w_poi, "w_competition": w_comp, **metrics})

    df = pd.DataFrame(results).sort_values("L1")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest by L1: poi_detail={best.w_poi_detail}, competition={best.w_competition} "
          f"-> L1={best.L1:.2f}, cosine={best.cosine:.4f}, spearman={best.spearman:.3f}")

    n_tied = (df["L1"] <= 18.11 + 0.01).sum()
    print(f"Combinations that tie or beat the v4/v5 baseline (L1<=18.11): {n_tied} / {len(df)}")

    zeroed = leave_one_out_eval(X, {**FIXED_WEIGHTS, "poi_detail": 1e-9, "competition": 1e-9}, ground_truth, k=3)
    print(f"\nSanity check (new groups ~zeroed): L1={zeroed['L1']:.2f} (should match v5's 18.11 closely)")


if __name__ == "__main__":
    main()
