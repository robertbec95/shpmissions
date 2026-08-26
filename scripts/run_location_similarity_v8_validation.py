"""Re-validates propagation on v8 - identical to v7 except the 9 raw
`banner_*` columns are replaced by 2 PCA-summarized components. Tests
directly whether PCA-summarizing fixes the "many sparse columns actively
hurt" failure mode found in v7 (zero of 125 weight combinations tied or
beat the v4 baseline there).

Same disciplined pattern: v4's original 4 groups + v5's catchment/
economic + v6's poi_detail/competition all held FIXED at their
established values. `growth`, `footfall_activity`, and `banner` (now 2
columns, not 9) are grid-searched together again - same 125-combination
sweep as v7, so the two results are directly comparable.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v8_validation.py`
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shopping_missions import location_similarity as loc_sim

V8_FILE = "data/location_features/store_feature_table_v8.json"
GROUND_TRUTH_FILE = "data/location_features/store_topic_prevalence_ground_truth.json"

FIXED_WEIGHTS = {
    "primary": 0.5, "poi": 2.0, "population": 4.0, "age": 0.5,
    "catchment": 1.0, "economic": 0.25, "poi_detail": 0.25, "competition": 0.25,
}
CANDIDATE_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]

FEATURE_GROUPS_V8 = {
    **loc_sim.FEATURE_GROUPS,
    "log_catchment_population": "catchment",
    "unemployment_share_pct": "economic",
    "gdp_per_capita_eur": "economic",
    "nearest_competitor_km": "competition",
    "poi_school_count": "poi_detail", "poi_kindergarten_count": "poi_detail",
    "poi_place_of_worship_count": "poi_detail", "poi_pharmacy_count": "poi_detail",
    "poi_bank_count": "poi_detail", "poi_competitor_supermarket_count": "poi_detail",
    "poi_competitor_convenience_count": "poi_detail", "poi_transit_stop_count": "poi_detail",
    "poi_mall_count": "poi_detail", "poi_anchor_institution_count": "poi_detail",
    "poi_event_venue_count": "poi_detail", "poi_ev_charging_count": "poi_detail",
    "poi_tourism_accommodation_count": "poi_detail",
    "population_cagr_pct": "growth", "unemployment_trend_pct": "growth",
    "school_enrollment_pct": "footfall_activity", "tourism_overnight_stays_per_capita": "footfall_activity",
    "log_car_ownership_county": "footfall_activity",
    "banner_PC1": "banner", "banner_PC2": "banner",
}
CATEGORICAL_COLUMNS_V8 = loc_sim.CATEGORICAL_COLUMNS


def leave_one_out_eval(X: pd.DataFrame, group_weights: dict, ground_truth: dict, k: int = 3) -> dict:
    distance = loc_sim.weighted_gower_distance(
        X, feature_groups=FEATURE_GROUPS_V8, group_weights=group_weights, categorical_columns=CATEGORICAL_COLUMNS_V8,
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
    v8 = json.load(open(V8_FILE))
    ground_truth = json.load(open(GROUND_TRUTH_FILE))
    X = pd.DataFrame(v8).T
    X = X.drop(columns=["pct_working_age"])  # exact linear function of 2 columns already in "age" - see v7 notes
    numeric_cols = [c for c, g in FEATURE_GROUPS_V8.items() if c not in CATEGORICAL_COLUMNS_V8]
    for col in numeric_cols:
        X[col] = X[col].astype(float)

    print("=== v4/v5/v6 baseline ===  L1=18.11  cosine=0.9750  spearman=0.851")
    print("=== v7 best (raw 9-column banner) ===  L1=18.95  cosine=0.9739  spearman=0.848\n")

    print("=== Grid search: growth x footfall_activity x banner (PCA'd, all else fixed) ===")
    results = []
    for w_growth in CANDIDATE_WEIGHTS:
        for w_foot in CANDIDATE_WEIGHTS:
            for w_banner in CANDIDATE_WEIGHTS:
                weights = {**FIXED_WEIGHTS, "growth": w_growth, "footfall_activity": w_foot, "banner": w_banner}
                metrics = leave_one_out_eval(X, weights, ground_truth, k=3)
                results.append({"w_growth": w_growth, "w_footfall": w_foot, "w_banner": w_banner, **metrics})

    df = pd.DataFrame(results).sort_values("L1")
    print(df.head(15).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest by L1: growth={best.w_growth}, footfall={best.w_footfall}, banner={best.w_banner} "
          f"-> L1={best.L1:.2f}, cosine={best.cosine:.4f}, spearman={best.spearman:.3f}")

    n_tied = (df["L1"] <= 18.11 + 0.01).sum()
    print(f"Combinations that tie or beat the v4/v5/v6 baseline (L1<=18.11): {n_tied} / {len(df)}")

    zeroed = leave_one_out_eval(X, {**FIXED_WEIGHTS, "growth": 1e-9, "footfall_activity": 1e-9, "banner": 1e-9}, ground_truth, k=3)
    print(f"Sanity check (new groups ~zeroed): L1={zeroed['L1']:.2f}")


if __name__ == "__main__":
    main()
