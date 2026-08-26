"""Re-validates propagation on v7 (v6's 27 columns + 14 new: population/
unemployment trend, school enrollment, tourism intensity, car ownership,
9 banner-level competitor counts). Ethnicity/religion/mother-tongue
excluded from this file entirely - kept as reference-only data per an
explicit user decision, not a candidate for the weighted pipeline.

Same disciplined pattern as v5/v6: all 8 previously-established group
weights held FIXED (v4's original 4 at 0.5/2.0/4.0/0.5, v5's
catchment/economic at 1.0/0.25, v6's poi_detail/competition at
0.25/0.25). Only the 3 NEW groups are grid-searched - "growth"
(population/unemployment trend), "footfall_activity" (enrollment/
tourism/car ownership), "banner" (9 chain-level counts) - 125
combinations, against leave-one-out k=3 accuracy on the joint NMF's
cached ground truth.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_v7_validation.py`
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shopping_missions import location_similarity as loc_sim

V7_FILE = "data/location_features/store_feature_table_v7.json"
GROUND_TRUTH_FILE = "data/location_features/store_topic_prevalence_ground_truth.json"

FIXED_WEIGHTS = {
    "primary": 0.5, "poi": 2.0, "population": 4.0, "age": 0.5,
    "catchment": 1.0, "economic": 0.25, "poi_detail": 0.25, "competition": 0.25,
}
CANDIDATE_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]

BANNER_COLS = [
    "banner_independent_other", "banner_mega_image", "banner_unnamed", "banner_carrefour", "banner_froo",
    "banner_la_doi_pasi", "banner_penny", "banner_auchan", "banner_annabella",
]

FEATURE_GROUPS_V7 = {
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
    **{col: "banner" for col in BANNER_COLS},
}
CATEGORICAL_COLUMNS_V7 = loc_sim.CATEGORICAL_COLUMNS


def leave_one_out_eval(X: pd.DataFrame, group_weights: dict, ground_truth: dict, k: int = 3) -> dict:
    distance = loc_sim.weighted_gower_distance(
        X, feature_groups=FEATURE_GROUPS_V7, group_weights=group_weights, categorical_columns=CATEGORICAL_COLUMNS_V7,
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
    v7 = json.load(open(V7_FILE))
    ground_truth = json.load(open(GROUND_TRUTH_FILE))
    X = pd.DataFrame(v7).T
    # pct_working_age = 100 - pct_children_0_14 - pct_seniors_65plus, an exact
    # linear function of two columns already in the "age" group - including
    # it too would silently triple-weight the same 2-degree-of-freedom age
    # signal, the same redundancy mistake already avoided elsewhere in this
    # project (dropped competitor_grocery/compositional columns). Kept in
    # the saved JSON as a convenient derived reference value, excluded here.
    X = X.drop(columns=["pct_working_age"])
    numeric_cols = [c for c, g in FEATURE_GROUPS_V7.items() if c not in CATEGORICAL_COLUMNS_V7]
    for col in numeric_cols:
        X[col] = X[col].astype(float)

    print("=== v4/v5/v6 baseline (for reference) ===")
    print("  L1=18.11  cosine=0.9750  spearman=0.851\n")

    print("=== Grid search: growth x footfall_activity x banner (all else fixed) ===")
    results = []
    for w_growth in CANDIDATE_WEIGHTS:
        for w_foot in CANDIDATE_WEIGHTS:
            for w_banner in CANDIDATE_WEIGHTS:
                weights = {**FIXED_WEIGHTS, "growth": w_growth, "footfall_activity": w_foot, "banner": w_banner}
                metrics = leave_one_out_eval(X, weights, ground_truth, k=3)
                results.append({"w_growth": w_growth, "w_footfall": w_foot, "w_banner": w_banner, **metrics})

    df = pd.DataFrame(results).sort_values("L1")
    print(df.head(15).to_string(index=False))
    print("...")
    print(df.tail(5).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest by L1: growth={best.w_growth}, footfall={best.w_footfall}, banner={best.w_banner} "
          f"-> L1={best.L1:.2f}, cosine={best.cosine:.4f}, spearman={best.spearman:.3f}")

    n_tied = (df["L1"] <= 18.11 + 0.01).sum()
    print(f"Combinations that tie or beat the v4/v5/v6 baseline (L1<=18.11): {n_tied} / {len(df)}")

    zeroed = leave_one_out_eval(X, {**FIXED_WEIGHTS, "growth": 1e-9, "footfall_activity": 1e-9, "banner": 1e-9}, ground_truth, k=3)
    print(f"\nSanity check (new groups ~zeroed): L1={zeroed['L1']:.2f} (should match 18.11 closely)")


if __name__ == "__main__":
    main()
