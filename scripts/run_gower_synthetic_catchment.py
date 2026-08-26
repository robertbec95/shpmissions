"""Weighted-nothing (uniform) Gower distance + k-medoids on the same
100-location synthetic catchment dataset used by
`run_kmeans_synthetic_catchment.py` - the mixed-type-correct counterpart
to that k-means run, for direct comparison.

Unlike that k-means run, Gower distance natively consumes the categorical
`urbanicity` column instead of discarding it - the whole point of using
Gower here. `gower.gower_matrix` needs a `cat_features` boolean mask;
`urbanicity` (nominal, 3 unordered levels) is the only column marked
categorical. `parking_availability` is ordinally encoded 0..3 (as in the
k-means script) rather than marked categorical, since Gower's categorical
handling is simple match/mismatch and would discard its order - encoding
it numerically lets Gower's continuous-column handling respect the order.

Uniform (unweighted) Gower is used deliberately, unlike the project's real
v4 weighted Gower - there, weights were grid-searched against a real
external validation target (joint-NMF leave-one-out propagation error).
No such target exists for these 100 fabricated locations, so there is
nothing to grid-search weights against; presenting invented weights as
tuned would be misleading. The same redundancy trimming as the k-means
script is still applied first (competitor-tier sum, compositional
education/occupation shares, income percentiles collapsed to a ratio) -
Gower averages every column with equal weight by default, so leaving
near-duplicate columns in would silently triple-count those factors here
too, for the same reason it would in Euclidean k-means.

Clustering on the resulting distance matrix uses k-medoids
(`mission_stage.kmedoids_fit`), the technique this project has used
throughout for precomputed/Gower distance matrices, since k-means has no
valid centroid for non-Euclidean, mixed-type distance.

Reproduce with: `PYTHONPATH=src python3 scripts/run_gower_synthetic_catchment.py`
"""

import gower
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from shopping_missions.config import RANDOM_STATE
from shopping_missions.mission_stage import kmedoids_fit

IN_CSV = "data/location_features/synthetic/synthetic_catchment_100.csv"
KMEANS_CSV = "data/location_features/synthetic/synthetic_catchment_100_kmeans.csv"

PARKING_ORDER = {"none": 0, "limited": 1, "moderate": 2, "ample": 3}

DROP_REDUNDANT = [
    "competitors_total_count",
    "pct_education_primary",
    "pct_other_of_employed",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["location_id"]).copy()
    X["parking_availability"] = X["parking_availability"].map(PARKING_ORDER)

    X["income_inequality_ratio"] = X["household_income_p90_ron"] / X["household_income_p10_ron"]
    X = X.drop(columns=["household_income_p10_ron", "household_income_p90_ron"])
    X = X.drop(columns=DROP_REDUNDANT)
    return X


def sweep_k(D: np.ndarray, k_range=range(2, 9), random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        _, labels = kmedoids_fit(D, k, random_state=random_state)
        sil = silhouette_score(D, labels, metric="precomputed")
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN_CSV)
    X = build_feature_matrix(df)

    cat_mask = np.array([col == "urbanicity" for col in X.columns])
    D = gower.gower_matrix(X, cat_features=cat_mask)

    sweep = sweep_k(D)
    print("=== k sweep (silhouette, Gower + k-medoids) ===")
    print(sweep.to_string(index=False))

    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette: {best_k}")

    medoids, labels = kmedoids_fit(D, best_k, random_state=RANDOM_STATE)
    df["gower_cluster"] = labels

    print(f"\n=== Cluster sizes (k={best_k}) ===")
    print(df["gower_cluster"].value_counts().sort_index().to_string())

    print("\n=== Cluster vs. urbanicity crosstab ===")
    print(pd.crosstab(df["gower_cluster"], df["urbanicity"]).to_string())

    print("\n=== Medoid locations (the actual representative point per cluster) ===")
    for c, m in zip(range(best_k), medoids):
        row = df.iloc[m]
        print(f"Cluster {c}: location_id={row.location_id}, urbanicity={row.urbanicity}, "
              f"pop_density={row.population_density_km2:.0f}, income_median={row.household_income_median_ron:.0f}, "
              f"tertiary_edu={row.pct_education_tertiary:.1f}%")

    try:
        kmeans_df = pd.read_csv(KMEANS_CSV)
        merged = df.merge(kmeans_df[["location_id", "cluster"]], on="location_id")
        print("\n=== Gower/k-medoids cluster vs. earlier k-means cluster crosstab ===")
        print(pd.crosstab(merged["gower_cluster"], merged["cluster"]).to_string())
    except FileNotFoundError:
        print("\n(no prior k-means output found to cross-tabulate against)")

    df.to_csv("data/location_features/synthetic/synthetic_catchment_100_gower.csv", index=False)
    print("\nWrote data/location_features/synthetic/synthetic_catchment_100_gower.csv")


if __name__ == "__main__":
    main()
