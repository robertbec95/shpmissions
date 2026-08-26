"""K-means clustering on the 100-location synthetic catchment dataset
(`data/location_features/synthetic/synthetic_catchment_100.csv`).

Unlike the project's real Gower-distance pipeline (mixed categorical +
continuous features, which k-means structurally cannot consume - no valid
centroid exists for a mixed-type distance), this synthetic table is mostly
continuous, so a literal k-means IS the right tool here - but only after
handling three things k-means silently gets wrong if ignored:

1. `urbanicity` is nominal (no numeric encoding is meaningful for
   Euclidean distance) - excluded from the feature matrix, kept aside to
   profile the resulting clusters against it post-hoc instead.
2. `parking_availability` is ordinal - encoded 0..3 (none..ample), which
   IS meaningful for Euclidean distance, unlike a one-hot nominal encoding
   would be.
3. Several columns are redundant/near-collinear by construction and would
   silently triple-count one underlying factor: `competitors_total_count`
   is the exact sum of the 4 tier counts (dropped); the 3 education shares
   and 3 occupation shares each sum to 100% (one column dropped from each
   triple - 2 of 3 shares carry the same information); the income
   percentiles are near-deterministic functions of the median plus a
   low-variance dispersion parameter (collapsed into one
   `income_inequality_ratio = p90/p10` feature alongside the median,
   instead of counting the "income" factor 3x).

All remaining features are z-scored (mandatory for k-means: unstandardized
income in RON would dominate percentages in 0-100 and counts in 0-10
purely from scale, not real importance). k is chosen by silhouette score,
swept 2..8.

Reproduce with: `PYTHONPATH=src python3 scripts/run_kmeans_synthetic_catchment.py`
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from shopping_missions.config import RANDOM_STATE

IN_CSV = "data/location_features/synthetic/synthetic_catchment_100.csv"

PARKING_ORDER = {"none": 0, "limited": 1, "moderate": 2, "ample": 3}

DROP_REDUNDANT = [
    "competitors_total_count",  # exact sum of the 4 tier columns
    "pct_education_primary",  # 3-way composition; secondary+tertiary retained
    "pct_other_of_employed",  # 3-way composition; white+blue collar retained
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["location_id", "urbanicity"]).copy()
    X["parking_availability"] = X["parking_availability"].map(PARKING_ORDER)

    X["income_inequality_ratio"] = X["household_income_p90_ron"] / X["household_income_p10_ron"]
    X = X.drop(columns=["household_income_p10_ron", "household_income_p90_ron"])
    X = X.drop(columns=DROP_REDUNDANT)
    return X


def sweep_k(X_scaled: np.ndarray, k_range=range(2, 9), random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN_CSV)
    X = build_feature_matrix(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sweep = sweep_k(X_scaled)
    print("=== k sweep (silhouette + inertia) ===")
    print(sweep.to_string(index=False))

    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette: {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    df["cluster"] = km.labels_

    print(f"\n=== Cluster sizes (k={best_k}) ===")
    print(df["cluster"].value_counts().sort_index().to_string())

    print("\n=== Cluster vs. urbanicity crosstab ===")
    print(pd.crosstab(df["cluster"], df["urbanicity"]).to_string())

    print("\n=== Per-cluster feature means (z-scored, top movers only) ===")
    Xz = pd.DataFrame(X_scaled, columns=X.columns)
    Xz["cluster"] = km.labels_
    cluster_means = Xz.groupby("cluster").mean()
    for c in cluster_means.index:
        top = cluster_means.loc[c].abs().sort_values(ascending=False).head(6).index
        print(f"\nCluster {c} (n={sum(km.labels_ == c)}):")
        for feat in top:
            print(f"  {feat}: {cluster_means.loc[c, feat]:+.2f} sd")

    df.to_csv("data/location_features/synthetic/synthetic_catchment_100_kmeans.csv", index=False)
    print("\nWrote data/location_features/synthetic/synthetic_catchment_100_kmeans.csv")


if __name__ == "__main__":
    main()
