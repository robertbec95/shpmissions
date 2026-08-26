"""Location-similarity propagation: extend a mission-identification method's
results from a subset of deeply-analyzed stores to the rest of a retailer's
network, on the assumption that stores with similar catchment
characteristics have similar shopping missions.

A modernized version of Applebaum's (1966) analog method - find existing
stores similar to a target, transfer their observed data to it - the
classic version relied on analyst judgment to pick analogs with no formal
validation step; this version quantifies similarity (weighted Gower
distance over store/catchment attributes) and empirically validates the
transfer (leave-one-out against this project's joint NMF topic-prevalence
profiles, the only method with a shared, directly-comparable profile
across stores - see [[shpmissions-nmf-joint-idf]]).

This module covers similarity computation and profile propagation only.
Raw feature *acquisition* (geocoding via Nominatim, POI/competitor counts
via the Overpass API, sociodemographics via Romania's INS TEMPO-Online API)
is NOT wrapped here - those were one-off pulls for this project's 6 known
stores, already cached to `data/location_features/*.json`, and wrapping
live external-API calls into an importable module risks surprising network
calls on import/test. See `data/location_features/store_feature_table_v3.json`
for the cached per-store feature table this module consumes, and the
`shpmissions-location-similarity` memory entry for how each column was
sourced and what to do to add a genuinely new (ungeocoded) store.

Feature-set history (v1-v3 are flat/unweighted Gower over progressively
richer features, all found to underperform the weighted v4 - see memory
for the L1-error progression):
    v1: format, locality_type, locality_type_l2 (categorical) + PCA-decorrelated
        POI/competitor density (PC1, PC2)
    v2: v1 + log(population), pct_female (from INS TEMPO-Online)
    v3: v2 + pct_children_0_14, pct_seniors_65plus (from INS TEMPO-Online,
        aggregated from 5-year age bands - INS has no built-in coarse bands)
    v4: v3's exact same 9 columns, but Gower-weighted (this module's default) -
        the best-validated configuration, L1=18.11 at k=3 vs v3's flat L1=20.30.

v4's weights were chosen from a grid search whose likelihood surface was
genuinely flat at n=6 (dozens of weight combinations tied at the same best
L1) - the search confidently supports the *direction* (population is the
strongest signal, POI density secondary, primary attributes and age
structure minor) but not the precise multipliers. Treat `DEFAULT_WEIGHTS`
as a reasonable choice from a tied cluster, not a fitted unique optimum,
and re-validate if this is ever extended past 6 analyzed stores."""

from dataclasses import dataclass, field

import gower
import numpy as np
import pandas as pd

from .config import RANDOM_STATE

# Column -> weight-group mapping for the v3/v4 feature table
# (`data/location_features/store_feature_table_v3.json`). Named by column,
# not position, so this stays correct if columns are ever reordered.
FEATURE_GROUPS: dict[str, str] = {
    "format": "primary", "locality_type": "primary", "locality_type_l2": "primary",
    "PC1": "poi", "PC2": "poi",
    "log_population": "population", "pct_female": "population",
    "pct_children_0_14": "age", "pct_seniors_65plus": "age",
}
CATEGORICAL_COLUMNS: set[str] = {"format", "locality_type", "locality_type_l2"}

# Grid-searched against leave-one-out k=3 L1 error over the 6 known stores'
# joint-NMF topic prevalence - see module docstring for the flat-landscape
# caveat. Direction (population > POI > primary/age) is the load-bearing
# finding; exact multipliers are one of several tied-best choices.
DEFAULT_WEIGHTS: dict[str, float] = {"primary": 0.5, "poi": 2.0, "population": 4.0, "age": 0.5}

DEFAULT_K = 3  # best-validated number of nearest analogs to average - see module docstring


def weighted_gower_distance(
    X: pd.DataFrame, feature_groups: dict[str, str] = FEATURE_GROUPS,
    group_weights: dict[str, float] = DEFAULT_WEIGHTS, categorical_columns: set[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """Gower distance (handles the categorical/continuous mix natively)
    with per-column weights derived from named feature groups, so a
    caller can reweight by group (e.g. "trust population more than POI
    density") without hand-building a raw per-column weight array.

    `gower.gower_matrix`'s `weight` parameter is per-column and the final
    distance is a weighted average normalized by the weight sum, so only
    the *relative* ratios between groups matter, not their absolute
    scale."""
    missing = set(X.columns) - set(feature_groups)
    if missing:
        raise ValueError(f"No feature-group assignment for columns: {sorted(missing)}")

    cat_mask = np.array([col in categorical_columns for col in X.columns])
    weight = np.array([group_weights[feature_groups[col]] for col in X.columns], dtype=float)

    dist = gower.gower_matrix(X, weight=weight, cat_features=cat_mask)
    return pd.DataFrame(dist, index=X.index, columns=X.index)


def propagate_profile(
    distance: pd.DataFrame, known_profiles: dict[str, list[float]], target: str,
    k: int = DEFAULT_K, exclude: set[str] | None = None,
) -> np.ndarray:
    """Predicts `target`'s profile (e.g. joint-NMF topic-prevalence vector)
    as the unweighted average of its `k` nearest analogs' known profiles,
    by weighted-Gower distance - the validated propagation rule from this
    project's leave-one-out testing (k=3-4 beats both a single nearest
    neighbor, too noisy at this sample size, and averaging every known
    store, which dilutes the signal with genuinely dissimilar stores).

    `exclude` defaults to just `{target}` (so a store never gets to "vote"
    on its own predicted profile, the standard leave-one-out setup used to
    validate this pipeline) - pass a larger set to additionally exclude
    stores you don't want contributing (e.g. other stores that are
    themselves only propagated estimates, not directly analyzed)."""
    if exclude is None:
        exclude = {target}
    candidates = {o: distance.loc[target, o] for o in known_profiles if o not in exclude}
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} candidate analog(s) available, need k={k}")
    nearest = sorted(candidates, key=candidates.get)[:k]
    return np.mean([known_profiles[o] for o in nearest], axis=0)


@dataclass
class PropagationResult:
    target: str
    nearest_analogs: list[str]  # the k stores actually used, nearest first
    distances: list[float]  # aligned to nearest_analogs
    predicted_profile: np.ndarray


def propagate_with_detail(
    distance: pd.DataFrame, known_profiles: dict[str, list[float]], target: str,
    k: int = DEFAULT_K, exclude: set[str] | None = None,
) -> PropagationResult:
    """Same prediction as `propagate_profile`, but also returns which
    analogs were used and at what distance - for reporting "this store's
    profile was propagated from these k neighbors", not just the number."""
    if exclude is None:
        exclude = {target}
    candidates = {o: distance.loc[target, o] for o in known_profiles if o not in exclude}
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} candidate analog(s) available, need k={k}")
    nearest = sorted(candidates, key=candidates.get)[:k]
    predicted = np.mean([known_profiles[o] for o in nearest], axis=0)
    return PropagationResult(
        target=target, nearest_analogs=nearest, distances=[candidates[o] for o in nearest],
        predicted_profile=predicted,
    )


def cluster_stores(distance: pd.DataFrame, k: int, random_state: int = RANDOM_STATE) -> dict[str, int]:
    """PAM/k-medoids clustering of stores on the same weighted-Gower
    distance matrix `propagate_profile` uses for k-NN - a coarser
    alternative propagation strategy: instead of each store's prediction
    depending on its own individually-ranked nearest analogs, every store
    in the same catchment cluster shares one predicted profile (simpler to
    explain to a retailer - "this store is in catchment-cluster 2" - at
    the cost of losing the fine-grained per-store ranking k-NN uses).

    Plain k-means cannot be used here: it requires computing a centroid
    (mean) in the original feature space, and there is no valid "average"
    of a mix of categorical and continuous Gower-distance features.
    k-medoids sidesteps this by picking an actual data point as each
    cluster's center - `mission_stage.kmedoids_fit` already implements
    this on any precomputed distance matrix (built originally for Jaccard-
    distance basket clustering), so it's reused unchanged here rather than
    reimplemented; Gower distance is just another precomputed matrix to it.

    Returns {loc_id: cluster_id}."""
    from . import mission_stage

    store_ids = list(distance.index)
    D = distance.loc[store_ids, store_ids].to_numpy()
    _, labels = mission_stage.kmedoids_fit(D, k, random_state=random_state)
    return {s: int(label) for s, label in zip(store_ids, labels)}


def propagate_via_cluster(
    distance: pd.DataFrame, known_profiles: dict[str, list[float]], cluster_labels: dict[str, int],
    target: str, exclude: set[str] | None = None,
) -> np.ndarray:
    """Predicts `target`'s profile as the average of its cluster-mates'
    known profiles (from `cluster_stores`), rather than `propagate_profile`'s
    individually-ranked k-NN average. `exclude` defaults to `{target}` (the
    standard leave-one-out setup - the clustering itself may legitimately
    use `target`'s catchment features, since those aren't the ground truth
    being predicted, but `target`'s own known profile is never allowed to
    vote on its own prediction).

    Falls back to the next-nearest OTHER cluster (by mean distance from
    `target` to that cluster's remaining members) if `target`'s own
    cluster has no members left after exclusion - a real edge case at
    n=6 once k gets large enough to produce near-singleton clusters."""
    if exclude is None:
        exclude = {target}
    target_cluster = cluster_labels[target]
    same_cluster = [
        s for s, c in cluster_labels.items()
        if c == target_cluster and s not in exclude and s in known_profiles
    ]
    if same_cluster:
        return np.mean([known_profiles[s] for s in same_cluster], axis=0)

    other_clusters = sorted(set(cluster_labels.values()) - {target_cluster})
    ranked = []
    for c in other_clusters:
        members = [s for s, cl in cluster_labels.items() if cl == c and s not in exclude and s in known_profiles]
        if members:
            avg_dist = float(np.mean([distance.loc[target, s] for s in members]))
            ranked.append((avg_dist, members))
    if not ranked:
        raise ValueError(f"No candidate analogs available for {target!r} under cluster-based propagation")
    ranked.sort(key=lambda pair: pair[0])
    fallback_members = ranked[0][1]
    return np.mean([known_profiles[s] for s in fallback_members], axis=0)


# Only the continuous catchment features - deliberately excludes the
# categorical ones (format, locality_type, locality_type_l2). A ridge
# regression with 6 training stores per leave-one-out fold (n-1, since the
# target is held out) cannot support many free parameters at all; the
# weighted-Gower work already found these continuous features (population,
# sex ratio, age structure, POI density) carry the dominant signal while
# categorical attributes matter least, so dropping the categoricals here
# keeps the feature count closer to what 5 training points can bear rather
# than compounding the small-n problem with one-hot columns on top.
NUMERIC_FEATURE_COLUMNS: list[str] = [
    "PC1", "PC2", "log_population", "pct_female", "pct_children_0_14", "pct_seniors_65plus",
]


@dataclass
class RegressionPropagationResult:
    target: str
    predicted_profile: np.ndarray  # (n_topics,)
    alphas: list[float]  # per-topic ridge regularization strength, chosen by inner LOO-CV
    coefficients: np.ndarray  # (n_topics, n_features), standardized (comparable magnitudes across features)
    feature_columns: list[str]


def propagate_via_regression(
    X: pd.DataFrame, known_profiles: dict[str, list[float]], target: str,
    feature_columns: list[str] = NUMERIC_FEATURE_COLUMNS, exclude: set[str] | None = None,
) -> RegressionPropagationResult:
    """Predicts `target`'s profile by fitting one ridge regression per
    topic/community - `topic_prevalence ~ catchment_features` - on the
    training stores, rather than averaging nearest analogs
    (`propagate_profile`) or cluster-mates (`propagate_via_cluster`). The
    payoff over those two: an explicit, inspectable relationship per topic
    (`coefficients`) - e.g. "this topic's prevalence rises with
    population" - not just a number with no stated reason behind it.

    This is a genuinely fragile fit, and that has to be said plainly, not
    footnoted: with `target` held out, only 5 stores remain to fit 6 free
    coefficients (even before regularization) per topic. Ridge regression
    (L2-shrinkage) is not optional here - unregularized OLS with 6
    predictors and 5 points is a saturated/degenerate system with a
    perfect, meaningless fit. `RidgeCV` with `cv=None` uses sklearn's
    efficient closed-form leave-one-out cross-validation to pick each
    topic's regularization strength from data alone (nested inside the
    outer per-store leave-one-out this function itself is called under,
    from a caller like `run_location_similarity_regression_test.py`) -
    principled for this sample size, but still a fit over single-digit
    data points, not a result to treat as precise.

    Predictions are clipped at 0 (prevalence cannot be negative) but not
    renormalized to sum to 100 - unlike `propagate_profile`'s average of
    real profiles, an extrapolated regression prediction has no guarantee
    of summing to the right total, and forcing one would hide rather than
    fix that."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    if exclude is None:
        exclude = {target}
    train_ids = [s for s in known_profiles if s not in exclude]
    if len(train_ids) < 2:
        raise ValueError(f"Only {len(train_ids)} training store(s) available - too few to fit a regression")

    X_train_raw = X.loc[train_ids, feature_columns].to_numpy(dtype=float)
    X_target_raw = X.loc[[target], feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_target = scaler.transform(X_target_raw)

    n_topics = len(next(iter(known_profiles.values())))
    y_train = np.array([known_profiles[s] for s in train_ids])  # (n_train, n_topics)

    # Wide enough that the selected alpha is very rarely pinned to either
    # edge of the grid (checked empirically: a narrower 1e-2..1e4 grid had
    # 3-8 of 12 topics per store landing exactly on its upper bound - the
    # model wanting more shrinkage than was on offer, silently capped
    # rather than actually resolved).
    alphas_grid = np.logspace(-2, 8, 40)
    predicted = np.zeros(n_topics)
    chosen_alphas = []
    coefs = []
    for t in range(n_topics):
        model = RidgeCV(alphas=alphas_grid, cv=None)  # cv=None -> exact closed-form LOO over train_ids
        model.fit(X_train, y_train[:, t])
        predicted[t] = model.predict(X_target)[0]
        chosen_alphas.append(float(model.alpha_))
        coefs.append(model.coef_.copy())

    predicted = np.clip(predicted, 0.0, None)
    return RegressionPropagationResult(
        target=target, predicted_profile=predicted, alphas=chosen_alphas,
        coefficients=np.array(coefs), feature_columns=list(feature_columns),
    )
