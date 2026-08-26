"""Step E - Stage 2: Jaccard distance + k-centroid (k-medoids) clustering.

Full pairwise Jaccard clustering does not scale to the population of
"relatively concrete" baskets (hundreds of thousands, and basket category
signatures barely deduplicate at this category granularity - see
docs/mvp-plan.md). Instead we use a CLARA-style approach:

  1. Draw a random sample of baskets small enough for an O(n^2) distance
     matrix (a few thousand).
  2. Run k-medoids (PAM) on that sample to find k representative "mission"
     centroids.
  3. Assign *every* relatively-concrete basket to its nearest centroid via a
     single vectorized Jaccard computation against just the k medoids
     (O(n*k), trivial even for millions of baskets).

This is a standard, well-understood approximation to full PAM and is exactly
what CLARA (Kaufman & Rousseeuw) was designed for.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from .cluster_selection import gap_statistic
from .config import DEFAULT_K, DEFAULT_SAMPLE_SIZE, RANDOM_STATE


def _jaccard_distance_matrix(X: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pairwise (weighted) Jaccard distance for a dense binary (n x d) array.

    With `weights=None`, plain Jaccard. With per-category `weights` (shape
    (d,)), computes the weighted generalization - for binary x, y this is
    sum_i w_i*min(x_i,y_i) / sum_i w_i*max(x_i,y_i), which since x_i,y_i are
    0/1 reduces to weighted-intersection/weighted-union with
    min(x_i,y_i)=x_i*y_i and max(x_i,y_i)=x_i+y_i-x_i*y_i. Used to downweight
    ubiquitous "staple" categories (see `compute_idf_weights`) the way
    stop-word downweighting works in text clustering - a category present in
    almost every basket contributes little to distinguishing baskets, so it
    shouldn't count as much as a rarer, more discriminating one."""
    X = X.astype(np.float64)
    if weights is None:
        intersection = X @ X.T
        row_sums = X.sum(axis=1)
    else:
        Xw = X * weights[None, :]
        intersection = X @ Xw.T
        row_sums = Xw.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    # weighted intersection/union are sums of floats rather than integers, so
    # a same-basket pair (similarity exactly 1) can come out as 1 + 1e-16 -
    # clip rather than let that tiny negative distance reach silhouette_score,
    # which rejects any negative value in a precomputed distance matrix.
    return np.clip(1.0 - similarity, 0.0, None)


def compute_idf_weights(matrix: csr_matrix, min_support: float = 0.01) -> np.ndarray:
    """Per-category weight w_i = log(n / df_i) - standard inverse document
    frequency, treating each basket as a "document" and each category as a
    "term". A category present in every basket gets weight 0 (no
    discriminating power under Jaccard); rarer categories get proportionally
    larger weight. Computed from the full population, not a CLARA sample, so
    the weighting is stable across runs regardless of which baskets get
    sampled for the k-medoids fit.

    `min_support` floors df before the log: 25 of 99 level-3 categories have
    <0.1% prevalence (near-noise/miscoded SKUs, e.g. a handful of GIFTS or
    FOOTWEAR line items), and raw idf gives those up to 14x weight - enough
    for one idiosyncratic rare category to dominate a basket's distance to
    everything else, the opposite of the intended effect (downweighting
    ubiquitous staples, not upweighting noise). Categories at or above
    `min_support` get their real idf value; everything rarer is flattened to
    the same capped weight, log(1/min_support)."""
    n = matrix.shape[0]
    df = np.asarray(matrix.sum(axis=0)).ravel()
    df_floor = np.maximum(df, min_support * n)
    with np.errstate(divide="ignore"):
        w = np.log(n / df_floor)
    return np.where(df > 0, w, 0.0)


# --- Bray-Curtis: clustering on a basket's spend *distribution* across
# categories (continuous), rather than category presence/absence (binary).
# Bray-Curtis is the natural generalization of Jaccard to non-negative
# weighted data - for binary 0/1 input the two are closely related (both are
# intersection/union-family measures), so this keeps the same conceptual
# shape as the paper's method while answering a different question: not
# "which categories did this basket touch" but "how did this basket's money
# split across the categories it touched".


def normalize_value_matrix(value_matrix: csr_matrix) -> csr_matrix:
    """Row-normalizes a basket x category value matrix so each basket's row
    sums to 1 - converts raw RON spend into a spend *distribution* (what
    fraction of this basket's value went to each category), so two baskets
    with the same shopping pattern but different total spend (e.g. a
    single-person top-up vs a family's identical-ratio bigger shop) are
    treated as similar rather than distant. Without this, Bray-Curtis on raw
    values partly re-encodes basket size, which is already captured by
    `variety`/`basket_value` elsewhere and isn't the mission signal being
    tested here."""
    row_sums = np.asarray(value_matrix.sum(axis=1)).ravel()
    row_sums = np.where(row_sums > 0, row_sums, 1.0)  # guard only - every variety>=1 basket has positive value
    return diags(1.0 / row_sums) @ value_matrix


def _bray_curtis_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Bray-Curtis dissimilarity for a dense non-negative (n x d)
    array: BC(a,b) = sum_i|a_i-b_i| / sum_i(a_i+b_i). Computed as
    L1(a,b)/(sum(a)+sum(b)) via `cdist(..., "cityblock")` rather than a
    direct elementwise-min formula, since there's no matmul shortcut for L1
    distance the way binary Jaccard has for intersection (min(x,y)=x*y only
    holds for 0/1 data) - `cdist` is still fast at CLARA sample sizes
    (~1500-6000 rows) since it's a single vectorized C call."""
    X = X.astype(np.float64)
    l1 = cdist(X, X, metric="cityblock")
    row_sums = X.sum(axis=1)
    denom = row_sums[:, None] + row_sums[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        bc = np.where(denom > 0, l1 / denom, 0.0)
    return bc


def _assign_to_medoids_bray_curtis(matrix: csr_matrix, medoid_vectors: np.ndarray):
    """Full-population nearest-medoid assignment under Bray-Curtis. Avoids
    ever densifying `matrix` (millions of rows): each medoid vector is
    itself sparse (a single basket's category profile, typically ~4-8
    nonzero categories out of ~99), so for the categories where a medoid is
    zero, |x_i - 0| = x_i (x is non-negative) needs no per-row work beyond a
    sum; only the medoid's own small nonzero support needs an actual dense
    per-category subtraction, extracted as a thin (n, |support|) slice."""
    n = matrix.shape[0]
    k = medoid_vectors.shape[0]
    matrix_csc = matrix.tocsc()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    medoid_sums = medoid_vectors.sum(axis=1)

    l1 = np.zeros((n, k))
    for j in range(k):
        support = np.nonzero(medoid_vectors[j])[0]
        if len(support) == 0:
            l1[:, j] = row_sums
            continue
        sub = np.asarray(matrix_csc[:, support].todense())
        on_support = np.abs(sub - medoid_vectors[j, support][None, :]).sum(axis=1)
        off_support = row_sums - sub.sum(axis=1)
        l1[:, j] = on_support + off_support

    denom = row_sums[:, None] + medoid_sums[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(denom > 0, l1 / denom, 0.0)
    labels = np.argmin(distance, axis=1)
    min_distance = distance[np.arange(n), labels]
    return labels, min_distance


def _kmedoids_dispersion_bray_curtis(X, k: int, seed: int):
    """Same dispersion measure as `_kmedoids_dispersion`, under Bray-Curtis."""
    if X is None or len(X) < k:
        return np.inf, None
    D = _bray_curtis_distance_matrix(np.asarray(X, dtype=np.float64))
    medoids, labels = kmedoids_fit(D, k, random_state=seed % (2**32 - 1))
    Wk = 0.0
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) <= 1:
            continue
        Wk += D[np.ix_(members, members)].sum() / (2 * len(members))
    return float(Wk), labels


def select_k_bray_curtis(
    X_sample: np.ndarray, k_range=range(2, 12), B: int = 10, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Gap statistic + silhouette for the number of clusters under
    Bray-Curtis, mirroring `select_k`. `X_sample` should be a dense
    non-negative (ideally already row-normalized via
    `normalize_value_matrix`, so every row sums to 1) sample.

    Reference (null) datasets are built by independently permuting each
    category's column - this preserves each category's exact marginal value
    distribution (including its exact sparsity/zero pattern, since
    permutation only reorders, never invents values) while destroying the
    real cross-category co-occurrence structure that clustering would be
    picking up on. Each reference row is then renormalized to sum to 1,
    matching the real (normalized) data - an earlier version of this
    function resampled independently per column without that renormalization
    step, so reference "baskets" weren't valid spend distributions the way
    real ones are, biasing the reference dispersion comparison.

    Independent per-column shuffling of ~99 sparse columns (each basket
    typically touches ~4) leaves a non-trivial chance - not the "very rare"
    this originally assumed - that any single row lands on zero in every
    column: with typical per-category presence around a few percent,
    roughly product(1-presence_rate) per row, which multiplied out over 99
    columns and ~1500 rows means *some* empty row is close to guaranteed
    every draw. Re-shuffling the whole matrix until every one of 1500 rows
    is simultaneously non-empty was hanging for exactly that reason (a
    single fixed retry probability per row compounds to "will not finish" at
    n=1500). Fixed by patching only the specific empty rows directly."""
    X = np.asarray(X_sample, dtype=np.float64)
    n, d = X.shape
    nonzero_by_col = [X[X[:, j] > 0, j] for j in range(d)]
    col_weights = np.array([len(v) for v in nonzero_by_col], dtype=float)
    col_weights = col_weights / col_weights.sum()

    def generate_reference(seed):
        rng = np.random.default_rng(seed)
        ref = X.copy()
        for j in range(d):
            rng.shuffle(ref[:, j])
        row_sums = ref.sum(axis=1)
        for i in np.where(row_sums <= 0)[0]:
            j = rng.choice(d, p=col_weights)
            ref[i, j] = rng.choice(nonzero_by_col[j])
        row_sums = ref.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # safety net only, shouldn't trigger
        return ref / row_sums[:, None]

    df = gap_statistic(X, k_range, _kmedoids_dispersion_bray_curtis, generate_reference, B=B, random_state=random_state)

    silhouettes = []
    for k in k_range:
        D = _bray_curtis_distance_matrix(X)
        _, labels = kmedoids_fit(D, k, random_state=random_state)
        silhouettes.append(silhouette_score(D, labels, metric="precomputed") if len(set(labels)) > 1 else float("nan"))
    df["silhouette"] = silhouettes
    return df


def _kmedoids_pp_init(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++-style seeding, adapted to a precomputed distance matrix."""
    n = D.shape[0]
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        nearest = D[:, medoids].min(axis=1)
        probs = nearest**2
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(n, 1.0 / n)
        medoids.append(int(rng.choice(n, p=probs)))
    return np.array(medoids)


def kmedoids_fit(D: np.ndarray, k: int, max_iter: int = 100, random_state: int = RANDOM_STATE):
    """Partitioning-Around-Medoids on a precomputed distance matrix.

    Returns (medoid_indices, labels).
    """
    rng = np.random.default_rng(random_state)
    medoids = _kmedoids_pp_init(D, k, rng)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()
        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                continue  # keep previous medoid for an emptied cluster
            sub = D[np.ix_(members, members)]
            total_cost = sub.sum(axis=1)
            new_medoids[cluster_id] = members[np.argmin(total_cost)]
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    labels = np.argmin(D[:, medoids], axis=1)
    return medoids, labels


def _kmedoids_dispersion(X, k: int, seed: int, weights: np.ndarray | None = None):
    """W_k = sum_r [D_r / (2 n_r)], D_r = sum of pairwise Jaccard distances
    within cluster r - the general gap-statistic dispersion measure, valid
    for any distance (here: Jaccard, via a fresh distance matrix per call
    since X differs for the observed sample vs. each reference draw)."""
    if X is None or len(X) < k:
        return np.inf, None
    D = _jaccard_distance_matrix(np.asarray(X, dtype=np.float64), weights=weights)
    medoids, labels = kmedoids_fit(D, k, random_state=seed % (2**32 - 1))
    Wk = 0.0
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) <= 1:
            continue
        Wk += D[np.ix_(members, members)].sum() / (2 * len(members))
    return float(Wk), labels


def select_k(
    X_sample: np.ndarray, k_range=range(2, 12), B: int = 10, random_state: int = RANDOM_STATE,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Gap statistic + silhouette diagnostics for the number of mission
    clusters, following Tibshirani, Walther & Hastie (2001) - the same
    method the paper uses (Section 5.5, via R's `clusGap`) to arrive at
    k=7 for its supermarket-pilot store.

    `X_sample` should be a small (~1000-2000 row) dense binary sample of
    relatively-concrete baskets, e.g. `mission_stage`'s own CLARA sample -
    this fits B+1 k-medoids models per k, each needing an O(n^2) Jaccard
    distance matrix, so keep the sample well below the ~6000 used for the
    final `fit_and_assign_missions` centroid fit.

    Reference (null) datasets are drawn by sampling each category
    independently at its observed marginal rate (Bernoulli(p_j) per column,
    matching the observed prevalence structure but destroying category
    co-occurrence) - the standard adaptation of Tibshirani's reference
    distribution for binary/basket data, any all-zero reference basket
    (no categories present, Jaccard-undefined) is resampled.
    """
    X = np.asarray(X_sample, dtype=np.float64)
    n, d = X.shape
    marginals = X.mean(axis=0)

    def generate_reference(seed):
        rng = np.random.default_rng(seed)
        ref = (rng.random((n, d)) < marginals).astype(np.float64)
        empty = ref.sum(axis=1) == 0
        while empty.any():
            ref[empty] = (rng.random((int(empty.sum()), d)) < marginals).astype(np.float64)
            empty = ref.sum(axis=1) == 0
        return ref

    dispersion_fn = partial(_kmedoids_dispersion, weights=weights)
    df = gap_statistic(X, k_range, dispersion_fn, generate_reference, B=B, random_state=random_state)

    silhouettes = []
    for k in k_range:
        D = _jaccard_distance_matrix(X, weights=weights)
        _, labels = kmedoids_fit(D, k, random_state=random_state)
        silhouettes.append(silhouette_score(D, labels, metric="precomputed") if len(set(labels)) > 1 else float("nan"))
    df["silhouette"] = silhouettes
    return df


@dataclass
class MissionClustering:
    k: int
    medoid_vectors: np.ndarray  # (k, d) binary
    sample_size: int
    labels: np.ndarray  # length = n_relatively_concrete, aligned to input order
    distances: np.ndarray  # distance to assigned medoid, same length as labels


def fit_and_assign_missions(
    matrix: csr_matrix,
    k: int = DEFAULT_K,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    weights: np.ndarray | None = None,
) -> MissionClustering:
    """Fit k-medoids on a random sample of `matrix`'s rows, then assign every
    row in `matrix` to its nearest medoid. `weights` (shape (d,), e.g. from
    `compute_idf_weights`) switches the distance metric to weighted Jaccard -
    used to downweight ubiquitous "staple" categories so they don't wash out
    the more discriminating ones (see docs/scenarios/level3_baseline_gapk_staple_downweighted.md)."""
    n = matrix.shape[0]
    rng = np.random.default_rng(random_state)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = np.asarray(matrix[sample_idx].todense())
    D = _jaccard_distance_matrix(X_sample, weights=weights)
    medoid_local_idx, _ = kmedoids_fit(D, k, random_state=random_state)
    medoid_vectors = X_sample[medoid_local_idx]

    labels, distances = _assign_to_medoids(matrix, medoid_vectors, weights=weights)

    return MissionClustering(
        k=k,
        medoid_vectors=medoid_vectors,
        sample_size=sample_size,
        labels=labels,
        distances=distances,
    )


@dataclass
class MultiInitMissionClustering(MissionClustering):
    n_init: int = 1
    best_init_seed: int = 0
    init_costs: list = None  # total within-cluster dispersion for every attempted init, in order


def fit_and_assign_missions_multi_init(
    matrix: csr_matrix,
    k: int = DEFAULT_K,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    n_init: int = 10,
    random_state: int = RANDOM_STATE,
    weights: np.ndarray | None = None,
) -> MultiInitMissionClustering:
    """Same CLARA shape as `fit_and_assign_missions`, but runs `n_init`
    independent k-medoids fits (different random seeds -> different
    k-medoids++ initializations) on the *same* CLARA sample, and keeps the
    one with the lowest total within-cluster Jaccard dispersion - the
    standard "multiple random restarts, keep the best" pattern (mirrors
    scikit-learn's `KMeans(n_init=...)`).

    `fit_and_assign_missions` only ever tries one initialization. This
    project's stability investigation (docs/stability/*.md) found that
    single-sample, single-init CLARA agrees with itself only ARI~0.2-0.3
    across random seeds - a single k-medoids++ init can converge to a
    different local optimum each time, which is a distinct problem from
    (and compounds) the sampling variance a full multi-sample CLARA would
    address. This function fixes the init-variance half of that; it does
    not draw multiple samples (still one CLARA sample per call - sample-level
    variance is a separate, unaddressed axis, see `stability.run_stability_test`
    for how much of the instability is attributable to which sample gets
    drawn vs. this initialization effect)."""
    n = matrix.shape[0]
    rng = np.random.default_rng(random_state)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = np.asarray(matrix[sample_idx].todense())
    D = _jaccard_distance_matrix(X_sample, weights=weights)

    best_medoids, best_cost, best_seed = None, np.inf, random_state
    init_costs = []
    for i in range(n_init):
        seed = random_state * 1000 + i
        medoids, labels = kmedoids_fit(D, k, random_state=seed)
        cost = 0.0
        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if len(members) > 1:
                cost += D[np.ix_(members, members)].sum() / 2
        init_costs.append(float(cost))
        if cost < best_cost:
            best_cost, best_medoids, best_seed = cost, medoids, seed

    medoid_vectors = X_sample[best_medoids]
    labels, distances = _assign_to_medoids(matrix, medoid_vectors, weights=weights)

    return MultiInitMissionClustering(
        k=k, medoid_vectors=medoid_vectors, sample_size=sample_size,
        labels=labels, distances=distances,
        n_init=n_init, best_init_seed=best_seed, init_costs=init_costs,
    )


def fit_and_assign_missions_bray_curtis(
    matrix: csr_matrix,
    k: int = DEFAULT_K,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> MissionClustering:
    """Same CLARA shape as `fit_and_assign_missions`, under Bray-Curtis on
    `matrix` (pass a value matrix, ideally row-normalized via
    `normalize_value_matrix`, not the binary presence matrix)."""
    n = matrix.shape[0]
    rng = np.random.default_rng(random_state)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = np.asarray(matrix[sample_idx].todense())
    D = _bray_curtis_distance_matrix(X_sample)
    medoid_local_idx, _ = kmedoids_fit(D, k, random_state=random_state)
    medoid_vectors = X_sample[medoid_local_idx]

    labels, distances = _assign_to_medoids_bray_curtis(matrix, medoid_vectors)

    return MissionClustering(
        k=k,
        medoid_vectors=medoid_vectors,
        sample_size=sample_size,
        labels=labels,
        distances=distances,
    )


def _assign_to_medoids(matrix: csr_matrix, medoid_vectors: np.ndarray, weights: np.ndarray | None = None):
    if weights is None:
        intersection = matrix.dot(medoid_vectors.T)  # (n, k), dense
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        medoid_sums = medoid_vectors.sum(axis=1)
    else:
        weighted_medoids = medoid_vectors * weights[None, :]
        intersection = matrix.dot(weighted_medoids.T)  # (n, k), dense
        row_sums = np.asarray(matrix.multiply(weights.reshape(1, -1)).sum(axis=1)).ravel()
        medoid_sums = weighted_medoids.sum(axis=1)
    union = row_sums[:, None] + medoid_sums[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    distance = 1.0 - similarity
    labels = np.argmin(distance, axis=1)
    min_distance = distance[np.arange(distance.shape[0]), labels]
    return labels, min_distance


@dataclass
class HDBSCANClustering:
    k_found: int  # number of non-noise clusters HDBSCAN found on the sample - NOT fixed in advance
    medoid_vectors: np.ndarray  # (k_found, d) binary - one per discovered cluster, for full-population assignment
    sample_size: int
    sample_noise_frac: float  # fraction of the sample HDBSCAN itself labeled noise (-1)
    labels: np.ndarray  # length = n basket, aligned to input order - full population, via nearest-medoid
    distances: np.ndarray


def fit_and_assign_missions_hdbscan(
    matrix: csr_matrix,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_cluster_size: int = 100,
    min_samples: int | None = None,
    random_state: int = RANDOM_STATE,
) -> HDBSCANClustering:
    """HDBSCAN in place of k-medoids for Stage 2: same CLARA-style sampling
    (draw a sample small enough for an O(n^2) distance matrix), but density-
    based clustering instead of a fixed k. Doesn't require specifying k in
    advance and can label sparse points as noise (-1) instead of forcing
    them into a cluster - the two properties motivating this as an
    alternative worth testing (see docs/stability/*.md - CLARA k-medoids was
    found to be highly unstable across random samples, mean ARI ~0.2).

    Extending HDBSCAN's sample-level assignment to the full population can't
    reuse "nearest of the sample_size sample points" directly (a dense
    n_basket x sample_size distance matrix is infeasible - e.g. 6.7M x 6000
    would be a ~320GB array). Instead, one medoid is extracted per
    HDBSCAN-discovered cluster (the sample member minimizing total in-cluster
    distance, same rule `kmedoids_fit` uses) and the full population is
    assigned via the same `_assign_to_medoids` used for k-medoids - so
    noise-vs-cluster distinctions only exist at the sample level; every
    full-population basket ends up assigned to its nearest discovered
    cluster's medoid, none labeled noise.
    """
    from sklearn.cluster import HDBSCAN

    n = matrix.shape[0]
    rng = np.random.default_rng(random_state)
    sample_size = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    X_sample = np.asarray(matrix[sample_idx].todense(), dtype=np.float64)
    D = _jaccard_distance_matrix(X_sample)

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed", copy=True)
    sample_labels = hdb.fit_predict(D)

    cluster_ids = sorted(set(sample_labels.tolist()) - {-1})
    if not cluster_ids:
        raise ValueError(
            f"HDBSCAN found no clusters at all (min_cluster_size={min_cluster_size}) - "
            f"everything was noise; try a smaller min_cluster_size"
        )

    medoid_local_idx = []
    for cid in cluster_ids:
        members = np.where(sample_labels == cid)[0]
        sub = D[np.ix_(members, members)]
        total_cost = sub.sum(axis=1)
        medoid_local_idx.append(int(members[np.argmin(total_cost)]))
    medoid_vectors = X_sample[medoid_local_idx]

    labels, distances = _assign_to_medoids(matrix, medoid_vectors)
    noise_frac = float((sample_labels == -1).sum()) / sample_size

    return HDBSCANClustering(
        k_found=len(cluster_ids),
        medoid_vectors=medoid_vectors,
        sample_size=sample_size,
        sample_noise_frac=noise_frac,
        labels=labels,
        distances=distances,
    )
