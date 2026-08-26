"""Cluster stability testing: repeat Stage 2 clustering across multiple
random seeds and measure pairwise agreement (Adjusted Rand Index, Normalized
Mutual Information) between the resulting basket-level labelings.

Motivated by a real gap flagged during a 2026-07-21 review of possible
methodology improvements: every scenario run in this project so far used a
single fixed random_state=0. We've never actually checked whether cluster
assignments are stable across different CLARA samples/k-medoids++
initializations, as opposed to just reflecting what that one particular
sample happened to produce - which matters a lot given how much reported
results have swung between otherwise-similar configs (e.g. the HOUSEHOLD
PRODUCTS exclusion test).

ARI and NMI are both permutation-invariant (cluster "0" in run A doesn't
need to correspond to cluster "0" in run B for these to work correctly) -
exactly what's needed since k-medoids cluster IDs are arbitrary per fit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import bigclam, dmr_missions, embeddings, graph_missions, mission_stage, topic_missions
from .categories import CategoryVocab
from .config import DEFAULT_K, DEFAULT_SAMPLE_SIZE


@dataclass
class StabilityResult:
    n_runs: int
    k: int
    sample_size: int
    pairwise_ari: np.ndarray  # (n_runs, n_runs), symmetric, diagonal = 1.0
    pairwise_nmi: np.ndarray  # (n_runs, n_runs), symmetric, diagonal = 1.0
    cluster_size_shares: pd.DataFrame  # n_runs rows x k columns, % share per run

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        n = self.n_runs
        mask = ~np.eye(n, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        n = self.n_runs
        mask = ~np.eye(n, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test(
    matrix: csr_matrix,
    k: int = DEFAULT_K,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    n_runs: int = 20,
    base_seed: int = 0,
) -> StabilityResult:
    """Fits Stage 2 clustering `n_runs` times on the same `matrix`, each with
    a distinct random seed (different CLARA sample AND different k-medoids++
    initialization), then computes every pairwise ARI/NMI between the
    resulting full-population labelings."""
    labelings = []
    size_shares = []
    n_baskets = matrix.shape[0]

    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        clustering = mission_stage.fit_and_assign_missions(
            matrix, k=k, sample_size=sample_size, random_state=seed
        )
        labelings.append(clustering.labels)
        shares = np.bincount(clustering.labels, minlength=k) / n_baskets * 100
        size_shares.append(shares)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    size_df = pd.DataFrame(size_shares, columns=[f"cluster_{c}" for c in range(k)])
    return StabilityResult(
        n_runs=n_runs, k=k, sample_size=sample_size,
        pairwise_ari=ari, pairwise_nmi=nmi, cluster_size_shares=size_df,
    )


@dataclass
class HDBSCANStabilityResult:
    n_runs: int
    sample_size: int
    min_cluster_size: int
    k_found_per_run: np.ndarray  # length n_runs - HDBSCAN doesn't fix k, so this varies run to run
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray
    noise_frac_per_run: np.ndarray  # fraction of the *sample* HDBSCAN itself called noise, per run

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_hdbscan(
    matrix: csr_matrix,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_cluster_size: int = 100,
    n_runs: int = 20,
    base_seed: int = 0,
) -> HDBSCANStabilityResult:
    """Same idea as `run_stability_test`, for `mission_stage.fit_and_assign_missions_hdbscan`.
    Unlike k-medoids, HDBSCAN doesn't fix the number of clusters in advance,
    so how much `k_found` itself varies across runs is a stability signal in
    its own right, tracked alongside ARI/NMI."""
    labelings = []
    k_found = []
    noise_fracs = []

    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        clustering = mission_stage.fit_and_assign_missions_hdbscan(
            matrix, sample_size=sample_size, min_cluster_size=min_cluster_size, random_state=seed
        )
        labelings.append(clustering.labels)
        k_found.append(clustering.k_found)
        noise_fracs.append(clustering.sample_noise_frac)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return HDBSCANStabilityResult(
        n_runs=n_runs, sample_size=sample_size, min_cluster_size=min_cluster_size,
        k_found_per_run=np.array(k_found), pairwise_ari=ari, pairwise_nmi=nmi,
        noise_frac_per_run=np.array(noise_fracs),
    )


@dataclass
class GraphStabilityResult:
    n_runs: int
    min_lift: float
    resolution: float
    n_communities_per_run: np.ndarray  # length n_runs - Louvain doesn't fix community count either
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_graph(
    matrix: csr_matrix,
    vocab: CategoryVocab,
    min_lift: float = 1.0,
    resolution: float = 1.0,
    n_runs: int = 20,
    base_seed: int = 0,
) -> GraphStabilityResult:
    """Same idea as the other two stability tests, but for graph community
    detection. Critically, the category co-occurrence graph itself is built
    ONCE, from the exact full population - it does not depend on any random
    seed at all. Only Louvain's own internal randomness (traversal order,
    tie-breaking) varies across runs, which is a much narrower source of
    variation than re-sampling 6,000 baskets out of millions each time (what
    `run_stability_test`/`run_stability_test_hdbscan` do). If this approach
    is meaningfully more stable, it should show up here as much higher
    ARI/NMI than the other two stability reports."""
    G = graph_missions.compute_pmi_graph(matrix, vocab, min_lift=min_lift)

    labelings = []
    n_communities = []
    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        community_map = graph_missions.detect_communities(G, resolution=resolution, seed=seed)
        assignment = graph_missions.assign_baskets_to_missions(matrix, vocab, community_map)
        labelings.append(assignment.hard_labels)
        n_communities.append(assignment.n_communities)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return GraphStabilityResult(
        n_runs=n_runs, min_lift=min_lift, resolution=resolution,
        n_communities_per_run=np.array(n_communities), pairwise_ari=ari, pairwise_nmi=nmi,
    )


@dataclass
class BrayCurtisStabilityResult:
    n_runs: int
    k: int
    sample_size: int
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray
    cluster_size_shares: pd.DataFrame

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_bray_curtis(
    value_matrix: csr_matrix,
    k: int = DEFAULT_K,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    n_runs: int = 20,
    base_seed: int = 0,
) -> BrayCurtisStabilityResult:
    """Same idea as `run_stability_test`, for
    `mission_stage.fit_and_assign_missions_bray_curtis` - clustering on each
    basket's spend *distribution* across categories (Bray-Curtis) rather
    than category presence/absence (Jaccard). `value_matrix` should already
    be row-normalized via `mission_stage.normalize_value_matrix`. Still
    CLARA (basket sampling) under the hood, so still subject to the same
    sampling-instability source as `run_stability_test` - this measures
    whether the sharper category lifts Bray-Curtis finds per-run are also
    more reproducible run-to-run, or just a better fit to whichever sample
    happened to get drawn."""
    labelings = []
    size_shares = []
    n_baskets = value_matrix.shape[0]

    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        clustering = mission_stage.fit_and_assign_missions_bray_curtis(
            value_matrix, k=k, sample_size=sample_size, random_state=seed
        )
        labelings.append(clustering.labels)
        shares = np.bincount(clustering.labels, minlength=k) / n_baskets * 100
        size_shares.append(shares)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    size_df = pd.DataFrame(size_shares, columns=[f"cluster_{c}" for c in range(k)])
    return BrayCurtisStabilityResult(
        n_runs=n_runs, k=k, sample_size=sample_size,
        pairwise_ari=ari, pairwise_nmi=nmi, cluster_size_shares=size_df,
    )


@dataclass
class ValueGraphStabilityResult:
    n_runs: int
    min_lift: float
    min_prevalence: float
    resolution: float
    n_communities_per_run: np.ndarray
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_value_graph(
    value_matrix: csr_matrix,
    vocab: CategoryVocab,
    min_lift: float = 1.0,
    min_prevalence: float = 1e-4,
    resolution: float = 2.0,
    n_runs: int = 20,
    base_seed: int = 0,
) -> ValueGraphStabilityResult:
    """Same idea as `run_stability_test_graph`, for
    `graph_missions.compute_value_pmi_graph` - the value-weighted
    generalization of graph community detection, clustering on spend
    distribution instead of category presence/absence. `value_matrix` should
    be row-normalized (`mission_stage.normalize_value_matrix`). Like the
    binary version, the graph itself is built once from the exact full
    population and does not depend on any seed - only Louvain's own
    randomness varies across runs, which is why this and
    `run_stability_test_graph` are both far more stable than anything CLARA-
    sampling-based (`run_stability_test`, `run_stability_test_bray_curtis`,
    `run_stability_test_hdbscan`)."""
    G = graph_missions.compute_value_pmi_graph(value_matrix, vocab, min_lift=min_lift, min_prevalence=min_prevalence)

    labelings = []
    n_communities = []
    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        community_map = graph_missions.detect_communities(G, resolution=resolution, seed=seed)
        assignment = graph_missions.assign_baskets_to_missions(value_matrix, vocab, community_map)
        labelings.append(assignment.hard_labels)
        n_communities.append(assignment.n_communities)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return ValueGraphStabilityResult(
        n_runs=n_runs, min_lift=min_lift, min_prevalence=min_prevalence, resolution=resolution,
        n_communities_per_run=np.array(n_communities), pairwise_ari=ari, pairwise_nmi=nmi,
    )


@dataclass
class EmbeddingGraphStabilityResult:
    n_runs: int
    dim: int
    min_cosine: float
    min_prevalence: float
    resolution: float
    n_communities_per_run: np.ndarray
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_embedding_graph(
    value_matrix: csr_matrix,
    vocab: CategoryVocab,
    dim: int = 32,
    min_cosine: float = 0.3,
    min_prevalence: float = 1e-4,
    resolution: float = 2.0,
    n_runs: int = 20,
    base_seed: int = 0,
) -> EmbeddingGraphStabilityResult:
    """Same idea as `run_stability_test_value_graph`, for the Item2Vec-style
    embedding similarity graph (`embeddings.compute_embedding_similarity_graph`)
    instead of direct pairwise lift. The PPMI matrix and its SVD are both
    fully deterministic given the population (no seed dependence at all -
    `np.linalg.svd` has no randomness), so like the other two graph methods,
    only Louvain's own randomness varies across runs here."""
    ppmi, category_ids = embeddings.compute_ppmi_matrix(value_matrix, vocab, min_prevalence=min_prevalence)
    emb = embeddings.train_item2vec(ppmi, category_ids, dim=dim)
    G = embeddings.compute_embedding_similarity_graph(emb, vocab, min_cosine=min_cosine)

    labelings = []
    n_communities = []
    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        community_map = graph_missions.detect_communities(G, resolution=resolution, seed=seed)
        assignment = graph_missions.assign_baskets_to_missions(value_matrix, vocab, community_map)
        labelings.append(assignment.hard_labels)
        n_communities.append(assignment.n_communities)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return EmbeddingGraphStabilityResult(
        n_runs=n_runs, dim=dim, min_cosine=min_cosine, min_prevalence=min_prevalence, resolution=resolution,
        n_communities_per_run=np.array(n_communities), pairwise_ari=ari, pairwise_nmi=nmi,
    )


@dataclass
class NMFStabilityResult:
    n_runs: int
    n_topics: int
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_nmf(
    value_matrix: csr_matrix,
    n_topics: int,
    n_runs: int = 20,
    base_seed: int = 0,
    max_iter: int = 300,
) -> NMFStabilityResult:
    """Unlike Louvain on a fixed graph (only the community-detection step is
    randomized) or CLARA-sampling methods (randomness is which baskets get
    drawn), NMF's own optimization is non-convex - different random
    initializations can converge to genuinely different local minima on the
    *same* full data, with no sampling involved at all. This test isolates
    exactly that: every run refits NMF on the identical `value_matrix` with
    `init="random"` (deliberately NOT the deterministic `nndsvda` default
    `topic_missions.fit_nmf_missions` otherwise uses) and a different seed,
    then compares the resulting hard (argmax) basket labels pairwise -
    genuinely testing solution stability under non-convexity, the NMF
    analogue of this project's k-medoids CLARA-init stability tests."""
    labelings = []
    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        fit = topic_missions.fit_nmf_missions(value_matrix, n_topics, random_state=seed, init="random", max_iter=max_iter)
        labelings.append(fit.hard_labels)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return NMFStabilityResult(n_runs=n_runs, n_topics=n_topics, pairwise_ari=ari, pairwise_nmi=nmi)


@dataclass
class DMRStabilityResult:
    n_runs: int
    n_topics: int
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_dmr(
    build_docs_fn, n_topics: int, n_runs: int = 5, iterations: int = 150, base_seed: int = 0,
) -> DMRStabilityResult:
    """Same idea as `run_stability_test_nmf` - refit from scratch with a
    different seed each run and compare hard (argmax) labels pairwise -
    but at a much-reduced `n_runs` (5, not this project's usual 20) and
    `iterations` (150, not `fit_dmr_missions`'s 1000 default). Both are a
    real, disclosed trade-off: at ~6s/iteration on the ~1.26M-document
    pooled corpus, 20 runs x 1000 iterations would take multiple hours on
    this machine. `build_docs_fn` must return a fresh iterable each call
    (see `dmr_missions.select_n_topics`) and must yield documents in the
    exact same order every time - the pairwise ARI/NMI comparison below is
    positional (`model.docs[i]` from run A vs run B), so any reordering
    between calls would silently corrupt the comparison rather than error."""
    labelings = []
    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        fit = dmr_missions.fit_dmr_missions(build_docs_fn(), n_topics=n_topics, iterations=iterations, step=iterations, seed=seed)
        labelings.append(fit.hard_labels)
        del fit

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return DMRStabilityResult(n_runs=n_runs, n_topics=n_topics, pairwise_ari=ari, pairwise_nmi=nmi)


@dataclass
class BigClamStabilityResult:
    n_runs: int
    n_communities: int
    pairwise_ari: np.ndarray
    pairwise_nmi: np.ndarray

    @property
    def off_diagonal_ari(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_ari[mask]

    @property
    def off_diagonal_nmi(self) -> np.ndarray:
        mask = ~np.eye(self.n_runs, dtype=bool)
        return self.pairwise_nmi[mask]


def run_stability_test_bigclam(
    G, matrix: csr_matrix, vocab: CategoryVocab, n_communities: int,
    n_runs: int = 10, base_seed: int = 0, n_iter: int = 150,
) -> BigClamStabilityResult:
    """Independent BigCLAM fits (fresh random init each run, on the
    identical fixed category graph - the graph itself never changes, only
    the gradient-ascent optimization's starting point and internal restart
    seeds), compared via basket-level hard-label ARI/NMI - the same
    pattern as `run_stability_test_graph`'s Louvain seed test, generalized
    to BigCLAM's continuous-then-argmax assignment."""
    labelings = []
    for i in range(n_runs):
        seed = base_seed * 10_000 + i
        fit = bigclam.fit_bigclam(G, n_communities, n_iter=n_iter, seed=seed, n_restarts=1)
        assignment = bigclam.assign_baskets_to_missions(matrix, vocab, fit)
        labelings.append(assignment.hard_labels)

    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m

    return BigClamStabilityResult(n_runs=n_runs, n_communities=n_communities, pairwise_ari=ari, pairwise_nmi=nmi)
