"""Step E alternative: NMF (non-negative matrix factorization) as a topic
model for missions, giving soft/probabilistic basket-mission membership
instead of the hard single-label assignment every other method in this
project produces (Jaccard k-medoids, HDBSCAN, value-weighted graph).

Motivation: a real shopping trip plausibly serves more than one need at
once (a "dinner + snacks" trip isn't purely one or the other), but every
clustering method tried so far forces exactly one label per basket. NMF
factorizes the basket x category matrix X (n_baskets x n_categories) into
W (n_baskets x n_topics, basket-topic loadings) and H (n_topics x
n_categories, topic-category loadings) such that X ~= W @ H - a basket's
mission membership is then the full row of W, not one winning label.

Fit on the row-normalized VALUE matrix (`mission_stage.normalize_value_matrix`),
not binary presence - matches this project's own finding that value-weighted
signal produces sharper, more interpretable structure (see
docs/graph_missions/level3_value_distribution_graph.md) than presence/absence
alone, and both W and H come out non-negative by construction, which reads
naturally as "share of this basket explained by this mission" /
"how much this category defines this mission".

NMF's objective is non-convex, so (unlike Louvain on a fixed graph) the
solution genuinely depends on initialization - `fit_nmf_missions` defaults
to `init="nndsvda"` (a deterministic SVD-based warm start, generally the
best-converging choice for a single reference fit); the stability test in
`stability.run_stability_test_nmf` instead uses `init="random"` across many
seeds specifically to measure how much that non-convexity moves the answer,
mirroring how this project already measures k-medoids' CLARA-sample
variance and gap-statistic's KMeans n_init variance elsewhere.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import NMF

from . import mission_stage
from .categories import CategoryVocab
from .config import RANDOM_STATE

DEGENERATE_LOADING = 0.90  # a topic whose single top category holds >=90% of its H-row weight isn't a real "mission"


def build_idf_weighted_matrix(quantity_matrix: csr_matrix, presence_matrix: csr_matrix, min_support: float = 0.01) -> csr_matrix:
    """x_bc = log(1 + quantity_bc) x idf_c - a TF-IDF-style feature, not the
    value-share weighting `normalize_value_matrix` uses elsewhere in this
    project. The distinction matters: raw/normalized *spend* can make
    inflation, premium products, and promotions look like distinct missions
    purely because they move money differently, not because they reflect a
    different shopping goal. Item *count* (log-dampened so a 20-unit bulk
    buy doesn't swamp everything else in the basket) times inverse document
    frequency (downweighting ubiquitous staples the way it already does for
    Jaccard elsewhere - reuses `mission_stage.compute_idf_weights` rather
    than reimplementing it) targets "how distinctively this basket used
    this category" without spend magnitude doing the talking.

    `presence_matrix` (binary) is used only to compute idf (document
    frequency needs presence/absence, not counts); `quantity_matrix` (raw
    item counts) supplies the log(1+count) term."""
    idf = mission_stage.compute_idf_weights(presence_matrix, min_support=min_support)
    log_qty = quantity_matrix.tocsr(copy=True)
    log_qty.data = np.log1p(log_qty.data)
    return log_qty @ diags(idf)


@dataclass
class NMFMissionFit:
    n_topics: int
    W: np.ndarray  # (n_baskets, n_topics), non-negative, NOT row-normalized
    H: np.ndarray  # (n_topics, n_categories), non-negative topic-category loadings
    hard_labels: np.ndarray  # argmax(W, axis=1) - for stability testing and comparison to other (hard-label) methods
    soft_membership: np.ndarray  # W row-normalized to sum 1 - the actual soft mission-membership distribution
    reconstruction_err: float  # ||X - W @ H||_F, as returned by sklearn


def fit_nmf_missions(
    value_matrix: csr_matrix, n_topics: int, random_state: int = RANDOM_STATE,
    init: str = "nndsvda", max_iter: int = 400,
) -> NMFMissionFit:
    model = NMF(
        n_components=n_topics, init=init, solver="cd", beta_loss="frobenius",
        max_iter=max_iter, random_state=random_state,
    )
    W = model.fit_transform(value_matrix)
    H = model.components_

    row_sums = W.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        soft = np.where(row_sums > 0, W / row_sums, 0.0)
    hard_labels = np.argmax(W, axis=1)

    return NMFMissionFit(
        n_topics=n_topics, W=W, H=H, hard_labels=hard_labels, soft_membership=soft,
        reconstruction_err=float(model.reconstruction_err_),
    )


def select_n_topics(
    value_matrix: csr_matrix, k_range=range(3, 10), random_state: int = RANDOM_STATE, max_iter: int = 300,
) -> pd.DataFrame:
    """Reconstruction-error scree curve across candidate topic counts. NMF
    has no equivalent of the gap statistic - there's no natural "null
    reference" distribution for a factorization rank the way there is for
    cluster dispersion - so topic count is instead chosen by the same elbow
    heuristic already used elsewhere in this project
    (`cluster_selection.find_elbow_k`), applied to reconstruction error, and
    the caller should cross-check the chosen k against per-topic degeneracy
    (see `DEGENERATE_LOADING`) rather than trusting the elbow blindly."""
    rows = []
    for k in k_range:
        fit = fit_nmf_missions(value_matrix, k, random_state=random_state, max_iter=max_iter)
        rows.append({"k": k, "reconstruction_err": fit.reconstruction_err})
    return pd.DataFrame(rows)


def bootstrap_stability(
    matrix: csr_matrix, n_topics: int, n_bootstrap: int = 10, random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Genuine bootstrap stability, distinct from `stability.run_stability_test_nmf`'s
    seed-variance test: each round resamples baskets *with replacement*
    (the actual bootstrap procedure) rather than refitting on the same
    fixed matrix with a different random init. Bootstrap resampling means
    each round's basket set differs, so basket-level ARI (which needs the
    same rows in the same order across runs) isn't available - instead,
    each bootstrap fit's topics are matched to the reference fit's topics
    by greedy cosine similarity between H-rows (topic-category loading
    vectors), and the best-match cosine similarity per reference topic is
    reported. High, consistently-matched similarity across bootstrap
    rounds means a topic's category composition doesn't depend on which
    particular baskets happened to be sampled; a topic that matches poorly
    or inconsistently is likely an artifact of specific baskets rather than
    a real recurring pattern."""
    n = matrix.shape[0]
    reference = fit_nmf_missions(matrix, n_topics, random_state=random_state)
    ref_H = reference.H / np.linalg.norm(reference.H, axis=1, keepdims=True).clip(min=1e-12)

    rows = []
    rng = np.random.default_rng(random_state)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = matrix[idx]
        boot = fit_nmf_missions(sample, n_topics, random_state=random_state + b + 1)
        boot_H = boot.H / np.linalg.norm(boot.H, axis=1, keepdims=True).clip(min=1e-12)
        cos_sim = ref_H @ boot_H.T  # (n_topics, n_topics)
        best_match = cos_sim.max(axis=1)
        for t in range(n_topics):
            rows.append({"bootstrap_round": b, "reference_topic": t, "best_match_cosine": float(best_match[t])})
    df = pd.DataFrame(rows)
    return df.groupby("reference_topic")["best_match_cosine"].agg(["mean", "std", "min", "max"]).reset_index()


def profile_topics(
    value_matrix: csr_matrix, fit: NMFMissionFit, vocab: CategoryVocab, baskets: pd.DataFrame,
    min_loading: float = 0.05, top_n: int = 8,
) -> list[dict]:
    """Two complementary views per topic, so it can be read either as a
    topic model or compared apples-to-apples against this project's other
    (hard-label) methods:

    - `native_top_categories`: NMF's own H-row loadings, normalized to sum
      1 per topic - directly what the model learned, with no assumption of
      a hard basket assignment at all.
    - `hard_top_categories`: value-share/lift statistic (same formula
      `run_value_graph_missions_2025.py`'s `_profile_communities` already
      uses) computed over the baskets hard-assigned (argmax of W) to this
      topic - lets an NMF topic's headline categories be compared directly
      against a value-graph community's.
    """
    n = value_matrix.shape[0]
    overall_value = np.asarray(value_matrix.sum(axis=0)).ravel()
    overall_share = overall_value / overall_value.sum() if overall_value.sum() > 0 else overall_value

    profiles = []
    for t in range(fit.n_topics):
        h = fit.H[t]
        h_norm = h / h.sum() if h.sum() > 0 else h
        loading_order = np.argsort(-h_norm)
        native_top = [
            {"category": vocab.names[vocab.ids[i]], "loading": round(float(h_norm[i]), 4)}
            for i in loading_order if h_norm[i] >= min_loading
        ][:top_n]
        is_degenerate = bool(len(native_top) and native_top[0]["loading"] >= DEGENERATE_LOADING)

        mask = fit.hard_labels == t
        nc = int(mask.sum())
        sub = value_matrix[mask]
        cshare = np.asarray(sub.sum(axis=0)).ravel()
        cshare = cshare / cshare.sum() if cshare.sum() > 0 else cshare
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_share > 0, cshare / overall_share, 0.0)
        lift_order = np.argsort(-cshare)
        hard_top = [
            {"category": vocab.names[vocab.ids[i]], "value_share": round(float(cshare[i]), 4), "lift": round(float(lift[i]), 3)}
            for i in lift_order if cshare[i] > 0
        ][:top_n]

        profiles.append({
            "topic_id": t,
            "is_degenerate": is_degenerate,
            "n_hard_baskets": nc,
            "hard_share_pct": round(nc / n * 100, 2) if n else 0,
            "avg_soft_weight_pct": round(float(fit.soft_membership[:, t].mean()) * 100, 2),
            "avg_value": round(float(baskets.loc[mask, "basket_value"].mean()), 2) if nc else None,
            "native_top_categories": native_top,
            "hard_top_categories": hard_top,
        })
    return profiles
