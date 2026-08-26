"""BigCLAM (Yang & Leskovec 2013): community-affiliation graph model, as a
genuinely overlapping alternative to both Louvain (`graph_missions`, hard
partition) and CPM (clique-based overlap - see project history: CPM found
real, interpretable overlap but structurally could not represent standalone
single-category missions like TOBACCO/BEER/WINE, since a node only joins a
community by sitting inside a dense k-clique).

Each category u gets a non-negative membership-strength vector F_u in R^C
(C = number of communities). Edge probability p(u,v) = 1 - exp(-F_u . F_v) -
categories with high combined community affinity are more likely to co-occur.
Fit by maximizing this log-likelihood over the fixed category graph
(`graph_missions.compute_value_pmi_graph`'s output) via block-coordinate
gradient ascent.

Unlike CPM, every node gets *some* membership vector (even a weak one), so
coverage is never partial by construction - the trade-off is two
hyperparameters to choose (C, and the relative threshold for extracting
hard membership from continuous F) versus CPM's one (clique size k)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx

from .categories import CategoryVocab
from .config import RANDOM_STATE

DEFAULT_GRAD_CLIP = 5.0


@dataclass
class BigClamFit:
    nodes: list  # category ids, aligned to F's rows
    F: np.ndarray  # (n_nodes, n_communities), non-negative membership strengths
    n_communities: int
    log_likelihood: float


def fit_bigclam(
    G: nx.Graph, n_communities: int, n_iter: int = 150, lr: float = 0.01,
    seed: int = RANDOM_STATE, n_restarts: int = 3, min_val: float = 1e-4,
    grad_clip: float = DEFAULT_GRAD_CLIP,
) -> BigClamFit:
    """Block-coordinate gradient ascent on the BigCLAM log-likelihood, using
    the standard O(degree) gradient trick (precompute the global sum of F
    once per sweep, subtract neighbor/self terms per node) rather than the
    naive O(n) per-node sum over non-neighbors - cheap at this project's
    graph sizes (~70-100 category nodes) regardless of how many baskets the
    underlying store has, since the graph is over categories, not baskets.

    `grad_clip` is not optional: the neighbor-term coefficient
    exp(-dot)/(1-exp(-dot)) diverges as dot->0, which happens easily early
    in optimization from a small random init. An unclipped fixed-lr step
    blew F values up to ~900 (from a healthy ~0-5 range) on a real test
    graph during development, silently producing a garbage fit with no
    error raised - just an absurd, easily-missed log-likelihood. Verified
    correct (post-clipping) against a hand-checked toy graph: two triangles
    sharing one bridge node recovers exactly that node as the sole
    dual-community member, everything else single-community."""
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    adj = [np.array(sorted(idx[v] for v in G.neighbors(u)), dtype=int) for u in nodes]

    def log_likelihood(F):
        sumF = F.sum(axis=0)
        ll = 0.0
        for i in range(n):
            if len(adj[i]):
                dots = F[adj[i]] @ F[i]
                ll += np.log(1 - np.exp(-dots) + 1e-12).sum() / 2  # each edge counted from both ends
            non_nbr_sum = sumF - F[i] - (F[adj[i]].sum(axis=0) if len(adj[i]) else 0)
            ll -= F[i] @ non_nbr_sum / 2
        return ll

    best_F, best_ll = None, -np.inf
    rng = np.random.default_rng(seed)
    for restart in range(n_restarts):
        F = rng.uniform(0.01, 0.5, size=(n, n_communities))
        for _ in range(n_iter):
            sumF = F.sum(axis=0)
            for i in range(n):
                nbrs = adj[i]
                if len(nbrs):
                    dots = np.clip(F[nbrs] @ F[i], 1e-6, None)
                    coef = np.exp(-dots) / (1 - np.exp(-dots) + 1e-12)
                    grad_nbr = (coef[:, None] * F[nbrs]).sum(axis=0)
                    non_nbr_sum = sumF - F[i] - F[nbrs].sum(axis=0)
                else:
                    grad_nbr = np.zeros(n_communities)
                    non_nbr_sum = sumF - F[i]
                grad = grad_nbr - non_nbr_sum
                grad_norm = np.linalg.norm(grad)
                if grad_norm > grad_clip:
                    grad = grad * (grad_clip / grad_norm)
                F[i] = np.maximum(F[i] + lr * grad, min_val)
        ll = log_likelihood(F)
        if np.isfinite(ll) and ll > best_ll:
            best_ll, best_F = ll, F.copy()

    return BigClamFit(nodes=nodes, F=best_F, n_communities=n_communities, log_likelihood=best_ll)


def select_n_communities(
    G: nx.Graph, c_range=range(4, 9), n_iter: int = 150, seed: int = RANDOM_STATE, n_restarts: int = 2,
) -> pd.DataFrame:
    """Log-likelihood scree curve across candidate community counts - like
    NMF's reconstruction-error curve, BigCLAM has no gap-statistic
    equivalent. Log-likelihood should improve (become less negative) as C
    grows, similar to how reconstruction error falls as k grows in a
    factorization model - use `cluster_selection.find_elbow_k` on
    `-log_likelihood` (so the curve is "decreasing-ish" as the helper
    expects) to pick an elbow rather than trusting the highest C blindly."""
    rows = []
    for c in c_range:
        fit = fit_bigclam(G, c, n_iter=n_iter, seed=seed, n_restarts=n_restarts)
        rows.append({"n_communities": c, "log_likelihood": fit.log_likelihood})
    return pd.DataFrame(rows)


def extract_communities(fit: BigClamFit, rel_threshold: float = 0.3, min_absolute_strength: float = 0.02) -> dict[int, list[int]]:
    """Node u belongs to community c if F[u,c] >= rel_threshold * max(F[u,:])
    AND max(F[u,:]) >= min_absolute_strength - a relative-strength
    threshold alone isn't safe: a degree-0 node (no edges survived the
    graph's min_lift filter - a real case, e.g. TOBACCO had zero edges at
    one store, all filtered out for lift<=1) never receives gradient signal
    and sits exactly at the optimizer's floor (`min_val`) in every
    community. Every entry is then trivially "within `rel_threshold` of the
    row max" because they're all identical, so the node would spuriously
    appear to belong to *every* community - the opposite of correct (an
    isolated node has no real community signal and should belong to none,
    matching how `graph_missions` already drops near-zero-prevalence nodes
    entirely rather than pretending they cluster). `min_absolute_strength`
    catches this: found via a real failure case during development, not
    picked blindly - healthy, edge-supported nodes reach F values an order
    of magnitude or more above this floor (e.g. 0.67, 0.086), so it
    separates "weak but real signal" from "no signal, stuck at the
    optimizer's minimum" cleanly. Returns {community_id: [category_ids]}."""
    communities = {c: [] for c in range(fit.n_communities)}
    for i, u in enumerate(fit.nodes):
        row_max = fit.F[i].max()
        if row_max < min_absolute_strength:
            continue
        for c in range(fit.n_communities):
            if fit.F[i, c] >= rel_threshold * row_max:
                communities[c].append(u)
    return communities


@dataclass
class BigClamMissionAssignment:
    n_communities: int
    hard_labels: np.ndarray  # (n_baskets,) argmax of continuous community affinity
    soft_scores: np.ndarray  # (n_baskets, n_communities) - normalized continuous affinity


def assign_baskets_to_missions(matrix: csr_matrix, vocab: CategoryVocab, fit: BigClamFit) -> BigClamMissionAssignment:
    """Generalizes `graph_missions.assign_baskets_to_missions` from a hard
    0/1 community_map to BigCLAM's continuous F matrix - a basket's
    affinity to community c is the sum of F[category, c] over categories it
    contains, so categories with strong-but-not-exclusive membership (e.g.
    WATER, which bridges two communities with real strength in both - see
    project history) contribute proportionally to every community they
    belong to, not just a single winner-take-all one."""
    d = vocab.size
    F_full = np.zeros((d, fit.n_communities))
    node_to_row = {u: i for i, u in enumerate(fit.nodes)}
    for i, cid in enumerate(vocab.ids):
        row = node_to_row.get(cid)
        if row is not None:
            F_full[i] = fit.F[row]

    community_affinity = matrix.dot(F_full)  # (n_baskets, n_communities), dense
    row_sums = community_affinity.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        soft_scores = np.where(row_sums > 0, community_affinity / row_sums, 0.0)
    hard_labels = np.argmax(community_affinity, axis=1)

    return BigClamMissionAssignment(n_communities=fit.n_communities, hard_labels=hard_labels, soft_scores=soft_scores)
