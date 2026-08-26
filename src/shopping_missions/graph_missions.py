"""Graph community detection for Stage 2 (missions), as an alternative to
CLARA-sampled k-medoids/HDBSCAN.

Motivation (see docs/stability/*.md): three independent attempts to fix
Stage 2's clustering instability - larger CLARA sample, HDBSCAN instead of
k-medoids, excluding TOBACCO/HOUSEHOLD PRODUCTS - all failed to meaningfully
improve cross-seed ARI/NMI (all stuck around 0.19-0.23, essentially
just-above-chance agreement between random samples). All three share a root
cause: they still cluster a small *sample* of baskets. This module sidesteps
that entirely.

Instead of clustering baskets, it builds a weighted graph over the category
vocabulary (~97-99 nodes for level 3) - edges are category pairs weighted by
pointwise mutual information (PMI = log(lift)), computed from the *exact*
full-population co-occurrence matrix. That computation is cheap regardless
of basket count: the output is only as big as the category vocabulary
(d x d), whether d comes from 2M or 200M baskets. No basket-level sampling
anywhere. Community detection (Louvain) then finds category groups that
co-occur more than chance would predict - candidate missions - and every
basket gets assigned by which community's categories it contains most of.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from .categories import CategoryVocab


def compute_cooccurrence(matrix: csr_matrix) -> np.ndarray:
    """(d x d) dense co-occurrence count matrix - co[i,i] is category i's
    total support (basket count), co[i,j] (i!=j) is the number of baskets
    containing both. Computed exactly over the full population: the
    sparse-sparse matmul cost scales with total nonzeros (~4 categories/
    basket x n_baskets), but the *output* is only d x d regardless of n -
    this is the property that avoids sampling entirely.

    Cast to float64 before the matmul, not after: `matrix`'s native dtype is
    uint8 (see baskets.py), so a same-dtype matmul silently overflows
    mod 256 the moment any category's basket count exceeds 255 - every
    multi-million-basket population hits this."""
    M = matrix.astype(np.float64)
    co = (M.T @ M).toarray()
    return co


def compute_pmi_graph(matrix: csr_matrix, vocab: CategoryVocab, min_lift: float = 1.0) -> nx.Graph:
    """Builds the category co-occurrence graph. Nodes = categories (with
    support/prevalence as attributes). Edges = category pairs with
    lift > min_lift (i.e. co-occurring more than chance predicts - lift<=1
    pairs are dropped rather than given a negative weight, since community
    detection expects non-negative edge weights and a "co-occurs less than
    chance" relationship isn't a meaningful attraction signal for a mission
    anyway). Edge weight = PMI = log(lift)."""
    n = matrix.shape[0]
    co = compute_cooccurrence(matrix)
    support = np.diag(co).copy()
    p_i = support / n
    d = vocab.size

    G = nx.Graph()
    for i, cid in enumerate(vocab.ids):
        G.add_node(cid, name=vocab.names[cid], support=int(support[i]), prevalence=float(p_i[i]))

    for i in range(d):
        for j in range(i + 1, d):
            pair_count = co[i, j]
            if pair_count == 0 or p_i[i] <= 0 or p_i[j] <= 0:
                continue
            p_ij = pair_count / n
            lift = p_ij / (p_i[i] * p_i[j])
            if lift <= min_lift:
                continue
            G.add_edge(
                vocab.ids[i], vocab.ids[j],
                weight=float(np.log(lift)), lift=float(lift), pair_count=int(pair_count),
            )
    return G


def compute_value_pmi_graph(
    value_matrix: csr_matrix, vocab: CategoryVocab, min_lift: float = 1.0, min_prevalence: float = 1e-4,
) -> nx.Graph:
    """Same idea as `compute_pmi_graph`, generalized from category
    presence/absence to a basket's *spend distribution* across categories
    (pass a row-normalized value matrix, e.g. `mission_stage.normalize_value_matrix`'s
    output, so every basket contributes equal total mass 1 regardless of its
    size - otherwise a handful of big-spend baskets would dominate the graph).

    The lift formula is the same ratio (observed joint mass over the mass
    expected under independence) and reduces to exactly `compute_pmi_graph`'s
    binary lift when every entry is 0/1 - but it is NOT implemented by
    reusing `compute_pmi_graph` directly, because that function reads each
    category's marginal off `compute_cooccurrence`'s *diagonal*
    (`co[i,i] = sum_b x_i(b)^2`), which only equals the true marginal
    `sum_b x_i(b)` when x is binary (0^2=0, 1^2=1). For continuous shares in
    [0,1], squaring shrinks them, silently corrupting every lift computed
    from it - this function reads the marginal from the matrix's actual
    column sums instead.

    `min_prevalence` excludes a near-untraded category (e.g. FOOTWEAR at
    ~6e-7 of total spend mass) from the graph *entirely* - not just its
    edges. With so little mass, a lift value is dominated by a handful of
    baskets and can come out enormous by chance (FOOTWEAR's observed lift
    hit x1.4M on 2 baskets in one run) - the continuous analog of the noise
    `compute_idf_weights` guards against for the long tail of rare
    categories. Unlike `compute_pmi_graph`'s edge-only filtering (where a
    filtered category still gets a singleton-community node), these
    categories get no node at all: at level-3 granularity there were ~20
    of them (mostly genuinely marginal merchandise - footwear, gifts,
    sunscreen - plus a few very-low-volume Romanian preserve categories),
    each forming its own near-empty "community" of a handful of baskets
    with nonsensical lift, cluttering what should be a small set of
    substantial missions. Dropping the node means `assign_baskets_to_missions`
    simply ignores that category when voting - a basket's other, more
    common categories decide its community instead."""
    n = value_matrix.shape[0]
    co = compute_cooccurrence(value_matrix)
    support = np.asarray(value_matrix.sum(axis=0)).ravel()
    p_i = support / n
    d = vocab.size

    G = nx.Graph()
    for i, cid in enumerate(vocab.ids):
        if p_i[i] < min_prevalence:
            continue
        G.add_node(cid, name=vocab.names[cid], support=float(support[i]), prevalence=float(p_i[i]))

    for i in range(d):
        if p_i[i] < min_prevalence:
            continue
        for j in range(i + 1, d):
            if p_i[j] < min_prevalence:
                continue
            joint_mass = co[i, j]
            if joint_mass <= 0:
                continue
            p_ij = joint_mass / n
            lift = p_ij / (p_i[i] * p_i[j])
            if lift <= min_lift:
                continue
            G.add_edge(
                vocab.ids[i], vocab.ids[j],
                weight=float(np.log(lift)), lift=float(lift), joint_mass=float(joint_mass),
            )
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 0) -> dict[int, int]:
    """Louvain community detection on the PMI graph. Returns {category_id:
    community_id}. Any node with zero edges surviving the min_lift filter
    ends up in its own singleton community - reported as-is rather than
    merged, since a category with no positive-lift partners genuinely
    doesn't co-occur with anything more than chance."""
    communities = nx.algorithms.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    community_map = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            community_map[node] = comm_id
    return community_map


@dataclass
class GraphMissionAssignment:
    n_communities: int
    community_map: dict[int, int]  # category_id -> community_id
    hard_labels: np.ndarray  # length = n_baskets
    soft_scores: np.ndarray  # (n_baskets, n_communities) - share of present categories per community


def assign_baskets_to_missions(matrix: csr_matrix, vocab: CategoryVocab, community_map: dict[int, int]) -> GraphMissionAssignment:
    """Hard assignment: which community do most of a basket's categories
    belong to (argmax). Soft assignment: the full distribution (share of
    present categories per community) - a basket with categories split
    across two communities gets partial credit for both, unlike the forced
    single-cluster assignment k-medoids/HDBSCAN give every basket."""
    n_communities = max(community_map.values()) + 1 if community_map else 0
    d = vocab.size
    membership = np.zeros((d, n_communities))
    for i, cid in enumerate(vocab.ids):
        comm = community_map.get(cid)
        if comm is not None:
            membership[i, comm] = 1.0

    community_counts = matrix.dot(membership)  # (n_baskets, n_communities), dense
    row_sums = community_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        soft_scores = np.where(row_sums > 0, community_counts / row_sums, 0.0)
    hard_labels = np.argmax(community_counts, axis=1)

    return GraphMissionAssignment(
        n_communities=n_communities, community_map=community_map,
        hard_labels=hard_labels, soft_scores=soft_scores,
    )
