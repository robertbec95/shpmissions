"""Item2Vec-style category embeddings, as a richer alternative to pairwise
lift for building the mission graph.

Motivation: `graph_missions.compute_value_pmi_graph` already uses pairwise
lift between every category pair - but two categories that rarely co-occur
directly, while both reliably co-occurring with a shared third category,
have a genuine similarity that pairwise lift can't see (A and B both show
up near C, but A and B themselves are rare together). Embeddings capture
that transitive/higher-order structure: each category gets a dense vector
positioned by ALL the baskets it appears in, not just its relationship to
one other category at a time.

Rather than training an actual word2vec/skip-gram-negative-sampling model
(which would reintroduce exactly the kind of stochastic-training instability
this project has spent most of its effort eliminating - a random init, SGD,
negative-sampling draws), this uses the proven equivalence between SGNS and
matrix factorization (Levy & Goldberg, 2014, "Neural Word Embedding as
Implicit Matrix Factorization"): skip-gram with negative sampling implicitly
factorizes a shifted PMI matrix. With a vocabulary this small (~80-99
categories), that factorization can be done exactly and deterministically
via a single dense SVD - no training loop, no randomness, no new heavy
dependency (gensim/PyTorch) on a resource-constrained machine, and the
result depends only on the (already-validated) full-population co-occurrence
statistics, not on any random seed - preserving the determinism that made
`graph_missions`'s value-weighted graph the most stable method in this
project.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from .categories import CategoryVocab
from .graph_missions import compute_cooccurrence


def compute_ppmi_matrix(
    value_matrix: csr_matrix, vocab: CategoryVocab, min_prevalence: float = 1e-4,
) -> tuple[np.ndarray, list[int]]:
    """Dense (k x k) positive-PMI matrix (negative/zero PMI clipped to 0,
    the standard PPMI convention, and the same clipping
    `compute_value_pmi_graph` already applies via its `min_lift` edge
    filter) over categories clearing `min_prevalence`. Reuses the corrected
    column-sum marginal (not the co-occurrence diagonal, which only equals
    the true marginal for binary data - see `compute_value_pmi_graph`'s
    docstring for why that distinction matters for continuous value data).

    Returns (ppmi, category_ids) - category_ids gives the row/column order,
    since excluded low-prevalence categories are dropped entirely (their
    near-zero mass makes PMI dominated by a handful of baskets, the same
    noise `compute_value_pmi_graph`'s node filtering guards against)."""
    n = value_matrix.shape[0]
    co = compute_cooccurrence(value_matrix)
    support = np.asarray(value_matrix.sum(axis=0)).ravel()
    p_i = support / n
    keep = np.where(p_i >= min_prevalence)[0]
    category_ids = [vocab.ids[i] for i in keep]

    k = len(keep)
    ppmi = np.zeros((k, k))
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            if a == b:
                continue
            joint_mass = co[i, j]
            if joint_mass <= 0:
                continue
            p_ij = joint_mass / n
            lift = p_ij / (p_i[i] * p_i[j])
            if lift <= 0:
                continue
            ppmi[a, b] = max(0.0, np.log(lift))
    return ppmi, category_ids


@dataclass
class CategoryEmbeddings:
    vectors: np.ndarray  # (k, dim)
    category_ids: list[int]
    explained_variance_ratio: np.ndarray  # (dim,) - share of the PPMI matrix's structure each dimension captures


def train_item2vec(ppmi: np.ndarray, category_ids: list[int], dim: int = 16) -> CategoryEmbeddings:
    """Embeds categories via truncated SVD of the PPMI matrix - the exact,
    deterministic equivalent of what skip-gram-negative-sampling training
    converges to (Levy & Goldberg 2014), without the training. PPMI is
    symmetric (co-occurrence is symmetric), so the left and right singular
    vectors are the same up to sign; `vectors = U * sqrt(S)` is the standard
    "symmetric" factorization, giving one vector per category rather than
    the separate word/context vector pair word2vec normally keeps - a
    mission-finding use needs "how similar are these categories", not a
    directional word/context distinction."""
    U, S, _Vt = np.linalg.svd(ppmi, full_matrices=False)
    dim = min(dim, len(S))
    vectors = U[:, :dim] * np.sqrt(S[:dim])[None, :]
    explained = (S**2) / (S**2).sum()
    return CategoryEmbeddings(vectors=vectors, category_ids=category_ids, explained_variance_ratio=explained[:dim])


def compute_embedding_similarity_graph(
    emb: CategoryEmbeddings, vocab: CategoryVocab, min_cosine: float = 0.0,
) -> nx.Graph:
    """Builds a category graph from embedding cosine similarity - same
    shape as `graph_missions.compute_value_pmi_graph`'s output (a drop-in
    input to `graph_missions.detect_communities`), but using embedding
    closeness instead of direct pairwise lift, so it can connect categories
    that rarely co-occur directly but consistently appear near the same
    others."""
    norms = np.linalg.norm(emb.vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = emb.vectors / norms
    sim = normed @ normed.T

    G = nx.Graph()
    for i, cid in enumerate(emb.category_ids):
        G.add_node(cid, name=vocab.names[cid])
    k = len(emb.category_ids)
    for a in range(k):
        for b in range(a + 1, k):
            s = sim[a, b]
            if s <= min_cosine:
                continue
            G.add_edge(emb.category_ids[a], emb.category_ids[b], weight=float(s), cosine=float(s))
    return G
