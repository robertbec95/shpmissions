"""Step D - Stage 1: cluster baskets by variety (category count).

Separates concrete / relatively concrete / abstract trips, per the paper's
premise: relatively concrete trips are the ones with enough co-occurring
categories to reveal an affinity pattern, but not so many that the signal
drowns in stock-up noise.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .cluster_selection import best_k_by_gap, gap_statistic
from .config import RANDOM_STATE

SEGMENT_ORDER = ["concrete", "relatively_concrete", "abstract"]


@dataclass
class VarietySegmentation:
    baskets: pd.DataFrame  # input baskets + `segment` column
    bounds: dict[str, tuple[int, int | None]]  # segment -> (min, max) variety
    summary: pd.DataFrame  # segment -> n, n_share, value, value_share


def fit_variety_segments(baskets: pd.DataFrame, random_state: int = RANDOM_STATE) -> VarietySegmentation:
    variety = baskets["variety"].to_numpy().reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=10, random_state=random_state).fit(variety)

    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)  # low -> high variety
    cluster_to_segment = {cluster_id: SEGMENT_ORDER[rank] for rank, cluster_id in enumerate(order)}

    labels = pd.Series(km.labels_, index=baskets.index).map(cluster_to_segment)
    out = baskets.copy()
    out["segment"] = labels

    bounds = {}
    for segment in SEGMENT_ORDER:
        vals = out.loc[out["segment"] == segment, "variety"]
        bounds[segment] = (int(vals.min()), int(vals.max()))
    # Make the top segment explicitly unbounded above, for documentation clarity.
    top_segment = SEGMENT_ORDER[-1]
    bounds[top_segment] = (bounds[top_segment][0], None)

    summary = out.groupby("segment").agg(n=("basket_id", "size"), value=("basket_value", "sum"))
    summary = summary.reindex(SEGMENT_ORDER)
    summary["n_share"] = summary["n"] / summary["n"].sum() * 100
    summary["value_share"] = summary["value"] / summary["value"].sum() * 100

    return VarietySegmentation(baskets=out, bounds=bounds, summary=summary)


@dataclass
class GapKVarietySegmentation:
    baskets: pd.DataFrame  # input baskets + `segment` column
    bounds: dict[str, tuple[int, int | None]]
    summary: pd.DataFrame
    gap_k: int  # number of raw k-means clusters the gap statistic selected
    cluster_to_segment: dict[int, str]  # which raw cluster fell into which of the 3 bands
    fallback_to_fixed_k3: bool  # True if gap_k < 3 forced a fallback (see docstring)


def fit_variety_segments_gap_k(
    baskets: pd.DataFrame, k_range=range(1, 8), B: int = 10, sample_size: int = 50_000,
    random_state: int = RANDOM_STATE,
) -> GapKVarietySegmentation:
    """Stage 1, with k chosen by the gap statistic rather than fixed at 3 -
    matching the paper's description more literally ("training and testing
    sets... and the gap statistic procedure derives the optimal number of
    clusters", stated once but framed as applying to both the variety
    segmentation and the mission-identification stages). `fit_variety_segments`
    (fixed k=3) remains the default used elsewhere in this project; this is
    the more literal alternative.

    Once gap_k clusters are fit, they still need to collapse into the
    paper's 3-band framing (concrete/relatively-concrete/abstract) - the
    paper's own text isn't fully specific about the grouping rule (it
    mentions fitting more than 3 clusters and grouping them, without stating
    exactly how). This implementation splits the gap_k clusters, sorted by
    ascending mean variety, into 3 contiguous bands via `numpy.array_split`
    (as-equal-as-possible group sizes - e.g. 5 clusters -> sizes [2,2,1]) -
    a reasonable, deterministic, clearly-documented choice, not a claim that
    it's the paper's own exact rule. If gap_k < 3, there aren't enough
    clusters to form 3 non-empty bands at all - the fallback in that case is
    the fixed k=3 fit (`fit_variety_segments`), with `fallback_to_fixed_k3`
    set so callers/reports can flag it rather than silently pretending the
    gap statistic actually chose 3."""
    variety = baskets["variety"].to_numpy().reshape(-1, 1)
    gdf = select_k(baskets["variety"].to_numpy(), k_range=k_range, B=B, sample_size=sample_size, random_state=random_state)
    gap_k = best_k_by_gap(gdf)

    if gap_k < 3:
        seg3 = fit_variety_segments(baskets, random_state=random_state)
        return GapKVarietySegmentation(
            baskets=seg3.baskets, bounds=seg3.bounds, summary=seg3.summary,
            gap_k=gap_k, cluster_to_segment={}, fallback_to_fixed_k3=True,
        )

    km = KMeans(n_clusters=gap_k, n_init=10, random_state=random_state).fit(variety)
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)  # raw cluster ids, low -> high mean variety
    band_groups = np.array_split(order, 3)  # 3 contiguous bands of raw clusters
    cluster_to_segment = {}
    for rank, group in enumerate(band_groups):
        for cluster_id in group:
            cluster_to_segment[int(cluster_id)] = SEGMENT_ORDER[rank]

    labels = pd.Series(km.labels_, index=baskets.index).map(cluster_to_segment)
    out = baskets.copy()
    out["segment"] = labels

    bounds = {}
    for segment in SEGMENT_ORDER:
        vals = out.loc[out["segment"] == segment, "variety"]
        bounds[segment] = (int(vals.min()), int(vals.max())) if len(vals) else (0, 0)
    top_segment = SEGMENT_ORDER[-1]
    bounds[top_segment] = (bounds[top_segment][0], None)

    summary = out.groupby("segment").agg(n=("basket_id", "size"), value=("basket_value", "sum"))
    summary = summary.reindex(SEGMENT_ORDER)
    summary["n_share"] = summary["n"] / summary["n"].sum() * 100
    summary["value_share"] = summary["value"] / summary["value"].sum() * 100

    return GapKVarietySegmentation(
        baskets=out, bounds=bounds, summary=summary,
        gap_k=gap_k, cluster_to_segment=cluster_to_segment, fallback_to_fixed_k3=False,
    )


def apply_variety_bounds(baskets: pd.DataFrame, bounds: dict[str, tuple[int, int | None]]) -> pd.DataFrame:
    """Label baskets by fixed bounds instead of refitting k-means (e.g. for
    validation runs on held-out stores/days, to keep segment definitions
    consistent with the fitted run)."""
    out = baskets.copy()

    def label(v: int) -> str:
        for segment, (lo, hi) in bounds.items():
            if v >= lo and (hi is None or v <= hi):
                return segment
        return "abstract"  # fallback for variety above all bounds

    out["segment"] = out["variety"].apply(label)
    return out


def _kmeans_dispersion(X, k: int, seed: int, n_init: int = 5):
    """W_k for Euclidean k-means = pooled within-cluster sum of squares
    (sklearn's `inertia_`), which is algebraically equal to the general
    gap-statistic dispersion sum_r[D_r/(2n_r)] for squared-Euclidean D_r."""
    if X is None or len(X) < k:
        return np.inf, None
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed % (2**32 - 1)).fit(X)
    return float(km.inertia_), km.labels_


def select_k(
    variety: np.ndarray,
    k_range=range(1, 8),
    B: int = 10,
    sample_size: int = 50_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Gap statistic + silhouette diagnostics for the number of variety
    clusters, following Tibshirani, Walther & Hastie (2001) - the same
    method the paper uses (Section 5.4) to pick its cluster count before
    grouping clusters into concrete/relatively-concrete/abstract bands.

    Fits on a random subsample (default 50k) rather than the full basket
    population - 1-D k-means inertia from a 50k sample is already a stable
    estimate, and fitting B+1 models per k directly on millions of rows
    (with sklearn's internal n_init restarts on top) is needlessly slow.

    This is a diagnostic, not a drop-in replacement for
    `fit_variety_segments` (which stays fixed at k=3 to match the paper's
    three-band framing) - use it to check whether 3 is actually well
    supported by this data, or whether the paper's finer-grained k
    (they cluster into 5 and then group into 3 bands) fits better here too.
    """
    rng = np.random.default_rng(random_state)
    variety = np.asarray(variety)
    if len(variety) > sample_size:
        variety = rng.choice(variety, size=sample_size, replace=False)

    X = variety.astype(float).reshape(-1, 1)
    n = len(X)
    lo, hi = float(X.min()), float(X.max())

    def generate_reference(seed):
        rng = np.random.default_rng(seed)
        return rng.uniform(lo, hi, size=(n, 1))

    df = gap_statistic(X, k_range, _kmeans_dispersion, generate_reference, B=B, random_state=random_state)

    silhouettes = []
    for k in k_range:
        if k < 2:
            silhouettes.append(float("nan"))
            continue
        _, labels = _kmeans_dispersion(X, k, random_state)
        silhouettes.append(
            silhouette_score(X, labels) if labels is not None and len(set(labels)) > 1 else float("nan")
        )
    df["silhouette"] = silhouettes
    return df
