"""Generic gap statistic (Tibshirani, Walther & Hastie, 2001) - the method
the paper actually uses to pick k, for both the variety-segmentation (Stage 1)
and mission (Stage 2) clustering steps.

Deliberately clustering-method-agnostic: callers inject how to fit+measure
dispersion for a candidate k, and how to draw one reference (null) dataset.
This lets the same gap-statistic machinery serve both Euclidean k-means
(Stage 1) and Jaccard k-medoids (Stage 2), which need different dispersion
and reference-generation logic but the same selection rule.
"""

import numpy as np
import pandas as pd


def gap_statistic(
    X_observed,
    k_range,
    fit_dispersion_fn,
    generate_reference_fn,
    B: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    fit_dispersion_fn(X, k, seed) -> (dispersion: float, labels: np.ndarray | None)
        Fits the clustering method for candidate k on X and returns its
        within-cluster dispersion W_k = sum_r [ D_r / (2 n_r) ], where D_r is
        the sum of pairwise (dis)similarities within cluster r - the general
        form from the original paper, valid for any distance measure.
    generate_reference_fn(seed) -> X_ref
        Draws one reference (null) dataset of the same shape as X_observed,
        sampled from an appropriate "no structure" distribution (e.g. uniform
        over the observed range for continuous features, independent
        per-column Bernoulli at the observed marginal rate for binary data).

    B reference datasets are drawn once and reused across all k, per the
    original algorithm, so gap(k) comparisons across k share the same Monte
    Carlo noise rather than each drawing fresh (noisier) references.

    Returns a DataFrame with columns: k, log_Wk, gap, sk, gap_rule_satisfied.
    Select k via `best_k_by_gap`: the smallest k with gap(k) >= gap(k+1) - sk(k+1).
    """
    references = [generate_reference_fn(random_state * 1000 + b + 1) for b in range(B)]

    rows = []
    for k in k_range:
        Wk, _ = fit_dispersion_fn(X_observed, k, random_state)
        log_Wk = np.log(Wk) if Wk > 0 else np.log(1e-12)

        ref_log_Wks = []
        for b, X_ref in enumerate(references):
            Wkb, _ = fit_dispersion_fn(X_ref, k, random_state * 1000 + b + 1)
            ref_log_Wks.append(np.log(Wkb) if Wkb > 0 else np.log(1e-12))

        gap = float(np.mean(ref_log_Wks) - log_Wk)
        sk = float(np.std(ref_log_Wks, ddof=0) * np.sqrt(1.0 + 1.0 / B))
        rows.append({"k": k, "log_Wk": log_Wk, "gap": gap, "sk": sk})

    df = pd.DataFrame(rows).reset_index(drop=True)
    df["gap_rule_satisfied"] = False
    for i in range(len(df) - 1):
        df.loc[i, "gap_rule_satisfied"] = bool(df.loc[i, "gap"] >= (df.loc[i + 1, "gap"] - df.loc[i + 1, "sk"]))
    return df


def best_k_by_gap(df: pd.DataFrame) -> int:
    """Smallest k satisfying the gap rule; falls back to argmax(gap) if no k
    in the swept range satisfies it (e.g. gap is still rising at the range's end)."""
    satisfied = df[df["gap_rule_satisfied"]]
    if len(satisfied):
        return int(satisfied.iloc[0]["k"])
    return int(df.loc[df["gap"].idxmax(), "k"])


def find_elbow_k(k_values, dispersion_values) -> int:
    """Elbow-method k selection: the point of maximum perpendicular distance
    from the chord connecting the first and last (k, dispersion) points on a
    min-max-normalized curve - the standard automated stand-in for
    "eyeballing" an elbow (the single-bend case of the kneedle heuristic,
    Satopaa et al. 2011), so the pick is reproducible rather than a judgment
    call on a plot.

    `dispersion_values` should be a monotonically-decreasing-ish measure of
    within-cluster spread (conventionally raw WCSS/inertia, i.e.
    `exp(log_Wk)` from a `gap_statistic` result - log_Wk itself compresses
    the curve and shifts where the bend falls).
    """
    k = np.asarray(k_values, dtype=float)
    w = np.asarray(dispersion_values, dtype=float)
    order = np.argsort(k)
    k, w = k[order], w[order]

    k_range, w_range = k.max() - k.min(), w.max() - w.min()
    k_n = (k - k.min()) / k_range if k_range > 0 else np.zeros_like(k)
    w_n = (w - w.min()) / w_range if w_range > 0 else np.zeros_like(w)

    p1 = np.array([k_n[0], w_n[0]])
    p2 = np.array([k_n[-1], w_n[-1]])
    line = p2 - p1
    line_len = np.hypot(*line)
    if line_len == 0:
        return int(k[0])

    vecs = np.stack([k_n, w_n], axis=1) - p1
    cross = line[0] * vecs[:, 1] - line[1] * vecs[:, 0]
    distances = np.abs(cross) / line_len
    return int(k[int(np.argmax(distances))])


def elbow_k_from_gap_df(df: pd.DataFrame) -> int:
    """Convenience wrapper: elbow k directly from a `gap_statistic` result
    (exponentiates its `log_Wk` column back to raw dispersion first)."""
    return find_elbow_k(df["k"], np.exp(df["log_Wk"]))
