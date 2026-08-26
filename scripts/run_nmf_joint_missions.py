"""Joint IDF-weighted NMF topic model for missions, fit ONCE across ALL 6
stores' full basket population (every basket with variety>=1, not just
"relatively concrete" or variety>=2 - a deliberate deviation from every
other method in this project, at explicit user request: "model all
baskets, rather than retaining only the relatively concrete group").

Feature representation: x_bc = log(1 + quantity_bc) x idf_c
(`topic_missions.build_idf_weighted_matrix`) - item COUNT, not spend, so
inflation/premium products/promotions don't masquerade as distinct
missions purely by moving more money. This is a genuine change from the
per-store NMF already in this project (`run_nmf_missions_2025.py`), which
weights by value share.

Store/format are NOT baked into the factorization itself - NMF has no
covariate mechanism the way DMR's Dirichlet-multinomial regression does.
Instead, one topic space is fit jointly across the pooled population, and
topic prevalence is broken out BY STORE post-hoc (average soft membership,
hard-assignment share) - the NMF-appropriate analogue of DMR's fitted
prevalence effect, honestly reported as a post-hoc breakdown of a jointly-
fit model, not a term in the optimization objective.

Topic count selection implements 2 of the 6 criteria the user specified
(reconstruction-error elbow as a coherence-adjacent proxy; minimum mission
prevalence flagging) plus a genuine bootstrap-stability check (basket
resampling with replacement, not just seed variance - see
`topic_missions.bootstrap_stability`). NOT implemented, disclosed rather
than silently skipped: formal temporal held-out likelihood (attempted as a
lighter proxy - held-out reconstruction error on a date-based split),
NPMI/UMass topic coherence scoring, retail-expert interpretability (needs a
human), and downstream predictive value (needs a defined business task).

Usage:
    PYTHONPATH=src python3 scripts/run_nmf_joint_missions.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.sparse import vstack

from shopping_missions import categories, cluster_selection, ingest_v2, stability, temporal, topic_missions
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "topic_missions"
K_RANGE = range(6, 16, 2)  # 6, 8, 10, 12, 14 - user asked for ~6-15
N_STABILITY_SEED_RUNS = 10
N_BOOTSTRAP_ROUNDS = 8
HELD_OUT_FRACTION = 0.2  # most recent ~20% of days per store, by basket timestamp
MIN_PREVALENCE_PCT = 1.0  # topics below this hard-share % are flagged thin


def main():
    print("=== Joint IDF-weighted NMF, ALL baskets (variety>=1), all 6 stores pooled ===")
    t_all = time.time()

    print("\n--- Ingest + pool ---")
    t0 = time.time()
    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)

    value_parts, qty_parts, presence_parts, meta_parts = [], [], [], []
    for loc_id in sorted(stores):
        info = stores[loc_id]
        con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
        data = baskets_mod.build_basket_data(
            con, vocab, include_timestamp=True, include_value_matrix=True, include_quantity_matrix=True,
        )
        con.close()
        keep_mask = ((data.baskets["variety"] >= 1) & (data.baskets["basket_value"] > 0)).to_numpy()
        baskets = temporal.add_daypart(data.baskets.loc[keep_mask].reset_index(drop=True))
        baskets["store_label"] = f"{info['format']}|{info['county']}"
        value_parts.append(data.value_matrix[keep_mask])
        qty_parts.append(data.quantity_matrix[keep_mask])
        presence_parts.append(data.matrix[keep_mask])
        meta_parts.append(baskets[["basket_id", "store_label", "day_part", "basket_value", "variety", "timestamp"]])
        print(f"  {info['format']} / {info['county']}: {int(keep_mask.sum()):,} baskets (variety>=1)")

    value_matrix = vstack(value_parts).tocsr()
    quantity_matrix = vstack(qty_parts).tocsr()
    presence_matrix = vstack(presence_parts).tocsr()
    meta = pd.concat(meta_parts, ignore_index=True)
    n = value_matrix.shape[0]
    print(f"pooled: {n:,} baskets, {vocab.size} categories ({time.time()-t0:.1f}s)")

    print("\n--- Feature construction: x_bc = log(1+qty)*idf ---")
    t1 = time.time()
    weighted = topic_missions.build_idf_weighted_matrix(quantity_matrix, presence_matrix)
    print(f"weighted matrix built, nnz={weighted.nnz:,} ({time.time()-t1:.1f}s)")

    n_holdout_days = None
    train_mask, test_mask = _temporal_split(meta, HELD_OUT_FRACTION)
    print(f"temporal split: {train_mask.sum():,} train baskets, {test_mask.sum():,} held-out baskets (most recent {HELD_OUT_FRACTION*100:.0f}% of days per store)")

    print(f"\n--- Topic-count sweep: k in {list(K_RANGE)} ---")
    t2 = time.time()
    scree_rows = []
    for kk in K_RANGE:
        tk = time.time()
        train_fit = topic_missions.fit_nmf_missions(weighted[train_mask], kk, random_state=RANDOM_STATE)
        held_out_err = _held_out_reconstruction_error(weighted, train_fit, train_mask, test_mask, kk)
        n_thin = sum(1 for share in _hard_shares(train_fit, kk) if share < MIN_PREVALENCE_PCT)
        scree_rows.append({
            "k": kk, "train_reconstruction_err": train_fit.reconstruction_err,
            "held_out_reconstruction_err": held_out_err, "n_thin_topics": n_thin,
        })
        print(f"  k={kk}: train_err={train_fit.reconstruction_err:.2f} held_out_err={held_out_err:.2f} thin_topics={n_thin} ({time.time()-tk:.1f}s)")
    scree = pd.DataFrame(scree_rows)
    k = cluster_selection.find_elbow_k(scree["k"], scree["held_out_reconstruction_err"])
    print(f"elbow k={k} by held-out reconstruction error (sweep total {time.time()-t2:.1f}s)")

    print(f"\n--- Reference fit: k={k}, full pooled population ---")
    t3 = time.time()
    fit = topic_missions.fit_nmf_missions(weighted, k, random_state=RANDOM_STATE)
    print(f"reference fit done, reconstruction_err={fit.reconstruction_err:.2f} ({time.time()-t3:.1f}s)")
    profiles = topic_missions.profile_topics(weighted, fit, vocab, meta, top_n=10)
    for p in profiles:
        top = ", ".join(f"{c['category']}({c['loading']:.2f})" for c in p["native_top_categories"][:5])
        flag = " [THIN]" if p["hard_share_pct"] < MIN_PREVALENCE_PCT else ""
        print(f"  topic {p['topic_id']}{flag}: hard_share={p['hard_share_pct']}% soft_wt={p['avg_soft_weight_pct']}% top=[{top}]")

    print(f"\n--- Stability: {N_STABILITY_SEED_RUNS} seed reruns + {N_BOOTSTRAP_ROUNDS} bootstrap rounds ---")
    t4 = time.time()
    seed_stab = stability.run_stability_test_nmf(weighted, k, n_runs=N_STABILITY_SEED_RUNS, base_seed=RANDOM_STATE)
    seed_ari, seed_nmi = seed_stab.off_diagonal_ari, seed_stab.off_diagonal_nmi
    print(f"  seed stability done ({time.time()-t4:.1f}s): ARI mean={seed_ari.mean():.4f} | NMI mean={seed_nmi.mean():.4f}")
    t5 = time.time()
    boot_stab = topic_missions.bootstrap_stability(weighted, k, n_bootstrap=N_BOOTSTRAP_ROUNDS, random_state=RANDOM_STATE)
    print(f"  bootstrap stability done ({time.time()-t5:.1f}s): mean best-match cosine={boot_stab['mean'].mean():.4f}")

    print("\n--- Store prevalence (post-hoc, not a model covariate) ---")
    meta_topic = meta.copy()
    meta_topic["topic"] = fit.hard_labels
    meta_topic["soft"] = list(fit.soft_membership)
    store_hard = _crosstab(meta_topic, "store_label")
    store_soft = _store_soft_prevalence(meta_topic, k)
    daypart_xtab = _crosstab(meta_topic, "day_part")

    _write_report(k, n, scree, fit, profiles, seed_stab, boot_stab, store_hard, store_soft, daypart_xtab, time.time() - t_all)
    print(f"\nTotal runtime {time.time()-t_all:.1f}s")
    print(f"Wrote {DOCS_DIR / 'level3_nmf_joint_all_baskets_2025.md'}")


def _temporal_split(meta: pd.DataFrame, held_out_fraction: float):
    """Per-store temporal split: the most recent `held_out_fraction` of
    each store's days are held out - a per-store split (not a single global
    date cutoff) because the 6 stores' data spans the same calendar year
    but a global cutoff would let a store with different day-of-week mix
    near year-end bias the held-out set unevenly. Cutoff is computed on the
    datetime64 timestamp directly (not `.dt.date`, an object-dtype Series
    that silently breaks `.quantile()`), then compared at day granularity."""
    train_mask = np.zeros(len(meta), dtype=bool)
    timestamps = pd.to_datetime(meta["timestamp"])
    for store in meta["store_label"].unique():
        store_mask = (meta["store_label"] == store).to_numpy()
        store_ts = timestamps[store_mask]
        cutoff = store_ts.quantile(1 - held_out_fraction)
        train_mask[store_mask] = (store_ts <= cutoff).to_numpy()
    return train_mask, ~train_mask


def _held_out_reconstruction_error(weighted, train_fit, train_mask, test_mask, kk):
    """Proxy for temporal held-out likelihood: NMF has no natural
    likelihood, so this projects held-out baskets onto the *fixed*
    topic-category loadings H learned from the training baskets only
    (`non_negative_factorization(..., H=train_fit.H, update_H=False)`
    solves for W with H frozen, rather than re-fitting a whole new model),
    then reports reconstruction error on that held-out set - the standard
    practical substitute for held-out likelihood in factorization models,
    not literal held-out likelihood."""
    from sklearn.decomposition import non_negative_factorization
    X_test = weighted[test_mask]
    W_test, _, _ = non_negative_factorization(
        X_test, H=train_fit.H, update_H=False, n_components=kk,
        init="custom", W=np.full((X_test.shape[0], kk), 1.0 / kk),
        solver="cd", beta_loss="frobenius", max_iter=400, random_state=RANDOM_STATE,
    )
    recon = W_test @ train_fit.H
    actual = np.asarray(X_test.todense())
    return float(np.linalg.norm(actual - recon, ord="fro"))


def _hard_shares(fit, kk):
    n = len(fit.hard_labels)
    return [float((fit.hard_labels == t).sum()) / n * 100 for t in range(kk)]


def _crosstab(meta: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    xtab = meta.groupby([dim_col, "topic"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _store_soft_prevalence(meta_topic: pd.DataFrame, k: int) -> pd.DataFrame:
    """Average soft-membership weight per topic, grouped by store - the
    NMF-appropriate analogue of DMR's fitted `lambdas`-based prevalence:
    computed post-hoc from a jointly-fit model's soft assignments, not from
    a covariate term in the optimization (NMF has none)."""
    rows = []
    for store, grp in meta_topic.groupby("store_label", observed=True):
        avg = np.mean(np.stack(grp["soft"].to_numpy()), axis=0)
        for t in range(k):
            rows.append({"store_label": store, "topic": t, "avg_soft_pct": round(float(avg[t]) * 100, 2)})
    return pd.DataFrame(rows)


def _write_report(k, n, scree, fit, profiles, seed_stab, boot_stab, store_hard, store_soft, daypart_xtab, elapsed):
    seed_ari, seed_nmi = seed_stab.off_diagonal_ari, seed_stab.off_diagonal_nmi

    lines = []
    lines.append("# Joint IDF-weighted NMF (all baskets, all 6 stores pooled, 2025 data)")
    lines.append("")
    lines.append(
        f"{n:,} baskets across all 6 stores (variety>=1, basket_value>0 - the FULL population, not just "
        f"relatively-concrete, at explicit user request). Feature: x_bc = log(1+quantity_bc) x idf_c "
        f"(`topic_missions.build_idf_weighted_matrix`) - item count, not spend, so premium products/promotions/"
        f"inflation don't masquerade as distinct missions. One NMF fit jointly across the full pooled population, "
        f"{k} topics selected by held-out reconstruction error elbow. Total run time {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Scope: which of the 6 requested k-selection criteria are implemented")
    lines.append("")
    lines.append("| Criterion | Status |")
    lines.append("|---|---|")
    lines.append("| Temporal held-out likelihood | Approximated - NMF has no likelihood; held-out reconstruction error on a per-store date-split instead (see table below) |")
    lines.append("| Topic coherence | Not implemented (would need NPMI/UMass scoring over category co-occurrence) |")
    lines.append("| Bootstrap stability | Implemented - genuine basket resampling with replacement, not just seed variance (see below) |")
    lines.append(f"| Minimum mission prevalence | Implemented - topics below {MIN_PREVALENCE_PCT}% hard share flagged [THIN] |")
    lines.append("| Interpretability by retail experts | Not automatable - top categories per topic are printed below for human review |")
    lines.append("| Downstream predictive value | Not implemented (no defined business task/label to predict against) |")
    lines.append("")

    lines.append("## Topic-count sweep")
    lines.append("")
    lines.append("| k | Train reconstruction err | Held-out reconstruction err | Thin topics (<1% share) |")
    lines.append("|---|---|---|---|")
    for _, row in scree.iterrows():
        marker = " <- elbow" if int(row["k"]) == k else ""
        lines.append(f"| {int(row['k'])} | {row['train_reconstruction_err']:.2f} | {row['held_out_reconstruction_err']:.2f}{marker} | {int(row['n_thin_topics'])} |")
    lines.append("")

    lines.append("## Topic profiles")
    lines.append("")
    for p in profiles:
        flag = " **[THIN]**" if p["hard_share_pct"] < MIN_PREVALENCE_PCT else ""
        words = ", ".join(f"{c['category']} ({c['loading']*100:.0f}%)" for c in p["native_top_categories"][:6])
        lines.append(f"### Topic {p['topic_id']}{flag} - hard share {p['hard_share_pct']}%, avg soft weight {p['avg_soft_weight_pct']}%")
        lines.append("")
        lines.append(f"Top categories (IDF-weighted loading): {words}")
        lines.append("")

    lines.append("## Store prevalence (post-hoc breakdown of the jointly-fit model, NOT a model covariate)")
    lines.append("")
    lines.append(
        "NMF has no covariate mechanism like DMR's prevalence regression - this is the empirical average "
        "soft-membership weight per topic, computed after fitting, grouped by store."
    )
    lines.append("")
    soft_pivot = store_soft.pivot(index="store_label", columns="topic", values="avg_soft_pct").fillna(0)
    soft_pivot.columns = [f"T{c}" for c in soft_pivot.columns]
    lines.append(soft_pivot.round(1).to_markdown())
    lines.append("")

    lines.append("## Stability")
    lines.append("")
    lines.append("### Seed variance (same full matrix, different random inits)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {seed_ari.mean():.4f} | {seed_ari.std():.4f} | {seed_ari.min():.4f} | {seed_ari.max():.4f} |")
    lines.append(f"| NMI | {seed_nmi.mean():.4f} | {seed_nmi.std():.4f} | {seed_nmi.min():.4f} | {seed_nmi.max():.4f} |")
    lines.append("")
    lines.append("### Bootstrap stability (basket resampling with replacement, H-row cosine match to reference)")
    lines.append("")
    lines.append(boot_stab.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## Cross-validation by day-part (hard assignment)")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    dp_pivot = daypart_xtab.pivot(index="day_part", columns="topic", values="share_within_dim").fillna(0)
    dp_pivot = dp_pivot.reindex([o for o in order if o in dp_pivot.index])
    dp_pivot.columns = [f"T{c}" for c in dp_pivot.columns]
    lines.append(dp_pivot.round(1).to_markdown())
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_joint_missions.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_nmf_joint_all_baskets_2025.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
