"""Joint structural-topic-model approximation (DMR - see
`shopping_missions.dmr_missions` for why DMR, not full STM), fit ONCE across
all 6 stores' pooled 2025 baskets, with store identity as a covariate on
topic prevalence - unlike every other method in this project, which fits
independently per store and so cannot ask "is this the same mission at a
different rate, or a genuinely different mission."

Scoped down from this project's usual budgets after measuring real
per-iteration cost on this machine - and scoped down a SECOND time after an
initial attempt at this reduced budget died with no traceback partway
through the k-sweep (consistent with an OS-level kill under sustained load
on this 8GB machine, not a code bug the isolated smoke tests that preceded
it didn't already rule out). Per-iteration cost varied 0.7-6s/iteration
across different runs of identical work - thermal throttling under
sustained load is the leading explanation, not confirmed. Current budget:
40 iterations per candidate k in the topic-count sweep, 150 for the
reference fit, 3 stability reruns at 100 iterations each - all well below
this project's usual depth elsewhere, disclosed in the combined report, not
smoothed over. `fit_dmr_missions(..., verbose=True)` now prints after every
training chunk specifically so a repeat failure is diagnosable (how far did
it get) rather than silent.

Usage:
    PYTHONPATH=src python3 scripts/run_dmr_missions_joint.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from shopping_missions import categories, cluster_selection, dmr_missions, ingest_v2, stability, temporal
from shopping_missions import baskets as baskets_mod
from shopping_missions.config import PROJECT_ROOT, RANDOM_STATE

DOCS_DIR = PROJECT_ROOT / "docs" / "dmr_missions"
DOCS_STABILITY_DIR = PROJECT_ROOT / "docs" / "stability"
# Scope reduced a second time after a live run died with no traceback partway
# through the k-sweep (consistent with an OS-level kill on this 8GB machine
# under sustained load, not a code bug - the isolated single-fit smoke tests
# that preceded it succeeded cleanly). Smaller K_RANGE/iteration counts than
# the first attempt, plus per-chunk `verbose` logging below, so a repeat
# failure is at least diagnosable instead of silent.
K_RANGE = range(3, 6)
SWEEP_ITERATIONS = 40
SWEEP_STEP = 10
REFERENCE_ITERATIONS = 150
REFERENCE_STEP = 25
STABILITY_RUNS = 3
STABILITY_ITERATIONS = 100
STABILITY_STEP = 25


def _store_matrices():
    """One (store_label, quantity_matrix, baskets_df) tuple per store, in a
    fixed order (sorted by location id) - re-ingests from the source
    parquet files, cheap (~13s total for all 6 stores) next to training
    cost. Called fresh inside `build_all_docs` every time a new fit needs
    the corpus, so at most one fit's worth of documents is ever held in
    memory - see `dmr_missions.quantity_matrix_to_docs`'s docstring for why
    this matters on this machine's 8GB RAM."""
    stores = ingest_v2.list_store_files()
    vocab_df = ingest_v2.build_global_category_vocab()
    vocab = categories.build_vocab_from_pairs(vocab_df)
    for loc_id in sorted(stores):
        info = stores[loc_id]
        con = ingest_v2.connect_store_lines(info["path"], vocab_df, include_daypart=True)
        data = baskets_mod.build_basket_data(con, vocab, include_timestamp=True, include_quantity_matrix=True)
        con.close()
        keep_mask = ((data.baskets["variety"] >= 2) & (data.baskets["basket_value"] > 0)).to_numpy()
        baskets = temporal.add_daypart(data.baskets.loc[keep_mask].reset_index(drop=True))
        qm = data.quantity_matrix[keep_mask]
        store_label = f"{info['format']}|{info['county']}"
        yield store_label, qm, baskets, vocab


def build_all_docs():
    """Fresh generator over (words, metadata) for the full pooled corpus,
    same store order and same within-store basket order (sorted by
    basket_id, from `baskets.build_basket_data`) every call - required so
    stability testing's positional ARI/NMI comparison across reruns is
    actually comparing the same basket to itself, not silently misaligned
    rows."""
    for store_label, qm, baskets, vocab in _store_matrices():
        yield from dmr_missions.quantity_matrix_to_docs(qm, vocab, store_label)


def build_basket_metadata() -> pd.DataFrame:
    """Basket-level (store_label, day_part, basket_value) in the exact same
    row order `build_all_docs()` yields documents - every variety>=2
    basket survives into at least one token (the quantity floor in
    `quantity_matrix_to_docs` guarantees every present category contributes
    >=1 token), so doc count equals basket count per store and this aligns
    positionally with a reference fit's `hard_labels`/`soft_membership`
    without needing to thread metadata through the model itself."""
    frames = []
    for store_label, qm, baskets, vocab in _store_matrices():
        df = baskets[["basket_id", "day_part", "basket_value"]].copy()
        df["store_label"] = store_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    print("=== DMR joint structural-topic-model approximation, all 6 stores pooled ===")
    t_all = time.time()

    print(f"\n--- Topic-count sweep: k in {list(K_RANGE)}, {SWEEP_ITERATIONS} iterations each ---")
    t0 = time.time()
    scree_rows = []
    for kk in K_RANGE:
        tk = time.time()
        sweep_fit = dmr_missions.fit_dmr_missions(
            build_all_docs(), n_topics=kk, iterations=SWEEP_ITERATIONS, step=SWEEP_STEP, seed=RANDOM_STATE, verbose=True,
        )
        scree_rows.append({"k": kk, "perplexity": float(sweep_fit.model.perplexity), "log_likelihood": sweep_fit.log_likelihood})
        print(f"  k={kk}: perplexity={sweep_fit.model.perplexity:.2f} ({time.time()-tk:.1f}s, {time.time()-t0:.1f}s elapsed)")
        del sweep_fit
    scree = pd.DataFrame(scree_rows)
    k = cluster_selection.find_elbow_k(scree["k"], scree["perplexity"])
    print(f"elbow k={k} (sweep total {time.time()-t0:.1f}s)")

    print(f"\n--- Reference fit: k={k}, {REFERENCE_ITERATIONS} iterations ---")
    t1 = time.time()
    fit = dmr_missions.fit_dmr_missions(
        build_all_docs(), n_topics=k, iterations=REFERENCE_ITERATIONS, step=REFERENCE_STEP, seed=RANDOM_STATE, verbose=True,
    )
    print(f"reference fit done, log_likelihood={fit.log_likelihood:.1f} ({time.time()-t1:.1f}s)")

    profiles = dmr_missions.profile_topics(fit, top_n=10)
    for p in profiles:
        top = ", ".join(f"{w['category']}({w['weight']:.2f})" for w in p["top_words"][:5])
        print(f"  topic {p['topic_id']}: hard_share={p['hard_share_pct']}% soft_wt={p['avg_soft_weight_pct']}% top=[{top}]")

    print(f"\n--- Stability: {STABILITY_RUNS} reruns, {STABILITY_ITERATIONS} iterations each ---")
    t2 = time.time()
    stab_labelings = []
    for i in range(STABILITY_RUNS):
        tr = time.time()
        run_seed = RANDOM_STATE * 10_000 + i
        run_fit = dmr_missions.fit_dmr_missions(
            build_all_docs(), n_topics=k, iterations=STABILITY_ITERATIONS, step=STABILITY_STEP, seed=run_seed, verbose=True,
        )
        stab_labelings.append(run_fit.hard_labels)
        print(f"  rerun {i+1}/{STABILITY_RUNS} done ({time.time()-tr:.1f}s, {time.time()-t2:.1f}s elapsed)")
        del run_fit
    stab = _pairwise_stability(stab_labelings, k)
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    print(f"stability done ({time.time()-t2:.1f}s): ARI mean={ari.mean():.4f} | NMI mean={nmi.mean():.4f}")

    print("\n--- Basket metadata + crosstabs ---")
    t3 = time.time()
    meta = build_basket_metadata()
    assert len(meta) == len(fit.hard_labels), f"basket/doc count mismatch: {len(meta)} vs {len(fit.hard_labels)}"
    meta = meta.copy()
    meta["topic"] = fit.hard_labels
    store_xtab = _crosstab(meta, "store_label")
    daypart_xtab = _crosstab(meta, "day_part")
    print(f"crosstabs built ({time.time()-t3:.1f}s)")

    _write_report(k, scree, fit, profiles, stab, store_xtab, daypart_xtab, meta, time.time() - t_all)
    print(f"\nTotal runtime {time.time()-t_all:.1f}s")
    print(f"Wrote {DOCS_DIR / 'level3_dmr_missions_joint_2025.md'}")


def _pairwise_stability(labelings: list, n_topics: int) -> stability.DMRStabilityResult:
    """Builds a `stability.DMRStabilityResult` from hard-label arrays
    collected by `main()`'s own per-rerun loop (kept inline there, rather
    than calling `stability.run_stability_test_dmr` directly, purely so
    each rerun's progress and timing prints live - a background job this
    long needs visible checkpoints, not a single opaque multi-hour call)."""
    n_runs = len(labelings)
    ari = np.eye(n_runs)
    nmi = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a = adjusted_rand_score(labelings[i], labelings[j])
            m = normalized_mutual_info_score(labelings[i], labelings[j])
            ari[i, j] = ari[j, i] = a
            nmi[i, j] = nmi[j, i] = m
    return stability.DMRStabilityResult(n_runs=n_runs, n_topics=n_topics, pairwise_ari=ari, pairwise_nmi=nmi)


def _crosstab(meta: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    xtab = meta.groupby([dim_col, "topic"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_report(k, scree, fit, profiles, stab, store_xtab, daypart_xtab, meta, elapsed):
    ari, nmi = stab.off_diagonal_ari, stab.off_diagonal_nmi
    prevalence = dmr_missions.store_topic_prevalence(fit)
    n_docs = len(fit.hard_labels)

    lines = []
    lines.append("# Joint DMR structural-topic-model approximation (2025 data, all 6 stores pooled)")
    lines.append("")
    lines.append(
        f"{n_docs:,} baskets across all 6 stores (variety>=2, basket_value>0), fit as ONE Dirichlet "
        f"Multinomial Regression model with store identity as a covariate on topic prevalence - see "
        f"`shopping_missions.dmr_missions` module docstring for why DMR rather than full STM (no maintained "
        f"Python STM port; R is not installed in this environment). {k} topics chosen by perplexity elbow. "
        f"Total run time {elapsed:.1f}s."
    )
    lines.append("")

    lines.append("## Disclosed compute trade-offs")
    lines.append("")
    lines.append(
        f"Measured at ~6s/iteration on this ~{n_docs:,}-document pooled corpus on this machine (8GB RAM, "
        f"single-threaded `workers=1` required for reproducibility) - this project's usual budgets (1000 "
        f"iterations, 20 stability reruns) would take multiple hours just for one stage. Scoped down instead: "
        f"topic-count sweep at {SWEEP_ITERATIONS} iterations/k (not full depth), reference fit at "
        f"{REFERENCE_ITERATIONS} iterations (not the library default's 1000), stability at {STABILITY_RUNS} "
        f"reruns x {STABILITY_ITERATIONS} iterations (not this project's usual 20 at full depth). Treat "
        f"perplexity/log-likelihood values here as directional, not fully converged."
    )
    lines.append("")

    lines.append("## Topic-count selection (perplexity scree)")
    lines.append("")
    lines.append("| k | Perplexity | Log-likelihood |")
    lines.append("|---|---|---|")
    for _, row in scree.iterrows():
        marker = " <- elbow" if int(row["k"]) == k else ""
        lines.append(f"| {int(row['k'])} | {row['perplexity']:.2f} | {row['log_likelihood']:.1f}{marker} |")
    lines.append("")

    lines.append("## Topic profiles (native top words + per-store prevalence)")
    lines.append("")
    for p in profiles:
        words = ", ".join(f"{w['category']} ({w['weight']*100:.0f}%)" for w in p["top_words"][:6])
        lines.append(f"### Topic {p['topic_id']} - hard share {p['hard_share_pct']}%, avg soft weight {p['avg_soft_weight_pct']}%")
        lines.append("")
        lines.append(f"Top words: {words}")
        lines.append("")
        lines.append("| Store | DMR prevalence |")
        lines.append("|---|---|")
        for s, v in sorted(p["store_prevalence_pct"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {s} | {v}% |")
        lines.append("")

    lines.append("## Store x topic, DMR-fitted prevalence vs. empirical hard-assignment share")
    lines.append("")
    lines.append(
        "The model's fitted prevalence (from `lambdas`, softmax-normalized per store) is a *typical* mix "
        "independent of any single basket; the empirical share is what baskets actually got hard-assigned to "
        "in this fit - close agreement between the two is a sanity check, not a given."
    )
    lines.append("")
    emp_pivot = store_xtab.pivot(index="store_label", columns="topic", values="share_within_dim").fillna(0)
    emp_pivot.columns = [f"T{c} (empirical)" for c in emp_pivot.columns]
    lines.append(emp_pivot.round(1).to_markdown())
    lines.append("")
    fit_pivot = prevalence.T
    fit_pivot.columns = [f"T{c.split('_')[1]} (fitted)" for c in fit_pivot.columns]
    lines.append((fit_pivot * 100).round(1).to_markdown())
    lines.append("")

    lines.append("## Stability (reduced budget - see disclosed trade-offs above)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |")
    lines.append(f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |")
    lines.append("")

    lines.append("## Cross-validation by day-part (hard assignment)")
    lines.append("")
    order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening"]
    dp_pivot = daypart_xtab.pivot(index="day_part", columns="topic", values="share_within_dim").fillna(0)
    dp_pivot = dp_pivot.reindex([o for o in order if o in dp_pivot.index])
    dp_pivot.columns = [f"T{c}" for c in dp_pivot.columns]
    lines.append(dp_pivot.round(1).to_markdown())
    lines.append("")

    lines.append("Reproduce with: `PYTHONPATH=src python3 scripts/run_dmr_missions_joint.py`")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "level3_dmr_missions_joint_2025.md").write_text("\n".join(lines))

    stab_lines = [
        "# Stability test: level3_dmr_missions_joint_2025", "",
        f"{stab.n_runs} independent DMR fits ({stab.n_topics} topics each, {STABILITY_ITERATIONS} iterations, "
        f"reduced from this project's usual 20 runs / full-depth iterations - see the combined report's "
        f"disclosed compute trade-offs) on the identical full pooled corpus - no basket sampling, isolates "
        f"non-convex solution variance under a reduced iteration budget.",
        "", "| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|",
        f"| ARI | {ari.mean():.4f} | {ari.std():.4f} | {ari.min():.4f} | {ari.max():.4f} |",
        f"| NMI | {nmi.mean():.4f} | {nmi.std():.4f} | {nmi.min():.4f} | {nmi.max():.4f} |",
        "", "Reproduce with: `PYTHONPATH=src python3 scripts/run_dmr_missions_joint.py`",
    ]
    DOCS_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_STABILITY_DIR / "level3_dmr_missions_joint_2025.md").write_text("\n".join(stab_lines))


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
