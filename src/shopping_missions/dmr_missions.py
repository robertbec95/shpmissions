"""Joint structural-topic-model approximation: Dirichlet Multinomial
Regression (DMR, Mimno & McCallum 2008), fit once across ALL stores'
baskets pooled together, with store identity as a document covariate on
topic *prevalence*.

Motivation: every method in this project so far - including the NMF topic
model - was fit independently per store, so there is no way to ask "is this
the same mission at a different rate, or a genuinely different mission" -
each store gets its own private topic space. A Structural Topic Model
(Roberts, Stewart & Tingley) is the standard tool for exactly this: one
shared topic space, with document metadata (here, store) allowed to shift
how much of each topic a document draws on. Real STM (R's `stm` package)
additionally lets topic *content* (the words themselves) vary by covariate
and has no maintained Python port - not available in this project's
Python-only environment (checked: no `R`/`Rscript`, no `rpy2`). DMR is the
model STM's own prevalence half is built on and is what tomotopy (a
C++-backed, pip-installable topic-modeling library) actually implements -
the closest genuine approximation available here, at the cost of only the
content-covariate half of full STM.

Represents each basket as a bag-of-words document: each category is a
token, repeated by how many units of it were purchased
(`baskets.build_basket_data(..., include_quantity_matrix=True)`) - genuine
integer counts, unlike the continuous value shares NMF/graph methods use,
because DMR/LDA-family models are built for discrete counts.

`lambdas` (shape k x n_stores after training) is DMR's fitted regression
coefficient of each topic on each store - the per-doc Dirichlet-alpha prior
for store s, topic t is `exp(lambdas[t, s])`; softmax-normalizing a store's
column of `lambdas` across topics gives that store's *typical* topic-
prevalence mix, all else equal - directly the "structural" signal this
model was chosen for."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tomotopy as tp
from scipy.sparse import csr_matrix

from .categories import CategoryVocab
from .config import RANDOM_STATE

MIN_TOKENS_PER_DOC = 1  # baskets with zero survivable tokens are dropped, not added as empty docs


def quantity_matrix_to_docs(quantity_matrix: csr_matrix, vocab: CategoryVocab, store_label: str):
    """Yields (words, metadata) pairs for `tp.DMRModel.add_doc`, one basket
    at a time, from one store's basket x category quantity matrix.

    A generator, not a list-builder, deliberately: on this machine (8GB
    total RAM, shared with everything else running), materializing all
    ~1.26M baskets' word lists into one Python list before handing them to
    `add_doc` roughly doubled peak memory versus adding each basket's words
    straight into the model and letting it be garbage-collected immediately
    after - confirmed via `/usr/bin/time -l` while chasing an OOM kill (no
    traceback, no error - the process just silently disappeared) on the
    first attempt at the full 6-store joint fit. `fit_dmr_missions` chains
    these generators across stores so at most one store's matrix and one
    basket's word list are ever live at once.

    Iterates the matrix's raw CSR arrays (`indptr`/`indices`/`data`)
    directly rather than `matrix.getrow(i)` per basket - `getrow` allocates
    a new sparse-matrix object per call, prohibitively slow at this scale (a
    naive per-row-object loop did not finish a single store within two
    minutes; this is a few seconds per store). Token lists index a plain
    Python list of category-name strings (`names[idx]`) rather than going
    through `numpy_array.tolist()`, which mints a fresh Python str object
    per element on every call - repeated across ~6.9M tokens, that alone
    duplicated the same ~99 distinct category names millions of times over
    and was the single largest source of the memory blowup above. Indexing
    a plain list reuses the same string object each time; list
    multiplication replicates references, not string content.

    Fractional quantities (weighted produce, e.g. 0.3 kg of a category) are
    rounded to the nearest integer with a floor of 1 whenever the raw value
    is positive, so a category's mere presence in a basket is never
    silently dropped just because its rounded weight happened to round to
    zero."""
    qm = quantity_matrix.tocsr()
    names = [vocab.names[cid] for cid in vocab.ids]
    indptr, indices = qm.indptr, qm.indices
    counts = np.maximum(1, np.round(qm.data)).astype(np.int64)

    for i in range(qm.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        words = []
        for idx, cnt in zip(indices[start:end], counts[start:end]):
            words.extend([names[idx]] * int(cnt))
        if len(words) >= MIN_TOKENS_PER_DOC:
            yield words, store_label


@dataclass
class DMRFit:
    model: tp.DMRModel
    n_topics: int
    store_labels: list[str]  # metadata_dict order, aligned to lambdas' columns
    doc_store: np.ndarray  # (n_docs,) which store each doc belongs to, aligned to model.docs
    hard_labels: np.ndarray  # (n_docs,) argmax per-doc topic distribution
    soft_membership: np.ndarray  # (n_docs, n_topics)
    log_likelihood: float


def fit_dmr_missions(
    docs, n_topics: int, iterations: int = 1000,
    step: int = 20, seed: int = RANDOM_STATE, workers: int = 1, verbose: bool = False,
) -> DMRFit:
    """`docs` is any iterable of (words, metadata) pairs - typically
    `itertools.chain` over several `quantity_matrix_to_docs` generators (one
    per store), so the full corpus is never materialized as a Python list;
    see that function's docstring for why this matters on this machine.

    `workers=1` is deliberate, not just the default: tomotopy's own docs
    warn that training is only reproducible across runs with a fixed seed
    when `workers=1` - the reference/reported fit should stay exactly
    reproducible, and `run_stability_test_dmr`'s whole point is measuring
    genuine solution variance, not variance introduced by thread
    scheduling. Trains in `step`-sized chunks so a caller can track
    log-likelihood convergence rather than trusting a single fixed
    iteration count blindly.

    `verbose=True` prints a line after every chunk (iteration count so far,
    elapsed seconds, running perplexity) - added after a real incident on
    this machine: a long unattended fit died with no traceback (consistent
    with an OS-level kill, not a Python exception) somewhere inside a
    single opaque `model.train()` call, and because nothing had printed
    since the fit started, there was no way to tell whether it died after
    1 iteration or after 590. Chunking `step` smaller than `iterations`
    when `verbose=True` is what actually buys the visibility - a `step`
    equal to `iterations` (one giant `model.train()` call) still leaves a
    single blind spot no smaller than before."""
    model = tp.DMRModel(k=n_topics, seed=seed)
    for words, meta in docs:
        model.add_doc(words=words, metadata=meta)
    if verbose:
        print(f"    [fit_dmr_missions] {len(model.docs)} docs added, k={n_topics}, training {iterations} iters in steps of {step}", flush=True)

    t_train = time.time()
    done = 0
    for _ in range(0, iterations, step):
        model.train(step, workers=workers)
        done += step
        if verbose:
            print(f"    [fit_dmr_missions] {min(done, iterations)}/{iterations} iters, perplexity={model.perplexity:.2f} ({time.time()-t_train:.1f}s elapsed)", flush=True)

    if verbose:
        print(f"    [fit_dmr_missions] training done ({time.time()-t_train:.1f}s total), extracting per-doc topic distributions for {len(model.docs)} docs...", flush=True)
    t_extract = time.time()
    n_docs = len(model.docs)
    soft = np.array([d.get_topic_dist() for d in model.docs])
    hard = np.argmax(soft, axis=1)
    doc_store = np.array([d.metadata for d in model.docs])
    if verbose:
        print(f"    [fit_dmr_missions] extraction done ({time.time()-t_extract:.1f}s)", flush=True)

    return DMRFit(
        model=model, n_topics=n_topics, store_labels=list(model.metadata_dict),
        doc_store=doc_store, hard_labels=hard, soft_membership=soft,
        log_likelihood=float(model.ll_per_word) * sum(len(d) for d in model.docs),
    )


def select_n_topics(build_docs_fn, k_range=range(3, 7), iterations: int = 60, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Perplexity-based scree curve across candidate topic counts - like
    `topic_missions.select_n_topics`'s reconstruction-error curve, DMR has
    no gap-statistic equivalent (no null-reference distribution for a topic
    count). `build_docs_fn` must return a *fresh* iterable of (words,
    metadata) on every call (e.g. a closure re-reading the source parquet
    files) rather than a single list handed out repeatedly - re-ingesting
    is cheap (~13s for all 6 stores) next to training cost, and it means at
    most one candidate k's corpus and model are ever live in memory at
    once, not `len(k_range)` of them simultaneously.

    `iterations` defaults far below `fit_dmr_missions`'s own 1000-iteration
    default - measured at ~6s/iteration on the full ~1.26M-document pooled
    corpus on this machine, so this project's usual budget would take
    multiple hours just for the k-sweep. This is a real, disclosed
    trade-off: fewer iterations means a rougher, less-converged perplexity
    estimate per k, adequate for picking an elbow but not for treating any
    single k's perplexity value as fully converged."""
    rows = []
    for k in k_range:
        fit = fit_dmr_missions(build_docs_fn(), n_topics=k, iterations=iterations, step=iterations, seed=seed)
        rows.append({"k": k, "perplexity": float(fit.model.perplexity), "log_likelihood": fit.log_likelihood})
        del fit
    return pd.DataFrame(rows)


def store_topic_prevalence(fit: DMRFit) -> pd.DataFrame:
    """Softmax-normalizes each store's column of `lambdas` across topics -
    the DMR-fitted "typical" topic mix for that store, independent of any
    single basket's assignment. Rows = topics, columns = stores, values sum
    to 1 down each column."""
    lambdas = fit.model.lambdas  # (k, n_stores)
    exp_l = np.exp(lambdas - lambdas.max(axis=0, keepdims=True))
    prevalence = exp_l / exp_l.sum(axis=0, keepdims=True)
    return pd.DataFrame(prevalence, columns=fit.store_labels, index=[f"topic_{t}" for t in range(fit.n_topics)])


def profile_topics(fit: DMRFit, top_n: int = 10) -> list[dict]:
    """Native topic-word distribution straight from the trained model
    (`get_topic_words`), plus each topic's overall hard-assignment share
    across the full pooled corpus and its per-store prevalence breakdown -
    the joint-fit analogue of `topic_missions.profile_topics`."""
    prevalence = store_topic_prevalence(fit)
    n_docs = len(fit.hard_labels)

    profiles = []
    for t in range(fit.n_topics):
        words = fit.model.get_topic_words(t, top_n=top_n)
        top_words = [{"category": w, "weight": round(float(p), 4)} for w, p in words]
        mask = fit.hard_labels == t
        profiles.append({
            "topic_id": t,
            "hard_n_docs": int(mask.sum()),
            "hard_share_pct": round(float(mask.sum()) / n_docs * 100, 2) if n_docs else 0,
            "avg_soft_weight_pct": round(float(fit.soft_membership[:, t].mean()) * 100, 2),
            "top_words": top_words,
            "store_prevalence_pct": {s: round(float(prevalence.loc[f"topic_{t}", s]) * 100, 2) for s in fit.store_labels},
        })
    return profiles
