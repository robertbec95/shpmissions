"""Run one or more named scenarios (src/shopping_missions/scenarios.py) end
to end and track results across runs.

Each scenario run:
  1. Ingests at the scenario's category_level (3/4/5), with day-part carried
     through if requested.
  2. Builds the basket x category matrix, filters to variety>=min_variety
     (settled at 2 - see scenarios.py docstring).
  3. Runs Stage 2 (Jaccard k-medoids) clustering.
  4. Profiles clusters (Table-6-style top categories per cluster) and flags
     "degenerate" clusters - ones a single category defines almost by itself
     (>=90% support), the failure mode confirmed in
     docs/table6-comparison-all-baskets.md when single-category baskets were
     wrongly included.
  5. If requested, cross-tabs cluster membership against county and/or
     day-part - validation only, these never feed the clustering itself.
  6. Optionally runs the gap statistic/silhouette/elbow k-selection diagnostic.
  7. Writes a detailed report to docs/scenarios/<name>.md and appends one
     summary row to outputs/scenarios/scenario_results.csv (created if
     missing, appended to otherwise) so results accumulate across separate
     invocations instead of being overwritten.

Usage:
    PYTHONPATH=src python3 scripts/run_scenarios.py                  # run all scenarios
    PYTHONPATH=src python3 scripts/run_scenarios.py level3_baseline level4
"""

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shopping_missions import multiday, mission_stage, store_formats, temporal
from shopping_missions.cluster_selection import best_k_by_gap, elbow_k_from_gap_df
from shopping_missions.config import OUTPUTS_DIR, PROJECT_ROOT, RANDOM_STATE, SCENARIOS_DIR
from shopping_missions.scenarios import SCENARIOS, Scenario

DOCS_SCENARIOS_DIR = PROJECT_ROOT / "docs" / "scenarios"
RESULTS_CSV = SCENARIOS_DIR / "scenario_results.csv"
DEGENERATE_SUPPORT_THRESHOLD = 0.90

LABEL_KEYWORDS = {
    "breakfast": {"BREAKFAST", "CEREALS", "BUTTER", "EGGS", "MILK", "BAKE OFF", "SUPPLIED BAKERY"},
    "coffee": {"COFFEE", "SUGAR", "CHOCOLATE"},
    "meal": {"MEAT", "CHICKEN", "PORK", "FISH", "VEGETABLES", "RICE", "PASTA", "GARNISHES", "COOKED MEALS", "SAUCES"},
    "light_meal": {"CHEESE", "COLD CUTS", "PREPARED MEAT", "SALADS", "BAKE OFF", "PACKED SAUSAGES"},
    "snack": {"SALTY SNACKS", "SWEET SNACKS", "CHEWING GUM", "CHOCOLATE", "CARBONATED DRINKS", "ICE CREAM"},
    "cleaning": {"DETERGENTS", "BLEACHES", "HOME CARE", "HOUSHOLD PAPER", "HOUSEHOLD PRODUCTS"},
}


def suggest_label(top_categories):
    names = {c["category"].upper() for c in top_categories}
    best_label, best_score = "unlabeled", 0.0
    for label, keywords in LABEL_KEYWORDS.items():
        hits = sum(1 for name in names if any(kw in name for kw in keywords))
        score = hits / max(len(top_categories), 1)
        if score > best_score:
            best_label, best_score = label, score
    return best_label, round(best_score, 2)


def run_scenario(scenario: Scenario) -> dict:
    print(f"\n=== Scenario: {scenario.name} ===")
    print(scenario)
    t0 = time.time()

    ingest_kwargs = {}
    if scenario.exclude_category_ids is not None:
        ingest_kwargs["exclude_category_ids"] = scenario.exclude_category_ids
    data = multiday.build_basket_data_multi_date(
        list(scenario.dates), category_level=scenario.category_level, include_daypart=scenario.include_daypart,
        **ingest_kwargs,
    )
    vocab = data.vocab
    print(f"  ingested: {len(data.baskets):,} baskets, {vocab.size} categories seen ({time.time()-t0:.1f}s)")

    keep_mask = (data.baskets["variety"] >= scenario.min_variety).to_numpy()
    baskets = data.baskets.loc[keep_mask].reset_index(drop=True)
    matrix = data.matrix[keep_mask]
    n = len(baskets)
    print(f"  after variety>={scenario.min_variety} filter: {n:,} baskets")

    clustering = mission_stage.fit_and_assign_missions(
        matrix, k=scenario.k, sample_size=scenario.sample_size, random_state=RANDOM_STATE
    )
    labels = clustering.labels
    assert len(labels) == n, "label/basket alignment broke"

    if scenario.include_county:
        baskets = store_formats.join_county(baskets)
        assert len(baskets) == n, "county join changed row count - alignment broke"
    if scenario.include_daypart:
        baskets = temporal.add_daypart(baskets)
        assert len(baskets) == n, "day-part derive changed row count - alignment broke"

    profiles = _profile_clusters(baskets, matrix, labels, vocab, scenario.k)
    n_degenerate = sum(1 for p in profiles if p["is_degenerate"])
    for p in profiles:
        top = ", ".join(c["category"] for c in p["top_categories"][:4])
        flag = " [DEGENERATE]" if p["is_degenerate"] else ""
        print(f"  cluster {p['cluster_id']}: {p['suggested_label']} n={p['n_baskets']:,} "
              f"({p['share_pct']}%) top=[{top}]{flag}")

    gap_k = sil_k = elbow_k = mean_sil = None
    if scenario.run_k_selection:
        gap_k, sil_k, elbow_k, mean_sil = _run_k_selection(matrix, scenario)
        print(f"  k-selection: gap_k={gap_k} sil_k={sil_k} elbow_k={elbow_k} silhouette@k={scenario.k}={mean_sil}")

    county_xtab = _crosstab(baskets, labels, "county", profiles) if scenario.include_county else None
    daypart_xtab = _crosstab(baskets, labels, "day_part", profiles) if scenario.include_daypart else None

    elapsed = time.time() - t0
    _write_report(scenario, n, vocab.size, profiles, gap_k, sil_k, elbow_k, mean_sil, county_xtab, daypart_xtab, elapsed)

    return {
        "scenario_name": scenario.name,
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category_level": scenario.category_level,
        "n_categories_seen": vocab.size,
        "min_variety": scenario.min_variety,
        "k": scenario.k,
        "n_baskets": n,
        "n_degenerate_clusters": n_degenerate,
        "gap_k": gap_k,
        "silhouette_k": sil_k,
        "elbow_k": elbow_k,
        "silhouette_at_k": mean_sil,
        "elapsed_sec": round(elapsed, 1),
        "notes": scenario.notes,
    }


def _profile_clusters(baskets, matrix, labels, vocab, k):
    n = len(labels)
    overall_support = np.asarray(matrix.sum(axis=0)).ravel() / n

    profiles = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        nc = int(mask.sum())
        sub = matrix[mask]
        cluster_support = np.asarray(sub.sum(axis=0)).ravel() / nc if nc else np.zeros(vocab.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_support > 0, cluster_support / overall_support, 0.0)

        candidates = [
            {"category": vocab.names[vocab.ids[i]], "support": round(float(cluster_support[i]), 4),
             "lift": round(float(lift[i]), 3)}
            for i in range(vocab.size)
            if cluster_support[i] >= 0.15 and lift[i] >= 1.2
        ]
        candidates.sort(key=lambda c: c["lift"], reverse=True)
        top_categories = candidates[:6]
        label, confidence = suggest_label(top_categories)
        is_degenerate = bool(len(candidates) and candidates[0]["support"] >= DEGENERATE_SUPPORT_THRESHOLD)

        profiles.append({
            "cluster_id": cluster_id,
            "n_baskets": nc,
            "share_pct": round(nc / n * 100, 2) if n else 0,
            "avg_basket_value": round(float(baskets.loc[mask, "basket_value"].mean()), 2) if nc else None,
            "avg_basket_variety": round(float(baskets.loc[mask, "variety"].mean()), 2) if nc else None,
            "top_categories": top_categories,
            "suggested_label": label,
            "label_confidence": confidence,
            "is_degenerate": is_degenerate,
        })
    return profiles


def _run_k_selection(matrix, scenario, k_range=range(2, 12), sample_size=1500, B=10):
    rng = np.random.default_rng(RANDOM_STATE)
    n = matrix.shape[0]
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_sample = np.asarray(matrix[idx].todense(), dtype=np.float64)

    gdf = mission_stage.select_k(X_sample, k_range=k_range, B=B, random_state=RANDOM_STATE)
    gap_k = best_k_by_gap(gdf)
    sil_k = int(gdf.loc[gdf["silhouette"].idxmax(), "k"])
    elbow_k = elbow_k_from_gap_df(gdf)
    row_at_k = gdf.loc[gdf["k"] == scenario.k]
    mean_sil = round(float(row_at_k["silhouette"].iloc[0]), 4) if len(row_at_k) and row_at_k["silhouette"].notna().all() else None
    return gap_k, sil_k, elbow_k, mean_sil


def _crosstab(baskets, labels, dim_col, profiles):
    df = baskets.copy()
    df["cluster"] = labels
    df["cluster_label"] = df["cluster"].map({p["cluster_id"]: p["suggested_label"] for p in profiles})
    xtab = df.groupby([dim_col, "cluster"], observed=True).size().rename("n").reset_index()
    xtab["share_within_dim"] = xtab.groupby(dim_col, observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return xtab


def _write_report(scenario, n, n_categories_seen, profiles, gap_k, sil_k, elbow_k, mean_sil, county_xtab, daypart_xtab, elapsed):
    lines = []
    lines.append(f"# Scenario: {scenario.name}")
    lines.append("")
    lines.append(scenario.notes)
    lines.append("")
    lines.append(
        f"**Config:** category_level={scenario.category_level} | dates={','.join(scenario.dates)} | "
        f"min_variety={scenario.min_variety} | k={scenario.k} | sample_size={scenario.sample_size} | "
        f"county_crosstab={scenario.include_county} | daypart_crosstab={scenario.include_daypart}"
    )
    lines.append("")
    lines.append(f"**Result:** {n:,} baskets, {n_categories_seen} categories seen, run in {elapsed:.1f}s")
    if gap_k is not None:
        lines.append(
            f". k-selection at this population: gap_k={gap_k}, silhouette_k={sil_k}, elbow_k={elbow_k}, "
            f"silhouette@k={scenario.k}={mean_sil}"
        )
    lines.append("")

    lines.append("## Cluster profiles")
    lines.append("")
    lines.append("| Cluster | Label | Share | Avg value | Avg variety | Top categories | Degenerate? |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in profiles:
        top = ", ".join(f"{c['category']} ({c['support']*100:.0f}%)" for c in p["top_categories"][:4])
        flag = "**YES**" if p["is_degenerate"] else "no"
        lines.append(
            f"| {p['cluster_id']} | {p['suggested_label']} (conf {p['label_confidence']}) | {p['share_pct']}% | "
            f"{p['avg_basket_value']} | {p['avg_basket_variety']} | {top} | {flag} |"
        )
    lines.append("")

    if county_xtab is not None:
        lines.append("## Cluster mix by county (top 10 counties by basket volume)")
        lines.append("")
        top_counties = county_xtab.groupby("county", observed=True)["n"].sum().sort_values(ascending=False).head(10).index
        pivot = county_xtab[county_xtab["county"].isin(top_counties)].pivot(index="county", columns="cluster", values="share_within_dim").fillna(0)
        pivot = pivot.loc[top_counties]
        cluster_labels = {p["cluster_id"]: p["suggested_label"] for p in profiles}
        pivot.columns = [f"C{c} ({cluster_labels.get(c,'?')})" for c in pivot.columns]
        lines.append(pivot.round(1).to_markdown())
        lines.append("")

    if daypart_xtab is not None:
        lines.append("## Cluster mix by day-part")
        lines.append("")
        order = ["night", "early_morning", "morning", "afternoon", "evening", "late_evening", "Unknown"]
        pivot = daypart_xtab.pivot(index="day_part", columns="cluster", values="share_within_dim").fillna(0)
        pivot = pivot.reindex([o for o in order if o in pivot.index])
        cluster_labels = {p["cluster_id"]: p["suggested_label"] for p in profiles}
        pivot.columns = [f"C{c} ({cluster_labels.get(c,'?')})" for c in pivot.columns]
        lines.append(pivot.round(1).to_markdown())
        lines.append(
            "\n*Read this as: within each day-part, what % of baskets fall in each cluster - a genuine "
            "\"breakfast\" cluster should show a bump in morning/early_morning here.*"
        )
        lines.append("")

    lines.append(f"Reproduce with: `PYTHONPATH=src python3 scripts/run_scenarios.py {scenario.name}`")

    out_path = DOCS_SCENARIOS_DIR / f"{scenario.name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def _append_results_csv(rows: list[dict]):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_CSV.exists()
    fieldnames = list(rows[0].keys())
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nAppended {len(rows)} row(s) to {RESULTS_CSV}")


def main():
    requested = sys.argv[1:]
    if requested:
        to_run = [get_scenario_or_die(name) for name in requested]
    else:
        to_run = SCENARIOS

    results = []
    for scenario in to_run:
        results.append(run_scenario(scenario))

    _append_results_csv(results)


def get_scenario_or_die(name):
    from shopping_missions.scenarios import get_scenario
    return get_scenario(name)


if __name__ == "__main__":
    main()
