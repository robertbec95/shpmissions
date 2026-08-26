"""Orchestrates Steps A-H end to end. CLI entry point.

Usage:
    python -m shopping_missions.pipeline --dates 20260509
    python -m shopping_missions.pipeline --dates 20260509,20260510,20260511 --k 6
"""

import argparse
import dataclasses
import json
from pathlib import Path

from . import ingest, categories, baskets as baskets_mod, variety_stage, mission_stage, profiling, validation
from .config import DEFAULT_K, DEFAULT_SAMPLE_SIZE, OUTPUTS_DIR, RANDOM_STATE


def run(dates: list[str], k: int = DEFAULT_K, sample_size: int = DEFAULT_SAMPLE_SIZE, output_dir: Path = OUTPUTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[A] Ingesting {dates} (offline channel, noise categories excluded)...")
    con = ingest.connect_lines(dates)

    print("[B] Building category vocabulary...")
    vocab = categories.build_vocab(con)
    print(f"    {vocab.size} categories in vocabulary")

    print("[C] Building basket table + basket x category matrix...")
    data = baskets_mod.build_basket_data(con, vocab)
    print(f"    {len(data.baskets)} baskets")

    print("[D] Stage 1 - variety segmentation...")
    seg = variety_stage.fit_variety_segments(data.baskets)
    print(seg.summary)
    print(f"    bounds: {seg.bounds}")

    rel_mask = (seg.baskets["segment"] == "relatively_concrete").to_numpy()
    rel_baskets = seg.baskets.loc[rel_mask].reset_index(drop=True)
    rel_matrix = data.matrix[rel_mask]
    print(f"    {len(rel_baskets)} relatively-concrete baskets carried into Stage 2")

    print(f"[E] Stage 2 - Jaccard k-medoids (k={k}, sample_size={sample_size})...")
    clustering = mission_stage.fit_and_assign_missions(
        rel_matrix, k=k, sample_size=sample_size, random_state=RANDOM_STATE
    )

    print("[F] Profiling & labeling mission clusters...")
    profiles = profiling.profile_clusters(rel_matrix, clustering.labels, vocab, rel_baskets)
    for p in profiles:
        top = ", ".join(c["category"] for c in p.top_categories[:4])
        print(f"    cluster {p.cluster_id}: {p.suggested_label} (conf={p.label_confidence}) "
              f"n={p.n_baskets} ({p.share_of_relatively_concrete}%) top=[{top}]")
    bridges = profiling.find_bridge_categories(profiles)
    print(f"    {len(bridges)} bridge categories found")

    print("[G] Validating by store-size tier...")
    store_tiers = validation.compute_store_size_tiers(data.baskets)
    segment_mix = validation.segment_mix_by_store_tier(seg.baskets, store_tiers)
    mission_mix = validation.mission_mix_by_store_tier(rel_baskets, clustering.labels, store_tiers)

    print("[H] Writing outputs...")
    _write_outputs(output_dir, rel_baskets, clustering, profiles, bridges, seg, segment_mix, mission_mix, dates, k, sample_size)
    print(f"Done. Outputs in {output_dir}")


def _write_outputs(output_dir, rel_baskets, clustering, profiles, bridges, seg, segment_mix, mission_mix, dates, k, sample_size):
    basket_missions = rel_baskets[["basket_id", "store_id", "variety", "basket_value"]].copy()
    basket_missions["segment"] = "relatively_concrete"
    basket_missions["mission_cluster"] = clustering.labels
    basket_missions["mission_label"] = [profiles[c].suggested_label for c in clustering.labels]
    basket_missions["distance_to_medoid"] = clustering.distances
    basket_missions.to_parquet(output_dir / "basket_missions.parquet", index=False)

    with open(output_dir / "mission_profiles.json", "w") as f:
        json.dump([dataclasses.asdict(p) for p in profiles], f, indent=2)

    with open(output_dir / "category_bridges.json", "w") as f:
        json.dump(bridges, f, indent=2)

    segment_mix.to_csv(output_dir / "segment_mix_by_store_tier.csv", index=False)
    mission_mix.to_csv(output_dir / "mission_mix_by_store_tier.csv", index=False)

    summary_lines = [
        f"# Shopping Missions MVP - run summary",
        "",
        f"Dates: {', '.join(dates)}  |  k={k}  |  sample_size={sample_size}",
        "",
        "## Stage 1 - Variety segments",
        "",
        seg.summary.to_markdown(),
        "",
        f"Bounds: `{seg.bounds}`",
        "",
        "## Stage 2 - Mission clusters",
        "",
    ]
    for p in profiles:
        summary_lines.append(
            f"- **Cluster {p.cluster_id} ({p.suggested_label}, confidence={p.label_confidence})** - "
            f"{p.n_baskets} baskets ({p.share_of_relatively_concrete}% of relatively-concrete), "
            f"avg value {p.avg_basket_value}"
        )
        summary_lines.append(
            "  top categories: " + ", ".join(f"{c['category']} (lift={c['lift']})" for c in p.top_categories)
        )
    summary_lines += ["", "## Bridge categories", ""]
    for b in bridges[:15]:
        clusters = ", ".join(f"{e['label']}#{e['cluster_id']}" for e in b["clusters"])
        summary_lines.append(f"- **{b['category']}**: present in {clusters}")

    (output_dir / "summary.md").write_text("\n".join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description="Run the shopping-missions MVP pipeline")
    parser.add_argument("--dates", required=True, help="Comma-separated YYYYMMDD dates, e.g. 20260509,20260510")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",")]
    run(dates, k=args.k, sample_size=args.sample_size, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
