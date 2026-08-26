"""Step G - Validate segment/mission results by store, using a store-size
proxy in place of the paper's convenience/supermarket/hypermarket format
label, which is not present in this dataset (see docs/mvp-plan.md, section 2).
"""

import pandas as pd

STORE_SIZE_TIERS = ["small", "medium", "large"]


def compute_store_size_tiers(all_baskets: pd.DataFrame) -> pd.DataFrame:
    """Bucket stores into size tertiles by daily basket (transaction) volume,
    as a stand-in for store format."""
    store_volume = all_baskets.groupby("store_id").size().rename("n_baskets").reset_index()
    store_volume["size_tier"] = pd.qcut(
        store_volume["n_baskets"], q=3, labels=STORE_SIZE_TIERS
    )
    return store_volume


def segment_mix_by_store_tier(baskets_with_segment: pd.DataFrame, store_tiers: pd.DataFrame) -> pd.DataFrame:
    merged = baskets_with_segment.merge(store_tiers[["store_id", "size_tier"]], on="store_id", how="left")
    mix = (
        merged.groupby(["size_tier", "segment"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    mix["share_within_tier"] = mix.groupby("size_tier", observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return mix


def mission_mix_by_store_tier(
    rel_concrete_baskets: pd.DataFrame, mission_labels, store_tiers: pd.DataFrame
) -> pd.DataFrame:
    """`rel_concrete_baskets` and `mission_labels` must be row-aligned (same
    order/length as produced by mission_stage.fit_and_assign_missions)."""
    df = rel_concrete_baskets.copy()
    df["mission_cluster"] = mission_labels
    df = df.merge(store_tiers[["store_id", "size_tier"]], on="store_id", how="left")

    mix = df.groupby(["size_tier", "mission_cluster"], observed=True).size().rename("n").reset_index()
    mix["share_within_tier"] = mix.groupby("size_tier", observed=True)["n"].transform(lambda s: s / s.sum() * 100)
    return mix
