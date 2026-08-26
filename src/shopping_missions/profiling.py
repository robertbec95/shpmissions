"""Step F - Profile and label mission clusters."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .categories import CategoryVocab

MIN_SUPPORT = 0.15  # a category must appear in >=15% of a cluster's baskets to count toward its profile
MIN_LIFT = 1.2  # and be at least 20% more common in-cluster than baseline, or it's just a ubiquitous item (e.g. produce bags), not mission-defining
TOP_N_CATEGORIES = 6
BRIDGE_MIN_SUPPORT = 0.20
BRIDGE_MIN_CLUSTERS = 2

# Keyword heuristics for auto-suggesting a human-readable mission label from
# its top categories. Mirrors the paper's six recurring missions; always
# reviewed by eye (see mission_profiles.json `label_confidence`) rather than
# trusted blindly.
LABEL_KEYWORDS = {
    "breakfast": {"BREAKFAST", "CEREALS", "BUTTER", "EGGS", "MILK", "BAKE OFF", "SUPPLIED BAKERY"},
    "coffee": {"COFFEE", "SUGAR", "CHOCOLATE"},
    "meal": {"MEAT", "CHICKEN", "PORK", "FISH", "VEGETABLES", "RICE", "PASTA", "GARNISHES", "COOKED MEALS", "SAUCES"},
    "light_meal": {"CHEESE", "COLD CUTS", "PREPARED MEAT", "SALADS", "BAKE OFF", "PACKED SAUSAGES"},
    "snack": {"SALTY SNACKS", "SWEET SNACKS", "CHEWING GUM", "CHOCOLATE", "CARBONATED DRINKS", "ICE CREAM"},
    "cleaning": {"DETERGENTS", "BLEACHES", "HOME CARE", "HOUSHOLD PAPER", "HOUSEHOLD PRODUCTS"},
}


@dataclass
class MissionProfile:
    cluster_id: int
    n_baskets: int
    share_of_relatively_concrete: float
    avg_basket_value: float
    top_categories: list[dict] = field(default_factory=list)  # [{category, support, lift}]
    suggested_label: str = "unlabeled"
    label_confidence: float = 0.0


def profile_clusters(
    matrix: csr_matrix,
    labels: np.ndarray,
    vocab: CategoryVocab,
    baskets: pd.DataFrame,
) -> list[MissionProfile]:
    """`matrix`/`baskets` must be the relatively-concrete subset, row-aligned
    with `labels` (the mission cluster assignment from Step E)."""
    dense = np.asarray(matrix.todense())
    overall_support = dense.mean(axis=0)  # baseline presence rate per category, across all rel-concrete baskets

    profiles = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        n_baskets = int(mask.sum())
        cluster_support = dense[mask].mean(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(overall_support > 0, cluster_support / overall_support, 0.0)

        candidates = [
            {
                "category": vocab.names[vocab.ids[i]],
                "category_id": vocab.ids[i],
                "support": round(float(cluster_support[i]), 4),
                "lift": round(float(lift[i]), 3),
            }
            for i in range(vocab.size)
            if cluster_support[i] >= MIN_SUPPORT and lift[i] >= MIN_LIFT
        ]
        candidates.sort(key=lambda c: c["lift"], reverse=True)
        top_categories = candidates[:TOP_N_CATEGORIES]

        label, confidence = _suggest_label(top_categories)

        profiles.append(
            MissionProfile(
                cluster_id=int(cluster_id),
                n_baskets=n_baskets,
                share_of_relatively_concrete=round(n_baskets / len(labels) * 100, 2),
                avg_basket_value=round(float(baskets.loc[mask, "basket_value"].mean()), 2),
                top_categories=top_categories,
                suggested_label=label,
                label_confidence=confidence,
            )
        )
    return profiles


def _suggest_label(top_categories: list[dict]) -> tuple[str, float]:
    names = {c["category"].upper() for c in top_categories}
    best_label, best_score = "unlabeled", 0.0
    for label, keywords in LABEL_KEYWORDS.items():
        hits = sum(1 for name in names if any(kw in name for kw in keywords))
        score = hits / max(len(top_categories), 1)
        if score > best_score:
            best_label, best_score = label, score
    return best_label, round(best_score, 2)


def find_bridge_categories(profiles: list[MissionProfile]) -> list[dict]:
    """Categories with meaningful presence (>=BRIDGE_MIN_SUPPORT) across
    multiple missions - candidates for cross-sell / bridging analysis."""
    presence: dict[str, list[dict]] = {}
    for profile in profiles:
        for cat in profile.top_categories:
            if cat["support"] >= BRIDGE_MIN_SUPPORT:
                presence.setdefault(cat["category"], []).append(
                    {"cluster_id": profile.cluster_id, "label": profile.suggested_label, "support": cat["support"]}
                )

    bridges = [
        {"category": category, "clusters": entries}
        for category, entries in presence.items()
        if len(entries) >= BRIDGE_MIN_CLUSTERS
    ]
    bridges.sort(key=lambda b: len(b["clusters"]), reverse=True)
    return bridges
