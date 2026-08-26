"""Closes the "by tier/banner" half of the competitor-count gap: the
existing pipeline only ever counted competitors by TIER (supermarket/
convenience/grocery, from the OSM shop=* tag). This groups the same
already-extracted `name` field by actual retail BANNER instead, for the
6 known stores' 1km catchments - no new data pull, just a different cut
of data already in hand.

Banner patterns were chosen from the top 20 most frequent competitor
names found nationally in `romania_poi_nodes.csv` (Profi's own banners
excluded via the same `OWN_BANNER_NAME_PATTERN` used for
`nearest_own_banner_km`). Generic Romanian terms for independent shops
("Magazin Mixt", "Mini Market", "Magazin alimentar" etc.) are NOT a
banner and are bucketed as "Independent/Other" rather than force-fit into
a fake chain.

Reproduce with: `PYTHONPATH=src python3 scripts/build_competitor_banner_breakdown.py`
"""

import json

import pandas as pd

from shopping_missions.catchment import COMPETITOR_CATEGORIES, OWN_BANNER_NAME_PATTERN, haversine_km

POI_CSV = "data/location_features/osm/romania_poi_nodes.csv"
GEOCODES_FILE = "data/location_features/store_geocodes.json"
OUT_FILE = "data/location_features/competitor_banner_breakdown.json"

BANNER_PATTERNS = {
    "Mega Image": r"mega image",
    "Carrefour": r"carrefour",
    "Penny": r"penny",
    "Lidl": r"lidl",
    "La Doi Pasi": r"la doi pa[sș]i",
    "Annabella": r"annabella",
    "Froo": r"froo",
    "Kaufland": r"kaufland",
    "Auchan": r"auchan",
    "Cora": r"\bcora\b",
}


def classify_banner(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unnamed"
    for banner, pattern in BANNER_PATTERNS.items():
        if pd.Series([name]).str.contains(pattern, case=False, regex=True).iloc[0]:
            return banner
    return "Independent/Other"


def main():
    poi_df = pd.read_csv(POI_CSV)
    comp = poi_df[poi_df["category"].isin(COMPETITOR_CATEGORIES)].copy()
    comp = comp.loc[~comp["name"].astype(str).str.contains(OWN_BANNER_NAME_PATTERN, case=False, regex=True, na=False)]
    comp["banner"] = comp["name"].apply(classify_banner)

    stores = json.load(open(GEOCODES_FILE))
    results = {}
    for s in stores:
        dist = haversine_km(s["lat"], s["lon"], comp["lat"].values, comp["lon"].values)
        within = comp.loc[dist <= 1.0]
        counts = within["banner"].value_counts().to_dict()
        results[s["loc_id"]] = {"store": s["raw_name"], "city": s["city"], "banner_counts": counts}

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_FILE}")
    for loc_id, r in results.items():
        print(f"  {r['store']} ({r['city']}): {r['banner_counts']}")


if __name__ == "__main__":
    main()
