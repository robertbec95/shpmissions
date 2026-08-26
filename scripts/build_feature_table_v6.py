"""Builds v6: v5's 12 columns unchanged, plus the 9 on-demand
`catchment.py` attributes (per-category POI counts + nearest-competitor
distance) that were computed as part of v5's build but never actually
saved as their own columns - only `log_catchment_population` made it into
v5's table.

`poi_competitor_grocery_count` is dropped: zero variance across all 6
stores (confirmed - every store shows 0), same reasoning the original
per-store PCA pipeline used to drop the equivalent raw column back when
it built PC1/PC2. A zero-variance column contributes nothing to Gower
distance except a wasted weight slot.

Two new feature groups, kept SEPARATE from `poi` (PC1/PC2, still from the
old live-Overpass pull, unchanged) rather than merged into it - PC1/PC2
already summarize the old POI signal, and folding 8 more raw counts
straight into that same group would double-count the same underlying
"how POI-dense is this catchment" factor. Keeping them as distinct
groups lets the v6 grid search (see `run_location_similarity_v6_validation.py`)
find out empirically whether the newer, more granular OSM-extract counts
add real information beyond what PC1/PC2 already capture, or are mostly
redundant with it - rather than assuming either answer up front:
    - "poi_detail": the 8 non-zero-variance POI category counts
    - "competition": `nearest_competitor_km` alone (a continuous distance,
      a different kind of measurement than a count)

Reproduce with: `PYTHONPATH=src python3 scripts/build_feature_table_v6.py`
"""

import json

import pandas as pd

from shopping_missions.catchment import catchment_profile
from shopping_missions.config import CATCHMENT_RADIUS_KM

V5_FILE = "data/location_features/store_feature_table_v5.json"
GEOCODES_FILE = "data/location_features/store_geocodes.json"
POI_CSV = "data/location_features/osm/romania_poi_nodes.csv"
POP_RASTER = "data/location_features/worldpop/rou_pop_2025_CN_100m_R2025A_v1.tif"
OUT_FILE = "data/location_features/store_feature_table_v6.json"

DROP_ZERO_VARIANCE = ["poi_competitor_grocery_count"]


def main():
    v5 = json.load(open(V5_FILE))
    geocodes = {g["loc_id"]: g for g in json.load(open(GEOCODES_FILE))}
    poi_df = pd.read_csv(POI_CSV)

    v6 = {}
    for loc_id, row in v5.items():
        geo = geocodes[loc_id]
        profile = catchment_profile(geo["lat"], geo["lon"], radius_km=CATCHMENT_RADIUS_KM, raster_path=POP_RASTER, poi_df=poi_df)

        new_cols = {k: v for k, v in profile.attributes.items() if k.startswith("poi_") or k == "nearest_competitor_km"}
        for col in DROP_ZERO_VARIANCE:
            new_cols.pop(col, None)

        v6[loc_id] = {**row, **new_cols}

    with open(OUT_FILE, "w") as f:
        json.dump(v6, f, indent=2)
    print(f"Wrote {OUT_FILE} ({len(next(iter(v6.values())))} columns)")
    for loc_id, row in v6.items():
        print(f"  {loc_id}: nearest_competitor={row['nearest_competitor_km']:.3f}km, "
              f"schools={row['poi_school_count']}, transit={row['poi_transit_stop_count']}")


if __name__ == "__main__":
    main()
