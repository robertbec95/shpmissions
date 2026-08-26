"""Builds v5 of the store feature table: v4's 9 columns UNCHANGED, plus 3
genuinely new columns from this session's national real-data pulls:
    - log_catchment_population: WorldPop-based population within 1km of
      the store (via `catchment.population_in_buffer`), log-transformed
      for the same reason `log_population` already is (population spans
      orders of magnitude) - NOT a replacement for `log_population` (that
      stays as the whole-city figure/market-size context), a genuinely
      different signal (immediate trade-area density).
    - unemployment_share_pct: INS TEMPO-Online SOM101F, per city.
    - gdp_per_capita_eur: Eurostat nama_10r_3gdp, per county, latest
      year (2024). Kept unlogged - unlike population, this only spans
      about one order of magnitude across these 5 counties.

Deliberately did NOT recompute PC1/PC2 from the new national OSM extract.
The extract undercounts schools/kindergartens/places of worship (node-tag
scope limitation - see `docs/.../catchment` validation), so replacing the
already-validated, already-weight-tuned live-Overpass-based PC1/PC2 with a
partially-degraded new source would very plausibly make the POI signal
WORSE, not better. The new national OSM data justified its own place in
this project (a national-coverage capability the old per-store pulls
didn't have) without needing to overwrite what already works for the 6
known stores specifically.

Reproduce with: `PYTHONPATH=src python3 scripts/build_feature_table_v5.py`
"""

import json

import numpy as np
import pandas as pd

from shopping_missions.catchment import catchment_profile
from shopping_missions.config import CATCHMENT_RADIUS_KM

V4_FILE = "data/location_features/store_feature_table_v3.json"
GEOCODES_FILE = "data/location_features/store_geocodes.json"
UNEMPLOYMENT_FILE = "data/location_features/ins_unemployment_by_city.json"
GDP_FILE = "data/location_features/eurostat_gdp_per_capita.json"
POI_CSV = "data/location_features/osm/romania_poi_nodes.csv"
POP_RASTER = "data/location_features/worldpop/rou_pop_2025_CN_100m_R2025A_v1.tif"
OUT_FILE = "data/location_features/store_feature_table_v5.json"


def main():
    v4 = json.load(open(V4_FILE))
    geocodes = {g["loc_id"]: g for g in json.load(open(GEOCODES_FILE))}
    unemployment = {u["city"]: u["unemployment_share_pct"] for u in json.load(open(UNEMPLOYMENT_FILE))}
    gdp = {g["county"]: g for g in json.load(open(GDP_FILE))}
    poi_df = pd.read_csv(POI_CSV)

    city_label_map = {  # geocodes uses "Piatra Neamt"/"Craiova", unemployment file uses INS's official city labels
        "Bucharest": "Bucuresti Municipality", "Piatra Neamt": "Piatra-Neamt Municipality",
        "Calarasi": "Calarasi Municipality", "Craiova": "Craiova Municipality", "Brasov": "Brasov Municipality",
    }

    v5 = {}
    for loc_id, row in v4.items():
        geo = geocodes[loc_id]
        profile = catchment_profile(geo["lat"], geo["lon"], radius_km=CATCHMENT_RADIUS_KM, raster_path=POP_RASTER, poi_df=poi_df)

        unemployment_key = city_label_map[geo["city"]]
        gdp_row = gdp[geo["county"]]
        latest_year = max(gdp_row["gdp_per_capita_eur_by_year"])

        v5[loc_id] = {
            **row,
            "log_catchment_population": float(np.log(profile.attributes["catchment_population"] + 1)),
            "unemployment_share_pct": unemployment[unemployment_key],
            "gdp_per_capita_eur": gdp_row["gdp_per_capita_eur_by_year"][latest_year],
        }

    with open(OUT_FILE, "w") as f:
        json.dump(v5, f, indent=2)
    print(f"Wrote {OUT_FILE}")
    for loc_id, row in v5.items():
        print(f"  {loc_id}: catchment_pop_1km(log)={row['log_catchment_population']:.2f}, "
              f"unemployment={row['unemployment_share_pct']}%, gdp={row['gdp_per_capita_eur']:,}")


if __name__ == "__main__":
    main()
