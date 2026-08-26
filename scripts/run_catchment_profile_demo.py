"""Runs the new `catchment.catchment_profile` against the 6 known stores'
real geocoded coordinates, at radius_km=1.0, and prints it next to the
ORIGINAL per-store Overpass POI counts (`store_poi_counts.json`, also
pulled at 1km) as a validation check - if the new national-extract-based
counts land close to the original per-store live-API counts, that
confirms the OSM extraction pipeline is correct, not just plausible.

Also prints each store's new WorldPop-based catchment population
(1km-radius circle) next to the OLD whole-city INS population figure it
was matched to before, to make concrete exactly how much more precise
the new pipeline is - a 1km circle around a single store is a very
different number from "the entire city's population."

Reproduce with: `PYTHONPATH=src python3 scripts/run_catchment_profile_demo.py`
"""

import json

import pandas as pd

from shopping_missions.catchment import catchment_profile
from shopping_missions.config import CATCHMENT_RADIUS_KM

STORES_FILE = "data/location_features/store_poi_counts.json"
POI_CSV = "data/location_features/osm/romania_poi_nodes.csv"
POP_RASTER = "data/location_features/worldpop/rou_pop_2025_CN_100m_R2025A_v1.tif"

OLD_CITY_POPULATION = {  # from ins_population_by_city.json, for the population comparison
    "Bucharest": 2_143_578, "Piatra Neamt": 106_814, "Calarasi": 70_861,
    "Craiova": 285_722, "Brasov": 283_848,
}

OLD_TO_NEW_CATEGORY = {
    "schools": "poi_school_count",
    "kindergartens": "poi_kindergarten_count",
    "places_of_worship": "poi_place_of_worship_count",
    "competitor_supermarket": "poi_competitor_supermarket_count",
    "competitor_convenience": "poi_competitor_convenience_count",
    "competitor_grocery": "poi_competitor_grocery_count",
    "pharmacies": "poi_pharmacy_count",
    "banks": "poi_bank_count",
    "public_transport_stops": "poi_transit_stop_count",
}


def main():
    stores = json.load(open(STORES_FILE))
    poi_df = pd.read_csv(POI_CSV)

    for store in stores:
        profile = catchment_profile(store["lat"], store["lon"], radius_km=CATCHMENT_RADIUS_KM, raster_path=POP_RASTER, poi_df=poi_df)
        print(f"\n=== {store['raw_name']} ({store['city']}) ===")
        print(f"  New catchment population (1km radius): {profile.attributes['catchment_population']:,.0f}")
        print(f"  Old whole-city population (INS):        {OLD_CITY_POPULATION.get(store['city'], 0):,}")
        print(f"  Nearest competitor: {profile.attributes['nearest_competitor_km']:.3f}km ({profile.attributes['nearest_competitor_name']})")
        print(f"  {'category':28s} {'old(live API)':>14s} {'new(extract)':>14s}")
        for old_key, new_key in OLD_TO_NEW_CATEGORY.items():
            old_val = store["poi_counts"].get(old_key, "n/a")
            new_val = profile.attributes.get(new_key, "n/a")
            print(f"  {old_key:28s} {old_val!s:>14s} {new_val!s:>14s}")


if __name__ == "__main__":
    main()
