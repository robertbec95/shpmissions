"""The first genuinely-new-site run of this project's full pipeline:
Gower distance (v4, weighted, validated) + k=3 nearest-analog propagation
of the joint NMF's topic-prevalence profiles, applied to 4 real candidate
locations that are NOT among the 6 known stores (Cluj-Napoca, Iasi,
Timisoara, Sighisoara) - everything before this was leave-one-out on the
6 known stores themselves.

Two categorical inputs are ASSUMPTIONS, stated explicitly rather than
silently guessed: `format` (set to "Convenience Store" for all 4 - the
most common format among the 6 known stores; this is a "what if we opened
this format here" scenario, not a real Profi classification) and
`locality_type`/`locality_type_l2` (assigned by population-tier analogy
to the closest known store - e.g. Cluj/Iasi/Timisoara at 300-390k people
are closest in scale to Craiova's "Urban Medium" tier). Profi's actual
internal locality-classification rules aren't recoverable from this
project's data for a site with no store there yet - these are the most
defensible estimates available, not precise values, and they carry the
LOWEST group weight (0.5) of the 4 established v4 groups, so their effect
on the result is real but secondary to the higher-weighted population/POI
signal.

PC1/PC2 (POI density) are computed by refitting the EXACT StandardScaler
+ PCA used for the 6 known stores' PC1/PC2 (verified to reproduce their
values exactly from `store_poi_counts.json`), then transforming each
candidate's raw POI counts (from `catchment.catchment_profile`, national
OSM extract, radius=1km) through that same fitted transform - not a
fresh PCA fit, which would use a different basis and break comparability.

Population/age figures pulled fresh from INS TEMPO-Online (POP107D),
same methodology as the original 5 cities.

Reproduce with: `PYTHONPATH=src python3 scripts/run_candidate_site_propagation.py`
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from shopping_missions import location_similarity as loc_sim
from shopping_missions.catchment import catchment_profile

V4_FILE = "data/location_features/store_feature_table_v3.json"
RAW_POI_FILE = "data/location_features/store_poi_counts.json"
GROUND_TRUTH_FILE = "data/location_features/store_topic_prevalence_ground_truth.json"
POI_CSV = "data/location_features/osm/romania_poi_nodes.csv"
POP_RASTER = "data/location_features/worldpop/rou_pop_2025_CN_100m_R2025A_v1.tif"

TOPIC_LABELS = [
    "T0 Vegetables+Household+Fruits", "T1 Dessert Drinks", "T2 Bake Off", "T3 Sweet Snacks",
    "T4 Beer+Tobacco", "T5 Dessert Ingr.+Rice/Beans/Pasta+Sugar", "T6 Yogurt/Dairy+Cheese+Sausages",
    "T7 Water+Carbonated", "T8 Ice Cream+Desserts", "T9 Pet Food",
    "T10 Salty Snacks+Chocolate+Carbonated", "T11 Coffee Corner+Tobacco+Others Fiigo",
]

CANDIDATES = {
    "Cluj-Napoca (Piata Unirii)": {
        "lat": 46.7697489, "lon": 23.5897267, "format": "Convenience Store",
        "locality_type": "Urban", "locality_type_l2": "Urban Medium",
        "population": 327_091, "pct_female": 53.18, "pct_children_0_14": 14.33, "pct_seniors_65plus": 17.46,
    },
    "Iasi (Piata Unirii)": {
        "lat": 47.1666910, "lon": 27.5801155, "format": "Convenience Store",
        "locality_type": "Urban", "locality_type_l2": "Urban Medium",
        "population": 389_895, "pct_female": 51.98, "pct_children_0_14": 16.07, "pct_seniors_65plus": 14.88,
    },
    "Timisoara (Piata Victoriei)": {
        "lat": 45.7526802, "lon": 21.2254959, "format": "Convenience Store",
        "locality_type": "Urban", "locality_type_l2": "Urban Medium",
        "population": 302_980, "pct_female": 53.67, "pct_children_0_14": 12.51, "pct_seniors_65plus": 18.95,
    },
    "Sighisoara (town center)": {
        "lat": 46.2202037, "lon": 24.7934483, "format": "Convenience Store",
        "locality_type": "Urban", "locality_type_l2": "Urban Small",
        "population": 31_784, "pct_female": 51.91, "pct_children_0_14": 11.07, "pct_seniors_65plus": 18.63,
    },
}

# raw poi_counts.json category name -> catchment.py poi_*_count name, in the exact
# order the original PCA was fit on (after dropping zero-variance competitor_grocery)
POI_CATEGORY_MAP = {
    "schools": "poi_school_count", "kindergartens": "poi_kindergarten_count",
    "places_of_worship": "poi_place_of_worship_count", "competitor_supermarket": "poi_competitor_supermarket_count",
    "competitor_convenience": "poi_competitor_convenience_count", "pharmacies": "poi_pharmacy_count",
    "banks": "poi_bank_count", "public_transport_stops": "poi_transit_stop_count",
}


def fit_original_pca():
    raw = json.load(open(RAW_POI_FILE))
    df = pd.DataFrame([s["poi_counts"] for s in raw], index=[s["loc_id"] for s in raw])
    keep = list(POI_CATEGORY_MAP.keys())
    scaler = StandardScaler().fit(df[keep])
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(df[keep]))
    return scaler, pca, keep


def main():
    v4 = json.load(open(V4_FILE))
    ground_truth = json.load(open(GROUND_TRUTH_FILE))
    poi_df = pd.read_csv(POI_CSV)
    scaler, pca, keep_order = fit_original_pca()

    candidate_rows = {}
    for name, c in CANDIDATES.items():
        profile = catchment_profile(c["lat"], c["lon"], radius_km=1.0, raster_path=POP_RASTER, poi_df=poi_df)
        raw_counts = [[profile.attributes[POI_CATEGORY_MAP[cat]] for cat in keep_order]]
        pc1, pc2 = pca.transform(scaler.transform(raw_counts))[0]

        candidate_rows[name] = {
            "format": c["format"], "locality_type": c["locality_type"], "locality_type_l2": c["locality_type_l2"],
            "PC1": float(pc1), "PC2": float(pc2), "log_population": float(np.log(c["population"])),
            "pct_female": c["pct_female"], "pct_children_0_14": c["pct_children_0_14"],
            "pct_seniors_65plus": c["pct_seniors_65plus"],
        }

    combined = pd.DataFrame({**v4, **candidate_rows}).T
    numeric_cols = ["PC1", "PC2", "log_population", "pct_female", "pct_children_0_14", "pct_seniors_65plus"]
    for col in numeric_cols:
        combined[col] = combined[col].astype(float)

    distance = loc_sim.weighted_gower_distance(combined)

    print("=== Predicted mission mix for 4 candidate sites (k=3 nearest known-store analogs) ===\n")
    for name in candidate_rows:
        result = loc_sim.propagate_with_detail(distance, ground_truth, name, k=3, exclude={name})
        print(f"--- {name} ---")
        print("  Nearest analogs:")
        for analog, dist in zip(result.nearest_analogs, result.distances):
            store_label = next((s["raw_name"] for s in json.load(open("data/location_features/store_geocodes.json"))
                                 if s["loc_id"] == analog), analog)
            print(f"    {store_label}  (Gower distance {dist:.3f})")
        top3_idx = np.argsort(result.predicted_profile)[::-1][:3]
        print("  Top 3 predicted missions:")
        for i in top3_idx:
            print(f"    {TOPIC_LABELS[i]}: {result.predicted_profile[i]:.1f}%")
        print()


if __name__ == "__main__":
    main()
