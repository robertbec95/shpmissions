"""Builds v7: v6's 27 columns unchanged, plus every remaining real
attribute pulled this session EXCEPT the 2021 census ethnicity/religion/
mother-tongue data - the user explicitly decided (asked directly, not
assumed) to keep that as reference-only data, not a weighted input to
the site-scoring algorithm. Everything else pulled this session goes in:

    - population_cagr_pct: 2019->2024 population growth trend (INS)
    - unemployment_trend_pct: 2020->2025 CHANGE in unemployment share
      (distinct from `unemployment_share_pct`, already in v5/v6, which is
      only the current LEVEL - this is the trend, a different signal)
    - school_enrollment_pct: enrollment / city population (INS SCL103D)
    - tourism_overnight_stays_per_capita: overnight stays / city
      population (INS TUR105E) - expect this to matter for Brasov
      specifically, given its 1.36M stays dwarfs every other city
    - log_car_ownership_county: log(registered passenger cars, natural
      persons) - COUNTY-level (no locality option exists for this INS
      matrix), kept as a raw log-scale count rather than manufacturing a
      per-capita rate with a mismatched (city vs. county) denominator
    - 9 banner-level competitor counts (Mega Image, Carrefour, Penny,
      Independent/Other, etc.) - a different cut of data already in v6's
      `poi_competitor_*_count` (by tier), this time by actual chain.
      Kaufland/Lidl/Cora dropped: zero variance, never appear in any of
      the 6 stores' 1km catchments.

3 NEW groups, grid-searched independently - all 8 previously-established
group weights (v4's original 4 + v5's catchment/economic + v6's
poi_detail/competition) held FIXED, same disciplined pattern used for
every prior version:
    - "growth": population_cagr_pct, unemployment_trend_pct
    - "footfall_activity": school_enrollment_pct,
      tourism_overnight_stays_per_capita, log_car_ownership_county
    - "banner": the 9 banner-count columns

Reproduce with: `PYTHONPATH=src python3 scripts/build_feature_table_v7.py`
"""

import json

import numpy as np

V6_FILE = "data/location_features/store_feature_table_v6.json"
GEOCODES_FILE = "data/location_features/store_geocodes.json"
GROWTH_FILE = "data/location_features/ins_population_growth_2019_2024.json"
UNEMP_TREND_FILE = "data/location_features/ins_unemployment_trend_2020_2025.json"
ENROLLMENT_FILE = "data/location_features/ins_school_enrollment_2024.json"
TOURISM_FILE = "data/location_features/ins_tourism_overnight_stays_2024.json"
CAR_FILE = "data/location_features/ins_car_ownership_2025.json"
BANNER_FILE = "data/location_features/competitor_banner_breakdown.json"
OUT_FILE = "data/location_features/store_feature_table_v7.json"

CITY_LABEL = {  # store_geocodes.json city -> the label used in each pulled file
    "Bucharest": "Bucharest", "Piatra Neamt": "Neamt (Piatra-Neamt)", "Calarasi": "Calarasi",
    "Craiova": "Dolj (Craiova)", "Brasov": "Brasov",
}
COUNTY_LABEL = {"Bucharest": "Bucharest", "Neamt": "Neamt", "Calarasi": "Calarasi", "Dolj": "Dolj", "Brasov": "Brasov"}

BANNER_COLS = [
    "Independent/Other", "Mega Image", "Unnamed", "Carrefour", "Froo",
    "La Doi Pasi", "Penny", "Auchan", "Annabella",
]

# raw city population (for computing the two new per-capita rates), from
# ins_population_by_city.json (2024, already used throughout this project)
CITY_POPULATION = {
    "Bucharest": 2_143_578, "Neamt (Piatra-Neamt)": 106_814, "Calarasi": 70_861,
    "Dolj (Craiova)": 285_722, "Brasov": 283_848,
}


def main():
    v6 = json.load(open(V6_FILE))
    geocodes = {g["loc_id"]: g for g in json.load(open(GEOCODES_FILE))}
    growth = json.load(open(GROWTH_FILE))
    unemp_trend = json.load(open(UNEMP_TREND_FILE))["by_city"]
    enrollment = json.load(open(ENROLLMENT_FILE))["by_city"]
    tourism = json.load(open(TOURISM_FILE))["by_city"]
    cars = json.load(open(CAR_FILE))
    banners = json.load(open(BANNER_FILE))

    v7 = {}
    for loc_id, row in v6.items():
        geo = geocodes[loc_id]
        city_key = CITY_LABEL[geo["city"]]
        pop = CITY_POPULATION[city_key]

        new_cols = {
            "population_cagr_pct": growth[city_key]["population_cagr_pct"],
            "unemployment_trend_pct": round(unemp_trend[city_key]["2025"] - unemp_trend[city_key]["2020"], 3),
            "school_enrollment_pct": round(100 * enrollment[city_key] / pop, 3),
            "tourism_overnight_stays_per_capita": round(tourism[city_key] / pop, 4),
            "log_car_ownership_county": float(np.log(cars[geo["county"]]["passenger_cars_natural_persons"])),
        }
        banner_counts = banners[loc_id]["banner_counts"]
        for col in BANNER_COLS:
            new_cols[f"banner_{col.lower().replace('/', '_').replace(' ', '_')}"] = banner_counts.get(col, 0)

        v7[loc_id] = {**row, **new_cols}

    with open(OUT_FILE, "w") as f:
        json.dump(v7, f, indent=2)
    print(f"Wrote {OUT_FILE} ({len(next(iter(v7.values())))} columns)")
    for loc_id, row in v7.items():
        print(f"  {loc_id}: cagr={row['population_cagr_pct']}%, tourism/capita={row['tourism_overnight_stays_per_capita']}, "
              f"enrollment%={row['school_enrollment_pct']}")


if __name__ == "__main__":
    main()
