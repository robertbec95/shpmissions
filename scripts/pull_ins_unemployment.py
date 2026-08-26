"""One-off pull of registered-unemployment share by locality (matrix
SOM101F, "Share of registered unemployed at the end of the month in the
total labor resources, by gender, counties and localities") from Romania's
INS TEMPO-Online API, for the 5 cities covering this project's 6 known
stores - the same API/matrix mechanics documented in the
`shpmissions-location-similarity` memory (POST /tempo-ins/pivot,
Bucharest's special county-dimension gotcha, etc.), just not previously
saved to a cached file (was run once inline and only printed).

County/locality nomItemIds reused unchanged from the population pull
(`ins_population_by_city.json`) - confirmed identical across matrices in
an earlier session. Period: "Year 2025" (nomItemId 4912), matching the
basket data's full-2025 coverage.

Reproduce with: `python3 scripts/pull_ins_unemployment.py`
"""

import json

import requests

OUT_FILE = "data/location_features/ins_unemployment_by_city.json"

CITIES = {
    "Bucuresti Municipality": {"county_id": 3104, "loc_id": 3048, "store_counties": ["Bucharest"]},
    "Piatra-Neamt Municipality": {"county_id": 3090, "loc_id": 2051, "store_counties": ["Neamt"]},
    "Calarasi Municipality": {"county_id": 3105, "loc_id": 1544, "store_counties": ["Calarasi"]},
    "Craiova Municipality": {"county_id": 3079, "loc_id": 1187, "store_counties": ["Dolj"]},
    "Brasov Municipality": {"county_id": 3071, "loc_id": 674, "store_counties": ["Brasov"]},
}

SEX_TOTAL = 105
PERIOD_YEAR_2025 = 4912
MU_PERCENTAGE = 10225

PIVOT_URL = "http://statistici.insse.ro:8077/tempo-ins/pivot"


def fetch_unemployment() -> list[dict]:
    county_ids = ",".join(str(v["county_id"]) for v in CITIES.values())
    loc_ids = ",".join(str(v["loc_id"]) for v in CITIES.values())
    enc_query = f"{SEX_TOTAL}:{county_ids}:{loc_ids}:{PERIOD_YEAR_2025}:{MU_PERCENTAGE}"

    payload = {
        "language": "en", "encQuery": enc_query, "matCode": "SOM101F",
        "matMaxDim": 5, "matUMSpec": 0, "matRegJ": 0,
    }
    resp = requests.post(PIVOT_URL, json=payload, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    rows = lines[1:]

    # "Counties" column (e.g. "Brasov", "Bucharest Municipality") is the
    # reliable join key - the "Localities" column only shows a SIRUTA
    # code + name (e.g. "40198 BRASOV MUNICIPALITY"), not the nomItemId
    # used to request it, so it can't be matched back to `loc_id` directly.
    by_county_label = {}
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        county_raw, value = parts[1], parts[-1]
        by_county_label[county_raw] = float(value) if value not in ("", ":") else None

    results = []
    for city, meta in CITIES.items():
        county = meta["store_counties"][0]
        label = "Bucharest Municipality" if county == "Bucharest" else county
        match = by_county_label.get(label)
        results.append({
            "city": city, "county_id": meta["county_id"], "loc_id": meta["loc_id"],
            "store_counties": meta["store_counties"], "year": 2025,
            "source": "INS TEMPO-Online SOM101F",
            "unemployment_share_pct": match,
        })
    return results


if __name__ == "__main__":
    results = fetch_unemployment()
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_FILE}")
    for r in results:
        print(f"  {r['city']:30s} {r['unemployment_share_pct']}%")
