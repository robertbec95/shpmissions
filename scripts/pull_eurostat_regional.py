"""One-off pull of Eurostat regional (NUTS3) economic indicators for the
5 counties covering this project's 6 known stores - the Eurostat
counterpart to the INS TEMPO-Online pulls (population, age, unemployment)
already cached in `data/location_features/`.

Dataset: `nama_10r_3gdp` - GDP at current market prices by NUTS 3 region,
unit=EUR_HAB (euro per inhabitant), 2018-2024. Queried via Eurostat's
JSON-stat REST API: `GET /eurostat/api/dissemination/statistics/1.0/data/
nama_10r_3gdp?format=JSON&lang=en&unit=EUR_HAB&sinceTimePeriod=2018`.
GDP per capita is used (not a raw wage/earnings figure) because it's
already correctly normalized per-inhabitant at exactly the granularity
this project needs (NUTS3/county) - Eurostat's wage/earnings datasets are
NOT available at this granularity for Romania (see gap notes below), and
deriving a per-employee wage from `nama_10r_2coe` (compensation of
employees, NUTS2, raw million-EUR totals) would need a second dataset
(employment counts) just to normalize it, for a redundant signal GDP per
capita already substantially captures.

Two genuine gaps found and NOT worked around (real, not silently
skipped):
- Wages/earnings: only available for Romania at NUTS1 (very coarse - a
  handful of national macro-regions) or as compensation-of-employees
  totals at NUTS2 (not per-capita/per-employee without a second dataset).
  No NUTS3-level wage data exists in Eurostat for Romania.
- Education attainment: only available at NUTS2 (2011 census,
  `cens_11aed_r2`) or NUTS3 (2001 census, `cens_01rews`/`cens_01reisco`)
  - both stale snapshots, not current data. This independently confirms
    the same gap already found when checking INS TEMPO-Online directly
    (see `shpmissions-location-similarity` memory) - two separate
    national/EU statistical systems, same missing granularity.

NUTS3 codes for the 5 counties (verified against Eurostat's own geo
dimension labels, not assumed): Bucharest=RO321, Brasov=RO122,
Neamt=RO214, Calarasi=RO312, Dolj=RO411.

Reproduce with: `python3 scripts/pull_eurostat_regional.py`
"""

import json

import requests

OUT_FILE = "data/location_features/eurostat_gdp_per_capita.json"

NUTS3_CODES = {
    "RO321": "Bucharest", "RO122": "Brasov", "RO214": "Neamt",
    "RO312": "Calarasi", "RO411": "Dolj",
}

API_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nama_10r_3gdp"
    "?format=JSON&lang=en&unit=EUR_HAB&sinceTimePeriod=2018"
)


def fetch_gdp_series() -> dict:
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    d = resp.json()

    dim_ids = d["id"]
    size = d["size"]
    geo_index = d["dimension"]["geo"]["category"]["index"]
    time_labels = list(d["dimension"]["time"]["category"]["label"].keys())
    values = d["value"]

    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]
    geo_pos, time_pos = dim_ids.index("geo"), dim_ids.index("time")

    results = []
    for code, county in NUTS3_CODES.items():
        gidx = geo_index[code]
        series = {}
        for t_i, year in enumerate(time_labels):
            offset = gidx * strides[geo_pos] + t_i * strides[time_pos]
            val = values.get(str(offset))
            if val is not None:
                series[year] = val
        years_sorted = sorted(series)
        growth_pct = None
        if len(years_sorted) >= 2:
            first, last = series[years_sorted[0]], series[years_sorted[-1]]
            n_years = int(years_sorted[-1]) - int(years_sorted[0])
            growth_pct = round(((last / first) ** (1 / n_years) - 1) * 100, 2) if n_years > 0 else None
        results.append({
            "nuts3_code": code,
            "county": county,
            "gdp_per_capita_eur_by_year": series,
            "gdp_per_capita_cagr_pct": growth_pct,
            "source": "Eurostat nama_10r_3gdp, unit=EUR_HAB",
        })
    return results


if __name__ == "__main__":
    results = fetch_gdp_series()
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_FILE}")
    for r in results:
        latest_year = max(r["gdp_per_capita_eur_by_year"])
        print(f"  {r['county']:10s} {r['gdp_per_capita_eur_by_year'][latest_year]:>7,} EUR/capita ({latest_year}), "
              f"CAGR {r['gdp_per_capita_cagr_pct']}%")
