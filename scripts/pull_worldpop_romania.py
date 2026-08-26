"""One-off pull of WorldPop's national Romania population grid - the
national-coverage counterpart to the 6-store-only INS TEMPO-Online pulls
in `location_similarity.py`. Unlike that per-city lookup, this raster
covers the ENTIRE country at 100m resolution, so any future candidate
site (not just the 6 known stores) can get a real population-per-km2
figure via a local zonal-statistics query against the cached file below -
no further network calls needed once pulled.

Source: WorldPop "Global2" R2025A release, constrained individual-country
100m population counts, 2025 (`G2_CN_POP_R25A_100m` in WorldPop's REST API
- queried via `https://hub.worldpop.org/rest/data/pop/G2_CN_POP_R25A_100m?iso3=ROU`
to find this exact file URL; other years 2015-2030 are available under the
same dataset if a different reference year is ever needed).

"Constrained" means population is redistributed only into built-up areas
(per WorldPop's settlement layer) rather than spread uniformly across
administrative units - the more accurate of WorldPop's two population
products for exactly this kind of within-city catchment work.

License: WorldPop, free for commercial/non-commercial use with
attribution (https://hub.worldpop.org/data/licence.txt) - CC BY 4.0, no
caching/storage restriction (unlike the Google Places API terms discussed
for the POI/competitor side of this pipeline).

Sanity check after download: summed raster = 18,962,338 - matches
Romania's real ~19M population, confirming this is the right file, not a
truncated or wrong-country download.

Deliberately NOT wrapped into an importable module (same reasoning as
Nominatim/Overpass/INS in `location_similarity.py`'s docstring) - this is
a one-off pull to a cached file, re-run only if a newer WorldPop release
year is wanted.

Reproduce with: `python3 scripts/pull_worldpop_romania.py`
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_missions.http_utils import retrying_session

OUT_DIR = Path("data/location_features/worldpop")
POP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/ROU/v1/100m/"
    "constrained/rou_pop_2025_CN_100m_R2025A_v1.tif"
)
POP_FILE = OUT_DIR / "rou_pop_2025_CN_100m_R2025A_v1.tif"
MANIFEST_FILE = OUT_DIR / "manifest.json"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with retrying_session().get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def write_manifest() -> None:
    import rasterio

    with rasterio.open(POP_FILE) as src:
        band = src.read(1, masked=True)
        manifest = {
            "dataset": "WorldPop Global2 R2025A, constrained, 100m",
            "country": "Romania",
            "iso3": "ROU",
            "year": 2025,
            "source_url": POP_URL,
            "license_url": "https://hub.worldpop.org/data/licence.txt",
            "crs": str(src.crs),
            "shape_rows_cols": list(src.shape),
            "resolution_deg": list(src.res),
            "bounds": list(src.bounds),
            "nodata": src.nodata,
            "total_population_check": float(band.sum()),
            "note": "Total population check should land close to Romania's real ~19M population.",
        }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    if not POP_FILE.exists():
        print(f"Downloading {POP_URL} -> {POP_FILE}")
        download(POP_URL, POP_FILE)
    write_manifest()
    print(f"Wrote {MANIFEST_FILE}")
