"""Turns the national-coverage raw data pulled for this project (WorldPop's
Romania population raster, the OSM/Geofabrik POI extract) into an actual
per-location attribute row - the missing piece between "we have national
data" and "we can score any candidate site in Romania."

Catchment definition: a fixed-radius circular buffer around a point, NOT
a routing-based isochrone. Isochrones (actual drive/walk-time polygons via
OSRM/GraphHopper) would be a more realistic catchment shape, but that's a
separate, not-yet-built piece - a circle is the standard, defensible
fallback (and is what a "1km around the store" radius already informally
meant in this project's earlier per-store POI pulls). Documented here so
this doesn't get silently presented as more precise than it is.

Buffering is done in an azimuthal-equidistant projection centered on the
query point itself (via pyproj), not a fixed UTM zone - Romania spans two
UTM zones (34N/35N), so a per-point local projection avoids edge-of-zone
distortion for any location in the country, not just ones near a
particular meridian.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.mask
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform

EARTH_RADIUS_KM = 6371.0088

POI_CATEGORIES = [
    "school", "kindergarten", "place_of_worship", "pharmacy", "bank",
    "competitor_supermarket", "competitor_convenience", "competitor_grocery",
    "transit_stop", "mall", "anchor_institution", "event_venue", "ev_charging",
    "tourism_accommodation",
]
COMPETITOR_CATEGORIES = ["competitor_supermarket", "competitor_convenience", "competitor_grocery"]

# Profi's own banner family, as it actually appears in OSM `name` tags
# (discovered by bug: scoring a REAL Profi location returned itself as
# its own "nearest competitor" at ~0.00006km, since the competitor
# categories are plain shop=supermarket/convenience/grocery tags with no
# brand exclusion). "profi" is matched anywhere in the name (substring,
# case-insensitive) to catch "Depozit Profi"/"Brana - Profi"-style
# entries too. "shop&go"/"shopgo" is matched only at the START of the
# name - "Mega Image Shop&Go" is a DIFFERENT retailer's own format using
# a similar name and must NOT be excluded as if it were Profi's.
OWN_BANNER_NAME_PATTERN = r"^(?:.*\bprofi\b.*|shop ?& ?go.*)$"


def make_buffer_geom(lat: float, lon: float, radius_km: float):
    """A circular buffer around (lat, lon), returned as a shapely polygon
    in EPSG:4326 - built by projecting to a local azimuthal-equidistant
    CRS (true distances at the buffer's center), buffering in meters,
    then reprojecting back so it can overlay the WGS84 raster/POI data."""
    aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m")
    wgs84 = pyproj.CRS.from_epsg(4326)
    to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(aeqd, wgs84, always_xy=True).transform

    point_aeqd = shapely_transform(to_aeqd, Point(lon, lat))
    buffer_aeqd = point_aeqd.buffer(radius_km * 1000)
    return shapely_transform(to_wgs84, buffer_aeqd)


def population_in_buffer(raster_path: str, buffer_geom, nodata: float = -99999.0) -> dict:
    """Sums WorldPop pixel values (= estimated people) intersecting the
    buffer polygon, via `rasterio.mask` (a windowed read/crop - never
    loads the full national raster into memory for a single-buffer
    query). NOTE: deliberately NOT using the `rasterstats` package here -
    it reliably segfaults in this environment (isolated via a minimal
    repro; likely a GDAL/PROJ binding conflict between rasterstats' and
    pyproj's separately-bundled native libraries), while calling
    `rasterio.mask.mask` directly on the same geometry gives an identical
    result with no crash."""
    with rasterio.open(raster_path) as src:
        out_image, _ = rasterio.mask.mask(src, [buffer_geom], crop=True, nodata=nodata)
        band = out_image[0]
        valid = band[band != nodata]
    return {
        "catchment_population": float(valid.sum()),
        "catchment_pixel_count": int(valid.size),
    }


def haversine_km(lat1: float, lon1: float, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance - lat2/lon2 may be arrays, for
    computing distance from one query point to every POI at once."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def poi_counts_in_radius(poi_df: pd.DataFrame, lat: float, lon: float, radius_km: float) -> dict:
    """Count of each POI category within `radius_km` of (lat, lon) -
    equivalent to counting points inside the same circular buffer used
    for the population raster, computed directly via haversine distance
    rather than a geometric point-in-polygon test (simpler and exact for
    a circular catchment, no per-point CRS transform needed)."""
    dist = haversine_km(lat, lon, poi_df["lat"].values, poi_df["lon"].values)
    within = poi_df.loc[dist <= radius_km]
    counts = within["category"].value_counts().to_dict()
    return {f"poi_{cat}_count": int(counts.get(cat, 0)) for cat in POI_CATEGORIES}


def nearest_poi_distance(
    poi_df: pd.DataFrame, lat: float, lon: float, category: str,
    exclude_name_pattern: str | None = None,
) -> dict:
    """Distance in km to the single nearest POI of `category`, plus its
    name if tagged - `None`/`inf` if that category has no POI anywhere
    in the loaded dataset (or none left after `exclude_name_pattern`
    filters out matching names, e.g. the query site's own banner)."""
    sub = poi_df.loc[poi_df["category"] == category]
    if exclude_name_pattern:
        sub = sub.loc[~sub["name"].astype(str).str.contains(exclude_name_pattern, case=False, regex=True, na=False)]
    if sub.empty:
        return {"distance_km": float("inf"), "name": None}
    dist = haversine_km(lat, lon, sub["lat"].values, sub["lon"].values)
    idx = np.argmin(dist)
    row = sub.iloc[idx]
    return {"distance_km": float(dist[idx]), "name": row.get("name") or None}


def nearest_own_banner_distance(
    poi_df: pd.DataFrame, lat: float, lon: float, own_banner_pattern: str = OWN_BANNER_NAME_PATTERN,
) -> dict:
    """Distance to the nearest node matching `own_banner_pattern` across
    ANY competitor category (a same-banner store might be tagged
    shop=supermarket at one location and shop=convenience at another) -
    the cannibalization-risk counterpart to `nearest_competitor_km`.
    NOTE: if the query point IS itself an existing own-banner store, this
    will trivially return ~0km (matching itself) - a real, expected edge
    case for scoring an existing site, not a bug, but worth checking for
    when interpreting the result."""
    sub = poi_df.loc[poi_df["category"].isin(COMPETITOR_CATEGORIES)]
    sub = sub.loc[sub["name"].astype(str).str.contains(own_banner_pattern, case=False, regex=True, na=False)]
    if sub.empty:
        return {"distance_km": float("inf"), "name": None}
    dist = haversine_km(lat, lon, sub["lat"].values, sub["lon"].values)
    idx = np.argmin(dist)
    row = sub.iloc[idx]
    return {"distance_km": float(dist[idx]), "name": row.get("name") or None}


@dataclass
class CatchmentProfile:
    lat: float
    lon: float
    radius_km: float
    attributes: dict = field(default_factory=dict)


def catchment_profile(
    lat: float, lon: float, radius_km: float, raster_path: str, poi_df: pd.DataFrame,
    own_banner_pattern: str | None = OWN_BANNER_NAME_PATTERN,
) -> CatchmentProfile:
    """The actual per-location feature row: population within the
    catchment (from the WorldPop raster) + POI counts by category within
    the same catchment (from the OSM extract) + distance to the nearest
    TRUE competitor (excluding `own_banner_pattern` matches - pass None
    to disable this and reproduce the pre-fix behavior) + distance to the
    nearest own-banner store (cannibalization risk)."""
    buffer_geom = make_buffer_geom(lat, lon, radius_km)
    attrs = {}
    attrs.update(population_in_buffer(raster_path, buffer_geom))
    attrs.update(poi_counts_in_radius(poi_df, lat, lon, radius_km))

    nearest = [nearest_poi_distance(poi_df, lat, lon, cat, exclude_name_pattern=own_banner_pattern)
               for cat in COMPETITOR_CATEGORIES]
    nearest = min(nearest, key=lambda x: x["distance_km"])
    attrs["nearest_competitor_km"] = nearest["distance_km"]
    attrs["nearest_competitor_name"] = nearest["name"]

    if own_banner_pattern:
        own = nearest_own_banner_distance(poi_df, lat, lon, own_banner_pattern)
        attrs["nearest_own_banner_km"] = own["distance_km"]
        attrs["nearest_own_banner_name"] = own["name"]

    return CatchmentProfile(lat=lat, lon=lon, radius_km=radius_km, attributes=attrs)
