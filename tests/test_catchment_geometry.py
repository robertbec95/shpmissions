"""Tests for the pure-geometry pieces of `shopping_missions.catchment` -
no raster/POI file dependency, fast."""

import math

import numpy as np
import pytest

from shopping_missions.catchment import haversine_km, make_buffer_geom

# Bucharest and Cluj-Napoca, city centers - real-world known distance is
# ~326km great-circle (Piata Victoriei to Piata Unirii coordinates used
# elsewhere in this project's candidate-site scoring).
BUCHAREST = (44.4325, 26.1039)
CLUJ = (46.7712, 23.6236)


def test_haversine_zero_distance_to_self():
    assert haversine_km(44.43, 26.10, 44.43, 26.10) == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_city_pair_distance():
    dist = haversine_km(*BUCHAREST, *CLUJ)
    assert dist == pytest.approx(326, rel=0.03)  # within 3%


def test_haversine_vectorized_over_array():
    lat2 = np.array([44.43, 46.77])
    lon2 = np.array([26.10, 23.62])
    dist = haversine_km(44.43, 26.10, lat2, lon2)
    assert dist.shape == (2,)
    assert dist[0] == pytest.approx(0.0, abs=1.0)  # near-self
    assert dist[1] > 300  # the Cluj leg


def test_haversine_one_degree_latitude_is_about_111km():
    # A pure latitude step at any longitude is ~111.19km/degree - a solid
    # sanity check on the formula independent of any real city pair.
    dist = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert dist == pytest.approx(111.19, rel=0.01)


def test_buffer_area_matches_circle_area_for_small_radius():
    """A 1km buffer should enclose ~pi km^2 - the whole point of the
    azimuthal-equidistant projection is that this holds accurately
    anywhere in Romania, not just near a reference meridian."""
    radius_km = 1.0
    geom = make_buffer_geom(45.0, 24.0, radius_km)

    # Rough planar-degree area isn't meaningful directly (lon degrees
    # aren't constant length) - reproject back through the same local
    # projection to measure area in real meters.
    import pyproj
    from shapely.ops import transform as shapely_transform

    aeqd = pyproj.CRS.from_proj4("+proj=aeqd +lat_0=45.0 +lon_0=24.0 +units=m")
    wgs84 = pyproj.CRS.from_epsg(4326)
    to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    geom_m = shapely_transform(to_aeqd, geom)

    expected_area_km2 = math.pi * radius_km**2
    actual_area_km2 = geom_m.area / 1e6
    assert actual_area_km2 == pytest.approx(expected_area_km2, rel=0.01)


def test_buffer_is_centered_on_the_query_point():
    geom = make_buffer_geom(45.0, 24.0, 1.0)
    centroid = geom.centroid
    assert centroid.x == pytest.approx(24.0, abs=1e-3)
    assert centroid.y == pytest.approx(45.0, abs=1e-3)
