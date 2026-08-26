"""Regression test for the real bug found this session: scoring an actual
Profi location returned itself as its own "nearest competitor"
(nearest_competitor_km ~ 0.00006km), because the competitor categories are
plain shop=supermarket/convenience/grocery OSM tags with no brand
exclusion. Fixed via `OWN_BANNER_NAME_PATTERN` in
`shopping_missions.catchment`.

The fixture below is deliberately small and hand-built (not the 41K-row
national extract) so this test is fast and has no external file
dependency, but it reproduces every case that actually mattered when the
bug was found and fixed:
  - a query point co-located with its own Profi node (the original bug)
  - "Shop&Go" (Profi's own convenience banner) - must be excluded
  - "Mega Image Shop&Go" - a DIFFERENT retailer's own format using a
    similar name - must NOT be excluded (this is why the own-banner
    pattern anchors "shop&go" at the start of the name instead of doing
    a plain substring match, unlike "profi" which IS matched anywhere)
"""

import pandas as pd
import pytest

from shopping_missions.catchment import (
    OWN_BANNER_NAME_PATTERN,
    nearest_own_banner_distance,
    nearest_poi_distance,
)

# Distances from the query point (0.0, 0.0), by construction:
#   Profi City            ~0m     (own banner, self-match)
#   Shop&Go                ~100m   (own banner)
#   Mega Image Shop&Go     ~200m   (different retailer - must NOT be excluded)
#   Mega Image              ~500m   (true competitor, farther)
#   Carrefour              ~1000m  (true competitor, farthest)
POI_FIXTURE = pd.DataFrame([
    {"name": "Profi City", "category": "competitor_supermarket", "lat": 0.0, "lon": 0.0},
    {"name": "Shop&Go", "category": "competitor_convenience", "lat": 0.0009, "lon": 0.0},
    {"name": "Mega Image Shop&Go", "category": "competitor_supermarket", "lat": 0.0018, "lon": 0.0},
    {"name": "Mega Image", "category": "competitor_supermarket", "lat": 0.0045, "lon": 0.0},
    {"name": "Carrefour", "category": "competitor_convenience", "lat": 0.009, "lon": 0.0},
])

QUERY_LAT, QUERY_LON = 0.0, 0.0


def test_nearest_competitor_excludes_self_match():
    """The original bug: without exclusion, the nearest 'competitor' to a
    Profi site is the Profi site itself."""
    result = nearest_poi_distance(POI_FIXTURE, QUERY_LAT, QUERY_LON, "competitor_supermarket")
    assert result["name"] == "Profi City"
    assert result["distance_km"] == pytest.approx(0.0, abs=1e-6)


def test_nearest_competitor_with_exclusion_skips_own_banner():
    """With the fix applied, the nearest supermarket-category competitor
    must be 'Mega Image Shop&Go', not 'Profi City' (self) - and NOT
    'Shop&Go' either, since that's the wrong category here anyway; this
    specifically proves the exclusion does not eat the real competitor
    just because its name also contains 'Shop&Go'."""
    result = nearest_poi_distance(
        POI_FIXTURE, QUERY_LAT, QUERY_LON, "competitor_supermarket",
        exclude_name_pattern=OWN_BANNER_NAME_PATTERN,
    )
    assert result["name"] == "Mega Image Shop&Go"


def test_mega_image_shop_and_go_is_not_treated_as_own_banner():
    """The specific false-positive trap this pattern was designed around:
    a plain substring match on 'shop&go' would wrongly exclude 'Mega Image
    Shop&Go' too. It must survive exclusion."""
    excluded = POI_FIXTURE["name"].str.contains(OWN_BANNER_NAME_PATTERN, case=False, regex=True)
    assert not excluded[POI_FIXTURE["name"] == "Mega Image Shop&Go"].iloc[0]


@pytest.mark.parametrize("name", ["Profi City", "Shop&Go", "Depozit Profi", "Brana - Profi"])
def test_profi_family_names_are_excluded(name):
    row = pd.DataFrame([{"name": name}])
    assert row["name"].str.contains(OWN_BANNER_NAME_PATTERN, case=False, regex=True).iloc[0]


def test_nearest_own_banner_distance_finds_nearest_profi_node():
    result = nearest_own_banner_distance(POI_FIXTURE, QUERY_LAT, QUERY_LON)
    assert result["name"] == "Profi City"
    assert result["distance_km"] == pytest.approx(0.0, abs=1e-6)


def test_nearest_poi_distance_returns_inf_when_category_absent():
    result = nearest_poi_distance(POI_FIXTURE, QUERY_LAT, QUERY_LON, "pharmacy")
    assert result["distance_km"] == float("inf")
    assert result["name"] is None
