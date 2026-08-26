"""Filters the full Romania OSM extract (`romania-latest.osm.pbf`, pulled
from Geofabrik, 326MB - all roads/buildings/everything) down to just the
POI categories this project's catchment work actually uses, national
coverage this time, not just 1km around the 6 known stores.

Why not Overpass for this: national-scale Overpass queries against the
free public mirrors hit a hard resource limit (`the dispatcher is turned
off`, HTTP 504) even for a simple national count - the per-store 1km
radius queries used earlier in this project work fine on those mirrors,
but whole-country queries don't. Downloading the raw extract and filtering
it locally with `pyosmium` (streaming, does not load the full file into
memory) avoids the shared-API rate/resource limit entirely, and is the
standard approach for country-scale OSM extraction.

Scope/limitation: NODE-tagged POIs only (point features), matching the
node-only Overpass queries this project already validated for the 6
known stores. Some larger supermarkets/schools are mapped as way
(building outline) geometries in OSM rather than nodes, and are NOT
captured here - a known undercount, not a bug, kept consistent with the
existing per-store methodology rather than silently changing what counts
as "captured" between the two.

Category tag mapping (mirrors the 9 categories pulled per-store earlier):
    amenity=school            -> school
    amenity=kindergarten      -> kindergarten
    amenity=place_of_worship  -> place_of_worship
    amenity=pharmacy          -> pharmacy
    amenity=bank              -> bank
    shop=supermarket          -> competitor_supermarket
    shop=convenience          -> competitor_convenience
    shop=grocery              -> competitor_grocery
    highway=bus_stop / public_transport=stop / railway=tram_stop|station -> transit_stop

Reproduce with: `PYTHONPATH=src python3 scripts/extract_osm_poi_romania.py`
"""

import csv

import osmium

IN_PBF = "data/location_features/osm/romania-latest.osm.pbf"
OUT_CSV = "data/location_features/osm/romania_poi_nodes.csv"

TAG_CATEGORY_MAP = {
    ("amenity", "school"): "school",
    ("amenity", "kindergarten"): "kindergarten",
    ("amenity", "place_of_worship"): "place_of_worship",
    ("amenity", "pharmacy"): "pharmacy",
    ("amenity", "bank"): "bank",
    ("shop", "supermarket"): "competitor_supermarket",
    ("shop", "convenience"): "competitor_convenience",
    ("shop", "grocery"): "competitor_grocery",
    ("highway", "bus_stop"): "transit_stop",
    ("public_transport", "stop"): "transit_stop",
    ("railway", "tram_stop"): "transit_stop",
    ("railway", "station"): "transit_stop",
    ("shop", "mall"): "mall",
    ("shop", "department_store"): "mall",
    ("amenity", "hospital"): "anchor_institution",
    ("amenity", "university"): "anchor_institution",
    ("amenity", "college"): "anchor_institution",
    ("office", "government"): "anchor_institution",
    ("amenity", "conference_centre"): "event_venue",
    ("leisure", "stadium"): "event_venue",
    ("amenity", "charging_station"): "ev_charging",
    ("tourism", "hotel"): "tourism_accommodation",
    ("tourism", "hostel"): "tourism_accommodation",
    ("tourism", "guest_house"): "tourism_accommodation",
}

# `mall` and `event_venue` are KNOWN SEVERE UNDERCOUNTS, worse than
# schools/churches: national tag scans found only 14 shop=mall NODES vs
# 254 WAYS, and 18 event-venue NODES (2 conference_centre + 16 stadium)
# vs 340 WAYS - both categories are overwhelmingly mapped as building/area
# outlines in OSM, not points. `anchor_institution` (hospitals,
# universities, government offices) has a real but milder undercount
# (node counts roughly half of way counts for hospital/university;
# government offices actually have MORE nodes than ways). `ev_charging`
# and `tourism_accommodation` have good node coverage (nodes >= ways for
# both) and should be trusted more than the others. Kept node-only for
# consistency with every other category here (see module docstring) -
# way-centroid support would fix the undercount properly, not attempted
# here (a real scope increase, not a "cheap" fix).


class POIExtractor(osmium.SimpleHandler):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.counts = {}
        self.n_seen = 0

    def node(self, n):
        self.n_seen += 1
        for (key, val), category in TAG_CATEGORY_MAP.items():
            if n.tags.get(key) == val:
                name = n.tags.get("name", "")
                brand = n.tags.get("brand", "")
                self.writer.writerow(
                    [n.id, category, key, val, name, brand, n.location.lat, n.location.lon]
                )
                self.counts[category] = self.counts.get(category, 0) + 1
                break  # first matching category wins, avoids double-counting multi-tagged nodes


def main():
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["osm_id", "category", "tag_key", "tag_value", "name", "brand", "lat", "lon"])
        handler = POIExtractor(writer)
        handler.apply_file(IN_PBF, locations=True)

    print(f"Scanned {handler.n_seen:,} nodes total")
    print(f"Wrote {OUT_CSV}")
    print("Counts by category:")
    for cat, n in sorted(handler.counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:28s} {n:,}")


if __name__ == "__main__":
    main()
