"""Retry/backoff wrapper for the external HTTP calls this project makes -
added after this session hit real, unrecoverable-without-manual-retry
failures against INS TEMPO-Online (transient POST failures, not just the
already-fixed parsing bugs) and while downloading the WorldPop raster.

Deliberately NOT applied to the Overpass API pattern - that path is
already superseded by the Geofabrik/pyosmium national extraction
(`scripts/extract_osm_poi_romania.py`) and isn't in the critical path of
any script still in active use.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
RETRY_STATUS_CODES = (500, 502, 503, 504)


def retrying_session(
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: tuple[int, ...] = RETRY_STATUS_CODES,
) -> requests.Session:
    """A `requests.Session` that automatically retries on connection
    errors and the given server-error status codes, with exponential
    backoff (`backoff_factor * (2 ** (retry_number - 1))` seconds between
    attempts) - used in place of a bare `requests.get`/`.post` call for
    every external pull script in this project (INS TEMPO-Online,
    Eurostat, WorldPop)."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
