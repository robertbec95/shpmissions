# shpmissions

Shopping-mission identification on Profi Romania POS data (replicating and
extending Sarantopoulos et al. 2015), plus an Applebaum-style location-similarity
pipeline that scores catchment areas anywhere in Romania and propagates
detected missions to stores without transaction data.

This README covers setup and how to run things. For the actual research
findings, methodology decisions, and dead ends already ruled out, see `docs/`
— it's organized by method (`docs/topic_missions/`, `docs/location_similarity/`,
`docs/bigclam/`, etc.), and the docs themselves are the source of truth for
"what did we conclude," not this file.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # includes requirements.txt + pytest
```

`requirements.txt` alone is enough to run the pipeline; `requirements-dev.txt`
adds `pytest` for the test suite.

## Secrets

This project reads credentials from the environment only — never hardcode a
token/key in source, a script, or a committed data file. See `.env.example`
for the convention (currently no live data source needs one; every INS/
Eurostat/WorldPop/OSM pull is unauthenticated).

## Running things

Most scripts are one-off pulls or analysis runs, invoked directly:

```bash
PYTHONPATH=src python3 scripts/<script_name>.py
```

Some scripts insert `src/` onto `sys.path` themselves and can be run without
`PYTHONPATH` — check the script's own docstring for its exact "Reproduce with"
line.

## Tests

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -v
```

Covers the reusable core library (`src/shopping_missions/catchment.py`,
`location_similarity.py`) — not the one-off pull/extraction scripts, which are
inherently exploratory and re-run rarely.

## Project layout

- `src/shopping_missions/` — the reusable library (mission detection methods,
  Gower-distance/catchment pipeline, HTTP retry wrapper, shared config).
- `scripts/` — one-off pulls, builds, and validation runs. Most are
  self-documenting (module docstring explains what it does and why).
- `data/` — pulled/derived data. Large raw source files (the national OSM
  extract, the WorldPop raster) and raw POS transaction data are gitignored;
  the smaller derived JSON/CSV artifacts under `data/location_features/` are
  tracked, since they're the actual hard-won pull results.
- `docs/` — findings, one file per method/experiment.
- `tests/` — pytest suite for the core library.
