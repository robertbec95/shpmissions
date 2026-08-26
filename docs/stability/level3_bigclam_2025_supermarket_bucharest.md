# Stability test: level3_bigclam_2025_supermarket_bucharest

10 independent BigCLAM refits (7 communities each, fresh random init) on the identical fixed value-weighted PMI graph for this store (see `docs/bigclam/level3_bigclam_2025_supermarket_bucharest.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.4689 | 0.1082 | 0.2541 | 0.8121 |
| NMI | 0.4831 | 0.0900 | 0.3273 | 0.7052 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`