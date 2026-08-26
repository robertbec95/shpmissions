# Stability test: level3_bigclam_2025_convenience_store_dolj

10 independent BigCLAM refits (5 communities each, fresh random init) on the identical fixed value-weighted PMI graph for this store (see `docs/bigclam/level3_bigclam_2025_convenience_store_dolj.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.3828 | 0.0844 | 0.2103 | 0.5423 |
| NMI | 0.4126 | 0.0614 | 0.3082 | 0.5595 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`