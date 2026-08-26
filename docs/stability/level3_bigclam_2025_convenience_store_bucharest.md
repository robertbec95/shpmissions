# Stability test: level3_bigclam_2025_convenience_store_bucharest

10 independent BigCLAM refits (6 communities each, fresh random init) on the identical fixed value-weighted PMI graph for this store (see `docs/bigclam/level3_bigclam_2025_convenience_store_bucharest.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.5312 | 0.0861 | 0.3694 | 0.7969 |
| NMI | 0.5174 | 0.0647 | 0.3991 | 0.7260 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`