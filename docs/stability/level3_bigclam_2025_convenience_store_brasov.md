# Stability test: level3_bigclam_2025_convenience_store_brasov

10 independent BigCLAM refits (7 communities each, fresh random init) on the identical fixed value-weighted PMI graph for this store (see `docs/bigclam/level3_bigclam_2025_convenience_store_brasov.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.5563 | 0.1459 | 0.3060 | 0.7818 |
| NMI | 0.4940 | 0.0688 | 0.3703 | 0.6378 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`