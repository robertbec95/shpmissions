# Stability test: level3_bigclam_2025_mini_market_calarasi

10 independent BigCLAM refits (7 communities each, fresh random init) on the identical fixed value-weighted PMI graph for this store (see `docs/bigclam/level3_bigclam_2025_mini_market_calarasi.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.6273 | 0.0674 | 0.5016 | 0.7799 |
| NMI | 0.5483 | 0.0588 | 0.4459 | 0.6840 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`