# Stability test: level3_bigclam_2025_supermarket_neamt

10 independent BigCLAM refits (7 communities each, fresh random init) on the identical fixed value-weighted PMI graph for this store (see `docs/bigclam/level3_bigclam_2025_supermarket_neamt.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.4688 | 0.0990 | 0.2607 | 0.6659 |
| NMI | 0.4628 | 0.0749 | 0.2994 | 0.6013 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`