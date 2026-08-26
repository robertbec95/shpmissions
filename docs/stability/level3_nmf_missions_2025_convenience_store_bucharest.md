# Stability test: level3_nmf_missions_2025_convenience_store_bucharest

20 independent NMF fits (7 topics each, `init="random"`) on the identical full value matrix for this store - no basket sampling, isolates non-convex solution variance alone (see `docs/topic_missions/level3_nmf_missions_2025_convenience_store_bucharest.md` for the reference fit and full topic profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.6589 | 0.1416 | 0.4277 | 0.9263 |
| NMI | 0.7271 | 0.0937 | 0.5615 | 0.9105 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`