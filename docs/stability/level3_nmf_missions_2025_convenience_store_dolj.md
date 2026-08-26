# Stability test: level3_nmf_missions_2025_convenience_store_dolj

20 independent NMF fits (6 topics each, `init="random"`) on the identical full value matrix for this store - no basket sampling, isolates non-convex solution variance alone (see `docs/topic_missions/level3_nmf_missions_2025_convenience_store_dolj.md` for the reference fit and full topic profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8408 | 0.1096 | 0.5097 | 0.9662 |
| NMI | 0.8359 | 0.0707 | 0.6085 | 0.9422 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`