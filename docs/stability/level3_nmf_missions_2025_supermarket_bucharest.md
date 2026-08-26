# Stability test: level3_nmf_missions_2025_supermarket_bucharest

20 independent NMF fits (5 topics each, `init="random"`) on the identical full value matrix for this store - no basket sampling, isolates non-convex solution variance alone (see `docs/topic_missions/level3_nmf_missions_2025_supermarket_bucharest.md` for the reference fit and full topic profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.7090 | 0.1658 | 0.3973 | 0.9866 |
| NMI | 0.7459 | 0.1195 | 0.4930 | 0.9797 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`