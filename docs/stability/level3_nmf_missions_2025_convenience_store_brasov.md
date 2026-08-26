# Stability test: level3_nmf_missions_2025_convenience_store_brasov

20 independent NMF fits (6 topics each, `init="random"`) on the identical full value matrix for this store - no basket sampling, isolates non-convex solution variance alone (see `docs/topic_missions/level3_nmf_missions_2025_convenience_store_brasov.md` for the reference fit and full topic profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8448 | 0.0550 | 0.6906 | 0.9533 |
| NMI | 0.8333 | 0.0424 | 0.7274 | 0.9316 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`