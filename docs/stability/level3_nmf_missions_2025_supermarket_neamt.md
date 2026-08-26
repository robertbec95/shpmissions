# Stability test: level3_nmf_missions_2025_supermarket_neamt

20 independent NMF fits (5 topics each, `init="random"`) on the identical full value matrix for this store - no basket sampling, isolates non-convex solution variance alone (see `docs/topic_missions/level3_nmf_missions_2025_supermarket_neamt.md` for the reference fit and full topic profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8619 | 0.0734 | 0.6417 | 0.9819 |
| NMI | 0.8631 | 0.0476 | 0.7359 | 0.9634 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`