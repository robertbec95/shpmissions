# Stability test: level3_nmf_missions_2025_mini_market_calarasi

20 independent NMF fits (7 topics each, `init="random"`) on the identical full value matrix for this store - no basket sampling, isolates non-convex solution variance alone (see `docs/topic_missions/level3_nmf_missions_2025_mini_market_calarasi.md` for the reference fit and full topic profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.7336 | 0.1107 | 0.5257 | 0.9448 |
| NMI | 0.7600 | 0.0764 | 0.6034 | 0.9223 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`