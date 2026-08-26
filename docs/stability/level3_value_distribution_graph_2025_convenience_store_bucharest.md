# Stability test: level3_value_distribution_graph_2025_convenience_store_bucharest

20 Louvain runs on the fixed value-weighted PMI graph for this store (see `docs/graph_missions/level3_value_distribution_graph_2025_convenience_store_bucharest.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8462 | 0.1367 | 0.5075 | 0.9999 |
| NMI | 0.8471 | 0.1221 | 0.6100 | 0.9995 |

Communities found per run: [8, 8, 10, 8, 9, 8, 8, 8, 10, 8, 10, 8, 8, 8, 8, 7, 9, 10, 8, 10]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`