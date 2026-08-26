# Stability test: level3_value_distribution_graph_2025_convenience_store_dolj

20 Louvain runs on the fixed value-weighted PMI graph for this store (see `docs/graph_missions/level3_value_distribution_graph_2025_convenience_store_dolj.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.7469 | 0.1501 | 0.5151 | 0.9996 |
| NMI | 0.8270 | 0.0898 | 0.6850 | 0.9987 |

Communities found per run: [10, 9, 9, 10, 9, 9, 10, 10, 10, 9, 10, 10, 10, 10, 9, 10, 8, 10, 10, 10]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`