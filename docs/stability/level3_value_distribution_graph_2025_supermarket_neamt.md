# Stability test: level3_value_distribution_graph_2025_supermarket_neamt

20 Louvain runs on the fixed value-weighted PMI graph for this store (see `docs/graph_missions/level3_value_distribution_graph_2025_supermarket_neamt.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8705 | 0.0945 | 0.6995 | 1.0000 |
| NMI | 0.8908 | 0.0775 | 0.7504 | 1.0000 |

Communities found per run: [10, 10, 9, 10, 10, 9, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`