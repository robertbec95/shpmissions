# Stability test: level3_value_distribution_graph_2025_convenience_store_brasov

20 Louvain runs on the fixed value-weighted PMI graph for this store (see `docs/graph_missions/level3_value_distribution_graph_2025_convenience_store_brasov.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9664 | 0.0220 | 0.9117 | 0.9993 |
| NMI | 0.9429 | 0.0339 | 0.8694 | 0.9982 |

Communities found per run: [12, 13, 12, 12, 12, 13, 13, 12, 13, 12, 12, 12, 13, 12, 12, 12, 12, 12, 13, 14]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`