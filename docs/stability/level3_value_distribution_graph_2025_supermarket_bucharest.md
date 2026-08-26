# Stability test: level3_value_distribution_graph_2025_supermarket_bucharest

20 Louvain runs on the fixed value-weighted PMI graph for this store (see `docs/graph_missions/level3_value_distribution_graph_2025_supermarket_bucharest.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9114 | 0.0663 | 0.7793 | 0.9999 |
| NMI | 0.9171 | 0.0594 | 0.8021 | 0.9998 |

Communities found per run: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`