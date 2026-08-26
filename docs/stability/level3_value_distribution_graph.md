# Stability test: level3_value_distribution_graph

20 Louvain runs on the fixed value-weighted PMI graph (see `docs/graph_missions/level3_value_distribution_graph.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9622 | 0.0722 | 0.7227 | 1.0000 |
| NMI | 0.9760 | 0.0377 | 0.8419 | 1.0000 |

Communities found per run: [10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions.py`