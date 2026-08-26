# Stability test: level3_value_distribution_graph_2025_mini_market_calarasi

20 Louvain runs on the fixed value-weighted PMI graph for this store (see `docs/graph_missions/level3_value_distribution_graph_2025_mini_market_calarasi.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8152 | 0.1423 | 0.5903 | 0.9999 |
| NMI | 0.8872 | 0.0744 | 0.7513 | 0.9995 |

Communities found per run: [7, 7, 8, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 8, 8, 7, 8, 7, 7, 7]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`