# Stability test: level3_value_distribution_graph_8day

20 Louvain runs on the fixed value-weighted PMI graph (see `docs/graph_missions/level3_value_distribution_graph_8day.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9664 | 0.0478 | 0.8129 | 1.0000 |
| NMI | 0.9684 | 0.0420 | 0.8375 | 1.0000 |

Communities found per run: [10, 10, 10, 10, 9, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10]

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions.py --name level3_value_distribution_graph_8day --dates 20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511`