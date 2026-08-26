# Stability test: level3_embedding_graph

20 Louvain runs on the fixed embedding-similarity graph (see `docs/graph_missions/level3_embedding_graph.md` for the graph itself and full community profiles).

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9925 | 0.0089 | 0.9685 | 1.0000 |
| NMI | 0.9899 | 0.0112 | 0.9585 | 1.0000 |

Communities found per run: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

Reproduce with: `PYTHONPATH=src python3 scripts/run_embedding_graph_missions.py --name level3_embedding_graph --dates 20260509,20260510,20260511`