# Graph community detection: level3_8day

Category co-occurrence graph (PMI edges, min_lift=2.0) built from the exact full 6,694,099-basket population (99 categories -> 99 graph nodes, 1047 positive-lift edges). Louvain community detection (resolution=1.0) found 9 communities at the reference seed; every basket assigned by which community's categories it contains the most of. Run in 412.9s total.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories (by support) | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | 7.54% | 24.16 | 2.71 | 1 | SUPPLIED BAKERY (100%) | **YES** |
| 1 | 47.85% | 46.68 | 4.52 | 34 | VEGETABLES (37%), FRUITS (35%), YOGURT  / SANA / KEFIR, DESSERT (20%), MILK / CREAM (20%), PACKED SAUSAGES (16%) | no |
| 2 | 20.74% | 30.7 | 3.12 | 9 | CARBONATED DRINKS (54%), SALTY SNACKS (40%), HOUSEHOLD PRODUCTS (32%), ICE CREAM AND DESSERTS (22%), CEREALS (5%) | no |
| 3 | 1.11% | 20.16 | 2.25 | 1 | CHEWING GUM (100%) | **YES** |
| 4 | 0.51% | 17.74 | 2.2 | 1 | ICE TEA (100%) | **YES** |
| 5 | 1.83% | 22.44 | 2.19 | 1 | OTHER DRINKS (100%) | **YES** |
| 6 | 3.05% | 40.74 | 2.07 | 1 | TOBACCO (100%) | **YES** |
| 7 | 15.57% | 38.19 | 3.74 | 30 | WATER (44%), SWEET SNACKS (37%), BEER (30%), CHOCOLATE (25%), DESSERT DRINKS (15%) | no |
| 8 | 1.8% | 29.26 | 3.38 | 21 | BAKE OFF (59%), COFFEE CORNER (54%), OTHERS FIIGO (38%), COOKED MEALS (26%), GARNISHES (25%) | no |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

20 independent community-detection runs on the *same* PMI graph (built once from the full population), each with a different Louvain seed. Unlike the CLARA k-medoids and HDBSCAN stability tests (docs/stability/*.md), no basket sampling happens here at all - this isolates how much Louvain's own internal randomness (traversal order, tie-breaking) affects the result, as opposed to which baskets happened to get sampled.

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.6703 | 0.2143 | 0.3381 | 1.0000 |
| NMI | 0.7214 | 0.1396 | 0.4524 | 1.0000 |

Communities found per run: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

Rule of thumb (same as the other stability reports): ARI/NMI > 0.9 = highly stable, 0.7-0.9 = reasonably stable, 0.5-0.7 = moderate instability worth investigating, < 0.5 = largely re-deriving noise.

Reproduce with: `PYTHONPATH=src python3 scripts/run_graph_missions.py level3_8day --n-runs 20`