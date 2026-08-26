# Graph community detection: level3_baseline

Category co-occurrence graph (PMI edges, min_lift=2.0) built from the exact full 2,364,042-basket population (99 categories -> 99 graph nodes, 1006 positive-lift edges). Louvain community detection (resolution=1.0) found 6 communities at the reference seed; every basket assigned by which community's categories it contains the most of. Run in 120.5s total.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories (by support) | Degenerate? |
|---|---|---|---|---|---|---|
| 0 | 6.47% | 22.26 | 2.56 | 1 | SUPPLIED BAKERY (100%) | **YES** |
| 1 | 14.15% | 23.33 | 2.73 | 23 | FRUITS (43%), COFFEE CORNER (26%), BAKE OFF (26%), OTHERS FIIGO (13%), ICE TEA (8%) | no |
| 2 | 43.03% | 37.88 | 3.39 | 18 | CARBONATED DRINKS (32%), SALTY SNACKS (29%), TOBACCO (28%), WATER (27%), BEER (25%) | no |
| 3 | 0.29% | 16.24 | 2.08 | 1 | OTHER DRINKS (100%) | **YES** |
| 4 | 3.03% | 44.23 | 3.73 | 22 | DESSERT DRINKS (31%), COSMETICS (29%), HOUSHOLD PAPER (23%), NONCARBONATED DRINKS (21%), SUGAR CONFECTIONERY (19%) | no |
| 5 | 33.03% | 55.91 | 5.26 | 34 | HOUSEHOLD PRODUCTS (68%), VEGETABLES (48%), MILK / CREAM (25%), YOGURT  / SANA / KEFIR, DESSERT (21%), PACKED SAUSAGES (20%) | no |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

20 independent community-detection runs on the *same* PMI graph (built once from the full population), each with a different Louvain seed. Unlike the CLARA k-medoids and HDBSCAN stability tests (docs/stability/*.md), no basket sampling happens here at all - this isolates how much Louvain's own internal randomness (traversal order, tie-breaking) affects the result, as opposed to which baskets happened to get sampled.

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.5565 | 0.1993 | 0.0603 | 1.0000 |
| NMI | 0.6121 | 0.1373 | 0.3157 | 1.0000 |

Communities found per run: [6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6]

Rule of thumb (same as the other stability reports): ARI/NMI > 0.9 = highly stable, 0.7-0.9 = reasonably stable, 0.5-0.7 = moderate instability worth investigating, < 0.5 = largely re-deriving noise.

Reproduce with: `PYTHONPATH=src python3 scripts/run_graph_missions.py level3_baseline --n-runs 20`