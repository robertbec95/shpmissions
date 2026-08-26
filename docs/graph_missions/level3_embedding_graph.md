# Item2Vec embedding similarity graph: level3_embedding_graph

Dates: 20260509,20260510,20260511 (3-day window). Category embeddings via exact SVD of the value-weighted PPMI matrix (`embeddings.train_item2vec`, dim=32, 96% of PPMI variance explained - deterministic, no training/random seed), then a cosine-similarity graph (`embeddings.compute_embedding_similarity_graph`, min_cosine=0.3) in place of `graph_missions.compute_value_pmi_graph`'s direct-lift edges. 80 nodes, 274 edges, Louvain resolution=2.0 -> 10 communities at the reference seed. Run in 132.2s total.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 5.03% | 33.36 | 3.87 | 8 | MILK / CREAM (21%, x8.1), EGGS (11%, x10.2), DESSERT INGREDIENTS (6%, x10.6), BREAKFAST PRODUCTS (4%, x12.2), CEREALS (4%, x8.0) |
| 1 | 20.7% | 57.04 | 4.0 | 9 | TOBACCO (41%, x3.0), PORK (8%, x3.0), CHICKEN MEAT (8%, x2.5), FROZEN READY TO COOK (1%, x2.5), PREPARED MEAT (1%, x2.5) |
| 2 | 5.3% | 24.21 | 2.96 | 6 | COFFEE CORNER (26%, x18.4), OTHERS FIIGO (13%, x15.3), MEAT (8%, x23.6), BAKE OFF (7%, x8.8), COOKED MEALS (6%, x23.7) |
| 3 | 15.18% | 47.19 | 4.73 | 9 | PACKED SAUSAGES (12%, x3.8), CHEESE PACKED (10%, x3.7), BULK SAUSAGES (9%, x4.3), SUPPLIED BAKERY (8%, x2.4), YOGURT  / SANA / KEFIR, DESSERT (8%, x3.0) |
| 4 | 8.39% | 37.29 | 4.19 | 11 | VEGETABLES (26%, x5.5), OIL, VINEGAR (9%, x6.9), RICE / BEANS / PASTA (5%, x5.5), SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU (4%, x5.8), SAUCES (3%, x6.3) |
| 5 | 24.27% | 29.08 | 3.49 | 11 | SALTY SNACKS (13%, x3.4), CARBONATED DRINKS (13%, x3.0), SWEET SNACKS (9%, x3.1), ICE CREAM AND DESSERTS (9%, x3.8), WATER (8%, x3.0) |
| 6 | 5.08% | 36.39 | 3.08 | 2 | BEER (59%, x10.9), ENTERTAINMENT (0%, x4.0) |
| 7 | 5.8% | 49.97 | 3.7 | 6 | DESSERT DRINKS (19%, x8.1), SPIRIT DRINKS (14%, x11.1), WINE (10%, x9.9), CHOCOLATE (10%, x5.1), CAKES (4%, x7.3) |
| 8 | 10.18% | 39.35 | 3.9 | 14 | FRUITS (17%, x4.2), COSMETICS (10%, x6.2), PET  FOOD (8%, x6.5), DETERGENTS / BLEACHES / FABRIC SOFTENER (5%, x7.2), HOUSEHOLD PRODUCTS (4%, x2.8) |
| 9 | 0.08% | 53.5 | 4.25 | 4 | PRODUSE DIN CARNE (23%, x508.5), LACTATE (11%, x485.1), PATISERIE (10%, x389.2), PANIFICATIE (7%, x313.4) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

20 independent community-detection runs on the same embedding-similarity graph (built once from the full population - the embeddings themselves have no seed dependence, since SVD is exact and deterministic), each with a different Louvain seed.

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9925 | 0.0089 | 0.9685 | 1.0000 |
| NMI | 0.9899 | 0.0112 | 0.9585 | 1.0000 |

Communities found per run: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

For reference: direct-lift value-weighted graph (`docs/stability/level3_value_distribution_graph.md`) scores ARI mean=0.9622. This embedding-based variant scores higher on this population - the richer, transitive similarity signal embeddings capture appears to produce not just comparable but *more* consistent community boundaries, not only a different lens on the same structure.

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  2.2 | 24.3 |  7.6 | 10.4 |  1.9 | 32   |  8   |  8.3 |  5.1 |  0.1 |
| early_morning |  4.7 | 21.7 | 10.3 | 16.8 |  8.2 | 20   |  2.9 |  5.4 |  9.9 |  0.1 |
| morning       |  6   | 21   |  5.9 | 17.6 | 12   | 16.1 |  3.6 |  5.4 | 12.3 |  0.1 |
| afternoon     |  5.1 | 20.3 |  5.6 | 14.5 |  9.1 | 24.2 |  4.9 |  5.7 | 10.6 |  0.1 |
| evening       |  4.8 | 20.2 |  3.5 | 14.5 |  6.6 | 28.8 |  6.3 |  5.9 |  9.3 |  0.1 |
| late_evening  |  3.9 | 21.2 |  3   | 13.5 |  3.6 | 33.5 |  7.1 |  6.5 |  7.5 |  0.1 |

## Cross-validation by county (top 10 by volume)

| county    |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Timis     |  4.8 | 23.9 |  5.4 | 12.3 |  7.5 | 26.3 |  5.9 |  5.2 |  8.5 |  0.1 |
| Cluj      |  6.5 | 22.1 |  5.7 | 13.8 |  8   | 23.7 |  5.3 |  5.7 |  9.1 |  0.1 |
| Bucharest |  5.7 | 21.6 |  4.8 | 14.7 |  8   | 24.7 |  4.5 |  6   |  9.7 |  0.2 |
| Dolj      |  4.6 | 18.5 |  6.9 | 18.9 |  8.8 | 22.8 |  4.3 |  4.9 | 10.3 |  0   |
| Bihor     |  5   | 21.5 |  7.1 | 13.9 |  8.1 | 25.4 |  5   |  5.3 |  8.5 |  0.2 |
| Constanta |  5   | 20.1 |  6.6 | 14.9 |  8.2 | 24.5 |  4.4 |  6.2 | 10.1 |  0   |
| Suceava   |  4.8 | 17.1 |  4.6 | 14.9 |  7.5 | 27.7 |  5   |  7   | 11.3 |  0.2 |
| Bacau     |  5   | 20.3 |  4.2 | 14.7 |  8.4 | 24.3 |  6   |  6.7 | 10.4 |  0   |
| Iasi      |  5.6 | 18.3 |  3.7 | 16.1 |  8.6 | 25.4 |  5.2 |  6.4 | 10.7 |  0.1 |
| Arges     |  4.9 | 20.2 |  5.8 | 17.4 |  7.5 | 22.7 |  4.5 |  6   | 10.9 |  0   |

Reproduce with: `PYTHONPATH=src python3 scripts/run_embedding_graph_missions.py --name level3_embedding_graph --dates 20260509,20260510,20260511`