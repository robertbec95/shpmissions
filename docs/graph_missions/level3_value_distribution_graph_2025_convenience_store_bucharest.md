# Value-weighted graph community detection (2025 data): Convenience Store — Bucharest

Store `65587f43fae8eed7bbc91c43`, Convenience Store, Bucharest. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` (value-weighted PMI graph, min_lift=1.0, min_prevalence=0.0001, Louvain resolution=2.0), run for this store alone. 267,631 baskets -> 68 graph nodes, 285 edges -> 8 communities. Run in 11.9s.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 2.25% | 7.72 | 3.4 | 3 | PET  FOOD (21%, x21.1), OTHERS FIIGO (17%, x21.8), CAKES (16%, x25.9) |
| 1 | 28.73% | 9.1 | 3.4 | 10 | TOBACCO (42%, x2.9), CARBONATED DRINKS (8%, x1.9), WATER (5%, x1.7), ICE CREAM AND DESSERTS (4%, x2.2), COFFEE CORNER (2%, x2.7) |
| 2 | 22.86% | 8.65 | 4.11 | 8 | CHEESE PACKED (16%, x3.0), PACKED SAUSAGES (14%, x3.1), SUPPLIED BAKERY (10%, x2.1), FRUITS (6%, x2.3), VEGETABLES (5%, x2.5) |
| 3 | 5.62% | 9.53 | 3.9 | 11 | COSMETICS (14%, x8.9), DETERGENTS / BLEACHES / FABRIC SOFTENER (10%, x11.3), HOUSHOLD PAPER (9%, x7.4), HOME CARE PRODUCTS (8%, x9.1), CEREALS (4%, x5.3) |
| 4 | 4.89% | 9.42 | 3.87 | 7 | CHICKEN MEAT (24%, x10.8), FROZEN READY TO COOK (11%, x11.8), FRUIT AND VEGETABLES (8%, x8.7), PORK (5%, x11.6), DRESSING (3%, x5.9) |
| 5 | 12.65% | 7.23 | 3.78 | 12 | MILK / CREAM (16%, x3.9), EGGS (10%, x4.6), OIL, VINEGAR (8%, x5.3), RICE / BEANS / PASTA (4%, x4.3), BUTTER / MARGARINE (4%, x4.9) |
| 6 | 0.32% | 10.72 | 3.78 | 5 | FROZEN MEAT (37%, x146.8), SEASONS (9%, x127.6), CONCENTRATED SYRUPS (7%, x105.3), FLOWERS + GARDEN ARTICLES (2%, x92.0), INTERNATIONAL CUISINE (0%, x42.4) |
| 7 | 22.68% | 8.13 | 3.62 | 12 | SALTY SNACKS (11%, x2.7), BEER (11%, x2.8), SWEET SNACKS (8%, x2.5), YOGURT  / SANA / KEFIR, DESSERT (8%, x2.2), DESSERT DRINKS (7%, x3.0) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8462 | 0.1367 | 0.5075 | 0.9999 |
| NMI | 0.8471 | 0.1221 | 0.6100 | 0.9995 |

Communities found per run: [8, 8, 10, 8, 9, 8, 8, 8, 10, 8, 10, 8, 8, 8, 8, 7, 9, 10, 8, 10]

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  2.7 | 34   | 17.6 |  5.1 |  3.5 |  5.8 |  0.3 | 31.2 |
| early_morning |  0   | 89.5 |  0   |  0   |  0   |  0   |  0   | 10.5 |
| morning       |  1.5 | 35.1 | 20.9 |  5.1 |  3.5 | 12.5 |  0.3 | 21.1 |
| afternoon     |  2   | 24.8 | 25.9 |  6.1 |  5.2 | 16   |  0.3 | 19.7 |
| evening       |  2.6 | 28.8 | 22.5 |  5.4 |  5.4 | 12.5 |  0.4 | 22.6 |
| late_evening  |  2.5 | 30.5 | 20.8 |  5.4 |  4.8 |  9.4 |  0.3 | 26.2 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`