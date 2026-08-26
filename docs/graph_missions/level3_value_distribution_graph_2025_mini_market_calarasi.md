# Value-weighted graph community detection (2025 data): Mini Market — Calarasi

Store `65587f43fae8eed7bbc91c04`, Mini Market, Calarasi. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` (value-weighted PMI graph, min_lift=1.0, min_prevalence=0.0001, Louvain resolution=2.0), run for this store alone. 101,155 baskets -> 69 graph nodes, 351 edges -> 7 communities. Run in 4.4s.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 13.39% | 13.12 | 5.27 | 8 | PACKED SAUSAGES (14%, x3.9), CHEESE PACKED (11%, x3.8), BULK SAUSAGES (9%, x4.4), SUPPLIED BAKERY (6%, x2.4), CANNED MEAT + FISH + READY MEALS (6%, x3.8) |
| 1 | 11.14% | 11.99 | 4.05 | 9 | BEER (23%, x4.7), DESSERT DRINKS (9%, x4.0), SPIRIT DRINKS (7%, x7.1), CHOCOLATE (6%, x3.2), WINE (5%, x6.2) |
| 2 | 10.56% | 10.2 | 4.65 | 8 | FRUITS (14%, x4.0), MILK / CREAM (13%, x4.2), YOGURT  / SANA / KEFIR, DESSERT (11%, x4.2), DESSERT INGREDIENTS (6%, x6.0), EGGS (3%, x4.2) |
| 3 | 11.9% | 16.39 | 5.13 | 10 | PORK (27%, x5.2), CHICKEN MEAT (17%, x4.0), FROZEN READY TO COOK (3%, x3.6), FRUIT AND VEGETABLES (2%, x3.1), PREPARED MEAT (2%, x4.1) |
| 4 | 6.87% | 15.39 | 5.05 | 10 | PET  FOOD (17%, x6.7), COSMETICS (9%, x5.8), DETERGENTS / BLEACHES / FABRIC SOFTENER (9%, x7.7), HOUSEHOLD PRODUCTS (6%, x3.6), HOUSHOLD PAPER (4%, x4.1) |
| 5 | 8.16% | 11.98 | 4.94 | 8 | OIL, VINEGAR (17%, x6.7), VEGETABLES (13%, x3.9), SUGAR (6%, x7.7), RICE / BEANS / PASTA (6%, x4.5), SAUCES (4%, x5.6) |
| 6 | 37.99% | 10.9 | 4.15 | 16 | TOBACCO (21%, x2.7), CARBONATED DRINKS (9%, x1.9), WATER (8%, x1.9), SALTY SNACKS (7%, x1.9), SWEET SNACKS (6%, x1.9) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8152 | 0.1423 | 0.5903 | 0.9999 |
| NMI | 0.8872 | 0.0744 | 0.7513 | 0.9995 |

Communities found per run: [7, 7, 8, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 8, 8, 7, 8, 7, 7, 7]

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 11.7 | 10.4 |  7.4 |  5.4 |  4.2 |  2   | 59   |
| early_morning | 12.5 |  0   | 12.5 |  0   |  0   |  0   | 75   |
| morning       | 14.2 |  9.2 | 11.1 | 12.6 |  7   |  9.6 | 36.3 |
| afternoon     | 14.6 | 10.7 | 11.5 | 14.4 |  7.5 | 10.5 | 30.8 |
| evening       | 12.6 | 12.7 | 10.4 | 10.7 |  6.9 |  6.8 | 39.9 |
| late_evening  | 11.4 | 11.2 |  8.6 |  8.1 |  5.6 |  4.5 | 50.6 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`