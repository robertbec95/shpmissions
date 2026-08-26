# Value-weighted graph community detection (2025 data): Supermarket — Neamt

Store `65587f43fae8eed7bbc917ff`, Supermarket, Neamt. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` (value-weighted PMI graph, min_lift=1.0, min_prevalence=0.0001, Louvain resolution=2.0), run for this store alone. 396,449 baskets -> 77 graph nodes, 387 edges -> 10 communities. Run in 17.4s.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 1.98% | 7.3 | 3.74 | 5 | CANNED MEAT + FISH + READY MEALS (26%, x20.0), CANNED VEGETABLES (14%, x19.4), SAUCES (8%, x15.2), INTERNATIONAL CUISINE (2%, x27.0), CANNED FRUITS (2%, x18.3) |
| 1 | 0.91% | 7.41 | 3.23 | 4 | CAKES (46%, x50.1), HEALTH (4%, x63.0), ELECTRICAL (4%, x69.8), CAR / BRICOLAGE (0%, x33.1) |
| 2 | 26.37% | 8.26 | 4.28 | 10 | FRUITS (13%, x2.4), VEGETABLES (9%, x2.4), SUPPLIED BAKERY (9%, x2.1), CHEESE PACKED (8%, x2.8), PACKED SAUSAGES (7%, x2.7) |
| 3 | 31.78% | 8.03 | 3.6 | 14 | TOBACCO (26%, x2.9), SALTY SNACKS (9%, x2.1), SWEET SNACKS (8%, x2.0), CARBONATED DRINKS (7%, x2.1), WATER (4%, x1.8) |
| 4 | 5.78% | 5.61 | 3.06 | 7 | OTHERS FIIGO (28%, x12.3), COFFEE CORNER (12%, x15.7), COOKED MEALS (6%, x19.5), BAKE OFF (6%, x6.2), MEAT (5%, x19.7) |
| 5 | 4.36% | 10.17 | 3.98 | 8 | COSMETICS (18%, x11.6), PET  FOOD (12%, x9.6), DETERGENTS / BLEACHES / FABRIC SOFTENER (7%, x13.5), HOUSHOLD PAPER (6%, x7.8), HOME CARE PRODUCTS (5%, x10.5) |
| 6 | 0.66% | 9.71 | 3.9 | 3 | FROZEN READY TO COOK (21%, x71.0), FRUIT AND VEGETABLES (16%, x40.1), FROZEN MEAT (16%, x77.4) |
| 7 | 14.04% | 9.44 | 3.44 | 10 | BEER (20%, x3.6), WINE (12%, x4.9), DESSERT DRINKS (11%, x3.8), SPIRIT DRINKS (9%, x5.6), CHOCOLATE (9%, x3.2) |
| 8 | 5.07% | 11.56 | 4.29 | 5 | PORK (28%, x11.3), CHICKEN MEAT (26%, x8.7), DRESSING (2%, x3.4), MINCED MEAT (1%, x6.7), PREPARED MEAT (1%, x8.2) |
| 9 | 9.04% | 7.44 | 4.04 | 11 | MILK / CREAM (11%, x4.7), OIL, VINEGAR (11%, x6.9), EGGS (7%, x5.4), DESSERT INGREDIENTS (6%, x7.3), RICE / BEANS / PASTA (6%, x5.1) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8705 | 0.0945 | 0.6995 | 1.0000 |
| NMI | 0.8908 | 0.0775 | 0.7504 | 1.0000 |

Communities found per run: [10, 10, 9, 10, 10, 9, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  1.3 |  1.5 | 15.7 | 47.5 |  4.7 |  4.2 |  0.7 | 19.2 |  1.3 |  3.9 |
| early_morning |  2.3 |  0.4 | 17.2 | 45.6 | 11   |  4.1 |  0.2 | 15.4 |  0.6 |  3.3 |
| morning       |  2.1 |  0.3 | 28.2 | 29.5 |  8   |  4.4 |  0.4 | 12.3 |  5.1 |  9.7 |
| afternoon     |  2.2 |  0.7 | 31.7 | 23.4 |  6.2 |  4.3 |  0.7 | 12   |  7   | 11.9 |
| evening       |  2.1 |  1   | 27.3 | 30.3 |  5   |  4.4 |  0.7 | 14   |  5.8 |  9.4 |
| late_evening  |  1.8 |  1.2 | 22.2 | 39.3 |  4.8 |  4.5 |  0.7 | 15.5 |  3.3 |  6.7 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`