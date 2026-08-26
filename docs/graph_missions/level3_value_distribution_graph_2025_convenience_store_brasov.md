# Value-weighted graph community detection (2025 data): Convenience Store — Brasov

Store `65587f43fae8eed7bbc91cd5`, Convenience Store, Brasov. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` (value-weighted PMI graph, min_lift=1.0, min_prevalence=0.0001, Louvain resolution=2.0), run for this store alone. 66,847 baskets -> 67 graph nodes, 424 edges -> 12 communities. Run in 2.9s.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 1.18% | 7.13 | 3.61 | 4 | CANNED MEAT + FISH + READY MEALS (31%, x37.7), CANNED VEGETABLES (15%, x32.1), CANNED FRUITS (4%, x32.6), CONCENTRATED SYRUPS (2%, x35.1) |
| 1 | 29.72% | 5.22 | 3.05 | 13 | SALTY SNACKS (14%, x3.1), CARBONATED DRINKS (13%, x2.9), SWEET SNACKS (9%, x3.0), COFFEE CORNER (7%, x3.3), ICE CREAM AND DESSERTS (7%, x3.7) |
| 2 | 13.29% | 9.18 | 3.29 | 9 | BEER (27%, x3.7), SPIRIT DRINKS (18%, x5.5), COSMETICS (6%, x4.0), HOUSHOLD PAPER (3%, x3.4), DETERGENTS / BLEACHES / FABRIC SOFTENER (2%, x4.8) |
| 3 | 11.28% | 7.57 | 3.93 | 5 | PACKED SAUSAGES (18%, x5.8), CHEESE PACKED (15%, x5.5), SUPPLIED BAKERY (12%, x3.7), YOGURT  / SANA / KEFIR, DESSERT (9%, x4.3), FISH PRODUCTS PACKED (2%, x5.9) |
| 4 | 1.54% | 8.69 | 3.78 | 5 | CHICKEN MEAT (22%, x27.0), CAKES (13%, x29.5), FROZEN MEAT (9%, x40.0), PORK (5%, x27.8), MINCED MEAT (2%, x26.4) |
| 5 | 0.82% | 8.73 | 3.62 | 3 | FROZEN READY TO COOK (30%, x71.8), FRUIT AND VEGETABLES (14%, x35.8), DRESSING (9%, x26.6) |
| 6 | 6.78% | 5.02 | 3.21 | 3 | FRUITS (35%, x9.9), VEGETABLES (22%, x8.4), OTHER PACKAGED PRODUCTS  FRESH (0%, x6.3) |
| 7 | 1.29% | 6.29 | 3.42 | 5 | OIL, VINEGAR (31%, x43.3), SAUCES (8%, x27.1), SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU (7%, x25.4), HERBS AND SPICES (7%, x25.9), INTERNATIONAL CUISINE (0%, x31.7) |
| 8 | 1.61% | 2.74 | 2.35 | 2 | BAKE OFF (57%, x45.5), OTHERS FIIGO (2%, x71.4) |
| 9 | 4.74% | 8.7 | 3.18 | 8 | WINE (29%, x13.9), CHOCOLATE (24%, x7.3), SEASONS + GIFTS (3%, x11.7), ELECTRICAL (2%, x9.6), ENTERTAINMENT (1%, x9.0) |
| 10 | 8.06% | 7.54 | 3.61 | 9 | DESSERT DRINKS (20%, x8.1), MILK / CREAM (12%, x5.6), EGGS (7%, x6.4), CEREALS (4%, x4.7), DESSERT INGREDIENTS (3%, x7.5) |
| 11 | 19.7% | 11.68 | 3.19 | 1 | TOBACCO (67%, x3.0) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9664 | 0.0220 | 0.9117 | 0.9993 |
| NMI | 0.9429 | 0.0339 | 0.8694 | 0.9982 |

Communities found per run: [12, 13, 12, 12, 12, 13, 13, 12, 13, 12, 12, 12, 13, 12, 12, 12, 12, 12, 13, 14]

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |   C10 |   C11 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| night         |  0.9 | 33.7 | 17.6 | 10.6 |  0.4 |  0.9 |  2.9 |  0.5 |  0.2 |  5.5 |   5.4 |  21.3 |
| early_morning |  0   | 26.3 |  5.3 | 10.5 |  0   |  0   |  0   |  0   |  0   |  5.3 |   5.3 |  47.4 |
| morning       |  1.4 | 26.3 |  8.4 | 11.5 |  1.1 |  0.4 |  6   |  0.8 |  5.1 |  3.1 |   9.6 |  26.1 |
| afternoon     |  1.2 | 28.5 | 10   | 12.6 |  2.1 |  0.8 |  9.5 |  1.8 |  2   |  4.1 |   9   |  18.4 |
| evening       |  1.2 | 30.1 | 15   | 10.4 |  1.5 |  0.9 |  6.3 |  1.3 |  1   |  5.2 |   7.4 |  19.7 |
| late_evening  |  1.1 | 31.8 | 16.9 | 10.6 |  1.1 |  0.9 |  4.5 |  0.9 |  0.7 |  5.6 |   7.4 |  18.4 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`