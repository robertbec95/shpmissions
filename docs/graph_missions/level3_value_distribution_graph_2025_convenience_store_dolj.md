# Value-weighted graph community detection (2025 data): Convenience Store — Dolj

Store `65587f43fae8eed7bbc91c35`, Convenience Store, Dolj. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` (value-weighted PMI graph, min_lift=1.0, min_prevalence=0.0001, Louvain resolution=2.0), run for this store alone. 127,899 baskets -> 67 graph nodes, 313 edges -> 10 communities. Run in 5.6s.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 16.23% | 5.71 | 3.26 | 9 | SALTY SNACKS (17%, x4.4), WATER (14%, x3.6), SWEET SNACKS (14%, x4.3), NONCARBONATED DRINKS (5%, x4.1), SUGAR CONFECTIONERY (4%, x4.6) |
| 1 | 4.44% | 6.69 | 3.58 | 6 | MILK / CREAM (25%, x9.1), BUTTER / MARGARINE (8%, x11.0), DESSERT INGREDIENTS (6%, x12.2), CEREALS (6%, x10.1), BREAKFAST PRODUCTS (5%, x13.8) |
| 2 | 16.37% | 8.69 | 3.41 | 12 | BEER (14%, x3.3), CHOCOLATE (10%, x3.0), ICE CREAM AND DESSERTS (9%, x3.7), DESSERT DRINKS (8%, x3.5), WINE (6%, x4.5) |
| 3 | 13.38% | 9.0 | 4.25 | 7 | CHEESE PACKED (19%, x4.2), PACKED SAUSAGES (16%, x4.2), SUPPLIED BAKERY (10%, x2.8), CANNED MEAT + FISH + READY MEALS (4%, x4.2), FISH PRODUCTS PACKED (2%, x4.2) |
| 4 | 6.42% | 8.52 | 3.98 | 9 | VEGETABLES (16%, x5.5), CHICKEN MEAT (13%, x9.3), FRUIT AND VEGETABLES (7%, x8.2), FROZEN READY TO COOK (5%, x9.3), PET  FOOD (4%, x6.5) |
| 5 | 3.78% | 9.59 | 3.94 | 7 | COSMETICS (20%, x13.1), HOUSHOLD PAPER (10%, x9.9), HOME CARE PRODUCTS (9%, x12.8), DETERGENTS / BLEACHES / FABRIC SOFTENER (8%, x16.2), HOUSEHOLD PRODUCTS (5%, x4.8) |
| 6 | 21.96% | 4.73 | 3.02 | 8 | FRUITS (15%, x3.5), CARBONATED DRINKS (14%, x3.3), YOGURT  / SANA / KEFIR, DESSERT (12%, x3.5), BAKE OFF (7%, x3.3), COFFEE CORNER (7%, x4.8) |
| 7 | 0.69% | 5.82 | 3.42 | 3 | SAUCES (24%, x55.6), SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU (15%, x43.5), HERBS AND SPICES (12%, x43.1) |
| 8 | 2.66% | 6.57 | 3.55 | 5 | OIL, VINEGAR (20%, x22.0), EGGS (19%, x15.4), RICE / BEANS / PASTA (11%, x13.2), SUGAR (5%, x16.8), OTHER PACKAGED PRODUCTS  FRESH (0%, x11.4) |
| 9 | 14.05% | 11.49 | 3.28 | 1 | TOBACCO (66%, x4.3) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.7469 | 0.1501 | 0.5151 | 0.9996 |
| NMI | 0.8270 | 0.0898 | 0.6850 | 0.9987 |

Communities found per run: [10, 9, 9, 10, 9, 9, 10, 10, 10, 9, 10, 10, 10, 10, 9, 10, 8, 10, 10, 10]

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 21.1 |  5.1 | 18.6 | 11.5 |  5.7 |  4.4 | 16.9 |  0.4 |  1.9 | 14.2 |
| early_morning | 14.3 |  0   | 21.4 |  3.6 |  7.1 |  0   | 10.7 |  0   |  0   | 42.9 |
| morning       | 14.5 |  3.9 | 11.9 | 11.4 |  4.1 |  3   | 33.6 |  0.6 |  2   | 15   |
| afternoon     | 14.6 |  4.4 | 15.1 | 14.4 |  6.6 |  3.7 | 24   |  0.7 |  3.3 | 13.1 |
| evening       | 16   |  4.5 | 17.3 | 13.5 |  7.5 |  4   | 19.5 |  0.8 |  2.7 | 14.3 |
| late_evening  | 18.8 |  4.5 | 19   | 13.2 |  6.1 |  3.9 | 17.1 |  0.6 |  2.1 | 14.8 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`