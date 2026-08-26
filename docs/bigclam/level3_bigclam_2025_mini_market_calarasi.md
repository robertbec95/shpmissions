# BigCLAM overlapping community detection (2025 data): Mini Market — Calarasi

Store `65587f43fae8eed7bbc91c04`, Mini Market, Calarasi. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same value-weighted PMI graph as `docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift=1.0, min_prevalence=0.0001), fit with BigCLAM instead of Louvain - 7 communities (log-likelihood elbow), relative threshold=0.5. 101,155 baskets -> 69 graph nodes, 351 edges -> 7 communities covering 69/69 categories. Run in 4.5s.

## Community-count selection (log-likelihood scree)

| C | Log-likelihood |
|---|---|
| 4 | -549.2 |
| 5 | -525.3 |
| 6 | -498.0 |
| 7 | -462.4 <- elbow |
| 8 | -443.3 |

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 18.88% | 14.26 | 5.39 | 13 | PACKED SAUSAGES (12%, x3.2), CHEESE PACKED (10%, x3.5), BULK SAUSAGES (7%, x3.3), SUPPLIED BAKERY (5%, x1.9), CANNED MEAT + FISH + READY MEALS (5%, x3.1) |
| 1 | 9.67% | 9.14 | 3.97 | 11 | FRUITS (16%, x4.6), MILK / CREAM (15%, x5.0), YOGURT  / SANA / KEFIR, DESSERT (9%, x3.5), CEREALS (1%, x1.9), FROZEN READY TO COOK (1%, x1.3) |
| 2 | 6.73% | 15.94 | 5.11 | 11 | PET  FOOD (12%, x4.9), COSMETICS (8%, x5.3), DETERGENTS / BLEACHES / FABRIC SOFTENER (8%, x7.3), HOUSHOLD PAPER (6%, x6.0), HOME CARE PRODUCTS (5%, x7.6) |
| 3 | 13.2% | 14.75 | 4.83 | 12 | CHICKEN MEAT (20%, x4.6), PORK (18%, x3.4), VEGETABLES (7%, x2.2), FRUIT AND VEGETABLES (3%, x4.2), PREPARED MEAT (2%, x4.3) |
| 4 | 4.41% | 12.15 | 3.48 | 15 | BEER (15%, x3.1), DESSERT DRINKS (10%, x4.5), HOUSEHOLD PRODUCTS (10%, x6.0), SPIRIT DRINKS (9%, x9.2), WINE (5%, x5.9) |
| 5 | 34.96% | 9.67 | 3.9 | 16 | TOBACCO (15%, x1.9), CARBONATED DRINKS (11%, x2.3), SALTY SNACKS (9%, x2.5), SWEET SNACKS (8%, x2.3), CHOCOLATE (5%, x2.6) |
| 6 | 12.14% | 14.64 | 5.62 | 14 | OIL, VINEGAR (10%, x3.9), RICE / BEANS / PASTA (5%, x4.4), VEGETABLES (5%, x1.6), DESSERT INGREDIENTS (5%, x4.7), SUGAR (4%, x5.0) |

## Categories in multiple communities (the overlap CPM/Louvain cannot represent)

| Category | Communities |
|---|---|
| CONCENTRATED SYRUPS | 0, 2, 3, 6 |
| EGGS | 0, 1, 3, 6 |
| ELECTRICAL | 1, 2, 4, 6 |
| SEASONS + GIFTS | 1, 4, 5 |
| CANNED VEGETABLES | 0, 6 |
| CHEESE BULK | 0, 4 |
| WATER | 0, 5 |
| YOGURT  / SANA / KEFIR, DESSERT | 0, 1 |
| CEREALS | 1, 5 |
| DIET PRODUCTS | 1, 5 |
| FROZEN READY TO COOK | 1, 3 |
| COSMETICS | 2, 4 |
| DETERGENTS / BLEACHES / FABRIC SOFTENER | 2, 4 |
| HOUSEHOLD PRODUCTS | 2, 4 |
| SAUCES | 3, 6 |
| VEGETABLES | 3, 6 |

## Stability across independent refits (fixed graph, different random init)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.6273 | 0.0674 | 0.5016 | 0.7799 |
| NMI | 0.5483 | 0.0588 | 0.4459 | 0.6840 |

## Cross-validation by day-part (hard assignment)

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 16.6 |  7.7 |  4.3 |  6.3 |  4.5 | 57   |  3.5 |
| early_morning | 25   | 12.5 |  0   |  0   |  0   | 62.5 |  0   |
| morning       | 19.5 |  9.6 |  6.7 | 13.7 |  3.9 | 32.6 | 13.9 |
| afternoon     | 20.2 | 10.2 |  7.3 | 15.5 |  4.2 | 26.9 | 15.6 |
| evening       | 18.1 |  9.9 |  6.8 | 12.5 |  5   | 37.6 | 10.1 |
| late_evening  | 16.7 |  8.4 |  5.7 |  9.3 |  4.2 | 49   |  6.8 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`