# BigCLAM overlapping community detection (2025 data): Convenience Store — Brasov

Store `65587f43fae8eed7bbc91cd5`, Convenience Store, Brasov. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same value-weighted PMI graph as `docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift=1.0, min_prevalence=0.0001), fit with BigCLAM instead of Louvain - 7 communities (log-likelihood elbow), relative threshold=0.5. 66,847 baskets -> 67 graph nodes, 424 edges -> 7 communities covering 66/67 categories. Run in 4.1s.

## Community-count selection (log-likelihood scree)

| C | Log-likelihood |
|---|---|
| 4 | -628.6 |
| 5 | -600.8 |
| 6 | -573.8 |
| 7 | -550.4 <- elbow |
| 8 | -537.9 |

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 7.47% | 7.05 | 3.54 | 16 | FRUITS (15%, x4.4), OIL, VINEGAR (8%, x11.6), VEGETABLES (5%, x2.0), DESSERT INGREDIENTS (3%, x7.0), FRUIT AND VEGETABLES (3%, x6.6) |
| 1 | 28.82% | 8.74 | 3.96 | 15 | PACKED SAUSAGES (9%, x2.7), SUPPLIED BAKERY (8%, x2.4), CHEESE PACKED (8%, x2.8), VEGETABLES (6%, x2.3), MILK / CREAM (5%, x2.5) |
| 2 | 49.18% | 6.56 | 2.83 | 16 | BEER (9%, x1.3), SALTY SNACKS (8%, x1.7), CARBONATED DRINKS (7%, x1.7), CHOCOLATE (6%, x1.8), SWEET SNACKS (5%, x1.8) |
| 3 | 1.94% | 11.38 | 3.46 | 11 | SPIRIT DRINKS (15%, x4.6), FROZEN READY TO COOK (12%, x28.3), CAKES (9%, x20.2), FROZEN MEAT (7%, x29.3), PORK (3%, x14.5) |
| 4 | 6.54% | 10.49 | 3.67 | 16 | COSMETICS (13%, x8.2), BEER (12%, x1.7), HOUSHOLD PAPER (7%, x8.6), DETERGENTS / BLEACHES / FABRIC SOFTENER (4%, x8.4), HOUSEHOLD PRODUCTS (4%, x4.5) |
| 5 | 3.52% | 10.24 | 2.88 | 9 | DESSERT DRINKS (28%, x11.2), WINE (15%, x7.1), PET  FOOD (4%, x8.1), YOGURT  / SANA / KEFIR, DESSERT (4%, x1.8), SUGAR (1%, x6.9) |
| 6 | 2.53% | 6.29 | 3.19 | 8 | CEREALS (18%, x22.1), FRUITS (8%, x2.2), BREAKFAST PRODUCTS (5%, x20.2), YOGURT  / SANA / KEFIR, DESSERT (5%, x2.4), DIET PRODUCTS (4%, x15.8) |

## Categories in multiple communities (the overlap CPM/Louvain cannot represent)

| Category | Communities |
|---|---|
| DRESSING | 0, 3, 4, 6 |
| DESSERT INGREDIENTS | 0, 4, 6 |
| FRUIT AND VEGETABLES | 0, 1, 3 |
| SUGAR | 0, 5, 6 |
| YOGURT  / SANA / KEFIR, DESSERT | 1, 5, 6 |
| CANNED VEGETABLES | 0, 1 |
| CLOTHING | 0, 4 |
| FRUITS | 0, 6 |
| PORK | 0, 3 |
| RICE / BEANS / PASTA | 0, 1 |
| VEGETABLES | 0, 1 |
| BREAKFAST PRODUCTS | 1, 6 |
| BUTTER / MARGARINE | 1, 5 |
| CHICKEN MEAT | 1, 3 |
| BEER | 2, 4 |
| HEALTH | 2, 4 |
| SPIRIT DRINKS | 2, 3 |
| ENTERTAINMENT | 4, 5 |
| PET  FOOD | 4, 5 |

## Stability across independent refits (fixed graph, different random init)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.5563 | 0.1459 | 0.3060 | 0.7818 |
| NMI | 0.4940 | 0.0688 | 0.3703 | 0.6378 |

## Cross-validation by day-part (hard assignment)

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  3.6 | 23.5 | 59.9 |  1.3 |  5.2 |  3.5 |  2.9 |
| early_morning |  5.3 | 21.1 | 68.4 |  0   |  5.3 |  0   |  0   |
| morning       |  5.5 | 25.7 | 52.9 |  1.3 |  5.3 |  5.3 |  4   |
| afternoon     |  9.6 | 32.6 | 44.1 |  2.1 |  6.2 |  3.3 |  2   |
| evening       |  7.7 | 28.4 | 49   |  2.2 |  7.1 |  3.3 |  2.3 |
| late_evening  |  5.5 | 26.2 | 53.3 |  1.8 |  7   |  3.3 |  2.9 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`