# BigCLAM overlapping community detection (2025 data): Convenience Store — Dolj

Store `65587f43fae8eed7bbc91c35`, Convenience Store, Dolj. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same value-weighted PMI graph as `docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift=1.0, min_prevalence=0.0001), fit with BigCLAM instead of Louvain - 5 communities (log-likelihood elbow), relative threshold=0.5. 127,899 baskets -> 67 graph nodes, 313 edges -> 5 communities covering 66/67 categories. Run in 4.8s.

## Community-count selection (log-likelihood scree)

| C | Log-likelihood |
|---|---|
| 4 | -489.2 |
| 5 | -453.1 <- elbow |
| 6 | -441.8 |
| 7 | -407.8 |
| 8 | -387.9 |

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 27.73% | 5.22 | 2.72 | 13 | WATER (10%, x2.6), CARBONATED DRINKS (9%, x2.1), YOGURT  / SANA / KEFIR, DESSERT (9%, x2.5), BAKE OFF (6%, x3.0), COFFEE CORNER (6%, x4.1) |
| 1 | 22.86% | 8.99 | 4.07 | 20 | CHEESE PACKED (13%, x2.8), PACKED SAUSAGES (10%, x2.6), MILK / CREAM (7%, x2.4), SUPPLIED BAKERY (6%, x1.9), EGGS (4%, x3.1) |
| 2 | 25.99% | 7.63 | 3.14 | 20 | CHOCOLATE (10%, x2.9), BEER (9%, x2.2), SALTY SNACKS (9%, x2.2), SWEET SNACKS (7%, x2.2), WINE (4%, x3.1) |
| 3 | 16.31% | 8.75 | 4.12 | 22 | VEGETABLES (12%, x4.3), FRUITS (9%, x2.1), CHICKEN MEAT (6%, x4.7), FRUIT AND VEGETABLES (3%, x4.1), CANNED VEGETABLES (2%, x4.0) |
| 4 | 7.1% | 10.09 | 4.1 | 10 | COSMETICS (14%, x9.0), HOUSHOLD PAPER (7%, x7.0), HOME CARE PRODUCTS (7%, x9.1), DETERGENTS / BLEACHES / FABRIC SOFTENER (5%, x9.9), HOUSEHOLD PRODUCTS (4%, x4.4) |

## Categories in multiple communities (the overlap CPM/Louvain cannot represent)

| Category | Communities |
|---|---|
| CEREALS | 0, 1, 2 |
| FISH PRODUCTS PACKED | 1, 2, 3 |
| CLOTHING | 2, 3, 4 |
| DIET PRODUCTS | 0, 3 |
| ELECTRICAL | 0, 4 |
| HEALTH | 0, 4 |
| ICE TEA | 0, 2 |
| CANNED FRUITS | 1, 3 |
| CANNED MEAT + FISH + READY MEALS | 1, 3 |
| HERBS AND SPICES | 1, 3 |
| OIL, VINEGAR | 1, 3 |
| OTHER PACKAGED PRODUCTS  FRESH | 1, 3 |
| RICE / BEANS / PASTA | 1, 3 |
| SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU | 1, 3 |
| SAUCES | 1, 3 |
| PREPARED MEAT | 2, 3 |

## Stability across independent refits (fixed graph, different random init)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.3828 | 0.0844 | 0.2103 | 0.5423 |
| NMI | 0.4126 | 0.0614 | 0.3082 | 0.5595 |

## Cross-validation by day-part (hard assignment)

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |
|:--------------|-----:|-----:|-----:|-----:|-----:|
| night         | 23   | 20.8 | 35.1 | 13   |  8   |
| early_morning | 53.6 | 10.7 | 28.6 |  7.1 |  0   |
| morning       | 44.8 | 18.1 | 19.7 | 12.3 |  5.1 |
| afternoon     | 29.9 | 23.9 | 22.2 | 17.1 |  6.9 |
| evening       | 23.5 | 23.7 | 26.7 | 18.6 |  7.5 |
| late_evening  | 21.9 | 23.3 | 32.1 | 14.9 |  7.8 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`