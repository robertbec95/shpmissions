# BigCLAM overlapping community detection (2025 data): Supermarket — Neamt

Store `65587f43fae8eed7bbc917ff`, Supermarket, Neamt. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same value-weighted PMI graph as `docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift=1.0, min_prevalence=0.0001), fit with BigCLAM instead of Louvain - 7 communities (log-likelihood elbow), relative threshold=0.5. 396,449 baskets -> 77 graph nodes, 387 edges -> 7 communities covering 77/77 categories. Run in 9.9s.

## Community-count selection (log-likelihood scree)

| C | Log-likelihood |
|---|---|
| 4 | -646.4 |
| 5 | -602.7 |
| 6 | -551.9 |
| 7 | -509.2 <- elbow |
| 8 | -491.6 |

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 31.65% | 7.15 | 3.42 | 14 | TOBACCO (16%, x1.8), SALTY SNACKS (11%, x2.5), SWEET SNACKS (9%, x2.4), CARBONATED DRINKS (8%, x2.5), WATER (4%, x1.9) |
| 1 | 15.32% | 11.14 | 4.58 | 13 | CHICKEN MEAT (10%, x3.3), PORK (10%, x4.1), CHEESE PACKED (9%, x3.4), PACKED SAUSAGES (8%, x3.0), BULK SAUSAGES (6%, x2.8) |
| 2 | 9.67% | 6.61 | 3.29 | 9 | OTHERS FIIGO (21%, x9.1), COFFEE CORNER (8%, x10.9), BAKE OFF (6%, x5.7), COOKED MEALS (4%, x11.7), CAKES (3%, x3.6) |
| 3 | 20.87% | 7.72 | 4.06 | 15 | VEGETABLES (15%, x3.9), SUPPLIED BAKERY (10%, x2.4), FRUITS (10%, x1.9), CANNED MEAT + FISH + READY MEALS (4%, x3.3), CANNED VEGETABLES (3%, x3.9) |
| 4 | 6.57% | 10.45 | 3.14 | 14 | DESSERT DRINKS (14%, x4.7), CHOCOLATE (14%, x4.9), SPIRIT DRINKS (13%, x8.1), WINE (13%, x5.3), BEER (11%, x1.9) |
| 5 | 5.87% | 10.42 | 4.06 | 14 | COSMETICS (15%, x9.5), PET  FOOD (6%, x4.9), HOUSHOLD PAPER (6%, x7.9), HOME CARE PRODUCTS (5%, x10.8), HOUSEHOLD PRODUCTS (5%, x4.9) |
| 6 | 10.04% | 8.37 | 4.25 | 15 | OIL, VINEGAR (10%, x6.6), DESSERT INGREDIENTS (7%, x7.8), MILK / CREAM (6%, x2.6), RICE / BEANS / PASTA (6%, x5.2), SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU (4%, x5.1) |

## Categories in multiple communities (the overlap CPM/Louvain cannot represent)

| Category | Communities |
|---|---|
| FLOWERS + GARDEN ARTICLES | 3, 4, 6 |
| DIET PRODUCTS | 0, 1 |
| HEALTH | 0, 5 |
| EGGS | 1, 6 |
| FISH PRODUCTS PACKED | 1, 3 |
| MILK / CREAM | 1, 6 |
| OTHER PACKED PRODUCTS | 2, 3 |
| CANNED FRUITS | 3, 5 |
| DRESSING | 3, 6 |
| FROZEN READY TO COOK | 3, 4 |
| FRUIT AND VEGETABLES | 3, 6 |
| OTHER PACKAGED PRODUCTS  FRESH | 3, 6 |
| BABY FOOD + NONFOOD | 4, 5 |
| COSMETICS | 4, 5 |
| SEASONS | 4, 5 |
| INTERNATIONAL CUISINE | 5, 6 |

## Stability across independent refits (fixed graph, different random init)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.4688 | 0.0990 | 0.2607 | 0.6659 |
| NMI | 0.4628 | 0.0749 | 0.2994 | 0.6013 |

## Cross-validation by day-part (hard assignment)

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 52.3 | 11.1 |  8.6 |  9.6 |  8.4 |  6.1 |  3.9 |
| early_morning | 41.1 | 11   | 19.8 | 10.5 |  8.6 |  5.2 |  3.7 |
| morning       | 25.6 | 15.3 | 14   | 22.2 |  6.7 |  5.6 | 10.6 |
| afternoon     | 21.9 | 17.3 |  9.6 | 26.5 |  5.9 |  5.6 | 13.3 |
| evening       | 30.2 | 16   |  8.5 | 22.2 |  6.3 |  6   | 10.7 |
| late_evening  | 41.1 | 13.9 |  8.5 | 16.1 |  6.6 |  6.4 |  7.4 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`