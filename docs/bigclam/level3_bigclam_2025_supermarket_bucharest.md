# BigCLAM overlapping community detection (2025 data): Supermarket — Bucharest

Store `65587f43fae8eed7bbc917e4`, Supermarket, Bucharest. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same value-weighted PMI graph as `docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift=1.0, min_prevalence=0.0001), fit with BigCLAM instead of Louvain - 7 communities (log-likelihood elbow), relative threshold=0.5. 301,311 baskets -> 90 graph nodes, 708 edges -> 7 communities covering 89/90 categories. Run in 9.6s.

## Community-count selection (log-likelihood scree)

| C | Log-likelihood |
|---|---|
| 4 | -1056.2 |
| 5 | -983.0 |
| 6 | -935.0 |
| 7 | -835.3 <- elbow |
| 8 | -841.0 |

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 1.26% | 17.14 | 4.83 | 12 | PRODUSE DIN CARNE (13%, x33.1), LACTATE (9%, x36.4), GEMURI, DULCETURI (6%, x41.0), PATISERIE (4%, x34.7), ZACUSCA (3%, x36.7) |
| 1 | 6.49% | 12.58 | 4.4 | 14 | COSMETICS (12%, x7.7), HOME CARE PRODUCTS (7%, x9.0), HOUSHOLD PAPER (7%, x6.6), DETERGENTS / BLEACHES / FABRIC SOFTENER (6%, x8.4), PET  FOOD (5%, x6.4) |
| 2 | 12.16% | 6.81 | 3.31 | 11 | COFFEE CORNER (12%, x7.0), BULK SAUSAGES (9%, x4.7), OTHERS FIIGO (8%, x2.2), COOKED MEALS (8%, x9.6), BAKE OFF (7%, x4.8) |
| 3 | 33.17% | 7.14 | 3.28 | 14 | CARBONATED DRINKS (9%, x2.5), SALTY SNACKS (8%, x2.5), SWEET SNACKS (7%, x2.7), WATER (7%, x2.4), CHOCOLATE (6%, x2.7) |
| 4 | 23.11% | 11.45 | 4.71 | 20 | CHEESE PACKED (10%, x3.1), FRUITS (8%, x1.7), SUPPLIED BAKERY (7%, x2.2), PACKED SAUSAGES (6%, x2.5), MILK / CREAM (6%, x2.5) |
| 5 | 4.51% | 12.63 | 2.99 | 15 | WINE (21%, x9.0), SPIRIT DRINKS (13%, x10.1), BEER (11%, x2.9), DESSERT DRINKS (5%, x2.2), SEASONS (3%, x10.9) |
| 6 | 19.29% | 13.29 | 5.2 | 24 | VEGETABLES (13%, x2.9), CHICKEN MEAT (8%, x3.6), PORK (8%, x3.7), BEER (3%, x0.8), EGGS (3%, x2.1) |

## Categories in multiple communities (the overlap CPM/Louvain cannot represent)

| Category | Communities |
|---|---|
| FISH | 0, 4, 6 |
| DESSERT DRINKS | 1, 4, 5 |
| CHEESE BULK | 2, 4, 6 |
| INTERNATIONAL CUISINE | 0, 6 |
| PATISERIE | 0, 5 |
| TURKEY MEAT | 0, 6 |
| BABY FOOD + NONFOOD | 1, 5 |
| CAR / BRICOLAGE | 1, 3 |
| CLOTHING | 1, 5 |
| SEASONS | 1, 5 |
| SUGAR | 1, 4 |
| DRESSING | 2, 6 |
| CHOCOLATE | 3, 5 |
| CANNED MEAT + FISH + READY MEALS | 4, 6 |
| EGGS | 4, 6 |
| OTHER PACKAGED PRODUCTS  FRESH | 4, 6 |
| RICE / BEANS / PASTA | 4, 6 |
| BEER | 5, 6 |

## Stability across independent refits (fixed graph, different random init)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.4689 | 0.1082 | 0.2541 | 0.8121 |
| NMI | 0.4831 | 0.0900 | 0.3273 | 0.7052 |

## Cross-validation by day-part (hard assignment)

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  0.9 |  6.4 |  2.9 | 53.3 | 19.2 |  7.3 |  9.9 |
| early_morning |  1   |  6.1 |  7.6 | 45.3 | 21.1 | 10.6 |  8.2 |
| morning       |  1   |  5.8 | 16.5 | 34.6 | 24.8 |  3.6 | 13.8 |
| afternoon     |  1.3 |  6.1 | 19.6 | 24.4 | 22.9 |  3   | 22.7 |
| evening       |  1.5 |  6.9 |  9.5 | 30.9 | 23.8 |  4.2 | 23.2 |
| late_evening  |  1.4 |  7.3 |  5.3 | 36.1 | 24.9 |  5.5 | 19.5 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`