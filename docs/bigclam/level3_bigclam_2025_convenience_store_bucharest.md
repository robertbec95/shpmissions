# BigCLAM overlapping community detection (2025 data): Convenience Store — Bucharest

Store `65587f43fae8eed7bbc91c43`, Convenience Store, Bucharest. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same value-weighted PMI graph as `docs/graph_missions/level3_value_distribution_graph_2025_*.md` (min_lift=1.0, min_prevalence=0.0001), fit with BigCLAM instead of Louvain - 6 communities (log-likelihood elbow), relative threshold=0.5. 267,631 baskets -> 68 graph nodes, 285 edges -> 6 communities covering 68/68 categories. Run in 7.3s.

## Community-count selection (log-likelihood scree)

| C | Log-likelihood |
|---|---|
| 4 | -477.3 |
| 5 | -433.9 |
| 6 | -391.6 <- elbow |
| 7 | -395.7 |
| 8 | -363.7 |

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 5.16% | 8.69 | 2.94 | 17 | CHOCOLATE (13%, x5.4), BEER (9%, x2.3), DESSERT DRINKS (9%, x3.8), SPIRIT DRINKS (8%, x14.1), WINE (8%, x6.5) |
| 1 | 15.53% | 7.55 | 3.65 | 14 | MILK / CREAM (17%, x4.3), EGGS (8%, x3.6), SUPPLIED BAKERY (7%, x1.4), FRUITS (6%, x2.3), BUTTER / MARGARINE (4%, x4.8) |
| 2 | 17.07% | 8.88 | 4.13 | 19 | CHICKEN MEAT (10%, x4.4), VEGETABLES (7%, x3.4), OIL, VINEGAR (6%, x4.0), FRUIT AND VEGETABLES (4%, x4.7), RICE / BEANS / PASTA (3%, x3.3) |
| 3 | 35.03% | 7.21 | 3.19 | 15 | TOBACCO (27%, x1.9), CARBONATED DRINKS (10%, x2.4), SALTY SNACKS (9%, x2.3), WATER (6%, x2.3), SWEET SNACKS (4%, x1.3) |
| 4 | 8.48% | 10.76 | 4.15 | 13 | COSMETICS (12%, x7.4), DETERGENTS / BLEACHES / FABRIC SOFTENER (7%, x8.1), HOME CARE PRODUCTS (7%, x7.9), HOUSHOLD PAPER (7%, x5.9), BEER (5%, x1.2) |
| 5 | 18.74% | 10.58 | 4.39 | 8 | CHEESE PACKED (18%, x3.3), PACKED SAUSAGES (14%, x3.2), YOGURT  / SANA / KEFIR, DESSERT (7%, x2.1), CANNED MEAT + FISH + READY MEALS (5%, x3.1), FISH PRODUCTS PACKED (2%, x3.4) |

## Categories in multiple communities (the overlap CPM/Louvain cannot represent)

| Category | Communities |
|---|---|
| CANNED FRUITS | 0, 1, 5 |
| DIET PRODUCTS | 1, 3, 4 |
| BEER | 0, 4 |
| CHOCOLATE | 0, 3 |
| CLOTHING | 0, 4 |
| DESSERT DRINKS | 0, 2 |
| FROZEN MEAT | 0, 2 |
| INTERNATIONAL CUISINE | 0, 2 |
| SEASONS | 0, 1 |
| SUGAR CONFECTIONERY | 0, 3 |
| SWEET SNACKS | 0, 3 |
| OTHER PACKAGED PRODUCTS  FRESH | 1, 2 |
| RICE / BEANS / PASTA | 1, 2 |
| SUGAR | 1, 2 |
| FROZEN READY TO COOK | 2, 4 |
| PET  FOOD | 4, 5 |

## Stability across independent refits (fixed graph, different random init)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.5312 | 0.0861 | 0.3694 | 0.7969 |
| NMI | 0.5174 | 0.0647 | 0.3991 | 0.7260 |

## Cross-validation by day-part (hard assignment)

| day_part      |   C0 |   C1 |   C2 |    C3 |   C4 |   C5 |
|:--------------|-----:|-----:|-----:|------:|-----:|-----:|
| night         |  7.1 |  9.8 |  9.6 |  47.8 |  8.4 | 17.3 |
| early_morning |  0   |  0   |  0   | 100   |  0   |  0   |
| morning       |  4.3 | 14.9 | 14.7 |  41.8 |  7.2 | 17.1 |
| afternoon     |  4.7 | 18.2 | 20.5 |  28.1 |  8.9 | 19.6 |
| evening       |  5.3 | 15.4 | 18   |  34.5 |  8.3 | 18.5 |
| late_evening  |  5.6 | 13.3 | 13.5 |  40.2 |  8.6 | 18.8 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_missions_2025.py`