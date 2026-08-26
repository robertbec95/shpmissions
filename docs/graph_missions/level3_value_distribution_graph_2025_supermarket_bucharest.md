# Value-weighted graph community detection (2025 data): Supermarket — Bucharest

Store `65587f43fae8eed7bbc917e4`, Supermarket, Bucharest. Full calendar-2025 purchase history, variety>=2, basket_value>0. Same method as `docs/graph_missions/level3_value_distribution_graph.md` (value-weighted PMI graph, min_lift=1.0, min_prevalence=0.0001, Louvain resolution=2.0), run for this store alone. 301,311 baskets -> 90 graph nodes, 708 edges -> 9 communities. Run in 14.3s.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 14.97% | 11.08 | 4.75 | 12 | FRUITS (14%, x3.0), CHEESE PACKED (12%, x3.6), PACKED SAUSAGES (9%, x3.8), SUPPLIED BAKERY (8%, x2.4), CANNED MEAT + FISH + READY MEALS (4%, x3.7) |
| 1 | 24.31% | 6.93 | 3.65 | 13 | CARBONATED DRINKS (11%, x3.0), SALTY SNACKS (11%, x3.4), SWEET SNACKS (9%, x3.3), WATER (8%, x2.9), YOGURT  / SANA / KEFIR, DESSERT (6%, x2.6) |
| 2 | 14.37% | 7.09 | 3.6 | 10 | OTHERS FIIGO (18%, x5.0), BULK SAUSAGES (11%, x5.7), COFFEE CORNER (11%, x6.1), COOKED MEALS (6%, x7.7), BAKE OFF (4%, x3.1) |
| 3 | 7.71% | 12.81 | 3.93 | 11 | WINE (18%, x7.4), DESSERT DRINKS (13%, x5.3), SPIRIT DRINKS (11%, x8.3), CHOCOLATE (9%, x4.3), CAKES (4%, x4.8) |
| 4 | 4.66% | 12.65 | 4.5 | 12 | COSMETICS (16%, x10.1), DETERGENTS / BLEACHES / FABRIC SOFTENER (9%, x12.0), HOUSHOLD PAPER (8%, x7.3), HOME CARE PRODUCTS (7%, x9.4), PET  FOOD (7%, x8.7) |
| 5 | 0.83% | 16.95 | 4.81 | 9 | PRODUSE DIN CARNE (18%, x44.0), LACTATE (11%, x45.0), GEMURI, DULCETURI (7%, x49.9), ZACUSCA (4%, x52.0), PATISERIE (4%, x40.8) |
| 6 | 8.35% | 11.68 | 4.25 | 9 | BEER (19%, x5.0), PORK (15%, x7.3), CHICKEN MEAT (13%, x5.9), FROZEN READY TO COOK (2%, x6.2), FRUIT AND VEGETABLES (2%, x4.4) |
| 7 | 10.33% | 10.77 | 4.88 | 13 | VEGETABLES (18%, x4.0), MILK / CREAM (9%, x3.7), EGGS (6%, x4.4), OIL, VINEGAR (4%, x5.2), RICE / BEANS / PASTA (3%, x4.5) |
| 8 | 14.46% | 12.72 | 3.71 | 1 | TOBACCO (62%, x4.6) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9114 | 0.0663 | 0.7793 | 0.9999 |
| NMI | 0.9171 | 0.0594 | 0.8021 | 0.9998 |

Communities found per run: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 11.5 | 36.3 |  5.7 |  9   |  4.2 |  0.5 |  8.5 |  5.8 | 18.7 |
| early_morning | 12.1 | 28.8 |  7   | 10.6 |  3.8 |  0.5 |  6.6 |  4.9 | 25.8 |
| morning       | 15.5 | 24.4 | 16.9 |  7.4 |  4.3 |  0.7 |  5   |  8.3 | 17.6 |
| afternoon     | 15.9 | 18.6 | 21.7 |  6.5 |  4.6 |  0.9 |  8   | 12   | 11.9 |
| evening       | 15.6 | 23.1 | 12.4 |  8   |  5   |  1   |  9.3 | 12.2 | 13.4 |
| late_evening  | 15.1 | 27.8 |  8.1 |  8.5 |  5.2 |  0.9 | 10.2 | 10.2 | 14.1 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions_2025.py`