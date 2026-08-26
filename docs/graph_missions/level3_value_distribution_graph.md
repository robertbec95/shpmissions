# Graph community detection on spend distribution: level3_value_distribution_graph

Category graph built from value-weighted co-occurrence (`graph_missions.compute_value_pmi_graph`, min_lift=1.0, min_prevalence=0.0001) over the exact full 2,363,118-basket population - no CLARA sampling, same property that made binary graph community detection far more stable than any sampling-based method. 99 categories -> 80 graph nodes (19 excluded below the prevalence floor - genuinely marginal merchandise like footwear/gifts/suncare, whose near-zero mass makes lift dominated by a handful of baskets), 449 edges, Louvain resolution=2.0 -> 10 communities at the reference seed. Run in 138.5s total.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 5.29% | 24.22 | 2.96 | 6 | COFFEE CORNER (26%, x18.4), OTHERS FIIGO (13%, x15.3), MEAT (8%, x23.8), BAKE OFF (7%, x8.8), COOKED MEALS (6%, x23.7) |
| 1 | 15.14% | 47.31 | 4.73 | 9 | PACKED SAUSAGES (12%, x3.8), CHEESE PACKED (10%, x3.7), BULK SAUSAGES (9%, x4.3), SUPPLIED BAKERY (8%, x2.4), YOGURT  / SANA / KEFIR, DESSERT (8%, x3.0) |
| 2 | 6.77% | 60.17 | 4.76 | 8 | PORK (22%, x8.2), CHICKEN MEAT (21%, x6.6), PREPARED MEAT (3%, x6.5), FROZEN READY TO COOK (3%, x6.3), FRUIT AND VEGETABLES (3%, x4.7) |
| 3 | 8.58% | 43.86 | 4.33 | 10 | DESSERT DRINKS (16%, x6.9), MILK / CREAM (11%, x4.3), OIL, VINEGAR (7%, x5.6), EGGS (6%, x5.4), RICE / BEANS / PASTA (4%, x4.5) |
| 4 | 5.95% | 32.37 | 3.82 | 7 | VEGETABLES (37%, x7.8), SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU (4%, x6.9), CANNED VEGETABLES (4%, x7.8), SAUCES (3%, x7.7), HERBS AND SPICES (2%, x6.3) |
| 5 | 24.88% | 29.39 | 3.52 | 13 | SALTY SNACKS (13%, x3.3), CARBONATED DRINKS (13%, x2.9), SWEET SNACKS (9%, x3.1), ICE CREAM AND DESSERTS (8%, x3.7), WATER (8%, x2.9) |
| 6 | 10.21% | 39.4 | 3.9 | 15 | FRUITS (17%, x4.2), COSMETICS (10%, x6.2), PET  FOOD (8%, x6.5), DETERGENTS / BLEACHES / FABRIC SOFTENER (5%, x7.2), HOUSEHOLD PRODUCTS (4%, x2.8) |
| 7 | 13.6% | 54.02 | 3.52 | 1 | TOBACCO (63%, x4.7) |
| 8 | 9.49% | 41.89 | 3.35 | 7 | BEER (31%, x5.7), SPIRIT DRINKS (11%, x8.6), WINE (8%, x7.7), CHOCOLATE (7%, x3.8), CAKES (3%, x5.5) |
| 9 | 0.08% | 53.4 | 4.24 | 4 | PRODUSE DIN CARNE (23%, x509.7), LACTATE (11%, x484.3), PATISERIE (10%, x391.3), PANIFICATIE (7%, x313.0) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

20 independent community-detection runs on the same value-weighted PMI graph (built once from the full population), each with a different Louvain seed - identical methodology to the binary graph stability tests (`docs/stability/level3_baseline.md`'s graph section), only the edge weighting differs.

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9622 | 0.0722 | 0.7227 | 1.0000 |
| NMI | 0.9760 | 0.0377 | 0.8419 | 1.0000 |

Communities found per run: [10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9]

For reference, every other method tested in this project: binary Jaccard CLARA ARI~0.21-0.23, Bray-Curtis (value distribution) CLARA ARI~0.35, binary graph community detection ARI~0.56-0.67. This is the most stable method found so far, by a wide margin - 'highly stable' (>0.9) by this project's rule of thumb, vs. 'largely re-deriving noise' (<0.5) for every CLARA-sampling method tested.

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  7.5 | 10.4 |  1.7 |  2.9 |  1.5 | 32.6 |  5.1 | 22.3 | 15.8 |  0.2 |
| early_morning | 10.3 | 16.8 |  6.5 |  9.1 |  5.6 | 20.6 | 10   | 15   |  6.2 |  0.1 |
| morning       |  5.9 | 17.6 | 10   | 11.5 |  8.2 | 16.6 | 12.4 | 10.7 |  7.1 |  0.1 |
| afternoon     |  5.6 | 14.4 |  7.4 |  8.8 |  6.5 | 24.7 | 10.6 | 12.6 |  9.3 |  0.1 |
| evening       |  3.5 | 14.4 |  5.1 |  7.3 |  4.9 | 29.4 |  9.3 | 14.7 | 11.2 |  0.1 |
| late_evening  |  3   | 13.4 |  3   |  5.1 |  2.9 | 34.3 |  7.5 | 17.8 | 12.9 |  0.1 |

## Cross-validation by county (top 10 by volume)

| county    |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Timis     |  5.4 | 12.3 |  5.8 |  7.4 |  5.7 | 26.9 |  8.5 | 17.8 | 10.1 |  0.1 |
| Cluj      |  5.7 | 13.7 |  4.5 |  9.5 |  6   | 24.5 |  9.1 | 17.2 |  9.6 |  0.1 |
| Bucharest |  4.8 | 14.7 |  4.9 |  8.5 |  6   | 25.4 |  9.7 | 16.4 |  9.3 |  0.2 |
| Dolj      |  6.9 | 18.9 |  8.7 |  8.6 |  6   | 23.3 | 10.3 |  9.5 |  7.7 |  0   |
| Bihor     |  7.1 | 13.9 |  6.4 |  8.1 |  5.8 | 26   |  8.5 | 14.7 |  9.2 |  0.2 |
| Constanta |  6.6 | 14.9 |  6.5 |  8.2 |  5.9 | 25.2 | 10.1 | 13.3 |  9.2 |  0   |
| Suceava   |  4.6 | 14.9 |  6   |  8.4 |  4.9 | 28.3 | 11.4 | 10.8 | 10.5 |  0.2 |
| Bacau     |  4.1 | 14.7 |  5.9 |  8.9 |  5.7 | 25   | 10.4 | 13.9 | 11.4 |  0   |
| Iasi      |  3.7 | 16   |  5.3 |  9.4 |  6   | 26.1 | 10.7 | 12.7 | 10.2 |  0.1 |
| Arges     |  5.8 | 17.3 |  8.2 |  8.5 |  5.3 | 23.2 | 10.9 | 11.6 |  9.1 |  0   |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions.py`