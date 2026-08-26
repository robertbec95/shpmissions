# Graph community detection on spend distribution: level3_value_distribution_graph_8day

Dates: 20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511 (8-day window). Category graph built from value-weighted co-occurrence (`graph_missions.compute_value_pmi_graph`, min_lift=1.0, min_prevalence=0.0001) over the exact full 6,687,923-basket population - no CLARA sampling, same property that made binary graph community detection far more stable than any sampling-based method. 99 categories -> 80 graph nodes (19 excluded below the prevalence floor - genuinely marginal merchandise like footwear/gifts/suncare, whose near-zero mass makes lift dominated by a handful of baskets), 433 edges, Louvain resolution=2.0 -> 10 communities at the reference seed. Run in 438.7s total.

## Community profiles

| Community | Share | Avg value | Avg variety | # categories | Top categories by value share (lift) |
|---|---|---|---|---|---|
| 0 | 6.14% | 22.8 | 2.89 | 6 | COFFEE CORNER (27%, x16.7), OTHERS FIIGO (13%, x13.7), MEAT (8%, x21.0), BAKE OFF (7%, x8.3), COOKED MEALS (6%, x21.0) |
| 1 | 12.42% | 45.82 | 4.6 | 8 | PACKED SAUSAGES (14%, x4.2), BULK SAUSAGES (11%, x5.0), CHEESE PACKED (11%, x4.1), SUPPLIED BAKERY (9%, x2.7), CANNED MEAT + FISH + READY MEALS (4%, x4.0) |
| 2 | 6.62% | 58.9 | 4.73 | 8 | PORK (22%, x8.3), CHICKEN MEAT (22%, x6.7), FROZEN READY TO COOK (3%, x6.4), FRUIT AND VEGETABLES (3%, x4.8), PREPARED MEAT (3%, x6.4) |
| 3 | 6.33% | 37.31 | 4.17 | 9 | MILK / CREAM (16%, x6.0), OIL, VINEGAR (10%, x8.1), EGGS (8%, x7.8), RICE / BEANS / PASTA (6%, x6.3), DESSERT INGREDIENTS (5%, x8.7) |
| 4 | 6.01% | 31.98 | 3.82 | 7 | VEGETABLES (37%, x7.7), SALT, BASIC FOOD INGREDIENTS, SOUPS, INSTANT PRODU (4%, x6.8), CANNED VEGETABLES (4%, x7.7), SAUCES (3%, x7.6), HERBS AND SPICES (2%, x6.2) |
| 5 | 28.34% | 28.79 | 3.56 | 14 | CARBONATED DRINKS (11%, x2.6), SALTY SNACKS (11%, x3.0), SWEET SNACKS (8%, x2.8), WATER (8%, x2.7), ICE CREAM AND DESSERTS (7%, x3.3) |
| 6 | 10.2% | 38.14 | 3.88 | 15 | FRUITS (18%, x4.2), COSMETICS (10%, x6.2), PET  FOOD (8%, x6.4), DETERGENTS / BLEACHES / FABRIC SOFTENER (6%, x7.2), HOUSHOLD PAPER (4%, x4.8) |
| 7 | 13.31% | 53.25 | 3.5 | 1 | TOBACCO (63%, x4.7) |
| 8 | 10.53% | 43.13 | 3.49 | 8 | BEER (24%, x4.8), DESSERT DRINKS (12%, x5.3), SPIRIT DRINKS (8%, x7.2), CHOCOLATE (7%, x3.5), WINE (6%, x6.5) |
| 9 | 0.09% | 52.04 | 4.14 | 4 | PRODUSE DIN CARNE (23%, x507.2), PATISERIE (13%, x399.9), LACTATE (11%, x475.2), PANIFICATIE (4%, x269.6) |

## Stability across seeds (Louvain-only randomness - graph itself is fixed)

20 independent community-detection runs on the same value-weighted PMI graph (built once from the full population), each with a different Louvain seed - identical methodology to the binary graph stability tests (`docs/stability/level3_baseline.md`'s graph section), only the edge weighting differs.

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.9664 | 0.0478 | 0.8129 | 1.0000 |
| NMI | 0.9684 | 0.0420 | 0.8375 | 1.0000 |

Communities found per run: [10, 10, 10, 10, 9, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10]

For reference, every other method tested in this project: binary Jaccard CLARA ARI~0.21-0.23, Bray-Curtis (value distribution) CLARA ARI~0.35, binary graph community detection ARI~0.56-0.67. This is the most stable method found so far, by a wide margin - 'highly stable' (>0.9) by this project's rule of thumb, vs. 'largely re-deriving noise' (<0.5) for every CLARA-sampling method tested.

## Cross-validation by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         |  7.7 |  9.4 |  1.6 |  2.1 |  1.7 | 34.2 |  5.5 | 23   | 14.7 |  0.1 |
| early_morning | 12.7 | 12.4 |  5   |  5.6 |  4.9 | 27.5 |  9.4 | 15.1 |  7.5 |  0.1 |
| morning       |  7.2 | 14.4 |  9.4 |  8.4 |  8.2 | 20.9 | 12.5 | 10.3 |  8.6 |  0.1 |
| afternoon     |  6.4 | 12.1 |  7.4 |  6.6 |  6.7 | 28.1 | 10.6 | 12.1 | 10   |  0.1 |
| evening       |  3.6 | 12.1 |  5.6 |  5.7 |  5.2 | 31.5 |  9.4 | 14.4 | 12.4 |  0.1 |
| late_evening  |  3   | 11.1 |  3.2 |  3.9 |  3   | 36.4 |  7.7 | 17.8 | 13.8 |  0.1 |

## Cross-validation by county (top 10 by volume)

| county    |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Timis     |  6.5 | 10.3 |  5.6 |  5.6 |  5.7 | 29.6 |  8.4 | 17.7 | 10.6 |  0.1 |
| Cluj      |  6.3 | 10.8 |  4.4 |  7.3 |  5.8 | 28.6 |  9.2 | 17   | 10.5 |  0.1 |
| Bucharest |  5.8 | 12   |  4.8 |  6.3 |  6.1 | 28.7 |  9.7 | 16.3 | 10.4 |  0.2 |
| Dolj      |  8.2 | 15.7 |  8.2 |  6.1 |  6.1 | 27   | 10.2 |  9.4 |  9.1 |  0   |
| Bihor     |  8.2 | 11.3 |  6.5 |  6.4 |  5.8 | 28.8 |  9   | 14.2 |  9.7 |  0.1 |
| Constanta |  7.5 | 12.4 |  6.9 |  6.1 |  6.1 | 27.9 | 10   | 12.9 | 10.2 |  0   |
| Suceava   |  5.2 | 12   |  5.9 |  6.1 |  5.1 | 32.5 | 10.9 | 10.6 | 11.6 |  0.2 |
| Bacau     |  4.8 | 12   |  6   |  6.4 |  5.8 | 29.1 | 10.1 | 13.5 | 12.4 |  0   |
| Iasi      |  4.2 | 12.3 |  5.2 |  6.9 |  6   | 31.4 | 10.3 | 12.4 | 11.2 |  0.1 |
| Arges     |  6.7 | 14.2 |  7.7 |  5.9 |  5.2 | 27.3 | 10.9 | 11.5 | 10.6 |  0   |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_graph_missions.py --name level3_value_distribution_graph_8day --dates 20260504,20260505,20260506,20260507,20260508,20260509,20260510,20260511`