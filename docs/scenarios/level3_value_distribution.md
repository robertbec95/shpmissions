# Scenario: level3_value_distribution

Stage 2 clustering on each basket's *spend distribution* across categories (Bray-Curtis dissimilarity on row-normalized RON value per category) instead of category presence/absence (Jaccard on the binary matrix, used everywhere else in this project). Same population as the frozen baseline `level3_baseline_gapk` (variety>=2, category level 3, 3-day window), minus 924 baskets with basket_value<=0 (all exactly 0 - promotional/data edge cases with no spend distribution to speak of).

**Result:** 2,363,118 baskets, 99 categories, run in 349.0s.
 k-selection: gap_k=2, silhouette_k=11, elbow_k=10. The gap rule is satisfied at nearly every k from 4 upward here (only k=3 fails it) - the same weak-signal pattern flagged for level-4 categories (docs/scenarios/level4.md), where the mechanical "smallest satisfying k" answer doesn't reflect a strong preference. Silhouette has a genuine peak (not just monotonic drift) around k=10-11 - 0.166-0.174, 2-3x higher than any silhouette value seen anywhere in this project under Jaccard (ceiling ~0.07). Both k=2 (gap statistic, mechanical) and k=10 (elbow; the featured result - see below) are profiled in full.

## k-selection curve

|   k |    gap |     sk | gap_rule_satisfied   |   silhouette |
|----:|-------:|-------:|:---------------------|-------------:|
|   2 | 0.0134 | 0.0023 | True                 |       0.061  |
|   3 | 0.0045 | 0.0062 | False                |       0.058  |
|   4 | 0.0452 | 0.0051 | True                 |       0.1118 |
|   5 | 0.0441 | 0.0123 | True                 |       0.1063 |
|   6 | 0.0427 | 0.0086 | True                 |       0.1231 |
|   7 | 0.0504 | 0.0108 | False                |       0.1367 |
|   8 | 0.0664 | 0.0148 | True                 |       0.1599 |
|   9 | 0.0694 | 0.0163 | True                 |       0.1573 |
|  10 | 0.084  | 0.0193 | True                 |       0.1664 |
|  11 | 0.0839 | 0.0235 | True                 |       0.1741 |
|  12 | 0.0797 | 0.029  | True                 |       0.1709 |
|  13 | 0.067  | 0.0278 | True                 |       0.1616 |
|  14 | 0.0583 | 0.0266 | True                 |       0.1585 |
|  15 | 0.0639 | 0.0234 | True                 |       0.1641 |
|  16 | 0.0639 | 0.0218 | True                 |       0.1663 |
|  17 | 0.06   | 0.0137 | True                 |       0.1638 |
|  18 | 0.0444 | 0.012  | True                 |       0.1639 |
|  19 | 0.0445 | 0.0105 | True                 |       0.165  |
|  20 | 0.0399 | 0.0091 | False                |       0.167  |

## Cluster profiles at k=2 (gap statistic, mechanical answer)

| Cluster | Share | Avg value | Avg variety | Top categories by value share (lift) |
|---|---|---|---|---|
| 0 | 39.27% | 41.73 | 3.33 | TOBACCO (33%, x2.4), CARBONATED DRINKS (9%, x2.0), BEER (6%, x1.1), SWEET SNACKS (2%, x0.8), SALTY SNACKS (2%, x0.6) |
| 1 | 60.73% | 40.38 | 4.22 | VEGETABLES (7%, x1.5), FRUITS (6%, x1.5), SALTY SNACKS (5%, x1.3), BEER (5%, x0.9), SUPPLIED BAKERY (5%, x1.4) |

### k=2: cluster mix by day-part

| day_part      |   C0 |   C1 |
|:--------------|-----:|-----:|
| night         | 54.9 | 45.1 |
| early_morning | 38.5 | 61.5 |
| morning       | 32   | 68   |
| afternoon     | 39   | 61   |
| evening       | 41.9 | 58.1 |
| late_evening  | 47.5 | 52.5 |

## Cluster profiles at k=10 (FEATURED)

| Cluster | Share | Avg value | Avg variety | Top categories by value share (lift) |
|---|---|---|---|---|
| 0 | 19.9% | 51.99 | 3.57 | TOBACCO (50%, x3.8), BEER (4%, x0.8), CARBONATED DRINKS (3%, x0.7), SPIRIT DRINKS (2%, x1.4), SALTY SNACKS (2%, x0.5) |
| 1 | 6.45% | 28.45 | 3.23 | COFFEE CORNER (22%, x15.4), CHOCOLATE (19%, x9.9), SWEET SNACKS (4%, x1.4), WATER (4%, x1.4), CARBONATED DRINKS (3%, x0.7) |
| 2 | 6.75% | 44.2 | 4.14 | DESSERT DRINKS (22%, x9.4), MILK / CREAM (17%, x6.5), CHEESE PACKED (3%, x1.2), YOGURT  / SANA / KEFIR, DESSERT (3%, x1.2), CHICKEN MEAT (3%, x0.9) |
| 3 | 15.4% | 29.46 | 3.42 | CARBONATED DRINKS (21%, x4.9), WATER (12%, x4.3), SALTY SNACKS (9%, x2.4), ICE CREAM AND DESSERTS (4%, x1.7), SWEET SNACKS (3%, x1.1) |
| 4 | 10.34% | 40.68 | 4.43 | VEGETABLES (27%, x5.8), CHICKEN MEAT (7%, x2.2), FRUITS (7%, x1.7), CHEESE PACKED (4%, x1.5), YOGURT  / SANA / KEFIR, DESSERT (3%, x1.0) |
| 5 | 8.38% | 57.88 | 5.23 | PACKED SAUSAGES (17%, x5.4), BULK SAUSAGES (14%, x6.3), YOGURT  / SANA / KEFIR, DESSERT (7%, x2.6), CHEESE PACKED (6%, x2.3), CHICKEN MEAT (4%, x1.1) |
| 6 | 2.79% | 78.84 | 5.53 | PORK (45%, x16.3), CHICKEN MEAT (7%, x2.1), VEGETABLES (5%, x1.0), BULK SAUSAGES (2%, x1.0), SUPPLIED BAKERY (2%, x0.6) |
| 7 | 14.9% | 37.41 | 3.73 | BEER (26%, x4.8), SALTY SNACKS (12%, x3.0), SWEET SNACKS (9%, x3.1), ICE CREAM AND DESSERTS (3%, x1.2), CARBONATED DRINKS (3%, x0.6) |
| 8 | 2.74% | 38.36 | 3.71 | HOUSHOLD PAPER (15%, x17.5), HOUSEHOLD PRODUCTS (14%, x9.5), CHICKEN MEAT (5%, x1.6), COSMETICS (5%, x2.8), ICE CREAM AND DESSERTS (4%, x1.6) |
| 9 | 12.36% | 26.98 | 3.52 | FRUITS (24%, x6.1), SUPPLIED BAKERY (17%, x4.9), YOGURT  / SANA / KEFIR, DESSERT (4%, x1.3), CHICKEN MEAT (3%, x1.1), CHEESE PACKED (3%, x1.2) |

### k=10: cluster mix by day-part

| day_part      |   C0 |   C1 |   C2 |   C3 |   C4 |   C5 |   C6 |   C7 |   C8 |   C9 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 29.3 |  8.7 |  3.2 | 20.9 |  2.4 |  6.6 |  0.1 | 22   |  1.6 |  5.1 |
| early_morning | 20.4 |  9.5 |  7.7 | 13.1 |  9.5 |  8.7 |  3.2 | 10.8 |  2.6 | 14.6 |
| morning       | 16.3 |  6   |  8.4 | 10.8 | 14.3 |  9.4 |  4.9 | 10.5 |  3.1 | 16.3 |
| afternoon     | 19.2 |  6.7 |  6.7 | 15.5 | 11.4 |  8.1 |  3   | 13.9 |  2.9 | 12.5 |
| evening       | 21.4 |  5.7 |  6.1 | 17.4 |  8.4 |  8.2 |  1.6 | 18.3 |  2.5 | 10.4 |
| late_evening  | 24.3 |  5.6 |  4.9 | 20.9 |  4.8 |  7.8 |  0.6 | 21.4 |  2.1 |  7.7 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_value_distribution_scenario.py`