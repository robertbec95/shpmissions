# NMF topic model for missions (2025 data): Supermarket — Neamt

Store `65587f43fae8eed7bbc917ff`, Supermarket, Neamt. Full calendar-2025 purchase history, variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), 5 topics chosen by reconstruction-error elbow, reconstruction_err=338.106. Run in 52.4s.

## Topic-count selection (reconstruction-error scree)

| k | Reconstruction error |
|---|---|
| 3 | 362.838 |
| 4 | 349.626 |
| 5 | 338.106 <- elbow |
| 6 | 327.884 |
| 7 | 317.604 |
| 8 | 307.444 |
| 9 | 299.539 |

## Topic profiles

| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |
|---|---|---|---|---|---|
| 0 | no | 8.65% | 8.08% | 13.35 | TOBACCO (80%) |
| 1 | no | 12.21% | 13.71% | 7.61 | FRUITS (70%) |
| 2 | no | 12.41% | 12.12% | 9.24 | BEER (77%) |
| 3 | no | 33.14% | 32.68% | 7.05 | SALTY SNACKS (27%), SWEET SNACKS (16%), CARBONATED DRINKS (14%), CHOCOLATE (6%), WATER (5%) |
| 4 | no | 33.59% | 33.4% | 8.32 | SUPPLIED BAKERY (26%), VEGETABLES (20%) |

## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)

| Topic | Top categories by value share (lift) |
|---|---|
| 0 | TOBACCO (60%, x10.1), CARBONATED DRINKS (3%, x0.7), BEER (2%, x0.4), SUPPLIED BAKERY (2%, x0.3), SWEET SNACKS (2%, x0.4) |
| 1 | FRUITS (40%, x6.3), SUPPLIED BAKERY (3%, x0.6), VEGETABLES (3%, x0.7), YOGURT  / SANA / KEFIR, DESSERT (3%, x1.2), WATER (3%, x0.8) |
| 2 | BEER (41%, x7.2), TOBACCO (4%, x0.7), WINE (3%, x2.1), SALTY SNACKS (3%, x0.6), SUPPLIED BAKERY (3%, x0.6) |
| 3 | SALTY SNACKS (13%, x2.5), SWEET SNACKS (11%, x2.3), CARBONATED DRINKS (10%, x2.3), CHOCOLATE (6%, x2.1), WATER (6%, x1.8) |
| 4 | SUPPLIED BAKERY (13%, x2.3), VEGETABLES (11%, x2.4), CHICKEN MEAT (5%, x2.2), MILK / CREAM (5%, x1.8), CHEESE PACKED (4%, x1.8) |

## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8619 | 0.0734 | 0.6417 | 0.9819 |
| NMI | 0.8631 | 0.0476 | 0.7359 | 0.9634 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |
|:--------------|-----:|-----:|-----:|-----:|-----:|
| night         | 12.3 |  5.1 | 19.5 | 46.1 | 16.8 |
| early_morning | 20.2 |  9.7 | 14.3 | 37.1 | 18.7 |
| morning       | 10.5 | 13.2 |  7.3 | 31.3 | 37.8 |
| afternoon     |  6.2 | 15.5 |  8.8 | 26.9 | 42.5 |
| evening       |  7.7 | 12.9 | 13   | 31.8 | 34.5 |
| late_evening  |  9.6 |  9.2 | 16.4 | 39.1 | 25.6 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`