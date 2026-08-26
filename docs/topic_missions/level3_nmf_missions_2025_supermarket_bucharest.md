# NMF topic model for missions (2025 data): Supermarket — Bucharest

Store `65587f43fae8eed7bbc917e4`, Supermarket, Bucharest. Full calendar-2025 purchase history, variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), 5 topics chosen by reconstruction-error elbow, reconstruction_err=283.695. Run in 73.1s.

## Topic-count selection (reconstruction-error scree)

| k | Reconstruction error |
|---|---|
| 3 | 304.318 |
| 4 | 294.038 |
| 5 | 283.695 <- elbow |
| 6 | 274.683 |
| 7 | 266.123 |
| 8 | 257.006 |
| 9 | 248.717 |

## Topic profiles

| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |
|---|---|---|---|---|---|
| 0 | no | 16.72% | 14.93% | 14.23 | TOBACCO (86%) |
| 1 | no | 20.14% | 20.19% | 5.78 | CARBONATED DRINKS (32%), WATER (11%), COFFEE CORNER (10%), SWEET SNACKS (9%), SALTY SNACKS (8%) |
| 2 | no | 43.84% | 39.56% | 10.45 | FRUITS (26%), VEGETABLES (14%), SUPPLIED BAKERY (7%) |
| 3 | no | 10.11% | 13.16% | 9.12 | OTHERS FIIGO (64%) |
| 4 | no | 9.2% | 12.17% | 10.45 | BEER (63%), SALTY SNACKS (6%) |

## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)

| Topic | Top categories by value share (lift) |
|---|---|
| 0 | TOBACCO (60%, x5.7), CARBONATED DRINKS (3%, x0.6), BEER (2%, x0.5), WATER (2%, x0.4), COFFEE CORNER (2%, x0.6) |
| 1 | CARBONATED DRINKS (18%, x3.4), COFFEE CORNER (11%, x3.5), WATER (10%, x2.3), SWEET SNACKS (9%, x2.5), SALTY SNACKS (8%, x2.1) |
| 2 | FRUITS (11%, x2.0), VEGETABLES (8%, x1.9), SUPPLIED BAKERY (6%, x1.6), YOGURT  / SANA / KEFIR, DESSERT (4%, x1.7), CHEESE PACKED (4%, x1.8) |
| 3 | OTHERS FIIGO (40%, x8.7), CARBONATED DRINKS (4%, x0.7), BAKE OFF (3%, x1.3), WATER (3%, x0.7), COFFEE CORNER (3%, x0.9) |
| 4 | BEER (38%, x9.2), WINE (6%, x3.9), SALTY SNACKS (4%, x1.1), SPIRIT DRINKS (3%, x4.1), WATER (3%, x0.7) |

## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.7090 | 0.1658 | 0.3973 | 0.9866 |
| NMI | 0.7459 | 0.1195 | 0.4930 | 0.9797 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |
|:--------------|-----:|-----:|-----:|-----:|-----:|
| night         | 21.5 | 28.4 | 27.1 |  8.6 | 14.4 |
| early_morning | 28.4 | 23.1 | 29.3 |  8.3 | 10.9 |
| morning       | 19.5 | 20.4 | 45.1 | 10.4 |  4.7 |
| afternoon     | 13.9 | 17.6 | 50.4 | 12.2 |  5.9 |
| evening       | 15.6 | 18.8 | 46.4 |  9.3 | 10   |
| late_evening  | 16.8 | 20   | 41.4 |  8.2 | 13.6 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`