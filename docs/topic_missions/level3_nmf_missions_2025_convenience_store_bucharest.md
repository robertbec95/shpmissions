# NMF topic model for missions (2025 data): Convenience Store — Bucharest

Store `65587f43fae8eed7bbc91c43`, Convenience Store, Bucharest. Full calendar-2025 purchase history, variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), 7 topics chosen by reconstruction-error elbow, reconstruction_err=255.006. Run in 91.5s.

## Topic-count selection (reconstruction-error scree)

| k | Reconstruction error |
|---|---|
| 3 | 290.416 |
| 4 | 281.522 |
| 5 | 272.509 |
| 6 | 263.739 |
| 7 | 255.006 <- elbow |
| 8 | 247.312 |
| 9 | 239.745 |

## Topic profiles

| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |
|---|---|---|---|---|---|
| 0 | no | 14.53% | 12.26% | 13.82 | TOBACCO (86%) |
| 1 | no | 7.38% | 10.61% | 5.6 | SUPPLIED BAKERY (49%), FRUITS (9%), WATER (6%) |
| 2 | no | 14.2% | 13.75% | 5.98 | CARBONATED DRINKS (57%), WATER (9%) |
| 3 | no | 16.18% | 16.53% | 6.94 | SALTY SNACKS (49%), SWEET SNACKS (14%), CHOCOLATE (6%) |
| 4 | no | 7.36% | 9.0% | 8.68 | BEER (73%), WATER (5%) |
| 5 | no | 16.33% | 17.8% | 11.32 | CHEESE PACKED (40%), PACKED SAUSAGES (22%), YOGURT  / SANA / KEFIR, DESSERT (6%) |
| 6 | no | 24.01% | 20.04% | 6.97 | MILK / CREAM (43%), YOGURT  / SANA / KEFIR, DESSERT (12%), FRUITS (6%) |

## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)

| Topic | Top categories by value share (lift) |
|---|---|
| 0 | TOBACCO (61%, x6.5), CARBONATED DRINKS (3%, x0.6), SUPPLIED BAKERY (3%, x0.4), BEER (2%, x0.6), SALTY SNACKS (2%, x0.3) |
| 1 | SUPPLIED BAKERY (38%, x6.3), FRUITS (6%, x1.6), WATER (4%, x1.2), VEGETABLES (4%, x1.5), EGGS (3%, x1.2) |
| 2 | CARBONATED DRINKS (27%, x5.2), WATER (11%, x3.0), COFFEE CORNER (7%, x4.4), BAKE OFF (5%, x1.9), ICE CREAM AND DESSERTS (4%, x2.2) |
| 3 | SALTY SNACKS (24%, x4.9), SWEET SNACKS (13%, x3.5), CHOCOLATE (8%, x3.2), WATER (4%, x1.0), SUPPLIED BAKERY (3%, x0.5) |
| 4 | BEER (44%, x11.0), SUPPLIED BAKERY (4%, x0.7), WATER (4%, x0.9), TOBACCO (3%, x0.4), CARBONATED DRINKS (3%, x0.6) |
| 5 | CHEESE PACKED (23%, x5.3), PACKED SAUSAGES (17%, x4.7), SUPPLIED BAKERY (5%, x0.9), YOGURT  / SANA / KEFIR, DESSERT (4%, x1.1), CANNED MEAT + FISH + READY MEALS (3%, x2.4) |
| 6 | MILK / CREAM (15%, x3.4), YOGURT  / SANA / KEFIR, DESSERT (9%, x2.3), FRUITS (8%, x2.2), VEGETABLES (5%, x2.2), EGGS (5%, x2.1) |

## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.6589 | 0.1416 | 0.4277 | 0.9263 |
| NMI | 0.7271 | 0.0937 | 0.5615 | 0.9105 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |   T5 |   T6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 18.4 |  4   | 16.9 | 20.3 | 13.1 | 12.7 | 14.6 |
| early_morning | 52.6 |  0   | 31.6 |  0   |  0   |  0   | 15.8 |
| morning       | 15.6 |  6.7 | 15.5 | 17.2 |  4.1 | 15.2 | 25.5 |
| afternoon     | 12   |  9.3 | 12.4 | 13.2 |  6   | 18.1 | 28.9 |
| evening       | 15   |  7.1 | 14.5 | 16.3 |  7.8 | 16.1 | 23.3 |
| late_evening  | 16.4 |  5.8 | 15.4 | 19.4 |  9   | 15.3 | 18.8 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`