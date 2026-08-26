# NMF topic model for missions (2025 data): Convenience Store — Dolj

Store `65587f43fae8eed7bbc91c35`, Convenience Store, Dolj. Full calendar-2025 purchase history, variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), 6 topics chosen by reconstruction-error elbow, reconstruction_err=185.793. Run in 31.0s.

## Topic-count selection (reconstruction-error scree)

| k | Reconstruction error |
|---|---|
| 3 | 205.939 |
| 4 | 199.176 |
| 5 | 192.265 |
| 6 | 185.793 <- elbow |
| 7 | 179.484 |
| 8 | 173.623 |
| 9 | 167.745 |

## Topic profiles

| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |
|---|---|---|---|---|---|
| 0 | yes | 14.78% | 12.74% | 12.11 | TOBACCO (91%) |
| 1 | no | 6.52% | 8.21% | 4.9 | CARBONATED DRINKS (69%), COFFEE CORNER (5%) |
| 2 | no | 11.08% | 12.49% | 7.25 | FRUITS (71%), VEGETABLES (8%) |
| 3 | no | 13.92% | 15.23% | 4.97 | WATER (64%), BAKE OFF (6%), COFFEE CORNER (5%) |
| 4 | no | 7.95% | 7.89% | 8.31 | BEER (78%) |
| 5 | no | 45.74% | 43.44% | 7.35 | SUPPLIED BAKERY (12%), SALTY SNACKS (11%), YOGURT  / SANA / KEFIR, DESSERT (10%), SWEET SNACKS (10%), BAKE OFF (7%) |

## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)

| Topic | Top categories by value share (lift) |
|---|---|
| 0 | TOBACCO (67%, x6.4), CARBONATED DRINKS (4%, x0.6), WATER (2%, x0.3), BEER (2%, x0.4), SALTY SNACKS (1%, x0.3) |
| 1 | CARBONATED DRINKS (49%, x8.8), COFFEE CORNER (5%, x1.9), BAKE OFF (4%, x1.1), CHOCOLATE (4%, x1.0), SWEET SNACKS (3%, x0.7) |
| 2 | FRUITS (39%, x7.5), VEGETABLES (5%, x1.6), SUPPLIED BAKERY (4%, x0.8), BAKE OFF (3%, x0.9), YOGURT  / SANA / KEFIR, DESSERT (3%, x0.9) |
| 3 | WATER (33%, x5.8), COFFEE CORNER (7%, x3.0), CHEWING GUM (4%, x2.9), CARBONATED DRINKS (4%, x0.8), BAKE OFF (4%, x1.0) |
| 4 | BEER (46%, x11.0), TOBACCO (4%, x0.4), SALTY SNACKS (4%, x0.9), CARBONATED DRINKS (4%, x0.6), SUPPLIED BAKERY (3%, x0.7) |
| 5 | SUPPLIED BAKERY (7%, x1.6), SALTY SNACKS (7%, x1.6), YOGURT  / SANA / KEFIR, DESSERT (7%, x1.7), SWEET SNACKS (6%, x1.6), BAKE OFF (5%, x1.3) |

## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8408 | 0.1096 | 0.5097 | 0.9662 |
| NMI | 0.8359 | 0.0707 | 0.6085 | 0.9422 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |   T5 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 15.1 |  7.3 |  9.1 | 15.7 | 10.5 | 42.3 |
| early_morning | 39.3 |  3.6 |  0   |  7.1 | 17.9 | 32.1 |
| morning       | 15.3 |  5   | 13.4 | 13.9 |  4.1 | 48.3 |
| afternoon     | 13.7 |  6.5 | 11.7 | 14.3 |  6.1 | 47.7 |
| evening       | 15.3 |  6.7 | 11.2 | 13   |  8.5 | 45.3 |
| late_evening  | 15.5 |  6.9 |  9.2 | 14   | 11.7 | 42.7 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`