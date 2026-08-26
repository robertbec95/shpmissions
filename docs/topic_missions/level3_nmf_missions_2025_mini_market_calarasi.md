# NMF topic model for missions (2025 data): Mini Market — Calarasi

Store `65587f43fae8eed7bbc91c04`, Mini Market, Calarasi. Full calendar-2025 purchase history, variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), 7 topics chosen by reconstruction-error elbow, reconstruction_err=151.803. Run in 20.6s.

## Topic-count selection (reconstruction-error scree)

| k | Reconstruction error |
|---|---|
| 3 | 175.996 |
| 4 | 169.798 |
| 5 | 163.601 |
| 6 | 157.701 |
| 7 | 151.803 <- elbow |
| 8 | 146.377 |
| 9 | 141.792 |

## Topic profiles

| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |
|---|---|---|---|---|---|
| 0 | no | 10.53% | 9.06% | 16.31 | TOBACCO (85%) |
| 1 | no | 7.67% | 10.03% | 8.54 | CARBONATED DRINKS (65%), SWEET SNACKS (6%) |
| 2 | no | 10.78% | 11.26% | 12.58 | BEER (73%) |
| 3 | no | 34.34% | 31.9% | 11.76 | FRUITS (17%), CHICKEN MEAT (13%), VEGETABLES (11%), SUPPLIED BAKERY (6%), MILK / CREAM (6%) |
| 4 | no | 6.9% | 6.82% | 22.13 | PORK (78%) |
| 5 | no | 16.67% | 15.87% | 9.63 | SALTY SNACKS (57%), SWEET SNACKS (14%) |
| 6 | no | 13.12% | 15.06% | 10.64 | WATER (62%) |

## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)

| Topic | Top categories by value share (lift) |
|---|---|
| 0 | TOBACCO (58%, x8.8), CARBONATED DRINKS (4%, x0.6), BEER (3%, x0.5), SALTY SNACKS (2%, x0.4), SWEET SNACKS (2%, x0.5) |
| 1 | CARBONATED DRINKS (43%, x6.9), SWEET SNACKS (5%, x1.3), CHOCOLATE (4%, x1.7), ICE CREAM AND DESSERTS (4%, x1.5), YOGURT  / SANA / KEFIR, DESSERT (2%, x0.8) |
| 2 | BEER (39%, x7.5), CARBONATED DRINKS (5%, x0.8), SUPPLIED BAKERY (3%, x1.0), ICE CREAM AND DESSERTS (2%, x1.0), PACKED SAUSAGES (2%, x0.8) |
| 3 | FRUITS (9%, x2.0), CHICKEN MEAT (8%, x2.2), VEGETABLES (7%, x2.0), MILK / CREAM (6%, x1.8), SUPPLIED BAKERY (6%, x1.6) |
| 4 | PORK (49%, x14.1), CHICKEN MEAT (4%, x1.1), VEGETABLES (4%, x1.0), FRUITS (2%, x0.5), MILK / CREAM (2%, x0.7) |
| 5 | SALTY SNACKS (26%, x4.8), SWEET SNACKS (11%, x2.8), CHOCOLATE (5%, x2.4), CARBONATED DRINKS (5%, x0.9), YOGURT  / SANA / KEFIR, DESSERT (3%, x1.1) |
| 6 | WATER (31%, x6.0), ICE CREAM AND DESSERTS (6%, x2.4), CARBONATED DRINKS (5%, x0.8), SWEET SNACKS (4%, x1.0), SUPPLIED BAKERY (3%, x0.9) |

## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.7336 | 0.1107 | 0.5257 | 0.9448 |
| NMI | 0.7600 | 0.0764 | 0.6034 | 0.9223 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |   T5 |   T6 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 16   | 11.9 |  9.1 | 20   |  2.1 | 26.6 | 14.2 |
| early_morning | 12.5 | 25   |  0   | 25   |  0   | 12.5 | 25   |
| morning       |  8.9 |  5.7 |  7.4 | 37.9 |  8.3 | 18.8 | 13.1 |
| afternoon     |  9.2 |  6.5 | 10.8 | 39.4 |  8.9 | 12.9 | 12.4 |
| evening       | 11.3 |  8.3 | 12   | 32.4 |  5.4 | 16.9 | 13.6 |
| late_evening  | 13   | 10.3 | 11.3 | 24.6 |  4   | 22.7 | 14.1 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`