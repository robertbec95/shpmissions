# NMF topic model for missions (2025 data): Convenience Store — Brasov

Store `65587f43fae8eed7bbc91cd5`, Convenience Store, Brasov. Full calendar-2025 purchase history, variety>=2, basket_value>0. NMF on the row-normalized value matrix (`topic_missions.fit_nmf_missions`), 6 topics chosen by reconstruction-error elbow, reconstruction_err=127.167. Run in 12.0s.

## Topic-count selection (reconstruction-error scree)

| k | Reconstruction error |
|---|---|
| 3 | 143.054 |
| 4 | 137.423 |
| 5 | 132.040 |
| 6 | 127.167 <- elbow |
| 7 | 122.561 |
| 8 | 118.206 |
| 9 | 114.063 |

## Topic profiles

| Topic | Degenerate? | Hard share | Avg soft weight | Avg value | Native top categories (loading) |
|---|---|---|---|---|---|
| 0 | no | 21.73% | 18.17% | 12.43 | TOBACCO (88%) |
| 1 | no | 10.71% | 11.98% | 7.91 | BEER (73%), SPIRIT DRINKS (5%) |
| 2 | no | 9.07% | 12.93% | 5.93 | CARBONATED DRINKS (59%), CHOCOLATE (6%), SWEET SNACKS (5%) |
| 3 | no | 29.58% | 27.21% | 6.68 | FRUITS (36%), VEGETABLES (10%), SUPPLIED BAKERY (9%) |
| 4 | no | 17.66% | 17.8% | 6.68 | SALTY SNACKS (50%), SWEET SNACKS (10%), CHOCOLATE (8%) |
| 5 | no | 11.24% | 11.91% | 4.06 | COFFEE CORNER (56%), WATER (13%), SWEET SNACKS (8%), BAKE OFF (8%) |

## Topic profiles, hard-assignment value-share/lift (for comparison against the graph method)

| Topic | Top categories by value share (lift) |
|---|---|
| 0 | TOBACCO (66%, x4.5), BEER (5%, x0.7), CARBONATED DRINKS (2%, x0.4), SPIRIT DRINKS (2%, x1.0), COFFEE CORNER (2%, x0.5) |
| 1 | BEER (48%, x6.7), SPIRIT DRINKS (9%, x4.4), SUPPLIED BAKERY (3%, x0.8), WATER (3%, x0.7), CHOCOLATE (2%, x0.6) |
| 2 | CARBONATED DRINKS (44%, x7.5), SWEET SNACKS (5%, x1.2), CHOCOLATE (5%, x1.2), WATER (4%, x1.1), BAKE OFF (3%, x1.2) |
| 3 | FRUITS (15%, x3.0), SUPPLIED BAKERY (9%, x2.2), VEGETABLES (8%, x2.7), MILK / CREAM (6%, x2.4), PACKED SAUSAGES (6%, x2.2) |
| 4 | SALTY SNACKS (25%, x4.7), SWEET SNACKS (11%, x2.6), CHOCOLATE (11%, x2.8), SUGAR CONFECTIONERY (4%, x2.9), ICE CREAM AND DESSERTS (3%, x1.6) |
| 5 | COFFEE CORNER (27%, x7.4), WATER (16%, x4.2), BAKE OFF (11%, x4.5), SWEET SNACKS (5%, x1.3), OTHER DRINKS (4%, x3.0) |

## Stability under non-convexity (20 random inits on the identical full matrix - no sampling)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.8448 | 0.0550 | 0.6906 | 0.9533 |
| NMI | 0.8333 | 0.0424 | 0.7274 | 0.9316 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |   T5 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|
| night         | 23.3 | 16.5 | 11.9 | 18.3 | 24.7 |  5.2 |
| early_morning | 47.4 |  5.3 | 21.1 | 15.8 |  5.3 |  5.3 |
| morning       | 27.2 |  4.5 |  5.3 | 32   | 11   | 20   |
| afternoon     | 20   |  6.5 |  8.1 | 36.7 | 13.9 | 14.8 |
| evening       | 22.1 | 12.9 |  9.7 | 27.7 | 18.5 |  9.1 |
| late_evening  | 21   | 15.2 | 10.7 | 23   | 23.6 |  6.5 |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`