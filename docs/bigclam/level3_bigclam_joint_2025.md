# Joint (pooled) BigCLAM overlapping community detection (2025 data, all 6 stores)

1,261,292 baskets across all 6 stores (variety>=2, basket_value>0), pooled into ONE value-weighted PMI graph (84 nodes, 501 edges) and fit with BigCLAM once - 8 communities (log-likelihood elbow), relative threshold=0.5. Unlike the per-store BigCLAM runs (`docs/bigclam/level3_bigclam_2025_summary.md`), this is ONE shared community space, so community IDs are directly comparable across stores. Total run time 13.5s.

## Community-count selection

| C | Log-likelihood |
|---|---|
| 4 | -802.9 |
| 5 | -738.4 |
| 6 | -738.4 |
| 7 | -670.2 |
| 8 | -603.9 <- elbow |
| 9 | -605.3 |
| 10 | -573.8 |

## Community profiles

| Community | Share | # categories | Top categories by value share (lift) |
|---|---|---|---|
| 0 | 7.08% | 11 | CHOCOLATE (13%, x5.3), BEER (12%, x2.7), WINE (12%, x6.2), SPIRIT DRINKS (11%, x8.5), DESSERT DRINKS (4%, x1.7) |
| 1 | 5.66% | 13 | OTHERS FIIGO (26%, x13.5), COOKED MEALS (5%, x15.7), GARNISHES (2%, x14.9), WINE (2%, x1.1), BAKE OFF (2%, x1.7) |
| 2 | 9.42% | 10 | MILK / CREAM (15%, x5.3), CHEESE PACKED (11%, x3.1), EGGS (5%, x3.8), YOGURT  / SANA / KEFIR, DESSERT (5%, x1.7), DESSERT INGREDIENTS (4%, x6.3) |
| 3 | 13.75% | 12 | SUPPLIED BAKERY (13%, x3.4), PACKED SAUSAGES (12%, x4.0), CHEESE PACKED (7%, x1.9), BULK SAUSAGES (6%, x4.7), CANNED MEAT + FISH + READY MEALS (5%, x3.8) |
| 4 | 3.43% | 14 | PORK (28%, x14.3), FRUITS (11%, x2.6), CHICKEN MEAT (3%, x1.3), FROZEN MEAT (3%, x15.3), VEGETABLES (3%, x0.9) |
| 5 | 7.3% | 13 | COSMETICS (13%, x8.1), HOUSHOLD PAPER (6%, x6.6), DETERGENTS / BLEACHES / FABRIC SOFTENER (5%, x7.6), HOME CARE PRODUCTS (5%, x7.8), PET  FOOD (5%, x4.0) |
| 6 | 34.22% | 16 | SALTY SNACKS (10%, x2.5), CARBONATED DRINKS (9%, x2.4), SWEET SNACKS (7%, x2.3), WATER (6%, x2.2), YOGURT  / SANA / KEFIR, DESSERT (4%, x1.5) |
| 7 | 19.16% | 13 | VEGETABLES (11%, x3.1), CHICKEN MEAT (9%, x3.7), OIL, VINEGAR (5%, x3.8), RICE / BEANS / PASTA (3%, x3.3), EGGS (3%, x1.8) |

## Categories in multiple communities

| Category | Communities |
|---|---|
| DESSERT DRINKS | 0, 2, 5 |
| DRESSING | 3, 4, 7 |
| SEASONS | 0, 5 |
| SUGAR CONFECTIONERY | 0, 6 |
| WINE | 0, 1 |
| BAKE OFF | 1, 6 |
| MEAT | 1, 3 |
| SALADS | 1, 3 |
| CANNED FRUITS | 2, 3 |
| CHEESE PACKED | 2, 3 |
| EGGS | 2, 7 |
| SUGAR | 2, 7 |
| YOGURT  / SANA / KEFIR, DESSERT | 2, 6 |
| CHICKEN MEAT | 4, 7 |
| CONCENTRATED SYRUPS | 4, 6 |
| FRUIT AND VEGETABLES | 4, 7 |
| VEGETABLES | 4, 7 |

## Per-store prevalence (shared community space - directly comparable across stores)

| Store | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|---|
| Supermarket | Bucharest | 8.1% | 12.8% | 8.0% | 12.0% | 7.7% | 7.8% | 32.0% | 11.6% |
| Supermarket | Neamt | 9.2% | 6.8% | 9.2% | 14.2% | 8.5% | 7.8% | 28.7% | 15.5% |
| Mini Market | Calarasi | 7.3% | 0.9% | 9.8% | 13.3% | 10.5% | 9.8% | 30.4% | 18.0% |
| Convenience Store | Dolj | 8.6% | 6.1% | 10.0% | 12.1% | 5.8% | 8.2% | 38.9% | 10.4% |
| Convenience Store | Bucharest | 6.8% | 3.7% | 14.1% | 15.9% | 5.8% | 8.3% | 30.7% | 14.7% |
| Convenience Store | Brasov | 13.5% | 3.9% | 8.7% | 11.5% | 5.0% | 7.6% | 39.9% | 9.8% |

Reproduce with: `PYTHONPATH=src python3 scripts/run_bigclam_joint_missions.py`