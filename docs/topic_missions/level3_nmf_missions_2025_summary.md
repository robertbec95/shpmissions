# NMF topic model for missions (2025 data): cross-store summary

6 Profi stores, full calendar-2025 purchase history each, NMF (non-negative matrix factorization) on the row-normalized value matrix, run independently per store - the soft/probabilistic alternative to this project's hard-label methods (Jaccard k-medoids, value-weighted graph). Total run time 280.6s.

## Overview by store

| Store | Format | County | Baskets | Topics | Degenerate | ARI | NMI | Largest topic (hard share) |
|---|---|---|---|---|---|---|---|---|
| c917e4 | Supermarket | Bucharest | 301,311 | 5 | 0 | 0.709 | 0.746 | 43.84% |
| c917ff | Supermarket | Neamt | 396,449 | 5 | 0 | 0.862 | 0.863 | 33.59% |
| c91c04 | Mini Market | Calarasi | 101,155 | 7 | 0 | 0.734 | 0.760 | 34.34% |
| c91c35 | Convenience Store | Dolj | 127,899 | 6 | 1 | 0.841 | 0.836 | 45.74% |
| c91c43 | Convenience Store | Bucharest | 267,631 | 7 | 0 | 0.659 | 0.727 | 24.01% |
| c91cd5 | Convenience Store | Brasov | 66,847 | 6 | 0 | 0.845 | 0.833 | 29.58% |

## Full topic breakdown by store

### Supermarket — Bucharest (`65587f43fae8eed7bbc917e4`)

| Topic | Degenerate? | Hard share | Native top categories (loading) |
|---|---|---|---|
| 2 | no | 43.84% | FRUITS (26%), VEGETABLES (14%), SUPPLIED BAKERY (7%) |
| 1 | no | 20.14% | CARBONATED DRINKS (32%), WATER (11%), COFFEE CORNER (10%), SWEET SNACKS (9%) |
| 0 | no | 16.72% | TOBACCO (86%) |
| 3 | no | 10.11% | OTHERS FIIGO (64%) |
| 4 | no | 9.2% | BEER (63%), SALTY SNACKS (6%) |

### Supermarket — Neamt (`65587f43fae8eed7bbc917ff`)

| Topic | Degenerate? | Hard share | Native top categories (loading) |
|---|---|---|---|
| 4 | no | 33.59% | SUPPLIED BAKERY (26%), VEGETABLES (20%) |
| 3 | no | 33.14% | SALTY SNACKS (27%), SWEET SNACKS (16%), CARBONATED DRINKS (14%), CHOCOLATE (6%) |
| 2 | no | 12.41% | BEER (77%) |
| 1 | no | 12.21% | FRUITS (70%) |
| 0 | no | 8.65% | TOBACCO (80%) |

### Mini Market — Calarasi (`65587f43fae8eed7bbc91c04`)

| Topic | Degenerate? | Hard share | Native top categories (loading) |
|---|---|---|---|
| 3 | no | 34.34% | FRUITS (17%), CHICKEN MEAT (13%), VEGETABLES (11%), SUPPLIED BAKERY (6%) |
| 5 | no | 16.67% | SALTY SNACKS (57%), SWEET SNACKS (14%) |
| 6 | no | 13.12% | WATER (62%) |
| 2 | no | 10.78% | BEER (73%) |
| 0 | no | 10.53% | TOBACCO (85%) |
| 1 | no | 7.67% | CARBONATED DRINKS (65%), SWEET SNACKS (6%) |
| 4 | no | 6.9% | PORK (78%) |

### Convenience Store — Dolj (`65587f43fae8eed7bbc91c35`)

| Topic | Degenerate? | Hard share | Native top categories (loading) |
|---|---|---|---|
| 5 | no | 45.74% | SUPPLIED BAKERY (12%), SALTY SNACKS (11%), YOGURT  / SANA / KEFIR, DESSERT (10%), SWEET SNACKS (10%) |
| 0 | yes | 14.78% | TOBACCO (91%) |
| 3 | no | 13.92% | WATER (64%), BAKE OFF (6%), COFFEE CORNER (5%) |
| 2 | no | 11.08% | FRUITS (71%), VEGETABLES (8%) |
| 4 | no | 7.95% | BEER (78%) |
| 1 | no | 6.52% | CARBONATED DRINKS (69%), COFFEE CORNER (5%) |

### Convenience Store — Bucharest (`65587f43fae8eed7bbc91c43`)

| Topic | Degenerate? | Hard share | Native top categories (loading) |
|---|---|---|---|
| 6 | no | 24.01% | MILK / CREAM (43%), YOGURT  / SANA / KEFIR, DESSERT (12%), FRUITS (6%) |
| 5 | no | 16.33% | CHEESE PACKED (40%), PACKED SAUSAGES (22%), YOGURT  / SANA / KEFIR, DESSERT (6%) |
| 3 | no | 16.18% | SALTY SNACKS (49%), SWEET SNACKS (14%), CHOCOLATE (6%) |
| 0 | no | 14.53% | TOBACCO (86%) |
| 2 | no | 14.2% | CARBONATED DRINKS (57%), WATER (9%) |
| 1 | no | 7.38% | SUPPLIED BAKERY (49%), FRUITS (9%), WATER (6%) |
| 4 | no | 7.36% | BEER (73%), WATER (5%) |

### Convenience Store — Brasov (`65587f43fae8eed7bbc91cd5`)

| Topic | Degenerate? | Hard share | Native top categories (loading) |
|---|---|---|---|
| 3 | no | 29.58% | FRUITS (36%), VEGETABLES (10%), SUPPLIED BAKERY (9%) |
| 0 | no | 21.73% | TOBACCO (88%) |
| 4 | no | 17.66% | SALTY SNACKS (50%), SWEET SNACKS (10%), CHOCOLATE (8%) |
| 5 | no | 11.24% | COFFEE CORNER (56%), WATER (13%), SWEET SNACKS (8%), BAKE OFF (8%) |
| 1 | no | 10.71% | BEER (73%), SPIRIT DRINKS (5%) |
| 2 | no | 9.07% | CARBONATED DRINKS (59%), CHOCOLATE (6%), SWEET SNACKS (5%) |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_missions_2025.py`