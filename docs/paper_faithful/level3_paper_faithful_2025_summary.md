# Faithful reproduction (2025 data): cross-store summary

6 Profi stores, full calendar-2025 purchase history each, same faithful procedure (real Stage 1 gating, 60/40 train/test split, gap statistic on train, Jaccard k-medoids) run independently per store - this is the paper's own framing ("For each store, a second clustering analysis..."), the first time this project has had genuine per-store data to honor that with. Total run time 36.7s.

## Stage 1 + Stage 2 overview by store

| Store | Format | County | Total baskets | Relatively-concrete | gap_k | Largest mission | Top categories |
|---|---|---|---|---|---|---|---|
| c917e4 | Supermarket | Bucharest | 421,770 | 163,748 | 3 | 39.32% (breakfast) | VEGETABLES, HOUSEHOLD PRODUCTS, SUPPLIED BAKERY |
| c917ff | Supermarket | Neamt | 587,487 | 215,038 | 2 | 56.42% (breakfast) | VEGETABLES, SUPPLIED BAKERY, BEER |
| c91c04 | Mini Market | Calarasi | 146,650 | 41,057 | 3 | 45.46% (snack) | SWEET SNACKS, SALTY SNACKS, BEER |
| c91c35 | Convenience Store | Dolj | 211,587 | 57,091 | 3 | 38.01% (breakfast) | CHOCOLATE, BAKE OFF, WATER |
| c91c43 | Convenience Store | Bucharest | 407,460 | 128,518 | 2 | 53.51% (snack) | SWEET SNACKS, SALTY SNACKS, CARBONATED DRINKS |
| c91cd5 | Convenience Store | Brasov | 118,593 | 29,488 | 3 | 39.59% (snack) | CARBONATED DRINKS, BEER, CHOCOLATE |

## Full mission breakdown by store

### Supermarket — Bucharest (`65587f43fae8eed7bbc917e4`)

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | snack | 31.26% | WATER (x1.8), OTHERS FIIGO (x1.8), TOBACCO (x1.8), SALTY SNACKS (x1.7) |
| 1 | breakfast | 39.32% | VEGETABLES (x2.2), HOUSEHOLD PRODUCTS (x2.0), SUPPLIED BAKERY (x1.9), FRUITS (x1.6) |
| 2 | snack | 29.42% | CARBONATED DRINKS (x2.6), SWEET SNACKS (x2.6), YOGURT  / SANA / KEFIR, DESSERT (x2.4) |

### Supermarket — Neamt (`65587f43fae8eed7bbc917ff`)

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | breakfast | 56.42% | VEGETABLES (x1.5), SUPPLIED BAKERY (x1.4), BEER (x1.3) |
| 1 | snack | 43.58% | SWEET SNACKS (x2.2), SALTY SNACKS (x2.2), FRUITS (x1.2) |

### Mini Market — Calarasi (`65587f43fae8eed7bbc91c04`)

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | snack | 45.46% | SWEET SNACKS (x1.9), SALTY SNACKS (x1.9), BEER (x1.5), CHOCOLATE (x1.3) |
| 1 | light_meal | 19.24% | PACKED SAUSAGES (x2.5), YOGURT  / SANA / KEFIR, DESSERT (x2.3), CARBONATED DRINKS (x2.2), CHEESE PACKED (x1.4) |
| 2 | breakfast | 35.29% | VEGETABLES (x1.9), MILK / CREAM (x1.8), FRUITS (x1.8), WATER (x1.5) |

### Convenience Store — Dolj (`65587f43fae8eed7bbc91c35`)

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | snack | 37.62% | BEER (x2.0), SALTY SNACKS (x1.8), CARBONATED DRINKS (x1.7), SUPPLIED BAKERY (x1.3) |
| 1 | breakfast | 38.01% | CHOCOLATE (x2.2), BAKE OFF (x2.2), WATER (x2.2) |
| 2 | breakfast | 24.37% | VEGETABLES (x3.1), FRUITS (x2.7), HOUSEHOLD PRODUCTS (x2.6), SUPPLIED BAKERY (x1.4) |

### Convenience Store — Bucharest (`65587f43fae8eed7bbc91c43`)

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | snack | 53.51% | SWEET SNACKS (x1.5), SALTY SNACKS (x1.4), CARBONATED DRINKS (x1.3), MILK / CREAM (x1.2) |
| 1 | breakfast | 46.49% | VEGETABLES (x2.0), BEER (x2.0), SUPPLIED BAKERY (x1.9), HOUSEHOLD PRODUCTS (x1.9) |

### Convenience Store — Brasov (`65587f43fae8eed7bbc91cd5`)

| Cluster | Label | Share | Top categories (lift) |
|---|---|---|---|
| 0 | snack | 39.59% | CARBONATED DRINKS (x1.8), BEER (x1.8), CHOCOLATE (x1.7), SALTY SNACKS (x1.6) |
| 1 | unlabeled | 27.22% | YOGURT  / SANA / KEFIR, DESSERT (x3.1), WATER (x2.9), TOBACCO (x1.7) |
| 2 | meal | 33.19% | VEGETABLES (x2.5), FRUITS (x2.4), HOUSEHOLD PRODUCTS (x2.1), SWEET SNACKS (x1.7) |

## Does mission structure differ by store format?

**Supermarket** (2 stores):
- Bucharest: gap_k=3, largest mission 39.32%
- Neamt: gap_k=2, largest mission 56.42%

**Mini Market** (1 store):
- Calarasi: gap_k=3, largest mission 45.46%

**Convenience Store** (3 stores):
- Dolj: gap_k=3, largest mission 38.01%
- Bucharest: gap_k=2, largest mission 53.51%
- Brasov: gap_k=3, largest mission 39.59%

Reproduce with: `PYTHONPATH=src python3 scripts/run_paper_faithful_v2.py`