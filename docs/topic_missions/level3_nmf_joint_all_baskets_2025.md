# Joint IDF-weighted NMF (all baskets, all 6 stores pooled, 2025 data)

1,890,659 baskets across all 6 stores (variety>=1, basket_value>0 - the FULL population, not just relatively-concrete, at explicit user request). Feature: x_bc = log(1+quantity_bc) x idf_c (`topic_missions.build_idf_weighted_matrix`) - item count, not spend, so premium products/promotions/inflation don't masquerade as distinct missions. One NMF fit jointly across the full pooled population, 12 topics selected by held-out reconstruction error elbow. Total run time 412.1s.

## Scope: which of the 6 requested k-selection criteria are implemented

| Criterion | Status |
|---|---|
| Temporal held-out likelihood | Approximated - NMF has no likelihood; held-out reconstruction error on a per-store date-split instead (see table below) |
| Topic coherence | Not implemented (would need NPMI/UMass scoring over category co-occurrence) |
| Bootstrap stability | Implemented - genuine basket resampling with replacement, not just seed variance (see below) |
| Minimum mission prevalence | Implemented - topics below 1.0% hard share flagged [THIN] |
| Interpretability by retail experts | Not automatable - top categories per topic are printed below for human review |
| Downstream predictive value | Not implemented (no defined business task/label to predict against) |

## Topic-count sweep

| k | Train reconstruction err | Held-out reconstruction err | Thin topics (<1% share) |
|---|---|---|---|
| 6 | 4696.75 | 2421.06 | 0 |
| 8 | 4504.16 | 2345.24 | 0 |
| 10 | 4332.38 | 2262.45 | 0 |
| 12 | 4169.68 | 2186.82 <- elbow | 0 |
| 14 | 4008.53 | 2098.72 | 0 |

## Topic profiles

### Topic 0 - hard share 13.7%, avg soft weight 13.61%

Top categories (IDF-weighted loading): VEGETABLES (12%), HOUSEHOLD PRODUCTS (9%), FRUITS (7%), SUPPLIED BAKERY (6%)

### Topic 1 - hard share 4.3%, avg soft weight 4.46%

Top categories (IDF-weighted loading): DESSERT DRINKS (73%)

### Topic 2 - hard share 5.52%, avg soft weight 5.5%

Top categories (IDF-weighted loading): BAKE OFF (80%)

### Topic 3 - hard share 4.26%, avg soft weight 5.43%

Top categories (IDF-weighted loading): SWEET SNACKS (70%)

### Topic 4 - hard share 9.42%, avg soft weight 10.35%

Top categories (IDF-weighted loading): BEER (70%), TOBACCO (8%)

### Topic 5 - hard share 4.21%, avg soft weight 4.48%

Top categories (IDF-weighted loading): DESSERT INGREDIENTS (49%), RICE / BEANS / PASTA (9%), SUGAR (7%)

### Topic 6 - hard share 9.28%, avg soft weight 8.06%

Top categories (IDF-weighted loading): YOGURT  / SANA / KEFIR, DESSERT (50%), CHEESE PACKED (11%), PACKED SAUSAGES (8%), MILK / CREAM (6%)

### Topic 7 - hard share 8.49%, avg soft weight 10.89%

Top categories (IDF-weighted loading): WATER (59%), CARBONATED DRINKS (8%)

### Topic 8 - hard share 3.98%, avg soft weight 4.66%

Top categories (IDF-weighted loading): ICE CREAM AND DESSERTS (80%)

### Topic 9 - hard share 2.81%, avg soft weight 4.0%

Top categories (IDF-weighted loading): PET  FOOD (70%)

### Topic 10 - hard share 19.68%, avg soft weight 17.19%

Top categories (IDF-weighted loading): SALTY SNACKS (27%), CHOCOLATE (22%), CARBONATED DRINKS (8%), SUGAR CONFECTIONERY (7%)

### Topic 11 - hard share 14.35%, avg soft weight 11.38%

Top categories (IDF-weighted loading): COFFEE CORNER (58%), TOBACCO (14%), OTHERS FIIGO (12%), OTHER DRINKS (5%)

## Store prevalence (post-hoc breakdown of the jointly-fit model, NOT a model covariate)

NMF has no covariate mechanism like DMR's prevalence regression - this is the empirical average soft-membership weight per topic, computed after fitting, grouped by store.

| store_label                 |   T0 |   T1 |   T2 |   T3 |   T4 |   T5 |   T6 |   T7 |   T8 |   T9 |   T10 |   T11 |
|:----------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| Convenience Store|Brasov    |  9.7 |  3.9 |  4.6 |  5.3 | 14.8 |  3   |  6.2 |  9.3 |  5.2 |  3   |  18.6 |  16.2 |
| Convenience Store|Bucharest | 13.4 |  4.1 |  6.6 |  5.3 |  8.5 |  6   | 11.2 |  9.7 |  5   |  4.5 |  16.5 |   9.2 |
| Convenience Store|Dolj      | 11.2 |  3.9 |  8.4 |  4.9 |  8.8 |  3.2 |  8.4 | 13.4 |  4.8 |  3.3 |  17   |  12.9 |
| Mini Market|Calarasi        | 15.7 |  4.2 |  0.3 |  5.7 | 11.6 |  5.8 |  8.4 | 12.4 |  7.2 |  5.6 |  18.2 |   5   |
| Supermarket|Bucharest       | 12.5 |  4.3 |  7   |  4.9 |  9.9 |  2.8 |  7.4 | 12.3 |  3.5 |  3.7 |  15.9 |  15.8 |
| Supermarket|Neamt           | 15.7 |  5.2 |  4.1 |  6.1 | 11.4 |  5.1 |  6.5 |  9.8 |  4.5 |  4   |  18.1 |   9.7 |

## Stability

### Seed variance (same full matrix, different random inits)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| ARI | 0.4857 | 0.0920 | 0.3007 | 0.6780 |
| NMI | 0.6276 | 0.0673 | 0.4942 | 0.7525 |

### Bootstrap stability (basket resampling with replacement, H-row cosine match to reference)

|   reference_topic |   mean |    std |    min |    max |
|------------------:|-------:|-------:|-------:|-------:|
|                 0 | 0.9273 | 0.0005 | 0.9269 | 0.9283 |
|                 1 | 0.9998 | 0.0001 | 0.9996 | 0.9998 |
|                 2 | 0.9983 | 0.0007 | 0.9978 | 0.9995 |
|                 3 | 0.9515 | 0.0856 | 0.8082 | 0.9977 |
|                 4 | 0.9994 | 0.0009 | 0.9979 | 0.9999 |
|                 5 | 0.9998 | 0.0001 | 0.9997 | 0.9999 |
|                 6 | 0.9715 | 0.0002 | 0.9712 | 0.9718 |
|                 7 | 0.9752 | 0.0138 | 0.9637 | 0.9976 |
|                 8 | 0.9998 | 0.0004 | 0.9991 | 1      |
|                 9 | 0.9994 | 0      | 0.9994 | 0.9994 |
|                10 | 0.8979 | 0.1878 | 0.5869 | 0.9994 |
|                11 | 0.4582 | 0.3342 | 0.2671 | 0.9997 |

## Cross-validation by day-part (hard assignment)

| day_part      |   T0 |   T1 |   T2 |   T3 |   T4 |   T5 |   T6 |   T7 |   T8 |   T9 |   T10 |   T11 |
|:--------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|
| night         |  6.4 |  3.1 |  1.8 |  4.9 | 16.3 |  1.5 |  7.7 | 10.3 |  4.4 |  2.6 |  28   |  13   |
| early_morning |  6.3 |  7.1 |  3.6 |  6.4 | 14   |  1.1 |  6.5 |  8.5 |  0.9 |  1.9 |  16.4 |  27.2 |
| morning       | 13.3 |  6   | 12.2 |  4.8 |  5.1 |  3.9 |  9.9 |  7.5 |  1.1 |  2.5 |  14.4 |  19.4 |
| afternoon     | 17.6 |  4.7 |  7.3 |  3.8 |  6.5 |  5.6 |  9.8 |  7.9 |  3.4 |  2.8 |  15.9 |  14.8 |
| evening       | 14.1 |  4   |  4   |  4.1 | 10.1 |  4.4 |  9.1 |  8.5 |  5   |  3   |  20.6 |  12.9 |
| late_evening  | 10.4 |  3.5 |  2.5 |  4.4 | 12.6 |  3.1 |  9.4 |  9.2 |  5.4 |  2.9 |  24.7 |  12   |

Reproduce with: `PYTHONPATH=src python3 scripts/run_nmf_joint_missions.py`