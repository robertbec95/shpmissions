# Location-similarity propagation of the joint NMF's mission profiles

Leave-one-out validation: for each of the 6 analyzed stores, predict its shopping-mission profile (the joint NMF's 12-topic prevalence vector - see `docs/topic_missions/level3_nmf_joint_all_baskets_2025.md`) from the weighted-Gower nearest 3 analogs among the OTHER 5 stores, then compare to that store's own real profile. Distance weights: `{'primary': 0.5, 'poi': 2.0, 'population': 4.0, 'age': 0.5}` (see `shopping_missions.location_similarity` module docstring for how these were chosen and the flat-grid-search caveat). Total run time 23.9s.

**This is not a test on a genuinely new/unanalyzed store** - all 6 stores here already have real transaction data and were fit into the joint NMF. It answers a narrower, still load-bearing question: if a store's own data were withheld, how well does its catchment similarity to the other stores reconstruct its actual mission mix? That is the honest confidence check before trusting this pipeline on a store with no transaction data at all.

## Topic definitions (reference)

| Topic | Label (top 3 categories) | Overall hard share |
|---|---|---|
| T0 | Vegetables + Household Products + Fruits | 13.7% |
| T1 | Dessert Drinks | 4.3% |
| T2 | Bake Off | 5.52% |
| T3 | Sweet Snacks | 4.26% |
| T4 | Beer + Tobacco | 9.42% |
| T5 | Dessert Ingredients + Rice / Beans / Pasta + Sugar | 4.21% |
| T6 | Yogurt  / Sana / Kefir, Dessert + Cheese Packed + Packed Sausages | 9.28% |
| T7 | Water + Carbonated Drinks | 8.49% |
| T8 | Ice Cream And Desserts | 3.98% |
| T9 | Pet  Food | 2.81% |
| T10 | Salty Snacks + Chocolate + Carbonated Drinks | 19.68% |
| T11 | Coffee Corner + Tobacco + Others Fiigo | 14.35% |

## k sweep (leave-one-out, averaged across all 6 stores)

| k | L1 error | Cosine similarity | Spearman rho |
|---|---|---|---|
| 1 | 22.63 | 0.9626 | 0.861 |
| 2 | 20.52 | 0.9705 | 0.848 |
| 3 (report default) | 18.11 | 0.9750 | 0.851 |
| 4 | 19.13 | 0.9728 | 0.850 |
| 5 | 19.81 | 0.9723 | 0.823 |

## Per-store propagation detail (k=3)

### Supermarket | Bucharest

Propagated from: Convenience Store | Bucharest (dist=0.126), Convenience Store | Brasov (dist=0.390), Convenience Store | Dolj (dist=0.413)

Accuracy: L1=13.16, cosine=0.9902, spearman=0.944

| Topic | Actual % | Predicted % | Diff |
|---|---|---|---|
| T0 (Vegetables + Household Products) | 12.50% | 11.42% | -1.08 |
| T1 (Dessert Drinks) | 4.31% | 3.97% | -0.34 |
| T2 (Bake Off) | 7.03% | 6.50% | -0.53 |
| T3 (Sweet Snacks) | 4.87% | 5.17% | +0.31 |
| T4 (Beer + Tobacco) | 9.89% | 10.69% | +0.80 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta) | 2.83% | 4.06% | +1.22 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed) | 7.36% | 8.63% | +1.28 |
| T7 (Water + Carbonated Drinks) | 12.28% | 10.80% | -1.47 |
| T8 (Ice Cream And Desserts) | 3.47% | 4.97% | +1.50 |
| T9 (Pet  Food) | 3.69% | 3.58% | -0.10 |
| T10 (Salty Snacks + Chocolate) | 15.94% | 17.41% | +1.48 |
| T11 (Coffee Corner + Tobacco) | 15.84% | 12.78% | -3.06 |

### Supermarket | Neamt

Propagated from: Convenience Store | Brasov (dist=0.316), Convenience Store | Dolj (dist=0.436), Mini Market | Calarasi (dist=0.498)

Accuracy: L1=13.27, cosine=0.9883, spearman=0.923

| Topic | Actual % | Predicted % | Diff |
|---|---|---|---|
| T0 (Vegetables + Household Products) | 15.70% | 12.18% | -3.52 |
| T1 (Dessert Drinks) | 5.17% | 4.01% | -1.16 |
| T2 (Bake Off) | 4.12% | 4.42% | +0.30 |
| T3 (Sweet Snacks) | 6.08% | 5.30% | -0.78 |
| T4 (Beer + Tobacco) | 11.36% | 11.73% | +0.37 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta) | 5.06% | 4.00% | -1.06 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed) | 6.51% | 7.67% | +1.16 |
| T7 (Water + Carbonated Drinks) | 9.76% | 11.71% | +1.95 |
| T8 (Ice Cream And Desserts) | 4.51% | 5.71% | +1.19 |
| T9 (Pet  Food) | 3.96% | 3.95% | -0.01 |
| T10 (Salty Snacks + Chocolate) | 18.06% | 17.96% | -0.10 |
| T11 (Coffee Corner + Tobacco) | 9.72% | 11.38% | +1.66 |

### Mini Market | Calarasi

Propagated from: Supermarket | Neamt (dist=0.498), Convenience Store | Brasov (dist=0.543), Convenience Store | Dolj (dist=0.630)

Accuracy: L1=26.88, cosine=0.9444, spearman=0.545

| Topic | Actual % | Predicted % | Diff |
|---|---|---|---|
| T0 (Vegetables + Household Products) | 15.67% | 12.18% | -3.48 |
| T1 (Dessert Drinks) | 4.24% | 4.32% | +0.08 |
| T2 (Bake Off) | 0.31% | 5.69% | +5.38 |
| T3 (Sweet Snacks) | 5.66% | 5.44% | -0.22 |
| T4 (Beer + Tobacco) | 11.58% | 11.65% | +0.07 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta) | 5.80% | 3.75% | -2.06 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed) | 8.35% | 7.06% | -1.30 |
| T7 (Water + Carbonated Drinks) | 12.40% | 10.83% | -1.57 |
| T8 (Ice Cream And Desserts) | 7.20% | 4.81% | -2.39 |
| T9 (Pet  Food) | 5.58% | 3.41% | -2.17 |
| T10 (Salty Snacks + Chocolate) | 18.17% | 17.92% | -0.25 |
| T11 (Coffee Corner + Tobacco) | 5.03% | 12.95% | +7.92 |

### Convenience Store | Dolj

Propagated from: Convenience Store | Brasov (dist=0.300), Convenience Store | Bucharest (dist=0.384), Supermarket | Bucharest (dist=0.413)

Accuracy: L1=11.27, cosine=0.9897, spearman=0.951

| Topic | Actual % | Predicted % | Diff |
|---|---|---|---|
| T0 (Vegetables + Household Products) | 11.16% | 11.87% | +0.71 |
| T1 (Dessert Drinks) | 3.88% | 4.11% | +0.23 |
| T2 (Bake Off) | 8.36% | 6.05% | -2.31 |
| T3 (Sweet Snacks) | 4.89% | 5.17% | +0.27 |
| T4 (Beer + Tobacco) | 8.77% | 11.06% | +2.29 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta) | 3.15% | 3.95% | +0.80 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed) | 8.41% | 8.28% | -0.13 |
| T7 (Water + Carbonated Drinks) | 13.40% | 10.43% | -2.98 |
| T8 (Ice Cream And Desserts) | 4.75% | 4.55% | -0.20 |
| T9 (Pet  Food) | 3.27% | 3.72% | +0.45 |
| T10 (Salty Snacks + Chocolate) | 17.05% | 17.04% | -0.01 |
| T11 (Coffee Corner + Tobacco) | 12.89% | 13.77% | +0.87 |

### Convenience Store | Bucharest

Propagated from: Supermarket | Bucharest (dist=0.126), Convenience Store | Dolj (dist=0.384), Convenience Store | Brasov (dist=0.447)

Accuracy: L1=22.41, cosine=0.9645, spearman=0.811

| Topic | Actual % | Predicted % | Diff |
|---|---|---|---|
| T0 (Vegetables + Household Products) | 13.40% | 11.12% | -2.28 |
| T1 (Dessert Drinks) | 4.11% | 4.04% | -0.08 |
| T2 (Bake Off) | 6.55% | 6.66% | +0.10 |
| T3 (Sweet Snacks) | 5.29% | 5.03% | -0.25 |
| T4 (Beer + Tobacco) | 8.47% | 11.16% | +2.70 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta) | 5.98% | 3.01% | -2.98 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed) | 11.24% | 7.34% | -3.90 |
| T7 (Water + Carbonated Drinks) | 9.68% | 11.67% | +1.99 |
| T8 (Ice Cream And Desserts) | 5.01% | 4.46% | -0.55 |
| T9 (Pet  Food) | 4.48% | 3.32% | -1.17 |
| T10 (Salty Snacks + Chocolate) | 16.54% | 17.21% | +0.67 |
| T11 (Coffee Corner + Tobacco) | 9.24% | 14.99% | +5.75 |

### Convenience Store | Brasov

Propagated from: Convenience Store | Dolj (dist=0.300), Supermarket | Neamt (dist=0.316), Supermarket | Bucharest (dist=0.390)

Accuracy: L1=21.68, cosine=0.9729, spearman=0.930

| Topic | Actual % | Predicted % | Diff |
|---|---|---|---|
| T0 (Vegetables + Household Products) | 9.70% | 13.12% | +3.42 |
| T1 (Dessert Drinks) | 3.91% | 4.46% | +0.55 |
| T2 (Bake Off) | 4.58% | 6.50% | +1.92 |
| T3 (Sweet Snacks) | 5.34% | 5.28% | -0.06 |
| T4 (Beer + Tobacco) | 14.83% | 10.01% | -4.82 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta) | 3.04% | 3.68% | +0.64 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed) | 6.25% | 7.43% | +1.18 |
| T7 (Water + Carbonated Drinks) | 9.32% | 11.81% | +2.49 |
| T8 (Ice Cream And Desserts) | 5.16% | 4.24% | -0.92 |
| T9 (Pet  Food) | 3.00% | 3.64% | +0.64 |
| T10 (Salty Snacks + Chocolate) | 18.64% | 17.02% | -1.63 |
| T11 (Coffee Corner + Tobacco) | 16.22% | 12.82% | -3.41 |

## Limitations

- **n=6.** The k=3 default and the specific distance weights were both chosen at this sample size - re-validate if the analyzed-store set grows.
- **Leave-one-out, not a genuinely new store.** Every store used as a "target" here already has its own real transaction data; the test measures reconstruction quality against ground truth that exists, not blind extrapolation.
- **Distance weights came from a grid search with a flat likelihood surface** (many tied-best weight combinations) - the *direction* (population >> POI > primary attributes/age structure) is well-supported; the exact multipliers are one reasonable choice among several statistically indistinguishable ones.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_propagation.py`