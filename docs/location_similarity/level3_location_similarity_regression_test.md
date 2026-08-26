# Ridge-regression propagation vs. k-NN / cluster baselines

Tests explicit ridge regression of topic prevalence on catchment features (`location_similarity.propagate_via_regression` - one `RidgeCV` fit per topic, standardized features, alpha chosen by closed-form leave-one-out over the training stores) as a more *explainable* alternative to k=3 nearest-neighbor averaging: instead of "the answer is whatever your 3 nearest analogs say", each topic gets an inspectable coefficient per catchment feature. The honest cost, stated up front and confirmed empirically below: fitting even a *regularized* 6-feature relationship from 5 training stores per leave-one-out fold is a fragile exercise. Total run time 40.9s.

## Headline: regression underperforms both existing strategies

| Strategy | NMF L1 | NMF cosine | NMF spearman | BigCLAM L1 | BigCLAM cosine | BigCLAM spearman |
|---|---|---|---|---|---|---|
| k-NN (k=3) | 18.11 | 0.9750 | 0.851 | 21.13 | 0.9773 | 0.810 |
| Ridge regression | 27.22 | 0.9434 | 0.797 | 25.13 | 0.9672 | 0.675 |

Regression is worse on every metric, for both ground truths - roughly 1.5x the L1 error of k-NN for NMF, 1.2x for BigCLAM.

## Per-store detail

| Store | NMF regression L1 | NMF k-NN L1 | BigCLAM regression L1 | BigCLAM k-NN L1 |
|---|---|---|---|---|
| Supermarket | Bucharest | 17.71 | 18.11 | 20.62 | 21.13 |
| Supermarket | Neamt | 22.71 | 18.11 | 36.90 | 21.13 |
| Mini Market | Calarasi | 63.10 | 18.11 | 28.90 | 21.13 |
| Convenience Store | Dolj | 17.32 | 18.11 | 13.94 | 21.13 |
| Convenience Store | Bucharest | 19.96 | 18.11 | 22.09 | 21.13 |
| Convenience Store | Brasov | 22.53 | 18.11 | 28.34 | 21.13 |

**Mini Market | Calarasi is the worst case by a wide margin** (NMF regression L1=63.10, vs. 17.32-22.71 everywhere else) - consistent with a known failure mode of linear extrapolation, not a bug: when this store is held out, its catchment features (smallest population, rural, in this dataset) sit outside the range the remaining 5 stores' regression was fit on, and linear models are least reliable exactly there.

## Why regularization couldn't fully rescue this

Across all 6 leave-one-out folds, 33 of 72 topic-fits (6 stores x 12 topics) had their selected ridge alpha land at or near the search grid's upper bound (1e8) - the cross-validation procedure repeatedly concluding that the safest prediction for that topic is close to the training stores' *mean* prevalence, effectively discarding the catchment features rather than trusting them. That's the regularization working as intended, not failing - but it also means a meaningful fraction of this model's "explainable relationship" is, honestly, closer to "just predict the average" once properly regularized at this sample size.

## Per-topic strongest predictor (NMF, averaged across folds)

| Topic | Strongest catchment predictor | Coefficient (standardized) |
|---|---|---|
| T0 (Vegetables + Household Products + Fruits) | pct_female | -0.13 |
| T1 (Dessert Drinks) | PC1 | -0.05 |
| T2 (Bake Off) | PC1 | +0.14 |
| T3 (Sweet Snacks) | PC2 | -0.05 |
| T4 (Beer + Tobacco) | PC1 | -1.11 |
| T5 (Dessert Ingredients + Rice / Beans / Pasta + Sugar) | PC2 | -0.25 |
| T6 (Yogurt  / Sana / Kefir, Dessert + Cheese Packed + Packed Sausages) | PC1 | +0.28 |
| T7 (Water + Carbonated Drinks) | PC2 | +0.47 |
| T8 (Ice Cream And Desserts) | pct_female | -0.22 |
| T9 (Pet  Food) | PC2 | -0.13 |
| T10 (Salty Snacks + Chocolate + Carbonated Drinks) | PC1 | -0.26 |
| T11 (Coffee Corner + Tobacco + Others Fiigo) | PC1 | -0.61 |

These coefficients are the actual deliverable this method was built to produce - an inspectable reason, not just a number - but given the regularization finding above and the small-n instability demonstrated throughout, they should be read as a rough directional signal, not a precise, stable effect size. Re-fitting on a larger analyzed-store set would very plausibly change several of these.

## Verdict

Not recommended over k=3 nearest-neighbor propagation at this sample size - the accuracy cost is real and substantial, and even the "explainability" benefit this method was built for is undercut by how often the fitted model falls back to near-constant predictions once honestly regularized. The idea itself (an explicit catchment-to-mission relationship) remains a reasonable one to revisit if the analyzed-store set grows meaningfully past 6 - the failure mode here is a small-n problem, not a conceptual one.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_regression_test.py`