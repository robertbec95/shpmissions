# Cluster-based (Gower + k-medoids) vs. k-NN propagation

Tests whether pre-clustering all 6 stores via weighted Gower distance + k-medoids (`location_similarity.cluster_stores`, reusing `mission_stage.kmedoids_fit` unchanged - it already works on any precomputed distance matrix) and predicting each store's profile as its cluster-mates' average is competitive with the validated k=3 nearest-neighbor default. Plain k-means could not be used for this - it requires a centroid in the original feature space, which does not exist for mixed categorical/continuous Gower-distance data; k-medoids picks an actual data point as each cluster's center instead. Both ground truths (joint NMF, joint BigCLAM) tested identically. Total run time 38.6s.

## k-NN baseline (k=3, already validated)

| Ground truth | L1 | Cosine | Spearman |
|---|---|---|---|
| Joint NMF | 18.11 | 0.9750 | 0.851 |
| Joint BigCLAM | 21.13 | 0.9773 | 0.810 |

## Cluster-based sweep

| Clusters (C) | NMF L1 | NMF cosine | NMF spearman | BigCLAM L1 | BigCLAM cosine | BigCLAM spearman |
|---|---|---|---|---|---|---|
| 2 | 19.80 | 0.9697 | 0.850 | 22.93 | 0.9730 | 0.718 |
| 3 | 22.03 | 0.9655 | 0.839 | 18.83 | 0.9778 | 0.690 |
| 4 | 22.24 | 0.9647 | 0.850 | 20.18 | 0.9744 | 0.698 |
| 5 | 22.63 | 0.9626 | 0.861 | 21.08 | 0.9731 | 0.690 |

## Cluster assignments by C

**C=2:**
- Cluster 0: Supermarket | Bucharest, Convenience Store | Dolj, Convenience Store | Bucharest
- Cluster 1: Supermarket | Neamt, Mini Market | Calarasi, Convenience Store | Brasov

**C=3:**
- Cluster 0: Convenience Store | Dolj, Convenience Store | Brasov
- Cluster 1: Supermarket | Neamt, Mini Market | Calarasi
- Cluster 2: Supermarket | Bucharest, Convenience Store | Bucharest

**C=4:**
- Cluster 0: Convenience Store | Dolj, Convenience Store | Brasov
- Cluster 1: Supermarket | Neamt
- Cluster 2: Supermarket | Bucharest, Convenience Store | Bucharest
- Cluster 3: Mini Market | Calarasi

**C=5:**
- Cluster 0: Convenience Store | Brasov
- Cluster 1: Supermarket | Neamt
- Cluster 2: Supermarket | Bucharest, Convenience Store | Bucharest
- Cluster 3: Mini Market | Calarasi
- Cluster 4: Convenience Store | Dolj

## Interpretation

Best cluster-based result for NMF ground truth is C=2 (L1=19.80), which does not beat the k-NN baseline (L1=18.11).
Best cluster-based result for BigCLAM ground truth is C=3 (L1=18.83), which beats the k-NN baseline (L1=21.13).
Cluster-based propagation forces every member of a group to share one predicted profile, discarding the fine-grained ranking within a group that k-NN preserves - at n=6 with clusters of 1-3 members, that's a real loss of resolution the k-NN approach doesn't have. The trade-off is interpretability: "this store is in catchment-cluster 2" is a much easier sentence for a retailer to act on than a specific nearest-3 weighted average, even where it costs some accuracy.

**The result genuinely splits by ground truth, not just by C.** For the joint NMF (12 topics), cluster-based propagation never beats k-NN at any C tested - k=3 nearest-neighbor stays the better strategy. For the joint BigCLAM (8 communities), C=3 clustering beats k-NN by a wide margin (18.83 vs 21.13, ~11% relative) on L1, though it's *worse* on Spearman rank agreement (0.690 vs 0.810) - so even the "win" is not a clean one, it trades rank fidelity for lower absolute error.

C=3's structure is worth naming directly: k-medoids found three clean pairs (Dolj+Brasov, Neamt+Calarasi, both Bucharest stores), so each store's cluster prediction is really "my one paired analog's profile," not a genuine multi-store average. That happened to work better for BigCLAM's coarser, larger-share communities than any individually-ranked k-NN average did (better than even k=1 nearest-neighbor's L1 of 21.08) - plausibly because BigCLAM's fewer, broader-share communities are less sensitive to which exact neighbor supplies the estimate, while NMF's finer 12-topic structure has more to lose from collapsing three ranked neighbors down to one fixed partner.

**Practical read:** cluster-based propagation is not a strict upgrade or a strict downgrade - it is a genuinely different, coarser strategy that happens to suit one of the two mission-detection methods tested and not the other. It is not recommended as a blanket replacement for the validated k=3 k-NN default, but it is worth keeping in mind specifically alongside coarser/overlapping-community ground truths like BigCLAM's, and its output ("stores in the same catchment cluster") is a genuinely more explainable artifact even when it isn't the most accurate one.

Reproduce with: `PYTHONPATH=src python3 scripts/run_location_similarity_cluster_test.py`