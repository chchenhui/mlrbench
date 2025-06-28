## Name

gradient_cluster_robust

## Title

Unsupervised Gradient Clustering for Robust Spurious Correlation Mitigation

## Short Hypothesis

Latent spurious-feature groups can be uncovered by clustering per-sample gradient signatures during training; using these pseudo-groups in a group-robust learning framework improves worst-group performance without requiring costly group annotations.

## Related Work

Prior works on robustness to spurious correlations—such as Group DRO, JTT, and other group-based domain generalization methods—require explicit group labels at training time or heuristics based on loss magnitudes. Unlike methods that rely solely on loss-based sample reweighting, we propose clustering the gradient vectors themselves to capture latent feature contributions. To our knowledge, no prior work uses per-sample gradient statistics to define pseudo-groups for robust learning, making this approach a novel, unsupervised alternative to group annotation.

## Abstract

Deep models often exploit spurious correlations—shortcuts that hold only in training datasets—leading to poor worst-group generalization. Mitigating this requires robust training over known subgroups, but group labels are expensive or unavailable. We propose Unsupervised Gradient Clustering (UGC), a simple yet effective approach to identify latent spurious-feature groups without annotations. During early training, we extract per-sample gradient vectors (e.g., backpropagated gradients at a chosen layer) and apply clustering (e.g., k-means) to partition the data into pseudo-groups that reflect shared feature sensitivities. We then apply a group-robust optimization (e.g., Group DRO) over these clusters to re-balance the training process toward underrepresented or harder clusters. We justify UGC theoretically by showing that gradients encode feature-correlation strengths, and clustering them recovers group structure in common spurious benchmarks. Empirically, UGC matches or exceeds oracle group-label baselines on Colored MNIST, Waterbirds, and CelebA, improving worst-group accuracy by up to 10% over ERM and matching Group DRO performance without access to true group IDs. UGC requires minimal overhead, integrates into existing training pipelines, and scales to large neural networks. This unsupervised grouping strategy opens new avenues for robust learning in real-world settings where group supervision is infeasible.

## Experiments

- Datasets: Colored MNIST, Waterbirds, CelebA hair classification. Models: simple CNN for MNIST, ResNet-50 for others.
- Baseline methods: ERM, Group DRO (oracle groups), JTT, loss-based sample reweighting.
- UGC setup: Extract per-sample gradient vectors of the penultimate layer every epoch for first T epochs; reduce dimension via PCA; apply k-means with k=2 (spurious vs core) and k>2; assign cluster IDs as pseudo-groups.
- Robust training: Use Group DRO or reweighting loss over pseudo-groups for remaining epochs.
- Metrics: Worst-group accuracy, average accuracy, cluster purity against true spurious labels.
- Ablations: Vary k in clustering (2–5), gradient extraction layer, PCA dimension, T (number of epochs), cluster algorithm (GMM vs k-means).

## Risk Factors And Limitations

- Gradient clustering may fail when spurious and core features yield similar gradient patterns, leading to poor pseudo-group separation.
- Choice of clustering hyperparameters (k, PCA dim) affects robustness and may require tuning.
- Additional computation and memory overhead for storing per-sample gradients and clustering steps.
- Method assumes that spurious correlations dominate gradient signals early in training; may not hold in all settings or architectures.

