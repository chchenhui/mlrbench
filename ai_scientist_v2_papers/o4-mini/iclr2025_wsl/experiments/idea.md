## Name

weight_primitives

## Title

Learning Compositional Weight Primitives for Neural Model Synthesis

## Short Hypothesis

Neural network weight collections from a model zoo can be efficiently represented as sparse linear combinations of a small set of shared 'weight primitives'. By learning a dictionary over flattened weight vectors, we can reconstruct, interpolate, and synthesize new models that match the performance of fully trained networks, enabling rapid model generation and transfer learning without full training from scratch.

## Related Work

Prior work applies dictionary learning to data patches (e.g., images) and uses meta-learning or hypernetworks to predict weights. Model soups merge full weights but lack factorization. To our knowledge, no existing method learns a basis directly in weight space for compositional synthesis. Our proposal differs by treating the model zoo as a dataset of weight vectors, learning an explicit sparse dictionary of weight primitives, and using sparse coding to generate novel weight configurations.

## Abstract

We introduce a new paradigm for neural model synthesis by treating trained network weights as data and learning a compact dictionary of 'weight primitives' that capture shared building blocks across models. Given a collection of pre-trained neural networks (the model zoo), we flatten their weight tensors into high-dimensional vectors and apply sparse dictionary learning (e.g., K-SVD or learned analysis transforms) to discover a small basis set. Any new model's weights can then be approximated by a sparse combination of these primitives. We demonstrate that models synthesized via sparse codes achieve comparable accuracy on vision benchmarks (CIFAR-10/100) with only a fraction of the training cost. Our method also enables smooth interpolation between models, rapid transfer to downstream tasks by updating sparse coefficients, and insight into weight space structure. We evaluate reconstruction error, classification performance of synthesized models, and convergence speed in fine-tuning scenarios. This approach opens new avenues for weight space manipulation, meta-learning, and democratized model generation without heavy computational resources.

## Experiments

- Dataset & Model Zoo: Train 50 small CNNs (e.g., ResNet-18, VGG variants) on CIFAR-10 and CIFAR-100 with different seeds and regularization settings. Flatten each model's weights into a vector.
- Dictionary Learning: Apply K-SVD (scikit-learn) and a trainable autoencoder with ℓ₁ sparsity penalty to learn dictionaries of varying sizes (e.g., 50 to 500 atoms) on the flattened weight vectors. Measure average reconstruction error (ℓ₂ norm).
- Model Synthesis via Sparse Coding: For held-out target architectures, estimate sparse codes over the learned dictionary to reconstruct weights. Evaluate synthesized models (zero additional gradient steps) by top-1 accuracy on CIFAR-10/100.
- Fine-tuning from Sparse Initialization: Initialize new models with reconstructed weights and fine-tune for 10 epochs. Compare convergence speed and final accuracy against random initialization and standard pre-training baselines.
- Interpolation & Transfer: Linearly interpolate sparse codes between two trained models to generate intermediate models. Evaluate performance smoothness. For few-shot transfer tasks (e.g., CIFAR-10 → SVHN), fix dictionary and learn sparse codes on limited data; compare to standard transfer learning.
- Ablation & Metrics: Vary dictionary size, sparsity level, and atom selection method. Report reconstruction error, test accuracy, convergence epochs, and sparsity statistics.

## Risk Factors And Limitations

- High dimensionality of weight vectors may make dictionary learning computationally heavy for large models; initial experiments focus on small-to-medium networks.
- Flattening weights ignores tensor structure and symmetries (e.g., channel permutations); future work could incorporate equivariant factorization.
- Learned primitives may overfit to the training model zoo and not generalize to very different architectures or tasks.
- Sparse coding optimization at inference may be slower than direct weight loading; need efficient solvers.
- Quality of synthesized models depends on dictionary expressivity; too small a dictionary may underfit, too large may lose compositional benefits.

