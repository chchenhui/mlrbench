## Name

benchmark_lifecycle

## Title

The Lifecycle of ML Benchmarks: Quantifying and Counteracting Dataset Aging

## Short Hypothesis

Static ML benchmarks gradually lose challenge and relevancy over time as models saturate performance and real-world data distributions evolve; by quantifying this 'benchmark decay' and introducing lightweight data rejuvenation techniques, we can dynamically restore challenge levels and extend the useful lifespan of benchmarks without extensive human reannotation.

## Related Work

Existing work on concept and domain drift (e.g., TWIN-ADAPT) focuses on streaming or evolving data in IoT/CPS systems, not on static benchmark datasets. Prior studies have noted 'benchmark saturation' in specific tasks, but lack a unified quantitative framework for measuring decay or for automated rejuvenation. Adversarial and synthetic data augmentation methods generate hard examples but are not tied to benchmark aging. Our proposal bridges these gaps by (1) defining decay metrics for static benchmarks over time and (2) designing an automated, GAN-based rejuvenation pipeline, distinguishing it from trivial domain adaptation or adversarial augmentation.

## Abstract

Machine learning benchmarks like MNIST, CIFAR-10, ImageNet, GLUE, and SQuAD have driven rapid model improvements, but as architectures advance and real-world data shifts, these static benchmarks risk becoming overly saturated and less representative. We introduce the concept of 'benchmark decay,' a measurable decline in a dataset's capacity to discriminate between competitive models over time. First, we propose a suite of decay metrics—including performance saturation gap, year-over-year challenge drop, and distributional shift indices—applied retrospectively to five canonical benchmarks using historical leaderboard data. Second, we present a lightweight rejuvenation pipeline that generates a small set of new, challenging test samples via conditional generative models, targeted at regions of high model uncertainty. Finally, we evaluate the efficacy of this approach by measuring post-rejuvenation model ranking shifts, challenge gap recovery, and human-perceived realism. Our results demonstrate that (1) benchmark decay is a widespread phenomenon across vision and language tasks, and (2) targeted synthetic rejuvenation can restore benchmark discriminative power with less than 5% additional test data, suggesting a practical path to sustainable, dynamic benchmarking.

## Experiments

- Decay Quantification: Collect historical top-1 and top-5 accuracy (or F1) from leaderboard archives for MNIST, CIFAR-10, ImageNet, GLUE, and SQuAD over the past 10 years. Compute metrics: saturation gap (difference between human and model performance), annual challenge drop rate, and dataset–model distribution divergence (e.g., via feature embeddings).
- Rejuvenation Pipeline Development: Train conditional GANs (e.g., StyleGAN2 for vision, conditional GPT-2 for text) on each benchmark’s training set. Generate candidate samples by targeting regions of highest ensemble model uncertainty (measured via entropy over a committee of recent SOTA models).
- Sample Selection and Integration: Use automatic quality filters (e.g., Fréchet Inception Distance for images, perplexity constraints for text) to retain 200–500 high-quality, challenging samples. Integrate these into the existing test set as 'rejuvenated benchmarks.'
- Evaluation: Re-run benchmarking experiments with recent SOTA models on original vs. rejuvenated test sets. Metrics: change in model ranking (Kendall’s tau), recovery of challenge gap (difference in accuracy drop), and human evaluation of sample realism (via crowdworkers, 100 samples per task). Compare with random sample addition and expert-curated samples.
- Ablations: Vary the size of synthetic additions (1%, 3%, 5% of test set) and uncertainty thresholds to study trade-offs between annotation effort and challenge restoration.

## Risk Factors And Limitations

- Synthetic sample realism may be insufficient, leading to unrealistic or out-of-distribution test cases.
- Performance shifts could reflect model overfitting to generator artifacts rather than true challenge restoration.
- Historical leaderboard data may be incomplete or biased toward certain model families.
- GAN training and sample filtering introduce additional complexity and computational cost, which may not scale to very large benchmarks.

