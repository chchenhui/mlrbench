## Name

meta_data_sampler

## Title

Learning to Value Data: A Meta-Learned Sampler for Efficient and Fair Pre-Training of Foundation Models

## Short Hypothesis

A meta-learned data valuation network (DVN) can predict each unlabeled sample’s contribution to held-out performance using lightweight features (loss, gradient norm, representation metrics), enabling adaptive sampling that accelerates pre-training, improves generalization, and mitigates bias more effectively than existing selection methods. This setting—foundation model pre-training on massive unlabeled corpora—is the ideal testbed because both data scale and compute cost make traditional influence-based or coreset methods infeasible.

## Related Work

Data Shapley (Ghorbani & Zou, NeurIPS 2019), GLISTER (Killamsetty et al., ICML 2020), and CRAIG (Mirzasoleiman et al., NeurIPS 2020) compute per-sample importance via costly influence or gradient‐based procedures that do not scale to the multi-billion-token regime. Active learning and coreset selection also target data efficiency but assume small models or labeled data. In contrast, our DVN amortizes valuation cost via a learned predictor, scales to foundation-scale corpora, and jointly optimizes for efficiency and fairness during pre-training.

## Abstract

Foundation models rely on immense unlabeled corpora, yet not all samples equally benefit pre-training: some accelerate convergence or reduce downstream error more than others, while others introduce bias. We propose a Meta-Learned Data Valuation Network (DVN) that predicts each token-sequence sample’s expected contribution to held-out performance using lightweight per-sample features (e.g., current loss, gradient norm, embedding diversity). During pre-training, the DVN is periodically updated using occasional ground-truth contribution measurements—actual loss reduction on a small held-out set computed for mini-batches—and then used to drive adaptive sampling probabilities. This amortizes the high cost of influence calculations and scales to billions of tokens. We demonstrate on GPT-2 small pre-trained over WebText that our adaptive sampler reduces the number of tokens required to reach target perplexities by up to 30% compared to random or gradient-norm baselines, while also improving zero-shot accuracy on LAMBADA and PIQA. Furthermore, by feeding subgroup-specific contribution signals into the DVN, we show it can adjust sampling to reduce fairness disparities (e.g., toxic vs. non-toxic text) with minimal loss in overall language modeling quality. Our contributions are: (1) a scalable, meta-learning-based data valuation framework for foundation model pre-training; (2) empirical evidence of improved efficiency and fairness; and (3) analyses of DVN design choices and trade-offs.

## Experiments

- Dataset & Model: Use a 100M-token subset of OpenWebText and BookCorpus; GPT-2 small (124M parameters).
- Baselines: Random sampling; gradient-norm based sampling; GLISTER coreset selection (ICML 2020).
- Metrics: Number of tokens to reach perplexity thresholds (20, 30); zero-shot accuracy on LAMBADA and PIQA; fairness metrics—perplexity disparity between toxic vs. non-toxic text subsets.
- DVN Training: Every N=5k update steps, compute ground-truth contributions by measuring held-out loss reduction for small mini-batches (e.g., 128 samples). Train DVN to predict these values from per-sample features.
- Adaptive Sampling: Use softmax of DVN predictions to sample next training batch; compare to uniform and gradient-norm weighting.
- Ablations: Frequency of DVN updates (N ∈ {1k, 5k, 10k}); feature subsets (loss only, loss+grad-norm, full feature set); effect of DVN capacity (MLP depth/width).
- Multi-Modal Extension: Apply the same pipeline to a small Vision-Language model (e.g., ViT+GPT2) using the CC12M image–caption dataset; evaluate image–caption retrieval accuracy and generation quality (BLEU, CLIPScore).

## Risk Factors And Limitations

- DVN update overhead (ground-truth measurements) may offset sampling efficiency gains if done too frequently.
- Contribution labels are noisy estimates, which could degrade DVN performance or lead to suboptimal sampling.
- Results on GPT-2 small may not directly scale to much larger models without further engineering or distributional shifts.
- Fairness‐focused sampling may trade off overall model quality if subgroup signals are noisy or imbalanced.

