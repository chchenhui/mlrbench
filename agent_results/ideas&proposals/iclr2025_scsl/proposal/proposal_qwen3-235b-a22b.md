# Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS)

## Introduction

### Background and Motivation

Deep learning models exhibit a well-documented vulnerability to spurious correlations, where they rely on superficial patterns unaligned with the true causal mechanisms of the task. This behavior manifests in diverse domains: medical imaging models may base cancer detection on imaging artifacts tied to specific scanner types, while hiring algorithm audits reveal reliance on non-causal demographic indicators. The roots of this failure trace back to deep learning’s inductive biases during data preprocessing, architecture design, and optimization. Traditional empirical risk minimization (ERM) frameworks optimize for minimal training loss without distinguishing between causal and spurious features, leading to poor generalization in real-world scenarios where spurious correlations dissipate.

Current techniques for addressing spurious correlations fall short in three critical areas: (1) reliance on explicit supervision of spurious groups (e.g., requiring known protected attributes for distributionally robust optimization), (2) limited exploration of interventionist mechanisms in unsupervised settings, and (3) insufficient adaptation to unknown spurious patterns. While methods like ElRep (Tao et al., 2025) impose norm penalties on representations, and SPUME (Zheng et al., 2024) utilizes meta-learning for spurious attribute detection, these approaches assume prior domain knowledge about potential spurious sources. The proposed AIFS framework directly attacks this limitation by introducing a synthetic intervention loop that discovers and mitigates *unknown* spurious factors through dynamic gradient analysis and latent representation perturbation.

### Research Objectives

The central aim of this research is to develop and validate AIFS, a universal framework for robust feature learning that:
1. **Automatically detects latent directions associated with spurious correlations** through iterative gradient-based attribution and synthetic intervention.
2. **Enforces invariance to adversarial-style perturbations** in identified spurious dimensions while preserving sensitivity to causal features.
3. **Achieves state-of-the-art robustness metrics** on benchmarks with hidden spurious structure, without requiring group annotations.

### Significance

The impact of AIFS spans multiple axes:
- **Practically**, it addresses the critical need for scalable, unsupervised robustification techniques as articulated in the workshop’s call for solutions when spurious labels are partially or completely unknown.
- **Theoretically**, it bridges gaps in understanding how interventionist paradigms interact with the dynamics of deep learning optimization, particularly in relation to how spurious features affect loss landscape curvature (a challenge highlighted in current foundational work reviewed above).
- **Methodologically**, it introduces a novel training loop that couples invariant risk minimization principles with adaptive representation engineering, extending beyond static regularization and fixed augmentation approaches.

AIFS’ modality-agnostic design directly supports the workshop’s interest in multimodal benchmarks, and its label-free nature aligns with recent calls for scalable robustification methods applicable to foundation models where exhaustive manual annotation becomes infeasible.

---

## Methodology

### System Overview

The core principle behind AIFS involves three interacting components trained end-to-end:
1. **Pretrained Encoder $E(\cdot)$**: Maps inputs $\mathcal{X} \rightarrow \mathbb{R}^d$ into a latent representation $\mathbf{z} \in \mathbb{R}^d$.
2. **Intervention Module $\mathcal{M}(\cdot)$**: Applies structured perturbations to selected subspaces of $\mathbf{z}$ through learned masks $\mathbf{m} \in [0,1]^d$.
3. **Classifier $C(\cdot)$**: Makes predictions $\hat{y}$ while being regularized to suppress spurious feature importance.

The training process follows an alternating optimization protocol with three phases:
1. **Intervention Application**: Randomly sample a batch, apply intervention $M(\cdot)$, and compute loss.
2. **Feature Importance Update**: Calculate gradient-based feature attribution scores.
3. **Mask Reoptimization**: Adjust intervention masks according to the latest attribution scores.

### Technical Components

#### 1. Latent Space Intervention Module

The intervention operates through a multiplicative mask:
$$
\mathbf{z}^{\text{intervened}} = \mathbf{z} + \lambda \cdot (\mathbf{m} \otimes \epsilon),
$$
where $\otimes$ denotes element-wise multiplication, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is isotropic Gaussian noise, and $\lambda$ controls the intervention strength. The mask $\mathbf{m}$ is constrained to preserve important gradients while maximizing disruption to spurious factors.

#### 2. Dual-Objective Loss Function

The system trains on two loss components:
1. **Invariance Loss**: Encourages consistent predictions on original and intervened representations:
$$
\mathcal{L}_{\text{inv}} = \frac{1}{B} \sum_{b=1}^{B} D_{\text{KL}}\left(C(\mathbf{z}_b) \| C(\mathbf{z}_b^{\text{intervened}})\right),
$$
where $D_{\text{KL}}$ denotes Kullback-Leibler divergence.
2. **Sensitivity Loss**: Penalizes over-reliance on dimensions with high mask weights:
$$
\mathcal{L}_{\text{sens}} = -\sum_{j=1}^{d} m_j \cdot |\nabla_{z_j} \log p(y | \mathbf{z})|,
$$
with $m_j$ being the j-th element of mask $\mathbf{m}$.

The full objective becomes:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \alpha \cdot \mathcal{L}_{\text{inv}} + \beta \cdot \mathcal{L}_{\text{sens}},
$$
where $\mathcal{L}_{\text{ce}}$ is the standard cross-entropy loss, and $\alpha, \beta$ control regularization strength.

#### 3. Iterative Feature Attributions and Mask Update

Following each training epoch:
1. Compute Integrated Gradients (Sundararajan et al., 2017) for the top-K classes with respect to the latent representation $\mathbf{z}$:
$$
IG_i = (z_i - z_i^{\text{baseline}}) \cdot \sum_{k=1}^K w_k \cdot \frac{\partial \log p(y_k | \mathbf{z})}{\partial z_i},
$$
where $w_k$ are class weights.
2. Update the intervention mask $\mathbf{m}$ with a softmax weighting across absolute IG values:
$$
m_j \propto \exp\left(\gamma \cdot \frac{1}{|IG_j| + \epsilon}\right),
$$
where $\gamma$ controls mask selectivity.

### Experimental Design

#### Datasets
- **Image Data**: Subset of CelebA with latent spurious hair color/gender correlations (Liu et al., 2015); synthetic colored MNIST with color labels spuriously associated with digit classes.
- **Tabular Data**: Civil Comments for toxicity prediction with latent demographic spurious features, and Compas recidivism data containing racial bias.

#### Benchmarks
- **Baselines**: ERM baseline, IRM (Arjovsky et al., 2019), ElRep (Tao et al., 2025), SPUME (Zheng et al., 2024), RaVL (Varma et al., 2024), ULE (Mitchell et al., 2024).
- **Metrics**: Worst-group accuracy, Worst-group ROC-AUC, Average Accuracy, Equal Opportunity difference ($\leq 1\%$).

#### Ablation Studies
- Mask update frequency (every epoch vs. every 5 epochs)
- Different intervention norms ($\ell_1$ vs $\ell_2$ vs random noise)
- Gradient attribution methods (Integrated Gradients vs Layerwise Relevance Propagation)

#### Implementation Details
- Architecture: ViT-B/16 for image data, RoBERTa-base for text.
- Optimization: AdamW with cosine decay; batch size 256.
- Hyperparameters: $\lambda=0.5$, $\alpha=1.0$, $\beta=0.5$
- Training Budget: Standard 200 epochs with early stopping.

---

## Expected Outcomes & Impact

### Technical Advancements

We anticipate three core contributions from this research:

1. **New Paradigm for Unsupervised Shortcut Learning Mitigation**: By iteratively identifying and perturbing spurious directions in latent space, AIFS will demonstrate superior robustness over baselines on benchmarks where spurious features lack explicit supervision. We hypothesize >2% absolute improvement in worst-group accuracy compared to ElRep and ULE.

2. **Mathematical Formulation of Adaptive Intervention Dynamics**: The gradient-guided mask update rule connects to fundamental learning theory, providing insights into how spurious feature importance evolves during training. Our analysis should reveal differential convergence patterns between spurious and causal features.

3. **Benchmark Expansion and Open-Source Tools**: We will release implementations and synthetic datasets that challenge robust learning algorithms in label-free settings. These resources will fill a gap identified in the workshop’s call for comprehensive benchmarks in understudied domains like healthcare and industrial ML.

### Theoretical Insights

From our experiments, we expect to uncover:
- The existence of an optimal intervention strength $\lambda$ that balances invariance enforcement without collapsing representation quality.
- Distinct training dynamics where causal features exhibit higher "learning robustness"—being acquired later but retained more stably during interventions.
- Empirical validation of the invariance-sensitivity duality principle in deep learning.

### Broader Impacts

This work directly addresses the workshop’s three priorities:
1. **New Evaluation Benchmarks**: We introduce a synthetic spurious correlation benchmark for multimodal tasks through combined image+metadata settings.
2. **Robustification Techniques**: AIFS provides a foundation for applying robustification to larger vision-language foundation models where manual annotation of spurious attributes becomes impractical. For example, adapting our pipeline to robustly fine-tune CLIP (Radford et al., 2021) could prevent shortcut learning in cross-modal retrieval tasks.
3. **Understanding Mechanisms**: Through detailed feature attribution tracking, we will produce empirical evidence regarding how spurious gradients dominate early training—as previously reported theoretically by Geirhos et al. (2020) and recently shown in NLP by Zhao et al. (2023)—but how our interventions can reverse these early biases.

Potential challenges remain in computational scalability for high-dimensional data and the need for careful hyperparameter tuning in complex models. However, by building on the success of gradient-guided attribution and intervening in compressed latent spaces rather than raw inputs, AIFS achieves a manageable computational overhead (estimated <15% increase) compared to base models.

In conclusion, AIFS represents a substantial step forward in the quest for machine learning systems that "learn for right reasons"—leveraging synthetic interventions as both a diagnostic tool and a regularization mechanism without requiring prior knowledge of spurious relationships.