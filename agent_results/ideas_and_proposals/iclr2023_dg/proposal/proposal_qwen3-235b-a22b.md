# Causal Structure-Aware Domain Generalization via Invariant Mechanism Learning

## Introduction

### Background  
Machine learning models often struggle to maintain performance when deployed in environments that differ from those during training. This challenge, known as **domain generalization (DG)**, arises because models frequently exploit **spurious correlations** between features and labels that hold only in the training distribution. Recent benchmarking studies in 2023–2025 have demonstrated that even specialized DG methods fail to outperform the baseline of **empirical risk minimization (ERM)** across diverse real-world tasks, particularly in computer vision applications. The fundamental issue lies in the lack of access to **invariant mechanisms**—principles in the data-generating process that remain stable across domains. In contrast to spurious correlations, causal relationships (e.g., causally related features) define such invariant mechanisms, offering a theoretical foundation for robust generalization.

### Research Objectives  
The goal of this work is to design a **causal structure-aware framework** that:  
1. **Infers domain-specific causal graphs** using environment labels as metadata.  
2. **Distills invariant causal mechanisms** from multi-domain data.  
3. **Enforces causal invariance** during end-to-end training via differentiable constraints.  
4. **Validates generalization performance** on established DG benchmarks (e.g., DomainBed) under realistic distribution shifts.  

By explicitly modeling causal dependencies and penalizing reliance on non-causal features, we aim to bridge the gap between **causal modeling** and **deep learning-based DG**, offering formal guarantees on robustness.

### Significance  
The proposed framework will advance several critical areas:  
- **Theoretical DG foundations**: By linking counterfactual invariance to domain shifts, addressing the conjecture from the workshop.  
- **Practical robustness**: Medical imaging, autonomous driving, and other high-stakes applications require reliability when domain-specific artifacts (e.g., lighting, sensor biases) dominate.  
- **Causal discovery in representation learning**: Prior causal DG approaches (e.g., Contrastive Causal Model (arXiv:2210.02655), CIRL (arXiv:2203.14237)) treat causal mechanisms heuristically; we formalize their integration with modern deep learning.  

## Methodology

### Framework Overview  
The method combines **causal discovery** with **domain-aware representation learning**. It proceeds in three stages:  
1. **Causal graph inference** to identify invariant mechanisms across domains.  
2. **Differentiable regularization** for neural networks to align feature representations with causal invariance.  
3. **Invariant feature distillation** through adversarial or contrastive objectives for stable prediction.  

The overarching hypothesis is that **causal factors (C) satisfy $ p(Y \mid C) $ invariance** across domains, while non-causal features (N) introduce domain-dependent dependencies. We formalize this as:  
$$
Y \perp\!\!\!\perp D \mid C, \quad \text{and} \quad N \perp\!\!\!\perp C,
$$  
where $ D $ denotes domain and $ Y $ is the target label.

### Stage 1: Causal Discovery with Domain-Level Metadata  
Current DG benchmarks (e.g., DomainBed) lack explicit knowledge of domain shifts. However, domain metadata (e.g., labels indicating environments like "hospital A" or "weather condition B") provides weak supervision for causal graph inference.  

#### Algorithmic Steps  
1. **Domain-aware causal graph learning**:  
   - Use domain labels to segment training data into distinct domains $ D_1, D_2, \dots, D_K $.  
   - Apply constraint-based or score-based causal discovery (e.g., **PC algorithm**, **NOTEARS**) to infer a domain-conditional graph $ G^{(k)} $ for each domain $ D_k $.  
   - Identify **invariant causal features (C)** as intersections of ancestral sets of $ Y $ across all $ G^{(k)} $:  
     $$
     \mathcal{C} = \bigcap_{k=1}^K \mathcal{A}^{(k)}(Y),
     $$  
     where $ \mathcal{A}^{(k)}(Y) $ is the set of ancestors of $ Y $ in $ G^{(k)} $.  

2. **Counterfactual validation of causal candidates**:  
   - Generate counterfactual samples by intervening on non-causal features $ N $.  
   - Retain features $ C $ iff $ p(Y \mid \text{do}(N=n)) = p(Y \mid C) $, ensuring insensitivity to $ N $.  
   - This aligns with the **structural counterfactual generation framework** (arXiv:2502.12013).  

3. **Latent causal structure estimation (in unobserved causal features)**:  
   - For high-dimensional inputs (e.g., images), adopt **CausalVAE** (ICML 2024) or **Neural Additive Models** to approximate latent causal variables.  

### Stage 2: Representation Learning with Causal Constraints  
Once causal features $ \mathcal{C} $ are identified, the model architecture must enforce reliance solely on these features.  

#### Neural Network Architecture  
1. **Causal feature extractor**:  
   - Use a backbone neural network $ f_\theta: \mathcal{X} \rightarrow Z $ to map inputs to a latent representation $ Z $.  
   - Split $ Z $ into causal ($ Z_C $) and non-causal ($ Z_N $) partitions via clustering on $ G $.  

2. **Causal invariance loss**:  
   - Penalize domain dependence of $ Z_C $:  
     $$
     \mathcal{L}_C(Z_C, D) = \text{MI}(Z_C, D),
     $$  
     where $ \text{MI} $ is mutual information.  
   - Enforce independence between $ Z_C $ and $ Z_N $:  
     $$
     \mathcal{L}_{\text{split}} = \text{MI}(Z_C, Z_N).
     $$  
   - Total loss combines task loss ($ \mathcal{L}_{\text{task}} $), causal invariance, and contrastive learning for class separation:  
     $$
     \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_C + \lambda_2 \mathcal{L}_{\text{split}} + \lambda_3 \mathcal{L}_{\text{contrastive}}.
     $$  

3. **Interventional robustness**:  
   - For tasks where $ Z_C $ must directly model interventions, add a **counterfactual consistency loss** to ensure stable predictions under feature perturbations:  
     $$
     \mathcal{L}_{\text{cf}} = \mathbb{E}_{\mathcal{D}_{\text{train}}} \left[ \| f_\theta(x) - f_\theta(x'_{\text{do}(Z_N)}) \|_2^2 \right],
     $$  
     where $ x'_{\text{do}(Z_N)} $ is a reconstruction of inputs with $ Z_N $ randomized.

#### Training Details  
- **Differentiable mutual information estimation**: Use InfoNCE loss (DeepMind, 2023) with variational autoencoders to approximate $ \text{MI}(Z_C, D) $.  
- **Contrastive learning component**: For each batch, enforce intra-class cohesion and inter-class separation in $ Z_C $-space.  
- **Backbone**: ViT-B/16 for vision tasks, ResNet-50 for benchmarks.  

### Stage 3: Experimental Validation  

#### Datasets & Benchmarks  
We validate our framework on the following:  
**1. DomainBed (2022 Revision)**: A multi-task DG benchmark with environments (PACS, OfficeHome, TerraIncognita, Colored MNIST).  
- **Colored MNIST**: Introduce spurious correlations between digit identity ($ \mathcal{C} $) and color ($ \mathcal{N} $).  
- **PACS**: Domains include art, cartoon, photo, and sketch (label distribution aligned with arXiv:2210.02655 experiments).  

**2. Real-world autonomous driving dataset**: Synthetic data from the nuScenes dataset (2024) with weather conditions (rain, snow, fog) as domains.  

**3. Medical imaging dataset**: Diabetic Retinopathy detection from Kaggle using hospital sites as domains.  

#### Baselines & Comparators  
- **ERM (Vanilla ResNet/ViT)**: Baseline risk minimization.  
- **State-of-the-art DG methods**:  
  - REx (VREx)  
  - GroupDRO  
  - ERD (Causality Inspired Representation Learning, arXiv:2203.14237).  

#### Evaluation Metrics  
1. **In-distribution (ID) accuracy**: Average accuracy across training domains.  
2. **Out-of-distribution (OOD) accuracy**: Accuracy on held-out domains (e.g., MNIST in DomainBed).  
3. **Causal invariance metrics**:  
   - **Domain dependence of $ Z_C $**: $ \text{MI}(Z_C, D) $ computed via kNN estimators.  
   - **Feature sensitivity**: $ \Delta = \max_{\delta} \|Y - Y_{x'_{\text{do}(N+\delta)}}\| $, measuring invariance under non-causal feature perturbations.  
4. **Stability under unseen spurious shifts**: Evaluate accuracy drop when test domains exaggerate spurious correlations (e.g., high chromaticity in Colored MNIST).

#### Implementation Strategy  
- **Causal graph learning**: Use **CausalDiscoveryToolbox (PyT)** with domain-aware conditional independence tests.  
  - Implement domain-specific conditional mutual information $ I(X; Y \mid D=d) $ to capture local independencies.  
- **Domain-invariant representation learning**: Integrate causal constraints into PyTorch, leveraging automatic differentiation.  
- **Hyperparameter study**: Search over $ \lambda_1, \lambda_2, \lambda_3 $ to balance task loss and causal penalties.  
- **Ablation studies**: Analyze how inclusion of $ \mathcal{L}_C $, $ \mathcal{L}_{\text{split}} $, and $ \mathcal{L}_{\text{contrastive}} $ individually contribute to OOD accuracy.

### Theoretical Analysis  

#### Identifiability Assumptions  
To ensure the learned features $ Z_C $ are truly causal, we assume:  
1. **Causal sufficiency**: No unmeasured confounders affect both $ X $ and $ Y $.  
2. **Faithfulness**: All statistical dependencies arise from the causal structure.  
3. **Domain-dependent noise**: Non-causal features $ Z_N $ have dependencies on $ D $.  

This aligns with the **structural causal model framework** (Pearl, 2022 extended by arXiv:2203.14237), where interventions on $ Z_N $ leave $ p(Y \mid Z_C) $ unchanged.  

#### Generalization Guarantees  
Let $ \mu_1, \dots, \mu_K $ be training domains and $ \mu^* $ an unseen domain. Assuming $ \mathcal{C} $ is correctly identified, we derive a generalization bound (similar to Muandet et al., 2013),  
$$
\mathbb{E}_{\mu^*}[|\hat{Y} - Y|] \leq \mathbb{E}_{\mu^* \mid \mathcal{C}}[|\hat{Y} - Y|].
$$  
This implies the model generalizes under structural shifts by focusing strictly on $ \mathcal{C} $.

### Expected Outcomes & Impact  

#### Core Deliverables  
1. A **domain generalization framework** that combines **causal discovery** with **feature distillation**, significantly outperforming ERM baselines.  
   - Target: Improve OOD accuracy on DomainBed by at least 5% compared to current methods.  
   - Validate the 2013 DICA assumption of a common conditional distribution through causal-aware feature partitioning.  

2. **Theoretical insights**:  
   - Demonstrate the necessity of explicit domain metadata for discovering $ \mathcal{C} $.  
   - Quantify the relationship between $ \text{MI}(Z_C, D) $ and generalization error.  

3. **Empirical tools**:  
   - Publicly released code for causal graph learning and feature alignment.  
   - Reproducible benchmarks for studying causal DG in vision and medical imaging.  

#### Broader Impact  
1. **High-stakes AI deployment**: Enabling robust models for medical imaging where domain-specific artifacts (e.g., imaging protocols) vary between training hospitals.  
2. **Autonomous systems**: Safer predictions for self-driving vehicles under diverse environmental conditions (rain, fog).  
3. **Advancing DG theory**: Our results would support the workshop’s conjecture that "additional information" (here, domain labels) is required for general-purpose DG, inspiring research into metadata-aware learning.  

4. **Computational feasibility**: By integrating causal discovery algorithms with deep learning via differentiable proxies (arXiv:2502.12013), we mitigate historical concerns about the scalability of structural causal approaches.  

## Timeline and Resources  

| Phase | Duration | Deliverables |  
|-------|----------|------------|  
| Causal graph learning development | 3 months | Prototype graph discovery, codebase on DomainBed |  
| Causal-aware representation learning | 4 months | Implementation of $ \mathcal{L}_C, \mathcal{L}_{\text{split}}, \mathcal{L}_{\text{cf}} $, first OOD results. |  
| Benchmarking and ablation studies | 2 months | Final results, publication-grade figures. |  
| Paper writing and reproducibility | 1 month | Open-source repository with trained models. |  

## Conclusion  

This proposal addresses the central question of the workshop: **What is required for domain generalization to succeed?** We argue that incorporating domain-level metadata into causal discovery and imposing constraints on learned representations is pivotal. This approach not only aligns with prior work on invariant learning (Muandet et al., 2013; CIRL, 2022) but advances it through formal differentiable integration with deep networks. Empirical evaluation on synthetic and real-world benchmarks will demonstrate whether causal-aware DG can reliably solve distribution shifts where ERM fails.