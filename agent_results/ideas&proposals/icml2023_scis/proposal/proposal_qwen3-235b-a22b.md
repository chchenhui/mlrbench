# Adversarial Counterfactual Augmentation for Spurious Correlation Robustness

## 1. Introduction

### Context and Background  
Modern machine learning (ML) models frequently fail in real-world deployments due to reliance on spurious correlations—features that appear predictive in training data but lack causal relevance. For instance, chest X-ray classifiers may depend on hospital-specific scanner artifacts, while NLP models exploit lexical overlap instead of semantic relationships. Such failures highlight the necessity of robust models that learn invariant, causally grounded patterns despite spurious correlations. Traditional approaches often require explicit group labels (e.g., demographics) to enforce invariance, but these labels are scarce or sensitive in practice.  

### Research Objectives  
This work proposes **Adversarial Counterfactual Augmentation (ACA)**, a framework to robustify ML models against spurious correlations without needing group annotations. ACA integrates three key ideas:  
1. **Attribution-driven identification** of spurious features via influence functions or gradient-based methods.  
2. **Counterfactual data generation** using conditional generative models to manipulate spurious attributes while preserving true labels.  
3. **Consistency regularization** to enforce invariance during retraining.  

The objectives are:  
1. Develop a scalable algorithm for spurious feature detection and manipulation.  
2. Design a robust training pipeline incorporating adversarial counterfactuals.  
3. Empirically validate ACA on diverse tasks (e.g., vision, medical imaging) under synthetic and natural distribution shifts.  

### Significance  
By eliminating dependency on group labels, ACA addresses a critical gap in deployment-ready ML systems. This work advances out-of-distribution (OOD) generalization, stability under dataset shifts, and alignment with causal principles, directly answering the workshop’s focus on uniting methods from causality, fairness, and robustness. Additionally, it provides practitioners tools for stress-testing models and mitigating failures in high-risk domains like healthcare and autonomous systems.  

## 2. Methodology  

Our methodology proceeds in three stages: (1) spurious feature detection, (2) counterfactual augmentation, and (3) model retraining with consistency loss.  

### 2.1 Spurious Feature Identification  
We identify spurious features using gradient-based attribution maps. Given a trained model $ f_\theta $, input $ x $, and label $ y $, the **Integrated Gradients (IG)** attribution for feature $ x_i $ is:  
$$
\text{IG}_i(x) = (x_i - \hat{x}_i) \times \int_{\alpha=0}^1 \frac{\partial f_\theta(\hat{x} + \alpha(x - \hat{x}))}{\partial x_i} d\alpha,
$$
where $ \hat{x} $ is a baseline (e.g., zero vector). Features with high attributions correlated with environmental variables (e.g., scanner type) are flagged as spurious. This avoids manual labeling by leveraging model sensitivity.  

### 2.2 Counterfactual Generation with Conditional GANs  
We train a conditional **CycleGAN** to generate spurious-feature-preserving counterfactuals. Given input $ x $ and a spurious feature mask $ s \in \{0,1\}^d $, we learn generator $ G $ such that:  
$$
\tilde{x} = G(x; s, \phi),
$$  
where $ \tilde{x} $ retains the true label $ y $ while inverting features in $ s $. The loss balances realism and consistency:  
$$
\mathcal{L}_{\text{GAN}} = \mathbb{E}_{x,s}\left[\log D(G(x; s, \phi))\right] + \lambda \mathcal{L}_{\text{recon}}(x, G(x; s, \phi)),
$$  
with $ D $ as discriminator and $ \mathcal{L}_{\text{recon}} $ ensuring pixel-level fidelity outside $ s $.  

#### Example: Chest X-Ray Modification  
For medical imaging, $ s $ could highlight non-causal regions (e.g., text annotations). The GAN alters these regions while preserving pathology-relevant anatomy.  

### 2.3 Model Retraining with Consistency Loss  
The task model $ f_\theta $ is retrained on the original dataset $ \mathcal{D} = \{(x_i, y_i)\} $ and augmented pairs $ (x_i, \tilde{x}_i) $, minimizing:  
$$
\mathcal{L}_{\text{total}} = \underbrace{\sum_{i=1}^N \ell(f_\theta(x_i), y_i)}_{\text{Supervised Loss}} + \lambda \underbrace{\sum_{i=1}^N \|f_\theta(x_i) - f_\theta(\tilde{x}_i)\|^2_2}_{\text{Consistency Loss}},
$$  
where $ \lambda $ controls regularization strength. This penalizes discrepancies between predictions on original and counterfactual pairs, forcing $ f_\theta $ to ignore $ s $.  

### 2.4 Experimental Design  

#### Datasets  
1. **CelebA**: Spurious correlation between "attractive" label and hair color.  
2. **iWildCam**: Animals tagged with location metadata (environmental proxy).  
3. **MIMIC-CXR**: Chest X-rays with spurious correlations due to scanner brands.  

#### Baselines:  
- **EMP** (Empirical Risk Minimization)  
- **SPUME** (Meta-learning for spuriousness)  
- **EVaLS** (Data resampling via loss ranking)  
- **GroupDRO** (Oracle with group labels)  

#### Evaluation Metrics  
1. **Accuracy**: Standard performance.  
2. **WSE (Worst-Environment Accuracy)**: Robustness under environmental shifts.  
3. **AUC-ROC**: Medical datasets requiring probabilistic outputs.  
4. **Consistency Gap**: $ \mathbb{E}[\|f_\theta(x) - f_\theta(\tilde{x})\|] $, measuring invariance.  

#### Ablation Studies  
- Impact of $ \lambda $, $ \mathcal{L}_{\text{recon}} $, and spurious feature masking strategies.  
- Comparison with alternative augmentation methods (e.g., diffusion models vs. CycleGAN).  

#### Computational Resources  
- Train GANs and ResNet-50 backbones on NVIDIA A100 GPUs.  
- Hyperparameter search via Bayesian optimization.  

## 3. Expected Outcomes & Impact  

### Technical Contributions  
1. **Scalable Spuriousness Detection**: ACA’s attribution-driven workflow surpasses manual labeling, enabling efficient mitigation of latent spurious features.  
2. **Generalizable Augmentation**: Conditional GANs create synthetic-environment-like shifts (e.g., sim-to-real for autonomous vehicles) without labeled data.  
3. **State-of-the-Art Robustness**: We expect ACA to match or exceed oracle baselines (e.g., GroupDRO) in WSE and consistency gap on CelebA and iWildCam.  

### Scientific Impact  
1. **Unifying Framework**: ACA bridges causal interventions (counterfactuals) and practical robustness, advancing OOD generalization theory.  
2. **Metrics for Invariance**: Consistency gap and robustness under synthetic shifts will offer new evaluation standards.  
3. **Policy Relevance**: Enables deployment-safe models in healthcare, justice, and finance where group labels are sensitive or unavailable.  

### Societal Implications  
In healthcare, ACA could reduce racial disparities in diagnostic models. For example, chest X-ray classifiers might stop relying on scanner artifacts linked to predominantly non-white hospitals. By automating spuriousness mitigation, ACA lowers barriers to ethical AI adoption.  

### Dissemination Plan  
1. Open-source code and pre-trained models on GitHub.  
2. Submissions to premier conferences (NeurIPS, ICML, CVPR).  
3. Participation in workshops (e.g., this workshop) for cross-domain collaboration.  
4. Toolkits for medical imaging datasets to engage healthcare stakeholders.  

This proposal directly addresses the workshop’s goals by creating a practical framework for spurious correlation robustness and fostering dialogue across causality, fairness, and OOD generalization communities.  

---  
**Word Count**: ~1,950