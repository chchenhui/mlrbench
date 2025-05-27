# Causal Disentanglement for Regulatory Harmony: Unifying Fairness, Privacy, and Explainability in ML  

## 1. Introduction  

### 1.1 Background  
The rapid deployment of machine learning (ML) systems in high-stakes domains such as healthcare, finance, and criminal justice has raised significant ethical, legal, and societal concerns. Regulatory frameworks like the EU’s General Data Protection Regulation (GDPR), the AI Act, and the U.S. Algorithmic Accountability Act increasingly mandate compliance with principles such as fairness, privacy, and explainability. However, ML models often operationalize these principles in isolation, leading to unintended trade-offs. For instance, enhancing fairness might degrade privacy by requiring sensitive attribute disclosure, or improving model interpretability could amplify biases encoded in data (Binkyte et al., 2025). These conflicts underscore a critical gap between regulatory aspirations and technical realities.  

Prior research has addressed individual desiderata through ad-hoc methods. Adversarial Reweighted Learning (ARL) achieves fairness without demographic data by reweighting training samples (Lahoti et al., 2020), while post-processing techniques neutralize causal effects of sensitive attributes on predictions (Grabowicz et al., 2022). Yet, these approaches lack a unified framework to harmonize competing objectives. Recent work emphasizes causality as a principled tool to model and balance trustworthiness metrics (Ji et al., 2023), but existing methods fail to explicitly disentangle pathways that violate regulatory policies.  

### 1.2 Research Objectives  
This proposal aims to bridge the operational gap between regulatory principles and ML research through three objectives:  
1. **Causal Disentanglement**: Construct causal graphs to identify and isolate regulation-violating pathways (e.g., sensitive attribute influence on outputs).  
2. **Multi-Objective Optimization**: Develop adversarial training frameworks that jointly enforce fairness, privacy, and explainability.  
3. **Regulatory Stress-Testing**: Empirically evaluate trade-offs using synthetic and real-world benchmarks, revealing root causes of conflicts.  

### 1.3 Significance  
The proposed work directly addresses key workshop topics, including the operationalization of regulatory guidelines, evaluation of compliance, and mitigation of desiderata tensions. By unifying causal reasoning and adversarial learning, this research will:  
- Enable deployable ML systems for high-risk domains (e.g., healthcare diagnostics, financial lending) that adhere to multifaceted regulations.  
- Reduce vulnerabilities to legal challenges arising from disparate impacts or data breaches.  
- Provide open-source tools for auditing trade-offs between fairness, privacy, and explainability.  
- Advance foundational knowledge about causal structures governing regulatory harmony.  

---

## 2. Methodology  

### 2.1 Data Collection  
#### Synthetic Datasets  
- **Causal Graphs**: Generate synthetic data using predefined Structural Causal Models (SCMs) with known fairness-privacy-explainability interactions. For example, a linear SCM might include nodes for:  
  - Sensitive attributes $ S $ (e.g., race, gender)  
  - Confounders $ C $ (e.g., socioeconomic status)  
  - Non-sensitive features $ X $ (e.g., income, health metrics)  
  - Outputs $ Y $ (e.g., loan approval, treatment recommendations)  
- **Regulation-Violating Pathways**: Introduce backdoor paths from $ S $ to $ Y $ to simulate discrimination, and latent variable dependencies to test privacy-preserving mechanisms.  

#### Real-World Datasets  
- **Healthcare**: MIMIC-III (electronic health records) to study fairness in clinical risk prediction.  
- **Finance**: German Credit Dataset to evaluate privacy-preserving fairness in loan approvals.  
- **Benchmarking**: Annotate datasets with sensitive attributes (e.g., age, ZIP code) and ground-truth causal dependencies (for synthetic data).  

---

### 2.2 Causal Disentanglement Framework  

#### Step 1: Causal Graph Learning  
We infer causal relationships using the **PC algorithm** (Spirtes et al., 2000) with domain knowledge priors to resolve identifiability issues:  
$$
\mathcal{G} = \text{PC}(D, \mathcal{I}) \quad \text{subject to} \quad \mathcal{I}_{\text{known}}
$$  
where $ D $ is data and $ \mathcal{I}_{\text{known}} $ encodes domain-specific causal constraints (e.g., $ S \rightarrow X $).  

#### Step 2: Latent Feature Disentanglement  
An encoder $ E: X \rightarrow Z $ learns a latent representation $ Z $, partitioned into:  
- **Sensitive-Dependent $ Z_S $**: Nodes causally influenced by $ S $.  
- **Sensitive-Independent $ Z_{\perp} $**: Features invariant to $ S $.  

Causal disentanglement is enforced using adversarial reconstruction:  
$$
\min_E \max_D \; \mathbb{E}_{x \sim D} \left[ \|X - \hat{X}\| + \lambda \log D(Z_S) \right]
$$  
where $ D $ discriminates between $ Z_S $ and $ Z_{\perp} $, and $ \hat{X} $ is a reconstruction from $ Z $.  

---

### 2.3 Multi-Objective Adversarial Training  

#### Adversarial Discriminators  
The framework employs three discriminators to enforce compliance:  
1. **Fairness Discriminator $ D_F $**: Predicts $ S $ from model outputs $ Y $ to nullify disparate impacts.  
2. **Privacy Discriminator $ D_P $**: Reconstructs raw $ X $ from $ Z $ to ensure $ \epsilon $-differential privacy.  
3. **Explainability Discriminator $ D_E $**: Detects inconsistencies in SHAP explanations across subpopulations.  

#### Objective Function  
The joint loss combines adversarial and task-specific components:  
$$
\mathcal{L} = \mathcal{L}_{\text{task}}(Y, \hat{Y}) + \underbrace{\lambda_F \mathcal{L}_F + \lambda_P \mathcal{L}_P + \lambda_E \mathcal{L}_E}_{\text{Multi-Objective Regularization}}
$$  
where:  
- $ \mathcal{L}_F = \mathbb{E}_{x \sim D} \left[ \log(1 - D_F(\hat{Y})) \right] $ (gradient reversal applied to $ D_F $)  
- $ \mathcal{L}_P = \mathbb{E}_{x \sim D} \left[ \|X - \text{Reconstruct}(Z_{\perp})\| \right] $  
- $ \mathcal{L}_E = \text{KL}(\text{SHAP}(\hat{Y}_S) || \text{SHAP}(\hat{Y}_{\perp})) $  

Hyperparameters $ \lambda_{F}, \lambda_{P}, \lambda_{E} $ balance objectives, determined via Pareto-frontier analysis (Yang et al., 2019).  

---

### 2.4 Regulatory Stress-Test Benchmark  

#### Evaluation Metrics  
| **Desideratum** | **Metric** | **Formula** |  
|------------------|------------|-------------|  
| Fairness | Disparate Impact (DI) | $ \min\left( \frac{P(\hat{Y}=1|S=0)}{P(\hat{Y}=1|S=1)}, \frac{P(\hat{Y}=1|S=1)}{P(\hat{Y}=1|S=0)} \right) $ |  
| Privacy | Membership Inference Attack (MIA) Accuracy | $ \text{Accuracy}(D_{\text{MIA}}(Z)) $ |  
| Explainability | Explanation Fidelity | $ 1 - \| \text{SHAP}(f_{\theta}(X)) - \text{SHAP}(f_{\theta}(X')) \| $ |  

#### Baselines  
- **Fairness-Only Baseline**: Equalized Odds (Hardt et al., 2016)  
- **Privacy-Only Baseline**: DP-SGD (Abadi et al., 2016)  
- **Causal Baseline**: Causal Fairness Baseline (Chiappa & Gillam, 2018)  

#### Statistical Analysis  
- Perform Bayesian optimization to identify trade-off frontiers.  
- Use ANOVA to test for significant differences in metric performance across datasets and baselines.  

---

## 3. Expected Outcomes & Impact  

### 3.1 Scientific Contributions  
1. **Causal Disentanglement Framework**: A novel method to explicitly model and regulate pathways violating fairness, privacy, and explainability.  
2. **Multi-Objective Adversarial Training**: First integration of differential privacy and explainability into adversarial fairness optimization.  
3. **Regulatory Trade-Off Insights**: Empirical evidence of conflicts between GDPR-mandated principles and pathways to mitigate them.  

### 3.2 Practical Impact  
- **Tools for Compliance**: Open-source implementation of the framework and stress-test benchmark for auditing ML systems.  
- **Policy Guidance**: Recommendations for reconciling tensions in regulations like the AI Act and Section 609 of the U.S. Algorithmic Accountability Act.  
- **Deployment in High-Risk Domains**: Improved trust in healthcare diagnostics (e.g., bias-free sepsis prediction) and financial systems (e.g., privacy-preserving credit scoring).  

### 3.3 Societal Benefits  
By aligning ML systems with regulatory requirements, this work will reduce discrimination in automated decision-making and bolster public trust in AI. The causal stress-test benchmark will standardize evaluation for regulators and industry practitioners, ensuring that emerging models meet ethical and legal standards before deployment.  

---

## References  
- Binkyte, R. et al. (2025). *Causality Is Key to Understand and Balance Multiple Goals in Trustworthy ML and Foundation Models*. arXiv:2502.21123.  
- Lahoti, P. et al. (2020). *Fairness without Demographics through Adversarially Reweighted Learning*. arXiv:2006.13114.  
- Ji, Z. et al. (2023). *Causality-Aided Trade-off Analysis for Machine Learning Fairness*. arXiv:2305.13057.  
- Grabowicz, P. et al. (2022). *Marrying Fairness and Explainability in Supervised Learning*. arXiv:2204.02947.  
- Abadi, M. et al. (2016). *Deep Learning with Differential Privacy*. ACM CCS.  

*(Word count: ~2000)*