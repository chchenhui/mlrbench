# RobustMDiff: Hierarchical Multimodal Diffusion Models for Robust Diagnostic Predictions with Missing Clinical Data

## Introduction

Healthcare diagnostics rely significantly on multiple forms of clinical data, including medical imaging (X-rays, CT scans, MRIs), electronic health records (EHRs), clinical notes, lab results, and genomic information. These diverse data modalities collectively provide a comprehensive view of a patient's health status, enabling more accurate diagnoses. However, current artificial intelligence systems in healthcare typically process these modalities independently, failing to capture critical cross-modal correlations that could enhance diagnostic accuracy. This siloed approach limits the potential of AI systems to mimic the holistic diagnostic reasoning employed by healthcare professionals.

The challenge becomes even more pronounced when dealing with rare diseases and underrepresented patient populations. These cases often lack sufficient training data across all modalities, leading to poor diagnostic performance precisely for those patients who might benefit most from advanced computational assistance. Additionally, in real-world clinical settings, data is frequently incomplete, with certain modalities missing or containing significant noise due to various factors such as equipment limitations, data collection procedures, or patient-specific constraints. These practical limitations have hindered the widespread adoption of AI systems in clinical practice.

Recent advancements in deep generative models, particularly diffusion models, have shown remarkable potential for synthesizing and processing complex data distributions. Diffusion models have demonstrated unprecedented capabilities in generating high-quality outputs across various domains. In the medical field, models like MedM2G (Zhan et al., 2024) and MedCoDi-M (Molino et al., 2025) have begun to explore multimodal medical data generation. Meanwhile, DiffMIC (Yang et al., 2023) and MedSegDiff (Wu et al., 2022) have applied diffusion models to specific medical imaging tasks. However, these approaches have not fully addressed the challenges of robustness to missing modalities and effective multimodal integration for diagnostic purposes, especially in the context of rare diseases.

This research proposes RobustMDiff, a novel hierarchical multimodal diffusion model specifically designed for robust healthcare diagnostics that can effectively process diverse clinical data types while maintaining resilience when certain modalities are missing. Our approach introduces several innovations: (1) a hierarchical architecture with modality-specific encoders and a unified latent space for diffusion processes, (2) a medical knowledge-enhanced attention mechanism to prioritize clinically relevant patterns, (3) an adaptive training strategy with deliberate modality masking to build robustness, and (4) an explainability framework to provide interpretable predictions through modality-specific feature attribution maps.

The significance of this research lies in its potential to address critical challenges in medical AI systems: enhancing diagnostic accuracy for rare conditions, maintaining robust performance despite missing data, providing transparent and interpretable predictions, and ultimately bridging the gap between advanced AI methodologies and practical clinical implementation. By developing a more resilient and interpretable approach to multimodal medical data processing, this work aims to contribute meaningfully to improving healthcare outcomes, particularly for underserved and challenging diagnostic scenarios.

## Methodology

### 3.1 Model Architecture

RobustMDiff employs a hierarchical architecture with three main components: modality-specific encoders, a multimodal integration module, and a conditional diffusion model operating in a unified latent space.

#### 3.1.1 Modality-Specific Encoders

We design separate encoders for each data modality to account for their unique characteristics:

1. **Imaging Encoder** $E_{img}$: For medical images (MRI, CT, X-ray), we employ a vision transformer (ViT) architecture enhanced with medical-specific attention mechanisms. For an input image $I$, the encoder produces a feature representation:
   $$H_{img} = E_{img}(I) \in \mathbb{R}^{d_h}$$
   
2. **Text Encoder** $E_{text}$: For clinical notes and reports, we utilize a clinical language model fine-tuned on medical text. For input text $T$, we generate:
   $$H_{text} = E_{text}(T) \in \mathbb{R}^{d_h}$$
   
3. **Structured Data Encoder** $E_{struct}$: For lab results, vital signs, and discrete EHR data, we implement a multi-layer perceptron with normalization layers. For structured data $S$:
   $$H_{struct} = E_{struct}(S) \in \mathbb{R}^{d_h}$$

Each encoder produces feature vectors in a shared dimensionality $d_h$ to facilitate subsequent integration.

#### 3.1.2 Multimodal Integration Module

The multimodal integration module combines features from available modalities into a unified representation. We implement a cross-modal attention mechanism inspired by transformer architectures, but with important medical-specific enhancements:

1. **Modality Presence Encoding**: We introduce a binary mask vector $m \in \{0,1\}^M$ indicating which of the $M$ modalities are available for a given sample.

2. **Medical Knowledge-Enhanced Attention**: We incorporate medical domain knowledge through a knowledge graph representation $K \in \mathbb{R}^{k \times d_h}$ containing $k$ medical concepts. The cross-attention between modality features and knowledge graph is computed as:
   $$A_{med} = \text{softmax}\left(\frac{H_i Q (K R)^T}{\sqrt{d_k}}\right)$$
   where $Q$ is a learnable query projection, $R$ is a relevance matrix weighted by disease prevalence statistics, and $d_k$ is the attention dimension.

3. **Unified Representation**: The modality features are combined using a weighted attention mechanism:
   $$Z = \sum_{i \in \{img, text, struct\}} m_i \cdot W_i \cdot (H_i + A_{med}K)$$
   where $W_i$ are learnable weight matrices that adapt based on the modality mask $m$.

This integration approach allows the model to focus on different modality combinations and weight them appropriately based on availability and relevance to the diagnostic task.

#### 3.1.3 Conditional Diffusion Model

We employ a conditional diffusion model operating in the unified latent space. Following the diffusion model framework, we define a forward process that gradually adds Gaussian noise to the latent representation, and a reverse process that learns to denoise the representation conditioned on the available modalities.

**Forward Process**: Starting with the integrated representation $Z_0 = Z$, we define a Markov chain that progressively adds noise over $T$ steps:
$$q(Z_t|Z_{t-1}) = \mathcal{N}(Z_t; \sqrt{1-\beta_t}Z_{t-1}, \beta_t\mathbf{I})$$
where $\{\beta_t\}_{t=1}^T$ is a noise schedule.

**Reverse Process**: We train a denoising network $\epsilon_\theta$ to predict the noise added at each step, conditioned on the available modality information:
$$p_\theta(Z_{t-1}|Z_t, m) = \mathcal{N}(Z_{t-1}; \mu_\theta(Z_t, t, m), \Sigma_\theta(Z_t, t, m))$$

The mean prediction is given by:
$$\mu_\theta(Z_t, t, m) = \frac{1}{\sqrt{\alpha_t}}\left(Z_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(Z_t, t, m)\right)$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.

The loss function for training the diffusion model is:
$$\mathcal{L}_{diff} = \mathbb{E}_{t,Z_0,\epsilon}\left[||\epsilon - \epsilon_\theta(Z_t, t, m)||^2\right]$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is the noise sampled to construct $Z_t$.

### 3.2 Adaptive Training Strategy

To enhance robustness to missing modalities, we implement a deliberate modality masking strategy during training.

#### 3.2.1 Masking Schedule

We define a curriculum-based masking schedule that progressively increases the probability of masking different modalities throughout training:
$$p_{mask}(i, e) = p_{min} + (p_{max} - p_{min}) \cdot \min\left(1, \frac{e}{e_{total} \cdot r}\right)$$

where $p_{mask}(i, e)$ is the probability of masking modality $i$ at epoch $e$, $p_{min}$ and $p_{max}$ are the minimum and maximum masking probabilities, $e_{total}$ is the total number of training epochs, and $r$ is a curriculum ratio controlling the masking rate increase.

#### 3.2.2 Rare Disease Focus

To address the challenge of rare diseases with limited training data, we implement a weighted sampling strategy that oversamples rare disease cases during training. For each disease category $c$ with frequency $f_c$ in the dataset, we assign a sampling weight:
$$w_c = \left(\frac{f_{max}}{f_c}\right)^{\gamma}$$

where $f_{max}$ is the frequency of the most common disease category, and $\gamma \in [0, 1]$ is a hyperparameter controlling the degree of oversampling.

#### 3.2.3 Consistency Regularization

To ensure consistent predictions regardless of which modalities are available, we introduce a consistency regularization term. For each training sample, we randomly generate two different modality masks $m_1$ and $m_2$, and minimize the difference between their latent representations:
$$\mathcal{L}_{cons} = ||Z(m_1) - Z(m_2)||^2$$

The final training objective combines the diffusion loss with the consistency regularization:
$$\mathcal{L}_{total} = \mathcal{L}_{diff} + \lambda_{cons} \cdot \mathcal{L}_{cons}$$

where $\lambda_{cons}$ is a hyperparameter controlling the weight of the consistency term.

### 3.3 Diagnostic Prediction and Explainability

#### 3.3.1 Diagnostic Classification

The final diagnostic prediction is performed using the denoised latent representation $\hat{Z}_0$. We implement a classification head consisting of a multi-layer perceptron that outputs probabilities across diagnostic categories:
$$p(y|Z) = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot \hat{Z}_0 + b_1) + b_2)$$

The model is trained with an auxiliary classification loss in addition to the diffusion objective:
$$\mathcal{L}_{class} = \text{CrossEntropy}(p(y|Z), y_{true})$$

#### 3.3.2 Modality-Specific Attribution Maps

To enhance explainability, we implement a gradient-based attribution technique that identifies the contribution of each modality to the final prediction. For each modality $i$ and diagnosis class $c$, we compute:
$$A_{i,c} = \left|\frac{\partial p(y=c|Z)}{\partial H_i}\right|$$

These attribution scores are normalized and visualized as heatmaps for imaging modalities or highlighted text for clinical notes, providing clinicians with interpretable insights into the model's decision-making process.

### 3.4 Experimental Design

#### 3.4.1 Datasets

We will evaluate RobustMDiff on multiple medical datasets containing diverse modalities:

1. **MIMIC-IV**: A comprehensive critical care database containing clinical notes, laboratory measurements, imaging studies, and structured EHR data.

2. **ADNI (Alzheimer's Disease Neuroimaging Initiative)**: A dataset focused on Alzheimer's disease progression, containing MRI, PET scans, genetic information, and clinical assessments.

3. **CheXpert**: A large chest X-ray dataset with associated radiologist reports and structured labels.

4. **Rare Disease Cohort**: A curated dataset focusing on 5-10 rare diseases with limited samples but multimodal data.

#### 3.4.2 Evaluation Metrics

We will assess model performance using the following metrics:

1. **Diagnostic Accuracy Metrics**:
   - Area Under the ROC Curve (AUC)
   - Precision, Recall, and F1-score
   - Sensitivity and Specificity

2. **Robustness Metrics**:
   - Performance degradation under different missing modality patterns
   - Relative performance on rare vs. common diseases

3. **Explainability Metrics**:
   - Human evaluation of attribution maps by clinical experts
   - Quantitative assessment of attribution quality using perturbation-based methods

#### 3.4.3 Baseline Methods

We will compare RobustMDiff against several baselines:

1. Single-modality models (CNN, BERT) trained on individual data types
2. Standard multimodal fusion approaches (early, late, and intermediate fusion)
3. State-of-the-art medical multimodal models (MedM2G, MedCoDi-M)
4. Traditional imputation methods for handling missing data

#### 3.4.4 Ablation Studies

We will conduct ablation studies to assess the contribution of each component:

1. Impact of the knowledge-enhanced attention mechanism
2. Effect of the adaptive training strategy with modality masking
3. Contribution of the consistency regularization term
4. Performance with different diffusion model configurations

## Expected Outcomes & Impact

The successful development and validation of RobustMDiff is expected to yield several significant outcomes with broad impact on healthcare AI applications:

### 4.1 Improved Diagnostic Accuracy for Rare Diseases

By addressing the data scarcity challenge through our adaptive training strategy and hierarchical multimodal architecture, we expect RobustMDiff to demonstrate superior performance on rare disease diagnosis compared to existing approaches. This improvement could translate to earlier detection and more appropriate treatment strategies for patients with rare conditions, who are often underserved by current diagnostic systems. We anticipate demonstrating at least a 15-20% improvement in diagnostic accuracy for rare diseases compared to current state-of-the-art methods.

### 4.2 Robustness to Missing Modalities

A key expected outcome is the model's maintained performance despite missing data modalities—a common challenge in real-world clinical settings. Through our deliberate modality masking and consistency regularization strategies, RobustMDiff should exhibit graceful degradation rather than catastrophic failure when certain data types are unavailable. We expect to show that our model maintains at least 85-90% of its optimal performance even when up to 50% of input modalities are missing, significantly outperforming existing approaches in incomplete data scenarios.

### 4.3 Transparent and Interpretable Diagnostics

The modality-specific attribution maps will provide interpretable explanations for diagnostic predictions, addressing a critical requirement for clinical adoption. These explanations will help clinicians understand which aspects of the patient data most influenced the model's decision, potentially uncovering subtle patterns that might be overlooked in traditional diagnostic workflows. We anticipate that clinical experts will rate the explanations provided by RobustMDiff as significantly more relevant and useful than those from comparison models in human evaluation studies.

### 4.4 Practical Impact on Clinical Workflows

Beyond technical advancements, RobustMDiff has the potential to meaningfully impact clinical practice in several ways:

1. **Decision Support**: By providing robust diagnostic suggestions with associated confidence levels and explanations, the system can serve as an effective decision support tool for clinicians, particularly for complex or rare cases.

2. **Resource Optimization**: The model's ability to make quality predictions even with incomplete data could reduce unnecessary diagnostic tests, decreasing healthcare costs and patient burden.

3. **Equity in Care**: Improved performance on rare diseases and underrepresented patient populations could help address disparities in healthcare quality and access.

4. **Knowledge Discovery**: The patterns identified by the model and highlighted through attribution maps may lead to new insights about disease manifestations and relationships between different clinical variables.

### 4.5 Methodological Contributions to Medical AI

From a methodological perspective, this research will contribute several innovations to the field of medical AI:

1. A novel approach to incorporating medical domain knowledge into diffusion models through the knowledge-enhanced attention mechanism.

2. A curriculum-based modality masking strategy that can be adapted to other multimodal learning problems beyond healthcare.

3. A unified framework for handling heterogeneous medical data types within a single model architecture.

4. New techniques for evaluating model robustness under different patterns of missing data.

In summary, RobustMDiff represents a significant step toward addressing the key challenges that have limited the practical application of AI systems in clinical diagnostics. By combining the generative power of diffusion models with a robust, explainable multimodal architecture, this research aims to bridge the gap between advanced AI methodologies and practical clinical implementation, ultimately contributing to improved healthcare outcomes for diverse patient populations.