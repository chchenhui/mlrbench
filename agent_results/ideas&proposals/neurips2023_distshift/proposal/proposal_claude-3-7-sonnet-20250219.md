# Preserving Distributional Robustness in Foundation Model Fine-tuning via Dual-Teacher Knowledge Distillation

## Introduction

Foundation models have revolutionized artificial intelligence by demonstrating remarkable capabilities across diverse tasks and domains. These large pre-trained models, such as CLIP, GPT, and BERT, have shown impressive robustness to distribution shifts - a critical advantage in real-world applications where deployment conditions often differ significantly from training environments. However, a concerning phenomenon has emerged: when foundation models are fine-tuned for specialized downstream tasks, their robustness to distribution shifts often diminishes substantially (Kumar et al., 2022). This degradation presents a significant challenge for high-stakes applications in healthcare, legal systems, and other domains where distribution shifts are inevitable and consequential failures unacceptable.

The underlying cause of this robustness degradation appears to be the distortion of pretrained features during fine-tuning. As models adapt to specialized tasks, they often overfit to the in-distribution characteristics of the fine-tuning dataset, losing the broader knowledge and generalization capabilities acquired during pre-training. Current fine-tuning approaches primarily optimize for in-distribution performance, inadvertently sacrificing the distributional robustness that makes foundation models valuable. This trade-off between specialization and robustness represents a fundamental limitation in the deployment of foundation models to critical real-world applications.

Recent work has begun exploring this problem. Wortsman et al. (2021) introduced WiSE-FT, which ensembles the weights of zero-shot and fine-tuned models to improve robustness. Zhou et al. (2023) proposed Discrete Adversarial Distillation to enhance vision model robustness through knowledge distillation. Yang et al. (2024) developed Self-Distillation Fine-Tuning to bridge distribution gaps in language model fine-tuning. While these approaches show promise, they either require additional inference-time computation, focus exclusively on specific modalities, or fail to fully preserve the robustness advantages of the original foundation model.

This research aims to address this critical gap by developing a comprehensive knowledge distillation framework that preserves distributional robustness during foundation model fine-tuning while enabling effective task adaptation. Our approach builds on recent advances in parameter-efficient fine-tuning (PEFT) and knowledge distillation but introduces novel components specifically designed to preserve robustness signals from the original model. We propose a dual-teacher architecture that simultaneously guides the model toward task specialization and distribution robustness, complemented by a regularization scheme that explicitly preserves critical activation patterns from the pre-trained model.

The significance of this research lies in its potential to enable the robust deployment of foundation models in specialized domains where distribution shifts are prevalent and consequential. By preserving robustness during fine-tuning, our approach could substantially reduce performance degradation in real-world settings, making foundation models more reliable for applications in medical imaging, legal text analysis, environmental monitoring, and other high-stakes domains. Moreover, our method aims to be modality-agnostic, applicable to both vision and language foundation models, and computationally efficient during both training and inference.

## Methodology

### Overview

Our proposed framework, Robust Distillation Fine-Tuning (RDFT), employs a dual-teacher knowledge distillation approach combined with parameter-efficient fine-tuning and robust regularization techniques. The core idea involves using the original foundation model as a "robustness teacher" alongside a task-specific teacher to guide the fine-tuning process. The student model learns to balance task performance with distributional robustness through a multi-objective optimization process. Figure 1 illustrates the overall architecture of our approach.

### Model Architecture

We adopt a parameter-efficient fine-tuning approach using Low-Rank Adaptation (LoRA) (Hu et al., 2021) as our base architecture. For a foundation model with parameters $\theta_0$, we introduce trainable low-rank adaptation matrices for each weight matrix $W \in \mathbb{R}^{d \times k}$ in the model:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are low-rank matrices with rank $r \ll \min(d, k)$. Only the parameters in $A$ and $B$ are updated during fine-tuning, keeping the original weights $W$ frozen. This approach significantly reduces the number of trainable parameters while maintaining performance comparable to full fine-tuning.

### Dual-Teacher Knowledge Distillation

Our key innovation is the dual-teacher distillation framework. Let $M_0$ be the original foundation model (robustness teacher), $M_T$ be a task-specialized teacher model (e.g., a fully fine-tuned model on the target task), and $M_S$ be the student model undergoing parameter-efficient fine-tuning. The dual distillation occurs as follows:

1. **Task Knowledge Distillation**: The student learns task-specific knowledge from the task teacher.
2. **Robustness Knowledge Distillation**: The student preserves distributional robustness by matching the original model's behavior on diverse inputs.

For each input sample $x$ from the training set, we compute:

$$L_{task}(x) = \alpha \cdot L_{CE}(M_S(x), y) + (1-\alpha) \cdot L_{KD}(M_S(x), M_T(x))$$

where $L_{CE}$ is the cross-entropy loss with the ground truth label $y$, $L_{KD}$ is the knowledge distillation loss (typically KL divergence between output distributions), and $\alpha$ is a weighting hyperparameter.

For robustness distillation, we generate perturbed samples $\tilde{x}$ using a distribution shift simulator (detailed in the next section) and compute:

$$L_{robust}(\tilde{x}) = L_{KD}(M_S(\tilde{x}), M_0(\tilde{x}))$$

Additionally, we introduce a feature-level distillation component that preserves intermediate representations:

$$L_{feature}(x) = \sum_{l \in \mathcal{L}} \beta_l \cdot \|F_l^S(x) - F_l^0(x)\|_2^2$$

where $F_l^S$ and $F_l^0$ are the feature maps at layer $l$ for the student and robustness teacher, respectively, $\mathcal{L}$ is a set of selected layers, and $\beta_l$ are layer-specific weights.

The total loss function combines these components:

$$L_{total} = L_{task}(x) + \lambda_1 \cdot L_{robust}(\tilde{x}) + \lambda_2 \cdot L_{feature}(x)$$

where $\lambda_1$ and $\lambda_2$ are hyperparameters that control the importance of robustness and feature preservation.

### Distribution Shift Simulation

A critical component of our approach is the generation of realistic distribution shifts during training. We employ multiple strategies to create diverse perturbations:

1. **Domain-Specific Transformations**: We apply transformations relevant to the specific domain (e.g., contrast/brightness changes for medical images, style shifts for natural images, vocabulary shifts for text).

2. **Adversarial Perturbations**: We generate adversarial examples that maximize the divergence between the original foundation model and the current student model:

$$\tilde{x} = x + \epsilon \cdot \text{sign}(\nabla_x D_{KL}(M_0(x) || M_S(x)))$$

where $\epsilon$ is a small perturbation magnitude and $D_{KL}$ is the KL divergence.

3. **Synthetic Distribution Shifts**: For language models, we use prompt-based techniques to generate text with specific stylistic or demographic attributes. For vision models, we employ style transfer models to create domain-shifted variants of training images.

4. **Retrieval-Based Shift Examples**: We retrieve examples from publicly available datasets representing different domains, demographics, or time periods relevant to the task.

These diverse perturbation strategies ensure that the student model encounters a wide range of potential distribution shifts during training, allowing it to maintain robustness across various deployment scenarios.

### Robust Activation Regularization

To further preserve the generalization capabilities of the foundation model, we introduce Robust Activation Regularization (RAR). This technique explicitly encourages the student model to maintain similar activation patterns to the original model on a diverse set of inputs.

For each layer $l$ in a set of critical layers $\mathcal{C}$, we calculate the activation correlation matrix between the original and student model:

$$C_l = \text{corr}(A_l^0, A_l^S)$$

where $A_l^0$ and $A_l^S$ are the activations of the original and student model at layer $l$, and $\text{corr}$ computes the correlation matrix.

We then add a regularization term to the loss function:

$$L_{RAR} = \sum_{l \in \mathcal{C}} \gamma_l \cdot (1 - \text{tr}(C_l) / d_l)$$

where $d_l$ is the dimensionality of layer $l$, $\text{tr}$ is the trace operator, and $\gamma_l$ are layer-specific weights.

### Training Procedure

Our training procedure consists of the following steps:

1. **Initialization**: Initialize the student model $M_S$ with the weights of the original foundation model $M_0$ and add LoRA modules to key layers.

2. **Teacher Preparation**: If a task-specific teacher is not available, train one using full fine-tuning on the target task.

3. **Robust Fine-tuning**: 
   - For each mini-batch of training data $(X, Y)$:
     - Generate perturbed samples $\tilde{X}$ using the distribution shift simulator
     - Forward pass $X$ and $\tilde{X}$ through $M_0$, $M_T$, and $M_S$
     - Compute task loss $L_{task}$, robustness loss $L_{robust}$, feature distillation loss $L_{feature}$, and activation regularization loss $L_{RAR}$
     - Compute total loss $L_{total}$ and update LoRA parameters through backpropagation
   - Periodically validate on both in-distribution and out-of-distribution validation sets
   - Adjust $\lambda_1$ and $\lambda_2$ based on validation performance to balance task accuracy and robustness

4. **Model Merging (Optional)**: After training, optionally merge the original model weights with the adapted weights using an interpolation factor $\mu$:

$$\theta_{final} = (1 - \mu) \cdot \theta_0 + \mu \cdot \theta_S$$

where $\theta_0$ are the original weights and $\theta_S$ are the student model weights after applying LoRA updates.

### Evaluation Protocol

We design a comprehensive evaluation protocol to assess both task performance and robustness across diverse distribution shifts:

1. **In-Distribution Evaluation**: Standard metrics (accuracy, F1-score, etc.) on the target dataset's test split.

2. **Natural Distribution Shift Evaluation**: Performance on established distribution shift benchmarks such as WILDS (for diverse applications), ImageNet-C/R/A (for vision), and RobustBench datasets.

3. **Synthetic Distribution Shift Evaluation**: Performance on systematically generated shifts specific to the domain, including:
   - Demographic shifts (for language)
   - Style and appearance shifts (for vision)
   - Temporal shifts (where historical data is available)
   - Geographic shifts (for geospatial applications)

4. **Robustness Gap Metric**: We quantify the robustness gap as the percentage drop in performance between in-distribution and out-of-distribution scenarios:

$$\text{RG} = 1 - \frac{\text{Perf}_{\text{OOD}}}{\text{Perf}_{\text{ID}}}$$

5. **Efficiency Metrics**: We measure computational efficiency in terms of:
   - Number of trainable parameters
   - Training time (GPU-hours)
   - Inference time overhead compared to standard fine-tuning

6. **Ablation Studies**: We conduct ablation studies to assess the individual contribution of each component (dual distillation, distribution shift simulation, activation regularization, etc.).

Our evaluation strategy is designed to provide a comprehensive assessment of how well our approach preserves distributional robustness while achieving competitive task performance, compared to alternative fine-tuning methods.

## Expected Outcomes & Impact

This research is expected to yield several significant outcomes with broad impact across machine learning research and applications:

### Technical Advancements

1. **Improved Robustness-Accuracy Trade-off**: Our approach aims to significantly reduce the robustness gap observed during foundation model fine-tuning while maintaining competitive in-distribution performance. We expect at least a 30-50% reduction in the robustness gap compared to standard fine-tuning methods, as measured on established distribution shift benchmarks.

2. **Modality-Agnostic Framework**: We anticipate demonstrating the effectiveness of our approach across both vision and language foundation models, showing its broad applicability across modalities. This generality would represent a significant advance over existing methods that are often modality-specific.

3. **Computational Efficiency**: By leveraging parameter-efficient fine-tuning techniques like LoRA, our method should require significantly fewer computational resources than alternative robustness methods. We expect to achieve these robustness improvements with minimal additional training cost compared to standard fine-tuning.

4. **Theoretical Insights**: This research will contribute to a deeper understanding of why robustness degrades during fine-tuning and how knowledge distillation can preserve critical generalization capabilities. These insights may lead to new theoretical frameworks for analyzing the robustness-specialization trade-off in foundation models.

### Practical Applications

1. **Healthcare**: Enabling more robust deployment of foundation models for medical imaging and clinical text analysis, reducing performance disparities across patient demographics, hospital systems, and imaging devices. This could accelerate the safe adoption of AI in healthcare settings.

2. **Legal AI**: Improving the reliability of language models for legal document analysis across different jurisdictions, time periods, and legal domains, supporting fairer and more consistent AI-assisted legal services.

3. **Environmental Monitoring**: Enhancing the performance of vision models for conservation and climate applications across diverse geographic regions and environmental conditions, supporting more effective environmental protection efforts.

4. **Educational Technology**: Developing more robust language models that work effectively across different educational contexts, student demographics, and subject domains, enabling more equitable educational AI tools.

### Broader Impact

1. **Democratizing Robust AI**: By reducing the computational requirements for achieving robust fine-tuning, our approach could democratize access to robust AI systems, enabling smaller organizations with limited computational resources to deploy foundation models reliably.

2. **Addressing AI Fairness**: Distribution shifts often disproportionately affect underrepresented groups. By preserving robustness during fine-tuning, our method could help reduce performance disparities across demographic groups, contributing to fairer AI systems.

3. **Research Community Resources**: We will release our implementation, pretrained models, and evaluation framework as open-source resources, fostering further research on robust fine-tuning methods for foundation models.

4. **Industry Adoption**: The parameter-efficient nature of our approach makes it practical for industry adoption, potentially influencing how foundation models are deployed in production systems across various sectors.

In conclusion, this research addresses a critical challenge in the deployment of foundation models - the preservation of distributional robustness during fine-tuning. By developing a dual-teacher knowledge distillation framework that explicitly maintains the robustness capabilities of the original model while enabling task specialization, we aim to significantly advance the reliable application of foundation models in real-world settings where distribution shifts are inevitable and consequential.