# Adversarial Counterfactual Augmentation for Robust Learning Against Spurious Correlations

## 1. Introduction

Machine learning models are increasingly deployed in high-stakes domains such as healthcare, finance, and criminal justice, where reliability and generalizability are paramount. Despite impressive performance on benchmarks, these models often fail catastrophically when deployed in real-world settings due to their tendency to exploit spurious correlations—statistical patterns in the training data that are not causally related to the target task but happen to be predictive within the training distribution. When these spurious correlations shift or disappear in deployment settings, model performance degrades dramatically.

Examples of this phenomenon abound across domains. In medical imaging, models trained to detect lung disease often rely on scanning artifacts or hospital-specific markers rather than physiological indicators of disease. Natural language processing models may base entailment decisions on lexical overlap rather than semantic relationships. In genomics, polygenic risk scores for diseases like diabetes disproportionately leverage genetic variants common in European populations, limiting their utility for other demographic groups.

Current approaches to address spurious correlations generally fall into three categories: (1) methods requiring explicit group annotations, which are often unavailable or expensive to obtain; (2) invariant learning approaches that assume access to multiple environments with varying spurious correlations; and (3) data augmentation techniques that often lack principled guidance on which features to manipulate. While these approaches have shown promise, they typically struggle with complex, unknown spurious features or require annotations that are prohibitively expensive in real-world settings.

This research proposes Adversarial Counterfactual Augmentation (ACA), a novel framework that systematically identifies potentially spurious features and generates counterfactual examples to enforce invariance without requiring group annotations. Our approach combines insights from feature attribution methods, generative modeling, and adversarial training to create a practical solution for improving model robustness against spurious correlations.

The key research objectives of this study are:
1. To develop a method for automatically identifying potentially spurious features using gradient-based attribution techniques and influence functions
2. To design and implement a conditional generative framework for creating counterfactual examples that modify only the identified spurious features while preserving causal features
3. To formulate a training procedure that leverages these counterfactuals to encourage invariance to spurious features
4. To evaluate the effectiveness of the proposed approach across diverse domains and spurious correlation types

The significance of this research extends beyond theoretical contributions to machine learning. By developing methods that are robust to spurious correlations without requiring expensive annotations, we can improve the reliability and fairness of deployed machine learning systems across critical applications. Furthermore, our approach offers interpretability benefits by explicitly identifying and visualizing potentially spurious features, allowing domain experts to validate model reasoning processes.

## 2. Methodology

Our proposed Adversarial Counterfactual Augmentation (ACA) framework consists of three main components: (1) spurious feature identification, (2) counterfactual generation, and (3) robust model training. The following sections detail each component and their integration into a comprehensive approach.

### 2.1 Spurious Feature Identification

We begin by training an initial model $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ on the original training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. After training converges, we employ a combination of influence functions and gradient-based attribution methods to identify potentially spurious features.

#### 2.1.1 Influence-Based Identification

Influence functions measure how model predictions would change if a training example were downweighted. We leverage this to identify examples that have disproportionate influence on the model's decision boundaries:

$$\mathcal{I}(z, z_{\text{test}}) = -\nabla_\theta L(z_{\text{test}}, \theta)^T H_\theta^{-1} \nabla_\theta L(z, \theta)$$

where $z = (x, y)$ represents a training example, $L$ is the loss function, $\theta$ represents the model parameters, and $H_\theta$ is the Hessian of the loss. Intuitively, training examples with high influence often contain features that the model has latched onto, some of which may be spurious.

To make this computation tractable, we use the approximation method from Koh and Liang (2017):

$$H_\theta^{-1} \nabla_\theta L(z, \theta) \approx \sum_{j=0}^{J-1} (I - \lambda \nabla_\theta^2 L(\mathcal{D}, \theta))^j \lambda \nabla_\theta L(z, \theta)$$

where $\lambda$ is a damping parameter and $J$ is the number of iterations.

#### 2.1.2 Gradient-Based Feature Attribution

For each high-influence example identified, we apply integrated gradients (Sundararajan et al., 2017) to quantify feature importance:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial f_\theta(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

where $x'$ is a baseline input (typically zero), and the integral captures the gradients along a straight-line path from $x'$ to $x$. We approximate this integral using the Riemann sum with $m$ steps:

$$\text{IG}_i(x) \approx (x_i - x'_i) \times \frac{1}{m}\sum_{k=1}^{m}\frac{\partial f_\theta(x' + \frac{k}{m}(x - x'))}{\partial x_i}$$

For image data, we aggregate these attributions into spatial regions using superpixel segmentation. For text data, we aggregate at the token or phrase level.

#### 2.1.3 Clustering and Ranking Potential Spurious Features

We cluster the identified important features across the training set and rank them based on:

1. Consistency of the feature's presence in examples of the same class
2. Discriminative power across classes
3. Domain-inconsistency score: how much the feature's importance varies across different subsets of the data

The top-k ranked features are selected as candidates for being spurious correlations. We denote this set as $\mathcal{S} = \{s_1, s_2, ..., s_k\}$.

### 2.2 Counterfactual Generation

Once potential spurious features are identified, we train a conditional generative model to create counterfactual examples that modify only these features while preserving the causal features.

#### 2.2.1 Conditional Generation Framework

We implement a conditional diffusion model $G$ that takes as input the original example $x$, the target label $y$, and a mask $m$ indicating which features to modify (the identified spurious features). The generative process is formalized as:

$$\hat{x} = G(x, y, m)$$

For diffusion models, we follow the formulation:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

$$p_\theta(x_{t-1}|x_t, y, m) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, y, m), \Sigma_\theta(x_t, t))$$

where $\beta_t$ is the noise schedule, and $\mu_\theta$ and $\Sigma_\theta$ are learned functions.

#### 2.2.2 Training Objective

The generative model is trained with the following objectives:

1. **Reconstruction Loss**: Ensures the generated counterfactuals maintain overall structure and semantics:
   $$\mathcal{L}_{\text{recon}} = \mathbb{E}_{x,y,m,\epsilon,t}[||\epsilon - \epsilon_\theta(x_t, t, y, m)||^2]$$

2. **Feature Modification Loss**: Ensures that spurious features are modified while causal features are preserved:
   $$\mathcal{L}_{\text{mod}} = ||m \odot (x - \hat{x})||_2^2 + ||(1-m) \odot (x - \hat{x})||_2^2$$

3. **Semantic Consistency Loss**: Ensures that counterfactuals maintain the semantic meaning required for the label:
   $$\mathcal{L}_{\text{sem}} = D_{\text{KL}}(f_{\text{enc}}(x) || f_{\text{enc}}(\hat{x}))$$
   where $f_{\text{enc}}$ is a pretrained encoder that extracts semantic features.

The total loss for the generative model is:
$$\mathcal{L}_G = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{mod}} + \lambda_2 \mathcal{L}_{\text{sem}}$$

where $\lambda_1$ and $\lambda_2$ are hyperparameters balancing the different objectives.

### 2.3 Robust Model Training

With the counterfactual generation capability in place, we train a robust classifier that is invariant to the identified spurious features.

#### 2.3.1 Augmented Dataset Creation

We create an augmented dataset $\mathcal{D}_{\text{aug}}$ by generating counterfactuals for the original training examples:

$$\mathcal{D}_{\text{aug}} = \mathcal{D} \cup \{(\hat{x}_i, y_i) | \hat{x}_i = G(x_i, y_i, m_i), (x_i, y_i) \in \mathcal{D}\}$$

For each original example, we generate multiple counterfactuals by varying the mask $m$ to cover different combinations of the identified spurious features.

#### 2.3.2 Training with Consistency Regularization

We train the final robust model $f_{\phi}$ using a combination of standard cross-entropy loss and a consistency regularization term:

$$\mathcal{L}_{\text{CE}} = \mathbb{E}_{(x,y) \in \mathcal{D}_{\text{aug}}}[-\log f_{\phi}(y|x)]$$

$$\mathcal{L}_{\text{cons}} = \mathbb{E}_{(x,y) \in \mathcal{D}, \hat{x} = G(x,y,m)}[D_{\text{KL}}(f_{\phi}(\cdot|x) || f_{\phi}(\cdot|\hat{x}))]$$

The total loss is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \gamma \mathcal{L}_{\text{cons}}$$

where $\gamma$ is a hyperparameter controlling the strength of the consistency regularization.

#### 2.3.3 Adversarial Training Component

To further enhance robustness, we incorporate an adversarial component. We train a discriminator $D$ that attempts to predict whether an example is from the original distribution or a counterfactual:

$$\mathcal{L}_D = \mathbb{E}_{x \in \mathcal{D}}[\log D(x)] + \mathbb{E}_{\hat{x} \in \mathcal{D}_{\text{aug}} \setminus \mathcal{D}}[\log(1 - D(\hat{x}))]$$

The main model is then trained with an additional adversarial term:

$$\mathcal{L}_{\text{adv}} = -\mathbb{E}_{\hat{x} \in \mathcal{D}_{\text{aug}} \setminus \mathcal{D}}[\log D(\hat{x})]$$

The final loss becomes:
$$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{CE}} + \gamma \mathcal{L}_{\text{cons}} + \delta \mathcal{L}_{\text{adv}}$$

where $\delta$ controls the strength of the adversarial component.

### 2.4 Experimental Design

We evaluate our approach on three domains with well-documented spurious correlation issues:

#### 2.4.1 Image Classification

**Datasets:**
- Waterbirds (birds with spurious water/land background correlations)
- CelebA (gender prediction with spurious hair color correlations)
- MNIST-CIFAR (digits superimposed on CIFAR backgrounds)

**Implementation Details:**
- Architecture: ResNet-50 for initial model and robust classifier
- Diffusion model with U-Net architecture for counterfactual generation
- Superpixel segmentation (SLIC algorithm) for image region identification

**Evaluation Metrics:**
- Overall accuracy
- Worst-group accuracy
- Average accuracy gap between groups
- Area Under RCP (Reliability-Coverage) curve

#### 2.4.2 Natural Language Processing

**Datasets:**
- MNLI (natural language inference with spurious lexical overlap)
- FEVER (fact verification with spurious lexical patterns)
- CivilComments (toxicity detection with demographic correlations)

**Implementation Details:**
- Initial model: BERT-base or RoBERTa-base
- Text generation: Conditional GPT-2 fine-tuned on the task
- Attribute masks based on token-level attributions

**Evaluation Metrics:**
- Overall accuracy/F1
- Group-disaggregated metrics
- Performance under synthetic distribution shifts

#### 2.4.3 Medical Diagnosis

**Datasets:**
- CheXpert (chest X-ray classification with hospital-specific artifacts)
- Dermatology dataset (skin condition diagnosis with spurious color/lighting)

**Implementation Details:**
- DenseNet-121 architecture for initial model
- CycleGAN-based approach for modifying medical images with careful preservation of diagnostic features
- Validation with medical experts to ensure counterfactuals maintain clinical validity

**Evaluation Metrics:**
- AUC-ROC overall and per subgroup
- Specificity and sensitivity
- Performance across different hospitals/demographic groups

#### 2.4.4 Baselines

We compare our method against:
1. Empirical Risk Minimization (ERM)
2. Group DRO (requires group labels)
3. JTT (Just Train Twice)
4. LISA (Learning from Identified Spurious Attributes)
5. Standard data augmentation approaches
6. IRM (Invariant Risk Minimization)

#### 2.4.5 Ablation Studies

We conduct ablation studies to understand the contribution of each component:
1. Effect of spurious feature identification method
2. Impact of counterfactual generation quality
3. Contribution of consistency vs. adversarial losses
4. Effect of the number of counterfactuals per original example
5. Sensitivity to hyperparameters $\lambda_1$, $\lambda_2$, $\gamma$, and $\delta$

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

This research is expected to yield the following outcomes:

1. **Improved Robustness to Spurious Correlations**: The ACA framework should significantly improve worst-group performance across all tested domains without requiring group annotations. We expect to observe at least a 10-15% improvement in worst-group accuracy compared to standard ERM, approaching the performance of methods that require explicit group labels.

2. **Effective Spurious Feature Identification**: Our approach should automatically identify the key spurious features across datasets, validated through quantitative metrics and qualitative visualization. We anticipate identifying at least 80% of the known spurious correlations in benchmark datasets.

3. **Realistic Counterfactual Generation**: The conditional generative model should produce realistic counterfactuals that modify only the identified spurious features while preserving the semantic content relevant to the label. These counterfactuals should be validated through both automated metrics and human evaluation.

4. **Generalizability Across Domains**: The ACA framework should demonstrate effectiveness across image classification, natural language processing, and medical diagnosis tasks, showing its versatility in handling different types of data and spurious correlations.

5. **Interpretability Gains**: By explicitly identifying and visualizing potentially spurious features, our approach will provide interpretable insights into model decision-making, allowing domain experts to validate and refine the process.

### 3.2 Broader Impact

The proposed research has several potential broader impacts:

1. **Clinical Applications**: More robust medical diagnostic systems that focus on physiological indicators rather than hospital-specific artifacts could improve healthcare outcomes and reduce disparities in medical AI.

2. **Fairness and Equity**: By reducing reliance on spurious correlations that may disproportionately affect certain demographic groups, our method can contribute to building more equitable AI systems.

3. **Deployment Reliability**: Models trained with our approach should maintain consistent performance when deployed in new environments where spurious correlations may differ from training data, increasing the practical utility of ML systems.

4. **Reduced Annotation Costs**: By eliminating the need for explicit group annotations, our approach can make robust training more accessible and cost-effective for real-world applications.

5. **Methodological Advances**: The integration of feature attribution, counterfactual generation, and adversarial training provides a novel framework that can inspire further research at the intersection of these areas.

6. **Domain Adaptation**: The insights from identifying and mitigating spurious correlations can inform better domain adaptation strategies, particularly for applications where collecting target domain data is expensive or impractical.

Our method's ability to identify and visualize potentially spurious features also provides transparency that can build trust with stakeholders and end-users, a critical factor for the adoption of AI systems in high-stakes domains.

By addressing spurious correlations without requiring expensive annotations, this research contributes to the development of more reliable, fair, and generalizable machine learning systems that can maintain their performance when deployed in real-world settings.