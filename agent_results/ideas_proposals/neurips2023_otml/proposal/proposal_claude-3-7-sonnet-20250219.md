# Adaptive Unbalanced Optimal Transport for Robust Domain Adaptation under Label Shift

## 1. Introduction

Machine learning models typically assume that training and test data follow similar distributions. However, in real-world applications, this assumption is frequently violated due to domain shifts, which can significantly degrade model performance. Domain adaptation (DA) techniques aim to mitigate this performance drop by adapting models trained on a source domain to perform well on a target domain. Within this context, Optimal Transport (OT) has emerged as a powerful mathematical framework for aligning distributions across domains.

Traditional OT approaches assume that the source and target distributions have equal mass, effectively enforcing balanced class distributions across domains. This assumption, however, is unrealistic in many practical scenarios where the proportion of examples in each class can vary significantly between domains, a phenomenon known as label shift. For example, a disease detection system trained on a population with a 5% disease prevalence may be deployed in a region where the prevalence is 20%, representing a substantial label shift.

Unbalanced Optimal Transport (UOT) partially addresses this limitation by relaxing the mass conservation constraint, allowing for variation in marginal distributions. However, current UOT methods typically rely on fixed relaxation parameters that must be manually specified, requiring domain expertise and potentially extensive hyperparameter tuning. Furthermore, these predetermined parameters cannot adapt to the specific characteristics of the data or varying degrees of label shift encountered during training.

This research proposes an Adaptive Unbalanced Optimal Transport (A-UOT) framework that automatically learns the optimal degree of mass variation between source and target distributions directly from the data. By integrating this approach within a deep domain adaptation model, we aim to develop a more robust technique that can effectively handle unknown label shifts without requiring manual parameter tuning. The framework dynamically adjusts the relaxation parameters guided by target domain statistics and pseudo-label estimates, enabling implicit estimation and compensation for label shifts during the adaptation process.

Our research objectives are threefold:
1. Develop a novel A-UOT framework that learns optimal mass variation parameters during training
2. Design an end-to-end deep domain adaptation architecture that integrates A-UOT with feature extraction and classification
3. Evaluate the effectiveness of A-UOT on benchmark datasets with varying degrees of label shift

The significance of this research lies in addressing a critical gap in current domain adaptation methods—the ability to automatically adapt to unknown label shifts between domains. By developing an approach that can handle this common real-world challenge without requiring manual parameter tuning, we aim to enhance the robustness and applicability of domain adaptation techniques across various fields, including computer vision, natural language processing, and healthcare analytics.

## 2. Methodology

Our proposed Adaptive Unbalanced Optimal Transport (A-UOT) framework integrates learnable relaxation parameters within an end-to-end deep domain adaptation architecture. The methodology consists of three main components: (1) feature extraction, (2) adaptive unbalanced optimal transport alignment, and (3) classification with label shift compensation.

### 2.1 Problem Formulation

Let $X_s = \{x_s^i\}_{i=1}^{n_s}$ and $X_t = \{x_t^j\}_{j=1}^{n_t}$ denote the source and target domain data, respectively. For the source domain, we have labels $Y_s = \{y_s^i\}_{i=1}^{n_s}$ where $y_s^i \in \{1, 2, ..., K\}$ with $K$ being the number of classes. For the target domain, labels $Y_t$ are unavailable during training. We denote the source and target distributions as $P_s$ and $P_t$, respectively.

The goal is to learn a model that performs well on the target domain despite domain shift and label shift. Label shift occurs when the class distributions differ between domains, i.e., $P_s(Y) \neq P_t(Y)$, while maintaining the class-conditional feature distributions: $P_s(X|Y=k) = P_t(X|Y=k)$ for all classes $k$.

### 2.2 Feature Extraction

We employ a deep neural network $F_{\theta}$ with parameters $\theta$ to extract domain-invariant features from both source and target domains:

$$Z_s = \{z_s^i\}_{i=1}^{n_s} = \{F_{\theta}(x_s^i)\}_{i=1}^{n_s}$$
$$Z_t = \{z_t^j\}_{j=1}^{n_t} = \{F_{\theta}(x_t^j)\}_{j=1}^{n_t}$$

The feature extractor is trained to minimize the source classification loss while aligning the feature distributions across domains using our adaptive unbalanced optimal transport mechanism.

### 2.3 Adaptive Unbalanced Optimal Transport

Standard OT minimizes the cost of transporting mass from a source distribution to a target distribution under the constraint that all mass must be transported. Unbalanced OT relaxes this constraint by allowing some mass to be created or destroyed, but typically uses fixed relaxation parameters.

In our A-UOT framework, we formulate the transport problem with learnable relaxation parameters. Given empirical distributions in the feature space:

$$\mu_s = \sum_{i=1}^{n_s} a_i \delta_{z_s^i}, \quad \mu_t = \sum_{j=1}^{n_t} b_j \delta_{z_t^j}$$

where $a_i$ and $b_j$ represent the weights of each sample (initially uniform), and $\delta_{z}$ denotes the Dirac delta function at point $z$.

The A-UOT problem is formulated as:

$$\text{A-UOT}_{\lambda_s, \lambda_t}(\mu_s, \mu_t) = \min_{\pi \in \mathbb{R}_{+}^{n_s \times n_t}} \sum_{i,j} c_{ij} \pi_{ij} + \lambda_s D_{\text{KL}}(\pi \mathbf{1}_{n_t} || \mathbf{a}) + \lambda_t D_{\text{KL}}(\pi^T \mathbf{1}_{n_s} || \mathbf{b})$$

where:
- $c_{ij} = ||z_s^i - z_t^j||^2$ is the squared Euclidean cost between features
- $\pi_{ij}$ represents the amount of mass transported from $z_s^i$ to $z_t^j$
- $D_{\text{KL}}$ is the Kullback-Leibler divergence
- $\lambda_s$ and $\lambda_t$ are learnable parameters controlling the strictness of marginal constraints

The key innovation is that $\lambda_s$ and $\lambda_t$ are no longer fixed hyperparameters but are learned during training. We parameterize them as:

$$\lambda_s = \sigma(w_s), \quad \lambda_t = \sigma(w_t)$$

where $\sigma$ is a sigmoid function scaled to an appropriate range (e.g., $[0.1, 100]$), and $w_s$, $w_t$ are learnable parameters. This parameterization ensures that relaxation parameters remain positive and within a reasonable range.

To compute the A-UOT efficiently, we employ the Sinkhorn-Knopp algorithm with entropic regularization:

$$\text{A-UOT}_{\lambda_s, \lambda_t, \varepsilon}(\mu_s, \mu_t) = \min_{\pi \in \mathbb{R}_{+}^{n_s \times n_t}} \sum_{i,j} c_{ij} \pi_{ij} + \lambda_s D_{\text{KL}}(\pi \mathbf{1}_{n_t} || \mathbf{a}) + \lambda_t D_{\text{KL}}(\pi^T \mathbf{1}_{n_s} || \mathbf{b}) + \varepsilon H(\pi)$$

where $H(\pi) = -\sum_{i,j} \pi_{ij} \log \pi_{ij}$ is the entropy of the transport plan and $\varepsilon > 0$ is a small regularization parameter.

### 2.4 Label Shift Estimation and Compensation

To guide the learning of relaxation parameters and improve adaptation under label shift, we incorporate label shift estimation into our framework. We utilize the optimal transport plan $\pi$ from A-UOT to estimate the target class proportions.

First, we compute class-wise source features:

$$z_s^{(k)} = \frac{1}{n_s^{(k)}} \sum_{i: y_s^i = k} z_s^i$$

where $n_s^{(k)}$ is the number of source samples in class $k$.

Then, we generate pseudo-labels for target samples using the classifier $G_{\phi}$ trained on source features:

$$\hat{y}_t^j = \arg\max_k G_{\phi}(z_t^j)_k$$

Using the transport plan and pseudo-labels, we estimate the target class proportions:

$$\hat{p}_t(k) = \frac{1}{n_t} \sum_{j: \hat{y}_t^j = k} \sum_{i=1}^{n_s} \pi_{ij}$$

To account for estimation uncertainty, we introduce a confidence-weighted version:

$$\tilde{p}_t(k) = \alpha \hat{p}_t(k) + (1-\alpha) \frac{1}{K}$$

where $\alpha \in [0,1]$ increases gradually during training as pseudo-labels become more reliable.

These estimated proportions are used to guide the learning of relaxation parameters:

$$L_{\text{prop}} = D_{\text{JS}}(\hat{p}_t || \tilde{p}_t)$$

where $D_{\text{JS}}$ is the Jensen-Shannon divergence, encouraging the model to learn relaxation parameters that accurately capture the target class distribution.

### 2.5 Classification and End-to-End Training

Our classifier $G_{\phi}$ with parameters $\phi$ maps features to class probabilities:

$$G_{\phi}: \mathbb{R}^d \rightarrow [0,1]^K$$

For source domain classification, we use the standard cross-entropy loss:

$$L_{\text{cls}}(\theta, \phi) = -\frac{1}{n_s} \sum_{i=1}^{n_s} \sum_{k=1}^K \mathbf{1}_{[y_s^i = k]} \log G_{\phi}(F_{\theta}(x_s^i))_k$$

To address label shift, we incorporate importance weighting based on estimated class proportions:

$$L_{\text{iw}}(\theta, \phi) = -\frac{1}{n_s} \sum_{i=1}^{n_s} \sum_{k=1}^K \mathbf{1}_{[y_s^i = k]} \frac{\tilde{p}_t(k)}{p_s(k)} \log G_{\phi}(F_{\theta}(x_s^i))_k$$

where $p_s(k) = n_s^{(k)}/n_s$ is the source class proportion.

For the domain alignment, we use the A-UOT cost:

$$L_{\text{ot}}(\theta, w_s, w_t) = \text{A-UOT}_{\sigma(w_s), \sigma(w_t), \varepsilon}(\mu_s, \mu_t)$$

Additionally, we employ MixUp regularization to mitigate negative transfer:

$$L_{\text{mix}}(\theta, \phi) = -\frac{1}{n_{\text{mix}}} \sum_{l=1}^{n_{\text{mix}}} \sum_{k=1}^K y_{\text{mix}}^l(k) \log G_{\phi}(z_{\text{mix}}^l)_k$$

where $z_{\text{mix}}^l = \beta z_s^i + (1-\beta) z_t^j$ with random indices $i,j$ and $\beta \sim \text{Beta}(\alpha_{\text{mix}}, \alpha_{\text{mix}})$, and $y_{\text{mix}}^l = \beta \mathbf{1}_{y_s^i} + (1-\beta) \mathbf{1}_{\hat{y}_t^j}$.

The total loss for our end-to-end training is:

$$L_{\text{total}} = L_{\text{iw}} + \gamma_1 L_{\text{ot}} + \gamma_2 L_{\text{mix}} + \gamma_3 L_{\text{prop}}$$

where $\gamma_1, \gamma_2, \gamma_3$ are hyperparameters balancing the different loss components.

### 2.6 Experimental Design

To evaluate our A-UOT framework, we will conduct extensive experiments on standard domain adaptation benchmarks with varying degrees of label shift:

1. **Office-31**: A dataset containing 4,110 images across 31 categories from three domains: Amazon (A), DSLR (D), and Webcam (W).
2. **Office-Home**: A more challenging dataset with 15,500 images across 65 categories from four domains: Art (Ar), Clipart (Cl), Product (Pr), and Real-World (Rw).
3. **VisDA-2017**: A large-scale simulation-to-real dataset with over 280K images across 12 categories.

For each dataset, we will artificially create label shift scenarios by subsampling classes with different ratios in the source and target domains. We will consider three levels of label shift:
- **Mild**: Target class proportions differ from source by a factor of up to 2
- **Moderate**: Differences by a factor of 2-5
- **Severe**: Differences by a factor of greater than 5

We will compare our A-UOT method against several baselines:
1. Source Only: Training only on source data without adaptation
2. Standard OT-based domain adaptation
3. Fixed UOT with various predetermined relaxation parameters
4. IWCV: Importance weighting with cross-validation
5. BBSE: Black-box shift estimation
6. State-of-the-art methods addressing label shift

**Evaluation Metrics**:
- Overall classification accuracy on the target domain
- Class-wise F1 scores to assess performance on minority classes
- Estimation error for target class proportions
- Convergence time and computational efficiency

We will conduct a comprehensive ablation study to analyze the contribution of each component:
1. Effect of adaptive vs. fixed relaxation parameters
2. Impact of label shift estimation
3. Contribution of MixUp regularization
4. Importance of weighted classification loss

For each experiment, we will use a ResNet-50 pretrained on ImageNet as the backbone feature extractor, followed by a domain-specific adaptation layer and a classifier. Training will be performed using the Adam optimizer with a learning rate of 1e-4 and a batch size of A-UOT will employ a modest entropic regularization parameter ($\varepsilon = 0.1$) for computational efficiency.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

Our research is expected to yield several significant outcomes:

1. **A Novel Adaptive UOT Framework**: We anticipate developing a mathematically sound and computationally efficient adaptive unbalanced optimal transport framework that automatically learns the optimal degree of mass variation between distributions. This theoretical contribution extends the current OT literature by removing the need for manual tuning of relaxation parameters.

2. **Improved Performance Under Label Shift**: We expect A-UOT to significantly outperform standard OT and fixed UOT methods on domain adaptation benchmarks with varying degrees of label shift. The performance improvements should be most pronounced under moderate to severe label shift scenarios, where traditional methods tend to struggle the most.

3. **Accurate Label Shift Estimation**: Our method should produce reliable estimates of target class proportions, with estimation errors decreasing as training progresses. These estimates will not only guide the adaptation process but also provide valuable insights about the target domain distribution.

4. **Enhanced Robustness to Negative Transfer**: By incorporating MixUp regularization and adaptive transport constraints, we expect A-UOT to better mitigate negative transfer effects, where incorrectly aligned samples from different classes degrade adaptation performance.

5. **Improved Generalization on Minority Classes**: Traditional domain adaptation methods often sacrifice performance on minority classes to optimize overall accuracy. We anticipate that A-UOT will achieve more balanced performance across all classes, including those with low representation in either domain.

### 3.2 Scientific Impact

This research will contribute to advancing the field of optimal transport for machine learning in several ways:

1. **Theoretical Foundations**: By developing a principled approach to learning relaxation parameters in UOT, we bridge the gap between theoretical OT formulations and practical applications, potentially inspiring new research directions in adaptive transport methods.

2. **Algorithmic Innovations**: Our adaptation of the Sinkhorn algorithm to incorporate learnable marginal constraints provides a new computational framework that could be applied to other OT problems beyond domain adaptation.

3. **Interdisciplinary Applications**: The proposed A-UOT framework has potential applications in various fields where distribution alignment under varying proportions is crucial, including single-cell genomics, natural language processing, and medical image analysis.

### 3.3 Practical Impact

Beyond its scientific contributions, our research has several practical implications:

1. **Reduced Need for Hyperparameter Tuning**: By automatically learning relaxation parameters, A-UOT minimizes the need for extensive hyperparameter tuning, making OT-based domain adaptation more accessible to practitioners without deep expertise in optimal transport theory.

2. **Improved Robustness in Real-World Applications**: Many real-world machine learning deployments face unpredictable shifts in class distributions. A-UOT's ability to adapt to unknown label shifts enhances model robustness in such scenarios, potentially improving reliability in critical applications such as medical diagnostics, autonomous driving, and financial fraud detection.

3. **Computational Efficiency**: By integrating adaptive parameter learning with the Sinkhorn algorithm, our method maintains computational efficiency while providing enhanced flexibility, making it suitable for large-scale applications.

4. **Model Interpretability**: The learned relaxation parameters and estimated class proportions provide interpretable insights about the nature and extent of distribution shifts between domains, helping practitioners better understand their data.

### 3.4 Limitations and Future Work

We acknowledge several potential limitations and directions for future research:

1. **Scalability to Very Large Datasets**: While our use of minibatch processing and entropic regularization improves efficiency, scaling to extremely large datasets may still present challenges. Future work could explore more efficient approximations or parallel implementations.

2. **Extension to Multi-Domain Scenarios**: Our current framework focuses on two-domain adaptation. Extending A-UOT to multi-domain scenarios, where data comes from multiple source and target domains, represents an important future direction.

3. **Handling Concept Shift**: While A-UOT addresses label shift, it assumes consistent class-conditional distributions across domains. Future work could explore combining our approach with methods that handle concept shift (changes in $P(X|Y)$).

4. **Theoretical Guarantees**: Developing stronger theoretical guarantees for the convergence and optimality of learned relaxation parameters would further strengthen the foundation of A-UOT.

In conclusion, our Adaptive Unbalanced Optimal Transport framework represents a significant advancement in domain adaptation under label shift conditions. By learning optimal mass variation parameters directly from data, A-UOT promises to enhance the robustness and applicability of domain adaptation techniques across a wide range of real-world scenarios where class distributions vary between domains.