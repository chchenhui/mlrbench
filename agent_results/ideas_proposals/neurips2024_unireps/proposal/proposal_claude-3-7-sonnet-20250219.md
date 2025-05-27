# Task-Conditioned Functional Alignment for Cross-Architecture Neural Model Merging

## 1. Introduction

### Background

In recent years, the field of machine learning has witnessed an explosion in the size and complexity of neural network models, with state-of-the-art systems requiring enormous computational resources for training. For instance, large language models (LLMs) like GPT-4 and vision transformers such as CLIP require thousands of GPU-days to train from scratch. This computational burden poses significant challenges for researchers with limited resources and raises important questions about the environmental impact and sustainability of AI development.

Simultaneously, a fascinating phenomenon has emerged in both artificial neural networks and biological brains: when exposed to similar stimuli or trained on related tasks, different learning systems tend to develop remarkably similar internal representations. This phenomenon has been observed across diverse neural architectures, training methodologies, and even between artificial and biological systems. The Canonical Representation Hypothesis (CRH) proposed by Ziyin et al. (2024) suggests that during training, latent representations, weights, and neuron gradients become mutually aligned, leading to compact representations that are invariant to task-irrelevant transformations. Similarly, work by Pepin Lehalleur et al. (2025) highlights how the structure of data distributions shapes the internal structure of trained models.

These observations suggest an intriguing possibility: if different neural models develop functionally similar representations, could we leverage this similarity to merge or combine pre-trained models efficiently? Model merging presents a promising approach to reuse computational resources already invested in training, potentially enabling more efficient utilization of existing models rather than training new ones from scratch.

However, current model merging techniques face significant challenges, particularly when dealing with models that have different architectures or have been trained on slightly different task distributions. Existing approaches often rely on direct parameter averaging or interpolation, which may fail when architectures differ substantially. As Insulla et al. (2025) note in their exploration of representation alignment, the interplay between different representations in the context of a task—a concept they term "stitching"—requires a more nuanced approach than simple parameter matching.

### Research Objectives

This research proposal aims to develop and validate a novel approach to cross-architecture model merging through task-conditioned functional alignment. Specifically, we seek to:

1. Develop a theoretical framework for understanding when and why different neural architectures learn functionally similar representations for related tasks.
2. Design a robust methodology for identifying and aligning functionally equivalent components across diverse neural architectures.
3. Create efficient "stitching" mechanisms that can connect these aligned components across architectures with minimal additional parameters.
4. Evaluate the effectiveness of the proposed approach across a range of model architectures and tasks.
5. Analyze the generalization properties of merged models compared to their constituent components.

### Significance

The proposed research has significant implications for both the theoretical understanding of neural representations and practical applications in machine learning:

**Theoretical Significance:** This work will contribute to our understanding of representation learning in neural networks, particularly regarding the emergence of functionally similar representations across different architectures. It will extend the theoretical foundations laid by recent work on representation alignment and the Canonical Representation Hypothesis.

**Practical Significance:** Successfully merging models with different architectures could dramatically reduce the computational resources required for developing specialized AI systems. Rather than training new models from scratch, researchers could merge existing pre-trained models to create systems with capabilities that combine those of the original models. This approach could democratize access to advanced AI capabilities by reducing the barriers to entry posed by computational requirements.

**Environmental Impact:** By enabling more efficient reuse of pre-trained models, this research could contribute to reducing the carbon footprint associated with training large neural networks, aligning with goals for more sustainable AI development.

## 2. Methodology

Our proposed Task-Conditioned Functional Alignment (TCFA) methodology consists of four main phases: (1) representation probing, (2) functional mapping, (3) alignment transformation learning, and (4) model merging through stitching. We detail each phase below.

### 2.1 Representation Probing

The first step in our approach involves comprehensive probing of internal representations within the source models we aim to merge. Unlike existing methods that focus on direct parameter comparisons, we will analyze activation patterns in response to task-specific stimuli.

**Algorithm 1: Task-Conditioned Representation Probing**

1. Define a set of task conditions $T = \{t_1, t_2, ..., t_m\}$ that represent different aspects of the downstream task (e.g., different object classes in image recognition, different semantic relationships in language tasks).
2. For each task condition $t_i \in T$, generate a diverse set of inputs $X_{t_i} = \{x_1, x_2, ..., x_n\}$ that satisfy that condition.
3. For each model $M_j$ in the set of models to be merged $\{M_1, M_2, ..., M_k\}$:
   a. For each layer $l$ in model $M_j$:
      i. Record activations $A_{j,l,t_i} = \{M_j^l(x_1), M_j^l(x_2), ..., M_j^l(x_n)\}$ for all inputs in $X_{t_i}$.
      ii. Compute statistical properties of the activation distribution, including mean $\mu_{j,l,t_i}$, covariance $\Sigma_{j,l,t_i}$, and principal components.
4. For each pair of models $(M_j, M_{j'})$ and each task condition $t_i$, compute an initial measure of representational similarity between layers using Centered Kernel Alignment (CKA):

$$CKA(A_{j,l,t_i}, A_{j',l',t_i}) = \frac{HSIC(A_{j,l,t_i}, A_{j',l',t_i})}{\sqrt{HSIC(A_{j,l,t_i}, A_{j,l,t_i}) \cdot HSIC(A_{j',l',t_i}, A_{j',l',t_i})}}$$

where HSIC is the Hilbert-Schmidt Independence Criterion.

This probing phase will produce a comprehensive map of how different layers in each model respond to task-specific stimuli, allowing us to identify candidate layers for alignment based on functional similarity rather than architectural position.

### 2.2 Functional Mapping

Using the representational similarity measures from the probing phase, we will construct a functional mapping between models that identifies which components across different architectures serve similar roles.

**Algorithm 2: Cross-Architecture Functional Mapping**

1. For each task condition $t_i \in T$, construct a bipartite graph $G_{t_i}$ where:
   - Nodes on the left represent layers from model $M_1$
   - Nodes on the right represent layers from model $M_2$
   - Edge weights are given by the CKA similarity scores: $w_{(l,l')} = CKA(A_{1,l,t_i}, A_{2,l',t_i})$

2. Find a maximum weight bipartite matching $\mathcal{M}_{t_i}$ in $G_{t_i}$ using the Hungarian algorithm to identify the optimal layer-to-layer correspondences for each task condition.

3. For each edge $(l,l') \in \mathcal{M}_{t_i}$, compute a functional similarity score:

$$S_{t_i}(l,l') = w_{(l,l')} \cdot \frac{\min(rank(A_{1,l,t_i}), rank(A_{2,l',t_i}))}{\max(rank(A_{1,l,t_i}), rank(A_{2,l',t_i}))}$$

where $rank(A)$ estimates the effective rank of the activation matrix.

4. Construct a consensus functional mapping $\mathcal{M}$ by aggregating the task-specific mappings:

$$\mathcal{M} = \{(l,l') \mid \sum_{t_i \in T} \mathbbm{1}_{(l,l') \in \mathcal{M}_{t_i}} \cdot S_{t_i}(l,l') > \theta\}$$

where $\theta$ is a threshold parameter and $\mathbbm{1}$ is the indicator function.

This process identifies functionally equivalent components across architectures, taking into account both the similarity of representations and their effective dimensionality across different task conditions.

### 2.3 Alignment Transformation Learning

Once we have identified functionally corresponding components between models, we learn transformations that align their activation spaces while preserving their functional properties.

**Algorithm 3: Task-Conditioned Transformation Learning**

1. For each pair of functionally mapped layers $(l,l') \in \mathcal{M}$:
   a. For each task condition $t_i \in T$:
      i. Collect activation matrices $A_{1,l,t_i}$ and $A_{2,l',t_i}$.
      ii. Center the activation matrices: $\hat{A}_{1,l,t_i} = A_{1,l,t_i} - \mu_{1,l,t_i}$ and $\hat{A}_{2,l',t_i} = A_{2,l',t_i} - \mu_{2,l',t_i}$.
      
   b. Learn a linear transformation matrix $W_{l \rightarrow l'}$ using Canonical Correlation Analysis (CCA) or Optimal Transport (OT):
   
      **CCA approach**:
      i. Compute singular value decompositions: $\hat{A}_{1,l,t_i} = U_1 S_1 V_1^T$ and $\hat{A}_{2,l',t_i} = U_2 S_2 V_2^T$.
      ii. Compute the transformation: $W_{l \rightarrow l'} = U_1 U_2^T$.
      
      **OT approach**:
      i. Compute the cost matrix $C = \|A_{1,l,t_i}^T A_{1,l,t_i} - A_{2,l',t_i}^T A_{2,l',t_i}\|_F^2$.
      ii. Solve the optimal transport problem: $P = \arg\min_P \langle P, C \rangle$ subject to $P \mathbf{1} = \frac{1}{n_1}\mathbf{1}, P^T \mathbf{1} = \frac{1}{n_2}\mathbf{1}, P \geq 0$.
      iii. Compute the transformation: $W_{l \rightarrow l'} = (A_{1,l,t_i}^T P A_{2,l',t_i})(A_{2,l',t_i}^T A_{2,l',t_i})^{-1}$.
   
   c. Add bias term: $b_{l \rightarrow l'} = \mu_{2,l',t_i} - W_{l \rightarrow l'} \mu_{1,l,t_i}$.
   
   d. For task conditions that require non-linear transformations, learn a small neural network $f_{l \rightarrow l'}$ that minimizes:
   
   $$\mathcal{L}(f_{l \rightarrow l'}) = \sum_{t_i \in T} \sum_{x \in X_{t_i}} \|f_{l \rightarrow l'}(M_1^l(x)) - M_2^{l'}(x)\|_2^2 + \lambda \cdot \Omega(f_{l \rightarrow l'})$$
   
   where $\Omega(f)$ is a complexity regularization term and $\lambda$ is a hyperparameter.

This step produces a set of lightweight transformation functions that can map activations from one model to functionally equivalent activations in the other model.

### 2.4 Model Merging through Stitching

The final phase involves creating a merged model by stitching together components from the source models using the learned transformations.

**Algorithm 4: Task-Conditioned Stitching**

1. Define an architecture for the merged model $M_{merged}$ that combines elements from the source models based on the functional mapping $\mathcal{M}$.

2. For each transition between components from different source models in $M_{merged}$:
   a. Insert the appropriate transformation function $f_{l \rightarrow l'}$ as a stitching layer.
   
3. Initialize the parameters of each component in $M_{merged}$ with the corresponding parameters from the source model.

4. Optionally, perform a light fine-tuning of the stitching layers while keeping the parameters of the source components frozen:
   
   $$\min_{\{f_{l \rightarrow l'}\}} \mathcal{L}_{task}(M_{merged}) + \gamma \sum_{(l,l') \in \mathcal{M}} \Omega(f_{l \rightarrow l'})$$
   
   where $\mathcal{L}_{task}$ is the task-specific loss function and $\gamma$ is a regularization hyperparameter.

5. For multi-task scenarios, learn task-specific routing weights $\alpha_t$ that determine how activations flow through the merged model for each task $t$:
   
   $$a_{l'}^t = \alpha_t \cdot f_{l \rightarrow l'}(a_l) + (1-\alpha_t) \cdot a_{l'}$$
   
   where $a_l$ and $a_{l'}$ are activations from corresponding layers in the source models.

### 2.5 Experimental Design and Evaluation

To validate our approach, we will conduct experiments across multiple domains and model architectures:

**Dataset Selection:**
1. Vision domain: ImageNet, CIFAR-100, DomainNet
2. Language domain: GLUE benchmark, WMT translation tasks
3. Multimodal domain: MS-COCO, Visual Question Answering

**Model Architectures:**
1. Vision: ResNets, Vision Transformers, ConvNeXt
2. Language: BERT variants, RoBERTa, GPT-based models
3. Multimodal: CLIP, DALL-E, multimodal transformers

**Experimental Conditions:**
1. Same-architecture merging (baseline)
2. Cross-architecture merging with similar model sizes
3. Cross-architecture merging with significantly different model sizes
4. Cross-domain merging (e.g., combining vision and language models)

**Evaluation Metrics:**
1. **Task Performance:**
   - Accuracy, F1-score, BLEU score (task-dependent)
   - Performance relative to source models: $P_{rel} = \frac{P_{merged}}{\max(P_{M_1}, P_{M_2})}$
   - Performance efficiency: $P_{eff} = \frac{P_{merged}}{\# parameters}$

2. **Representation Quality:**
   - Representation similarity between merged and source models using CKA
   - t-SNE visualization of representations for qualitative assessment

3. **Efficiency Metrics:**
   - Number of parameters in stitching layers relative to total model size
   - Inference time compared to source models
   - Transfer efficiency to new tasks: $E_{transfer} = \frac{P_{new task}}{P_{baseline}}$

4. **Generalization Metrics:**
   - Performance on out-of-distribution data
   - Calibration error
   - Task composition effectiveness: ability to handle tasks that combine capabilities from both source models

## 3. Expected Outcomes & Impact

### Expected Outcomes

1. **Theoretical Framework:** We expect to develop a comprehensive theoretical framework that explains when and why neural networks with different architectures develop functionally similar representations. This framework will extend current understanding of representation alignment and provide principled guidelines for identifying mergeable components across architectures.

2. **Functional Similarity Measures:** Our research will yield novel metrics and methodologies for quantifying functional similarity between neural network components, going beyond simple activation correlation to capture deeper aspects of representational alignment conditioned on specific task requirements.

3. **Efficient Stitching Methods:** We anticipate developing lightweight transformation techniques that can effectively bridge between functionally similar components with minimal additional parameters. These methods will be adaptable to various architectural differences and capable of handling both linear and non-linear transformations as needed.

4. **Cross-Architecture Merged Models:** The primary tangible outcome will be a set of successfully merged models that combine components from different architectures while maintaining or exceeding the performance of the source models. These merged models will demonstrate the practical viability of our approach.

5. **Task-Conditioned Routing Mechanisms:** For multi-task scenarios, we expect to develop effective routing mechanisms that dynamically adjust the flow of information through the merged model based on the specific task being performed.

### Broader Impact

The successful development of Task-Conditioned Functional Alignment for cross-architecture model merging has the potential for significant impact across multiple dimensions:

**Computational Efficiency:** By enabling effective reuse and combination of pre-trained models, our approach could substantially reduce the computational resources required for developing new AI capabilities. This efficiency gain is particularly important as model sizes continue to grow, making training from scratch increasingly prohibitive for many researchers and organizations.

**Democratization of AI:** Reducing the resource requirements for advanced AI capabilities could help democratize access to these technologies, allowing smaller research groups and organizations with limited computational resources to leverage and build upon existing models more effectively.

**Environmental Sustainability:** The reduced need for training models from scratch would directly translate to lower energy consumption and carbon emissions associated with AI research and development, contributing to more environmentally sustainable AI practices.

**Knowledge Transfer:** Our work will advance understanding of how knowledge is represented across different neural architectures, potentially leading to insights about more effective transfer learning approaches and multi-modal integration.

**Model Composition:** The ability to merge models based on functional similarity rather than architectural compatibility opens new possibilities for composing AI systems with complementary capabilities, potentially enabling more complex intelligent behaviors through the combination of specialized components.

**Neuroscience Connections:** The principles developed in this research may provide insights relevant to neuroscience, particularly regarding how different brain regions with distinct neural architectures can nonetheless develop functionally similar representations that enable coordinated information processing.

In conclusion, the proposed research on Task-Conditioned Functional Alignment has the potential to not only address the practical challenge of cross-architecture model merging but also to contribute fundamental insights to our understanding of representation learning in neural systems, with broad implications for the future development of more efficient, accessible, and sustainable artificial intelligence.