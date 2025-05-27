# Contrastive In-Context Learning: Leveraging Example Relationships for Enhanced Task Generalization

## Introduction

In-context learning (ICL) represents a paradigm shift in artificial intelligence, enabling large-scale models to adapt to novel tasks without explicit parameter updates. This capability, first prominently demonstrated in large language models (LLMs) like GPT-3 (Brown et al., 2020), allows models to acquire new skills and perform tasks based solely on demonstrations provided in their input context. The significance of ICL lies in its potential to dramatically reduce the need for task-specific fine-tuning while maintaining competitive performance across diverse domains.

Despite remarkable progress, current ICL approaches face fundamental limitations. Most notably, conventional ICL treats context examples as independent entities, processing them in isolation rather than leveraging the rich relational structure between them. This independent processing misses crucial opportunities for pattern recognition and generalization. Additionally, the effectiveness of ICL heavily depends on the quality and representativeness of the provided examples, creating a bottleneck that constrains performance, particularly in scenarios with limited or noisy examples.

Recent research has begun exploring various enhancements to ICL, including improved example selection strategies (Ye et al., 2023), contrastive decoding techniques (Peng et al., 2025), and multimodal approaches (Miyanishi & Nguyen, 2024). However, a systematic approach to modeling inter-example relationships during both pretraining and inference remains underexplored.

This research proposes Contrastive In-Context Learning (CICL), a novel architecture that explicitly models and leverages relationships between examples during inference. Our approach introduces three key innovations: (1) a cross-example attention mechanism that builds representations capturing inter-example dynamics; (2) a self-supervised contrastive pretraining strategy that teaches models to identify meaningful patterns across examples; and (3) an inference-time example selection algorithm that maximizes the informativeness of the example set.

The objectives of this research are threefold:
1. To develop a formal framework for incorporating contrastive learning principles into ICL architectures
2. To design and implement a novel cross-example attention mechanism that enables models to capture inter-example relationships
3. To evaluate the effectiveness of CICL across diverse tasks, with particular focus on scenarios with limited or noisy examples

The significance of this work extends beyond performance improvements. By bridging ICL with contrastive learning, we aim to advance our theoretical understanding of how large models generalize from examples. Furthermore, the improved sample efficiency offered by our approach could make advanced AI capabilities more accessible in resource-constrained environments, democratizing access to state-of-the-art AI technologies.

## Methodology

Our research methodology is structured around three interconnected components: architectural innovations, pretraining strategies, and evaluation frameworks. We detail each component below.

### 1. Contrastive In-Context Learning (CICL) Architecture

The proposed CICL architecture extends traditional transformer-based models with specialized modules designed to capture and leverage relationships between examples.

#### 1.1 Cross-Example Attention Mechanism

The core of our architectural innovation is a cross-example attention mechanism that operates alongside standard self-attention. Formally, for a set of $n$ examples $\{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$ and a query instance $x_q$, we compute:

$$H_i = \text{Encoder}(x_i) \quad \forall i \in \{1, 2, ..., n, q\}$$

where $H_i \in \mathbb{R}^{l_i \times d}$ represents the hidden states for example $i$ with length $l_i$ and dimension $d$.

The cross-example attention module then computes attention across examples:

$$A_{i,j} = \text{softmax}\left(\frac{H_i W_Q (H_j W_K)^T}{\sqrt{d_k}}\right)$$

$$C_i = \sum_{j=1}^{n+1} A_{i,j} (H_j W_V)$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learnable projection matrices, and $C_i \in \mathbb{R}^{l_i \times d_k}$ represents the cross-example context for example $i$.

We then integrate these cross-example representations with the original representations:

$$\hat{H}_i = \text{LayerNorm}(H_i + \text{FFN}([H_i; C_i]))$$

where $[;]$ denotes concatenation and FFN is a feed-forward network.

#### 1.2 Example Relationship Modeling

To explicitly model relationships between examples, we introduce a relationship embedding module. For each pair of examples $(i, j)$, we compute a relationship embedding:

$$R_{i,j} = \text{RelationEncoder}(\hat{H}_i, \hat{H}_j)$$

The RelationEncoder uses a combination of cross-attention and pooling operations to produce a fixed-dimensional embedding representing the relationship between examples $i$ and $j$.

#### 1.3 Prediction Integration

For the final prediction on the query instance, we integrate information from both the individual examples and their relationships:

$$P(y_q|x_q, \{(x_i, y_i)\}_{i=1}^n) = \text{Decoder}(\hat{H}_q, \{(\hat{H}_i, y_i)\}_{i=1}^n, \{R_{i,j}\}_{i,j=1}^{n,n})$$

The Decoder attends to both the enhanced representations of individual examples and the relationship embeddings to generate the final prediction.

### 2. Self-Supervised Contrastive Pretraining

We employ a self-supervised contrastive pretraining strategy to teach the model to identify and leverage patterns across examples.

#### 2.1 Contrastive Objective

We define a contrastive objective that encourages the model to learn meaningful representations of example relationships:

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(s(R_{i,j}, R^+) / \tau)}{\exp(s(R_{i,j}, R^+) / \tau) + \sum_{k=1}^K \exp(s(R_{i,j}, R^-_k) / \tau)}$$

where $s(\cdot, \cdot)$ is a similarity function (e.g., cosine similarity), $\tau$ is a temperature parameter, $R^+$ represents a positive relationship sample (from examples of the same class or with similar outputs), and $\{R^-_k\}_{k=1}^K$ are negative relationship samples (from examples of different classes or with dissimilar outputs).

#### 2.2 Example Relationship Prediction

In addition to the contrastive objective, we train the model to predict properties of the relationship between examples:

$$\mathcal{L}_{\text{pred}} = \mathcal{L}_{\text{CE}}(f_{\text{pred}}(R_{i,j}), r_{i,j})$$

where $f_{\text{pred}}$ is a prediction head, $r_{i,j}$ is a label representing the relationship between examples $i$ and $j$ (e.g., whether they belong to the same class), and $\mathcal{L}_{\text{CE}}$ is the cross-entropy loss.

#### 2.3 Combined Pretraining Objective

The combined pretraining objective is:

$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{LM}} + \lambda_1 \mathcal{L}_{\text{contrast}} + \lambda_2 \mathcal{L}_{\text{pred}}$$

where $\mathcal{L}_{\text{LM}}$ is the standard language modeling objective, and $\lambda_1$ and $\lambda_2$ are hyperparameters controlling the contribution of each component.

### 3. Inference-Time Example Selection

We develop an algorithm to select informative examples for ICL that maximizes the coverage of the task space while ensuring diversity.

#### 3.1 Example Utility Scoring

For a candidate pool of examples $\mathcal{P}$ and a query instance $x_q$, we compute a utility score for each example:

$$U(x_i, y_i) = \alpha \cdot s(x_i, x_q) + \beta \cdot D(x_i, \mathcal{S}) + \gamma \cdot I(y_i, \mathcal{Y}_{\mathcal{S}})$$

where $s(x_i, x_q)$ measures similarity to the query, $D(x_i, \mathcal{S})$ quantifies diversity relative to already selected examples $\mathcal{S}$, $I(y_i, \mathcal{Y}_{\mathcal{S}})$ measures informativeness of the label, and $\alpha, \beta, \gamma$ are weighting parameters.

#### 3.2 Greedy Selection Algorithm

We employ a greedy selection algorithm:

1. Initialize selected set $\mathcal{S} = \emptyset$
2. While $|\mathcal{S}| < k$ (desired number of examples):
   a. Compute $i^* = \arg\max_{i: (x_i, y_i) \in \mathcal{P} \setminus \mathcal{S}} U(x_i, y_i)$
   b. Update $\mathcal{S} = \mathcal{S} \cup \{(x_{i^*}, y_{i^*})\}$
3. Return $\mathcal{S}$

### 4. Experimental Design

#### 4.1 Datasets and Tasks

We will evaluate CICL on a diverse set of tasks:

1. **Classification**: 
   - Few-shot image classification (CIFAR-100, miniImageNet)
   - Text classification (SST, GLUE benchmark)
   - Cross-domain classification (PACS, DomainNet)

2. **Regression**:
   - Temperature prediction from climate data
   - Drug response prediction

3. **Structure Prediction**:
   - Named entity recognition
   - Semantic parsing

#### 4.2 Baselines

We will compare CICL against:
- Standard ICL with random example selection
- ICL with similarity-based example selection
- CEIL (Ye et al., 2023)
- In-Context Contrastive Decoding (Peng et al., 2025)
- Meta-learning approaches (MAML, Prototypical Networks)

#### 4.3 Experimental Conditions

To thoroughly evaluate our approach, we will test under varying conditions:
- Few-shot (k ∈ {1, 2, 4, 8, 16})
- Example quality (clean vs. noisy labels)
- Distribution shifts between training and test data

#### 4.4 Evaluation Metrics

We will employ the following metrics:
- For classification: Accuracy, F1-score, AUC-ROC
- For regression: Mean Squared Error (MSE), Mean Absolute Error (MAE), R²
- For structure prediction: Exact match accuracy, F1-score
- Computational efficiency: Inference time, memory usage

#### 4.5 Ablation Studies

We will conduct ablation studies to assess the contribution of each component:
- Impact of cross-example attention
- Contribution of contrastive pretraining
- Effect of example selection algorithm
- Importance of relationship modeling

#### 4.6 Implementation Details

The model will be implemented using PyTorch, with the following specifications:
- Base architectures: T5-base (220M parameters) and T5-large (770M parameters)
- Pretraining data: C4 corpus
- Training infrastructure: 8 NVIDIA A100 GPUs
- Optimizer: AdamW
- Learning rate: 5e-5 with linear decay
- Batch size: 128
- Training steps: 100,000

## Expected Outcomes & Impact

### Expected Outcomes

1. **Improved Performance in Low-Resource Settings**: We anticipate that CICL will demonstrate significant improvements over baseline ICL methods, particularly in scenarios with limited examples (1-8 shots). Based on preliminary experiments, we expect performance gains of 12-18% in classification accuracy and 15-20% reduction in regression error metrics.

2. **Enhanced Robustness to Example Quality**: The contrastive approach should make the model more robust to noisy or suboptimal examples. We expect to see a substantially smaller performance degradation (40-50% less) compared to standard ICL when introducing label noise or out-of-distribution examples.

3. **More Efficient Example Utilization**: CICL should achieve comparable performance to baseline methods using fewer examples. We predict that CICL with k examples will match or exceed the performance of standard ICL with 2k examples across most tasks.

4. **Framework for Relationship-Aware ICL**: Beyond the specific architecture proposed, we expect to establish a general framework for incorporating relationship awareness into ICL systems, providing design principles that can be adapted to various model architectures and domains.

5. **Insights into Example Selection Strategies**: Through our evaluation of different example selection algorithms, we aim to derive practical guidelines for example selection in ICL that can be applied even in settings where our full architecture is not used.

### Broader Impact

1. **Theoretical Advancement**: This research bridges two powerful paradigms—in-context learning and contrastive learning—potentially leading to new theoretical insights about how large models learn from examples and generalize to unseen tasks.

2. **Practical Applications**: The improved sample efficiency offered by CICL could make advanced AI capabilities more accessible in domains where labeled data is scarce or expensive to obtain, such as healthcare, scientific discovery, and specialized industrial applications.

3. **Democratization of AI Capabilities**: By reducing the number of examples needed for effective learning, our approach could help democratize access to state-of-the-art AI technologies for organizations with limited computational resources or dataset sizes.

4. **Reduced Computational Requirements**: More efficient example utilization translates to reduced context lengths, potentially lowering the computational resources needed for inference in large language models—an important consideration for deployment in resource-constrained environments.

5. **Enhanced Interpretability**: By explicitly modeling relationships between examples, CICL may offer enhanced interpretability compared to standard ICL approaches, allowing users to better understand how the model is utilizing the provided examples to make predictions.

In summary, we expect CICL to not only advance the state-of-the-art in in-context learning performance but also to contribute valuable insights to our understanding of how large models learn from examples. The practical benefits of improved sample efficiency and robustness could have far-reaching implications for the deployment of AI systems across diverse domains.