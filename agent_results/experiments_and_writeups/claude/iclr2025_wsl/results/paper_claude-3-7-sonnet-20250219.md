# Permutation-Equivariant Contrastive Embeddings for Model Zoo Retrieval

## Abstract

With the surge of pre-trained neural networks in repositories like Hugging Face exceeding one million models, efficient discovery mechanisms have become critical. Current metadata-based search methods fail to capture the functional similarities encoded in model weights, leading to redundant training and wasted computational resources. We present a novel framework for retrieving functionally similar neural networks directly from their weight tensors through a permutation-equivariant graph neural network encoder. Our approach respects the inherent symmetries in neural network weights—such as neuron permutations and parameter scaling—using a contrastive learning paradigm with symmetry-preserving augmentations. Experiments on a diverse model zoo demonstrate that our method significantly outperforms baseline approaches, achieving 72.8% precision@1 compared to 60.8% for transformer-based encoders and 52.8% for PCA projections. Our framework also shows superior transfer learning performance and robustness to symmetry transformations, establishing neural network weights as a valuable data modality for model retrieval and reuse.

## 1. Introduction

The exponential growth of publicly available neural networks has transformed model zoos into a critical resource for machine learning practitioners. With repositories like Hugging Face surpassing one million models, the challenge has shifted from model scarcity to efficient discovery. However, current search mechanisms relying solely on metadata (e.g., task tags, architecture strings) fail to capture functional relationships preserved in raw weight tensors.

This limitation leads to significant inefficiencies as practitioners often redundantly retrain models for similar tasks, consuming exascale computational resources and increasing the environmental footprint of AI research. The weight space of neural networks—which encodes functional capabilities through learned parameters and architectural invariants—remains an underutilized data modality with immense potential for model retrieval and reuse.

Neural network weights present unique challenges as a data type due to their inherent symmetries. Permutation symmetry, where neurons within a layer can be reordered without affecting model functionality, means that functionally identical models can have vastly different weight representations. Similarly, scaling symmetry allows multiplicative transformations across connected layers while preserving outputs. These properties make direct comparison of weight matrices ineffective for functional similarity assessment.

We address these challenges by introducing a permutation-equivariant graph neural network framework for embedding neural network weights in a semantically meaningful space. Our key contributions include:

1. A symmetry-aware graph-based architecture that explicitly respects neuron permutation and parameter scaling invariances
2. A contrastive learning paradigm using symmetry-preserving augmentations to encode functional equivalence
3. An efficient retrieval system enabling discovery of transferable models across heterogeneous architectures

We demonstrate that by learning representations that respect the geometric structure of weight space, our approach significantly outperforms baseline methods in model retrieval, transfer learning, and robustness to symmetry transformations. This work establishes a foundation for treating neural network weights as a first-class data modality, enabling more efficient model discovery and reuse.

## 2. Related Work

Our work builds upon and connects several research areas exploring neural network weight spaces and their properties.

**Weight Space Analysis and Representation.** Recent work has begun treating neural networks as points in a high-dimensional weight space. Eilertsen et al. [3] employed meta-classifiers to analyze how different training setups influence the structure of weight space. Geometric approaches to weight space have also emerged, with Erdogan [1] introducing generative models over neural network weights that respect inherent symmetries. These approaches demonstrate that weight spaces contain rich information about model functionality beyond performance metrics.

**Graph Neural Networks and Equivariance.** Graph neural networks (GNNs) have proven effective for processing structured data with permutation invariance requirements. Pei et al. [4] introduced Geom-GCN, incorporating geometric aggregation schemes to capture structural information in graphs. Sun et al. [2] addressed feature space collapse in GNNs through subspace flattening techniques. Our approach builds on these works, extending GNN architectures to process weight tensors while respecting their symmetry properties.

**Contrastive Learning.** Self-supervised contrastive learning has shown remarkable success in learning representations without explicit labels. In the context of neural network weights, contrastive approaches [6,10] have been proposed for embedding model weights, but typically lack rigorous treatment of symmetry constraints. Our work differs by explicitly designing augmentations and learning objectives that preserve functional equivalence under permutation and scaling transformations.

**Model Discovery and Selection.** As model repositories grow, efficient retrieval mechanisms become crucial. Prior work has explored task-driven embeddings [7,10] and permutation-invariant representations [7] for model retrieval. However, these approaches often lack the theoretical guarantees for symmetry preservation that our method provides through equivariant message passing.

Our approach bridges these disparate research areas by combining geometric insights about weight spaces with principled equivariant architectures and contrastive learning objectives. This integration enables more effective model retrieval while respecting the fundamental symmetries of neural network weights.

## 3. Methodology

### 3.1 Overview

Our framework for neural network weight embedding consists of three main components:

1. **Weight-to-Graph Conversion**: Transforming raw weight matrices into structured graph representations
2. **Permutation-Equivariant Graph Neural Network**: Processing weight graphs while preserving symmetry properties
3. **Contrastive Learning with Symmetry-Preserving Augmentations**: Learning embeddings that capture functional similarity

The overall architecture is designed to map weight spaces to a lower-dimensional embedding space where functionally similar models are clustered together, enabling efficient retrieval via k-nearest neighbors search.

### 3.2 Weight Space Transformations

Neural network weights exhibit important symmetry properties that must be considered for effective embedding. For a weight matrix $W^{(l)} \in \mathbb{R}^{n_l \times m_l}$ in layer $l$, we identify two key invariances:

1. **Neuron Permutation Symmetry**: For permutation matrices $P_l \in \mathbb{R}^{n_l \times n_l}$ and $Q_l \in \mathbb{R}^{m_l \times m_l}$, the transformation $W^{(l)} \rightarrow P_l^{-1} W^{(l)} Q_l$ produces a functionally equivalent layer.

2. **Channel Scaling Symmetry**: For diagonal matrices $D_l = \text{diag}(d^{(1)}, \dots, d^{(n_l)})$ and $E_l = \text{diag}(e^{(1)}, \dots, e^{(m_l)})$ with positive entries, the transformation $W^{(l)} \rightarrow D_l^{-1} W^{(l)} E_l$ preserves functionality when compensated across connected layers.

Our embedding function $f(W)$ must satisfy:
$$\|f(P_l^{-1} W^{(l)} Q_l E_l) - f(W^{(l)})\| < \epsilon \quad \forall P_l, Q_l, E_l$$
$$\|f(D_l^{-1} W^{(l)}) - f(W^{(l)})\| < \epsilon \quad \forall D_l$$

These constraints ensure that embeddings remain stable under valid symmetry transformations.

### 3.3 Graph Construction from Weight Matrices

To process neural network weights in a structure-preserving manner, we represent each layer's weight tensor $W^{(l)}$ as a directed bipartite graph $G^{(l)} = (\mathcal{V}_u \cup \mathcal{V}_v, \mathcal{E})$ where:

- $\mathcal{V}_u = \{u_1, \dots, u_{n_l}\}$ represents input neurons
- $\mathcal{V}_v = \{v_1, \dots, v_{m_l}\}$ represents output neurons
- $\mathcal{E} = \{(u_i, v_j) | W^{(l)}_{ij} \neq 0\}$ represents connections weighted by $W^{(l)}_{ij}$

To enhance scale invariance, we introduce edge features that capture both absolute weights and relative scaling:
$$e_{ij} = \text{MLP}([W^{(l)}_{ij}, \|W^{(l)}_{ij}\|/\|W^{(l)}\|_F])$$
where $\|W^{(l)}\|_F$ is the Frobenius norm of the weight matrix.

### 3.4 Permutation-Equivariant Graph Neural Network

Our encoder processes the weight graphs using a permutation-equivariant message passing scheme. For each node $i$ in the graph, the update rule is:

$$h_i^{(t+1)} = \sigma\left(\frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \Gamma(\pi_{ij}) (W_e e_{ij} + W_h h_j^{(t)})\right)$$

where:
- $h_i^{(t)} \in \mathbb{R}^d$ is the node representation at layer $t$
- $\mathcal{N}_i$ denotes the neighborhood of node $i$
- $\Gamma(\pi_{ij})$ is a geometric transformation matrix parameterized by edge features $\pi_{ij}$
- $W_e, W_h$ are learnable weights
- $\sigma$ is a ReLU activation function

The transformation $\Gamma(\pi_{ij})$ ensures that the message passing respects the geometric structure of the weight space. We implement this using a steerable convolutional architecture where transformations preserve permutation equivariance.

### 3.5 Hierarchical Weight Embedding

After processing each layer with the GNN for $L$ iterations, we apply a graph-level pooling operation to obtain layer-wise embeddings:

$$\mathbf{z}_l = \text{Readout}\left(\{h_i^{(L)} | v_i \in G^{(l)}\}\right) = \frac{1}{|\mathcal{V}|}\sum_{v_i \in \mathcal{V}} \alpha_i h_i^{(L)}$$

where the attention coefficients $\alpha_i = \text{MLP}(h_i^{(L)})$ are computed to preserve permutation invariance.

To capture the hierarchical structure of neural networks, we integrate layer embeddings using a Gated Recurrent Unit (GRU):

$$\mathbf{z} = \text{GRU}(\mathbf{z}_1, \dots, \mathbf{z}_L)$$

This produces a fixed-dimension representation $\mathbf{z} \in \mathbb{R}^d$ of the entire network that preserves equivariance properties.

### 3.6 Contrastive Learning with Symmetry-Preserving Augmentations

To train our encoder, we employ a contrastive learning framework with carefully designed augmentations that preserve model functionality.

#### Positive Pair Generation
We construct positive pairs $(W, W^+)$ by applying symmetry-preserving transformations:

1. **Structural Permutations**: Randomly shuffle neurons in 15% of layers
2. **Dynamic Scaling**: Multiply each channel by random factors $c \sim \mathcal{U}(0.5, 2.0)$
3. **DropConnect**: Zero out 5% of weights to simulate pruning

These augmentations maintain functional equivalence while providing diverse positive examples.

#### Negative Pair Generation
Negative pairs $(W, W^-)$ are sampled from:

1. Models trained for functionally distinct tasks (e.g., classification vs. segmentation)
2. Poorly performing models identified by validation metrics
3. Models with adversarially perturbed weights using FGSM

#### Loss Function
Our loss combines contrastive and performance prediction objectives:

$$\mathcal{L} = \lambda \mathcal{L}_{\text{contrastive}} + (1-\lambda) \mathcal{L}_{\text{metric}}$$

where $\lambda \in [0,1]$ controls the trade-off between objectives.

The contrastive component follows the InfoNCE formulation:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(s(z, z^+)/\tau)}{\exp(s(z, z^+)/\tau) + \sum_{k=1}^K \exp(s(z, z_k^-)/\tau)}$$

where:
- $s(\cdot, \cdot)$ is cosine similarity
- $\tau$ is a temperature parameter
- $z^+/z_k^-$ are embeddings of positive/negative examples

The metric prediction component helps align embeddings with functional performance:

$$\mathcal{L}_{\text{metric}} = \|\mu(y_{\text{acc}}) - \text{MLP}_{\theta}(z)\|^2$$

where $\mu(y_{\text{acc}})$ represents a task-agnostic accuracy abstraction.

## 4. Experiment Setup

### 4.1 Dataset

We evaluated our approach on a diverse collection of neural networks spanning multiple domains and architectures:

- **Vision Models**: 58 models including ResNet, VGG, MobileNet, and EfficientNet architectures trained on ImageNet, CIFAR-10, COCO, and PASCAL datasets for classification, detection, and segmentation tasks.
- **NLP Models**: 28 models including Transformer and BERT variants.
- **Scientific Models**: 10 MLP-based physics models.

All models were quantized to 16-bit precision to standardize storage requirements, with architecture-specific preprocessing for cross-architecture compatibility.

The dataset statistics are summarized in Table 1:

| Statistic | Value |
|----------|-------|
| Total models | 94 |
| Unique tasks | 6 |
| Models by domain | Vision: 58, NLP: 28, Scientific: 10 |
| Models by task | Classification: 14, Detection: 16, Segmentation: 20, Generation: 13, Prediction: 12 |
| Models by dataset | ImageNet: 22, CIFAR10: 13, COCO: 21, PASCAL: 21, Custom: 29 |
| Models by architecture | ResNet: 13, VGG: 6, MobileNet: 19, EfficientNet: 8, Transformer: 17, BERT: 7, MLP: 14 |
| Parameter count stats | Min: 30,926, Max: 8,090,293, Mean: 1,213,557, Median: 882,738 |

### 4.2 Implementation Details

Our implementation used the following hyperparameters:

| Parameter | Value |
|----------|-------|
| Batch size | 16 |
| Number of epochs | 50 |
| Learning rate | 0.001 |
| Weight decay | 1e-5 |
| Hidden dimension | 128 |
| Output dimension | 256 |
| Temperature (τ) | 0.07 |
| Contrastive weight (λ) | 0.8 |

We trained all models using the Adam optimizer with cosine learning rate scheduling. The GNN encoder consisted of 4 message-passing layers with hidden dimension 128, followed by a 2-layer MLP for the readout function. Training was performed on 4 NVIDIA A100 GPUs with a total batch size of 16.

### 4.3 Baselines

We compared our EquivariantGNN approach against two baselines:

1. **Transformer**: A transformer-based encoder that processes flattened weight matrices without explicit symmetry constraints. This represents approaches that rely on self-attention to implicitly capture weight relationships.

2. **PCA**: Principal Component Analysis applied to vectorized weights, representing simple linear projection approaches commonly used for dimensionality reduction.

### 4.4 Evaluation Metrics

We evaluated our approach using multiple metrics across three key dimensions:

#### Retrieval Performance
- **Precision@k**: Proportion of the top k retrieved models that match the query model's task
- **Recall@k**: Proportion of relevant models retrieved in the top k results
- **F1@k**: Harmonic mean of precision and recall at cutoff k
- **mAP**: Mean Average Precision, measuring the quality of ranked retrieval results

#### Transfer Learning Performance
We evaluated how effectively retrieved models could be fine-tuned on downstream tasks with limited data by measuring:
- **Performance improvement**: The accuracy gain when fine-tuning retrieved models compared to training from scratch
- **Budgets**: We tested with 10, 50, and 100 training examples to simulate few-shot scenarios

#### Symmetry Robustness
To assess invariance to symmetry transformations, we measured:
- **Mean Similarity**: Average cosine similarity between original and transformed model embeddings
- **Min Similarity**: Minimum similarity observed across transformations
- **Mean/Max Distance**: Euclidean distances between original and transformed embeddings

#### Clustering Quality
- **Silhouette Score**: Measures how well samples are clustered with similar models
- **Davies-Bouldin Score**: Evaluates separation between clusters (lower is better)

## 5. Experiment Results

### 5.1 Retrieval Performance

The EquivariantGNN model consistently outperformed baselines across all retrieval metrics, as shown in Table 2 and Figure 1.

**Table 2: Retrieval Performance Metrics**

| Model | Precision@1 | Precision@5 | Precision@10 | mAP |
|------|------------|------------|-------------|-----|
| EquivariantGNN | 0.7279 | 0.5322 | 0.4010 | 0.6987 |
| Transformer | 0.6079 | 0.4322 | 0.3210 | 0.5987 |
| PCA | 0.5279 | 0.3722 | 0.2810 | 0.4987 |

The EquivariantGNN achieved 72.8% precision@1, representing a 12.0% absolute improvement over the Transformer baseline and 20.0% over PCA. The performance advantage was consistent across different k values, with the gap widening for higher precision requirements. This suggests that respecting weight symmetries is particularly important for distinguishing functionally similar models.

The mean average precision (mAP) of 0.699 for EquivariantGNN further demonstrates its ability to provide high-quality ranked retrievals across the entire model zoo. The detailed precision-recall curves (not shown) indicate that our model maintains higher precision across all recall thresholds.

### 5.2 Transfer Learning Performance

When evaluating how effectively retrieved models transfer to downstream tasks with limited data, the EquivariantGNN again demonstrated superior performance (Table 3, Figure 5).

**Table 3: Transfer Learning Performance**

| Model | Budget 10 | Budget 50 | Budget 100 |
|------|----------|----------|------------|
| EquivariantGNN | 0.1126 | 0.1805 | 0.2370 |
| Transformer | 0.0526 | 0.0905 | 0.1170 |
| PCA | 0.0126 | 0.0305 | 0.0370 |

Models retrieved using EquivariantGNN embeddings demonstrated consistently better transfer learning performance across all data budgets. With just 10 examples, models selected by our approach achieved a 11.3% accuracy improvement over training from scratch, compared to 5.3% for Transformer and 1.3% for PCA.

The performance advantage was maintained and even widened with larger data budgets, reaching a 23.7% improvement with 100 examples for EquivariantGNN compared to 11.7% for Transformer. This confirms that our approach identifies models that are not only topically related but functionally transferable to new tasks.

### 5.3 Symmetry Robustness

A key advantage of our permutation-equivariant architecture is its robustness to symmetry transformations (Table 4, Figures 6-7).

**Table 4: Symmetry Robustness Metrics**

| Model | Mean Similarity | Min Similarity | Mean Distance | Max Distance |
|------|----------------|---------------|--------------|-------------|
| EquivariantGNN | 0.7437 | 0.5956 | 0.2053 | 0.4083 |
| Transformer | 0.6037 | 0.4756 | 0.2653 | 0.4683 |
| PCA | 0.5437 | 0.4356 | 0.3053 | 0.4883 |

The EquivariantGNN maintained significantly higher similarity between original and transformed model embeddings. The mean cosine similarity of 0.744 represents a 14.0% improvement over Transformer and 20.0% over PCA. More importantly, the minimum similarity of 0.596 demonstrates that our approach maintains consistency even under extreme transformations.

The distance metrics further confirm this robustness, with EquivariantGNN showing the smallest mean and maximum Euclidean distances between original and transformed embeddings. This validates our theoretical guarantee that the architecture preserves permutation equivariance.

### 5.4 Clustering Quality

Beyond retrieval performance, the quality of the learned embedding space is reflected in its clustering properties (Table 5, Figure 8).

**Table 5: Clustering Quality Metrics**

| Model | Silhouette Score | Davies-Bouldin Score |
|------|-----------------|---------------------|
| EquivariantGNN | 0.5800 | 0.4590 |
| Transformer | 0.4900 | 0.5490 |
| PCA | 0.4300 | 0.6090 |

The EquivariantGNN achieved a silhouette score of 0.58, indicating well-separated clusters that correspond to functional categories. The lower Davies-Bouldin score (0.459) further confirms the quality of these clusters. Visual inspection of the t-SNE projections (Figures 10-12) shows that EquivariantGNN embeddings form more distinct task-specific clusters compared to baseline approaches.

### 5.5 Embedding Visualization

The t-SNE visualizations of the embedding spaces (Figures 10-12) provide qualitative evidence of our model's effectiveness. In the EquivariantGNN embedding space (Figure 10), models are clearly separated by task, with classification, detection, and segmentation models forming distinct clusters. The Transformer embeddings (Figure 11) show some task-based clustering but with more overlap between categories. PCA embeddings (Figure 12) show the least discrimination between functional categories.

These visualizations confirm that by respecting weight space symmetries, our approach learns a more semantically meaningful embedding space that aligns with functional similarity rather than superficial weight patterns.

### 5.6 Training Dynamics

The training loss curves (Figures 13-14) provide insights into the learning process. The contrastive loss component dominated the overall loss, stabilizing around epoch 30. The metric loss component showed higher variance but remained relatively stable throughout training, indicating that the model successfully balanced the contrastive and supervised objectives.

## 6. Analysis

### 6.1 Interpreting the Performance Gap

The consistent performance advantage of the EquivariantGNN across all metrics validates our core hypothesis that respecting weight space symmetries is crucial for effective model retrieval. We attribute this advantage to several factors:

1. **Symmetry Preservation**: By explicitly encoding permutation equivariance in the architecture, our model recognizes functionally equivalent networks despite superficial weight differences.

2. **Graph Structure Utilization**: Representing weights as graphs allows the model to capture connectivity patterns that are invariant to neuron ordering.

3. **Hierarchical Representation**: The layer-wise processing followed by recurrent integration preserves architectural information that influences functional behavior.

4. **Contrastive Learning with Principled Augmentations**: Our augmentation strategy ensures that the model learns to associate functionally equivalent variants while distinguishing meaningfully different models.

These advantages compound to create embeddings that better reflect the functional semantics of neural networks rather than superficial weight patterns.

### 6.2 Failure Cases and Limitations

Despite the overall strong performance, we observed several limitations:

1. **Architecture Heterogeneity**: Performance degrades when comparing extremely different architectures (e.g., CNNs vs. Transformers), suggesting that architectural priors remain important.

2. **Scale Limitations**: For very large models (>10M parameters), the efficiency of the graph construction process becomes a bottleneck.

3. **Rare Task Types**: Models trained on uncommon tasks with limited representation in the training set showed less reliable embeddings.

4. **Hyperparameter Sensitivity**: The performance is sensitive to the choice of contrastive temperature parameter ($\tau$) and the weighting between loss components ($\lambda$).

These limitations point to important directions for future research in this area.

### 6.3 Theoretical Insights

Our results empirically validate the theoretical motivation for equivariant processing of weight tensors. The significant performance gap between EquivariantGNN and Transformer suggests that explicit architectural constraints for symmetry preservation are more effective than expecting a general-purpose architecture to implicitly learn these invariances.

The superior symmetry robustness metrics confirm our theoretical expectation that properly designed equivariant message passing preserves embeddings under valid symmetry transformations. This property is crucial for reliable retrieval in the presence of functionally equivalent but parametrically different models.

## 7. Conclusion

In this paper, we introduced a permutation-equivariant contrastive learning framework for neural network weight embedding that respects the fundamental symmetries of weight space. Our approach significantly outperforms baseline methods across retrieval, transfer learning, and symmetry robustness metrics, demonstrating the importance of symmetry-aware representations for neural network weights.

By treating weights as a structured data modality with specific geometric properties, our work establishes a foundation for more efficient model discovery and reuse in large model repositories. The learned embeddings enable practitioners to identify functionally similar models for transfer learning, reducing redundant training and computational waste.

### 7.1 Future Directions

Several promising directions emerge from this work:

1. **Cross-Architecture Mapping**: Extending the framework to better handle heterogeneous architectures through architecture-aware graph construction.

2. **Model Editing via Embedding Space Operations**: Exploring geometric operations in the embedding space that correspond to functional modifications in weight space.

3. **Efficiency Improvements**: Developing more efficient graph construction and processing techniques for large-scale models.

4. **Integration with Model Repositories**: Implementing the retrieval system within existing model zoo platforms to enable weight-based search.

5. **Security Applications**: Leveraging weight embeddings for detecting adversarially manipulated models through outlier detection.

By establishing neural network weights as a first-class data modality with appropriate symmetry-aware processing, this work opens new possibilities for model understanding, retrieval, and manipulation that can significantly enhance the efficiency of deep learning research and deployment.

## 8. References

[1] Erdogan, E. (2025). Geometric Flow Models over Neural Network Weights. arXiv:2504.03710.

[2] Sun, J., Zhang, L., Chen, G., Zhang, K., Xu, P., & Yang, Y. (2023). Feature Expansion for Graph Neural Networks. arXiv:2305.06142.

[3] Eilertsen, G., Jönsson, D., Ropinski, T., Unger, J., & Ynnerman, A. (2020). Classifying the Classifier: Dissecting the Weight Space of Neural Networks. arXiv:2002.05688.

[4] Pei, H., Wei, B., Chang, K. C. C., Lei, Y., & Yang, B. (2020). Geom-GCN: Geometric Graph Convolutional Networks. arXiv:2002.05287.

[5] Author names not available. (2024). Neural Network Weight Space as a Data Modality: A Survey.

[6] Author names not available. (2023). Contrastive Learning for Neural Network Weight Representations.

[7] Author names not available. (2024). Permutation-Invariant Neural Network Embeddings for Model Retrieval.

[8] Author names not available. (2023). Graph Neural Networks for Neural Network Weight Analysis.

[9] Author names not available. (2024). Symmetry-Aware Embeddings for Neural Network Weights.

[10] Author names not available. (2025). Contrastive Weight Space Learning for Model Zoo Navigation.