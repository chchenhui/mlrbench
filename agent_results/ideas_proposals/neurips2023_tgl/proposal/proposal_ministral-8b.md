# HyTECL – Hyperbolic Temporal Contrastive Learning for Dynamic Graphs

## 1. Introduction

### Background
Graphs are ubiquitous in various domains, including social networks, natural language processing, computer vision, and financial networks. Traditional graph machine learning algorithms assume static networks, limiting their applicability to real-world dynamic graphs. Temporal graphs, which evolve over time, present unique challenges and opportunities. Existing methods struggle to capture both hierarchical structures and temporal dynamics effectively. Hyperbolic geometry offers a promising framework for modeling hierarchical data, but its integration with temporal learning remains underexplored. This research aims to bridge this gap by proposing HyTECL, a novel framework combining hyperbolic graph neural networks with contrastive temporal learning.

### Research Objectives
The primary objectives of this research are:
1. To develop a hyperbolic graph neural network architecture that captures hierarchical structures and temporal dynamics in dynamic graphs.
2. To integrate contrastive learning techniques to enhance the learning of temporal patterns and improve prediction accuracy.
3. To evaluate the proposed method on dynamic knowledge graph forecasting and fraud detection benchmarks to demonstrate its effectiveness and robustness.

### Significance
The proposed HyTECL framework addresses the limitations of current temporal graph learning methods by leveraging the strengths of hyperbolic geometry and contrastive learning. By capturing hierarchical structures and temporal evolutions, HyTECL has the potential to significantly improve performance in various applications, such as recommendation systems, event forecasting, and fraud detection. Additionally, the research contributes to the broader field of temporal graph learning by providing a novel approach that combines hyperbolic geometry with contrastive learning.

## 2. Methodology

### Research Design

#### 2.1. Hyperbolic Graph Convolutional Layer
The core of HyTECL is the Hyperbolic Graph Convolutional Layer (HGCL). This layer embeds node features and edge updates in hyperbolic space using exponential and logarithmic maps. The hyperbolic diffusion graph convolution and hyperbolic dilated causal convolution are employed to capture spatial and temporal dynamics effectively.

#### 2.2. Temporal Augmentation
To model dynamics, HyTECL generates two temporally shifted graph views via augmentations such as time-aware edge masking and subgraph sampling. These augmentations create variations in the graph structure at different time steps, enabling the model to learn temporal patterns.

#### 2.3. Contrastive Temporal Learning
A contrastive loss function is optimized to align positive pairs (same node across views) while pushing apart negatives (different nodes or distant timestamps). This approach enhances the learning of temporal patterns by encouraging the model to distinguish between similar and dissimilar nodes across different time steps.

#### 2.4. Temporal Memory Module
A temporal memory module aggregates past hyperbolic embeddings to capture long-range dependencies. This module is crucial for maintaining the temporal context and improving the model's ability to forecast future states and detect anomalies.

### Algorithmic Steps

1. **Input**: Temporal graph $G_t = (V, E_t)$, where $V$ is the set of nodes and $E_t$ is the set of edges at time $t$.
2. **Node Feature Embedding**: Embed node features in hyperbolic space using exponential and logarithmic maps.
3. **Edge Update**: Update edges based on hyperbolic diffusion graph convolution and hyperbolic dilated causal convolution.
4. **Temporal Augmentation**: Generate two temporally shifted graph views $G_{t-1}$ and $G_{t+1}$ using time-aware edge masking and subgraph sampling.
5. **Contrastive Learning**: Compute contrastive loss to align positive pairs and push apart negatives.
6. **Temporal Memory Aggregation**: Aggregate past hyperbolic embeddings using a temporal memory module.
7. **Output**: Hyperbolic embeddings at time $t$ and temporal predictions.

### Mathematical Formulations

#### 2.5. Hyperbolic Embedding
Given a node $v_i \in V$, its hyperbolic embedding is computed as:
\[ \mathbf{h}_i = \tanh(\mathbf{W}_h \mathbf{x}_i + \mathbf{b}_h) \]
where $\mathbf{x}_i$ is the node feature vector, $\mathbf{W}_h$ and $\mathbf{b}_h$ are learnable parameters.

#### 2.6. Hyperbolic Diffusion Graph Convolution
The hyperbolic diffusion graph convolution is defined as:
\[ \mathbf{h}_i^{(t+1)} = \sum_{j \in \mathcal{N}(i)} \frac{\exp(-\alpha \cdot \mathbf{d}(i, j))}{\sum_{k \in \mathcal{N}(i)} \exp(-\alpha \cdot \mathbf{d}(i, k))} \mathbf{h}_j^{(t)} \]
where $\mathcal{N}(i)$ is the neighborhood of node $i$, $\mathbf{d}(i, j)$ is the hyperbolic distance between nodes $i$ and $j$, and $\alpha$ is a hyperparameter.

#### 2.7. Contrastive Loss
The contrastive loss is defined as:
\[ \mathcal{L}_{\text{con}} = -\sum_{i=1}^{N} \log \frac{\exp(\mathbf{h}_i^{(t)} \cdot \mathbf{h}_i^{(t-1)})}{\sum_{j=1}^{N} \exp(\mathbf{h}_i^{(t)} \cdot \mathbf{h}_j^{(t-1)})} \]
where $N$ is the number of nodes, and $\mathbf{h}_i^{(t)}$ and $\mathbf{h}_i^{(t-1)}$ are the hyperbolic embeddings at times $t$ and $t-1$, respectively.

### Experimental Design

#### 2.8. Datasets
The proposed method will be evaluated on dynamic knowledge graph forecasting and fraud detection benchmarks. The datasets include:
- **Dynamic Knowledge Graphs**: Datasets such as Temporal Knowledge Graph (TKG) and Dynamic Knowledge Graph (DKG).
- **Fraud Detection**: Datasets such as ISBM Fraud Detection and Enron Email Dataset.

#### 2.9. Evaluation Metrics
The evaluation metrics will include:
- **Accuracy**: Measure the correctness of the predictions.
- **Precision, Recall, and F1 Score**: Evaluate the performance of the model in fraud detection tasks.
- **Hierarchy Preservation**: Measure the ability of the model to capture hierarchical structures in the data.

#### 2.10. Baseline Methods
The proposed method will be compared with state-of-the-art temporal graph learning methods, including:
- **Temporal Graph Networks (TGN)**: A baseline temporal graph network.
- **Hyperbolic Graph Neural Networks (HGNN)**: A baseline hyperbolic graph neural network.
- **Contrastive Learning for Graph Neural Networks (CL-GNN)**: A baseline contrastive learning method for graph neural networks.

## 3. Expected Outcomes & Impact

### Expected Outcomes
1. **Improved Performance**: The proposed HyTECL framework is expected to demonstrate significant improvements in accuracy, robustness, and hierarchy preservation on dynamic knowledge graph forecasting and fraud detection benchmarks.
2. **Scalability**: The method will be evaluated for its scalability to large, real-world datasets, ensuring that it can handle the complexity of real-world temporal graphs.
3. **Novel Contributions**: The research will contribute novel techniques for integrating hyperbolic geometry with contrastive learning in temporal graph learning.

### Impact
The proposed HyTECL framework has the potential to revolutionize the field of temporal graph learning by providing a scalable and effective approach for modeling dynamic graphs. The method's ability to capture hierarchical structures and temporal dynamics can lead to significant improvements in various applications, such as recommendation systems, event forecasting, and fraud detection. Additionally, the research will contribute to the broader field of graph machine learning by providing a novel approach that combines hyperbolic geometry with contrastive learning.

By addressing the challenges of temporal graph learning and integrating hyperbolic geometry with contrastive learning, the proposed HyTECL framework has the potential to drive significant advancements in the field and enable more accurate and robust modeling of dynamic graphs.