# HyTECL: Hyperbolic Temporal Contrastive Learning for Dynamic Graphs

## 1. Introduction

Temporal graphs, which represent entities and their relationships as they evolve over time, are ubiquitous in real-world applications ranging from social networks and financial systems to biological interactions. The dynamic nature of these graphs presents unique challenges for representation learning, as models must capture both the structural properties of the graph at each timestamp and the temporal patterns in how these structures evolve.

A significant limitation of existing temporal graph learning approaches is their reliance on Euclidean geometry, which is ill-suited for representing the latent hierarchical structures often present in real-world graphs. Many real-world networks exhibit tree-like properties, power-law degree distributions, and hierarchical organization that create high distortion when embedded in Euclidean space. For instance, social networks often display hierarchical communities, financial transaction networks contain hierarchical patterns of fund flows, and biological networks show nested functional relationships.

Hyperbolic geometry, with its exponentially expanding space, offers a more natural representation for such hierarchical structures. The negative curvature of hyperbolic space allows trees to be embedded with minimal distortion, making it particularly well-suited for representing graphs with latent hierarchies. Recent work has demonstrated the effectiveness of hyperbolic neural networks for static graphs, but their integration with temporal learning mechanisms remains underexplored.

Concurrently, contrastive learning has emerged as a powerful paradigm for self-supervised representation learning, enabling models to learn meaningful features by contrasting positive and negative examples. In the context of graph learning, contrastive approaches have shown remarkable success by creating different views of the same graph through various augmentation techniques and maximizing agreement between representations of the same nodes across these views.

This research proposes HyTECL (Hyperbolic Temporal Contrastive Learning), a novel framework that synergistically combines hyperbolic graph neural networks with contrastive temporal learning to address the challenges of representation learning on dynamic graphs. HyTECL operates in hyperbolic space to better capture hierarchical structures while employing contrastive learning to model temporal dynamics effectively. By embedding nodes in hyperbolic space and learning from temporally correlated views, our approach aims to produce representations that preserve both the structural and temporal properties of dynamic graphs.

The objectives of this research are:
1. Develop a hyperbolic graph neural network architecture capable of processing temporal graph data
2. Design a contrastive learning framework specifically tailored for hyperbolic space and temporal graph dynamics
3. Incorporate a temporal memory mechanism to capture long-range dependencies in evolving graphs
4. Evaluate the proposed approach on challenging temporal graph learning tasks, including dynamic knowledge graph completion and financial fraud detection

The significance of this research lies in its potential to advance the state-of-the-art in temporal graph representation learning by addressing fundamental limitations of existing approaches. By better modeling the underlying hierarchical structures of real-world networks in a geometry that naturally accommodates them, while simultaneously capturing temporal dynamics through contrastive learning, HyTECL can potentially unlock improved performance on a wide range of temporal graph learning tasks. Moreover, this work contributes to the broader understanding of how non-Euclidean geometries can be leveraged for machine learning on dynamic, structured data.

## 2. Methodology

### 2.1 Preliminaries

#### 2.1.1 Dynamic Graphs

We define a dynamic graph as a sequence of graph snapshots $\mathcal{G} = \{G^{(1)}, G^{(2)}, ..., G^{(T)}\}$, where each $G^{(t)} = (V^{(t)}, E^{(t)}, X^{(t)})$ represents the graph at timestamp $t$. Here, $V^{(t)}$ is the set of nodes, $E^{(t)}$ is the set of edges, and $X^{(t)} \in \mathbb{R}^{|V^{(t)}| \times d}$ contains node features at time $t$, where $d$ is the feature dimension. For simplicity, we assume the set of nodes remains constant across timestamps (i.e., $V^{(t)} = V$ for all $t$), though our method can be extended to handle node additions and deletions.

#### 2.1.2 Hyperbolic Geometry

We work with the Poincaré ball model of hyperbolic space, denoted as $\mathbb{B}^n_c = \{x \in \mathbb{R}^n : \|x\|^2 < 1/c\}$, where $c > 0$ is the curvature parameter. The Poincaré ball is equipped with the Riemannian metric tensor:

$$g_x = \lambda_x^2 g_E$$

where $\lambda_x = \frac{2}{1 - c\|x\|^2}$ is the conformal factor and $g_E$ is the Euclidean metric tensor. This induces a non-Euclidean distance function given by:

$$d_{\mathbb{B}}(x, y) = \frac{2}{\sqrt{c}} \tanh^{-1}(\sqrt{c}\|-x \oplus_c y\|)$$

where $\oplus_c$ is the Möbius addition:

$$x \oplus_c y = \frac{(1 + 2c\langle x, y \rangle + c\|y\|^2)x + (1 - c\|x\|^2)y}{1 + 2c\langle x, y \rangle + c^2\|x\|^2\|y\|^2}$$

To operate in hyperbolic space, we use the exponential map $\exp_x^c: T_x\mathbb{B}_c^n \rightarrow \mathbb{B}_c^n$ and logarithmic map $\log_x^c: \mathbb{B}_c^n \rightarrow T_x\mathbb{B}_c^n$, which allow us to move between hyperbolic space and its tangent space at point $x$:

$$\exp_x^c(v) = x \oplus_c \left(\tanh\left(\sqrt{c}\frac{\lambda_x\|v\|}{2}\right)\frac{v}{\sqrt{c}\|v\|}\right)$$

$$\log_x^c(y) = \frac{2}{\sqrt{c}\lambda_x}\tanh^{-1}(\sqrt{c}\|-x \oplus_c y\|)\frac{-x \oplus_c y}{\|-x \oplus_c y\|}$$

### 2.2 HyTECL Architecture

The proposed HyTECL framework consists of four main components:
1. Hyperbolic Feature Transformation
2. Hyperbolic Graph Convolutional Layer
3. Temporal Memory Module
4. Contrastive Learning Framework

We describe each component in detail below.

#### 2.2.1 Hyperbolic Feature Transformation

For each timestamp $t$, we first project the Euclidean node features $X^{(t)}$ into hyperbolic space. We define a learnable reference point $o \in \mathbb{B}^n_c$ (typically set to the origin) and use the exponential map to transform the features:

$$H^{(t)}_0 = \exp_o^c(W X^{(t)} + b)$$

where $W \in \mathbb{R}^{n \times d}$ and $b \in \mathbb{R}^n$ are learnable parameters, and $H^{(t)}_0 \in \mathbb{B}^n_c$ represents the initial hyperbolic embeddings at time $t$.

#### 2.2.2 Hyperbolic Graph Convolutional Layer

We extend graph convolutional networks to hyperbolic space by implementing the message passing operation using Möbius operations. For each node $i$ at time $t$ and layer $l$, we define:

$$H^{(t)}_{l+1,i} = \exp_{H^{(t)}_{l,i}}^c\left(\sum_{j \in \mathcal{N}_i^{(t)}} \frac{\alpha_{ij}^{(t)}}{\sqrt{|\mathcal{N}_i^{(t)}||\mathcal{N}_j^{(t)}|}} W_l \log_{H^{(t)}_{l,i}}^c(H^{(t)}_{l,j})\right)$$

where $\mathcal{N}_i^{(t)}$ is the set of neighbors of node $i$ at time $t$, $\alpha_{ij}^{(t)}$ is an attention coefficient, and $W_l$ is a learnable weight matrix that operates in the tangent space. The attention coefficient is computed as:

$$\alpha_{ij}^{(t)} = \frac{\exp(f(H^{(t)}_{l,i}, H^{(t)}_{l,j}))}{\sum_{k \in \mathcal{N}_i^{(t)}} \exp(f(H^{(t)}_{l,i}, H^{(t)}_{l,k}))}$$

where $f(x, y) = -d_{\mathbb{B}}(x, y)$ is a similarity function based on hyperbolic distance.

After $L$ layers of hyperbolic graph convolution, we obtain the final node representations $Z^{(t)} = H^{(t)}_L \in \mathbb{B}^n_c$ for timestamp $t$.

#### 2.2.3 Temporal Memory Module

To capture long-range temporal dependencies, we implement a memory module that aggregates information from past embeddings. For each node $i$ at time $t$, we maintain a memory state $M_i^{(t)} \in \mathbb{B}^n_c$ that is updated as:

$$M_i^{(t)} = \gamma_i^{(t)} \otimes_c M_i^{(t-1)} \oplus_c ((1 - \gamma_i^{(t)}) \otimes_c Z_i^{(t)})$$

where $\otimes_c$ is the Möbius scalar multiplication, $\oplus_c$ is the Möbius addition, and $\gamma_i^{(t)} \in [0, 1]$ is a gating parameter that controls how much past information is retained. The gating parameter is computed as:

$$\gamma_i^{(t)} = \sigma\left(W_\gamma \log_o^c(Z_i^{(t)}) + U_\gamma \log_o^c(M_i^{(t-1)}) + b_\gamma\right)$$

where $W_\gamma$, $U_\gamma$, and $b_\gamma$ are learnable parameters, and $\sigma$ is the sigmoid function.

The final representation for node $i$ at time $t$ is a combination of the current embedding and the memory state:

$$\hat{Z}_i^{(t)} = Z_i^{(t)} \oplus_c (r_i^{(t)} \otimes_c M_i^{(t)})$$

where $r_i^{(t)} \in [0, 1]$ is another learnable gating parameter.

#### 2.2.4 Contrastive Learning Framework

We employ contrastive learning to model temporal dynamics by creating multiple views of the graph at different timestamps. For each timestamp $t$, we generate two augmented views of the graph, $\tilde{G}^{(t,1)}$ and $\tilde{G}^{(t,2)}$, using the following augmentation strategies:

1. **Time-aware edge masking**: Randomly remove edges with probability $p_e$, with higher probability for edges that appear less frequently across timestamps.

2. **Temporal subgraph sampling**: Sample a connected subgraph centered around randomly selected seed nodes, with the sampling probability weighted by node centrality.

3. **Feature perturbation**: Add Gaussian noise to node features, with the noise magnitude proportional to the feature variance over time.

Let $\tilde{Z}_i^{(t,1)}$ and $\tilde{Z}_i^{(t,2)}$ be the embeddings of node $i$ in the two augmented views at time $t$. We also consider embeddings from adjacent timestamps $\tilde{Z}_i^{(t-1,1)}$ and $\tilde{Z}_i^{(t+1,1)}$. The contrastive loss for node $i$ is defined as:

$$\mathcal{L}_i^{cont} = -\log \frac{\exp(-d_{\mathbb{B}}(\tilde{Z}_i^{(t,1)}, \tilde{Z}_i^{(t,2)}) / \tau)}{\sum_{j=1}^{|V|} \exp(-d_{\mathbb{B}}(\tilde{Z}_i^{(t,1)}, \tilde{Z}_j^{(t,2)}) / \tau) + \sum_{s \in \{t-1, t+1\}} \exp(-d_{\mathbb{B}}(\tilde{Z}_i^{(t,1)}, \tilde{Z}_i^{(s,1)}) / \tau)}$$

where $\tau$ is a temperature parameter. This loss encourages embeddings of the same node in different augmented views at the same timestamp to be close, while pushing apart embeddings of different nodes or the same node at different timestamps.

Additionally, we introduce a temporal consistency loss to ensure smooth transitions between consecutive timestamps:

$$\mathcal{L}^{temp} = \frac{1}{T-1} \sum_{t=2}^T \frac{1}{|V|} \sum_{i=1}^{|V|} d_{\mathbb{B}}(\hat{Z}_i^{(t)}, \hat{Z}_i^{(t-1)})$$

The overall loss function is a combination of the contrastive loss, temporal consistency loss, and a task-specific supervised loss $\mathcal{L}^{task}$ (e.g., link prediction or node classification):

$$\mathcal{L} = \mathcal{L}^{task} + \lambda_1 \frac{1}{|V|} \sum_{i=1}^{|V|} \mathcal{L}_i^{cont} + \lambda_2 \mathcal{L}^{temp}$$

where $\lambda_1$ and $\lambda_2$ are hyperparameters controlling the contribution of each loss term.

### 2.3 Downstream Tasks and Evaluation

We evaluate HyTECL on two main temporal graph learning tasks:

#### 2.3.1 Dynamic Knowledge Graph Completion

For temporal knowledge graph completion, we predict missing links at future timestamps. Given a knowledge graph represented as a sequence of triples $(s, r, o, t)$ where $s$ is the subject entity, $r$ is the relation, $o$ is the object entity, and $t$ is the timestamp, we aim to predict missing triples at time $T+1$ given the history up to time $T$.

We use the entity and relation embeddings from HyTECL to compute a score for each potential triple $(s, r, o, T+1)$:

$$score(s, r, o, T+1) = -d_{\mathbb{B}}(\hat{Z}_s^{(T)} \oplus_c f_r(\hat{Z}_o^{(T)}))$$

where $f_r$ is a relation-specific transformation in hyperbolic space.

The model is evaluated using standard metrics: Mean Reciprocal Rank (MRR), Hits@1, Hits@3, and Hits@10.

#### 2.3.2 Financial Fraud Detection

For fraud detection in financial transaction networks, we formulate it as a node classification task where each node (account) is labeled as fraudulent or legitimate. The temporal graph captures transactions between accounts over time.

Using the node embeddings from HyTECL, we apply a hyperbolic multi-layer perceptron (MLP) for classification:

$$p(y_i = 1) = \sigma(W_{out} \log_o^c(\hat{Z}_i^{(T)}) + b_{out})$$

where $W_{out}$ and $b_{out}$ are learnable parameters, and $\sigma$ is the sigmoid function.

The model is evaluated using Accuracy, Precision, Recall, F1-score, and Area Under the ROC Curve (AUC).

### 2.4 Experimental Setup

#### 2.4.1 Datasets

We evaluate HyTECL on the following datasets:

1. **ICEWS18**: A temporal knowledge graph dataset containing political events between countries.
2. **GDELT**: A larger temporal knowledge graph dataset with global events.
3. **YELP**: A dynamic graph of Yelp user reviews and business interactions.
4. **ELLIPTIC**: A Bitcoin transaction network for fraud detection, with 203k transactions and 234k directed payment flows, including labeled fraudulent transactions.
5. **IEEE-CIS**: A financial transaction dataset for fraud detection.

#### 2.4.2 Baselines

We compare HyTECL against several state-of-the-art methods:

1. **Static Methods**: HGCN, HGNN (hyperbolic GNNs for static graphs)
2. **Temporal Euclidean Methods**: TGN, TGAT, DyRep, EvolveGCN (temporal GNNs in Euclidean space)
3. **Hyperbolic Temporal Methods**: HTGN, HGWaveNet, HVGNN (existing hyperbolic approaches for temporal graphs)
4. **Contrastive Methods**: HGCL (hyperbolic contrastive learning for static graphs), TGC (temporal graph contrastive learning in Euclidean space)

#### 2.4.3 Implementation Details

We implement HyTECL using PyTorch and the geoopt library for hyperbolic operations. The model is trained using the Riemannian Adam optimizer with the following hyperparameters:
- Embedding dimension: 64
- Number of GNN layers: 2
- Batch size: 256
- Learning rate: 0.001
- Curvature parameter $c$: 1.0 (learnable)
- Temperature $\tau$: 0.1
- Loss weights $\lambda_1 = 0.1$, $\lambda_2 = 0.01$
- Edge masking probability $p_e$: 0.15

We use early stopping with a patience of 10 epochs based on validation performance. The model is trained on NVIDIA V100 GPUs with 16GB memory.

## 3. Expected Outcomes & Impact

### 3.1 Expected Technical Contributions

This research is expected to make several significant technical contributions to the field of temporal graph learning:

1. **A novel hyperbolic temporal graph neural network architecture** that effectively captures both hierarchical structures and temporal dynamics in evolving graphs. By operating in hyperbolic space, HyTECL should provide more faithful representations of tree-like structures commonly found in real-world networks while maintaining the ability to model temporal patterns.

2. **A hyperbolic contrastive learning framework for temporal graphs** that leverages multiple augmented views to learn robust representations. This approach establishes principled ways to perform contrastive learning in hyperbolic space, addressing the unique challenges posed by non-Euclidean geometry.

3. **A temporal memory mechanism in hyperbolic space** that enables the model to capture long-range dependencies across time while preserving the hierarchical properties of the representations. This module should be particularly effective for graphs with evolving hierarchical structures.

4. **New insights into the geometry of temporal graph representations**, providing a better understanding of how hyperbolic space can be leveraged to model dynamic networks with latent hierarchies. These insights may guide future research in geometric deep learning for temporal data.

### 3.2 Expected Performance Improvements

Based on the properties of hyperbolic geometry and the design of HyTECL, we anticipate the following performance improvements over existing methods:

1. **Knowledge Graph Completion**: We expect HyTECL to achieve at least a 5-10% improvement in MRR and Hits@10 metrics compared to Euclidean temporal graph methods and a 3-5% improvement over existing hyperbolic temporal methods. This improvement should be more pronounced on datasets with strong hierarchical structures, such as ICEWS18.

2. **Fraud Detection**: For financial fraud detection tasks, we anticipate a 3-7% improvement in F1-score and AUC compared to state-of-the-art methods, particularly in detecting complex fraud patterns that involve hierarchical fund flows or organizational structures.

3. **Model Efficiency**: Despite operating in hyperbolic space, which can be computationally intensive, the focused representation power of hyperbolic geometry should allow HyTECL to achieve better performance with lower-dimensional embeddings compared to Euclidean methods, potentially reducing the overall parameter count by 30-50% for comparable performance.

4. **Robustness to Sparse Data**: Hyperbolic representations are expected to be more robust to data sparsity, a common challenge in temporal graphs. We anticipate HyTECL to maintain good performance even when trained on smaller subsets of the data (e.g., 50% of training data) compared to Euclidean baselines.

### 3.3 Broader Impact

The successful development of HyTECL could have several broader impacts across various domains:

1. **Financial Security**: Improved fraud detection models can help financial institutions identify suspicious activities more accurately, potentially saving billions of dollars annually and protecting consumers from financial crimes.

2. **Knowledge Discovery**: Better temporal knowledge graph completion can enhance information retrieval systems, recommendation engines, and question-answering systems, facilitating more efficient knowledge discovery in scientific research, healthcare, and business intelligence.

3. **Social Network Analysis**: More accurate modeling of evolving social networks can improve our understanding of information diffusion, community formation, and influence propagation, with applications in public health (e.g., tracking disease spread), marketing, and social science research.

4. **Computational Biology**: In biological networks, hierarchical structures are prevalent (e.g., protein interaction networks, metabolic pathways), and their temporal evolution is crucial for understanding disease progression and drug responses. HyTECL could enable more accurate modeling of these complex biological systems.

5. **Theoretical Advancement**: This research contributes to the broader understanding of geometric deep learning by exploring the intersection of hyperbolic geometry and temporal learning, potentially inspiring new approaches for other types of non-Euclidean data.

### 3.4 Limitations and Future Work

While HyTECL addresses several limitations of existing approaches, we acknowledge potential limitations that could be addressed in future work:

1. **Computational Complexity**: Operations in hyperbolic space are generally more computationally intensive than their Euclidean counterparts. Future work could focus on developing more efficient algorithms for hyperbolic operations.

2. **Mixed Curvature Spaces**: Some real-world graphs might be better represented in spaces with varying curvature rather than a single hyperbolic space. Exploring product spaces that combine regions of different curvature could be a promising direction.

3. **Incorporating External Events**: Many temporal graphs are influenced by external events not captured in the graph structure. Extending HyTECL to incorporate such external information could enhance its predictive power.

4. **Theoretical Understanding**: Developing a deeper theoretical understanding of how contrastive learning in hyperbolic space affects representation quality and generalization ability could guide further improvements in the approach.

By addressing these limitations in future work, the impact of HyTECL could be extended to even more challenging temporal graph learning problems and application domains.