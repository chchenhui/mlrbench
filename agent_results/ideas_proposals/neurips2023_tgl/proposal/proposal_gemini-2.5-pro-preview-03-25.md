Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:**

**HyTECL: Hyperbolic Temporal Contrastive Learning for Hierarchy-Aware Dynamic Graph Representation**

**2. Introduction**

**2.1 Background**
Graphs provide a powerful abstraction for modeling complex relational systems across diverse domains, including social networks, biological interactions, financial transactions, and knowledge bases [6, 8]. While Graph Neural Networks (GNNs) have achieved remarkable success on static graph tasks like node classification and link prediction [5, 9], many real-world systems are inherently dynamic, evolving continuously over time. Static graph models fail to capture these temporal dynamics, limiting their applicability and predictive power in real-world scenarios such as event forecasting, fraud detection, and recommendation systems [6].

Temporal Graph Networks (TGNs) have emerged to address this limitation by explicitly modeling the evolving structure and features of graphs [6, 8]. These models typically combine graph neural network architectures with sequence models (like RNNs or Transformers) or employ time-encoding techniques to capture temporal dependencies. However, a significant challenge remains: many real-world dynamic networks exhibit latent hierarchical or tree-like structures [3, 5]. For example, social communities often merge or split, organizations undergo structural changes, and knowledge graphs evolve with new concepts branching from existing ones. Standard TGNs, operating primarily in Euclidean space, struggle to effectively represent and capture the dynamics of such hierarchical structures due to Euclidean geometry's limitations in embedding tree-like structures with low distortion [5, 9].

Hyperbolic geometry offers a compelling alternative for representing hierarchical data. Hyperbolic spaces naturally accommodate tree-like structures with significantly lower embedding distortion compared to Euclidean spaces of the same dimension [5, 9]. Research on static graphs has demonstrated the effectiveness of Hyperbolic Graph Neural Networks (HGNNs) in capturing latent hierarchies, leading to improved performance on tasks like node classification and link prediction [5, 9, 10]. Recently, researchers have begun exploring the integration of hyperbolic geometry with temporal graphs [1, 3, 4], showing promise in capturing both hierarchical structure and temporal evolution. However, these approaches often rely on direct adaptation of sequence models or specific temporal convolution mechanisms within hyperbolic space, potentially overlooking more robust ways to learn temporal representations, especially concerning long-range dependencies and subtle evolutionary patterns.

Contrastive learning has emerged as a powerful self-supervised learning paradigm for representation learning, including on graphs [7]. By learning representations that pull semantically similar nodes (positive pairs) closer while pushing dissimilar nodes (negative pairs) apart, contrastive methods can learn robust embeddings without explicit labels. Adapting contrastive learning for *temporal* graphs offers a promising way to explicitly model temporal consistency and evolution by defining positive pairs based on temporal proximity or node identity across time [2]. While contrastive learning has been explored for static hyperbolic graphs [2], its potential for learning *dynamic*, hierarchy-aware representations in hyperbolic space remains largely untapped.

**2.2 Research Objectives**
This research proposes **HyTECL (Hyperbolic Temporal Contrastive Learning)**, a novel framework designed to learn effective node representations on dynamic graphs by synergistically combining the hierarchical modeling capabilities of hyperbolic geometry with the temporal representation power of contrastive learning. The primary objectives are:

1.  **Develop a Novel Hyperbolic Temporal Graph Learning Framework (HyTECL):** Design and implement an end-to-end framework that integrates hyperbolic graph convolutions, a temporal contrastive learning objective specifically tailored for hyperbolic space, and a temporal memory mechanism operating within hyperbolic geometry.
2.  **Design Hyperbolic Temporal Contrastive Mechanisms:** Formulate effective temporal data augmentation strategies (e.g., time-aware subgraph sampling, temporal edge masking) suitable for generating meaningful positive and negative pairs for contrastive learning in dynamic hyperbolic graph settings. Develop a contrastive loss function based on hyperbolic distance metrics.
3.  **Integrate Hyperbolic Temporal Memory:** Incorporate and adapt a temporal memory module (e.g., hyperbolic GRU or attention) to capture long-range dependencies within the learned dynamic hyperbolic representations.
4.  **Comprehensive Evaluation:** Evaluate HyTECL's performance on benchmark dynamic graph datasets for tasks like temporal link prediction (including dynamic knowledge graph forecasting) and dynamic node classification (e.g., fraud detection).
5.  **Analyze Hierarchy Preservation and Robustness:** Investigate HyTECL's ability to preserve latent hierarchical structures over time compared to Euclidean counterparts and assess its robustness to noisy or missing temporal information.

**2.3 Significance**
This research addresses critical limitations in existing temporal graph learning methods. By explicitly modeling latent hierarchies using hyperbolic geometry and leveraging temporal contrastive learning for robust dynamic representation learning, HyTECL is expected to:

*   **Improve Performance:** Achieve state-of-the-art performance on dynamic graph tasks where underlying hierarchical structures are prevalent and evolve, such as dynamic knowledge graph forecasting and evolving fraud network analysis.
*   **Enhance Representation Quality:** Produce node embeddings that better reflect both the hierarchical organization and the temporal dynamics of the graph, leading to more interpretable and meaningful representations.
*   **Advance Methodological Understanding:** Provide insights into the effective integration of non-Euclidean geometry (hyperbolic space) with self-supervised learning paradigms (contrastive learning) for complex temporal data.
*   **Broaden Applicability:** Offer a powerful tool applicable to various domains involving evolving hierarchical systems, including computational biology (phylogenetic trees, protein interaction networks), finance (transaction networks), and social science (organizational structures, community evolution).

This work directly aligns with the Temporal Graph Learning Workshop's focus on "Temporal Graph Modelling & Representation Learning," "Temporal Graph Theory" (by exploring geometric representation spaces), and "Temporal Graph Applications" (targeting forecasting and anomaly detection). It tackles key challenges identified in the literature, such as the integration of hyperbolic geometry with temporal dynamics and the adaptation of contrastive learning frameworks for non-Euclidean spaces [1, 2, 3].

**3. Methodology**

**3.1 Overview**
The proposed HyTECL framework learns dynamic node representations by processing a sequence of graph snapshots. At each timestamp $t$, HyTECL performs the following key operations: (i) it encodes the current graph structure and node features into hyperbolic space using a hyperbolic graph convolutional layer; (ii) it employs a temporal contrastive learning objective on augmented temporal views of the graph to capture dynamics; (iii) it utilizes a hyperbolic temporal memory module to integrate past information and capture long-range dependencies.

**3.2 Hyperbolic Space Preliminaries**
We will primarily utilize the Poincaré ball model of hyperbolic space $(\mathbb{B}^d, g_{\mathbb{B}})$, defined as $\mathbb{B}^d = \{ \mathbf{x} \in \mathbb{R}^d : \| \mathbf{x} \| < 1 \}$, where $d$ is the dimension and $\|\cdot\|$ denotes the Euclidean norm. The associated distance metric is given by:
$$ d_{\mathbb{B}}(\mathbf{x}, \mathbf{y}) = \text{arccosh} \left( 1 + 2 \frac{\| \mathbf{x} - \mathbf{y} \|^2}{(1 - \| \mathbf{x} \|^2)(1 - \| \mathbf{y} \|^2)} \right) $$
Key operations are defined via Möbius gyrovector space theory:
*   **Möbius Addition ($\oplus$):** $\mathbf{x} \oplus \mathbf{y} = \frac{(1 + 2 \langle \mathbf{x}, \mathbf{y} \rangle + \| \mathbf{y} \|^2) \mathbf{x} + (1 - \| \mathbf{x} \|^2) \mathbf{y}}{1 + 2 \langle \mathbf{x}, \mathbf{y} \rangle + \| \mathbf{x} \|^2 \| \mathbf{y} \|^2}$
*   **Exponential Map ($\exp_{\mathbf{o}}^c(\mathbf{v})$):** Maps a tangent vector $\mathbf{v} \in T_{\mathbf{o}} \mathbb{B}^d \cong \mathbb{R}^d$ at the origin $\mathbf{o}$ to $\mathbb{B}^d$:
    $$ \exp_{\mathbf{o}}^c(\mathbf{v}) = \tanh(\sqrt{c} \| \mathbf{v} \| / 2) \frac{\mathbf{v}}{\sqrt{c} \| \mathbf{v} \|} $$
    where $c \geq 0$ is the curvature (we typically set $c = 1$).
*   **Logarithmic Map ($\log_{\mathbf{o}}^c(\mathbf{y})$):** Maps a point $\mathbf{y} \in \mathbb{B}^d$ back to the tangent space at the origin:
    $$ \log_{\mathbf{o}}^c(\mathbf{y}) = \frac{2}{\sqrt{c}} \text{arctanh}(\sqrt{c} \| \mathbf{y} \|) \frac{\mathbf{y}}{\| \mathbf{y} \|} $$
These operations allow us to perform calculations analogous to Euclidean vector addition and transformations while respecting the hyperbolic geometry.

**3.3 HyTECL Framework Details**

Let $G_t = (V_t, E_t, X_t)$ represent the graph snapshot at time $t$, where $V_t$ is the set of nodes, $E_t$ the set of edges, and $X_t$ the node features.

**3.3.1 Hyperbolic Graph Convolution Layer**
Inspired by HGCN [9] and HGNN [5], we define a hyperbolic graph convolution operation at time $t$. For a node $v$, its representation $\mathbf{h}_v^{(l+1)}(t) \in \mathbb{B}^d$ at layer $l+1$ is computed based on its representation $\mathbf{h}_v^{(l)}(t)$ and its neighbors $\mathcal{N}_v(t)$ at layer $l$:

1.  **Feature Transformation:** Map Euclidean features $X_v(t)$ or previous layer's hyperbolic embeddings $\mathbf{h}_v^{(l)}(t)$ to the tangent space at the origin:
    *   If $l=0$, $\mathbf{m}_v^{(0)}(t) = \log_{\mathbf{o}}^c (\phi_f(X_v(t)))$, where $\phi_f$ is a function mapping features to $\mathbb{B}^d$ (e.g., linear layer + projection).
    *   If $l>0$, $\mathbf{m}_v^{(l)}(t) = \log_{\mathbf{o}}^c (\mathbf{h}_v^{(l)}(t))$.
2.  **Tangent Space Aggregation:** Aggregate transformed neighbor messages in the tangent space:
    $$ \mathbf{a}_v^{(l)}(t) = \text{AGG} \left( \{ W^{(l)} \mathbf{m}_u^{(l)}(t) \mid u \in \mathcal{N}_v(t) \cup \{v\} \} \right) $$
    where $W^{(l)}$ is a learnable weight matrix (potentially identity or graph-attention based weights) and AGG is an aggregation function (e.g., mean, sum).
3.  **Map back to Hyperbolic Space:** Project the aggregated tangent vector back to the Poincaré ball:
    $$ \mathbf{\tilde{h}}_v^{(l+1)}(t) = \exp_{\mathbf{o}}^c (\sigma ( \mathbf{a}_v^{(l)}(t) ) ) $$
    where $\sigma$ is a non-linear activation function suitable for hyperbolic space (e.g., identity or element-wise tanh applied in tangent space before mapping).
4.  **Möbius Combination (Optional Refinement):** Combine with the node's previous layer representation using Möbius addition for residual-like connections if needed:
    $$ \mathbf{h}_v^{(l+1)}(t) = \mathbf{h}_v^{(l)}(t) \oplus \mathbf{\tilde{h}}_v^{(l+1)}(t) $$ (requires careful implementation to stay within the ball).

**3.3.2 Temporal Contrastive Learning Module**
To capture temporal dynamics, we employ a contrastive learning strategy operating on the hyperbolic embeddings $\mathbf{h}_v(t)$ (output of the hyperbolic GCN layers).

1.  **Temporal View Generation:** For a given graph snapshot $G_t$, we generate two augmented temporal views, $G_t'$ and $G_t''$. Augmentations focus on temporal aspects:
    *   **Time-Aware Subgraph Sampling:** Sample subgraphs centered around nodes, potentially prioritizing recent interactions or structurally important nodes within a temporal window.
    *   **Temporal Edge Masking/Perturbation:** Randomly mask or perturb edges based on their age or frequency.
    *   **Node Feature Masking (Time-dependent):** Mask node features potentially based on their temporal volatility.
    These augmentations aim to create views that are temporally correlated but distinct, forcing the model to learn robust representations invariant to minor temporal perturbations but sensitive to significant shifts.
2.  **Hyperbolic Contrastive Loss:** Let $\mathbf{z}_v'(t)$ and $\mathbf{z}_v''(t)$ be the hyperbolic embeddings of node $v$ obtained from the two views $G_t'$ and $G_t''$ respectively (potentially after passing through a hyperbolic projection head). We adapt the InfoNCE loss to hyperbolic space:
    $$ \mathcal{L}_{CL}(t) = - \sum_{v \in V_t} \log \frac{\exp(-d_{\mathbb{B}}(\mathbf{z}_v'(t), \mathbf{z}_v''(t)) / \tau)}{\sum_{u \in V_t} \exp(-d_{\mathbb{B}}(\mathbf{z}_v'(t), \mathbf{z}_u''(t)) / \tau)} $$
    where $d_{\mathbb{B}}$ is the Poincaré distance, $\tau$ is a temperature hyperparameter, and the summation in the denominator includes the positive pair $(v, v)$ and negative pairs $(v, u)$ for $u \neq v$. This loss encourages the embeddings of the same node under different temporal augmentations to be close in hyperbolic space, while pushing embeddings of different nodes apart. We might also contrast nodes across nearby time steps $t$ and $t-\delta$.

**3.3.3 Hyperbolic Temporal Memory Module**
To capture longer-range dependencies beyond immediate snapshots, we integrate a temporal memory module operating directly in hyperbolic space. We adapt the Gated Recurrent Unit (GRU) architecture, similar to HTGN [3], using Möbius operations:

Let $\mathbf{h}_v(t)$ be the final hyperbolic embedding of node $v$ at time $t$ from the hyperbolic GCN. The memory state $\mathbf{s}_v(t) \in \mathbb{B}^d$ is updated as follows:

1.  **Map to Tangent Space:** $\mathbf{h}_{v, \text{tan}}(t) = \log_{\mathbf{o}}^c(\mathbf{h}_v(t))$, $\mathbf{s}_{v, \text{tan}}(t-1) = \log_{\mathbf{o}}^c(\mathbf{s}_v(t-1))$.
2.  **Compute Gates (in Tangent Space):**
    $$ \mathbf{r}_{v, \text{tan}}(t) = \sigma_g(W_r [\mathbf{h}_{v, \text{tan}}(t), \mathbf{s}_{v, \text{tan}}(t-1)] + \mathbf{b}_r) $$
    $$ \mathbf{u}_{v, \text{tan}}(t) = \sigma_g(W_u [\mathbf{h}_{v, \text{tan}}(t), \mathbf{s}_{v, \text{tan}}(t-1)] + \mathbf{b}_u) $$
    where $[\cdot, \cdot]$ denotes concatenation, $W_r, W_u, \mathbf{b}_r, \mathbf{b}_u$ are learnable parameters, and $\sigma_g$ is the sigmoid function.
3.  **Compute Candidate State (in Hyperbolic Space):**
    $$ \mathbf{\tilde{s}}_{v, \text{tan}}(t) = \tanh( W_s [\mathbf{h}_{v, \text{tan}}(t), \exp_{\mathbf{o}}^c(\mathbf{r}_{v, \text{tan}}(t)) \odot_M \mathbf{s}_{v, \text{tan}}(t-1) ] + \mathbf{b}_s ) $$
    where $\odot_M$ denotes Möbius multiplication (element-wise multiplication after mapping the second operand to the tangent space at the origin).
    $$ \mathbf{\tilde{s}}_v(t) = \exp_{\mathbf{o}}^c(\mathbf{\tilde{s}}_{v, \text{tan}}(t)) $$
4.  **Update State (in Hyperbolic Space):**
    $$ \mathbf{s}_v(t) = \exp_{\mathbf{o}}^c(\mathbf{u}_{v, \text{tan}}(t)) \odot_M \mathbf{s}_v(t-1) \oplus \exp_{\mathbf{o}}^c(\mathbf{1} - \mathbf{u}_{v, \text{tan}}(t)) \odot_M \mathbf{\tilde{s}}_v(t) $$
    This involves interpolation using Möbius addition and multiplication, ensuring the state remains within $\mathbb{B}^d$. The final representation for downstream tasks can be $\mathbf{s}_v(t)$ or a combination of $\mathbf{h}_v(t)$ and $\mathbf{s}_v(t)$.

**3.4 Overall Loss Function**
The model can be trained end-to-end by minimizing the contrastive loss $\mathcal{L}_{CL}$. For task-specific training (e.g., link prediction, node classification), a task specific loss $\mathcal{L}_{Task}$ (e.g., cross-entropy for classification, margin-based loss for link prediction adapted for hyperbolic distances) is added:
$$ \mathcal{L}_{Total} = \mathcal{L}_{CL} + \lambda \mathcal{L}_{Task} $$
where $\lambda$ is a hyperparameter balancing the two objectives. Pre-training using only $\mathcal{L}_{CL}$ followed by fine-tuning with $\mathcal{L}_{Total}$ is also a viable strategy.

**3.5 Algorithm Pseudocode**

```
Algorithm 1: HyTECL Training Step
Input: Graph snapshots G = {G_1, ..., G_T}, augmentation functions A1, A2, temperature τ, task loss weight λ
Output: Trained HyTECL model parameters Θ

Initialize model parameters Θ
Initialize node memory states s_v(0) for all v (e.g., at origin or random in B^d)

for t = 1 to T do
    // Temporal Contrastive Learning Part
    Apply augmentations: G'_t = A1(G_t), G''_t = A2(G_t)
    // Compute embeddings for both views
    h'_v(t), s'_v(t) = HyTECL_Forward(G'_t, s_v(t-1), Θ) for all v in V_t
    h''_v(t), s''_v(t) = HyTECL_Forward(G''_t, s_v(t-1), Θ) for all v in V_t
    // Compute hyperbolic contrastive loss
    L_CL(t) = Calculate_Hyperbolic_InfoNCE(z'_v(t), z''_v(t), τ) // z are projected embeddings
    
    // Task-Specific Part (using one view, e.g., G'_t output)
    Predict task outputs Y_pred(t) using h'_v(t) or s'_v(t)
    Compute task loss L_Task(t) = Calculate_Task_Loss(Y_pred(t), Y_true(t))
    
    // Combine Losses
    L_Total(t) = L_CL(t) + λ * L_Task(t)
    
    // Backpropagation and Parameter Update
    Update Θ using gradient descent on L_Total(t)
    
    // Update Memory State (using output from one view, e.g., s'_v(t))
    s_v(t) = s'_v(t) or use average/other combination // Detach gradient if needed
end for

Function HyTECL_Forward(G_t, s_prev, Θ):
    h(t) = HyperbolicGCN(G_t, Θ_gcn)        // Multiple layers
    s(t) = HyperbolicMemoryUpdate(h(t), s_prev, Θ_mem)
    Return h(t), s(t)
```

**3.6 Experimental Design**

*   **Datasets:**
    *   *Dynamic Knowledge Graph Forecasting:* ICEWS14, ICEWS05-15, GDELT. These datasets involve predicting future relational triples (links) and inherently contain evolving hierarchical concept structures.
    *   *Fraud/Anomaly Detection:* Elliptic dataset (Bitcoin transactions), DGraph dataset (simulated evolving fraud). These represent dynamic node classification tasks where fraudulent patterns might exhibit specific structural (potentially hierarchical community) and temporal characteristics.
    *   *Social/Interaction Networks:* AS-733, CollegeMsg. Used for temporal link prediction, potentially exhibiting evolving community hierarchies.
*   **Baselines:**
    *   *Static GNNs:* GCN, GAT (applied snapshot-wise).
    *   *Euclidean Temporal GNNs:* CTDNE, DyRep, TGN, EvolveGCN.
    *   *Hyperbolic Static GNNs:* HGNN [5], HGCN [9] (applied snapshot-wise).
    *   *Hyperbolic Temporal GNNs:* HTGN [3], HGWaveNet [1], HVGNN [4].
    *   *Contrastive Methods:* GraphCL [7] adapted for temporal graphs (snapshot-wise or simple temporal pairing), potentially HGCL [2] adapted similarly.
*   **Tasks & Evaluation Metrics:**
    *   *Temporal Link Prediction / KG Forecasting:* Predict edges/triples at time $t+1$ given graph history up to $t$. Metrics: Mean Reciprocal Rank (MRR), Hits@k (k=1, 3, 10).
    *   *Dynamic Node Classification (Fraud Detection):* Classify nodes at time $t$ based on history up to $t$. Metrics: Area Under ROC Curve (AUC), F1-Score (Macro/Micro).
    *   *Hierarchy Preservation Analysis:*
        *   *Embedding Visualization:* Use Poincaré disk visualization to qualitatively assess if known hierarchical structures are captured.
        *   *Distortion Metrics:* Measure the average distortion when embedding synthetic trees or known hierarchical subgraphs from the datasets into the learned hyperbolic space compared to Euclidean embeddings. Calculate $\frac{1}{\binom{N}{2}} \sum_{i<j} |\frac{d_{\mathbb{H}}(\mathbf{h}_i, \mathbf{h}_j)}{d_G(i,j)} - 1|$, where $d_G$ is the graph distance.
*   **Implementation Details:** The framework will be implemented using Python with libraries like PyTorch and the `geoopt` library for hyperbolic operations. Experiments will be run on GPU infrastructure. Hyperparameters (embedding dimension, learning rate, temperature $\tau$, loss weight $\lambda$, curvature $c$, optimizer details) will be tuned using standard validation procedures (e.g., time-aware splitting). Ablation studies will be conducted to evaluate the contribution of each component (hyperbolic geometry, contrastive loss, memory module).

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A Novel HyTECL Framework:** A fully implemented and publicly available open-source codebase for the HyTECL framework, enabling reproducible research.
    2.  **State-of-the-Art Performance:** Demonstrably superior performance of HyTECL compared to existing Euclidean and hyperbolic baseline methods on benchmark tasks, particularly dynamic KG forecasting and fraud detection where hierarchy matters.
    3.  **Quantifiable Hierarchy Preservation:** Empirical evidence showing HyTECL's enhanced ability to capture and preserve latent hierarchical structures in dynamic graphs over time, supported by distortion metrics and visualizations.
    4.  **Robust Temporal Representations:** Validation of the effectiveness of hyperbolic temporal contrastive learning in learning representations robust to temporal noise and capturing meaningful dynamic patterns.
    5.  **Peer-Reviewed Publications:** Dissemination of findings through publications in top-tier machine learning or data mining conferences and journals, including the Temporal Graph Learning Workshop.

*   **Potential Impact:**
    *   **Methodological Advancement:** This research will push the boundaries of temporal graph learning by successfully integrating hyperbolic geometry and contrastive self-supervision, offering a new paradigm for handling dynamic hierarchical data.
    *   **Improved Real-World Applications:** By providing more accurate and insightful representations of dynamic systems, HyTECL could significantly improve applications in areas like:
        *   *Financial Security:* Better detection of evolving fraud rings and prediction of market structure changes.
        *   *Recommendation Systems:* More accurate modeling of user interest evolution and item category hierarchies.
        *   *Computational Biology:* Improved analysis of dynamic protein interactions or evolving phylogenetic relationships.
        *   *Social Network Analysis:* Deeper understanding of community evolution and information diffusion dynamics.
    *   **Stimulating Further Research:** This work is expected to stimulate further exploration into geometric deep learning for temporal data, contrastive learning in non-Euclidean spaces, and the development of more sophisticated models for complex evolving systems. It directly contributes to the themes of the workshop by offering novel methods for temporal graph representation learning, forecasting, and applications like anomaly detection, while also touching upon theoretical aspects of representation spaces.

---
**References** *(Based on the provided literature review)*

[1] Bai, Q., Nie, C., Zhang, H., Zhao, D., & Yuan, X. (2023). *HGWaveNet: A Hyperbolic Graph Neural Network for Temporal Link Prediction*. arXiv preprint arXiv:2304.07302.
[2] Liu, J., Yang, M., Zhou, M., Feng, S., & Fournier-Viger, P. (2022). *Enhancing Hyperbolic Graph Embeddings via Contrastive Learning*. arXiv preprint arXiv:2201.08554.
[3] Yang, M., Zhou, M., Kalander, M., Huang, Z., & King, I. (2021). *Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning in Hyperbolic Space*. arXiv preprint arXiv:2107.03767.
[4] Sun, L., Zhang, Z., Zhang, J., Wang, F., Peng, H., Su, S., & Yu, P. S. (2021). *Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs*. arXiv preprint arXiv:2104.02228.
[5] Chami, I., Ying, Z., Ré, C., & Leskovec, J. (2019). *Hyperbolic Graph Neural Networks*. arXiv preprint arXiv:1901.04598.
[6] Kazemi, S. M., Goel, R., Jain, K., Kulkarni, J., Srinivasa, V., Hashemian, A. M., ... & Poupart, P. (2020). *Temporal graph networks for deep learning on dynamic graphs*. Representation Learning on Graphs and Manifolds Workshop, ICLR 2020. *(Note: Cites a related survey concept, the provided arXiv link points to a different TGN survey)*
[7] You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). *Graph contrastive learning with augmentations*. Advances in Neural Information Processing Systems, 33, 5812-5823.
[8] Skarding, J., Gaber, M. M., & Krecja, M. (2021). *Foundations and modelling of dynamic networks using dynamic graph neural networks: A survey*. ACM Computing Surveys (CSUR), 54(5), 1-39. *(Note: Cites survey concept, provided arXiv link points to the survey)*
[9] Liu, R., Nickel, M., & Kiela, D. (2019). *Hyperbolic graph convolutional neural networks*. Advances in Neural Information Processing Systems, 32. *(Note: Cites HGCN concept, provided arXiv link points to the paper)*
[10] Zhang, J., Sun, L., Peng, H., Su, S., & Yu, P. S. (2021). *Hyperbolic graph neural networks with self-attention*. arXiv preprint arXiv:2106.07845.