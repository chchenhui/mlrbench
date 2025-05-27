# **HyTECL – Hyperbolic Temporal Contrastive Learning for Dynamic Graphs**

---

## **1. Introduction**

### **Background**
Graphs are fundamental to modeling relational data across domains like social networks, biology, and finance. While traditional graph neural networks (GNNs) assume static graphs, real-world networks evolve dynamically over time, necessitating temporal graph learning (TGNNs) to capture interactions like node/edge updates and structural shifts. Recent advancements in TGNNs have addressed temporal dynamics through recurrent networks, attention mechanisms, and temporal aggregation. However, most methods operate in Euclidean space, which poorly approximates latent hierarchies inherent in many real-world graphs (e.g., hierarchical social influence, taxonomies in knowledge graphs). Hyperbolic geometry, a non-Euclidean space with exponential volume growth, naturally accommodates hierarchical structures but remains underexplored for temporal graphs.

Existing methods like **HTGN** (Yang et al., 2021) and **HGWaveNet** (Bai et al., 2023) combine hyperbolic GCNs with temporal models (e.g., GRUs, causal convolutions) for link prediction. However, they neglect contrastive learning, a self-supervised framework that improves robustness by pulling similar instances closer in representation space. Meanwhile, **HGCL** (Liu et al., 2022) introduced contrastive learning to hyperbolic GNNs but focused on static graphs. The synergy of contrastive learning, hyperbolic geometry, and temporal modeling in dynamic graphs remains unexplored, motivating our work.

---

### **Research Objectives**
This research aims to develop **HyTECL**, a novel framework that bridges hyperbolic graph neural networks, contrastive learning, and temporal dynamics. Specifically:
1. Formulate a hyperbolic graph convolutional layer with temporal augmentations to capture hierarchical and dynamical patterns.
2. Design a contrastive loss in hyperbolic space that aligns node embeddings across temporally shifted graph views.
3. Introduce a temporal memory module to aggregate historical embeddings and model long-range dependencies.
4. Evaluate HyTECL on dynamic knowledge graph forecasting and fraud detection, outperforming state-of-the-art methods in accuracy, robustness, and hierarchy preservation.

---

### **Significance**
HyTECL addresses critical gaps in temporal graph learning:
- **Theoretical Impact**: Integrates hyperbolic geometry with temporal contrastive learning, extending GNN expressiveness for hierarchical data.
- **Practical Applications**: Enhances fraud detection (e.g., anomalous transaction graphs) and temporal knowledge graphs (e.g., drug-gene interactions).
- **Benchmarking**: Establishes hyperbolic temporal baselines for dynamic graph prediction tasks, fostering future research.

---

## **2. Methodology**

### **2.1 Overall Framework**
HyTECL processes dynamic graphs $\mathcal{G} = \{G_t\}_{t=1}^T$, where each $G_t = (\mathcal{V}_t, \mathcal{E}_t, \mathbf{X}_t)$ represents a snapshot with nodes $\mathcal{V}_t$, edges $\mathcal{E}_t$, and node features $\mathbf{X}_t$. The framework has three components:
1. **Hyperbolic Graph Convolution**: Maps Euclidean features to hyperbolic space and aggregates neighborhood information.
2. **Temporal Augmentations & Contrastive Learning**: Generates two temporally augmented views of $G_t$ and aligns embeddings via a hyperbolic contrastive loss.
3. **Temporal Memory**: Aggregates past embeddings to capture long-term dependencies.

---

### **2.2 Hyperbolic Graph Convolutional Layer**

Hyperbolic space $\mathbb{H}^n$ is defined on the $n$-dimensional manifold:
$$
\mathbb{H}^n = \left\{ \mathbf{x} \in \mathbb{R}^{n+1} \mid \langle \mathbf{x}, \mathbf{x} \rangle_{\text{Lor}} = -x_0^2 + \sum_{i=1}^n x_i^2 = -1, x_0 > 0 \right\}
$$
where $\langle \cdot, \cdot \rangle_{\text{Lor}}$ denotes the Lorentzian inner product. Features $\mathbf{x} \in \mathbb{R}^d$ are first embedded into $\mathbb{H}^n$ via an exponential map $\exp^\kappa_\mathbf{0}(\cdot)$, and GCN operations are computed in tangent space $T_\mathbf{0}\mathbb{H}^n$ before mapping back to $\mathbb{H}^n$:
$$
\mathbf{h}_i^{(l)} = \exp^\kappa_\mathbf{0} \left( \sum_{j \in \mathcal{N}(i)} \mathrm{LogSumExp}(\mathbf{W}^{(l)} \log^\kappa_{\mathbf{0}}(\mathbf{h}_j^{(l-1)})) \right),
$$
where $\mathbf{W}^{(l)}$ is a learnable projection matrix, and $\mathrm{LogSumExp}$ combines hyperbolic addition and activation.

---

### **2.3 Temporal Augmentations and Contrastive Learning**

Two temporally augmented views $G_t^v$ ($v \in \{1,2\}$) are generated from $G_t$ via:
- **Time-aware edge masking**: Randomly masks edges with probability proportional to $t - t_e$ ($t_e$: edge age).
- **Subgraph sampling**: Samples neighborhoods within a temporal window $[t - \Delta, t]$.

The goal is to maximize agreement between embeddings of the same node across views:
$$
\mathcal{L}_{\text{c}} = -\log \frac{\exp(\mathrm{dist}_\mathcal{H}(f_t^1(\mathbf{v}_t^1), f_t^2(\mathbf{v}_t^2)) / \tau)}{\sum_{\mathbf{v}_k \in N(\mathbf{v}_t)} \exp(\mathrm{dist}_\mathcal{H}(f_t^1(\mathbf{v}_t), f_t^2(\mathbf{v}_k)) / \tau)},
$$
where $\mathrm{dist}_\mathcal{H}(\cdot, \cdot)$ is the hyperbolic distance:
$$
\mathrm{dist}_\mathcal{H}(\mathbf{x}, \mathbf{y}) = \cosh^{-1} \left(1 + 2 \frac{\|\mathbf{x} - \mathbf{y}\|^2}{(1 - \|\mathbf{x}\|^2)(1 - \|\mathbf{y}\|^2)} \right),
$$
$\tau$ is a temperature, and $N(\mathbf{v}_t)$ includes negatives from other nodes or timestamps.

---

### **2.4 Temporal Memory Module**

Historical embeddings $\mathbf{h}_i^{\leq t}$ are stored in a memory bank. Using the Möbius transformation, the memory updates:
$$
\mathbf{m}_t = \frac{1}{|\mathcal{V}_t|} \oplus_{\kappa} \left( \log^\kappa_{\mathbf{0}}(\mathbf{h}_i^t)_{i \in \mathcal{V}_t} \right),
$$
where $\oplus_\kappa$ denotes Möbius addition. A hyperbolic GRU (HGRU) updates the memory:
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \otimes \mathbf{m}_t), \quad \mathbf{z}_t = \sigma(\mathbf{W}_z \otimes \mathbf{m}_t), \quad \mathbf{h}_t = (1 - \mathbf{z}_t) \otimes \mathbf{m}_{t-1} + \mathbf{z}_t \otimes \mathbf{r}_t.
$$

---

### **2.5 Training Procedure**
1. **Initialization**: Pre-train hyperbolic GCN on initial $G_1$ using link prediction loss $\mathcal{L}_{\text{link}}$.
2. **Contrastive Learning**: Optimize $\mathcal{L} = \alpha \mathcal{L}_{\text{link}} + \beta \mathcal{L}_{\text{c}} + \gamma \mathcal{L}_{\text{memory}}$ via Riemannian Adam.
3. **Evaluation**: Perform early stopping using validation loss on unseen edge timestamps.

---

### **2.6 Experimental Design**

#### **Datasets**
- **Dynamic Knowledge Graphs**: **ICEWS18** (international events), **Wikidata** (entity-relation updates).
- **Financial Fraud Detection**: **AIRDROP** (cryptocurrency transactions with known fraudulent accounts).

#### **Baselines**
- **Euclidean TGNNs**: TGN (Rossi et al., 2020), DySAT (Sankar et al., 2020).
- **Hyperbolic TGNNs**: HTGN, HGWaveNet.
- **Contrastive GNNs**: HGCL, GCN-Contrast (You et al., 2020).

#### **Metrics**
- **Link Prediction**: MRR, Hits@10, AUC-ROC.
- **Fraud Detection**: F1-score, AUC.
- **Hierarchy Preservation**: Lowest Common Ancestor (LCA) score between embeddings.
- **Scalability**: Training time per epoch on 10 million edges.

#### **Ablation Studies**
- Impact of memory length $T$.
- Choice of hyperbolic curvatures $\kappa$.
- Sensitivity to augmentation strategies.

---

## **3. Expected Outcomes & Impact**

### **3.1 Expected Outcomes**
1. **Performance Improvements**:
   - HyTECL will outperform baselines on link prediction accuracy by 5-8% (e.g., MRR on ICEWS18) and fraud detection F1 by 10%.
   - Hierarchy Preservation Score will improve by 15% over Euclidean models, validated via LCA consistency.

2. **Theoretical Contributions**:
   - Formalize contrastive learning in hyperbolic space with time-aware augmentations.
   - Demonstrate that Möbius-addition memory improves long-range dependency modeling.

3. **Scalability**:
   - Hybrid CPU-GPU implementation will handle 10M+ edges with linear time complexity via subgraph sampling.

---

### **3.2 Impact on Applications**
- **Fraud Detection**: Enables early identification of Ponzi schemes in cryptocurrency via hierarchical transaction patterns.
- **Drug Discovery**: Models evolving molecular graphs to predict synergistic drug–target interactions.
- **Social Media**: Detects polarization clusters in temporal networks with higher fidelity.

---

### **3.3 Ethical Considerations**
- **Data Privacy**: Use anonymized social and financial datasets. Apply differential privacy to public benchmarks.
- **Bias Mitigation**: Avoid overfitting to majority node types via focal loss adjustments.
- **Transparency**: Visualize learned hierarchies using hyperbolic embeddings (e.g., Poincaré disks) for auditability.

---

### **3.4 Open-Source Release**
We will open-source HyTECL’s codebase with Docker containers and pre-trained models. A PyTorch Geometric extension will facilitate adoption in academia and industry.

---

## **Conclusion**
HyTECL pioneers the integration of hyperbolic geometry, contrastive learning, and temporal dynamics for dynamic graphs. By explicitly modeling hierarchical and time-variant patterns, we aim to set a new benchmark in temporal graph learning with broad implications for security, healthcare, and network science. This work bridges the theoretical gap between non-Euclidean spaces and temporal representations, accelerating progress toward robust, hierarchy-aware AI for evolving networks.