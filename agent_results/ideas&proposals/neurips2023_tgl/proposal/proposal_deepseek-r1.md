### **Research Proposal: HyTECL – Hyperbolic Temporal Contrastive Learning for Dynamic Graphs**

---

#### **1. Title**  
**HyTECL – Hyperbolic Temporal Contrastive Learning for Dynamic Graphs**

---

#### **2. Introduction**  
**Background**  
Temporal graphs model dynamic interactions in networks such as social connections, financial transactions, and biological processes. Traditional graph neural networks (GNNs) often assume static structures, limiting their ability to capture time-evolving patterns. Recent work highlights the importance of hyperbolic geometry for representing hierarchical relationships inherent in real-world graphs, but integrating hyperbolic embeddings with temporal dynamics remains a key challenge. Existing methods either focus on static hyperbolic embeddings or temporal models in Euclidean space, failing to jointly optimize hierarchical and time-aware representations.  

**Research Objectives**  
This proposal addresses these gaps by introducing **HyTECL**, a novel framework that unifies hyperbolic graph neural networks with temporal contrastive learning. The core objectives are:  
1. Develop a hyperbolic-temporal GNN architecture that embeds nodes in hyperbolic space while modeling evolving graph structures.  
2. Design a contrastive learning mechanism tailored for hyperbolic spaces to align node representations across temporally shifted views.  
3. Integrate a memory module to capture long-range dependencies in hyperbolic embeddings.  
4. Validate HyTECL’s efficacy on dynamic forecasting and anomaly detection tasks, ensuring scalability and robustness.  

**Significance**  
HyTECL bridges hyperbolic geometry and temporal graph learning, unlocking new capabilities for hierarchy-aware dynamic modeling. Its success will advance applications like fraud detection (where transaction hierarchies evolve over time) and disease spread prediction (where contact networks exhibit both hierarchy and temporal progression).  

---

#### **3. Methodology**  

##### **3.1 Data Collection**  
We evaluate HyTECL on:  
- **Temporal Knowledge Graphs**: ICEWS (event-based), GDELT (news-based).  
- **Fraud Detection**: YelpChi (review spam), Elliptic (blockchain transactions).  
- **Social Networks**: Reddit Hyperlink (dynamic communities).  
Each dataset provides timestamped edges, node features, and task-specific labels (e.g., anomalous nodes).  

##### **3.2 Model Architecture**  

**A. Hyperbolic Graph Convolutional Layer**  
HyTECL leverages the Poincaré ball model $\mathbb{D}^n$ with curvature $c$. For a node $v$ at time $t$:  
1. **Project features to hyperbolic space**: Map input features $x_v^t \in \mathbb{R}^d$ to $\mathbb{D}^n$ via the exponential map:  
$$
h_v^{t,0} = \exp_0^c(x_v^t) = \tanh\left(\sqrt{c}\|x_v^t\|\right)\frac{x_v^t}{\sqrt{c}\|x_v^t\|}
$$
2. **Neighborhood aggregation**: For neighbors $\mathcal{N}(v)$, compute hyperbolic messages using Fermi-Dirac attention:  
$$
m_{v}^t = \bigoplus_{u \in \mathcal{N}(v)} \alpha_{vu} \cdot \log_0^c\left(h_u^{t,k}\right), \quad \alpha_{vu} = \text{softmax}\left(\sigma\left(a^T[\log_0^c(h_v^{t,k}) \Vert \log_0^c(h_u^{t,k})]\right)\right)
$$
3. **Update node embeddings**: Combine messages with a hyperbolic residual connection:  
$$
h_v^{t,k+1} = \exp_{h_v^{t,k}}^c\left(\text{MLP}\left(\log_{h_v^{t,k}}^c(m_v^t)\right)\right)
$$

**B. Temporal Contrastive Learning Module**  
To capture temporal dynamics:  
1. **Time-aware graph augmentation**: Generate two views $(G^{t_1}, G^{t_2})$ via:  
   - Edge masking: Remove edges with probability proportional to their age.  
   - Temporal subgraph sampling: Extract subgraphs using a sliding time window.  
2. **Contrastive loss**: Maximize similarity between positive pairs $(h_v^{t_1}, h_v^{t_2})$ and minimize similarity for negatives:  
$$
\mathcal{L}_{\text{cont}} = -\sum_{v} \log \frac{\exp\left(-d_{\mathbb{D}}(h_v^{t_1}, h_v^{t_2}) / \tau\right)}{\sum_{u \neq v} \exp\left(-d_{\mathbb{D}}(h_v^{t_1}, h_u^{t_2}) / \tau\right)}
$$
where $d_{\mathbb{D}}(x,y) = \frac{2}{\sqrt{c}} \tanh^{-1}\left(\sqrt{c}\|(-x) \oplus y\|\right)$ is the hyperbolic distance.  

**C. Temporal Memory Module**  
Store historical embeddings $\{h_v^{t-k}\}$ and aggregate them via a hyperbolic gated recurrent unit (GRU):  
$$
r_v^t = \sigma\left(\log_0^c\left(W_r \left[\log_0^c(h_v^{t}) \Vert \log_0^c(m_v^{t-1})\right]\right)\right)
$$
$$
z_v^t = \sigma\left(\log_0^c\left(W_z \left[\log_0^c(h_v^{t}) \Vert \log_0^c(m_v^{t-1})\right]\right)\right)
$$
$$
\tilde{m}_v^t = \exp_0^c\left(\tanh\left(W_h \left[\log_0^c(h_v^t) \Vert \log_0^c(r_v^t \odot m_v^{t-1})\right]\right)\right)
$$
$$
m_v^t = \exp_0^c\left(\log_0^c(z_v^t \odot m_v^{t-1}) + \log_0^c((1 - z_v^t) \odot \tilde{m}_v^t)\right)
$$

**D. Training Objective**  
Combine contrastive and task-specific losses (e.g., cross-entropy for node classification):  
$$
\mathcal{L} = \mathcal{L}_{\text{cont}} + \lambda \mathcal{L}_{\text{task}}
$$

##### **3.3 Experimental Design**  
- **Baselines**: Compare against HGWaveNet (hyperbolic-temporal), HTGN (hyperbolic-GRU), TGN (Euclidean-temporal), and DySAT (attention-based).  
- **Metrics**:  
  - **Forecasting**: Mean Reciprocal Rank (MRR), Hits@10.  
  - **Fraud Detection**: AUC-ROC, F1-score.  
  - **Hierarchy Preservation**: Gromov $\delta$-hyperbolicity.  
- **Ablation Studies**: Validate contributions of contrastive learning, hyperbolic layers, and memory module.  

---

#### **4. Expected Outcomes & Impact**  

**Expected Outcomes**  
1. **Improved Accuracy**: HyTECL is expected to outperform Euclidean-temporal GNNs by 5–10% on dynamic link prediction and fraud detection tasks.  
2. **Hierarchy Preservation**: Embeddings will exhibit lower Gromov $\delta$-hyperbolicity (higher hierarchy) compared to baselines.  
3. **Robustness**: Temporal contrastive learning will enhance resilience to noisy or sparse temporal edges.  

**Impact**  
HyTECL will advance temporal graph learning by:  
- Enabling hierarchy-aware modeling for applications like financial fraud detection (e.g., tracing hierarchical money-laundering networks over time).  
- Providing a modular framework for integrating hyperbolic geometry with contrastive learning in dynamic settings.  
- Delivering scalable codebases and benchmarks to accelerate research.  

---

The proposed work bridges critical gaps in temporal and geometric graph learning, offering a foundation for future research in dynamic, hierarchy-rich networks.