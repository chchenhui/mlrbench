# Hierarchical Spatiotemporal Graphs for Unified Scene Representation in Autonomous Driving

## 1. Introduction

### Background and Motivation
Autonomous driving systems rely on interpreting complex environments through perception, prediction, and planning modules. However, traditional approaches often isolate these components, processing static elements (e.g., roads, traffic lights) and dynamic agents (e.g., vehicles, pedestrians) in separate pipelines. This fragmentation leads to error propagation, inefficiency, and reduced robustness in dynamic urban scenarios. Recent works like **VAD** (Vectorized Scene Representation) and **UniScene** (Unified Occupancy-centric Generation) have highlighted the potential of unified intermediate representations, but critical gaps remain in modeling hierarchical interactions and enabling scalability.

### Research Objectives
This work proposes a **hierarchical spatiotemporal graph (HSTG)** to unify scene representation for autonomous driving. The key objectives are:
1. Integrate static map features and dynamic agent states into a single graph structure.
2. Model multiscale interactions through adaptive edge weights and temporal convolution.
3. Enable joint optimization of perception, prediction, and planning via self-supervised contrastive learning.
4. Enhance generalization to novel scenarios while reducing dependency on labeled data.

### Significance
By unifying geometric priors from HD maps, LiDAR point clouds, and camera data into a structured graph, the proposed framework addresses the limitations of siloed architectures identified in **HDGT** and **Trajectron++**. The integration of contrastive learning builds on **Social-STGCNN**'s social interaction modeling but extends it to handle heterogeneous sensor inputs, aiming to achieve:
- **Robustness**: Reduced error propagation via joint optimization.
- **Generalization**: Improved handling of edge cases through self-supervised adaptation.
- **Efficiency**: Real-time processing via hierarchical graph sparsification.

---

## 2. Methodology

### 2.1 Data Collection and Preprocessing
**Datasets**: The model will be trained and validated on:
- **nuScenes**: 1,000 scenes with 3D LiDAR, camera, and HD map annotations.
- **Argoverse 2**: Sensor data with dense urban traffic interactions.

**Preprocessing**:
1. **Static Layer Extraction**: Road topology and traffic rules from HD maps are encoded as nodes with features $\mathbf{m}_i \in \mathbb{R}^{d_m}$ (e.g., lane direction, traffic light states).
2. **Dynamic Layer Extraction**: Agent positions and velocities from LiDAR/camera fusion are converted to nodes with states $\mathbf{a}_j(t) \in \mathbb{R}^{d_a}$.
3. **Temporal Alignment**: Multi-sensor data are synchronized at 10 Hz using Kalman filtering.

### 2.2 Hierarchical Spatiotemporal Graph Construction
The graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ includes three layers:

- **Static Layer**: Nodes represent lanes, intersections, and traffic signs. Edges encode connectivity (e.g., adjacent lanes).
- **Dynamic Layer**: Nodes represent agents. Edges model interactions using *adaptive attention weights*:
$$
e_{jk} = \sigma\left(\frac{\mathbf{W}_q \mathbf{a}_j}{\sqrt{d}} \cdot (\mathbf{W}_k \mathbf{a}_k)^T + \phi(\|\mathbf{p}_j - \mathbf{p}_k\|)\right)
$$
where $\phi(\cdot)$ is a learned distance kernel, and $\mathbf{W}_q, \mathbf{W}_k$ are projection matrices.
- **Temporal Layer**: Edges connect the same agent across timesteps, modeled via temporal convolutions:
$$
\mathbf{h}_j^{(t+1)} = \text{TCN}(\mathbf{h}_j^{(t)}, \mathbf{h}_j^{(t-1)}, \dots, \mathbf{h}_j^{(t-\tau)})
$$

### 2.3 Joint Perception-Prediction Learning
A dynamic graph neural network (DGNN) processes the HSTG in two stages:

1. **Spatial Encoding**:
   - GATv2 layers update node features by aggregating messages from neighbors:
   $$
   \mathbf{h}_i' = \text{ReLU}\left(\sum_{j \in \mathcal{N}_i} e_{ij} \mathbf{W} \mathbf{h}_j\right)
   $$
2. **Temporal Encoding**:
   - Dilated TCNs capture trajectory evolution over time:
   $$
   \mathbf{z}_i^{(t)} = \sum_{k=0}^{K-1} \mathbf{\Theta}_k \mathbf{h}_i^{(t - k \cdot d)}
   $$
   where $d$ is the dilation factor.

### 2.4 Self-Supervised Contrastive Learning
To enhance generalization, we employ scene graph contrastion:
- **Positive Pairs**: Augmented views of the same scene (e.g., rotated LiDAR scans).
- **Negative Pairs**: Graphs from different scenes.
- **Loss Function**:
$$
\mathcal{L}_{\text{cont}} = -\log \frac{\exp(\text{sim}(\mathbf{g}_s, \mathbf{g}_s^+)/\tau)}{\sum_{k=1}^B \exp(\text{sim}(\mathbf{g}_s, \mathbf{g}_k^-)/\tau)}
$$
where $\mathbf{g}_s$ is the graph embedding, and $\tau$ is a temperature parameter.

### 2.5 Experimental Design
**Baselines**: Compare against **VAD**, **HDGT**, and **Trajectron++** on:
1. **Perception Metrics**: mAP for 3D object detection.
2. **Prediction Metrics**: Average/Final Displacement Error (ADE/FDE) for trajectories.
3. **Generalization**: Performance on unseen nuScenes mini-val vs. Argoverse test splits.

**Training Protocol**:
- **Phase 1**: Pretrain contrastive module on unlabeled data.
- **Phase 2**: Fine-tune with joint loss $\mathcal{L} = \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{cont}}$.

---

## 3. Expected Outcomes & Impact

1. **Performance Gains**: Anticipate **15–20% reduction in ADE** and **10% improvement in mAP** over raster-based baselines in dense urban scenarios.
2. **Reduced Label Dependency**: Contrastive pretraining aims to cut annotation needs by 30–50% for downstream tasks.
3. **Safety Enhancement**: Explicit interaction modeling via adaptive edges could reduce collision risks in multi-agent simulations by modeling right-of-way conflicts.
4. **Broader Impact**: A unified HSTG representation could streamline the transition from modular stacks to end-to-end driving systems, enabling safer deployment in complex environments.

---

This proposal bridges the gap between static and dynamic scene understanding, offering a scalable framework to advance autonomous driving systems toward real-world reliability.