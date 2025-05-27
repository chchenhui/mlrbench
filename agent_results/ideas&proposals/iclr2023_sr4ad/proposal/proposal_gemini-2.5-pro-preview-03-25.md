Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

## **1. Title: Hierarchical Spatiotemporal Graphs for Unified Scene Representation and Joint Perception-Prediction in Autonomous Driving**

## **2. Introduction**

**2.1 Background**
Autonomous driving (AD) systems promise transformative changes in transportation safety, efficiency, and accessibility. However, achieving robust performance in complex, dynamic urban environments remains a significant challenge. A core difficulty lies in interpreting the intricate interplay between static infrastructure (roads, buildings, traffic signs) and dynamic actors (vehicles, pedestrians, cyclists). Traditional AD pipelines often employ modular approaches, where perception, prediction, and planning are handled by separate components (Caesar et al., 2020). While successful in many scenarios, this modularity can lead to information loss between stages, error propagation, and suboptimal decisions, particularly when complex interactions occur (Liang et al., 2020). Furthermore, these fragmented systems struggle to efficiently reason about the joint evolution of the scene, limiting their ability to anticipate rare but critical events ("edge cases").

Recent research has explored unified scene representations to overcome these limitations. Approaches like bird's-eye-view (BEV) grids (Philion & Fidler, 2020), occupancy grids (Li et al., 2024; Min et al., 2023), and vectorized representations (Jiang et al., 2023) aim to provide a holistic view of the driving environment. Occupancy-based methods offer dense geometric understanding but can struggle with instance-level reasoning and scalability. Vectorized methods excel at representing sparse elements like agents and map features efficiently, facilitating downstream planning, but may lack the rich contextual encoding of denser representations.

Graph-based representations have emerged as a powerful tool for modeling interactions, particularly in trajectory prediction tasks (Jia et al., 2022; Ivanovic & Pavone, 2020; Sadeghian et al., 2020; Tang et al., 2019). These methods represent entities as nodes and their relationships as edges, allowing Graph Neural Networks (GNNs) to explicitly reason about dependencies. However, many existing graph-based approaches focus primarily on dynamic agents or specific tasks like trajectory forecasting (Zhao et al., 2019; Song et al., 2020), often neglecting the integration of static environmental context or joint optimization across perception and prediction. The challenge identified by Trajectron++ regarding heterogeneous data integration (Salzmann et al., 2020) remains pertinent. Integrating static and dynamic elements within a single framework, ensuring scalability, fusing multi-modal sensor data effectively, achieving generalization, and informing safety-critical decisions are persistent challenges highlighted in the literature.

**2.2 Research Gap and Proposed Idea**
There is a need for a unified scene representation that seamlessly integrates diverse scene elements (static and dynamic), explicitly models their spatiotemporal interactions, supports joint learning across traditional AD tasks (perception, prediction), scales efficiently, and generalizes well to unseen scenarios.

This research proposes a **Hierarchical Spatiotemporal Graph (HSG)** as a novel unified scene representation for autonomous driving. The core idea is to structure the scene understanding problem using a graph that captures relationships at multiple levels of abstraction, both spatially and temporally. The HSG will represent static infrastructure elements (e.g., lane segments, traffic lights) and dynamic agents (e.g., vehicles, pedestrians) as distinct node types within a shared geometric space. Edges will represent relationships (e.g., spatial proximity, lane adherence, potential interaction), with adaptive weights learned via attention mechanisms to capture the dynamic relevance of interactions. The hierarchical structure could, for instance, represent individual agents and map elements at a lower level, local traffic configurations (e.g., intersection interactions) at an intermediate level, and broader scene context at a higher level. Temporally, the graph evolves, and its dynamics are modeled using GNNs coupled with temporal sequence models (e.g., Temporal Convolutional Networks - TCNs or GRU/LSTMs). This framework facilitates **joint learning of perception and prediction**, allowing the model to simultaneously refine object detection/tracking and forecast future trajectories by leveraging shared contextual information encoded in the graph. Furthermore, we propose using **self-supervised contrastive learning** on the graph representations to enhance robustness and generalization, reducing reliance on extensively labeled data.

**2.3 Research Objectives**
The primary objectives of this research are:

1.  **Develop the Hierarchical Spatiotemporal Graph (HSG) formalism:** Define the node types, edge types, hierarchical structure, and adaptive interaction modeling mechanisms suitable for comprehensive driving scene representation.
2.  **Integrate Multi-Modal Sensor Data:** Design methods to effectively fuse information from LiDAR, cameras, and potentially radar and HD maps into the HSG node and edge features.
3.  **Implement a Dynamic Graph Neural Network Architecture:** Develop a GNN architecture coupled with temporal models (e.g., GNN-TCN) that operates on the HSG to perform joint perception refinement (e.g., state estimation, tracking) and multi-agent trajectory prediction.
4.  **Incorporate Self-Supervised Learning:** Integrate a contrastive learning framework operating on the HSG representations to improve model generalization and robustness to variations in scenarios and sensor inputs.
5.  **Validate the Approach:** Empirically evaluate the proposed HSG framework on large-scale autonomous driving datasets, comparing its performance against state-of-the-art modular pipelines and other unified representation methods for perception and prediction tasks.
6.  **Analyze Safety Implications:** Investigate how the explicit interaction modeling within the HSG can potentially benefit downstream safety-critical planning tasks by providing richer, more reliable predictions.

**2.4 Significance**
This research addresses critical limitations in current AD systems by proposing a novel, unified, and interaction-aware scene representation. The potential significance includes:

*   **Improved Accuracy and Robustness:** Jointly optimizing perception and prediction within a unified framework informed by explicit interaction modeling is expected to yield more accurate and robust performance, especially in complex, interactive scenarios.
*   **Enhanced Generalization:** The combination of a structured representation (HSG) and self-supervised learning aims to improve generalization to unseen environments and edge cases, a key requirement for real-world deployment.
*   **Better Interaction Understanding:** The HSG explicitly models relationships between static and dynamic elements, providing deeper insights into scene dynamics compared to purely grid-based or agent-centric approaches.
*   **Foundation for Safer Planning:** By providing more accurate and context-aware multi-agent predictions, the HSG can serve as a stronger foundation for downstream motion planning and decision-making components, contributing to overall system safety.
*   **Advancing Representation Learning:** This work contributes a novel hierarchical graph structure for dynamic, multi-modal environments, potentially inspiring applications beyond autonomous driving.
*   **Alignment with Workshop Themes:** The proposed research directly aligns with the workshop's focus on representation learning, interaction modeling, ML for safety/generalization, and new perspectives on AD system integration.

## **3. Methodology**

**3.1 Data Collection and Preparation**
We will leverage large-scale, publicly available autonomous driving datasets that provide synchronized multi-modal sensor data and ground truth annotations. Primary datasets will include:

*   **nuScenes:** Provides 360-degree camera and LiDAR coverage, radar, IMU, GPS, along with 3D bounding box annotations, trajectories, and HD map information. Ideal for evaluating multi-modal fusion and interaction modeling.
*   **Waymo Open Dataset (WOD):** Offers high-resolution LiDAR and camera data with high-quality 3D labels and trajectories, suitable for complex urban scenarios.
*   **Argoverse 2:** Features rich map information and long-horizon trajectory data, particularly useful for evaluating prediction performance and map integration.

**Data Pre-processing:**
1.  **Sensor Fusion:** Calibrate sensors and project data into a common coordinate frame (e.g., ego-vehicle frame or a world frame).
2.  **Feature Extraction:**
    *   **LiDAR:** Process point clouds using techniques like PointNet++ (Qi et al., 2017) or VoxelNet (Zhou & Tuzel, 2018) to extract point-wise or object-level geometric features.
    *   **Camera:** Employ pre-trained Convolutional Neural Networks (CNNs, e.g., ResNet, EfficientNet) to extract visual appearance features from relevant image regions corresponding to detected objects or map elements.
    *   **HD Maps:** Extract relevant map elements (lane centerlines, boundaries, stop lines, traffic lights) within the region of interest, representing them geometrically (e.g., polylines) and semantically (e.g., lane type, light status).
3.  **Initial State Estimation:** Use an off-the-shelf 3D object detector and tracker or incorporate a detection head within our model to obtain initial estimates of dynamic agent states (position, velocity, heading, bounding box) over a historical time window $T_h$. Static elements are identified via map matching.

**3.2 Hierarchical Spatiotemporal Graph (HSG) Construction**
At each timestep $t$, we construct a scene graph $G_t = (V_t, E_t)$. Over a time horizon $T = T_h + T_f$ (history + future), this forms a spatiotemporal graph sequence.

*   **Node Set ($V_t$):**
    *   **Dynamic Agent Nodes ($V_t^{dyn}$):** Represent vehicles, pedestrians, cyclists. Each node $v_i \in V_t^{dyn}$ is associated with a state vector $s_{i,t}$ (position, velocity, heading, size) and a feature vector $h_{i,t}^{(0)}$ incorporating geometric features (from LiDAR) and appearance features (from cameras).
    *   **Static Element Nodes ($V_t^{sta}$):** Represent relevant map features like lane segments, intersection areas, traffic lights, stop signs. Each node $v_j \in V_t^{sta}$ has features representing its type, geometry (polyline/polygon vertices), and state (e.g., traffic light color, if applicable). $h_{j,t}^{(0)}$.
    *   **Ego-Vehicle Node ($v_{ego}$):** A special node representing the autonomous vehicle.

*   **Edge Set ($E_t$):** Edges represent relationships. Edge types include:
    *   **Agent-Agent Edges ($E_t^{dyn-dyn}$):** Connect dynamic agents within a certain proximity or potential interaction zone (e.g., based on predicted path overlap).
    *   **Agent-Lane Edges ($E_t^{dyn-sta}$):** Connect dynamic agents to the lane segment(s) they occupy or are near.
    *   **Lane-Lane Edges ($E_t^{sta-sta}$):** Connect adjacent or intersecting lane segments based on map topology.
    *   **Agent-TrafficLight/StopSign Edges ($E_t^{dyn-sta}$):** Connect agents to relevant traffic control elements they are approaching.

*   **Adaptive Edge Weights:** The influence of connected nodes is not uniform. We propose learning adaptive edge weights using an attention mechanism, similar to Graph Attention Networks (GAT) (Veličković et al., 2018). For an edge between node $i$ and node $j$ at layer $l$, the attention coefficient $\alpha_{ij}^{(l)}$ can be computed based on their features $h_i^{(l)}$ and $h_j^{(l)}$ and potentially their relative state:
    $$e_{ij}^{(l)} = \text{LeakyReLU}(\mathbf{a}^{(l)T} [ \mathbf{W}^{(l)} h_i^{(l)} || \mathbf{W}^{(l)} h_j^{(l)} || \phi(s_i, s_j) ])$$
    $$\alpha_{ij}^{(l)} = \text{softmax}_j (e_{ij}^{(l)}) = \frac{\exp(e_{ij}^{(l)})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik}^{(l)})}$$
    where $\mathbf{W}^{(l)}$ is a learnable weight matrix, $\mathbf{a}^{(l)}$ is a learnable attention vector, $||$ denotes concatenation, $\mathcal{N}(i)$ is the neighborhood of node $i$, and $\phi(s_i, s_j)$ represents optional relative state features (e.g., distance, relative velocity).

*   **Hierarchical Structure:** The hierarchy can be implemented spatially. Level 1 nodes represent individual agents/map primitives. Level 2 nodes could be generated via graph pooling (e.g., DiffPool, Ying et al., 2018) or defined semantically (e.g., an 'intersection node' aggregating connected lanes and agents within the intersection area). Information propagates up and down the hierarchy, allowing the model to reason about both fine-grained interactions and coarse-grained scene context. An alternative is a temporal hierarchy where different levels process information at different time scales. We will initially focus on spatial hierarchy.

**3.3 Joint Perception-Prediction Model: HSG-Net**

We propose **HSG-Net**, a model operating on the HSG sequence.

1.  **Input:** A sequence of HSGs $\{G_{t-T_h+1}, ..., G_t\}$ constructed over the history window, with initial node features derived from sensors and detectors.
2.  **Spatiotemporal Encoding:**
    *   **Spatial GNN:** At each historical timestep $\tau \in [t-T_h+1, t]$, a GNN (e.g., multi-layer GAT using the adaptive weights) processes the graph $G_\tau$. It updates node representations by aggregating information from neighbours:
        $$h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(l)} \mathbf{W}^{(l)} h_j^{(l)} \right)$$
        where $\sigma$ is a non-linear activation function (e.g., ReLU, GeLU). This refines object states based on interactions and context. The output is an updated feature vector $h_{i,\tau}^{(L)}$ for each node $i$ at time $\tau$.
    *   **Temporal RNN/TCN:** The sequence of updated node features $\{h_{i, t-T_h+1}^{(L)}, ..., h_{i, t}^{(L)}\}$ for each node $i$ is fed into a temporal model (e.g., GRU, LSTM, or TCN) to capture temporal dynamics and dependencies.
        $$z_{i,t} = \text{TemporalModel}(\{h_{i, \tau}^{(L)}\}_{\tau=t-T_h+1}^t)$$
        The output $z_{i,t}$ summarizes the spatiotemporal context for node $i$ up to time $t$.
3.  **Prediction Heads:**
    *   **Trajectory Prediction:** For each dynamic agent $i$, a prediction head (e.g., MLP or another RNN decoder) takes $z_{i,t}$ as input and predicts future states (position, velocity) over the prediction horizon $T_f$. To handle uncertainty and multimodality, we can predict parameters of a mixture density network (MDN) or use query-based approaches like DETR (Carion et al., 2020).
        $$(\hat{s}_{i, t+1}, ..., \hat{s}_{i, t+T_f}) = \text{PredictorHead}(z_{i,t})$$
    *   **Perception Refinement (Optional):** A perception head can take $z_{i,t}$ to output refined estimates of the current state $s_{i,t}$ (e.g., bounding box adjustments, velocity refinement) or classification scores.

**3.4 Joint Learning and Optimization**

*   **Loss Function:** We use a composite loss function to train HSG-Net end-to-end:
    $$\mathcal{L} = \lambda_{pred} \mathcal{L}_{pred} + \lambda_{perc} \mathcal{L}_{perc} + \lambda_{reg} \mathcal{L}_{reg}$$
    *   $\mathcal{L}_{pred}$: Trajectory prediction loss. For deterministic prediction, use ADE/FDE based on L2 distance. For probabilistic prediction, use Negative Log-Likelihood (NLL) of the ground truth trajectory under the predicted distribution.
        $$\mathcal{L}_{pred} = \frac{1}{N_{dyn}} \sum_{i \in V_t^{dyn}} \frac{1}{T_f} \sum_{k=1}^{T_f} || \hat{s}_{i, t+k} - s_{i, t+k} ||_2^2 \quad \text{(Example: ADE/FDE style loss)}$$
    *   $\mathcal{L}_{perc}$: Perception refinement loss (if applicable), e.g., smooth L1 loss for bounding box regression or cross-entropy for classification refinement.
    *   $\mathcal{L}_{reg}$: Regularization terms (e.g., L2 weight decay).
    *   $\lambda_{pred}, \lambda_{perc}, \lambda_{reg}$ are weighting hyperparameters.

**3.5 Self-Supervised Contrastive Learning**
To improve representation robustness and generalization, we incorporate contrastive learning during pre-training or as an auxiliary task during supervised training.

*   **Augmentations:** Create positive pairs by applying augmentations to the input HSG sequence that should not change the core semantics, e.g., minor sensor noise simulation, temporal jittering, masking some nodes/edges, slight variations in agent starting positions within physical plausibility.
*   **Positive/Negative Samples:** A positive pair consists of two augmented views of the same scene graph sequence. Negative pairs consist of views from different scene graph sequences.
*   **Contrastive Loss:** Use a contrastive loss objective, such as InfoNCE (Oord et al., 2018), on the graph-level or node-level representations ($z_{i,t}$ or a pooled graph embedding $z_{graph, t}$). For node $i$ with representation $z_i$, let $z_i^+$ be its positive counterpart and $\{z_{k}^-\}_{k=1}^K$ be negative samples:
    $$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\exp(\text{sim}(z_i, z_i^+) / \tau) + \sum_{k=1}^K \exp(\text{sim}(z_i, z_{k}^-) / \tau)}$$
    where $\text{sim}(\cdot, \cdot)$ is a similarity function (e.g., cosine similarity) and $\tau$ is a temperature hyperparameter. This encourages the model to learn representations invariant to superficial changes but sensitive to meaningful scene differences.

**3.6 Experimental Design**

*   **Baselines:**
    *   *Modular Approach:* State-of-the-art detector (e.g., CenterPoint) + tracker (e.g., AB3DMOT) + predictor (e.g., Trajectron++, LaneGCN).
    *   *Unified Grid-based:* BEVFormer, FIERY.
    *   *Unified Vector-based:* VAD (Jiang et al., 2023).
    *   *Graph-based (Prediction Focused):* HDGT (Jia et al., 2022), Social-STGCNN (Sadeghian et al., 2020) adapted to handle heterogeneous agents and map data.
*   **Ablation Studies:**
    *   Impact of hierarchy (HSG vs. flat SG).
    *   Impact of adaptive edge weights (attention vs. fixed/distance-based weights).
    *   Contribution of static map nodes.
    *   Effectiveness of joint perception-prediction vs. prediction-only.
    *   Impact of self-supervised pre-training/auxiliary loss.
    *   Sensitivity to different sensor modalities (LiDAR-only, Camera-only, Fusion).
*   **Evaluation Metrics:**
    *   **Perception:** If refinement is performed: Improvement in mAP, MOTA, IDF1 over the initial detector/tracker.
    *   **Prediction:** Standard trajectory forecasting metrics:
        *   Average Displacement Error (ADE): Mean L2 distance over all predicted timesteps.
        *   Final Displacement Error (FDE): L2 distance at the final prediction timestep.
        *   Miss Rate (MR): Percentage of predictions where FDE exceeds a threshold (e.g., 2 meters).
        *   For probabilistic methods: minADE$_k$/minFDE$_k$ (minimum error over k samples), NLL.
    *   **Computational Cost:** Training time, inference latency (FPS), model parameter count. Comparison will highlight trade-offs between accuracy and efficiency.
*   **Qualitative Analysis:** Visualize predicted trajectories overlaid on sensor data and map context, particularly in complex interaction scenarios (e.g., intersections, merges, cut-ins) to assess the model's understanding of scene dynamics. Analyze failure cases to identify remaining challenges.

## **4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **State-of-the-Art Performance:** We expect the HSG-Net to outperform baseline methods, particularly the modular approaches and non-integrated graph models, on standard prediction benchmarks (ADE, FDE, MR) on datasets like nuScenes and Argoverse 2, especially in scenarios with dense interactions. We anticipate improvements in the range of 10-20% in key metrics like FDE compared to strong graph-based prediction baselines.
    2.  **Improved Robustness and Generalization:** The self-supervised contrastive learning component is expected to yield representations that generalize better to novel scenarios and variations in sensor input, measurable through cross-dataset evaluation or evaluation on challenging long-tail distributions within datasets.
    3.  **Effective Fusion Demonstration:** The framework will demonstrate the effective integration of multi-modal sensor data (LiDAR, camera) and HD map information within the unified graph structure, quantified through ablation studies.
    4.  **Validation of Hierarchical Structure:** Ablation studies will provide empirical evidence for the benefits (or lack thereof) of the proposed hierarchical structure compared to flat graph representations for scene understanding tasks.
    5.  **Quantifiable Benefits of Joint Learning:** We expect to show measurable improvements stemming from the joint perception-prediction formulation compared to sequential or independent task execution.
    6.  **Computational Characterization:** A clear analysis of the computational trade-offs (latency, memory) associated with the HSG compared to grid-based and simpler graph methods. While potentially more complex than prediction-only graphs, the unified nature might offer efficiencies over separate large models for each task.

*   **Impact:**
    1.  **Academic Contribution:** This research introduces a novel, structured representation (HSG) bridging the gap between dense grid-based methods and sparse vectorized/graph-based approaches for AD scene understanding. It provides a principled framework for integrating static and dynamic elements and modeling their interactions hierarchically and temporally.
    2.  **Practical Advancements in AD:** By improving the accuracy and robustness of scene perception and prediction, particularly concerning interactions, this work can directly contribute to the development of safer and more reliable autonomous driving systems. The explicit interaction modeling can lead to more predictable and interpretable behavior.
    3.  **Foundation for End-to-End Systems:** The HSG can serve as a powerful intermediate representation for more integrated or end-to-end driving models, facilitating information flow from perception to planning by providing a rich, structured summary of the scene dynamics.
    4.  **Reduced Data Dependency:** The successful application of self-supervised learning can potentially reduce the reliance on vast amounts of meticulously labeled data for training robust AD models.
    5.  **Informing Future Research:** The methodologies and findings can inform future research in representation learning for dynamic systems, multi-agent systems, and robotics beyond autonomous driving.

## **5. References**

*(Includes references from the provided literature review and standard methods mentioned)*

1.  Barshan, E., et al. (2020). Value-Based Off-Policy Evaluation Method of Autonomous Driving Policies. *arXiv:2010.04130*.
2.  Caesar, H., et al. (2020). nuScenes: A multimodal dataset for autonomous driving. *CVPR*.
3.  Carion, N., et al. (2020). End-to-End Object Detection with Transformers. *ECCV*.
4.  Ivanovic, B., & Pavone, M. (2020). Graph Neural Networks for Modeling Traffic Participant Interactions. *arXiv:2005.06136*.
5.  Ji, S., et al. (2021). Graph pooling methods in graph neural networks—A survey. *arXiv:2107.08805*.
6.  Jia, X., et al. (2022). HDGT: Heterogeneous Driving Graph Transformer for Multi-Agent Trajectory Prediction via Scene Encoding. *arXiv:2205.09753*.
7.  Jiang, B., et al. (2023). VAD: Vectorized Scene Representation for Efficient Autonomous Driving. *arXiv:2303.12077*.
8.  Liang, M., et al. (2020). Learning Lane Graph Representations for Motion Forecasting. *ECCV*.
9.  Li, B., et al. (2024). UniScene: Unified Occupancy-centric Driving Scene Generation. *arXiv:2412.05435* (Note: Placeholder arXiv ID assumed fictitious based on date). *[Adjust if real ID is known]*
10. Min, C., et al. (2023). UniScene: Multi-Camera Unified Pre-training via 3D Scene Reconstruction for Autonomous Driving. *arXiv:2305.18829*.
11. Oord, A. V. D., et al. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*.
12. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. *ECCV*.
13. Qi, C. R., et al. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS*.
14. Sadeghian, A., et al. (2020). Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction. *CVPR*.
15. Salzmann, T., et al. (2020). Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data. *ECCV*.
16. Song, Y., et al. (2020). Spatio-Temporal Graph Neural Networks for Pedestrian Trajectory Prediction. *arXiv:2005.08514*.
17. Tang, Y. C., et al. (2019). Graph-Based Trajectory Prediction for Autonomous Driving. *arXiv:1912.08233*.
18. Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.
19. Ying, R., et al. (2018). Hierarchical Graph Representation Learning with Differentiable Pooling. *NeurIPS*.
20. Zhao, P., et al. (2019). STGAT: Spatio-Temporal Graph Attention Networks for Pedestrian Trajectory Prediction. *arXiv:1904.09439*.
21. Zhou, Y., & Tuzel, O. (2018). VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. *CVPR*.

---