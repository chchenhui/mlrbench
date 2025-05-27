Title  
Hierarchical Spatiotemporal Graphs for Unified Scene Representation in Autonomous Driving  

Introduction  
Background  
Autonomous driving systems today are built upon a pipeline of perception, prediction, planning, and control modules. While each module has seen significant advances driven by machine learning—e.g., object detection, semantic segmentation, trajectory forecasting—these components often operate on separate intermediate representations. Static maps and dynamic agent tracks are processed in isolation, leading to error propagation and inefficient use of information. For example, a mis‐detected vehicle in the perception stage can degrade downstream trajectory predictions; conversely, knowledge of likely trajectories could reinforce perception in ambiguous situations (e.g., occlusions). Moreover, existing representations frequently struggle to capture the rich interactions among road infrastructure, traffic participants, and environmental context, limiting both safety and generalization in complex urban scenes.  

Research Objectives  
This proposal aims to develop a unified scene representation based on a hierarchical spatiotemporal graph (HSTG) that tightly integrates static infrastructure and dynamic actors in a single, learnable structure. The core objectives are:  
1. Define a graph topology that encodes static elements (lanes, curbs, traffic signs) and dynamic agents (vehicles, pedestrians, cyclists) in a shared geometric frame.  
2. Design a dynamic graph neural network (GNN) architecture combining spatial message passing, temporal convolution, and self‐supervised contrastive learning to jointly optimize perception, scene‐flow estimation, and multi‐horizon trajectory prediction.  
3. Develop multi‐sensor fusion algorithms to embed 3D LiDAR, multi‐view camera imagery, and inertial measurements into node and edge features.  
4. Validate the approach on large‐scale driving benchmarks, comparing against state‐of‐the‐art modular and end‐to‐end methods in terms of accuracy, robustness, and runtime performance.  

Significance  
A unified hierarchical spatiotemporal graph representation (HSTG) promises several key benefits:  
• Error mitigation: By allowing feedback between prediction and perception sub‐modules, the system can correct spurious detections and refine forecasts.  
• Rich interaction modeling: Explicit edges among static and dynamic nodes capture scene context (e.g., pedestrian crosswalks) and inter‐agent influence (e.g., yielding behavior).  
• Data efficiency: Self‐supervised pre‐training on graph structure and dynamics reduces dependence on expensive human annotations.  
• Safety and interpretability: A graph‐based representation is more amenable to formal verification and human inspection than large monolithic neural networks.  

Methodology  
Overview  
Our method constructs, at each discrete time step $t$, a graph $G_t=(V^s\cup V^d,E_t,A_t)$, where $V^s$ are static nodes (road elements), $V^d$ are dynamic nodes (agents), $E_t$ are edges encoding pairwise relations, and $A_t$ denotes edge attributes (e.g., relative distance, semantic compatibility). We then process the sequence $\{G_{t-T+1},\dots,G_t\}$ with a dynamic GNN that interleaves spatial graph attention and temporal convolutional layers. A self‐supervised contrastive loss over graph embeddings encourages invariance to sensor noise and scene perturbations. Finally, task heads decode node embeddings for detection, segmentation, scene‐flow, and trajectory prediction.  

Data Collection and Preprocessing  
• Datasets: We will use nuScenes, Waymo Open Dataset, and Argoverse 2. Each provides synchronized 3D LiDAR point clouds, multi‐view camera images, and inertial measurements with ground‐truth labels for detection and tracking.  
• Static Node Extraction: From HD‐map inputs or LiDAR‐based map reconstruction, we extract lanes (as polylines), crosswalks, traffic signals, and road boundaries. Each such element becomes one or more static nodes with geometric and semantic attributes.  
• Dynamic Node Initialization: We run a baseline 3D detection and tracking pipeline (e.g., CenterPoint + Kalman filter) to generate candidate agent tracks; each track at time $t$ yields a dynamic node with attributes including bounding‐box center, velocity, object class, and LiDAR intensity statistics.  
• Edge Construction:  
  – Static–static edges: connect nearby map elements within a radius $r_s$, with attribute $a_{ij}^s$ encoding semantic adjacency (e.g., lane‐lane continuity).  
  – Static–dynamic edges: connect each agent to nearby map nodes within radius $r_{sd}$, capturing which lane or crosswalk an agent occupies.  
  – Dynamic–dynamic edges: connect agents within distance $r_d$, with attribute $a_{ij}^d$ storing relative displacement and velocity difference.  
  All distances are computed in the global metric frame.  

Graph Representation  
At time $t$, each node $i$ has feature vector $x_i^t\in\mathbb{R}^F$ combining:  
• Geometry: 2D/3D position, orientation.  
• Appearance: CNN features extracted from projected camera images.  
• Lidar: point‐cloud occupancy histograms in a local grid.  
• Kinematics: velocity, acceleration (from IMU).  
Edge attributes $a_{ij}^t\in\mathbb{R}^P$ encode relative geometry and semantic masks.  

Dynamic GNN Architecture  
1. Spatial Message Passing (per time step):  
   We apply $L_s$ layers of graph attention networks (GAT). For node $i$, layer $l$:  
   $$ h_i^{(l+1),t} = \sigma\Big(\sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{(l),t} W^{(l)} h_j^{(l),t}\Big), $$  
   where attention coefficients  
   $$ \alpha_{ij}^{(l),t} = \frac{\exp\big(\text{LeakyReLU}(a^\top [W^{(l)} h_i^{(l),t} \,\|\, W^{(l)} h_j^{(l),t}\,\|\, f(a_{ij}^t)])\big)}{\sum_{k\in\mathcal{N}(i)} \exp\big(\dots\big)}, $$  
   $\|\,$ denotes concatenation, and $f(\cdot)$ is a small MLP embedding edge attributes. The initial $h_i^{(0),t}$ is a linear embedding of $x_i^t$.  

2. Temporal Modeling:  
   We gather for each node $i$ the sequence $\{h_i^{(L_s),t-T+1},\dots,h_i^{(L_s),t}\}$ and apply $L_t$ layers of a temporal convolutional network (TCN) or a transformer encoder to capture dynamics:  
   $$ z_i^{t} = \text{TCN}\big(h_i^{(L_s),t-T+1:t}\big)\in\mathbb{R}^D. $$  

3. Self‐Supervised Contrastive Pre‐training:  
   We generate two augmented views of the same sequence by random edge dropping, sensor noise injection, or small spatial perturbations. The graph‐level embeddings $g^1,g^2$ are obtained by pooling node codes $z_i^t$. We minimize the NT‐Xent loss:  
   $$ \ell = -\log \frac{\exp(\mathrm{sim}(g^1,g^2)/\tau)}{\sum_{k}\exp(\mathrm{sim}(g^1,g^k)/\tau)}, $$  
   where $\mathrm{sim}(u,v)=u^\top v/\|u\|\|v\|$ and $\tau$ is a temperature.  

4. Task Heads and Joint Training:  
   We append lightweight MLP heads to $z_i^t$ for:  
   • Object detection: predict class $\hat y_i$ and bounding‐box offsets.  
   • Semantic segmentation (for static nodes): predict semantic labels.  
   • Scene‐flow estimation: regress per‐node 3D motion vectors.  
   • Trajectory prediction: generate multi‐modal future waypoints $\{\hat p_i^{t+1},\dots,\hat p_i^{t+H}\}$.  
   The overall loss is  
   $$ \mathcal{L} = \lambda_{\mathrm{det}}\mathcal{L}_{\mathrm{det}} + \lambda_{\mathrm{seg}}\mathcal{L}_{\mathrm{seg}} + \lambda_{\mathrm{flow}}\mathcal{L}_{\mathrm{flow}} + \lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{contrast}}. $$  

Implementation Details  
• Hidden dimensions: $F=128$, $W^{(l)}\in\mathbb{R}^{128\times128}$, $D=256$.  
• Layers: $L_s=3$ spatial GAT blocks, $L_t=2$ TCN blocks (kernel size = 3).  
• Optimization: AdamW with initial learning rate $3\mathrm{e}{-4}$, cosine decay over 100 epochs, batch size = 16 sequences.  
• Hardware: 8× NVIDIA A100 GPUs.  

Experimental Design  
Benchmarks and Baselines  
• nuScenes, Waymo, Argoverse 2. Metrics:  
  – Detection: mean Average Precision (mAP).  
  – Segmentation: mean Intersection over Union (mIoU).  
  – Scene flow: End‐Point Error (EPE).  
  – Trajectory forecasting: average/final displacement error (ADE/FDE) over 1–6 s horizons.  
  – Runtime: frames per second (FPS) on commodity hardware.  
• Baselines: CenterPoint + VectorNet, HDGT, VAD, UniScene variants.  

Ablation Studies  
1. Without self‐supervised pre‐training.  
2. Static vs. dynamic graph components only.  
3. Graph attention vs. plain GCN.  
4. TCN vs. transformer for temporal modeling.  
5. Single‐task vs. multi‐task losses.  

Robustness and Generalization  
• Edge Cases: night‐time, heavy occlusion, rare maneuvers (e.g., U‐turns).  
• Cross‐city generalization: train on Waymo SF, test on Miami.  

Real‐Time Evaluation  
Integrate the model into a ROS‐based stack and test inference on NVIDIA DRIVE AGX, measuring latency and CPU/GPU utilization.  

Expected Outcomes & Impact  
We anticipate that HSTG will outperform current state‐of‐the‐art by:  
• Improving 3D detection mAP by 3–5% through joint perception–prediction feedback.  
• Reducing ADE/FDE by 10–15% on long‐term trajectory horizons (4–6 s).  
• Achieving real‐time inference (≥ 15 FPS) with optimized graph batching.  
• Demonstrating 20–30% reduction in labeled data requirements via self‐supervised pre‐training.  

Broader Impacts  
• Safety: Explicit modeling of agent interactions supports formally verifiable planning and safer decisions in dense traffic.  
• Interpretability: Graph nodes and edges offer human‐readable diagnostics (e.g., which lane influenced a prediction).  
• Generalization: The self‐supervised and multi‐modal nature improves robustness to new cities, weather, and sensor noise.  
• Research and Industry Adoption: The unified HSTG framework can serve as a drop‐in intermediate representation for end‐to‐end driving systems, simulators (CARLA, LGSVL), and digital twins, promoting rapid prototyping and deployment of next‐generation autonomous vehicles.  

Timeline (12 months)  
Month 1–3: Data preparation, static/dynamic graph construction.  
Month 4–6: Implement spatial GAT and temporal modules; initial supervised training.  
Month 7–8: Develop self‐supervised pre‐training pipeline and ablations.  
Month 9–10: Comprehensive benchmarking and real‐time integration.  
Month 11–12: Paper writing, open‐source release of code and pre‐trained models.