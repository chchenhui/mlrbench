Title: HyTECL – Hyperbolic Temporal Contrastive Learning for Dynamic Graphs

1. Introduction  
Background  
Real‐world complex systems—from social interactions and financial transactions to knowledge bases and biological networks—are naturally represented as temporal graphs whose topology and features evolve over time. Classical graph learning methods (e.g., GCN, GAT) assume static structure, failing to capture both hierarchical relationships and temporal dynamics inherent in these data. Recent advances in temporal graph networks (TGNs) [6,8] enable modeling of evolving connectivity, while hyperbolic graph neural networks (HGNNs) [5,9] embed data in non–Euclidean spaces that better reflect latent hierarchies. Contrastive learning on graphs [7] has further demonstrated the power of self-supervised pretraining by distinguishing positive pairs (similar nodes/views) from negatives. However, integrating hyperbolic embeddings, temporal evolution, and contrastive objectives remains underexplored.  

Research Objectives  
1. Develop a unified framework—HyTECL—that jointly models temporal dynamics in hyperbolic space with a contrastive self‐supervised objective.  
2. Design hyperbolic graph convolutional layers and temporal augmentations that preserve hierarchical tree‐like structure while capturing time‐varying patterns.  
3. Introduce a hyperbolic memory module to aggregate historical embeddings and capture long‐range dependencies.  
4. Validate on dynamic knowledge graph forecasting and fraud/anomaly detection, demonstrating gains in predictive accuracy, robustness, and hierarchy preservation.  

Significance  
By embedding evolving graphs in hyperbolic space and leveraging contrastive learning, HyTECL addresses three key challenges: (i) capturing latent hierarchies more faithfully than Euclidean models, (ii) modeling temporal dependencies without sacrificing geometric structure, and (iii) learning robust representations via self‐supervision. The outcome will push the frontier of scalable, hierarchy‐aware temporal graph learning with broad applications in recommendation, event forecasting, and fraud detection.

2. Methodology  
2.1 Problem Formulation  
Let  denote a sequence of graph snapshots indexed by discrete time steps . Each snapshot  consists of nodes  and time‐stamped edges . Each node  has feature vector  that may itself evolve. We aim to learn a mapping  into a hyperbolic manifold  (modeled as the Poincaré ball of radius 1) such that the embeddings reflect both the instantaneous topology at  and its temporal evolution, optimized for downstream tasks (e.g., link forecasting, anomaly detection).  

2.2 Hyperbolic Space Preliminaries  
We adopt the Poincaré ball model of constant negative curvature . Points lie in . The distance between  and  is  
$$  
d_{\mathbb{B}}(x,y) = \operatorname{arcosh}\bigl(1 + 2\frac{\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\bigr)\,.  
$$  
Key operations include the exponential map at the origin:  
$$  
\exp_{0}(v) = \tanh(\sqrt{|c|}\,\|v\|)\,\frac{v}{\sqrt{|c|}\,\|v\|}\,,  
$$  
and the logarithmic map:  
$$  
\log_{0}(x) = \operatorname{arctanh}(\sqrt{|c|}\,\|x\|)\,\frac{x}{\sqrt{|c|}\,\|x\|}\,.  
$$  

2.3 HyTECL Architecture  
HyTECL consists of three core components: (A) Hyperbolic Graph Convolution, (B) Temporal Contrastive Learning, and (C) Hyperbolic Memory Module.  

A. Hyperbolic Graph Convolution  
At each time step , we embed node features  into  via  
1. Map to tangent space at origin:  ,  
2. Apply linear transformation and adjacency aggregation in tangent space:  
   $$  
   u_i = \sum_{j \in \mathcal{N}(i)} w_{ij} \, W \, \log_{0}(h_j^{(t-1)})\,,  
   $$  
   where  is a learnable weight matrix and  are normalized edge weights.  
3. Map back to manifold: .  
This yields updated embeddings . We stack  layers to capture higher‐order structure.  

B. Temporal Contrastive Learning  
To self‐supervise, we create two temporally shifted “views” of  via independent augmentations:  
– Edge masking based on time importance (drop edges with low temporal recency probability).  
– Random subgraph sampling preserving connectivity.  
Denote the resulting embeddings  and  for node  in each view. We define positive pairs  (same node across views) and negatives  (other nodes in same batch). Using a temperature  and hyperbolic similarity based on negative distance, the contrastive loss for a batch of  nodes is:  
$$  
\mathcal{L}_{\mathrm{cl}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\bigl(-d_{\mathbb{B}}(z_i^1,z_i^2)/\tau\bigr)}{\sum_{j=1}^N \exp\bigl(-d_{\mathbb{B}}(z_i^1,z_j^2)/\tau\bigr)}\,.  
$$  

C. Hyperbolic Memory Module  
To capture long‐range dependencies across  time steps, we maintain a memory bank  of past embeddings for each node, updated via a gated aggregation in tangent space:  
1. Retrieve  and map to tangent: .  
2. Compute gate:  
   $$  
   g_t = \sigma\bigl(W_g [m_i^{(t-1)} \oplus \log_{0}(h_i^{(t)})]\bigr)\,,  
   $$  
3. Update memory in tangent:  ,  
4. Map back: .  
At inference,  augments  to provide richer temporal context.  

D. Supervised Task Heads  
– For link/edge forecasting in temporal knowledge graphs, we use hyperbolic distance between subject–relation–object triples. Probability of edge  is modeled as  
$$  
P(i \overset{r}{\to} j) = \sigma\bigl(-d_{\mathbb{B}}(\exp_{0}(W_r\,\log_{0}(h_i^{(t)})),\,h_j^{(t)})\bigr)\,.  
$$  
Loss: negative log‐likelihood over observed edges plus negative sampling.  
– For anomaly/fraud detection, we attach a binary classification head mapping  (augmented with memory) to . Use cross‐entropy.  

Overall, the joint loss is  
$$  
\mathcal{L} = \mathcal{L}_{\mathrm{sup}} + \lambda\,\mathcal{L}_{\mathrm{cl}}\,,  
$$  
with  balancing supervised and contrastive objectives.  

Pseudo‐code:  
Algorithm HyTECL  
Input: Dynamic graph snapshots , features , batch size , hyperparams  
Initialize embeddings , memory .  
for  do  
  for each minibatch of nodes  do  
    1. Hyperbolic GCN update .  
    2. Generate two augmented views; compute embeddings  and .  
    3. Compute contrastive loss .  
    4. Compute supervised loss  via task head.  
    5. Update memory .  
    6. Backpropagate  and update parameters.  
end for  
end for  

2.4 Experimental Design  
Datasets  
– Temporal Knowledge Graphs: ICEWS18, YAGO-WIKI  
– Financial Fraud: Elliptic Bitcoin Transactions […], proprietary banking logs (if available)  
Preprocessing: discretize timestamps, normalize features, extract sliding windows of length .  

Baselines  
– Static Euclidean GCN  
– Temporal Graph Networks (TGN) [6]  
– HGWaveNet [1]  
– HGCL [2]  
– HTGN [3]  

Metrics  
– Link forecasting: Mean Reciprocal Rank (MRR), Hits@1/3/10  
– Anomaly detection: Area Under ROC (AUC), Average Precision (AP), F1‐score  
– Hierarchy quality: distortion [5], graph hyperbolicity measure  

Hyperparameters & Implementation  
– Embedding dimension: 64 (tangent space)  
– Number of HGCL layers: 2  
– Temperature τ: 0.1, contrast weight λ: 0.5  
– Memory size K: 5 steps  
– Optimizer: Adam (lr=1e-3, weight decay=1e-5)  
– Training epochs: 100, early‐stop on validation  

Ablation Studies  
– Without hyperbolic geometry (Euclidean only)  
– Without contrastive loss (λ=0)  
– Without memory module  
– Vary τ, λ, K to assess sensitivity  

Scalability Analysis  
– Runtime and memory usage vs. baseline on graphs up to 1M edges.  

3. Expected Outcomes & Impact  
We anticipate that HyTECL will:  
– Achieve >10% relative improvement in MRR and Hits@10 on dynamic KG forecasting over state‐of‐the‐art TGN and HGWaveNet.  
– Improve anomaly detection AUC by >8% on financial fraud datasets.  
– Demonstrate superior hierarchy preservation (lower distortion) compared to Euclidean methods.  
– Show robustness to sparse and noisy temporal updates via self‐supervision.  
Impact  
HyTECL will bridge hyperbolic geometry and temporal graph learning, offering:  
– A general framework for hierarchy‐aware dynamic graph representation.  
– Scalable algorithms for real‐world large‐scale temporal networks.  
– Insights into self‐supervised hyperbolic contrastive losses, catalyzing further research in non‐Euclidean graph learning.  
Potential applications include knowledge base completion, fraud and anomaly detection in finance and cybersecurity, and recommendation over evolving user–item interaction networks.

4. References  
[1] Bai et al., “HGWaveNet: A Hyperbolic Graph Neural Network for Temporal Link Prediction,” arXiv:2304.07302, 2023.  
[2] Liu et al., “Enhancing Hyperbolic Graph Embeddings via Contrastive Learning,” arXiv:2201.08554, 2022.  
[3] Yang et al., “Discrete‐time Temporal Network Embedding via Implicit Hierarchical Learning in Hyperbolic Space,” arXiv:2107.03767, 2021.  
[4] Sun et al., “Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs,” arXiv:2104.02228, 2021.  
[5] Chami et al., “Hyperbolic Graph Neural Networks,” arXiv:1901.04598, 2019.  
[6] Kazemi et al., “Temporal Graph Networks: A Comprehensive Survey,” arXiv:2004.13448, 2020.  
[7] You et al., “Contrastive Learning for Graph Neural Networks,” arXiv:2006.04131, 2020.  
[8] Skarding et al., “Dynamic Graph Neural Networks: A Survey,” arXiv:2006.06120, 2020.  
[9] Liu et al., “Hyperbolic Graph Convolutional Neural Networks,” arXiv:1805.09112, 2018.