Title: Geometric Message Passing Neural Networks on Riemannian Manifolds for Dynamic Graph Learning

1. Introduction  
Background  
Dynamic graphs arise in many real-world systems—traffic networks, social interaction graphs, biological networks—where both node attributes and edge structure evolve over time. Classical static Graph Neural Networks (GNNs) and their recurrent extensions (e.g., EvolveGCN, ROLAND, TGNs) capture some temporal dependencies but treat each snapshot or embedding update in Euclidean space. As graph topology and feature geometry become more complex and highly non-linear, Euclidean embeddings lose the ability to represent curvature, global structure, and constraints that naturally govern evolution. Differential geometry and Riemannian manifold theory offer mature mathematical tools to model curved spaces, parallel transport, geodesics, and curvature—concepts that can provide richer representations of temporal graph trajectories than standard vector spaces.

Research Objectives  
This proposal aims to develop a novel Geometric Message Passing framework for dynamic graph learning that:  
1. Represents each temporal snapshot as a point on a chosen Riemannian manifold \( (\mathcal{M}, g) \).  
2. Operates on tangent spaces via logarithmic/exponential maps and parallel transport to respect manifold geometry when aggregating messages across nodes and time.  
3. Introduces geodesic attention mechanisms to capture long-range temporal and structural dependencies using manifold distances.  
4. Develops curvature-aware aggregation functions that adapt to local geometric properties.  
5. Validates the framework on real-world dynamic graph tasks (traffic forecasting, epidemic spread, social link prediction), benchmarking against state-of-the-art.

Significance  
By integrating Riemannian geometry into dynamic graph learning, we expect to:  
• Enhance representation power for complex, non-linear graph evolution.  
• Improve generalization to unseen temporal patterns and missing data scenarios.  
• Provide interpretable insights through curvature measures and geodesic paths.  
• Open a new research direction that bridges mathematical machine learning and dynamic graph modeling.

2. Methodology  
Our proposed framework consists of four main components: manifold representation, message passing on tangent spaces, geodesic attention, and curvature-aware aggregation. We detail each component, data collection, algorithmic steps, and experimental design.

2.1 Data Collection and Preprocessing  
We will use multiple publicly available benchmarks:  
• Traffic forecasting: METR-LA, PEMS-BAY datasets. Node features: traffic speed/flow. Graph: fixed road network.  
• Epidemic modeling: simulated SIR processes on evolving contact graphs (e.g., high-resolution mobility data).  
• Social interaction: Dynamic link prediction on temporal social networks (e.g., Reddit threads, Wikipedia edits).  
Preprocessing steps:  
1. Standardize node features to zero mean, unit variance.  
2. Construct time-indexed graph snapshots \(G^t = (V^t, E^t, X^t)\).  
3. Select a Riemannian manifold \(\mathcal{M}\): candidates include the manifold of symmetric positive definite (SPD) matrices for covariance embeddings, or hyperbolic space \(\mathbb{H}^d\) for hierarchical structures.

2.2 Manifold Representation  
At each time \(t\), we embed node \(i\) into \(\mathcal{M}\) as a point \(y_i^t \in \mathcal{M}\). To operate in a vector space, we map \(y_i^t\) to the tangent space \(T_{y^t}\mathcal{M}\) via the logarithmic map
$$
\log_{y^t}(y_i^t) = v_i^t \in T_{y^t}\mathcal{M}.
$$
Conversely, to update embeddings and return to the manifold, we use the exponential map
$$
\exp_{y^t}(v) = y^t +_M v,
$$
where \(\exp\) respects the manifold geometry.

2.3 Geometric Message Passing  
We extend the standard message passing paradigm to manifold settings. For each node \(i\) at time \(t\), we define incoming messages from neighbors \(j\in \mathcal{N}(i)\) via:
1. **Parallel Transport**: Transport tangent vectors from neighbor’s tangent base \(y_j^t\) to base \(y_i^t\):
   $$
   \Gamma_{y_j^t \to y_i^t}(v_j^t) \in T_{y_i^t}\mathcal{M}.
   $$
2. **Edge Message Function**: Compute a message in the tangent space:
   $$
   m_{j\to i}^t = \mathrm{MLP}_m\big(\Gamma_{y_j^t \to y_i^t}(v_j^t), d_{\mathcal{M}}(y_i^t,y_j^t), x_{ij}^t\big),
   $$
   where \(d_{\mathcal{M}}\) is the geodesic distance and \(x_{ij}^t\) optional edge attributes.
3. **Aggregation**: Aggregate messages using curvature-aware weights (Section 2.4):
   $$
   M_i^t = \sum_{j\in \mathcal{N}(i)} w_{ij}^t \, m_{j\to i}^t.
   $$
4. **Node Update**: Update the node tangent representation via a manifold-aware update:
   $$
   v_i^{t+1} = \mathrm{MLP}_u\big(v_i^t \oplus_\mathcal{M} M_i^t\big),
   $$
   where \(\oplus_\mathcal{M}\) denotes manifold addition via exponential and logarithmic maps. Finally, set
   $$
   y_i^{t+1} = \exp_{y_i^t}(v_i^{t+1}).
   $$

2.4 Geodesic Attention Mechanism  
To capture long-range temporal dependencies, we define attention weights based on manifold distances along geodesics. For each pair \((i,t)\) and \((j,s)\) (possibly \(s\neq t\)):
$$
\alpha_{(i,t),(j,s)} = \frac{\exp\big(-\beta\, d_{\mathcal{M}}(y_i^t, y_j^s)\big)}{\sum_{k\,\in\,\mathcal{V},r} \exp\big(-\beta\, d_{\mathcal{M}}(y_i^t, y_k^r)\big)},
$$
with learnable inverse temperature \(\beta\). The message becomes
$$
m_{(j,s)\to (i,t)} = \alpha_{(i,t),(j,s)} \,\Gamma_{y_j^s\to y_i^t}(v_j^s).
$$
We include both spatial neighbors (\(s=t\)) and temporal neighbors (\(j=i\), \(s\neq t\)).

2.5 Curvature-Aware Aggregation  
Local curvature \(\kappa_i^t\) at \(y_i^t\) provides signals on how geometry bends. We estimate discrete Ricci curvature or sectional curvature on the node’s neighborhood, then incorporate it into aggregation weights:
$$
w_{ij}^t = \phi\big(d_{\mathcal{M}}(y_i^t,y_j^t),\, \kappa_i^t\big),
$$
where \(\phi\) is a small MLP that biases aggregation toward regions of high or low curvature.

2.6 Loss Functions and Training  
For forecasting tasks (e.g., traffic speed prediction), we supervise the next-step node features \(\hat{x}_i^{t+1} = g(y_i^{t+1})\) via mean squared error:
$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{|\mathcal{V}|T}\sum_{i,t}\big\| \hat{x}_i^{t+1} - x_i^{t+1}\big\|^2.
$$
For link prediction, we compute affinity scores using geodesic distances and optimize a binary cross-entropy loss. We add regularization
$$
\mathcal{L}_{\mathrm{reg}} = \lambda\,\sum_{i,t} \|\mathrm{Log}_{y_i^t}(y_i^t)\|^2,
$$
to avoid degenerate embeddings. The total loss is \(\mathcal{L}=\mathcal{L}_{\mathrm{task}} + \mathcal{L}_{\mathrm{reg}}\).

2.7 Experimental Design & Evaluation Metrics  
We will conduct comprehensive experiments:  
• Baselines: EvolveGCN, ROLAND, GN-CDE, DynGEM, TGNs.  
• Ablations: remove geodesic attention, curvature module, parallel transport.  
• Metrics:  
  – Forecasting: MAE, RMSE, MAPE.  
  – Link prediction: AUC, AP.  
  – Runtime and memory footprint.  
  – Robustness: performance under missing node features or edges.  

Hyperparameters will be tuned via grid search on validation sets. Each result will be averaged over multiple random seeds.

3. Expected Outcomes & Impact  
Performance Improvements  
We anticipate that our geometric framework will achieve:  
• Lower forecasting error (MAE, RMSE) by 5–15% relative to Euclidean baselines.  
• Higher AUC/AP in dynamic link prediction tasks, especially for long-range temporal links.  
• Improved robustness to missing data due to manifold regularization and curvature smoothing.

Interpretable Insights  
Our use of curvature and geodesic metrics will allow:  
• Visualization of regions of high curvature corresponding to structural changes (e.g., rush-hour traffic anomalies).  
• Analysis of geodesic paths to explain how information propagates in the network over time.  
• Diagnostic tools for detecting abnormal graph evolution via curvature spikes.

Theoretical Contributions  
• Extension of message passing neural networks to Riemannian manifolds, including full algorithmic specification of parallel transport, geodesic attention, and curvature aggregation.  
• Analytical characterization of the expressivity gains afforded by manifold geometry, with potential generalization bounds leveraging manifold Lipschitz continuity.

Broader Impact  
• Traffic and transportation: More accurate and interpretable forecasting models can support congestion management and route planning.  
• Epidemiology: Improved models for disease spread on contact networks can inform public health interventions.  
• Recommendation systems: Better capture of user–item interaction dynamics in social and e-commerce graphs.  
• Mathematical machine learning: Bridges topology, differential geometry, and deep learning, opening pathways for future TAG-ML collaborations.

4. Conclusion  
This proposal outlines a comprehensive research plan to develop Geometric Message Passing Neural Networks for dynamic graphs, leveraging Riemannian manifold tools—parallel transport, geodesic attention, curvature-aware aggregation—to capture complex temporal and structural dependencies. Through rigorous algorithm design, mathematical formulation, and extensive empirical evaluation, we aim to demonstrate significant performance gains, interpretability benefits, and foundational theoretical contributions that advance the integration of topology, algebra, and geometry in machine learning.