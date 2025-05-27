Okay, here is a research proposal based on the provided task description, idea, and literature review.

---

## **Research Proposal**

**1. Title:** Geometric Temporal Message Passing on Manifolds for Dynamic Graph Learning

**2. Introduction**

*   **Background:**
    Dynamic graphs, where both node attributes and the underlying network topology evolve over time, are ubiquitous in modeling real-world systems. Examples span diverse domains, including social interaction networks (You et al., 2022), communication systems, biological networks, transportation systems (Pareja et al., 2019), and financial transactions. Effectively learning from these dynamic structures is crucial for tasks like predicting future interactions, forecasting system states (e.g., traffic flow), detecting anomalies, and understanding the evolution of complex phenomena.

    Machine learning, particularly Graph Neural Networks (GNNs), has shown significant promise in learning representations from static graphs. However, extending these models to dynamic settings presents substantial challenges (Garg et al., 2020). The temporal dimension introduces complexities related to capturing evolving patterns, handling non-stationarity, maintaining computational efficiency as the graph changes, and modeling the intricate interplay between structural and feature dynamics.

    Existing approaches to dynamic graph learning often adapt static GNN architectures. Methods like EvolveGCN (Pareja et al., 2019) use recurrent networks to update GNN parameters over time. Others, such as TGN (Rossi et al., 2020) and ROLAND (You et al., 2022), employ memory modules or hierarchical state updates to incorporate temporal information. Continuous-time models like GN-CDE (Qin et al., 2023) leverage neural differential equations to model evolution. While these methods have advanced the field, they predominantly operate within Euclidean spaces or implicitly model temporal dependencies. They often struggle to explicitly capture and leverage the intrinsic geometric structures that may underlie the observed dynamics, potentially limiting their ability to model complex, non-linear evolution patterns and generalize effectively, especially over longer time horizons or under significant structural shifts.

    The Topology, Algebra, and Geometry in Machine Learning (TAG-ML) workshop highlights the growing potential of mathematical tools to address challenges in machine learning, particularly for high-dimensional, complex data. Geometric deep learning has already demonstrated benefits for static graphs by incorporating symmetries and structural priors. We posit that the principles of differential geometry and Riemannian manifold theory offer a powerful, yet underexplored, avenue for understanding and modeling the *dynamics* of graphs. The evolution of a graph over time can be conceptualized as a trajectory through a space of possible graph structures or embeddings. If this underlying space possesses non-Euclidean geometry (e.g., hyperbolic spaces for hierarchies, spaces of positive definite matrices for covariance structures), modeling the dynamics directly on the relevant manifold, respecting its geometric properties, could unlock significant improvements in modeling fidelity and predictive power.

*   **Research Gap and Proposed Idea:**
    Current dynamic graph learning frameworks lack a principled mechanism to explicitly model graph evolution as movement on a potentially curved geometric manifold. They typically rely on Euclidean assumptions for aggregating temporal information or evolving representations, which may not be optimal if the underlying data generating process has inherent geometric constraints or symmetries.

    This proposal introduces **Geometric Temporal Message Passing (GeoDyMP)**, a novel framework for dynamic graph learning that explicitly leverages Riemannian geometry. The core idea is to represent the state of the graph (or its nodes) at each time step as a point on a suitable Riemannian manifold $\mathcal{M}$. The evolution from one time step to the next is then viewed as movement along a path (approximated by a geodesic segment) on this manifold. To enable effective learning in this framework, we propose specialized neural message passing operations designed to function intrinsically on the manifold structure:
    1.  **Tangent Space Computation:** Since manifolds are only locally Euclidean, complex computations like feature aggregation and transformations are performed in the tangent spaces at specific points. This requires mapping between the manifold and its tangent spaces using logarithmic ($Log$) and exponential ($Exp$) maps.
    2.  **Parallel Transport:** To combine information across different time steps (i.e., different points on the manifold), we need to transport feature vectors (residing in tangent spaces) along the path connecting these points without distorting their geometric meaning. Parallel transport ($\Gamma$) provides a principled way to achieve this, ensuring geometric consistency.
    3.  **Geodesic Attention:** We propose an attention mechanism where the relevance between graph states (or nodes) across time is measured using the geodesic distance ($d_{\mathcal{M}}$) on the manifold, offering a more natural way to capture temporal dependencies in curved spaces compared to Euclidean distance.
    4.  **Curvature-Aware Aggregation:** The local curvature of the manifold influences how distances and volumes behave. We will explore incorporating manifold curvature ($K$) into the aggregation functions to allow the model to adapt its behavior based on the local geometry (e.g., adjusting aggregation weights based on local expansion or contraction).

*   **Research Objectives:**
    1.  **Develop the GeoDyMP Framework:** Formalize the mathematical foundations of representing dynamic graphs as manifold trajectories and define the core operations: tangent space projection, parallel transport for temporal propagation, geodesic attention, and curvature-aware aggregation.
    2.  **Implement GeoDyMP:** Instantiate the framework using specific choices of manifolds (e.g., Hyperbolic, Stiefel, Symmetric Positive Definite (SPD) matrices) and implement the geometric operations efficiently, potentially leveraging existing geometric deep learning libraries.
    3.  **Validate GeoDyMP Experimentally:** Evaluate the proposed model on diverse benchmark dynamic graph datasets across various tasks (e.g., dynamic node classification, link prediction, traffic forecasting). Compare performance against state-of-the-art dynamic GNN baselines identified in the literature review.
    4.  **Analyze Geometric Contributions:** Conduct thorough ablation studies to quantify the performance gains attributable to each geometric component (parallel transport, geodesic attention, curvature). Investigate the interpretability of the learned geometric representations and dynamics.
    5.  **Explore Manifold Learning:** Investigate methods for adaptively learning the underlying manifold structure from data, rather than assuming a fixed geometry a priori.

*   **Significance:**
    This research promises several significant contributions. Firstly, it introduces a fundamentally new perspective on dynamic graph learning by explicitly incorporating Riemannian geometry, bridging the gap between geometric deep learning and temporal graph analysis. Secondly, by respecting the intrinsic geometry of the data evolution, GeoDyMP is expected to achieve superior performance and generalization, particularly for graphs with complex, non-linear dynamics or inherent hierarchical/structural properties best captured by non-Euclidean spaces. Thirdly, the framework offers enhanced interpretability by potentially revealing underlying geometric structures governing the system's evolution. This work directly aligns with the goals of the TAG-ML workshop by applying advanced geometric concepts (manifolds, tangent spaces, geodesics, parallel transport, curvature) to tackle challenging machine learning problems, potentially leading to novel algorithms and deeper understanding. Success in this research could pave the way for more robust and accurate modeling of dynamic systems across science and engineering.

**3. Methodology**

*   **Problem Formulation:**
    Let $\mathcal{G} = \{G_1, G_2, ..., G_T\}$ be a sequence of graph snapshots, where each $G_t = (V_t, E_t, X_t)$ consists of a set of nodes $V_t$, edges $E_t \subseteq V_t \times V_t$, and node features $X_t \in \mathbb{R}^{|V_t| \times d}$. The sets $V_t$ and $E_t$ may change over time. The goal is typically to predict future graph properties, such as node labels $Y_{t+1}$, edge existence $E_{t+1}$, or future node features $X_{t+1}$, based on the history $\mathcal{G}_{\leq t}$.

*   **Geometric Representation:**
    We hypothesize that the essential dynamics of the graph sequence can be effectively captured by representing each snapshot $G_t$ (or node states within it) as a point $p_t$ on a chosen $n$-dimensional Riemannian manifold $(\mathcal{M}, g)$, where $g$ is the Riemannian metric. The choice of $\mathcal{M}$ can be data-dependent or task-dependent. Potential choices include:
    *   **Hyperbolic space ($\mathbb{H}^n$):** Suitable for hierarchical or scale-free structures. Possesses constant negative curvature.
    *   **Sphere ($\mathbb{S}^n$):** Suitable for data with bounded norm or directional characteristics. Constant positive curvature.
    *   **Manifold of Symmetric Positive Definite (SPD) matrices ($Sym_d^+$):** Useful for representing covariance structures or ellipsoid-based representations. Variable curvature.
    *   **Grassmannian manifold ($\mathcal{G}(k, d)$):** Space of $k$-dimensional subspaces in $\mathbb{R}^d$, useful for representing evolving subspaces.
    *   **Product Manifolds:** Combinations of simpler manifolds.
    Initially, we will experiment with fixed, standard manifolds like $\mathbb{H}^n$ and $Sym_d^+$. An encoder network $f_{enc}: G_t \mapsto p_t \in \mathcal{M}$ will map graph snapshots (e.g., using a static GNN) to points on the manifold. Node features $h_{v,t}$ will be represented as vectors in the tangent space $T_{p_t}\mathcal{M}$ at the corresponding graph embedding $p_t$.

*   **GeoDyMP Algorithmic Steps:**
    The core of GeoDyMP involves updating node representations $h_{v,t} \in T_{p_t}\mathcal{M}$ (or potentially Euclidean representations influenced by the manifold structure) over time. Let $p_t = f_{enc}(G_t)$. The process for computing representations at time $t$ involves:

    1.  **Intra-time Message Passing (Snapshot $t$):** Perform message passing within the current snapshot $G_t$ to capture instantaneous structural information. This can use standard GNN layers (e.g., GCN, GAT) adapted for tangent spaces or incorporate curvature:
        $$ m_{v,t}^{(l)} = \text{AGGREGATE}^{(l)} \left( \left\{ \phi^{(l)}(h_{u,t}^{(l-1)}, h_{v,t}^{(l-1)}, e_{uv,t}, K(p_t)) \mid u \in \mathcal{N}(v) \cup \{v\} \right\} \right) $$
        $$ h_{v,t}^{(l)} = \sigma \left( W^{(l)} m_{v,t}^{(l)} \right) $$
        where $h_{v,t}^{(0)}$ are initial node features mapped to $T_{p_t}\mathcal{M}$, $\phi^{(l)}$ is a message function (potentially curvature-aware $K(p_t)$), AGGREGATE$^{(l)}$ is an aggregation function (e.g., sum, mean, max), $W^{(l)}$ is a learnable weight matrix (acting linearly within the tangent space), and $\sigma$ is a non-linearity (potentially manifold-aware, like point-wise application after mapping back to the manifold and then projecting to the next tangent space, though operating linearly in the tangent space is simpler). For simplicity, we can start with standard GNN operations assuming features reside in a common Euclidean space but are influenced by the manifold positions $p_t$ during temporal aggregation. A more advanced version would perform these operations directly within tangent spaces, requiring mapping features between tangent spaces of neighbors if using manifold-based node embeddings.

    2.  **Inter-time Information Propagation (from $t-1$ to $t$):** Propagate relevant information from the previous time step $t-1$ to the current time step $t$. This requires transporting representations from $T_{p_{t-1}}\mathcal{M}$ to $T_{p_t}\mathcal{M}$.
        *   Map the aggregated representation from time $t-1$, let's call it $h_{v, t-1}^{agg}$ (e.g., the final layer output $h_{v,t-1}^{(L)}$), from $T_{p_{t-1}}\mathcal{M}$ to $T_{p_t}\mathcal{M}$ using parallel transport $\Gamma_{p_{t-1} \to p_t}$ along the geodesic connecting $p_{t-1}$ and $p_t$:
            $$ \tilde{h}_{v, t-1 \to t} = \Gamma_{p_{t-1} \to p_t}(h_{v, t-1}^{agg}) $$
        *   Parallel transport preserves the geometric properties (inner products, norms) of tangent vectors when moved along curves on the manifold. It can be implemented numerically (e.g., Schild's ladder) or using closed-form solutions if available for the specific manifold and geodesic. $\tilde{h}_{v, t-1 \to t}$ now resides in $T_{p_t}\mathcal{M}$.

    3.  **Temporal Aggregation with Geodesic Attention:** Combine the intra-time aggregated information $h_{v,t}^{(L)}$ with the transported information from previous steps using an attention mechanism based on geodesic distances. We can attend over a history of transported states $\{\tilde{h}_{v, t' \to t} \mid t' \in \{t-k, ..., t-1\}\}$.
        *   Compute attention scores based on geodesic distance $d_{\mathcal{M}}(p_{t'}, p_t)$:
            $$ \alpha_{v, t', t} = \text{softmax}_{t' \in \text{hist}} \left( \frac{(W_Q h_{v,t}^{(L)})^T (W_K \tilde{h}_{v, t' \to t})}{ \sqrt{d_k} } \cdot f(d_{\mathcal{M}}(p_{t'}, p_t)) \right) $$
            where $W_Q, W_K$ are learnable query/key matrices, $d_k$ is the dimension, and $f$ is a function penalizing large geodesic distances, e.g., $f(d) = \exp(-d^2 / \tau)$ or $f(d)=1/(1+d)$.
        *   Compute the temporally aggregated context vector $c_{v,t} \in T_{p_t}\mathcal{M}$:
            $$ c_{v,t} = \sum_{t' \in \{t-k, ..., t-1\}} \alpha_{v, t', t} (W_V \tilde{h}_{v, t' \to t}) $$
            where $W_V$ projects the transported value vectors.

    4.  **Final Representation Update:** Combine the current snapshot's information with the temporal context. This can be done via concatenation followed by a linear layer, a gating mechanism (like GRU or LSTM adaptations), or simple addition, all within the tangent space $T_{p_t}\mathcal{M}$ or mapped back to Euclidean space.
        $$ h_{v,t}^{final} = \text{UPDATE}(h_{v,t}^{(L)}, c_{v,t}) $$
        where UPDATE is a function like addition, concatenation followed by MLP, or a recurrent gate.

    5.  **Output Layer:** A final layer (e.g., MLP) maps $h_{v,t}^{final}$ to the desired output (node labels, link probabilities, forecasted features). If $h_{v,t}^{final}$ is in $T_{p_t}\mathcal{M}$, it might be mapped back to Euclidean space first using $Exp_{p_t}$ or a learned linear map.

*   **Mathematical Formalism:**
    *   **Riemannian Manifold $(\mathcal{M}, g)$:** A smooth manifold with a smoothly varying inner product $g_p: T_p\mathcal{M} \times T_p\mathcal{M} \to \mathbb{R}$ on each tangent space $T_p\mathcal{M}$.
    *   **Exponential Map $Exp_p: T_p\mathcal{M} \to \mathcal{M}$:** Maps a tangent vector $v \in T_p\mathcal{M}$ to a point $q \in \mathcal{M}$ by moving along the geodesic starting at $p$ in the direction $v$ for unit time, such that $q = \gamma(1)$ where $\gamma(0)=p, \dot{\gamma}(0)=v$.
    *   **Logarithmic Map $Log_p: \mathcal{M} \to T_p\mathcal{M}$:** The inverse of the exponential map (locally defined). $Log_p(q)$ gives the tangent vector $v$ at $p$ such that $Exp_p(v) = q$.
    *   **Geodesic Distance $d_{\mathcal{M}}(p, q)$:** The length of the shortest path (geodesic) between $p$ and $q$. In many cases, $d_{\mathcal{M}}(p, q) = ||Log_p(q)||_{g_p} = ||Log_q(p)||_{g_q}$.
    *   **Parallel Transport $\Gamma_{p \to q}: T_p\mathcal{M} \to T_q\mathcal{M}$:** Transports a vector $v \in T_p\mathcal{M}$ along the geodesic from $p$ to $q$ to obtain a vector $w \in T_q\mathcal{M}$ such that $w$ is "parallel" to $v$ along the path. Defined via the connection/covariant derivative.
    *   **Curvature:** Intrinsic property of the manifold (e.g., Ricci scalar $K(p)$) describing how geometry deviates from Euclidean space locally. Can be incorporated by making parameters of aggregation or update functions dependent on $K(p_t)$. For instance, weights in aggregation could be scaled based on local volume changes indicated by curvature.

*   **Experimental Design:**
    *   **Datasets:**
        *   *Traffic Forecasting:* METR-LA, PEMS-BAY (continuous features, fixed topology, dynamic weights). Task: Predict future traffic speed.
        *   *Social/Communication Networks:* Enron email dataset, Facebook friendships (discrete time events, evolving topology). Task: Link prediction or dynamic node classification.
        *   *Collaboration Networks:* DBLP (evolving co-authorship). Task: Link prediction.
        *   *Physical Simulation:* N-body simulation dataset where nodes (bodies) move based on forces (edges). Task: Predict future positions/velocities.
    *   **Baselines:**
        *   *Static Methods:* GCN, GAT applied snapshot-by-snapshot.
        *   *Temporal GNNs:* EvolveGCN (Pareja et al., 2019), TGN (Rossi et al., 2020), ROLAND (You et al., 2022), GC-LSTM (sequence models on node embeddings), CTDNE (continuous models), potentially GN-CDE (Qin et al., 2023).
        *   *Non-Geometric Dynamic Models:* DynGEM (Goyal et al., 2018).
    *   **Evaluation Metrics:**
        *   *Traffic Forecasting:* Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE).
        *   *Link Prediction:* Area Under the ROC Curve (AUC), Average Precision (AP).
        *   *Node Classification:* Accuracy, F1-score (micro/macro).
    *   **Implementation Details:** Utilize libraries like PyTorch Geometric or PyTorch Geometric Temporal, potentially integrating with Geoopt (Kochurov et al.) or Hypyr (library for hyperbolic embeddings) for manifold operations. Implement parallel transport numerically or use closed-form solutions where available.
    *   **Ablation Studies:**
        1.  *Effect of Geometry:* Compare full GeoDyMP against a Euclidean variant (replace manifold operations with standard Euclidean counterparts, e.g., parallel transport with identity, geodesic distance with Euclidean distance).
        2.  *Impact of Parallel Transport:* Replace $\Gamma_{p_{t-1} \to p_t}$ with a simple linear transformation or identity mapping.
        3.  *Impact of Geodesic Attention:* Replace geodesic distance in attention with Euclidean distance or simple temporal distance ($|t-t'|$).
        4.  *Impact of Curvature Awareness:* Remove curvature modulation from aggregation/update functions.
        5.  *Impact of Manifold Choice:* Compare performance using different manifolds (e.g., Hyperbolic vs. Euclidean vs. SPD).
    *   **Hyperparameter Tuning:** Use standard techniques like grid search or Bayesian optimization (e.g., using Optuna) on a validation set to tune learning rates, embedding dimensions, manifold parameters (e.g., curvature if learnable), attention heads, historical window size $k$, etc.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A Novel Framework (GeoDyMP):** A fully developed and formalized framework for dynamic graph learning based on Riemannian geometry, including the core components of tangent space computation, parallel transport, geodesic attention, and optional curvature awareness.
    2.  **High-Performance Model:** An implementation of GeoDyMP demonstrating state-of-the-art or comparable performance against leading baselines on diverse dynamic graph benchmark datasets across various tasks.
    3.  **Validated Geometric Contributions:** Clear empirical evidence from ablation studies quantifying the benefits of each geometric component, demonstrating the value of the manifold perspective.
    4.  **Geometric Insights:** Potential for new insights into the underlying structure of dynamic graph data by analyzing the learned manifold representations, the trajectories of graph embeddings on the manifold, and the role of curvature. This could involve visualizations of trajectories or learned manifold properties.
    5.  **Open-Source Implementation:** Release of the GeoDyMP codebase to facilitate reproducibility and further research by the community.

*   **Impact:**
    1.  **Scientific Advancement:** This research will significantly advance the field of dynamic graph learning by providing a new class of models grounded in differential geometry. It will open up new avenues for incorporating geometric priors into temporal machine learning models, potentially influencing related areas like time-series analysis on manifolds or geometric reinforcement learning. It directly contributes to the TAG-ML community's goal of leveraging geometric tools for ML challenges.
    2.  **Improved Practical Applications:** By potentially offering more accurate and robust predictions on dynamic graph data, GeoDyMP could lead to tangible improvements in real-world applications such as traffic management systems, recommender systems adapting to changing user preferences, epidemic modeling, and financial fraud detection.
    3.  **Enhanced Understanding and Interpretability:** The geometric perspective may offer a more intuitive understanding of complex system dynamics compared to black-box Euclidean models. Analyzing the learned geometry and dynamics on the manifold could provide interpretable insights into how systems evolve.
    4.  **Foundation for Future Work:** This work could serve as a foundation for future research exploring adaptive manifold learning for dynamic graphs, geometric control theory applications in graph dynamics, and extensions to continuous-time geometric models on manifolds.

This research directly addresses the need for tools that help structure and understand complex, high-dimensional dynamic data, perfectly aligning with the themes of the TAG-ML workshop. By rigorously integrating concepts from differential geometry with message passing neural networks, we aim to develop a powerful new tool for analyzing and predicting the behavior of evolving systems represented as dynamic graphs.

---