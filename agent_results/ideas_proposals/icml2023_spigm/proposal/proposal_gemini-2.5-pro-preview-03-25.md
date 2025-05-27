## 1. Title:

**Uncertainty-Aware Bayesian Graph Neural Networks: Integrating Aleatoric and Epistemic Uncertainty for Robust Graph-Based Decisions**

## 2. Introduction

**Background:**
Graph Neural Networks (GNNs) have emerged as a powerful paradigm for machine learning on graph-structured data, achieving state-of-the-art results in diverse domains such as molecular science, social network analysis, recommendation systems, and transportation networks (Zhou et al., 2020; Wu et al., 2020). These networks operate by iteratively aggregating information from neighboring nodes, allowing them to learn complex relational patterns and node representations sensitive to the underlying graph topology. However, the vast majority of GNN models provide only point estimates for their predictions. This lack of uncertainty quantification (UQ) poses a significant limitation, particularly in high-stakes applications where the cost of incorrect predictions is substantial. For instance, in drug discovery, misjudging the predicted efficacy or toxicity of a molecule can have severe consequences. Similarly, in financial fraud detection or autonomous driving systems relying on traffic prediction, understanding the confidence associated with a GNN's output is crucial for reliable and safe decision-making.

**Problem Statement:**
The need for reliable UQ in GNNs is increasingly recognized (Wang et al., 2024 - Survey). However, existing approaches often fall short. Many methods treat UQ as a post-hoc addition or rely on computationally expensive techniques like deep ensembles (Lakshminarayanan et al., 2017; Mallick et al., 2022). While ensembles can provide good uncertainty estimates, their training and inference costs can be prohibitive for large graphs. Bayesian Neural Networks (BNNs), including Bayesian GNNs, offer a principled framework for capturing uncertainty, typically using Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016) or variational inference (VI) approaches. Yet, standard applications often only place priors over weights, neglecting uncertainty inherent in the message-passing process itself. Furthermore, methods struggle to reliably distinguish between *aleatoric uncertainty* (inherent noise or randomness in the data) and *epistemic uncertainty* (uncertainty due to limited knowledge or gaps in the model) (Kendall & Gal, 2017). This distinction is vital, as epistemic uncertainty can potentially be reduced with more data or a better model, while aleatoric uncertainty represents an irreducible limit. Recent works like LGNSDE (Bergna et al., 2024) integrate randomness via SDEs, while others focus on conformal prediction (Huang et al., 2023; Cha et al., 2023) for guaranteed coverage or post-hoc evidential methods (Yu et al., 2025; Fuchsgruber et al., 2024). Despite these advances, there remains a need for GNN architectures that natively integrate and propagate uncertainty through the graph structure in a computationally tractable manner, explicitly separating uncertainty sources and demonstrating robustness, especially on out-of-distribution (OOD) data. The key challenges identified in the literature, including deep integration of UQ, separation of uncertainty types, scalability, and OOD robustness (Wang et al., 2024), motivate our research.

**Proposed Solution:**
We propose a novel **Uncertainty-Aware Bayesian Graph Neural Network (UA-BGNN)** framework designed to address these limitations. Our core idea is to integrate uncertainty quantification directly into the GNN's message-passing mechanism using a principled variational inference scheme. Instead of only propagating point estimates of node features, our framework propagates *distributions* over node representations layer by layer. Specifically, we model node features at each layer as latent random variables and learn approximate posterior distributions over them. This allows the model to capture and propagate uncertainty arising from both the input data and the model's own parameters (epistemic uncertainty). Crucially, we design the output layer to explicitly model aleatoric uncertainty by predicting the parameters of the output distribution (e.g., mean and variance for regression tasks). This principled separation provides more interpretable uncertainty estimates. Furthermore, we introduce a novel **uncertainty-aware attention mechanism** that adaptively weights messages from neighboring nodes based on their associated uncertainty, allowing the model to rely more heavily on confident information sources.

**Research Objectives:**
The primary objectives of this research are:
1.  To develop the theoretical framework for the Uncertainty-Aware Bayesian GNN (UA-BGNN), including the variational inference formulation for propagating distributions through GNN layers.
2.  To design and implement specific GNN layers (e.g., graph convolution, attention) that operate on and propagate distributions over node features, explicitly modeling learnable uncertainty parameters.
3.  To formulate and implement mechanisms within the framework to disentangle and quantify both aleatoric and epistemic uncertainty.
4.  To develop and incorporate an uncertainty-aware attention mechanism that modulates information flow based on propagated uncertainty estimates.
5.  To empirically validate the UA-BGNN framework on multiple benchmark datasets from diverse domains (e.g., molecular property prediction, traffic forecasting, node classification in social/citation networks).
6.  To demonstrate that the proposed UA-BGNN achieves competitive predictive performance while providing superior, well-calibrated, and interpretable uncertainty estimates compared to existing state-of-the-art UQ methods for GNNs, particularly under distribution shifts (OOD scenarios).

**Significance:**
This research addresses a critical gap in the reliability and trustworthiness of GNNs. By developing a GNN framework that intrinsically reasons about uncertainty, we aim to enhance the robustness of graph-based AI systems deployed in safety-critical and high-stakes domains. Providing well-calibrated and interpretable uncertainty estimates (distinguishing aleatoric and epistemic sources) empowers decision-makers to better understand the risks associated with model predictions. Successfully achieving our objectives will contribute significantly to the fields of graph representation learning, probabilistic machine learning, and trustworthy AI. The proposed framework has the potential to improve applications ranging from scientific discovery (e.g., more reliable material or drug design) to infrastructure management (e.g., safer traffic control) and financial modeling (e.g., robust risk assessment). This work directly aligns with the scope of the Structured Probabilistic Inference & Generative Modeling workshop, focusing on inference methods for graphs, unsupervised representation learning (implicitly via Bayesian framework), uncertainty quantification, and applications in science.

## 3. Methodology

This section details the proposed research methodology, covering the theoretical formulation, algorithmic steps, data collection, experimental design, and evaluation metrics.

**Theoretical Framework: Variational Inference for Propagating Node Feature Distributions**

Our UA-BGNN framework is grounded in Bayesian principles and approximated using variational inference (VI). We treat the node representations (features) at each layer, $\mathbf{h}_i^{(l)}$ for node $i$ at layer $l$, as latent random variables. The goal is to learn an approximate posterior distribution $q_\phi(\mathbf{H}^{(1)}, ..., \mathbf{H}^{(L)} | \mathbf{X}, \mathcal{G})$ over these representations, where $\mathbf{H}^{(l)} = \{\mathbf{h}_i^{(l)}\}_{i \in \mathcal{V}}$, $\mathbf{X}$ are initial node features, $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ is the graph structure, and $\phi$ are the variational parameters.

We adopt a layer-wise VI approach. Assume we have an approximate posterior distribution for layer $l-1$, denoted by $q_\phi(\mathbf{H}^{(l-1)})$. We define the $l$-th GNN layer as a transition probability $p(\mathbf{h}_i^{(l)} | \mathbf{h}_i^{(l-1)}, \{ \mathbf{h}_j^{(l-1)} \}_{j \in \mathcal{N}(i)})$, parameterized by learnable weights $\theta^{(l)}$. We then define a variational distribution $q_\phi(\mathbf{h}_i^{(l)})$ that approximates the true posterior $p(\mathbf{h}_i^{(l)} | \mathcal{D})$.

For tractability, we often assume factorized forms for the variational distributions, e.g., $q_\phi(\mathbf{H}^{(l)}) = \prod_{i \in \mathcal{V}} q_\phi(\mathbf{h}_i^{(l)})$. We typically model $q_\phi(\mathbf{h}_i^{(l)})$ as a Gaussian distribution with parameterized mean and (diagonal) covariance:
$$ q_\phi(\mathbf{h}_i^{(l)}) = \mathcal{N}(\mathbf{h}_i^{(l)}; \mu_i^{(l)}, \text{diag}((\sigma_i^{(l)})^2)) $$
The parameters $\mu_i^{(l)}$ and $\sigma_i^{(l)}$ are computed by GNN-like functions that take the parameters of the distributions from the previous layer ($\mu_i^{(l-1)}, \sigma_i^{(l-1)}$ and neighboring $\mu_j^{(l-1)}, \sigma_j^{(l-1)}$) as input.

The overall objective function is the Evidence Lower Bound (ELBO):
$$ \mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{Z})}[\log p(\mathbf{Y} | \mathbf{Z}_L)] - \sum_{l=1}^L \text{KL}(q_\phi(\mathbf{H}^{(l)}) || p(\mathbf{H}^{(l)} | \mathbf{H}^{(l-1)})) - \text{KL}(q_\phi(\mathbf{W}) || p(\mathbf{W})) $$
where $\mathbf{Z} = (\mathbf{H}^{(1)}, ..., \mathbf{H}^{(L)}, \mathbf{W})$ includes latent features and possibly Bayesian weights $\mathbf{W}$ with prior $p(\mathbf{W})$. The term $p(\mathbf{Y} | \mathbf{Z}_L)$ is the likelihood of the observed labels $\mathbf{Y}$ given the final layer representations. The KL divergence terms regularize the approximate posteriors towards the priors (or transition probabilities). We will primarily focus on propagating uncertainty in features $\mathbf{H}$, potentially simplifying the weight uncertainty aspect initially (e.g., using point estimates for weights or simple priors).

**Algorithmic Steps: Uncertainty Propagation Layers**

We will implement specific GNN layers that operate on input distributions $q_\phi(\mathbf{h}^{(l-1)})$ and output parameters for $q_\phi(\mathbf{h}^{(l)})$.

1.  **Input:** Distribution parameters for node $i$ and its neighbors at layer $l-1$: $\{ (\mu_k^{(l-1)}, \sigma_k^{(l-1)}) \}_{k \in \{i\} \cup \mathcal{N}(i)}$.
2.  **Propagation/Transformation:** Apply linear transformations parameterized by weights $W_\mu^{(l)}, W_\sigma^{(l)}$ to the means and variances. For instance, a simplified linear transform:
    $$ \tilde{\mu}_i^{(l)} = W_\mu^{(l)} \mu_i^{(l-1)} $$
    $$ (\tilde{\sigma}_i^{(l)})^2 = (W_\sigma^{(l)})^2 (\sigma_i^{(l-1)})^2 \quad (\text{element-wise square for diagonal covariance}) $$
    These transformations capture the initial processing of node features, analogous to the feature transformation step in standard GNNs. $W_\sigma^{(l)}$ are learnable parameters controlling epistemic uncertainty propagation/scaling.
3.  **Aggregation:** Aggregate transformed distributions from neighbors. Aggregating distributions analytically can be complex. We will explore two approaches:
    *   **Moment Matching:** Approximate the distribution of the aggregated features by matching moments (mean and variance). For sum aggregation of independent Gaussians $\mathcal{N}(\mu_j, \sigma_j^2)$:
        $$ \mu_{agg, i}^{(l)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \tilde{\mu}_j^{(l)} $$
        $$ (\sigma_{agg, i}^{(l)})^2 = \sum_{j \in \mathcal{N}(i) \cup \{i\}} (\tilde{\sigma}_j^{(l)})^2 $$
        Mean aggregation would involve averaging means and variances (divided by $|\mathcal{N}(i)|$ and $|\mathcal{N}(i)|^2$ respectively).
    *   **Monte Carlo Sampling:** Sample features from neighbor distributions $q_\phi(\mathbf{h}_j^{(l-1)})$, aggregate samples, then fit the output distribution $q_\phi(\mathbf{h}_i^{(l)})$ to the aggregated samples. This is more flexible but increases computational cost. We will primarily focus on moment matching for efficiency.
4.  **Update & Non-linearity:** Combine the aggregated information with the node's own transformed information and pass through a non-linearity. Applying non-linearities $\sigma(\cdot)$ to distributions is non-trivial. We can approximate the output distribution using:
    *   **Linearization:** Taylor expansion around the mean.
    *   **Sampling:** Propagate samples through the non-linearity.
    *   **Analytic Approximation (e.g., for ReLU):** Use known results for expectations and variances of rectified Gaussians.
    Alternatively, the update step directly parameterizes the output distribution:
    $$ \mu_i^{(l)} = \text{Update}_\mu (\tilde{\mu}_i^{(l)}, \mu_{agg, i}^{(l)}) $$
    $$ (\sigma_i^{(l)})^2 = \text{Update}_\sigma (\tilde{\sigma}_i^{(l)}, (\sigma_{agg, i}^{(l)})^2, \text{gradient info}) $$
    The Update functions (e.g., MLP layers) learn to combine these moments appropriately. Training relies on the reparameterization trick: $\mathbf{h}_i^{(l)} = \mu_i^{(l)} + \sigma_i^{(l)} \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$, allowing backpropagation through the stochastic nodes.

**Distinguishing Aleatoric and Epistemic Uncertainty**

*   **Epistemic Uncertainty:** This uncertainty arises from the model's limited knowledge. In our framework, it is captured by the variance $\sigma_i^{(l)}$ propagated through the layers. The variance in the final layer's feature distribution $q_\phi(\mathbf{h}_i^{(L)})$ reflects the model's uncertainty about the learned representation. If Bayesian weights are used ($q_\phi(\mathbf{W})$), their posterior variance also contributes. We can estimate epistemic uncertainty by examining the variance of the predictive distribution obtained by marginalizing over $q_\phi(\mathbf{H}^{(L)})$ (and $q_\phi(\mathbf{W})$ if applicable), often approximated via MC sampling from these distributions.
*   **Aleatoric Uncertainty:** This uncertainty is inherent in the data. We model it directly in the likelihood function $p(\mathbf{y}_i | \mathbf{h}_i^{(L)})$. For a regression task, instead of predicting only the mean $\hat{y}_i$, the final layer predicts both the mean and the variance of a Gaussian likelihood:
    $$ p(y_i | \mathbf{h}_i^{(L)}) = \mathcal{N}(y_i; \hat{y}_i, \sigma_{al, i}^2) $$
    where both $\hat{y}_i = f_\mu(\mathbf{h}_i^{(L)}; \phi_\mu)$ and $\log \sigma_{al, i}^2 = f_\sigma(\mathbf{h}_i^{(L)}; \phi_\sigma)$ are outputs of the network derived from the final mean representation $\mu_i^{(L)}$. The negative log-likelihood loss incorporates this predicted variance. For classification, similar ideas can be applied using Dirichlet distributions (via evidential deep learning, related to Yu et al., 2025) or by modeling uncertainty over softmax outputs.

**Uncertainty-Aware Attention Mechanism**

Building upon standard attention mechanisms like Graph Attention Network (GAT, Veličković et al., 2018), we incorporate uncertainty. The attention weight $\alpha_{ij}^{(l)}$ for the message from node $j$ to node $i$ at layer $l$ should depend not only on the feature means $\mu_i^{(l)}, \mu_j^{(l)}$ but also on their uncertainties $\sigma_i^{(l)}, \sigma_j^{(l)}$. A possible formulation:
$$ e_{ij}^{(l)} = \text{LeakyReLU} \left( \mathbf{a}^{(l) T} \left[ W_\mu^{(l)} \mu_i^{(l-1)} || W_\mu^{(l)} \mu_j^{(l-1)} \right] \right) $$
$$ \text{uncertainty_penalty}_{ij}^{(l)} = \lambda \cdot (\text{Tr}(\text{diag}((\sigma_i^{(l-1)})^2)) + \text{Tr}(\text{diag}((\sigma_j^{(l-1)})^2))) $$
$$ \tilde{e}_{ij}^{(l)} = \frac{e_{ij}^{(l)}}{1 + \text{uncertainty_penalty}_{ij}^{(l)}} $$
$$ \alpha_{ij}^{(l)} = \text{softmax}_j(\tilde{e}_{ij}^{(l)}) = \frac{\exp(\tilde{e}_{ij}^{(l)})}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(\tilde{e}_{ik}^{(l)})} $$
Here, $W_\mu^{(l)}$ and $\mathbf{a}^{(l)}$ are learnable weights, and $\lambda$ is a hyperparameter controlling the sensitivity to uncertainty. The aggregation step then uses these attention weights:
$$ \mu_i^{(l)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(l)} W_\mu^{(l)} \mu_j^{(l-1)} \right) $$
$$ (\sigma_i^{(l)})^2 = \sigma' \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} (\alpha_{ij}^{(l)})^2 (W_\sigma^{(l)})^2 (\sigma_j^{(l-1)})^2 \right) \quad (\text{Approximate propagation})$$
The variance propagation needs careful derivation depending on the exact aggregation and update rules.

**Data Collection**

We will use publicly available benchmark datasets relevant to the target applications:
1.  **Molecular Property Prediction:** QM9 (Ramakrishnan et al., 2014), MoleculeNet benchmarks (e.g., ESOL, FreeSolv, Lipophilicity for regression; BBBP, Tox21 for classification) (Wu et al., 2018).
2.  **Traffic Forecasting:** METR-LA, PEMS-BAY (Li et al., 2018). These are spatio-temporal graphs where nodes are sensors and edges represent proximity. Task is time series regression.
3.  **Node Classification:** Standard citation networks (Cora, CiteSeer, PubMed) (Sen et al., 2008) and potentially larger social network datasets if scalability permits.
4.  **OOD Evaluation:** We will generate OOD datasets by:
    *   Perturbing graph structures (adding/removing edges/nodes).
    *   Perturbing node features (adding noise).
    *   Using datasets from related but different domains (e.g., testing a molecular model trained on QM9 on a different chemical dataset).
    *   Using temporal shifts in traffic data (testing on unseen time periods).

**Experimental Design**

1.  **Tasks:** Graph classification/regression (Molecules), Node classification (Citation/Social), Time series forecasting (Traffic).
2.  **Baselines:** We will compare UA-BGNN against:
    *   Standard GNNs (GCN, GAT, GraphSAGE) providing point estimates.
    *   MC Dropout applied to standard GNNs.
    *   Deep Ensembles of standard GNNs (Lakshminarayanan et al., 2017; Mallick et al., 2022).
    *   Conformalized GNNs (CF-GNN) (Huang et al., 2023).
    *   Evidential GNNs (using framework like EPN, Yu et al., 2025, possibly re-implemented on our backbones).
    *   Energy-Based GNN UQ (GEBM, Fuchsgruber et al., 2024).
    *   Latent GNSDE (LGNSDE, Bergna et al., 2024) if implementation is feasible.
3.  **Implementation:** Models will be implemented using PyTorch and PyTorch Geometric. Training will use Adam optimizer with appropriate learning rate scheduling and hyperparameter tuning (e.g., dimensionality, number of layers, $\lambda$ for attention). Experiments will be run on GPU clusters.
4.  **Evaluation Protocol:** Standard train/validation/test splits will be used. Multiple runs with different random seeds will be performed for statistical significance.

**Evaluation Metrics**

*   **Predictive Performance:**
    *   Classification: Accuracy, F1-score.
    *   Regression: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
*   **Uncertainty Quantification Quality:**
    *   **Calibration:** Expected Calibration Error (ECE) for classification; Reliability diagrams and calibration metrics (e.g., sharpness conditional on calibration) for regression. Assess if predicted confidence matches empirical accuracy/error.
    *   **Negative Log-Likelihood (NLL):** Assesses the quality of the entire predictive distribution. Lower is better.
    *   **Proper Scoring Rules:** Continuous Ranked Probability Score (CRPS) for regression, Brier score for classification.
    *   **Sharpness:** Average predictive variance/entropy. Lower is better, given good calibration.
    *   **Separation of Uncertainty:** Analyze predicted aleatoric vs. epistemic uncertainty on ID and OOD data. Epistemic uncertainty should increase significantly on OOD samples. Aleatoric uncertainty should reflect data noise.
*   **Robustness (OOD performance):**
    *   Evaluate metrics above on OOD datasets.
    *   Analyze correlation between uncertainty estimates and prediction errors on OOD data.
    *   Misclassification / Outlier Detection: Use uncertainty scores (e.g., predictive variance, entropy) to rank samples and evaluate AUC-ROC or Precision-Recall for detecting OOD samples or misclassified ID samples.
*   **Computational Cost:** Measure training time and inference time (including time to generate uncertainty estimates, e.g., MC sampling if needed).

## 4. Expected Outcomes & Impact

**Expected Outcomes:**

1.  **A Novel UA-BGNN Framework:** The primary outcome will be a fully developed and implemented Bayesian GNN framework that integrates uncertainty propagation directly within the message-passing layers. This includes novel GNN layer implementations (convolutional, attentional) operating on distributions.
2.  **Algorithmic Contributions:** Development of tractable methods for distribution propagation, uncertainty-aware attention, and principled separation of aleatoric and epistemic uncertainty within GNNs.
3.  **Open-Source Code:** A publicly available codebase implementing the UA-BGNN framework and experimental setups, facilitating reproducibility and further research.
4.  **Empirical Validation:** Comprehensive experimental results on diverse benchmark datasets demonstrating the effectiveness of UA-BGNN. We expect to show:
    *   Competitive or superior predictive performance compared to baseline GNNs.
    *   Significantly better calibrated and informative uncertainty estimates (lower ECE, NLL, higher correlation with errors) compared to existing UQ methods like MC Dropout, Ensembles, and potentially other recent approaches.
    *   Demonstrated ability to distinguish aleatoric and epistemic uncertainty.
    *   Improved robustness and reliable uncertainty increase on OOD data.
5.  **Publications:** High-quality publications in top-tier machine learning or AI conferences (e.g., NeurIPS, ICML, ICLR) and potentially journals. Presentation at relevant workshops, such as the proposed Structured Probabilistic Inference & Generative Modeling workshop.
6.  **Insights into Uncertainty in GNNs:** Deeper understanding of how uncertainty propagates through graph structures and how architectural choices impact UQ quality.

**Impact:**

*   **Enhanced Trustworthiness of AI:** By providing reliable and interpretable uncertainty estimates, UA-BGNNs can significantly increase the trustworthiness and safe adoption of GNNs in critical applications. Decision-makers can make more informed choices by explicitly considering the model's confidence.
*   **Advancement in Probabilistic ML and Graph Learning:** This research pushes the boundaries of Bayesian deep learning and graph representation learning by developing scalable and integrated UQ methods specifically tailored for graph data's unique challenges.
*   **Societal Benefits:** Improved reliability in areas like drug discovery (faster development, reduced failures), traffic management (safer and more efficient transportation), financial modeling (better risk management), and scientific discovery (more robust analysis of structured data) can lead to significant societal and economic benefits.
*   **Foundation for Future Research:** The developed framework and insights can serve as a foundation for future work on more complex probabilistic graph models, active learning with uncertainty, causal inference on graphs, and continual learning in graph settings.

In conclusion, this research proposes a principled and novel approach to uncertainty quantification in GNNs. By embedding uncertainty estimation into the core architecture, UA-BGNN aims to deliver highly reliable and interpretable predictions, paving the way for more robust and trustworthy graph-based intelligent systems.

## 5. References

*(Note: The full list would include all cited papers, matching the provided literature review entries and any additional citations like Zhou et al. 2020, Wu et al. 2020, Lakshminarayanan et al. 2017, Gal & Ghahramani 2016, Kendall & Gal 2017, Veličković et al. 2018, Ramakrishnan et al. 2014, Wu et al. 2018, Li et al. 2018, Sen et al. 2008 etc. Here references correspond to the literature review list provided in the prompt for brevity)*

1.  Bergna, R., Calvo-Ordoñez, S., Opolka, F. L., Liò, P., & Hernandez-Lobato, J. M. (2024). Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations. *arXiv preprint arXiv:2408.16115*.
2.  Vinchurkar, T., Abdelmaqsoud, K., & Kitchin, J. R. (2025). Uncertainty Quantification in Graph Neural Networks with Shallow Ensembles. *arXiv preprint arXiv:2504.12627*.
3.  Yu, L., Li, K., Saha, P. K., Lou, Y., & Chen, F. (2025). Evidential Uncertainty Probes for Graph Neural Networks. *arXiv preprint arXiv:2503.08097*.
4.  Wang, F., Liu, Y., Liu, K., Wang, Y., Medya, S., & Yu, P. S. (2024). Uncertainty in Graph Neural Networks: A Survey. *arXiv preprint arXiv:2403.07185*.
5.  Huang, K., Jin, Y., Candès, E., & Leskovec, J. (2023). Uncertainty Quantification over Graph with Conformalized Graph Neural Networks. *arXiv preprint arXiv:2305.14535*.
6.  Cha, S., Kang, H., & Kang, J. (2023). On the Temperature of Bayesian Graph Neural Networks for Conformal Prediction. *arXiv preprint arXiv:2310.11479*.
7.  Jiang, S., Qin, S., Van Lehn, R. C., Balaprakash, P., & Zavala, V. M. (2023). Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search. *arXiv preprint arXiv:2307.10438*.
8.  Fuchsgruber, D., Wollschläger, T., & Günnemann, S. (2024). Energy-based Epistemic Uncertainty for Graph Neural Networks. *arXiv preprint arXiv:2406.04043*.
9.  Wang, Q., Wang, S., Zhuang, D., Koutsopoulos, H., & Zhao, J. (2023). Uncertainty Quantification of Spatiotemporal Travel Demand with Probabilistic Graph Neural Networks. *arXiv preprint arXiv:2303.04040*.
10. Mallick, T., Balaprakash, P., & Macfarlane, J. (2022). Deep-Ensemble-Based Uncertainty Quantification in Spatiotemporal Graph Neural Networks for Traffic Forecasting. *arXiv preprint arXiv:2204.01618*.
*(Additional implied references: Zhou et al., 2020; Wu et al., 2020; Lakshminarayanan et al., 2017; Gal & Ghahramani, 2016; Kendall & Gal, 2017; Veličković et al., 2018; Ramakrishnan et al., 2014; Wu et al., 2018; Li et al., 2018; Sen et al., 2008)*