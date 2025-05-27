Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **Adaptive Unbalanced Optimal Transport for Robust Deep Domain Adaptation under Label Shift**

**2. Introduction**

**2.1 Background**
Domain Adaptation (DA) techniques aim to leverage knowledge learned from a labeled source domain ($D_S$) to perform tasks on a related but different, often unlabeled or sparsely labeled, target domain ($D_T$). This is crucial in machine learning as collecting labeled data for every possible target domain is often prohibitively expensive or impractical. Deep learning models, while powerful, are known to suffer performance degradation when deployed in domains different from their training domain due to domain shift – discrepancies in the data distributions between $D_S$ and $D_T$.

Optimal Transport (OT) has emerged as a powerful geometric tool for DA (Courty et al., 2017). By viewing domain shift as the cost of "transporting" the probability mass of the source distribution to match the target distribution, OT provides a principled way to align feature distributions across domains. The Wasserstein distance, derived from OT, offers a meaningful discrepancy measure between distributions, often used as a regularization term in deep DA models to encourage domain-invariant feature representations. Standard OT formulations typically compute a transport plan $\gamma$ that minimizes the transport cost $\langle C, \gamma \rangle$ subject to marginal constraints, ensuring that the transported source distribution exactly matches the target distribution ($\gamma \mathbf{1} = \mu_S$, $\gamma^T \mathbf{1} = \mu_T$, where $\mu_S$ and $\mu_T$ are the source and target distributions, respectively).

However, a critical limitation of standard OT in DA is the assumption of *balanced* domains, meaning the total mass and, implicitly, the class proportions are assumed to be identical. This rarely holds in practice. Real-world DA scenarios often involve **label shift**, where the marginal distribution of labels $P(y)$ differs between the source and target domains ($P_S(y) \neq P_T(y)$), even if the conditional distributions $P(x|y)$ remain similar. Applying standard OT under label shift can force incorrect alignments between classes, potentially leading to *negative transfer* – degraded performance compared to using only source data (Fatras et al., 2022).

Unbalanced Optimal Transport (UOT) offers a relaxation of the strict marginal constraints of OT (Chizat et al., 2018; Liero et al., 2018). UOT introduces penalty terms, typically Kullback-Leibler (KL) divergences, that allow the marginals of the transport plan $\gamma$ to deviate from the original $\mu_S$ and $\mu_T$. This allows UOT to handle distributions with different total mass and provides robustness to outliers. Fatras et al. (2021) demonstrated the effectiveness of UOT, particularly in a minibatch setting, for DA, showing improved performance over standard OT. Tran et al. (2022) further extended UOT to the COOT framework for aligning samples and features robustly.

Despite its advantages, UOT typically requires specifying hyperparameters ($\lambda_1, \lambda_2$) that control the degree of relaxation for each marginal constraint. Selecting optimal values for these parameters a priori is challenging and data-dependent. In the context of DA under unknown label shift, fixing these parameters limits the model's ability to flexibly adapt to the specific, unknown discrepancy in class proportions between domains. While methods like Rakotomamonjy et al. (2020) address label shift by estimating target proportions and using importance weighting with OT, they don't directly adapt the UOT mechanism itself.

**2.2 Research Gap and Proposed Idea**
The key research gap lies in developing a domain adaptation method that can robustly handle **unknown label shifts** without requiring prior knowledge or explicit estimation of target label proportions, while leveraging the geometric strengths of Optimal Transport. Existing UOT approaches for DA (Fatras et al., 2021; 2022) rely on predefined, fixed relaxation parameters, limiting their adaptability to varying degrees of label shift.

We propose **Adaptive Unbalanced Optimal Transport (A-UOT)**, a novel framework integrated within deep domain adaptation models. The core idea is to **learn the optimal degree of marginal relaxation in UOT directly from the data during end-to-end training**. Instead of fixed hyperparameters $\lambda_1, \lambda_2$, A-UOT introduces learnable parameters that control these relaxation penalties. These parameters are optimized jointly with the feature extractor and classifier weights, guided by the overall learning objective (e.g., minimizing source classification loss and target adaptation loss). This allows the model to implicitly infer the necessary alignment flexibility required by the underlying label shift, effectively adapting the transport plan's marginals to better match the (unknown) target label distribution structure embedded within the feature space.

**2.3 Research Objectives**
The primary objectives of this research are:
1.  **Formulate the A-UOT Framework:** Define the mathematical formulation for A-UOT, incorporating learnable marginal relaxation parameters into the UOT objective function within a deep DA setting.
2.  **Develop an End-to-End Training Algorithm:** Design and implement an algorithm for jointly optimizing the deep model parameters (feature extractor, classifier) and the A-UOT relaxation parameters using stochastic gradient descent.
3.  **Evaluate A-UOT Performance:** Empirically evaluate the proposed A-UOT method on standard DA benchmark datasets under various controlled label shift scenarios.
4.  **Compare with Baselines:** Compare the performance of A-UOT against relevant baselines, including source-only models, standard DA methods (e.g., DANN), standard OT-based DA, and UOT-based DA with fixed relaxation parameters (e.g., UOTDA based on Fatras et al., 2021).
5.  **Analyze Adaptability:** Investigate the behavior of the learned relaxation parameters under different types and magnitudes of label shifts to understand the adaptive mechanism.

**2.4 Significance**
This research holds significant potential contributions:
1.  **Improved Robustness:** A-UOT promises more robust domain adaptation performance in realistic scenarios where label distributions change between domains, a common challenge currently limiting the deployment of many DA techniques.
2.  **Enhanced Practicality:** By automatically adapting to unknown label shifts, A-UOT reduces the need for manual tuning of UOT parameters or explicit label proportion estimation steps, simplifying the application pipeline.
3.  **Advancement in OT for ML:** This work contributes to the growing field of OTML by proposing a novel adaptive variant of UOT tailored for a specific machine learning challenge, potentially inspiring similar adaptive mechanisms in other UOT applications.
4.  **Potential for Broader Impact:** The principle of adaptive regularization within OT could potentially be extended to other problems involving distribution alignment under uncertainty or partial information.

**3. Methodology**

**3.1 Preliminaries: Optimal Transport and Unbalanced Optimal Transport**

Let $\mu_S \in \mathcal{P}(\mathcal{X})$ and $\mu_T \in \mathcal{P}(\mathcal{X})$ be the probability distributions of source and target domain features, respectively, residing in a feature space $\mathcal{X} \subseteq \mathbb{R}^d$. Let $C: \mathcal{X} \times \mathcal{X} \to \mathbb{R}^+$ be a cost function, typically the squared Euclidean distance $C(x, x') = \|x - x'\|_2^2$.

**Standard Optimal Transport (Kantorovich Formulation):** Seeks a coupling (transport plan) $\gamma \in \Pi(\mu_S, \mu_T)$ that minimizes the total transport cost:
$$
\min_{\gamma \in \Pi(\mu_S, \mu_T)} \int_{\mathcal{X} \times \mathcal{X}} C(x, x') d\gamma(x, x')
$$
where $\Pi(\mu_S, \mu_T) = \{ \gamma \in \mathcal{P}(\mathcal{X} \times \mathcal{X}) \mid \gamma(A \times \mathcal{X}) = \mu_S(A), \gamma(\mathcal{X} \times B) = \mu_T(B) \}$ is the set of all joint distributions with marginals $\mu_S$ and $\mu_T$. In the discrete case with empirical distributions $\mu_S = \sum_{i=1}^{n_S} a_i \delta_{x_i^S}$ and $\mu_T = \sum_{j=1}^{n_T} b_j \delta_{x_j^T}$ (where $a_i=1/n_S, b_j=1/n_T$), this becomes:
$$
\min_{\gamma \in \mathbb{R}_+^{n_S \times n_T}} \sum_{i,j} C_{ij} \gamma_{ij} \quad \text{s.t.} \quad \gamma \mathbf{1}_{n_T} = \mathbf{a}, \quad \gamma^T \mathbf{1}_{n_S} = \mathbf{b}
$$
where $C_{ij} = C(x_i^S, x_j^T)$, $\mathbf{a} = (a_1, ..., a_{n_S})^T$, $\mathbf{b} = (b_1, ..., b_{n_T})^T$. The constraint $\sum a_i = \sum b_j$ is required (balanced assumption).

**Unbalanced Optimal Transport (KL-Regularized):** Relaxes the marginal constraints using KL-divergence penalties:
$$
\min_{\gamma \ge 0} \sum_{i,j} C_{ij} \gamma_{ij} + \lambda_1 D_{KL}(\gamma \mathbf{1}_{n_T} || \mathbf{a}) + \lambda_2 D_{KL}(\gamma^T \mathbf{1}_{n_S} || \mathbf{b})
$$
where $D_{KL}(p || q) = \sum_k p_k \log(p_k / q_k) - p_k + q_k$ is the generalized KL divergence. The parameters $\lambda_1, \lambda_2 \ge 0$ control the trade-off between minimizing the transport cost and satisfying the marginal constraints. Larger $\lambda$ values enforce stricter adherence to the original marginals. Often, entropic regularization $\epsilon H(\gamma) = -\epsilon \sum_{i,j} \gamma_{ij}(\log \gamma_{ij} - 1)$ is added for computational efficiency via the Sinkhorn algorithm. The UOT cost is denoted $UOT_{\lambda_1, \lambda_2}(\mu_S, \mu_T)$.

**3.2 Proposed A-UOT Framework**

We propose to integrate UOT into a deep domain adaptation model, where the relaxation parameters are learned.

**3.2.1 Model Architecture:**
The model consists of:
- A feature extractor $F_\phi: \mathcal{X}_{input} \to \mathcal{X}$ parameterized by $\phi$. This network maps input data (e.g., images) to a feature space $\mathcal{X}$ where domain alignment is performed.
- A classifier $G_\psi: \mathcal{X} \to \mathbb{R}^K$ parameterized by $\psi$, predicting class probabilities for $K$ classes.

**3.2.2 A-UOT Formulation:**
Let $X_S = \{x_i^S\}_{i=1}^{n_S}$ and $X_T = \{x_j^T\}_{j=1}^{n_T}$ be minibatches of source and target data, respectively. Their feature representations are $Z_S = \{z_i^S = F_\phi(x_i^S)\}$ and $Z_T = \{z_j^T = F_\phi(x_j^T)\}$. We consider the empirical distributions $\hat{\mu}_S = \frac{1}{n_S} \sum_{i=1}^{n_S} \delta_{z_i^S}$ and $\hat{\mu}_T = \frac{1}{n_T} \sum_{j=1}^{n_T} \delta_{z_j^T}$.

Instead of fixed $\lambda_1, \lambda_2$, we introduce learnable parameters $\boldsymbol{\theta} = (\theta_1, \theta_2)$, possibly dependent on batches or global statistics. We parameterize the relaxation coefficients positively, e.g., using an exponential function: $\lambda_1(\boldsymbol{\theta}) = \exp(\theta_1)$ and $\lambda_2(\boldsymbol{\theta}) = \exp(\theta_2)$. The A-UOT cost is then defined as:
$$
\mathcal{L}_{A-UOT}(\phi, \boldsymbol{\theta}) = UOT_{\lambda_1(\boldsymbol{\theta}), \lambda_2(\boldsymbol{\theta})}(\hat{\mu}_S, \hat{\mu}_T)
$$
$$
= \min_{\gamma \ge 0} \sum_{i,j} C(z_i^S, z_j^T) \gamma_{ij} + \lambda_1(\boldsymbol{\theta}) D_{KL}(\gamma \mathbf{1}_{n_T} || \mathbf{a}) + \lambda_2(\boldsymbol{\theta}) D_{KL}(\gamma^T \mathbf{1}_{n_S} || \mathbf{b}) + \epsilon H(\gamma)
$$
where $\mathbf{a} = \frac{1}{n_S} \mathbf{1}_{n_S}$, $\mathbf{b} = \frac{1}{n_T} \mathbf{1}_{n_T}$, and $\epsilon$ is a fixed small constant for entropic regularization.

**3.2.3 Overall Objective Function:**
The total loss function combines the standard supervised loss on the source domain, the A-UOT loss for domain alignment, and potentially other regularization terms:
$$
\mathcal{L}_{total}(\phi, \psi, \boldsymbol{\theta}) = \mathcal{L}_{cls}(\phi, \psi) + \beta \mathcal{L}_{A-UOT}(\phi, \boldsymbol{\theta})
$$
where:
- $\mathcal{L}_{cls}(\phi, \psi) = \frac{1}{n_S} \sum_{i=1}^{n_S} \mathcal{L}_{CE}(G_\psi(F_\phi(x_i^S)), y_i^S)$ is the cross-entropy loss on labeled source samples.
- $\beta > 0$ is a hyperparameter balancing the classification task and domain alignment.
- The parameters $\phi, \psi, \boldsymbol{\theta}$ are optimized jointly.

**3.2.4 Learning Mechanism for $\boldsymbol{\theta}$:**
The gradients $\frac{\partial \mathcal{L}_{A-UOT}}{\partial \boldsymbol{\theta}}$ can be computed using the implicit differentiation of the Sinkhorn iterations or the envelope theorem applied to the UOT dual formulation. Assuming the optimal dual potentials $f, g$ for the UOT problem are found (via Sinkhorn-like iterations), the gradient with respect to $\theta_k$ (for $k=1, 2$) is related to the deviation of the optimal coupling's marginals from the reference measures $\mathbf{a}, \mathbf{b}$. Specifically, if $\lambda_k = e^{\theta_k}$, then $\frac{\partial \mathcal{L}_{A-UOT}}{\partial \theta_k} = \frac{\partial \mathcal{L}_{A-UOT}}{\partial \lambda_k} \frac{\partial \lambda_k}{\partial \theta_k} = \frac{\partial \mathcal{L}_{A-UOT}}{\partial \lambda_k} e^{\theta_k}$. The partial derivative $\frac{\partial \mathcal{L}_{A-UOT}}{\partial \lambda_k}$ depends on the KL divergence term evaluated at the optimum. The key idea is that $\boldsymbol{\theta}$ will be adjusted via gradient descent to find relaxation levels that contribute optimally to minimizing the *total* loss $\mathcal{L}_{total}$. If aligning strictly (large $\lambda_k$) hinders classification performance (e.g., due to label shift causing misalignment), the gradient descent process will push $\theta_k$ lower, allowing more marginal relaxation. Conversely, if more alignment is beneficial, $\theta_k$ might increase.

**3.3 Algorithmic Steps**

The training proceeds as follows:

1.  Initialize network parameters $\phi, \psi$ and adaptive UOT parameters $\boldsymbol{\theta}$.
2.  **For** each training iteration:
    a. Sample a minibatch of source data $\{(x_i^S, y_i^S)\}_{i=1}^{n_S}$ and target data $\{x_j^T\}_{j=1}^{n_T}$.
    b. Compute source features $Z_S = F_\phi(X_S)$ and target features $Z_T = F_\phi(X_T)$.
    c. Compute the cost matrix $C_{ij} = \|z_i^S - z_j^T\|_2^2$.
    d. Compute the current relaxation parameters $\lambda_1 = \exp(\theta_1), \lambda_2 = \exp(\theta_2)$.
    e. Solve the (entropically regularized) UOT problem between $\hat{\mu}_S = \frac{1}{n_S}\sum \delta_{z_i^S}$ and $\hat{\mu}_T = \frac{1}{n_T}\sum \delta_{z_j^T}$ with costs $C$ and relaxation parameters $\lambda_1, \lambda_2$ to obtain the optimal coupling $\gamma^*$ and the A-UOT loss $\mathcal{L}_{A-UOT}$. This typically involves Sinkhorn-like iterations.
    f. Compute the source classification loss $\mathcal{L}_{cls}$ using $Z_S$, $G_\psi$, and source labels $Y_S$.
    g. Compute the total loss $\mathcal{L}_{total} = \mathcal{L}_{cls} + \beta \mathcal{L}_{A-UOT}$.
    h. Perform backpropagation to compute gradients $\nabla_\phi \mathcal{L}_{total}$, $\nabla_\psi \mathcal{L}_{total}$, and $\nabla_{\boldsymbol{\theta}} \mathcal{L}_{total}$. The gradient w.r.t. $\boldsymbol{\theta}$ requires differentiating through the UOT solution (e.g., via implicit function theorem on Sinkhorn iterations).
    i. Update parameters using an optimizer (e.g., Adam):
       $\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}_{total}$
       $\psi \leftarrow \psi - \eta \nabla_\psi \mathcal{L}_{total}$
       $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}_{total}$
3.  Repeat step 2 until convergence.

**3.4 Data Collection / Datasets**

We will use standard benchmark datasets for unsupervised domain adaptation, focusing on those where label shift can be easily simulated or is known to exist:
1.  **Digits Datasets (MNIST, USPS, SVHN):** Adapting between these datasets (e.g., MNIST $\to$ USPS) allows for easy simulation of label shift by subsampling classes in the target domain during evaluation.
2.  **Office-Caltech10:** Contains images from 4 domains (Amazon, Caltech, DSLR, Webcam) across 10 classes. Label shift can be introduced by altering class proportions in the target domain splits.
3.  **Office-Home:** A more challenging dataset with 4 domains (Art, Clipart, Product, Real-World) and 65 classes. Provides a complex scenario for evaluating robustness to label shift.
4.  **VisDA-2017:** Large-scale dataset for Synthetic-to-Real adaptation, where label shifts are naturally present.

For controlled experiments, we will create specific label shift scenarios by manually adjusting the sampling probabilities of target domain classes while keeping the source domain balanced or having a known distribution. We will consider scenarios like:
-   **Target Class Imbalance:** Some target classes are significantly more/less frequent than others (and different from source proportions).
-   **Missing Target Classes:** Some classes present in the source are absent in the target.
-   **Novel Target Classes:** (Potentially future work) Target domain contains classes not seen in source. A-UOT might naturally handle this by allowing mass destruction via small $\lambda_2$.

**3.5 Experimental Design**

1.  **Baselines:**
    *   Source-Only: Train on source, test on target (lower bound).
    *   Deep Adaptation Network (DANN) (Ganin et al., 2016): Adversarial adaptation baseline.
    *   Conditional Adversarial Domain Adaptation (CDAN) (Long et al., 2018): Uses conditional adversarial loss.
    *   Standard OTDA: Deep DA using standard OT (balanced) for alignment (e.g., based on Courty et al., 2017).
    *   Fixed UOTDA: Deep DA using UOT with fixed, manually tuned $\lambda_1, \lambda_2$ (e.g., reimplementation based on Fatras et al., 2021 concepts). We will test various fixed $\lambda$ pairs.
    *   Label Shift Correction Method (e.g., Rakotomamonjy et al., 2020): A method explicitly estimating target proportions.

2.  **Evaluation Scenarios:** We will evaluate performance on standard UDA tasks (assuming balanced target during training if labels are absent, but evaluate on shifted target) and on explicitly constructed label shift scenarios (varying target proportions as described above).

3.  **Evaluation Metrics:**
    *   **Classification Accuracy:** Primary metric is the classification accuracy on the target domain test set.
    *   **Per-Class Accuracy & F1-Score:** To better understand performance on minority/majority classes under label shift.
    *   **Analysis of Learned $\boldsymbol{\theta}$:** We will track the values of $\theta_1, \theta_2$ during training and analyze their final values under different label shift conditions to verify if they adapt meaningfully (e.g., does $\lambda_2 = \exp(\theta_2)$ decrease when target mass should be reduced due to missing classes?).

4.  **Implementation Details:**
    *   Backbone Networks: Standard architectures like ResNet-