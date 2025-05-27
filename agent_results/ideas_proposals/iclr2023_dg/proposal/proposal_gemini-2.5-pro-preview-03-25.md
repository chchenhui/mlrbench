Okay, here is a detailed research proposal based on the provided task description, research idea, and literature review.

---

# **Research Proposal**

## **1. Title: Leveraging Inferred Causal Structure for Robust Domain Generalization via Invariant Mechanism Alignment**

## **2. Introduction**

### 2.1 Background
Machine learning models, particularly deep neural networks, have achieved remarkable success across various domains. However, a critical limitation is their often-poor performance when deployed in environments whose data distribution differs from the training distribution. This phenomenon, known as distribution shift, hinders the reliable application of ML in real-world scenarios such as medical diagnosis (where patient populations or imaging equipment vary), autonomous driving (encountering diverse weather and lighting conditions), and financial modeling (adapting to market volatility).

Domain Generalization (DG) aims to address this challenge by training a model on data from multiple related, yet distinct, *source* domains such that it generalizes well to unseen *target* domains without requiring access to target domain data during training. Despite significant research efforts, many proposed DG algorithms have struggled to consistently outperform the simple baseline of Empirical Risk Minimization (ERM) trained on the union of source domains (Gulrajani & Lopez-Paz, 2020). This suggests that merely minimizing the average loss across source domains is often insufficient to capture the underlying invariant patterns necessary for out-of-distribution (OOD) generalization.

The central question posed by the workshop, "What do we need for successful domain generalization?", highlights the community's search for crucial ingredients missing from standard learning paradigms. A prevailing hypothesis, which motivates this proposal, is that successful DG requires incorporating *additional information* beyond the input-label pairs $(X, Y)$. This information could guide the learning process towards representations that are robust to domain-specific variations while capturing the stable, underlying mechanisms relating inputs to outputs.

Causality offers a powerful lens through which to view this problem (Pearl, 2009; Peters et al., 2017). A causal model describes the underlying data-generating process, distinguishing mechanisms that cause an effect from mere statistical correlations. Crucially, fundamental causal mechanisms (e.g., the physics governing image formation, the biological basis of a disease) are expected to remain invariant across different domains or environments, even if the distributions of inputs or confounding factors change (Schölkopf et al., 2012). In contrast, spurious correlations (e.g., associating cows with green pastures because training images predominantly feature them) are often domain-specific and break down under distribution shift. Therefore, leveraging causal principles to identify and isolate these invariant causal mechanisms, while discounting domain-specific spurious correlations, emerges as a highly promising direction for achieving robust domain generalization.

Recent work has started exploring the intersection of causality and DG (Lv et al., 2022; Miao et al., 2022; Wang et al., 2021), often focusing on learning representations that satisfy certain causal properties or align causal mechanisms. However, challenges remain, particularly in reliably inferring relevant causal structures from observational multi-domain data and effectively integrating this structural knowledge into deep learning frameworks to enforce invariance (as highlighted in the literature review).

### 2.2 Research Objectives
This research proposes a novel framework, **C**ausal **S**tructure-**A**ware **I**nvariant **L**earning (**CSAIL**), designed to explicitly leverage inferred causal structures from multi-domain data to guide representation learning for improved domain generalization. The primary goal is to learn representations that capture domain-invariant causal relationships while actively mitigating the influence of domain-specific spurious correlations.

Our specific objectives are:

1.  **Develop a Causal Constraint Inference Module:** To design an algorithm that analyzes data from multiple source domains, potentially utilizing domain labels or metadata, to infer a set of plausible causal constraints. This module will aim to distinguish features or relationships likely involved in the invariant (potentially causal) mechanism predicting the target variable $Y$ from those exhibiting domain-specific correlations (spurious). This might involve identifying conditional independencies that hold across domains or detecting relationships whose strength or nature varies significantly between domains.
2.  **Integrate Causal Constraints into Representation Learning:** To formulate a learning objective that incorporates the inferred causal constraints as differentiable regularization terms or architectural biases within a deep neural network. The objective will penalize representations that violate the identified invariance properties and reward alignment with the stable, cross-domain structure.
3.  **Design the CSAIL Framework:** To combine the causal constraint inference module and the constraint-aware representation learning objective into an end-to-end trainable framework. This involves defining the interplay between structure inference (which might be iterative or occur upfront) and representation learning.
4.  **Empirical Validation and Analysis:** To rigorously evaluate the CSAIL framework on established domain generalization benchmarks (e.g., DomainBed suite: PACS, VLCS, OfficeHome, TerraIncognita). This includes comparing its performance against ERM and state-of-the-art (SOTA) DG methods, including recent causality-inspired approaches (e.g., CCM, CIRL). We will also perform ablation studies to understand the contribution of each component and analyze the learned representations to verify if they exhibit the desired invariance properties.

### 2.3 Significance
This research directly addresses the central question of the workshop by proposing *causal structure* as a critical form of additional information necessary for successful DG. By moving beyond simple statistical correlations and focusing on the underlying generative mechanisms, we hypothesize that CSAIL can overcome the brittleness of existing methods.

The potential significance of this work is threefold:

1.  **Improved Robustness:** By explicitly identifying and leveraging invariant causal structures while mitigating spurious correlations, CSAIL aims to produce models that are significantly more robust to distribution shifts encountered in unseen target domains. This has direct implications for deploying reliable ML systems in critical applications like healthcare and autonomous systems.
2.  **Advancing DG Methodology:** This work contributes a novel framework that systematically integrates ideas from causal inference (structure identification from multi-domain data) with deep representation learning (constraint-based optimization). Success would provide strong evidence for the utility of causal approaches in DG and potentially inspire new algorithms that explicitly model and exploit data-generating processes.
3.  **Understanding Generalization:** By analyzing *why* CSAIL succeeds or fails, and examining the properties of the learned representations (e.g., disentanglement of causal vs. spurious factors, invariance metrics), this research can provide valuable insights into the fundamental mechanisms underlying OOD generalization and the specific role causal invariance plays.

## **3. Methodology**

### 3.1 Overall Framework: CSAIL
The proposed Causal Structure-Aware Invariant Learning (CSAIL) framework comprises three main components:

1.  **Feature Extractor:** A neural network backbone $f_{\theta}: \mathcal{X} \rightarrow \mathcal{Z}$ (e.g., a ResNet for image data) parameterized by $\theta$, which maps high-dimensional input $X$ to a lower-dimensional representation $Z$.
2.  **Causal Constraint Inference Module:** A procedure $\mathcal{C}$ that takes multi-domain data $\{(X_d, Y_d)\}_{d=1}^D$ (optionally with domain indices $d$) as input and outputs a set of causal constraints $\mathcal{G}$. These constraints encode hypotheses about which aspects of the relationship between $X$ (or features derived from $X$) and $Y$ are invariant across domains versus domain-specific.
3.  **Constraint-Aware Predictor & Training:** A predictor head $h_{\phi}: \mathcal{Z} \rightarrow \mathcal{Y}$ parameterized by $\phi$, trained jointly with $f_{\theta}$ using an objective function that combines a standard supervised loss (ERM term) with a regularization term derived from the causal constraints $\mathcal{G}$.

The core idea is that $\mathcal{G}$ guides the learning of $\theta$ and $\phi$ such that the learned representation $Z$ and predictor $h_{\phi}$ primarily rely on the invariant aspects identified by $\mathcal{C}$.

![Conceptual Diagram of CSAIL Framework](placeholder_diagram.png)
*(Self-note: A diagram would ideally show Data -> Causal Constraint Inference -> Constraints G; Data -> Feature Extractor -> Representation Z; Representation Z -> Predictor -> Prediction Y; Constraints G -> influences training of Feature Extractor and/or Predictor via Loss Function)*

### 3.2 Data Collection
We will primarily use standard benchmark datasets from the DomainBed suite (Gulrajani & Lopez-Paz, 2020), including:
*   **PACS:** Photos, Art Painting, Cartoon, Sketch. Object recognition task. Significant style variation.
*   **VLCS:** Caltech101, LabelMe, SUN09, VOC2007. Object recognition task. Variation in camera/dataset source.
*   **OfficeHome:** Artistic, Clipart, Product, Real-World images. Object recognition task. Different visual styles.
*   **TerraIncognita:** Wildlife camera trap images from different locations. Animal classification task. Challenging variations in background, lighting, animal pose, and potential confounding factors (e.g., location correlates with species distribution).

These datasets provide multiple labeled source domains, allowing for the application of multi-domain causal inference techniques and standardized evaluation using the leave-one-domain-out protocol. Domain labels are available and can be used by the Causal Constraint Inference Module.

### 3.3 Causal Constraint Inference Module ($\mathcal{C}$)
This module aims to identify potential causal versus spurious relationships without necessarily recovering the full latent causal graph. We propose to operationalize this by identifying features or feature-target relationships whose properties are stable across domains versus those that vary.

**Assumption:** We assume the data $(X_d, Y_d)$ for each domain $d$ is generated from an underlying Structural Causal Model (SCM) where $Y$ is caused by a set of latent factors $C$ (invariant causal features), potentially influenced by $X$. $X$ itself might be influenced by $C$ and additional domain-specific factors $S_d$ (spurious or style factors). $X = g(C, S_d)$, $Y = h(C)$. Our goal is to learn a function that approximates $h(\cdot)$ based on $X$, implicitly recovering $C$.

**Proposed Approach:** Instead of full SCM discovery, we focus on deriving constraints from observational data by analyzing stability across domains.
Let $Z = f_\theta(X)$ be the representation learned by the feature extractor. We approximate causal structure by analyzing dependencies involving $Z$ (or intermediate features) and $Y$ across domains.

1.  **Feature-Level Stability Analysis:** Consider partitioning the representation $Z$ into blocks or analyzing individual feature dimensions $Z_i$. We can assess the stability of the relationship $P(Y|Z_i)$ across domains.
    *   Hypothesis: If $Z_i$ primarily captures invariant causal information ($C$), then $P_d(Y|Z_i) \approx P_{d'}(Y|Z_i)$ for different domains $d, d'$.
    *   Hypothesis: If $Z_i$ primarily captures spurious domain-specific information ($S_d$), then $P_d(Y|Z_i)$ may differ significantly across domains, or $Z_i$ might be predictive of the domain index $d$.
    *   Implementation: We can measure the divergence (e.g., Jensen-Shannon Divergence or Maximum Mean Discrepancy - MMD) between conditional distributions $P_d(Y|Z_i)$ estimated across pairs of domains. Features $Z_i$ with consistently low divergence are potentially "causal", while those with high divergence are potentially "spurious". Alternatively, we can train auxiliary classifiers to predict $Y$ from $Z_i$ for each domain and measure the variance of their parameters or performance. Another approach is to test for conditional independence: is $Y \perp d | Z_i$? If this holds, $Z_i$ might be causally sufficient.

2.  **Gradient Stability Analysis:** Inspired by Invariant Risk Minimization (IRM) (Arjovsky et al., 2019), causal predictors should yield stable gradients. We can analyze the stability of $\nabla_Z \mathcal{L}(h_\phi(Z), Y)$ across domains. Features contributing to unstable gradients might be spurious.

**Output ($\mathcal{G}$):** The module $\mathcal{C}$ will output a set of constraints $\mathcal{G}$. This could take the form of:
*   A partition of the representation indices into potentially causal $\mathcal{I}_C$ and potentially spurious $\mathcal{I}_S$.
*   A set of stability scores $s_i$ for each feature $Z_i$.
*   A set of conditional independence constraints $(A \perp B | C)_d$ that hold consistently across domains $d$.

This inference process might run once before training or be updated iteratively as the representation $f_\theta$ evolves. We will initially explore a pre-training inference step followed by fixed constraints during main training for simplicity.

### 3.4 Constraint-Aware Representation Learning and Optimization
Given the constraints $\mathcal{G}$, we modify the standard ERM objective. The total loss function will be:
$$L_{total}(\theta, \phi) = L_{ERM}(\theta, \phi) + \lambda L_{CausalConstraint}(\theta, \phi; \mathcal{G})$$

where $L_{ERM}$ is the average supervised loss over all source domains:
$$L_{ERM}(\theta, \phi) = \frac{1}{D} \sum_{d=1}^D \mathbb{E}_{(x,y) \sim P_d} [\ell(h_{\phi}(f_{\theta}(x)), y)]$$
with $\ell$ being a standard loss like cross-entropy. $\lambda$ is a hyperparameter balancing the two terms.

The crucial part is the $L_{CausalConstraint}$ term, which enforces the constraints $\mathcal{G}$. We propose several potential instantiations based on the inferred constraints:

1.  **Feature Disentanglement Regularization:** If $\mathcal{G}$ provides a partition $Z = (Z_C, Z_S)$, where $Z_C = Z[\mathcal{I}_C]$ and $Z_S = Z[\mathcal{I}_S]$:
    *   **Invariance for $Z_C$:** Penalize domain-predictability of $Z_C$. Use Gradient Reversal Layer (GRL) (Ganin et al., 2016) or MMD between distributions $P_d(Z_C)$ across domains.
        $$L_{Inv}(Z_C) = \sum_{d \neq d'} MMD(P_d(Z_C), P_{d'}(Z_C))$$
    *   **Information Minimization for $Z_S$:** Penalize mutual information between $Z_S$ and $Y$, encouraging the predictor $h_\phi$ to ignore $Z_S$.
        $$L_{InfoMin}(Z_S, Y) = I(Z_S; Y)$$ (estimated using methods like MINE (Belghazi et al., 2018)).
    *   $L_{CausalConstraint} = w_1 L_{Inv}(Z_C) + w_2 L_{InfoMin}(Z_S, Y)$

2.  **Conditional Distribution Alignment:** If $\mathcal{G}$ identifies features $Z_i$ with stable relationships $P(Y|Z_i)$:
    *   Enforce alignment of these conditional distributions across domains.
    *   Could use contrastive methods (like Miao et al., 2022) adapted based on $\mathcal{G}$, or align conditional generators adversarially.
    *   Example: Minimize divergence between predictors based on stable features across domains. Let $h_{\phi, i}$ be a predictor using only $Z_i$.
        $$L_{CondAlign} = \sum_{i \in \mathcal{I}_C} \sum_{d \neq d'} D_{KL}(\mathcal{N}(h_{\phi, i}(Z_{i,d}), \sigma^2) || \mathcal{N}(h_{\phi, i}(Z_{i,d'}), \sigma^2))$$ (Assuming Gaussian output for simplicity, adaptable for classification).

3.  **Causal Effect Stability:** Inspired by Wang et al. (2021), if we can perform interventions (or approximate them):
    *   Enforce that the estimated causal effect of stable features $Z_C$ on $Y$ (e.g., difference in prediction when changing $Z_C$ while keeping $Z_S$ fixed) is consistent across domains. This requires careful definition of interventions in the latent space.

We will initially focus on **Instantiation 1 (Feature Disentanglement)**, as it provides a clear mechanism for incorporating the causal/spurious distinction derived from $\mathcal{C}$. The partition $\mathcal{I}_C, \mathcal{I}_S$ will be determined by the stability analysis (e.g., features with stability scores above/below a threshold).

### 3.5 Algorithmic Steps (CSAIL with Feature Disentanglement)

1.  **Input:** Multi-domain data $\{(X_d, Y_d)\}_{d=1}^D$. Hyperparameters $\lambda, w_1, w_2$.
2.  **Phase 1: Constraint Inference (Pre-training or Initial Epochs):**
    a. Train an initial feature extractor $f_{\theta_0}$ and predictor $h_{\phi_0}$ using ERM.
    b. Use the Causal Constraint Inference Module $\mathcal{C}$ on the representations $Z_d = f_{\theta_0}(X_d)$ ( potentially using intermediate layer activations too) and labels $Y_d$ across domains $d=1...D$.
    c. Compute stability scores $s_i$ for features (or feature blocks) $Z_i$.
    d. Define the partition $\mathcal{I}_C = \{i | s_i > \tau\}$, $\mathcal{I}_S = \{i | s_i \le \tau\}$ based on a threshold $\tau$. Output $\mathcal{G} = (\mathcal{I}_C, \mathcal{I}_S)$.
3.  **Phase 2: Constraint-Aware Training:**
    a. Initialize (or continue training) $f_{\theta}$ and $h_{\phi}$. Optionally add auxiliary heads (e.g., domain classifier for GRL on $Z_C$, information estimator for $Z_S$).
    b. For each training batch:
        i. Sample data from source domains.
        ii. Compute representations $Z = f_{\theta}(X)$, partition into $Z_C = Z[\mathcal{I}_C]$ and $Z_S = Z[\mathcal{I}_S]$.
        iii. Compute predictions $\hat{Y} = h_{\phi}(Z)$.
        iv. Compute $L_{ERM} = \frac{1}{D} \sum_d \ell(\hat{Y}_d, Y_d)$.
        v. Compute $L_{Inv}(Z_C)$ (e.g., using MMD or reversed gradient from a domain classifier on $Z_C$).
        vi. Compute $L_{InfoMin}(Z_S, Y)$ (e.g., using a mutual information estimator).
        vii. Compute $L_{CausalConstraint} = w_1 L_{Inv}(Z_C) + w_2 L_{InfoMin}(Z_S, Y)$.
        viii. Compute total loss $L_{total} = L_{ERM} + \lambda L_{CausalConstraint}$.
        ix. Backpropagate gradients and update $\theta, \phi$ (and parameters of auxiliary heads).
4.  **Output:** Trained model $(f_{\theta}, h_{\phi})$.

### 3.6 Experimental Design

*   **Datasets & Protocol:** Use PACS, VLCS, OfficeHome, TerraIncognita following the DomainBed leave-one-domain-out training/testing protocol. This ensures fair comparison with prior work.
*   **Baselines:**
    *   ERM (Pooled source domains).
    *   IRM (Arjovsky et al., 2019)
    *   VREx (Krueger et al., 2021)
    *   GroupDRO (Sagawa et al., 2019)
    *   Mixup (Yan et al., 2020)
    *   MLDG (Li et al., 2018)
    *   CORAL (Sun & Saenko, 2016)
    *   Recent Causal DG: CCM (Miao et al., 2022), CIRL (Lv et al., 2022) (if reproducible implementations are available).
*   **Implementation Details:** Use standard backbones (e.g., ResNet-18, ResNet-50) pretrained on ImageNet, consistent with DomainBed practices. Use Adam optimizer with hyperparameters selected based on performance on a validation set constructed from source domains (e.g., holding out 20% of each source domain). Report results averaged over multiple random seeds (e.g., 3-5 seeds).
*   **Evaluation Metrics:**
    *   **Primary:** Average accuracy across all target domains.
    *   **Secondary:** Worst-domain accuracy (to assess robustness against the most challenging shift). Accuracy per target domain.
*   **Ablation Studies:**
    1.  **CSAIL vs. ERM:** Demonstrate the benefit of the causal constraints ($\lambda=0$).
    2.  **Impact of Constraint Components:** Evaluate variants using only $L_{Inv}(Z_C)$ ($\lambda > 0, w_2=0$) or only $L_{InfoMin}(Z_S, Y)$ ($\lambda > 0, w_1=0$).
    3.  **Sensitivity to $\lambda, w_1, w_2, \tau$:** Analyze how performance changes with different hyperparameter values.
    4.  **Effectiveness of Constraint Inference:** Compare different methods for generating $\mathcal{G}$ (e.g., stability vs. CI tests vs. gradient stability) if time permits. Assess the quality of the inferred partition $\mathcal{I}_C, \mathcal{I}_S$.
*   **Representation Analysis:**
    1.  **Visualization:** Use t-SNE or UMAP to visualize the learned representations $Z$, colored by domain and class, for both CSAIL and ERM. Assess if CSAIL achieves better domain alignment for $Z_C$ and class separation.
    2.  **Domain Invariance Metric:** Quantify domain invariance of $Z_C$ vs $Z_S$ using domain classification accuracy or MMD scores.
    3.  **Predictive Power:** Measure how predictive $Z_C$ and $Z_S$ are of $Y$ and domain $d$. Expect $Z_C$ to be predictive of $Y$ and invariant to $d$, while $Z_S$ might be predictive of $d$ and less predictive of $Y$.

## **4. Expected Outcomes & Impact**

### 4.1 Expected Outcomes
We anticipate the following outcomes from this research:

1.  **A Functional CSAIL Framework:** A complete implementation of the proposed framework, including the Causal Constraint Inference Module and the Constraint-Aware Training procedure integrated with standard deep learning pipelines.
2.  **State-of-the-Art Performance:** We expect CSAIL to outperform the ERM baseline significantly on standard DG benchmarks. We hypothesize it will also achieve competitive or superior performance compared to existing SOTA DG methods, particularly on datasets known to contain strong spurious correlations (e.g., TerraIncognita, potentially PACS).
3.  **Demonstrated Importance of Causal Constraints:** Ablation studies are expected to show that incorporating the causal constraints ($\lambda > 0$) is crucial for performance gains, and that both invariance enforcement on causal features ($L_{Inv}$) and mitigation of spurious features ($L_{InfoMin}$) contribute positively.
4.  **Improved Representation Properties:** Analysis of the learned representations is expected to quantitatively and qualitatively demonstrate that CSAIL learns features $Z_C$ that are more domain-invariant and aligned with class labels, while effectively isolating domain-specific information in $Z_S$, compared to representations learned by ERM or other baselines.
5.  **Insights into Causal DG:** The research will provide empirical evidence regarding the feasibility and effectiveness of using stability across domains as a proxy for causal structure in the context of deep representation learning for DG. It will also highlight the challenges and limitations, such as the accuracy of the inferred constraints and the choice of regularization techniques.

### 4.2 Impact
This research has the potential for significant impact:

*   **Methodological Advancement:** It offers a concrete approach to integrating causal reasoning (specifically, structure inference via multi-domain stability analysis) into deep learning for robustness. If successful, it will provide a strong empirical case for causality as a key ingredient for DG, directly contributing to the workshop's theme.
*   **Practical Applications:** By improving OOD generalization, CSAIL could lead to more reliable and trustworthy AI systems in safety-critical domains like healthcare (generalizing across hospitals/scanners), autonomous driving (generalizing across weather/locations), and finance (adapting to market changes).
*   **Theoretical Implications:** While primarily empirical, the results could motivate further theoretical investigation into why and under what precise assumptions leveraging inferred causal structure aids generalization. It relates to ongoing theoretical work on invariant prediction and causal representation learning.
*   **Future Research Directions:** This work could open up avenues for exploring more sophisticated causal discovery techniques within DG, investigating the interplay between observational and interventional data (if available), and developing adaptive methods where causal constraints are refined during training.

In conclusion, this research proposes a principled approach, CSAIL, to tackle the domain generalization problem by explicitly identifying and leveraging invariant causal structures inferred from multi-domain data. By focusing on the stability of mechanisms rather than superficial correlations, we aim to develop models with substantially improved robustness to distribution shifts, thereby advancing the reliability of machine learning systems in complex, real-world environments.

---
**References** *(A selection based on proposal content and lit review)*

*   Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. *arXiv preprint arXiv:1907.02893*.
*   Belghazi, M. I., Baratin, A., Rajeshwar, S., Ozair, S., Bengio, Y., Courville, A., & Hjelm, D. (2018). Mutual Information Neural Estimation. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.
*   Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1-35.
*   Gulrajani, I., & Lopez-Paz, D. (2020). In Search of Lost Domain Generalization. *arXiv preprint arXiv:2007.01434*. (ICLR 2021)
*   Kher, K. V., Badisa, L. V. S. M., Harsha, K. V. D. S., Sowmya, C. G., & Jagarlapudi, S. (2025). Unsupervised Structural-Counterfactual Generation under Domain Shift. *arXiv preprint arXiv:2502.12013*.
*   Krueger, D., Caballero, E., Jacobsen, J. H., Zhang, A., Binas, J., Krueger F., ... & Zhang, C. (2021). Out-of-Distribution Generalization via Risk Extrapolation (V-REx). *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
*   Li, D., Yang, Y., Song, Y. Z., & Hospedales, T. M. (2018). Learning to Generalize: Meta-Learning for Domain Generalization. *Proceedings of the AAAI Conference on Artificial Intelligence*.
*   Lv, F., Liang, J., Li, S., Zang, B., Liu, C. H., Wang, Z., & Liu, D. (2022). Causality Inspired Representation Learning for Domain Generalization. *arXiv preprint arXiv:2203.14237*. (CVPR 2022)
*   Miao, Q., Yuan, J., & Kuang, K. (2022). Domain Generalization via Contrastive Causal Learning. *arXiv preprint arXiv:2210.02655*.
*   Muandet, K., Balduzzi, D., & Schölkopf, B. (2013). Domain Generalization via Invariant Feature Representation. *Proceedings of the 30th International Conference on Machine Learning (ICML)*.
*   Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge university press.
*   Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT press.
*   Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2019). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. *arXiv preprint arXiv:1911.08731*. (ICLR 2020)
*   Schölkopf, B., Janzing, D., Peters, J., Sgouritsa, E., Zhang, K., & Mooij, J. (2012). On causal and anticausal learning. *Proceedings of the 29th International Conference on Machine Learning (ICML)*.
*   Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. *European Conference on Computer Vision (ECCV) Workshops*.
*   Wang, Y., Liu, F., Chen, Z., Lian, Q., Hu, S., Hao, J., & Wu, Y. C. (2021). Contrastive ACE: Domain Generalization Through Alignment of Causal Mechanisms. *arXiv preprint arXiv:2106.00925*.
*   Yan, S., Song, H., Li, N., Zhang, L., & Kratschmar, T. (2020). Improve Domain Generalization With Mixup. *Domain Adaptation and Representation Transfer, and Distributed and Collaborative Learning (DART) workshop at CVPR*.