Okay, here is a research proposal based on the provided task description, research idea (AIFS), and literature review.

---

## **1. Title:**

**Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS): Enhancing Model Robustness by Automatically Discounting Unknown Spurious Correlations**

## **2. Introduction**

### **2.1 Background**

Deep learning models have achieved remarkable success across various domains, yet their reliability in real-world applications is often undermined by their tendency to exploit spurious correlations present in the training data (Geirhos et al., 2020; Sagawa et al., 2020). Spurious correlations are patterns that hold statistically within the training distribution but do not reflect the underlying causal mechanisms of the task. Models relying on these "shortcuts" exhibit poor generalization when deployed in environments with slightly different data distributions, particularly affecting performance on under-represented groups or minority populations where the spurious correlation does not hold (Arjovsky et al., 2019; Ye et al., 2024). This phenomenon, often exacerbated by the implicit biases of optimization algorithms like Stochastic Gradient Descent (SGD) and model architectures themselves, represents a fundamental challenge across all branches of AI, hindering the development of truly robust and trustworthy systems (Workshop Overview). Shortcut learning is not merely a niche problem but a key obstacle to understanding how deep models learn and generalize effectively.

Current approaches to mitigate spurious correlations often rely on explicit knowledge of these undesirable patterns, typically requiring group labels that identify subgroups where the correlation between the spurious feature and the target label differs (Sagawa et al., 2020; Sohoni et al., 2020). Methods like Group Distributionally Robust Optimization (Group DRO) aim to optimize worst-group performance but necessitate costly and often unavailable group annotations. Furthermore, human annotation may fail to capture subtle or unintuitive spurious correlations that models might exploit (Workshop Objectives; Sun et al., 2023). While techniques like feature reweighting (Izmailov et al., 2022; Hameed et al., 2024) or regularization (Wen et al., 2025) show promise, they often focus on post-hoc adjustments or specific layers, and may not dynamically adapt to the most problematic features learned during training. Recent work has explored meta-learning (Zheng et al., 2024) or teacher-student approaches (Mitchell et al., 2024) to implicitly identify spurious cues, but these can introduce significant complexity or rely on auxiliary models (e.g., VLMs). There remains a critical need for methods that can automatically identify and mitigate *unknown* spurious correlations *during* training, without relying on explicit group supervision, thereby promoting invariance to these nuisance factors and encouraging reliance on causal features (Yao et al., 2024).

### **2.2 Research Gap and Proposed Solution**

The primary research gap lies in developing robustification techniques that operate effectively *without* prior knowledge or annotation of spurious attributes. Existing methods are often constrained by the need for group labels, limiting their applicability in real-world scenarios where such information is scarce or the relevant spurious features are unknown. Furthermore, many approaches lack mechanisms to *adaptively* focus mitigation efforts on the specific spurious features that a particular model instance learns to exploit during its training trajectory.

To address this gap, we propose **Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS)**. The core idea of AIFS is to integrate a generative intervention loop directly into the model training process. AIFS leverages synthetic perturbations applied within the model's latent representation space to simulate distributional shifts related to potential spurious factors. Crucially, these interventions are guided *adaptively*: the system identifies latent dimensions most influential on the model's predictions under perturbation (i.e., dimensions likely corresponding to non-robust features) and focuses subsequent interventions on these sensitive dimensions. By optimizing the model to maintain prediction invariance across these targeted synthetic interventions, AIFS encourages the model to discount sensitive (likely spurious) dimensions and rely more heavily on invariant (likely causal) features. AIFS requires no group labels or explicit knowledge of spurious attributes, making it broadly applicable.

### **2.3 Research Objectives**

This research aims to:

1.  **Develop the AIFS Framework:** Formulate and implement the AIFS methodology, including the latent intervention module, the adaptive mask generation mechanism based on sensitivity analysis, and the dual-objective loss function promoting task performance and intervention invariance.
2.  **Validate AIFS Effectiveness:** Empirically evaluate the ability of AIFS to improve model robustness against known spurious correlations on standard benchmarks (e.g., image classification, tabular data), specifically measuring worst-group accuracy *without* using group labels during training.
3.  **Compare AIFS with Baselines:** Benchmark AIFS against standard Empirical Risk Minimization (ERM) and relevant state-of-the-art methods for mitigating spurious correlations (including methods that use group labels, serving as an upper bound reference).
4.  **Analyze AIFS Mechanisms:** Investigate the internal workings of AIFS, analyzing how the adaptive intervention strategy identifies and targets latent dimensions associated with spurious features, and how this impacts the learned representations.

### **2.4 Significance**

This research holds significant potential for advancing the field of robust machine learning. By providing a method to automatically mitigate unknown spurious correlations without group labels, AIFS could drastically improve the reliability and fairness of AI systems deployed in diverse and shifting environments. Its modality-agnostic design based on latent space intervention suggests broad applicability across various data types (images, text, tabular, etc.) and potentially foundation models (Workshop Topics). Successfully demonstrating AIFS would contribute a novel robustification technique, offer insights into the foundations of shortcut learning by observing the adaptive process, and provide a practical tool for building more dependable AI, aligning directly with the core goals of the Workshop on Spurious Correlation and Shortcut Learning.

## **3. Methodology**

### **3.1 Overall Research Design**

AIFS integrates an adaptive intervention mechanism into a standard deep learning training pipeline. The framework consists of four main components: (1) a base encoder network $E$, (2) a latent intervention module $\mathcal{I}$, (3) a classifier network $C$, and (4) an adaptive masking strategy based on sensitivity analysis. The training proceeds iteratively, optimizing the model to perform the main task while simultaneously encouraging invariance to targeted synthetic perturbations in the latent space.

Let $x$ be an input sample and $y_{true}$ be its corresponding label. The process within a single training step (or periodically incorporating the adaptive update) is as follows:

1.  **Encoding:** Obtain the latent representation $z = E(x)$, where $z \in \mathbb{R}^d$.
2.  **Intervention:** Apply a synthetic intervention using the module $\mathcal{I}$ conditioned on a learnable or dynamically updated mask $M \in [0, 1]^d$ to produce a perturbed representation $z' = \mathcal{I}(z, M)$.
3.  **Prediction:** Obtain predictions from the original and perturbed representations: $p = C(z)$ and $p' = C(z')$, where $p, p'$ are typically output logits or probabilities.
4.  **Loss Calculation:** Compute a combined loss function incorporating task performance and invariance.
5.  **Parameter Update:** Update the parameters of $E$ and $C$ via gradient descent on the combined loss.
6.  **Sensitivity Analysis & Mask Update (Periodic):** Periodically analyze the sensitivity of the model's predictions or loss to perturbations in different latent dimensions and update the mask $M$ to focus future interventions on highly sensitive dimensions.

### **3.2 Data Collection and Preparation**

We will utilize existing, well-established benchmark datasets known to contain strong spurious correlations, primarily focusing on scenarios where group labels are available for *evaluation* but **not** used during *training* AIFS. This allows us to rigorously measure robustness gains (specifically worst-group accuracy). Potential datasets include:

1.  **Waterbirds** (Sagawa et al., 2020): Images of landbirds and waterbirds, where the background (land/water) acts as a strong spurious cue correlated with the bird type. Group labels define (bird type, background type) combinations.
2.  **CelebA** (Liu et al., 2015): Facial attribute dataset. We will use tasks like hair color prediction (e.g., Blond vs. Non-Blond), where gender is often spuriously correlated. Group labels define (hair color, gender) combinations.
3.  **CivilComments** (Borkan et al., 2019): Text dataset for toxicity detection, where mentions of certain demographic identities are spuriously correlated with toxicity. Group labels define (toxicity, identity mention) combinations.
4.  **Adult Income** (UCI ML Repository): A standard tabular dataset where attributes like gender or race can be spuriously correlated with income level.

Data preprocessing will follow standard practices for each dataset (e.g., image resizing/normalization for vision tasks, text tokenization for NLP). Crucially, group labels will be withheld from the AIFS training algorithm and only used for post-training evaluation of worst-group performance.

### **3.3 Algorithmic Steps: AIFS Framework**

**1. Base Model Architecture:**
We assume a standard deep learning architecture composed of an encoder $E_{\theta_E}: \mathcal{X} \to \mathbb{R}^d$ and a classifier $C_{\theta_C}: \mathbb{R}^d \to \mathbb{R}^K$ (where $K$ is the number of classes), parameterized by $\theta_E$ and $\theta_C$. For image tasks, $E$ could be a ResNet backbone; for text, a BERT-based encoder. $C$ is typically a linear layer followed by a softmax. The encoder $E$ can be pre-trained on a large dataset (e.g., ImageNet, standard language model pre-training) or trained from scratch alongside $C$.

**2. Latent Intervention Module ($\mathcal{I}$):**
Given a latent representation $z = E_{\theta_E}(x)$ and an intervention mask $M \in [0, 1]^d$, the intervention module generates a perturbed representation $z'$. A simple and flexible intervention is additive noise scaled by the mask:
$$z' = \mathcal{I}(z, M) = z + M \odot \epsilon$$
where $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$ is random noise sampled independently for each instance, $\odot$ denotes element-wise multiplication, and $\sigma$ is a hyperparameter controlling the intervention strength. Other interventions, like applying random affine transformations to the masked subspace, can also be explored. The mask $M$ determines *which* latent dimensions are subject to intervention and *to what degree*.

**3. Dual-Objective Loss Function:**
The total loss function combines the standard task loss with an invariance-promoting loss:
$$\mathcal{L}_{total}(x, y_{true}) = \mathcal{L}_{task}(C_{\theta_C}(z), y_{true}) + \lambda_{inv} \mathcal{L}_{inv}(C_{\theta_C}(z), C_{\theta_C}(z'))$$
where:
*   $\mathcal{L}_{task}$ is the primary objective function (e.g., Cross-Entropy loss): $\mathcal{L}_{CE}(p, y_{true})$.
*   $\mathcal{L}_{inv}$ penalizes differences between predictions on the original and perturbed latent vectors. A suitable choice is the Kullback-Leibler (KL) divergence between the output distributions (softmax applied to logits $p, p'$): $\mathcal{L}_{KL}(softmax(p) || softmax(p'))$. Alternatively, Mean Squared Error (MSE) on the logits could be used: $\mathcal{L}_{MSE}(p, p')$.
*   $\lambda_{inv}$ is a hyperparameter balancing the task performance and the invariance objective.

**4. Adaptive Masking Strategy:**
The core novelty of AIFS lies in adaptively updating the mask $M$ to focus interventions on dimensions the model relies on in a non-robust way. This is achieved periodically (e.g., every few epochs):

*   **Sensitivity Calculation:** We need to estimate how sensitive the model's predictions (or loss) are to perturbations in each latent dimension $j \in \{1, ..., d\}$. A practical approach is to use gradient information. For instance, we can compute the expected magnitude of the gradient of the invariance loss with respect to the *unperturbed* latent variable $z$, specifically focusing on the contribution from each dimension:
    $$S_j = \mathbb{E}_{x, \epsilon} \left| \frac{\partial \mathcal{L}_{inv}(C_{\theta_C}(z), C_{\theta_C}(z + M \odot \epsilon))}{\partial z_j} \right|$$
    Alternatively, sensitivity could be measured by the gradient of the task loss w.r.t. latent dimensions, or by directly observing the change in loss/prediction when perturbing individual dimensions. The expectation $\mathbb{E}$ can be approximated by averaging over a mini-batch or a larger set of samples.

*   **Mask Update Rule:** Based on the sensitivity scores $S = [S_1, ..., S_d]$, we update the mask $M$. A simple strategy is to select the top-$k\%$ of dimensions with the highest sensitivity scores and set their corresponding mask values $M_j$ to 1 (or a high value), while setting others to 0 (or a low value).
    $$ M_j = \begin{cases} 1 & \text{if } S_j \text{ is in the top } k\% \text{ of sensitivity scores} \\ 0 & \text{otherwise} \end{cases} $$
    The percentage $k$ can be fixed or annealed during training. More sophisticated update rules could involve assigning continuous mask values proportional to sensitivity, or using techniques like Exponential Moving Average (EMA) to smooth the sensitivity scores over time. The initial mask $M_0$ could be uniform ($M_j=1$ for all $j$) or randomly initialized.

**5. Training Algorithm:**
The overall training proceeds as follows:
```
Initialize E_theta_E, C_theta_C, M (e.g., M_j = 1 for all j)
For each training epoch t = 1 to T:
  For each mini-batch (x_batch, y_batch):
    z_batch = E_theta_E(x_batch)
    epsilon_batch ~ N(0, sigma^2 I_d)
    z'_batch = z_batch + M_batch * epsilon_batch  # M_batch might be same M for batch, or per-instance
    p_batch = C_theta_C(z_batch)
    p'_batch = C_theta_C(z'_batch)
    L_task = compute_task_loss(p_batch, y_batch)
    L_inv = compute_invariance_loss(p_batch, p'_batch)
    L_total = L_task + lambda_inv * L_inv
    Compute gradients of L_total w.r.t. theta_E, theta_C
    Update theta_E, theta_C using an optimizer (e.g., Adam)

  If t % update_frequency == 0:
    Compute sensitivity scores S based on current model (E_theta_E, C_theta_C) over a sample of data
    Update mask M based on S using the chosen update rule
```

### **3.4 Experimental Design and Validation**

**1. Baselines:**
We will compare AIFS against the following baselines:
*   **ERM (Empirical Risk Minimization):** Standard training using only $\mathcal{L}_{task}$. Represents the lower bound for robustness.
*   **Group DRO (Sagawa et al., 2020):** Trained using group labels to optimize worst-group loss. Represents a strong baseline/upper bound when group labels are available (AIFS does *not* use these labels for training).
*   **Last Layer Retraining (Izmailov et al., 2022):** Train ERM, then retrain the final layer on a group-balanced subset. AIFS aims for end-to-end robustness.
*   **Invariant Risk Minimization (IRM, Arjovsky et al., 2019):** A foundational approach attempting to learn invariant predictors across environments (often simulated by grouping). Will require simulating environments if group labels are withheld.
*   **Potential Label-Free Baselines:** Methods like SPUME (Zheng et al., 2024) or ULE (Mitchell et al., 2024) if feasible to implement/adapt, to compare against other techniques that do not require pre-defined group labels.

**2. Evaluation Metrics:**
The primary metrics will focus on robustness and generalization under distribution shift caused by spurious correlations:
*   **Worst-Group Accuracy (WGA):** Accuracy calculated on the subgroup of the test set for which the model performs worst. This is the key metric for robustness against spurious correlations, calculated using ground-truth group labels *only* at test time.
*   **Average Accuracy (AvgAcc):** Standard accuracy calculated over the entire test set.
*   **Accuracy Gap:** The difference between AvgAcc and WGA (AvgAcc - WGA). A smaller gap indicates better fairness/robustness across groups.

**3. Evaluation Protocol:**
For each dataset, models (AIFS and baselines) will be trained on the training split. A validation split will be used for hyperparameter tuning (e.g., learning rate, $\lambda_{inv}$, $\sigma$, $k$, update frequency for AIFS). Final performance will be reported on the held-out test split. We will report AvgAcc and WGA. Multiple runs with different random seeds will be performed to ensure statistical significance of the results.

**4. Ablation Studies:**
To understand the contribution of different components of AIFS, we will conduct ablation studies:
*   **Impact of $\lambda_{inv}$:** Varying the weight of the invariance loss.
*   **Impact of Intervention Strength $\sigma$:** Assessing sensitivity to the noise level.
*   **Impact of Adaptivity:** Comparing adaptive masking vs. static masking (e.g., $M_j=1$ for all $j$ always) or random masking.
*   **Impact of Mask Update Frequency:** How often the sensitivity analysis and mask update should occur.
*   **Impact of Sensitivity Measure:** Comparing different ways to calculate sensitivity scores $S$.

**5. Analysis of Learned Representations:**
We will employ techniques to analyze the learned representations ($z$) and the behavior of the adaptive mask:
*   **Correlation Analysis:** Calculate the correlation between the identified high-sensitivity latent dimensions (where $M_j \approx 1$) and known spurious attributes (using ground-truth group labels post-hoc). This aims to verify if AIFS correctly identifies dimensions related to spurious features.
*   **Visualization:** Use dimensionality reduction techniques (e.g., t-SNE, UMAP) to visualize the latent space $z$ for different groups, comparing representations learned by ERM vs. AIFS. We expect AIFS representations to show less clustering based on spurious attributes.
*   **Interpretability Methods:** Apply feature attribution methods (e.g., Integrated Gradients, SHAP - cf. Sun et al., 2023) to both the input space and the latent space to understand which features drive predictions for AIFS vs. ERM, especially on minority group examples.

## **4. Expected Outcomes & Impact**

**Expected Outcomes:**

1.  **A Fully Developed AIFS Framework:** A documented and implemented version of the AIFS algorithm applicable to standard deep learning pipelines.
2.  **Demonstrated Robustness Improvement:** Empirical evidence showing that AIFS significantly improves Worst-Group Accuracy compared to standard ERM training on benchmark datasets with known spurious correlations, without requiring group labels during training.
3.  **Competitive Performance:** AIFS performance potentially approaching or, in some scenarios, even matching methods that utilize group labels (like Group DRO), demonstrating the effectiveness of unsupervised identification and mitigation of spurious cues.
4.  **Insights into Adaptive Intervention:** Analysis results elucidating how the adaptive masking strategy evolves during training, which latent dimensions are targeted, and how this correlates with known spurious attributes, contributing to the understanding of shortcut learning dynamics.
5.  **Comparative Benchmarking:** Clear quantitative comparison of AIFS against relevant state-of-the-art methods, establishing its position in the landscape of robust learning techniques.

**Impact:**

*   **Advancement in Robust AI:** AIFS offers a novel and practical approach to building more reliable AI systems, particularly valuable in label-scarce or safety-critical domains where unknown spurious correlations pose significant risks.
*   **Broader Applicability:** The label-free and modality-agnostic nature of AIFS could enable its application to a wide range of problems, including robustness challenges in large language models (LLMs) and multimodal models, areas highlighted as important by the workshop.
*   **Contribution to Foundational Understanding:** By analyzing the adaptive mechanism and its effect on representations, this research can shed light on how models learn and utilize features, contributing to the fundamental understanding of generalization and shortcut learning in deep neural networks.
*   **Practical Tool for Practitioners:** If successful, AIFS could provide ML practitioners with a valuable tool to enhance model robustness without the prohibitive cost and limitations of extensive group annotation.
*   **Stimulating Further Research:** This work may inspire further research into adaptive and intervention-based methods for causal representation learning and robust optimization in various learning paradigms (supervised, self-supervised, reinforcement learning).

In conclusion, the proposed research on AIFS directly addresses the critical challenge of spurious correlations outlined by the workshop, offering a novel solution that operates without group supervision and adaptively targets model-specific vulnerabilities. We expect this work to yield significant improvements in model robustness and contribute valuable insights into the foundations of reliable machine learning.

---
**References** (Incorporating provided literature review and key foundational papers)

*   Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant Risk Minimization. *arXiv preprint arXiv:1907.02893*.
*   Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. *Companion Proceedings of The 2019 World Wide Web Conference*.
*   Chen, H., Yang, X., & Yang, Q. (2023). Towards Causal Representation Learning and Deconfounding from Indefinite Data. *arXiv preprint arXiv:2305.02640*.
*   Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*.
*   Hameed, H. W., Nanfack, G., & Belilovsky, E. (2024). Not Only the Last-Layer Features for Spurious Correlations: All Layer Deep Feature Reweighting. *arXiv preprint arXiv:2409.14637*.
*   Izmailov, P., Kirichenko, P., Gruver, N., & Wilson, A. G. (2022). On Feature Learning in the Presence of Spurious Correlations. *arXiv preprint arXiv:2210.11369*.
*   Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. *Proceedings of the IEEE international conference on computer vision*.
*   Mitchell, J., Martínez del Rincón, J., & McLaughlin, N. (2024). UnLearning from Experience to Avoid Spurious Correlations. *arXiv preprint arXiv:2409.02792*.
*   Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. *International Conference on Learning Representations (ICLR)*.
*   Sohoni, N. S., Sagawa, S., Khetan, A., Raghunathan, A., Koh, P. W., Liang, P., ... & Hashimoto, T. B. (2020). No Regret Left Behind: Implicit Regularization via Greedy Learning Dynamics. *arXiv preprint arXiv:2011.10916*.
*   Sun, S., Koch, L. M., & Baumgartner, C. F. (2023). Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations?. *arXiv preprint arXiv:2307.12344*.
*   Varma, M., Delbrouck, J. B., Chen, Z., Chaudhari, A., & Langlotz, C. (2024). RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models. *arXiv preprint arXiv:2411.04097*.
*   Wen, T., Wang, Z., Zhang, Q., & Lei, Q. (2025). Elastic Representation: Mitigating Spurious Correlations for Group Robustness. *arXiv preprint arXiv:2502.09850*. ([Note: Year adjusted based on typical pre-print to publication lag potential, assuming 2024 preprint])
*   Yao, D., Rancati, D., Cadei, R., Fumero, M., & Locatello, F. (2024). Unifying Causal Representation Learning with the Invariance Principle. *arXiv preprint arXiv:2409.02772*.
*   Ye, W., Zheng, G., Cao, X., Ma, Y., & Zhang, A. (2024). Spurious Correlations in Machine Learning: A Survey. *arXiv preprint arXiv:2402.12715*.
*   Zheng, G., Ye, W., & Zhang, A. (2024). Spuriousness-Aware Meta-Learning for Learning Robust Classifiers. *arXiv preprint arXiv:2406.10742*.