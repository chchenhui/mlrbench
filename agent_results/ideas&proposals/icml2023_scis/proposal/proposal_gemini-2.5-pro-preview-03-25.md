Okay, here is a research proposal draft based on the provided task description, research idea, and literature review.

## Research Proposal

**1. Title:** **Adversarial Counterfactual Augmentation for Mitigating Spurious Correlations without Group Annotations**

**2. Introduction**

**2.1 Background**
Machine learning models, despite achieving superhuman performance on various benchmarks, often exhibit unexpected failures when deployed in real-world scenarios. A primary reason for this brittleness is their tendency to exploit *spurious correlations* – patterns that are predictive in the training data distribution but are not causally related to the target label (Geirhos et al., 2020; Arjovsky et al., 2019). As highlighted by the workshop's call, examples abound across domains: models relying on scanner artifacts in medical imaging instead of disease pathology (Zech et al., 2018), leveraging lexical artifacts in natural language inference instead of semantic understanding (Gururangan et al., 2018), or exhibiting demographic biases in genetic risk prediction due to ancestral confounding (Martin et al., 2019). These shortcuts lead to poor out-of-distribution (OOD) generalization and instability when encountering data distributions different from the training set.

Addressing this challenge is critical for building reliable and trustworthy AI systems. Significant research effort has focused on developing robust models. Many successful approaches leverage *group annotations* that explicitly identify the spurious attribute (e.g., background type alongside the main object class). Methods like Group Distributionally Robust Optimization (GroupDRO) (Sagawa et al., 2019) aim to minimize the worst-group loss, while Invariant Risk Minimization (IRM) (Arjovsky et al., 2019) seeks representations invariant across predefined environments or groups. However, obtaining such group labels is often expensive, impractical, or even impossible in many real-world applications. This necessitates methods that can enhance robustness *without* relying on explicit group supervision.

Recent work has started exploring group-free robustness techniques. Some approaches attempt to infer pseudo-group labels (e.g., Han & Zou, 2024; Nam et al., 2020) or identify and upweight difficult examples assumed to belong to minority groups (e.g., Ghaznavi et al., 2024; Liu et al., 2021). Others focus on extracting robust subnetworks (Le et al., 2024) or using meta-learning strategies (Zheng et al., 2024a). Counterfactual reasoning and data augmentation have also emerged as promising directions (Reddy et al., 2023; Qin et al., 2024), aiming to expose models to variations that break spurious associations. However, generating meaningful counterfactuals, especially for complex and unknown spurious features, remains a significant hurdle. The challenge lies in modifying only the spurious attribute while preserving the core, label-defining features, without explicit guidance on what constitutes "spurious."

**2.2 Research Objectives**
This research proposes a novel framework, **Adversarial Counterfactual Augmentation (ACA)**, designed to improve model robustness against spurious correlations without requiring group annotations. ACA aims to automate the process of identifying potentially spurious features, generating targeted counterfactual examples that vary these features, and retraining the model to be invariant to them. The core objectives are:

1.  **Develop a pipeline for identifying potentially spurious input features** leveraged by a pre-trained model, using techniques like gradient-based attribution or influence functions, which do not require group labels.
2.  **Design and train conditional generative models** (e.g., based on GANs or Diffusion Models) capable of generating counterfactual data points ($x'$) from original inputs ($x$). These counterfactuals should modify the identified spurious features while preserving the original label ($y$) and core semantic content.
3.  **Formulate a robust training strategy** that utilizes both original data and generated counterfactuals. This involves incorporating a consistency regularization term that encourages the model to produce similar predictions or representations for original-counterfactual pairs $(x, x')$.
4.  **Empirically validate the effectiveness of ACA** in improving worst-group accuracy and OOD generalization on established benchmark datasets known for spurious correlations, comparing its performance against standard Empirical Risk Minimization (ERM) and state-of-the-art group-free robustness methods.
5.  **Analyze the behavior of ACA**, including the quality of generated counterfactuals and the impact of different spurious feature identification techniques, to understand its mechanisms and limitations.

**2.3 Significance**
This research directly addresses a critical bottleneck in deploying reliable machine learning models: the reliance on spurious correlations and the frequent lack of group annotations to mitigate them. By proposing a group-free counterfactual augmentation strategy, ACA offers several significant contributions:

*   **Practical Robustness:** It provides a potentially powerful tool for practitioners to enhance model robustness in scenarios where group labels are unavailable, which is common in real-world datasets across science, industry, and society.
*   **Advancing Group-Free Methods:** It contributes a novel approach to the rapidly growing field of group-free robustness, complementing existing methods by focusing on targeted data augmentation rather than primarily loss re-weighting or group inference.
*   **Bridging Causality and Deep Learning:** It operationalizes ideas from causality (counterfactuals) within a deep learning framework, using generative models to approximate interventions on spurious features.
*   **Addressing Workshop Themes:** The proposed work aligns perfectly with the workshop's solicited topics, particularly "Learning robust models in the presence of spurious correlations," "Methods for discovering and diagnosing spurious correlations" (via feature identification), and exploring relationships with causal ML.
*   **Potential for Broader Impact:** Improved robustness can lead to more reliable and equitable AI systems in sensitive domains like healthcare, autonomous driving (as explored in RL by Ding et al., 2023), and content moderation, reducing the risk of failures due to unforeseen data shifts or reliance on biased correlations.

**3. Methodology**

Our proposed Adversarial Counterfactual Augmentation (ACA) framework consists of three main stages: (1) Spurious Feature Identification, (2) Counterfactual Generation, and (3) Robust Model Retraining.

**3.1 Overall Framework**
Given a dataset $D = \{(x_i, y_i)\}_{i=1}^N$, we first train a standard model $f_{\theta}$ using Empirical Risk Minimization (ERM). This model, potentially reliant on spurious features, serves as the basis for identifying them. Subsequently, ACA proceeds as follows:
1.  **Identify Spurious Features:** Use the trained model $f_{\theta}$ to identify input regions or features most likely contributing to the prediction but potentially being spurious for each example $x_i$. This yields cues $s_i$ (e.g., attention masks, influential feature sets).
2.  **Generate Counterfactuals:** Train a conditional generative model $G$ that takes an input $x_i$, its label $y_i$, and the spurious feature cue $s_i$, and generates a counterfactual $x'_i = G(x_i, y_i, s_i)$. The goal is for $x'_i$ to retain the label $y_i$ but differ from $x_i$ primarily in the regions indicated by $s_i$.
3.  **Retrain with Consistency:** Train a new task model $f_{\phi}$ (potentially initialized from $f_{\theta}$) on an augmented dataset $D_{aug} = D \cup D'$, where $D' = \{(x'_i, y_i)\}_{i=1}^N$. The training objective includes a standard classification loss and a consistency loss encouraging $f_{\phi}(x_i) \approx f_{\phi}(x'_i)$.

**3.2 Stage 1: Spurious Feature Identification**
The goal is to pinpoint input features that the ERM model $f_{\theta}$ relies heavily upon, without knowing *a priori* which are spurious. We hypothesize that features receiving disproportionately high importance scores from attribution methods, especially if they are known to correlate spuriously in the dataset (e.g., backgrounds in object recognition, common words in NLI), are good candidates.

*   **Gradient-based Attribution:** Methods like Integrated Gradients (Sundararajan et al., 2017) or Grad-CAM (Selvaraju et al., 2017) compute the importance of input features (pixels, tokens) for the model's prediction on a specific input $(x_i, y_i)$. We can compute an attribution map $A_i = \text{Attribution}(f_{\theta}, x_i, y_i)$. A mask $M_i$ highlighting potentially spurious features can be derived by thresholding $A_i$ or selecting top-k% features. For example:
    $$ M_i(j) = 1 \quad \text{if} \quad |A_i(j)| \ge \tau \cdot \max_{k} |A_i(k)|, \quad \text{else} \quad 0 $$
    where $j$ indexes features and $\tau$ is a threshold hyperparameter. The mask $M_i$ serves as the spurious feature cue $s_i$.
*   **Influence Functions:** Alternatively, influence functions (Koh & Liang, 2017) can estimate the impact of perturbing or removing specific features or training points on the model's predictions or loss. High-influence features could be flagged as potentially spurious. This is generally more computationally expensive but might provide a different perspective on feature importance.

Crucially, these methods operate solely on the ERM model $f_{\theta}$ and the input data $x_i$, without needing group labels.

**3.3 Stage 2: Adversarial Counterfactual Generation**
We aim to generate a counterfactual $x'_i$ for each $x_i$ such that the label $y_i$ is preserved, but the features identified by the mask $M_i$ are altered. This requires a conditional generative model sensitive to the mask.

*   **Conditional Generative Models:**
    *   **Diffusion Models:** Denoising Diffusion Probabilistic Models (DDPMs) (Ho et al., 2020) have shown remarkable generation quality and controllability. We propose training a conditional diffusion model $p_{\psi}(x' | x, M, y)$ that learns to reverse a noising process, conditioned on the original image $x$, the mask $M$ indicating regions to modify, and the label $y$ to preserve semantics. During sampling, the model would inpaint or modify regions specified by $M$.
    *   **GAN-based Approaches:** Conditional GANs (Mirza & Osindero, 2014) or specialized architectures like CycleGAN (Zhu et al., 2017) could be adapted. For instance, a CycleGAN could learn mappings between images with certain feature types (e.g., different backgrounds), potentially guided by masks $M$. Alternatively, an inpainting GAN could be trained to modify masked regions based on noise or semantic guidance, conditioned on $y$.

*   **Ensuring Label Preservation:** Conditioning on the label $y$ is crucial. Additionally, we might incorporate perceptual losses or auxiliary classifiers during the generator training to ensure $x'_i$ still belongs to class $y_i$. The "adversarial" nature comes from generating examples specifically designed to challenge the classifier's reliance on the features targeted by $M_i$, forcing it to learn more robust representations.

**3.4 Stage 3: Robust Model Retraining**
The final stage involves training a classifier $f_{\phi}$ using both the original data $D$ and the generated counterfactuals $D' = \{(x'_i, y_i)\}$. The objective function combines a standard classification loss (e.g., cross-entropy) with a consistency regularization term:

$$ \mathcal{L}_{total}(f_{\phi}) = \frac{1}{|D_{aug}|} \sum_{(x, y) \in D_{aug}} \mathcal{L}_{CE}(f_{\phi}(x), y) + \lambda \frac{1}{|D|} \sum_{(x_i, y_i) \in D} \mathcal{L}_{cons}(f_{\phi}(x_i), f_{\phi}(x'_i)) $$

where $D_{aug} = D \cup D'$, $\mathcal{L}_{CE}$ is the cross-entropy loss, $\lambda > 0$ is a hyperparameter balancing the two terms, and $\mathcal{L}_{cons}$ is the consistency loss.

*   **Consistency Loss:** This term forces the model $f_{\phi}$ to be invariant to the modifications introduced in the counterfactuals $x'_i$. Several forms can be explored:
    *   **Prediction Consistency:** Minimize the divergence between output probability distributions: $\mathcal{L}_{cons}(p_i, p'_i) = D_{KL}(p_i || p'_i)$, where $p_i = \text{softmax}(f_{\phi}(x_i))$ and $p'_i = \text{softmax}(f_{\phi}(x'_i))$.
    *   **Representation Consistency:** Encourage similar representations in a hidden layer (e.g., the penultimate layer): $\mathcal{L}_{cons}(h_i, h'_i) = 1 - \text{cosine\_similarity}(h_i, h'_i)$, where $h_i$ and $h'_i$ are the feature vectors for $x_i$ and $x'_i$, respectively.

By minimizing this combined loss, $f_{\phi}$ is incentivized to rely less on the features identified as potentially spurious (since they vary between $x_i$ and $x'_i$ while the prediction should remain consistent) and more on the core, invariant features related to the true label $y_i$.

**3.5 Experimental Design**

*   **Datasets:** We will evaluate ACA on standard benchmarks designed to test robustness to spurious correlations:
    *   **Waterbirds** (Sagawa et al., 2019): Classify land birds vs. water birds, where the background (land/water) is spuriously correlated with the class. Contains group labels for evaluation.
    *   **CelebA** (Liu et al., 2015): Predict attributes like hair color (e.g., Blond), where correlations exist with other attributes like gender, often leading to biased predictions. Group labels (e.g., Blond Male, Blond Female) available.
    *   **Colored MNIST** (Arjovsky et al., 2019): MNIST digits where color is spuriously correlated with the digit label in training but decorrelated or anti-correlated at test time.
    *   **CivilComments** (Borkan et al., 2019): Toxicity detection where toxicity is correlated with mentions of certain demographic identity groups. Group labels (identifying comments mentioning specific groups) available.

*   **Baselines:** We will compare ACA against:
    *   **ERM:** Standard training on the original dataset.
    *   **Group-Supervised Methods (for context/upper bound):** GroupDRO (Sagawa et al., 2019), IRM (Arjovsky et al., 2019) - using ground-truth group labels during training.
    *   **State-of-the-Art Group-Free Methods:**
        *   LfF (Nam et al., 2020): Learning from Failure (reweighting based on high-loss examples).
        *   GEORGE (Sohoni et al., 2020) / SSA (Nam et al., 2022): Methods inferring group structures.
        *   EIIL (Creager et al., 2021) / CNC (Zhang et al., 2022): Contrastive or reweighting methods focusing on specific biases.
        *   Recent methods from literature review: e.g., EVaLS (Ghaznavi et al., 2024), SPUME (Zheng et al., 2024a), the subnetwork approach (Le et al., 2024), GIC (Han & Zou, 2024).

*   **Evaluation Metrics:**
    *   **Worst-Group Accuracy (WGA):** The primary metric for robustness against spurious correlations. Accuracy measured on the group with the lowest performance (requires group labels *only* for evaluation).
    *   **Average Accuracy:** Standard accuracy across all test examples.
    *   **Out-of-Distribution (OOD) Accuracy:** Performance on test sets with known distribution shifts (e.g., the OOD split of Waterbirds, anti-correlated Colored MNIST).
    *   **Calibration Error:** Expected Calibration Error (ECE) to assess model confidence reliability.

*   **Implementation Details:** We will primarily use PyTorch. For generative models, pre-trained backbones (e.g., Stable Diffusion, StyleGAN) might be fine-tuned for conditional counterfactual generation where applicable. Standard architectures (e.g., ResNet-50 for vision, BERT for NLP) will be used for the task models $f_{\theta}$ and $f_{\phi}$. Hyperparameters (e.g., $\lambda$, learning rates, generator specifics) will be tuned using validation sets constructed to reflect the OOD/worst-group performance objective, potentially using methods like IRM validation (Gulrajani & Lopez-Paz, 2020) if group labels are available for validation *only*, or relying on ID validation if not.

*   **Ablation Studies:** We will analyze the contribution of each component:
    *   Effectiveness of different spurious feature identification methods (gradients vs. influence functions).
    *   Impact of the generative model choice (Diffusion vs. GAN).
    *   Importance of the consistency loss ($\lambda=0$ vs. $\lambda>0$) and its formulation (prediction vs. representation).
    *   Qualitative analysis of generated counterfactuals $x'_i$ to assess their plausibility and effectiveness in modifying target features. Visualize attribution maps for $f_{\phi}$ compared to $f_{\theta}$ to verify reduced reliance on spurious cues.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
We anticipate the following outcomes from this research:

1.  **A validated ACA framework:** We expect to demonstrate that ACA significantly improves worst-group accuracy and OOD generalization compared to ERM and potentially matches or surpasses existing group-free methods on benchmark datasets.
2.  **Effective counterfactual generation:** We expect the conditional generative models to produce plausible counterfactual examples that successfully modify identified spurious features while preserving the core content and label, providing a valuable data augmentation resource.
3.  **Demonstrated invariance:** We expect the model $f_{\phi}$ trained with ACA to exhibit reduced reliance on spurious features, as evidenced by improved WGA/OOD metrics and confirmed through attribution map analysis comparing $f_{\phi}$ and $f_{\theta}$.
4.  **Insights into group-free robustness:** The research will provide insights into the feasibility of using model internals (attributions) to guide counterfactual generation for robustness without explicit group supervision. Ablation studies will clarify the relative importance of different components.
5.  **Comparative analysis:** A thorough comparison with state-of-the-art methods will position ACA within the current landscape of robust machine learning techniques.
6.  **Open-source contribution:** We plan to release code implementing the ACA framework to facilitate reproducibility and further research by the community.

**4.2 Impact**
This research holds the potential for significant impact:

*   **Enhanced Model Reliability:** By providing a practical method to mitigate spurious correlations without expensive group labels, ACA can contribute to building more reliable and trustworthy AI systems suitable for deployment in critical, real-world applications (e.g., healthcare diagnostics, autonomous systems, fairness-sensitive decision making).
*   **Theoretical Advancements:** The framework bridges ideas from causal inference (counterfactuals) and generative modeling for the purpose of robustness, potentially inspiring further theoretical work on the connections between these fields.
*   **Contribution to the Workshop:** The work directly addresses the core themes of the Workshop on Spurious Correlations, Invariance and Stability, offering a novel learning methodology, contributing to the discussion on evaluation, and potentially stimulating collaborations on applying ACA to diverse real-world problems.
*   **Democratizing Robustness:** Providing effective group-free methods lowers the barrier for deploying robust models, particularly in resource-constrained settings or domains where group annotation is infeasible.
*   **Future Research Directions:** This work can open up several avenues for future exploration, including extending ACA to other data modalities (e.g., tabular, time-series, reinforcement learning environments like Ding et al., 2023), investigating more sophisticated methods for spurious feature identification, exploring theoretical guarantees for ACA, and integrating ACA with other robustness techniques like adversarial training.

In conclusion, the proposed Adversarial Counterfactual Augmentation framework offers a promising, principled approach to tackle the pervasive issue of spurious correlations in machine learning without relying on group annotations. By leveraging model internals and conditional generative models, ACA aims to enhance model robustness and contribute significantly to the development of more reliable AI systems.

**References** (*A full list of cited papers would be included here in a final proposal, including those mentioned in the introduction and methodology like Geirhos et al. 2020, Arjovsky et al. 2019, Zech et al. 2018, Gururangan et al. 2018, Martin et al. 2019, Sagawa et al. 2019, Nam et al. 2020, Liu et al. 2021, Sundararajan et al. 2017, Selvaraju et al. 2017, Koh & Liang 2017, Ho et al. 2020, Mirza & Osindero 2014, Zhu et al. 2017, Liu et al. 2015, Borkan et al. 2019, Sohoni et al. 2020, Nam et al. 2022, Creager et al. 2021, Zhang et al. 2022, Gulrajani & Lopez-Paz 2020, and all papers from the provided literature review.*)