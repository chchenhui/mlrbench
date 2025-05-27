## 1. Title: LLM-Driven Discovery and Mitigation of Unknown Spurious Correlations

## 2. Introduction

Deep learning models have demonstrated remarkable capabilities across various domains, yet their reliability is often undermined by a tendency to learn spurious correlations – patterns that are statistically predictive in the training data but not causally related to the true underlying task (Geirhos et al., 2020). This phenomenon, also known as shortcut learning, arises from the statistical nature and inductive biases inherent in deep learning algorithms, encompassing data preprocessing, architectural choices, and optimization procedures (Workshop Overview). Consequently, models relying on spurious features exhibit poor generalization to out-of-distribution (OOD) data, fail in real-world scenarios involving under-represented groups or minority populations, and can perpetuate harmful biases (Ye et al., 2024; Workshop Overview). The foundational nature and widespread occurrence of shortcut learning make it a critical research area, pivotal for understanding model learning mechanisms and enhancing their robustness and trustworthiness.

Current research in mitigating spurious correlations often relies on methods requiring pre-defined knowledge of the spurious attributes, typically through group annotations (e.g., labeling images with "water background" vs. "land background"). While effective for known biases, this approach is not scalable, is labor-intensive, and critically, fails to address *unknown* spurious correlations that may be subtle, unanticipated, or not align with human perception (Workshop Objectives; Key Challenges). As noted in the workshop objectives, "Current evaluations do not inform us about the scenarios when the spurious correlation is unknown or annotations are missing." This highlights a significant gap: the lack of automated methods for discovering and mitigating these latent biases.

The recent advancements in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) present a unique opportunity to address this gap. Their sophisticated reasoning, contextual understanding, and pattern recognition capabilities (Wu et al., 2024) can be harnessed not only to understand existing biases (Hosseini et al., 2025; Zhou et al., 2023) but also to actively discover previously unidentified ones. This research proposes an innovative, interactive framework, "LLM-Assisted Spuriousity Scout" (LASS), that leverages LLMs to automate the discovery of unknown spurious correlations from model errors and guide subsequent mitigation strategies. This aligns with the workshop's aims to foster "automated methods for detecting spurious correlations" and find "solutions for robustness... when information regarding spurious feature is completely or partially unknown."

**Research Objectives:**

1.  **Develop an LLM-driven framework for discovering unknown spurious correlations:** This involves designing a pipeline where an LLM analyzes clusters of model errors to hypothesize potential spurious features without prior human specification of these features.
2.  **Design and implement targeted mitigation strategies guided by LLM-generated hypotheses:** Based on validated hypotheses, we will explore diverse robustification techniques, including counterfactual data augmentation, adaptive sample re-weighting, and auxiliary disentanglement tasks.
3.  **Rigorously evaluate the proposed framework's efficacy in improving model robustness and OOD generalization:** This includes comparison against state-of-the-art baselines on established and potentially newly curated benchmarks, focusing on worst-group accuracy and other relevant metrics.
4.  **Investigate the capabilities and limitations of LLMs in identifying complex, non-obvious spurious patterns:** This contributes to understanding the foundational aspects of LLM reasoning in the context of model debugging and scientific discovery.

**Significance:**

This research will contribute significantly to the field by:
*   Providing a novel, scalable approach to identify and mitigate *unknown* spurious correlations, thereby enhancing the robustness and reliability of AI systems in diverse real-world applications.
*   Reducing the manual effort and prior knowledge required for debiasing models, making robust AI more accessible.
*   Advancing our understanding of how LLMs can be utilized as scientific tools for hypothesis generation in machine learning research.
*   Potentially leading to the development of new benchmarks and evaluation methodologies specifically targeting unknown spurious correlations, addressing a key need highlighted by the workshop.
*   Contributing directly to the workshop's focus on "foundations and solutions" by exploring both the discovery mechanism (foundation) and the mitigation techniques (solutions) for spurious correlations.

## 3. Methodology

Our proposed research will be conducted in three main phases: (1) Error-Driven Spurious Hypothesis Generation using LLMs, (2) Interactive Hypothesis Validation and Refinement, and (3) LLM-Guided Robustification and Iterative Learning.

### 3.1 Data Collection and Preparation

We will utilize a diverse set of publicly available benchmark datasets known to contain spurious correlations, enabling rigorous evaluation and comparison with existing methods. Initial datasets include:
*   **Waterbirds** (Sagawa et al., 2019): Images of landbirds and waterbirds, where landbirds are spuriously correlated with land backgrounds and waterbirds with water backgrounds.
*   **CelebA** (Liu et al., 2015) with spurious attribute "Blond Hair" for "Smiling."
*   **CivilComments** (Borkan et al., 2019): Text data where toxicity is spuriously correlated with mentions of certain demographic identities.
*   **NICO (Non-ImageNet Contexts) Benchmark** (He et al., 2021): Designed to evaluate OOD generalization by providing different contexts for the same object classes.
*   **Multimodal datasets (e.g., based on VQA or image captioning, potentially extending SpurLens (Hosseini et al., 2025) findings):** Given the increasing importance of MLLMs (Workshop Topics), we will explore the application of our framework to a multimodal setting, focusing on visual and textual spurious cues.

For each dataset, we will define standard training, validation, and OOD test splits. The OOD test splits will specifically contain examples where the spurious correlation observed in the training set is broken.

### 3.2 Phase 1: Error-Driven Spurious Hypothesis Generation (LLM-Scribe)

This phase aims to automatically generate hypotheses about potential unknown spurious correlations by analyzing the errors of an initial task model.

**1. Initial Task Model Training:**
A standard task model $M_0$ (e.g., ResNet-50 for image tasks, BERT-base for text tasks) will be trained using Empirical Risk Minimization (ERM) on the training set of a chosen dataset.
$$ \min_{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{task}(M_0(x_i; \theta), y_i) $$
where $(x_i, y_i)$ are training examples and $\mathcal{L}_{task}$ is the task-specific loss (e.g., cross-entropy).

**2. Confident Error Identification and Clustering:**
The trained model $M_0$ will be applied to a diverse, unlabeled (or labeled, if available for analysis) dataset $D_{eval}$ (this could be the training set itself, or a separate validation set). We will identify instances where $M_0$ makes confident misclassifications. A confident error $e_i$ on sample $x_i$ is defined as:
$$ e_i = (x_i, y_i, \hat{y}_i) \quad \text{s.t.} \quad \hat{y}_i = \arg\max_k P(y=k|x_i; \theta) \neq y_i \quad \text{and} \quad P(\hat{y}_i|x_i; \theta) > \tau $$
where $\hat{y}_i$ is the predicted label, $y_i$ is the true label (if inspecting on labeled data), and $\tau$ is a confidence threshold (e.g., 0.7).

To identify systematic errors, embeddings of these confidently misclassified samples (e.g., from the penultimate layer of $M_0$) will be clustered using an algorithm like k-means or DBSCAN. Each cluster $C_j$ represents a group of inputs on which $M_0$ makes similar errors.

**3. LLM-Powered Hypothesis Generation:**
For each prominent error cluster $C_j$, representative samples (e.g., k-medoids or random samples from the cluster) will be presented to an LLM (e.g., GPT-4, Claude 3, or a powerful open-source MLLM like LLaVA for visual tasks). The LLM will be prompted to identify common, potentially non-causal patterns exclusive to these error samples that might explain the misclassifications.
An example prompt structure for an image classification task:
```
You are an AI assistant helping to debug an image classification model.
The model was trained to classify images into categories like [True Category A], [True Category B], etc.
Below are [K] images that the model INCORRECTLY and CONFIDENTLY classified as [Predicted Category P], but their TRUE category is [True Category T].
[Image 1 (misclassified as P, true T)]
[Image 2 (misclassified as P, true T)]
...
[Image K (misclassified as P, true T)]
Please carefully examine these images. What common visual features (e.g., background elements, co-occurring objects, textures, lighting, artistic style, camera angles) are present across many of these images that are:
1. Distinctive to this group of misclassified images compared to typical images of [True Category T].
2. Potentially UNRELATED to the definition of [True Category T] and might be acting as spurious cues, misleading the model.
List up to 5 such potential spurious features or patterns. For each, briefly explain your reasoning.
```
For text data, raw text snippets or summaries would be provided. The LLM's output, a set of textual descriptions of potential spurious features $H = \{h_1, h_2, ..., h_m\}$, will be automatically parsed. This leverages the LLM's ability to perform abductive reasoning from examples, akin to methods like RaVL (Varma et al., 2024) and SpurLens (Hosseini et al., 2025) which identify features, but here specifically targeted at error clusters representing *unknown* issues.

### 3.3 Phase 2: Interactive Hypothesis Validation and Refinement

The LLM-generated hypotheses $H$ are not guaranteed to be perfect. This phase involves human-in-the-loop validation to ensure relevance and accuracy.

**1. Hypothesis Presentation and Initial Validation:**
The generated hypotheses will be presented to a human evaluator (researchers in this project) through a simple interface. The evaluator will assess each hypothesis $h_k \in H$ for:
*   **Plausibility:** Is the hypothesized feature genuinely present in the error samples?
*   **Spuriousness:** Is the feature likely non-causal for the true task label?
*   **Actionability:** Can this hypothesis be used to guide a mitigation strategy?

**2. LLM-Assisted Refinement (Optional):**
If a hypothesis is too vague or partially correct, the evaluator can provide feedback. The LLM can then be re-prompted with this feedback and the original samples to refine its hypothesis. For example: "Your previous hypothesis was 'unusual background'. Can you be more specific about what aspects of the background are unusual in these misclassified images of [True Category T]?"

Validated hypotheses $H_{valid} \subseteq H$ will proceed to the mitigation phase. This interactive step addresses the challenge that LLM outputs can sometimes be noisy or require common-sense filtering.

### 3.4 Phase 3: LLM-Guided Robustification and Iterative Learning

Validated hypotheses $H_{valid}$ will guide the selection and implementation of targeted mitigation strategies to retrain the task model $M_k$.

**1. Counterfactual Data Augmentation:**
Inspired by approaches like Feder et al. (2023) and Zhou et al. (2023), we will generate or select counterfactual training data that explicitly breaks the identified spurious correlation.
*   If hypothesis $h_{valid}$ is "presence of feature S is spuriously linked to class Y," we will aim to:
    *   Augment data to include samples of class Y *without* feature S.
    *   Augment data to include samples of other classes *with* feature S.
*   LLMs (potentially generative ones) can assist in describing or even generating these counterfactuals (e.g., prompting a text-to-image model to "generate an image of a landbird on a water background").

**2. Adaptive Sample Re-weighting / Sub-sampling:**
For each validated spurious hypothesis $h_k \in H_{valid}$, we can use the same LLM (or a specialized classification model trained on $h_k$) to score each training sample $x_i$ for the presence of the spurious feature, yielding a spuriousness score $s_{ik}$. Samples exhibiting high spuriousness can be down-weighted in the training loss. For a single spurious feature $s_i$:
$$ \mathcal{L}_{re-weighted} = \sum_{i=1}^N w_i \mathcal{L}_{task}(M_k(x_i; \theta), y_i) $$
where $w_i = g(s_i)$ is a monotonically decreasing function of the spuriousness score (e.g., $w_i = 1/(1 + \alpha s_i)$). This is related to ideas in CCR (Zhou & Zhu, 2024) but with LLM-identified features.

**3. Auxiliary Disentanglement Tasks:**
If an LLM identifies a specific spurious feature $S_j$, an auxiliary task can be designed to explicitly encourage the model to learn representations that disentangle the true feature $F_{true}$ from $S_j$. For instance, an auxiliary head can be added to $M_k$ to predict the presence of $S_j$. The overall loss would be:
$$ \mathcal{L}_{total} = \mathcal{L}_{task}(M_k(x_i; \theta_{shared}, \theta_{task}), y_i) + \lambda \mathcal{L}_{aux}(M_k(x_i; \theta_{shared}, \theta_{aux}), l_{sj}) $$
where $l_{sj}$ is a label indicating the presence/absence of spurious feature $S_j$. Techniques similar to "UnLearning from Experience" (ULE) (Mitchell et al., 2024) or contrastive methods for unlearning spurious connections (Le et al., 2024) can be adapted here, where the 'student' or 'spurious path' is defined by the LLM's hypothesis.

**4. Iterative Model Refinement:**
After applying mitigation strategies and retraining the model (now $M_1$), the process (Phase 1-3) can be repeated. $M_1$ might now exhibit different error patterns, potentially revealing new, more subtle spurious correlations. This iterative loop aims to progressively enhance model robustness.

### 3.5 Experimental Design and Evaluation

**1. Baseline Methods:**
We will compare our LASS framework against several baselines:
*   **ERM (Empirical Risk Minimization):** Standard model training without any robustness intervention.
*   **Group-DRO (Sagawa et al., 2019):** An oracle method that requires group annotations (representing known spurious correlations).
*   **Methods not requiring group labels:**
    *   **SPUME (Zheng et al., 2024):** A meta-learning framework using VLM-extracted attributes.
    *   **Out of Spuriousity (Le et al., 2024):** Extracts a robust subnetwork via contrastive loss.
    *   **ULE (Mitchell et al., 2024):** Teacher-student unlearning.
    *   **Last Layer Retraining (LLR) / Deep Feature Reweighting (DFR) (Kirichenko et al., 2023):** Simple yet strong baselines.

**2. Evaluation Metrics:**
*   **Worst-Group Accuracy (WGA):** Primary metric, measuring accuracy on the most challenging predefined group (where spurious correlation is typically violated). If groups are unknown, we will report performance on OOD test sets designed to break the most common spurious correlations found in literature or by our LLM.
*   **Average Accuracy:** Overall accuracy on standard test sets and OOD test sets.
*   **Spurious Feature Reliance Score:** We may develop metrics to quantify how much the model relies on LLM-identified spurious features (e.g., using feature attribution methods).
*   **LLM Hypothesis Quality:**
    *   Precision/Recall of LLM identifying known spurious features (on benchmarks where they are documented).
    *   Human rating of the plausibility and actionability of novel hypotheses generated by the LLM.
*   **Efficiency:** Reduction in human effort for identifying spurious correlations compared to manual exploration.

**3. Ablation Studies:**
We will conduct ablation studies to understand the contribution of each component of LASS:
*   Impact of LLM choice (e.g., GPT-4 vs. Claude 3 vs. open-source models).
*   Effectiveness of different mitigation strategies guided by LLM hypotheses.
*   Importance of the human validation step.
*   Performance of single-iteration vs. multi-iteration LASS.

**4. Novel Scenarios and Benchmarks (Exploratory):**
In line with workshop objectives, we may explore creating a new small-scale benchmark focused on a type of unknown spurious correlation discovered through our method, or applying LASS to less-explored modalities like multimodal data with subtle cross-modal spuriousness.

## 4. Expected Outcomes & Impact

This research is poised to deliver several significant outcomes and impacts:

**Expected Outcomes:**

1.  **A Novel, Iterative Framework (LASS):** The primary outcome will be a fully developed and tested framework that integrates LLMs for the automated discovery of unknown spurious correlations and guides targeted mitigation. This includes the algorithmic steps, prompt engineering strategies, and human-in-the-loop validation protocols.
2.  **Improved Model Robustness on Benchmark and Novel Datasets:** We expect models trained using LASS to demonstrate significantly improved OOD generalization, particularly in worst-group accuracy, compared to ERM and competitive non-group-annotated methods. The goal is to approach, or even surpass in some scenarios of unknown bias, the performance of methods requiring full group supervision.
3.  **Identification and Characterization of Previously Unknown Spurious Correlations:** The application of LASS to diverse datasets is likely to uncover novel spurious correlations that are not well-documented, providing new insights into how models can fail. These findings can contribute to a richer understanding of shortcut learning.
4.  **Quantitative Evaluation of LLM Capabilities in Scientific Hypothesis Generation:** The research will provide data on the effectiveness of LLMs in identifying subtle patterns from model errors, contributing to the broader understanding of LLMs as tools for scientific discovery and debugging within AI itself.
5.  **Guidelines for LLM-Assisted Spurious Correlation Mitigation:** We will produce best practices for prompting LLMs, validating their outputs, and translating these outputs into actionable mitigation strategies for different data modalities.
6.  **Open-Source Code and Potentially New Benchmarks:** The LASS framework will be released as open-source software. If significant novel spurious correlations are discovered and validated, we may contribute to the development of new benchmark slices or datasets for the community.

**Impact:**

*   **Enhanced Trustworthiness and Reliability of AI Systems:** By addressing unknown spurious correlations, LASS will contribute to building AI models that are more dependable and less prone to unexpected failures when deployed in real-world, uncontrolled environments. This is crucial for applications in sensitive domains like medicine, finance, and autonomous systems.
*   **Democratization of Robust AI Development:** Current advanced debiasing often requires significant expertise and resources for annotating spurious attributes. LASS aims to lower this barrier by automating a significant part of the discovery process, making robust model development more accessible.
*   **Advancement of Foundational Understanding of Shortcut Learning:** This work will shed light on the mechanisms by which models pick up on various types of spurious cues, especially those not immediately obvious to humans. It will also explore the "foundations of spurious correlations" by investigating how LLMs can reason about these failures, potentially informing theories of model learning.
*   **Contribution to Workshop Themes:** This research directly addresses key workshop objectives, including "automated methods for detecting spurious correlations," creating "novel solutions for building robust models," finding "solutions for robustness ... when information regarding spurious feature is completely or partially unknown," and exploring "foundational large language models ... in terms of robustness."
*   **Stimulating Further Research:** The novel interactive paradigm proposed, where LLMs act as "debuggers" or "scientific assistants" for other ML models, could inspire new research directions in AI-assisted AI development, model interpretability, and human-AI collaboration.

In conclusion, this research proposes a paradigm shift from manually identifying and mitigating known spurious correlations to an LLM-driven, semi-automated discovery and mitigation of *unknown* biases. Success in this endeavor will mark a significant step towards building more robust, reliable, and ethically sound AI systems, capable of generalizing beyond the confines of their training data.

## References (Illustrative - to be completed based on final proposal text)
*   Borkan, D., et al. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. *arXiv:1903.04561*.
*   Feder, A., et al. (2023). Data Augmentations for Improved (Large) Language Model Generalization. *arXiv:2310.12803*.
*   Geirhos, R., et al. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*.
*   He, K., et al. (2021). Towards Non-ImageNet Contexts: A New Benchmark for OOD Generalization. *NeurIPS Datasets and Benchmarks Track*.
*   Hosseini, P., et al. (2025). Seeing What's Not There: Spurious Correlation in Multimodal LLMs. *arXiv:2503.08884*.
*   Kirichenko, P., et al. (2023). Last Layer Re-training is Sufficient for Robustness to Spurious Correlations. *ICML*.
*   Le, P. Q., et al. (2024). Out of Spuriousity: Improving Robustness to Spurious Correlations Without Group Annotations. *arXiv:2407.14974*.
*   Liu, Z., et al. (2015). Deep Learning Face Attributes in the Wild. *ICCV*.
*   Mitchell, J., et al. (2024). UnLearning from Experience to Avoid Spurious Correlations. *arXiv:2409.02792*.
*   Sagawa, S., et al. (2019). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. *ICLR*.
*   Varma, M., et al. (2024). RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models. *arXiv:2411.04097*.
*   Wu, A., et al. (2024). Causality for Large Language Models. *arXiv:2410.15319*.
*   Ye, W., et al. (2024). Spurious Correlations in Machine Learning: A Survey. *arXiv:2402.12715*.
*   Zheng, G., et al. (2024). Spuriousness-Aware Meta-Learning for Learning Robust Classifiers. *arXiv:2406.10742*.
*   Zhou, Y., et al. (2023). Explore Spurious Correlations at the Concept Level in Language Models for Text Classification. *arXiv:2311.08648*.
*   Zhou, Y., & Zhu, Z. (2024). Towards Robust Text Classification: Mitigating Spurious Correlations with Causal Learning. *arXiv:2411.01045*.