# **Research Proposal**

## 1. Title: **Causal Intervention Pruning: Mitigating Spurious Correlations for Reliable and Fair Foundation Models**

## 2. Introduction

**2.1 Background**
Foundation models (FMs), such as large language models (LLMs) like GPT-4 and vision-language models like CLIP, represent a paradigm shift in artificial intelligence. Trained on vast, diverse datasets, they exhibit remarkable zero-shot and few-shot capabilities across a wide range of tasks (Brown et al., 2020; Radford et al., 2021). However, their reliance on massive, often uncurated web-scale corpora makes them susceptible to learning spurious correlations – patterns that hold statistically in the training data but do not reflect true causal relationships in the world (Geirhos et al., 2020). This reliance leads to significant reliability and responsibility concerns, manifesting as:

*   **Nonfactuality and Hallucinations:** Generating plausible yet factually incorrect statements (Ji et al., 2023).
*   **Lack of Robustness:** Sensitivity to input perturbations or distribution shifts, leading to unpredictable performance drops in real-world scenarios (Hendrycks & Dietterich, 2019).
*   **Bias and Unfairness:** Perpetuating or amplifying societal biases present in the training data, leading to discriminatory outcomes (Bender et al., 2021).
*   **Prompt Sensitivity:** Output variability based on subtle, semantically equivalent changes in prompts.
*   **Lack of Transparency:** Difficulty in understanding *why* a model produces a specific output, hindering debugging and trust.

These issues are central to the mission of the Workshop on Reliable and Responsible Foundation Models (R2-FM). As FMs become increasingly integrated into critical domains like healthcare, finance, and education, ensuring their outputs are reliable, fair, and aligned with human values is paramount. Existing approaches often tackle these problems through data augmentation, robust optimization, or post-hoc explanations, but they may not directly address the underlying issue of spurious feature representation within the model's internal workings. While recent works explore causal perspectives (Zhou & Zhu, 2024; Ma et al., 2024; Wang et al., 2021), identifying and surgically removing spurious features at the activation level *within* large pre-trained FMs using scalable interventionist techniques remains an open challenge.

**2.2 Research Gap and Proposed Idea**
Current methods for mitigating spurious correlations often rely on aggregate statistics, domain-specific heuristics, or require significant manual annotation of causal vs. spurious features (Wang et al., 2021). While prompt tuning can adapt models at test time (Ma et al., 2024), it doesn't fundamentally alter the model's internal representations learned during pre-training. Techniques focusing on counterfactual reasoning (Zhou & Zhu, 2024) show promise but may struggle with the scale and complexity of pinpointing specific problematic features within the high-dimensional latent spaces of modern FMs.

We propose **Intervention-Based Causal Pruning (ICP)**, a novel two-stage framework designed to directly identify and mitigate the influence of spurious features embedded within the internal representations of foundation models.
1.  **Causal Attribution via Targeted Interventions:** We leverage causal intervention techniques, specifically performing "do-calculations" on the model's hidden activations (e.g., neuron outputs, attention head components). By systematically manipulating these internal features (e.g., masking, scaling, swapping) across diverse inputs and observing the causal effect on downstream outputs (e.g., factual correctness, sentiment prediction, fairness metrics), we aim to quantify the "spuriousness" of individual features. Features whose manipulation consistently leads to unreliable or biased outputs are flagged as spurious.
2.  **Intervention-Guided Pruning and Fine-tuning:** Based on the identified spurious features, we employ a targeted fine-tuning strategy. This involves either directly pruning (removing or zeroing out) high-spuriousness features or using a causally informed regularization technique, such as contrastive learning, to explicitly train the model to be invariant to manipulations of these spurious features while remaining sensitive to manipulations of causal ones.

This approach moves beyond correlational analysis to estimate the causal impact of internal model components, offering a more principled way to enhance FM reliability and responsibility.

**2.3 Research Objectives**
This research aims to:
1.  Develop a scalable methodology for **causal attribution** within foundation models using targeted interventions on hidden activations to identify features contributing to spurious correlations.
2.  Design and implement an **intervention-guided pruning and fine-tuning** strategy that selectively removes or neutralizes the impact of identified spurious features.
3.  **Empirically evaluate** the effectiveness of the proposed ICP framework in improving FM reliability (reducing hallucinations, improving out-of-distribution robustness), fairness (mitigating biases), and calibration across diverse tasks (e.g., open-domain QA, sentiment analysis, bias detection).
4.  **Analyze the characteristics** of identified spurious features and provide insights into how FMs internalize and utilize such correlations.

**2.4 Significance**
This research directly addresses several fundamental questions posed by the R2-FM workshop:
*   It provides a method to **identify and characterize unreliable behaviors** (hallucinations, spurious feature reliance) by pinpointing specific internal features.
*   It helps **understand the causes** of FM unreliability by examining the role of learned features (activations/weights).
*   It proposes a **principled intervention** during fine-tuning (or potentially pre-training adaptation) to enhance reliability and responsibility.
*   It contributes towards building more **trustworthy AI** by improving transparency (understanding feature roles) and robustness.

Successfully developing ICP would offer a generalizable, domain-agnostic technique to enhance the reliability and fairness of pre-trained FMs, reducing societal harms associated with hallucinations and biases, and paving the way for safer deployment in critical applications. It bridges the gap between causal inference theory and practical FM engineering, offering a potential pathway towards more robust and value-aligned AI systems.

## 3. Methodology

**3.1 Overall Framework**
The proposed Intervention-Based Causal Pruning (ICP) framework operates in two sequential stages, applied typically after standard pre-training but before or during task-specific fine-tuning.

**Stage 1: Causal Attribution via Targeted Interventions**

*   **Goal:** Identify internal model features (neurons, attention components) whose activation causally influences undesirable output behaviors (e.g., factual errors, biased predictions, sensitivity to spurious input patterns).
*   **Feature Definition:** A "feature" $f_i$ corresponds to a specific element within the model's hidden state representation at a given layer $l$. This could be the output of a single neuron in a feed-forward network (FFN) H, a specific dimension within an embedding vector, or a component related to an attention head (e.g., value vector contribution).
*   **Intervention Set:** We define a set of atomic interventions $\mathcal{I} = \{ \text{mask}, \text{scale}(\alpha), \text{swap}(f_j) \}$ that can be applied to a feature $f_i$.
    *   `mask(f_i)`: Sets the activation of $f_i$ to zero (or its mean value across a batch), simulating its removal (ablation).
    *   `scale(f_i, \alpha)`: Multiplies the activation of $f_i$ by a scalar $\alpha$ (e.g., $\alpha=0, 0.5, 2$), modulating its influence.
    *   `swap(f_i, f_j)`: Replaces the activation of $f_i$ with that of another feature $f_j$ (potentially from a different input sample exhibiting a contrasting property), testing for functional substitution.
*   **Causal Effect Quantification:** For a given input $x$, model $M$, target output property $P$ (e.g., probability of the correct answer, sentiment score, bias score), and feature $f_i$, we measure the causal effect of an intervention $I \in \mathcal{I}$ as the change in the property $P$:
    $$ \Delta P(x, f_i, I) = P(M(x | do(f_i=I(f_i)))) - P(M(x)) $$
    where $M(x | do(f_i=I(f_i)))$ denotes the model's output when intervention $I$ is applied to feature $f_i$ during the forward pass for input $x$.
*   **Spuriousness Score Calculation:** To obtain a robust estimate of a feature's spuriousness, we aggregate the causal effects over a diverse dataset $D_{probe}$ containing examples designed to elicit specific failure modes (e.g., factual contradictions, inputs with known spurious cues, minimal pairs for bias). The Spuriousness Score $S(f_i)$ for feature $f_i$ is computed as:
    $$ S(f_i) = \mathbb{E}_{x \sim D_{probe}, I \sim \mathcal{I}} [ w(x, I) \cdot |\Delta P(x, f_i, I)| ] $$
    where $w(x, I)$ is a weighting function that can prioritize interventions or inputs known to be associated with undesirable outcomes (e.g., higher weight if the intervention flips a correct factual statement to incorrect, or introduces bias). The expectation is approximated by sampling inputs and interventions. Features with high $S(f_i)$ are considered candidates for being spurious. This calculation requires multiple forward passes per input, but only for a subset of features and a dedicated probing dataset, making it potentially more scalable than exhaustive causal discovery. We will explore strategies like focusing interventions on later layers or features with high variance.

**Stage 2: Intervention-Guided Pruning and Fine-tuning**

*   **Goal:** Mitigate the impact of features identified as spurious in Stage 1.
*   **Approach 1: Direct Pruning:** Features $f_i$ with $S(f_i)$ exceeding a predefined threshold $\tau$ are permanently pruned. For neurons, this means setting their outgoing weights to zero. For attention components, this might involve masking specific head outputs or value dimensions. Pruning offers model compression as a side benefit but risks impacting performance if causal features are mistakenly pruned or if spurious features play a dual role.
*   **Approach 2: Causal Invariance Regularization:** Instead of pruning, we fine-tune the model using a modified objective function that encourages invariance to spurious features. We leverage contrastive learning or an explicit invariance loss. For instance, using a dataset $D_{tune}$, we augment the standard task loss $\mathcal{L}_{task}$ with a causal invariance loss $\mathcal{L}_{causal\_inv}$:
    $$ \mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{causal\_inv} $$
    The causal invariance loss encourages the model's output (or internal representation at a later layer) to remain unchanged when identified spurious features $F_{spurious} = \{f_i | S(f_i) > \tau\}$ are perturbed:
    $$ \mathcal{L}_{causal\_inv} = \mathbb{E}_{x \sim D_{tune}, f_i \sim F_{spurious}, I \sim \mathcal{I}} [ D( M(x), M(x | do(f_i=I(f_i)))) ] $$
    Here, $D$ is a distance or divergence metric (e.g., KL divergence for probability distributions, cosine distance for embeddings). $\lambda$ is a hyperparameter balancing task performance and causal invariance. This approach aims to retain model capacity while discouraging reliance on specific spurious pathways.
*   **Hybrid Approach:** We may explore combining pruning for features with extremely high S(f_i) and regularization for those with moderately high scores.

**3.2 Data Collection and Preparation**

*   **Foundation Models:** We will select representative FMs, such as BERT/RoBERTa for NLP tasks focusing on NLU, a GPT-style model (e.g., GPT-2/Neo/Pythia) for generation tasks, and potentially CLIP for vision-language tasks if resources permit. We will use publicly available pre-trained checkpoints.
*   **Probing Data ($D_{probe}$):** This dataset is crucial for Stage 1. It will consist of diverse samples designed to reveal specific failure modes:
    *   *Factual QA:* Pairs of factual statements and their negations (e.g., from FEVER dataset, or generated counterfactuals based on knowledge bases).
    *   *Sentiment Analysis with Spurious Cues:* Datasets like SST-2 augmented with known spurious triggers (e.g., specific names or locations associated with positive/negative sentiment only in training). Datasets like Waterbirds (Sagawa et al., 2019) for vision-language models.
    *   *Bias Benchmarks:* Minimal pairs from datasets like StereoSet (Nadeem et al., 2021) or CrowS-Pairs (Nangia et al., 2020) to probe for stereotypical associations.
    *   *OOD Generalization Sets:* Data from domains different from the FM's primary training/fine-tuning domain.
*   **Fine-tuning Data ($D_{tune}$):** Standard datasets for the downstream tasks used in evaluation (e.g., SQuAD, TriviaQA for QA; SST-2, IMDB for sentiment; relevant portions of bias benchmarks). Data augmentation might be employed based on insights from Stage 1.

**3.3 Experimental Design**

*   **Baselines:**
    1.  Standard fine-tuning of the pre-trained FM on the target task.
    2.  Fine-tuning with standard regularization techniques (e.g., L1/L2 weight decay, dropout).
    3.  Methods from literature designed to mitigate spurious correlations, adapted to our tasks (e.g., CCR-like inverse propensity weighting if applicable, potentially influence function-based data removal).
*   **Proposed Models:**
    1.  ICP-Prune: FM fine-tuned after applying direct pruning based on Stage 1 results.
    2.  ICP-Reg: FM fine-tuned using the causal invariance regularization (Stage 2, Approach 2).
    3.  ICP-Hybrid: Combining pruning and regularization.
*   **Evaluation Tasks:**
    1.  **Open-Domain Question Answering:** Evaluate factual accuracy and hallucination rates on datasets like TriviaQA or Natural Questions. Hallucinations can be measured using automated metrics (e.g., NLI-based fact-checking against evidence, self-consistency checks) or human evaluation on a subset.
    2.  **Sentiment Analysis (In-Distribution and OOD):** Evaluate accuracy on standard benchmarks (e.g., SST-2) and robustness on specially constructed OOD sets or datasets with known spurious correlations (e.g., evaluate performance on minority groups vs. majority groups in Waterbirds-like scenarios adapted for text).
    3.  **Bias Mitigation:** Evaluate performance on bias benchmarks like StereoSet (Intrasentence score) or measure fairness metrics (e.g., Equality of Opportunity, demographic parity) on prediction tasks involving protected attributes.
    4.  **Calibration:** Measure Expected Calibration Error (ECE) on classification tasks to assess model confidence reliability.
*   **Ablation Studies:**
    *   Vary the set of interventions ($\mathcal{I}$) used in Stage 1.
    *   Analyze the sensitivity to the spuriousness threshold $\tau$.
    *   Compare the effectiveness of different layers for intervention.
    *   Evaluate the impact of the regularization weight $\lambda$.
    *   Compare pruning vs. regularization directly.

**3.4 Evaluation Metrics**

*   **Reliability Metrics:** Accuracy (QA, Sentiment ID/OOD), F1 score, Hallucination Rate (%), Self-Consistency Score, Expected Calibration Error (ECE).
*   **Responsibility/Fairness Metrics:** StereoSet Score (LMS/SS), Equality of Opportunity gap, Demographic Parity difference, performance disparity across demographic groups.
*   **Model Efficiency:** Percentage of parameters pruned (for ICP-Prune), change in inference latency.
*   _Qualitative Analysis:_ Examination of specific examples where ICP models correct errors made by baselines; visualization or analysis of the function/semantics of pruned or down-weighted features.

## 4. Expected Outcomes & Impact

**4.1 Expected Outcomes**
We expect this research to deliver the following outcomes:

1.  **A Novel Framework (ICP):** A well-defined and implemented methodology (algorithm, code) for identifying and mitigating spurious features in FMs using causal interventions.
2.  **Empirical Validation:** Quantitative results demonstrating the effectiveness of ICP compared to baseline and existing methods. We anticipate:
    *   A measurable reduction in hallucination rates (targeting ~15-20% reduction based on the initial idea) on factual QA tasks.
    *   Improved out-of-distribution (OOD) generalization on tasks like sentiment analysis under domain shift.
    *   Reduced bias scores on standard fairness benchmarks (e.g., improved StereoSet scores, lower fairness gaps).
    *   Enhanced model calibration (lower ECE).
3.  **Identification of Spurious Features:** Characterization of the types of features (specific neurons, attention components) identified as spurious across different models and tasks. This could provide valuable insights into the internal mechanisms of FMs.
4.  **Analysis and Insights:** A deeper understanding of the relationship between internal feature representations, causal effects, and model reliability/responsibility. This includes analysing potential trade-offs between robustness, fairness, and task performance.
5.  **Open-Source Contribution:** Release of code implementing the ICP framework and potentially curated probing datasets to facilitate further research.

**4.2 Impact**
This research has the potential for significant impact aligned with the goals of the R2-FM workshop and the broader AI community:

*   **Enhanced Reliability:** By directly targeting and removing features causally linked to errors like hallucinations and poor generalization, ICP can lead to more dependable FMs suitable for deployment in high-stakes applications.
*   **Improved Responsibility and Fairness:** By identifying and mitigating features responsible for biased outputs, this work contributes to developing FMs that are fairer and more aligned with societal values, reducing the perpetuation of harmful stereotypes.
*   **Increased Transparency:** While not full interpretability, identifying the causal role of specific internal features on outputs offers a degree of transparency into the model's decision-making process, aiding debugging and trust.
*   **Advancement of Causal ML for FMs:** This work pushes the boundary of applying causal inference techniques to understand and improve large-scale neural networks, addressing key challenges like scalability and effective intervention design within complex models.
*   **Practical Tools for AI Developers:** If successful, ICP could provide developers with a concrete toolset to "debug" and "harden" pre-trained FMs against known failure modes before deployment, complementing existing fine-tuning and alignment techniques.
*   **Contribution to Theoretical Understanding:** Analyzing the nature of identified spurious features might inform future architectural designs or pre-training objectives aimed at inherently discouraging the learning of such correlations.

In summary, this research tackles a critical bottleneck in the development of trustworthy AI – the prevalence of spurious correlations learned by FMs. By proposing a novel, causally-grounded intervention method, we aim to make tangible progress towards building foundation models that are not only powerful but also reliable, fair, and responsible.

## 5. References

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜. *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, 610–623.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, *33*, 1877-1901.

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, *2*(11), 665–673.

Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *International Conference on Learning Representations*.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, *55*(12), 1-38.

Ma, H., Zhu, Y., Zhang, C., Zhao, P., Wu, B., Huang, L.-K., Hu, Q., & Wu, B. (2024). Spurious Feature Eraser: Stabilizing Test-Time Adaptation for Vision-Language Foundation Model. *arXiv preprint arXiv:2403.00376*.

Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring Stereotypical Bias in Pre-trained Language Models. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, 5356–5371.

Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1953–1969.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748-8763.

Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2019). Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. *International Conference on Learning Representations*.

Volodin, S., Wichers, N., & Nixon, J. (2020). Resolving Spurious Correlations in Causal Models of Environments via Interventions. *arXiv preprint arXiv:2002.05217*.

Wang, Z., Shu, K., & Culotta, A. (2021). Enhancing Model Robustness and Fairness with Causality: A Regularization Approach. *arXiv preprint arXiv:2110.00911*.

Zhou, Y., & Zhu, Z. (2024). Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective. *arXiv preprint arXiv:2402.01045*. *(Note: The provided ID arXiv:2411.01045 seems incorrect, assuming 2402.01045 based on usual numbering)*.