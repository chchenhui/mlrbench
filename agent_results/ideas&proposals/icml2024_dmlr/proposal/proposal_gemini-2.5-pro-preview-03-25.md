## 1. Title: Adaptive Model-Assisted Dataset Construction via Diversity-Aware Feedback Loops for Foundation Models

## 2. Introduction

**Background:**
The rapid advancement of large-scale foundation models has marked a significant paradigm shift in machine learning, demonstrating remarkable capabilities across diverse tasks, primarily in natural language processing and computer vision (Bommasani et al., 2021). However, the success of these models is inextricably linked to the massive datasets upon which they are trained. While architectural innovations previously dominated research focus, the critical role of data – its quality, scale, diversity, and provenance – is now increasingly recognized as paramount (Sambasivan et al., 2021). This shift towards data-centric AI (DCAI) emphasizes that improving data characteristics can yield substantial performance gains, often exceeding those achieved through model modifications alone (Mazumder et al., 2022).

Despite this recognition, constructing high-quality, large-scale datasets, especially for emerging or specialized domains such as climate science, robotics, genomics, or specialized biomedical imaging, remains a formidable challenge. These domains often lack readily available, well-curated data, making traditional dataset creation methods, reliant on extensive manual annotation, prohibitively expensive and time-consuming. Model-assisted dataset construction (MADS) techniques have emerged as a promising avenue, leveraging existing models to aid in data selection, labeling, or generation (Ratner et al., 2017). However, current MADS approaches often prioritize dataset scale or rudimentary quality checks, frequently overlooking the crucial aspect of data *diversity*. This oversight can lead to the creation of datasets with hidden biases, limited coverage of edge cases, or poor representation of rare but critical phenomena. Consequently, foundation models trained on such datasets may exhibit suboptimal performance, lack robustness to distributional shifts, and perpetuate societal biases (Buolamwini & Gebru, 2018).

Furthermore, the very integration of models into the data curation pipeline introduces complex dynamics, particularly the risk of "data feedback loops" where model biases are amplified over successive iterations of data generation or selection (Taori & Hashimoto, 2022; Wyllie et al., 2024). Models trained on synthetically augmented data may inadvertently reinforce existing biases, leading to degraded performance and increased unfairness in subsequent model generations (Wyllie et al., 2024). While techniques exist for fairness-aware augmentation (Erfanian et al., 2024) and integrating diverse forms of feedback for model alignment (Yu et al., 2023), a systematic framework that adaptively guides dataset construction towards *explicit diversity goals* while actively mitigating bias amplification is lacking.

**Research Problem and Gap:**
The core research problem is the absence of efficient and principled methodologies for constructing large-scale, diverse, and high-quality datasets for foundation models in specialized domains using model assistance, while proactively managing the risks inherent in model-data feedback loops. Current MADS approaches lack adaptive mechanisms to: (1) systematically identify and target underrepresented data subpopulations or phenomena, (2) integrate human expertise efficiently to validate quality and fill critical diversity gaps, and (3) continuously monitor and optimize for dataset diversity alongside quality throughout the construction process. This gap hinders the development of truly robust, generalizable, and fair foundation models for a wider range of applications beyond standard benchmarks.

**Proposed Solution:**
We propose **Adaptive Diversity-aware Dataset Enhancement (ADDE)**, an iterative framework for model-assisted dataset construction that explicitly incorporates diversity-aware feedback loops. ADDE leverages foundation models not just for annotation or generation, but to actively *guide* the data curation process towards maximizing diversity and quality. The framework iteratively performs:
1.  **Diversity Analysis:** Utilizes model embeddings to identify underrepresented regions or patterns in the current dataset's latent space.
2.  **Targeted Data Acquisition:** Employs model-based synthetic data generation specifically targeting these identified gaps and/or selects informative unlabeled real data.
3.  **Efficient Human Validation:** Integrates active learning strategies to prioritize the most valuable data points (synthetic or real) for human review, focusing on quality, relevance, and filling diversity gaps.
4.  **Adaptive Refinement:** Continuously monitors quantitative metrics for both data diversity (e.g., distributional coverage, cluster entropy in latent space) and quality (e.g., label consistency, downstream task performance) to dynamically adjust the generation and selection strategies in subsequent iterations.

**Research Objectives:**
The primary objectives of this research are:
1.  To design and implement the ADDE framework, including its core components: diversity analysis, targeted generation/selection, active learning validation, and adaptive refinement loop.
2.  To develop and evaluate novel quantitative metrics for assessing dataset diversity relevant to foundation model training in specialized domains, going beyond simple class balance.
3.  To investigate and adapt generative models and active learning strategies suitable for targeted data acquisition within the ADDE feedback loop.
4.  To empirically evaluate the effectiveness of ADDE in constructing datasets for specific emerging domains (e.g., biomedical imaging, climate informatics) compared to baseline MADS and traditional methods.
5.  To analyze ADDE's ability to enhance downstream foundation model performance, particularly regarding robustness to distribution shifts and fairness across subgroups.
6.  To assess the framework's efficiency in terms of reducing human annotation effort and computational cost.
7.  To investigate the stability of the ADDE feedback loop and its capacity to mitigate, rather than amplify, dataset biases compared to naive feedback systems.

**Significance:**
This research directly addresses the critical need for advanced data-centric methodologies highlighted by the workshop. By developing ADDE, we aim to:
*   Provide a more efficient, scalable, and principled approach for building high-quality, diverse datasets in domains underserved by current data resources.
*   Enhance the performance, robustness, and fairness of foundation models by improving the data they are trained on.
*   Reduce the significant human labor and cost associated with dataset creation for specialized applications.
*   Contribute to the understanding of model-data feedback loops and develop strategies for their beneficial and ethical management, explicitly addressing concerns raised by recent literature (Wyllie et al., 2024; Taori & Hashimoto, 2022).
*   Promote ethical data practices by embedding diversity considerations directly into the dataset construction pipeline.

The successful development of ADDE would represent a significant step forward in data-centric AI, enabling the broader and more responsible application of foundation models across diverse scientific and societal domains.

## 3. Methodology

The proposed ADDE framework operates iteratively, refining a dataset $D$ over multiple cycles. Let $f_\theta$ represent the foundation model being used or trained, $D_t$ be the dataset at iteration $t$, $D_{seed}$ be the initial small seed dataset, and $D_{pool}$ be a large pool of unlabeled data (optional, but beneficial if available).

**Overall Framework:**

The ADDE process can be summarized as:

Initialize $D_0 = D_{seed}$.
For $t = 0, 1, 2, ... T$ (until stopping criteria met):
1.  **Train/Update Model:** Train or fine-tune the model $f_{\theta_t}$ on the current dataset $D_t$.
2.  **Analyze Diversity & Identify Gaps:** Compute latent representations $Z_t = \{f_{\theta_t}(x) | x \in D_t\}$. Analyze the distribution of $Z_t$ to identify underrepresented regions $A_{sparse, t}$. Compute diversity metrics $M_{div}(D_t)$.
3.  **Targeted Data Acquisition:**
    a.  Generate synthetic candidates $D_{synth, t}$ targeting $A_{sparse, t}$ using $f_{\theta_t}$ or a derived generator $G_{\phi_t}$.
    b.  Optionally, select candidate real samples $D_{real, t}$ from $D_{pool}$ that potentially map to $A_{sparse, t}$ or have high uncertainty under $f_{\theta_t}$.
    c.  Combine candidates: $D_{cand, t} = D_{synth, t} \cup D_{real, t}$.
4.  **Active Learning & Human Validation:**
    a.  Select a subset $D_{select, t} \subset D_{cand, t}$ for human review using an active learning strategy $S_{AL}$.
    b.  Obtain human feedback $F_{human, t}$ (validation, correction, labels) on $D_{select, t}$.
    c.  Form the validated set $D_{valid, t}$ based on $F_{human, t}$.
5.  **Dataset Augmentation & Refinement:**
    a.  Update the dataset: $D_{t+1} = D_t \cup D_{valid, t}$.
    b.  Evaluate quality metrics $M_{qual}(D_{t+1})$ (e.g., performance on holdout validation set $D_{val}$).
    c.  Optionally, adapt parameters for the next iteration (e.g., clustering method, generation strategy, $S_{AL}$) based on $M_{div}(D_t)$, $M_{qual}(D_{t+1})$, and $F_{human, t}$.
6.  **Check Stopping Criteria:** Halt if budget exhausted, diversity/quality metrics plateau, or target performance achieved.

**Detailed Algorithmic Steps:**

1.  **Phase 1: Model Training & Diversity Analysis:**
    *   **Model:** We will utilize a pre-trained foundation model relevant to the target domain (e.g., a Vision Transformer for imaging, a large language model for text-based climate reports) and fine-tune it ($f_{\theta_t}$) on $D_t$.
    *   **Embeddings:** Extract final layer hidden states or pooled outputs as latent representations $z = f_{\theta_t}(x)$.
    *   **Diversity Analysis & Gap Identification:**
        *   Apply dimensionality reduction (e.g., UMAP, PCA) followed by density-based clustering (e.g., HDBSCAN) or k-means on $Z_t$ to identify clusters and their densities. $A_{sparse, t}$ will correspond to low-density regions or small, isolated clusters.
        *   **Diversity Metrics ($M_{div}$):** We will define and track metrics such as:
            *   *Latent Space Coverage:* Volume or entropy of the convex hull or alpha-shape of $Z_t$.
            *   *Cluster Balance:* Entropy of the cluster size distribution: $H(P) = -\sum_i p_i \log p_i$, where $p_i$ is the proportion of data in cluster $i$.
            *   *Minimum Inter-Cluster Distance / Maximum Intra-Cluster Distance:* To assess separation and compactness.
            *   *Subgroup Representation:* If known sensitive attributes or categories exist (even if coarsely defined), track their representation within clusters and overall. Define a target distribution if possible (e.g., aiming for uniform coverage).

2.  **Phase 2: Targeted Data Acquisition:**
    *   **Synthetic Generation ($D_{synth, t}$):**
        *   Option 1 (Domain-specific generators): Train a conditional generative model (cGAN, cVAE) $G_{\phi_t}$ conditioned on cluster IDs or representations from $A_{sparse, t}$. Generate $x_{synth} \sim G_{\phi_t}(c)$ where $c$ represents a target sparse region.
        *   Option 2 (Leveraging Large Generative Models): For text/image domains, use prompting techniques with large pre-trained models (e.g., Stable Diffusion, GPT-4) describing the characteristics of the underrepresented data (derived from analyzing samples near $A_{sparse, t}$).
        *   *Quality Control:* Filter synthetic samples using model confidence scores from $f_{\theta_t}$, or agreement scores across multiple diverse models (inspired by Erfanian et al., 2024).
    *   **Real Data Selection ($D_{real, t}$):**
        *   If $D_{pool}$ exists, select samples based on:
            *   *Uncertainty Sampling:* Select $x \in D_{pool}$ where $f_{\theta_t}(x)$ has maximum uncertainty (e.g., highest entropy over predicted probabilities, lowest maximum probability).
            *   *Diversity/Representativeness Sampling:* Select $x \in D_{pool}$ whose embeddings $f_{\theta_t}(x)$ fall into or are closest to the identified sparse regions $A_{sparse, t}$.

3.  **Phase 3: Active Learning & Human Validation ($D_{select, t}$, $F_{human, t}$, $D_{valid, t}$):**
    *   **Active Learning Strategy ($S_{AL}$):** Prioritize samples from $D_{cand, t}$ for human review based on a combination of:
        *   Uncertainty under the current model $f_{\theta_t}$.
        *   Diversity contribution (e.g., samples furthest from existing cluster centroids in $Z_t$, samples predicted to fall in $A_{sparse, t}$).
        *   Predicted quality/realism (if using synthetic data, potentially prioritize those with medium confidence - not too easy, not obvious junk).
        *   We may explore hybrid strategies, potentially framing it as a multi-objective optimization problem. The query selection can be formalized as:
          $$ x^* = \arg \max_{x \in D_{cand, t}} \alpha \cdot U(x | f_{\theta_t}) + \beta \cdot Div(x | Z_t) + \gamma \cdot Q(x) $$
          where $U$ is uncertainty, $Div$ is diversity contribution, $Q$ is predicted quality, and $\alpha, \beta, \gamma$ are weighting factors potentially adapted over time.
    *   **Human Feedback Interface:** Design an interface allowing annotators to efficiently:
        *   Validate/reject synthetic data realism.
        *   Assign/correct labels.
        *   Provide brief critiques or flags (e.g., "biased content," "edge case," "ambiguous"), inspired by Yu et al. (2023). This qualitative feedback can inform future generation or analysis.
    *   **Validation:** Only data points confirmed or corrected by humans form $D_{valid, t}$.

4.  **Phase 4: Iteration & Refinement:**
    *   The loop continues, with $f_{\theta_{t+1}}$ trained on $D_{t+1} = D_t \cup D_{valid, t}$. Metrics $M_{div}(D_{t+1})$ and $M_{qual}(D_{t+1})$ (performance on $D_{val}$) are tracked.
    *   **Adaptation:** If diversity metrics plateau while quality improves, the generation strategy might be broadened. If quality drops (e.g., due to poor synthetic data), filters might be tightened, or more weight given to real data selection. Weights in $S_{AL}$ ($\alpha, \beta, \gamma$) can be adjusted based on which types of samples yielded the most informative feedback or performance gain.
    *   **Stopping Criteria:** Predefined budget (number of annotations, computation hours), number of iterations, or convergence (e.g., negligible improvement in $M_{div}$ and $M_{qual}$ over several iterations).

**Experimental Design:**

*   **Domains & Datasets:**
    *   *Domain 1: Biomedical Imaging (Rare Disease Detection):* Use a public dataset like CheXpert (Irvin et al., 2019) or MIMIC-CXR (Johnson et al., 2019). Simulate rarity by downsampling specific pathology classes for $D_{seed}$. $D_{pool}$ will be the unlabeled images. $D_{val}$ will be a held-out set with balanced representation of pathologies.
    *   *Domain 2: Climate Informatics (Extreme Weather Event Detection):* Use satellite imagery datasets (e.g., Sentinel-2) and corresponding weather records. $D_{seed}$ will contain common weather patterns. Aim to improve detection of rare events (e.g., specific types of cyclones, localized floods) by targeting associated visual features. $D_{pool}$ = unlabeled satellite tiles. $D_{val}$ = curated set of diverse weather events.
*   **Foundation Models:** Use relevant pre-trained models (e.g., ResNet/ViT variant pre-trained on ImageNet/medical images for Domain 1; potentially a spatio-temporal model for Domain 2). Generative models might include StyleGAN variants or Diffusion Models fine-tuned on the domain.
*   **Baseline Methods:**
    1.  **Baseline-Seed:** Model trained only on $D_{seed}$.
    2.  **Baseline-Random:** Augment $D_{seed}$ by randomly sampling data from $D_{pool}$ for human annotation (same budget as ADDE).
    3.  **Baseline-AL:** Standard active learning (uncertainty sampling) on $D_{pool}$ without targeted generation or diversity focus.
    4.  **Baseline-MADS-Naive:** Simple MADS: Generate synthetic data (or use model predictions for pseudo-labeling) without diversity targeting or active learning selection, potentially mimicking naive feedback loops. Filter based on simple confidence threshold.
    5.  **Baseline-Static-Aug:** Generate a batch of diversity-enhancing data (e.g., using a method like Chameleon - Erfanian et al., 2024) once based on $D_{seed}$, then train. No iterative refinement.
*   **Evaluation Metrics:**
    *   *Dataset Metrics:* $M_{div}$ (as defined above), annotation cost (number of labeled samples vs. performance), ratio of synthetic vs. real data added. Measure distribution distance (e.g., Maximum Mean Discrepancy - MMD) between the latent space of the constructed dataset $Z_t$ and a target distribution (if available/definable for $D_{val}$).
    *   *Downstream Task Metrics:*
        *   Performance: Accuracy, F1-score (macro/micro), AUC, domain-specific metrics (e.g., Dice score for segmentation tasks if applicable, event detection accuracy). Evaluate on $D_{val}$ and potentially a separate OOD test set $D_{ood}$.
        *   Robustness: Performance degradation between $D_{val}$ and $D_{ood}$.
        *   Fairness: Measure performance gaps across identified subgroups (e.g., demographic groups if available, or clusters representing different data types/pathologies). Use metrics like Equalized Odds, Demographic Parity difference. Track bias amplification by comparing fairness metrics over iterations $t$ for ADDE vs. Baseline-MADS-Naive.
*   **Analysis:**
    *   Ablation studies: Evaluate ADDE variants by removing components (diversity targeting, active learning strategy, synthetic generation).
    *   Sensitivity analysis: Vary hyperparameters (cluster number, generation temperature, AL weights $\alpha, \beta, \gamma$).
    *   Qualitative analysis: Visualize latent space evolution (UMAP plots colored by cluster/iteration), examine samples selected by AL and generated synthetically. Analyze human feedback patterns.

## 4. Expected Outcomes & Impact

**Expected Outcomes:**

1.  **A Functional ADDE Framework:** We expect to deliver a robust, open-source implementation of the ADDE framework adaptable to different domains and foundation models.
2.  **Demonstrably Improved Datasets:** Datasets constructed using ADDE in our chosen experimental domains (biomedical imaging, climate informatics) are expected to exhibit significantly higher diversity scores ($M_{div}$) and better coverage of rare phenomena compared to datasets built using baseline methods under the same annotation budget. We anticipate achieving a measurable increase in latent space coverage and cluster balance entropy.
3.  **Enhanced Model Performance and Robustness:** Foundation models trained on ADDE-generated datasets are expected to show improved performance on the primary downstream task (e.g., accuracy, F1-score on $D_{val}$), particularly demonstrating greater robustness against distributional shifts (i.e., smaller performance drop on $D_{ood}$) compared to models trained on baseline datasets.
4.  **Improved Fairness and Bias Mitigation:** We expect ADDE to produce models with reduced performance disparities across identified data subgroups compared to baselines, especially the naive MADS approach. By explicitly monitoring diversity and potentially incorporating fairness constraints into the feedback loop, ADDE should actively mitigate the bias amplification documented in prior work (Taori & Hashimoto, 2022; Wyllie et al., 2024).
5.  **Quantifiable Efficiency Gains:** We hypothesize that ADDE will achieve target levels of model performance and data diversity with substantially fewer human annotations compared to random sampling or standard active learning. We aim to demonstrate annotation cost reductions in the range of 30-50% for achieving comparable performance levels, as suggested in the initial idea.
6.  **Novel Insights:** The research will yield insights into: (a) effective quantitative metrics for dataset diversity in the context of MADS, (b) synergistic combinations of generative models and active learning for targeted data acquisition, and (c) the dynamics of managing diversity and bias within iterative model-data feedback loops.

**Impact:**

*   **Scientific Impact:** This research will make significant contributions to the field of data-centric AI by providing a novel, principled framework for adaptive dataset construction. It advances the state-of-the-art in model-assisted data curation, active learning, and the application of generative models for targeted data enhancement. Furthermore, it directly addresses a critical challenge in contemporary ML research – understanding and controlling model-data feedback loops – offering a constructive approach to harness them for improving data quality and diversity rather than amplifying bias.
*   **Practical Impact:** ADDE offers a pathway to overcome the data bottleneck hindering the application of powerful foundation models in specialized, high-impact domains where data is scarce or expensive to acquire. This could accelerate progress in areas like personalized medicine (by improving diagnosis on rare conditions), climate change modeling (by better capturing extreme events), and safe robotics (by covering edge cases in perception). The potential for significant cost reduction in annotation makes advanced AI more accessible for researchers and organizations with limited resources.
*   **Ethical Impact:** By embedding diversity analysis and bias mitigation directly into the dataset creation process, ADDE promotes the development of more equitable and responsible AI systems. It provides a concrete methodology for operationalizing ethical AI principles related to data representation and fairness from the ground up, contributing to best practices for data governance and curation in the age of large models.
*   **Workshop Relevance:** This proposal directly aligns with the workshop's core themes, including model-assisted dataset construction, data quality and diversity signals, datasets for specific applications, ethical considerations, and managing data feedback dynamics. The expected outcomes will provide valuable insights and practical tools relevant to the workshop community, fostering discussion and further research in data-centric approaches for foundation models.

In summary, the proposed research on the ADDE framework promises to deliver both foundational advancements in data-centric machine learning methodology and practical solutions for building better, fairer, and more robust foundation models across a wider spectrum of critical application domains.