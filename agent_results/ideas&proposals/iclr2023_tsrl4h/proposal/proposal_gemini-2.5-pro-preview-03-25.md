# **Clin-ACT: Clinician-in-the-Loop Active Contrastive Learning for Robust and Interpretable Pediatric ICU Time Series Representation**

## 1. Introduction

### 1.1 Background
Time series data are ubiquitous in modern healthcare, particularly in high-acuity settings like the Pediatric Intensive Care Unit (PICU). Vital signs, laboratory results, and therapeutic interventions, recorded over time, offer a rich, dynamic view of a patient's physiological state. Machine learning (ML), especially representation learning, holds immense potential to unlock insights from these complex datasets, driving advancements in clinical decision support systems for tasks such as early disease detection (e.g., sepsis), patient stratification, and outcome prediction (Johnson et al., 2016; Harutyunyan et al., 2019).

However, translating ML potential into clinical practice faces significant hurdles, especially in challenging domains like pediatrics. PICU time series data are characterized by:
*   **High Dimensionality and Multimodality:** Data streams from various sources (monitors, labs, notes) create high-dimensional inputs.
*   **Irregular Sampling and Missingness:** Measurements are often taken at irregular intervals based on clinical need, leading to sparse data with significant missing values, patterns of which can themselves be informative.
*   **Label Scarcity:** Obtaining high-quality labels for clinical events (e.g., sepsis onset) requires time-consuming review by expert clinicians, whose availability is limited. This bottleneck severely restricts the scale of supervised learning.
*   **Need for Robustness and Trust:** Clinical applications demand models that are robust to data imperfections (outliers, missingness) and whose predictions are interpretable and trustworthy for high-stakes decision-making.

Recent advances in self-supervised learning (SSL) offer promising avenues for learning representations from unlabeled time series data (Yue et al., 2022). Methods like contrastive learning (Chen et al., 2020) learn representations by maximizing agreement between different augmented views of the same data instance while minimizing agreement with other instances. Several studies have adapted SSL for clinical time series, addressing missing data (Ghaderi et al., 2023; Tipirneni & Reddy, 2021; Wever et al., 2021) and multimodality (Baldenweg et al., 2024). These approaches effectively leverage large amounts of unlabeled data.

Despite these advancements, significant gaps remain. Existing SSL methods often treat all unlabeled data equally and may not optimally utilize the limited labeling budget available. Standard augmentations might not fully capture the nuances of clinical time series irregularities and missingness patterns. Furthermore, the learned representations, while effective for downstream tasks, often lack direct interpretability, hindering clinician adoption. Simple fine-tuning or linear probing on learned representations doesn't guarantee that the model leverages clinically meaningful patterns, nor does it provide transparency into the reasoning process.

This proposal introduces **Clin-ACT (Clinician-in-the-Loop Active Contrastive Learning)**, a novel framework designed to address these challenges specifically within the context of pediatric ICU time series. Clin-ACT synergistically combines:
1.  **Imputation-Aware Contrastive Learning:** An SSL encoder trained with tailored augmentations that explicitly model clinically relevant data characteristics like irregular sampling and informative missingness, enforcing representation robustness without resorting to potentially harmful pre-processing imputation.
2.  **Uncertainty-Diversity Active Learning:** An intelligent sampling strategy that identifies the most informative time series windows for which to request clinician labels, maximizing model performance gain while minimizing the annotation burden.
3.  **Prototype-Based Interpretability:** A lightweight module that maps the learned representations to clinically intuitive archetypes and provides feature saliency maps, enhancing transparency and trust.

By integrating these components, Clin-ACT aims to learn label-efficient, robust, and interpretable representations from sparse, irregular pediatric ICU time series, making representation learning more actionable for critical care applications.

### 1.2 Research Objectives
The primary goal of this research is to develop and validate the Clin-ACT framework for learning effective representations from pediatric ICU time series data. The specific objectives are:

1.  **Develop the Clin-ACT Framework:** Implement the three core components:
    *   An imputation-aware contrastive learning encoder suitable for multivariate, irregular time series.
    *   An active learning module integrating uncertainty and diversity criteria for efficient sample selection.
    *   A prototype-based interpretability layer mapping embeddings to clinical archetypes and generating saliency maps.
2.  **Enhance Label Efficiency:** Demonstrate that Clin-ACT significantly reduces the number of clinician-provided labels required to achieve high performance on a clinically relevant downstream task (e.g., sepsis prediction), compared to random sampling and standard supervised/self-supervised baselines. We hypothesize a label reduction of at least 60% for comparable performance.
3.  **Improve Downstream Task Performance:** Show that the representations learned by Clin-ACT lead to superior performance (targeting a +12% improvement in AUROC/AUPRC for sepsis detection compared to standard SSL with the same label budget) on downstream clinical prediction tasks, particularly in low-label regimes.
4.  **Ensure Robustness:** Evaluate the robustness of the learned representations to varying levels of missing data and sampling irregularity compared to baseline methods.
5.  **Validate Interpretability and Clinical Utility:** Assess the interpretability of the prototype layer and saliency maps through qualitative evaluation with pediatric clinicians, focusing on clinical relevance, understandability, and trustworthiness.

### 1.3 Significance
This research is significant for several reasons:

*   **Addresses Critical Clinical Needs:** It tackles the pervasive problem of label scarcity in clinical data analysis, particularly in specialized, under-resourced areas like pediatric critical care. By enabling high-performance models with fewer labels, Clin-ACT can accelerate the development of decision support tools.
*   **Enhances Trust and Adoption:** The integrated interpretability layer directly addresses the "black box" problem, a major barrier to ML adoption in healthcare. Providing clinicians with intuitive explanations can foster trust and facilitate the integration of ML insights into clinical workflows.
*   **Methodological Innovation:** Clin-ACT introduces a novel integration of imputation-aware contrastive learning, active learning tailored for time series, and prototype-based interpretability. The proposed augmentations and active learning strategies are specifically designed for the challenges of clinical time series data.
*   **Focus on Underserved Population:** By concentrating on pediatric ICU data, this research addresses a minority data group with unique physiological characteristics and data challenges, aligning with the workshop's emphasis on actionable research in specific clinical areas.
*   **Contribution to Time Series Representation Learning:** The findings will contribute to the broader field of time series representation learning by providing insights into robust augmentation strategies, efficient active learning techniques for sequential data, and methods for interpretable representation learning in high-stakes domains.

## 2. Methodology

### 2.1 Data Collection and Preparation
*   **Dataset:** We plan to utilize publicly available pediatric ICU datasets, primarily the pediatric subset of MIMIC-IV (Medical Information Mart for Intensive Care) (Johnson et al., 2023). MIMIC-IV contains de-identified data for thousands of pediatric patients, including high-resolution vital signs, laboratory measurements, medication administrations, and clinical notes. If feasible and ethically approved (IRB review), we may supplement this with data from local collaborating institutions for external validation.
*   **Data Modalities:** We will focus on multivariate time series data, including:
    *   **Vital Signs:** Heart Rate (HR), Respiratory Rate (RR), Peripheral Oxygen Saturation (SpO2), Blood Pressure (Systolic, Diastolic, Mean), Temperature. Sampled at relatively high frequency but often irregularly.
    *   **Laboratory Results:** Blood gas values, complete blood count, electrolytes, lactate, C-reactive protein (CRP), etc. Sampled sparsely and irregularly.
*   **Cohort Selection:** We will define specific patient cohorts based on age (e.g., neonates, infants, children) and potentially admission diagnosis, focusing on patients at risk for conditions like sepsis.
*   **Preprocessing:**
    *   **Time Windowing:** Segment continuous patient stays into fixed-length overlapping or non-overlapping windows (e.g., 4, 8, or 12 hours).
    *   **Feature Engineering:** Minimal feature engineering will be performed. We will include time elapsed since the previous measurement for each variable as an explicit input feature to handle irregularity, similar to (Tipirneni & Reddy, 2021). We will also include indicators for whether a value is observed or missing.
    *   **Normalization:** Apply appropriate normalization (e.g., z-score normalization based on population statistics, or robust scaling) to each feature channel. Extreme outliers might be clipped based on clinically plausible ranges after consultation with domain experts.
    *   **Imputation:** Importantly, *no upfront imputation* of missing values will be performed. The model architecture and training process are designed to handle missingness directly.
*   **Labeling:** For the downstream task (e.g., Sepsis-3 prediction), labels will be generated based on established clinical criteria applied retrospectively to the EHR data. For the active learning component, we will simulate the clinician interaction by revealing these ground-truth labels only for the actively selected samples.

### 2.2 Clin-ACT Framework: Algorithmic Details

The Clin-ACT framework consists of three main components trained iteratively or sequentially.

**2.2.1 Component 1: Imputation-Aware Contrastive Encoder**
*   **Architecture:** We will employ a Transformer-based architecture, inspired by models like STraTS (Tipirneni & Reddy, 2021), which are effective at capturing long-range dependencies and handling set-based inputs (suitable for irregular observations). The input to the Transformer will consist of observation tuples $(v_j, t_j, m_j)$, where $v_j$ is the measurement value (or a special token if missing), $t_j$ is the time of measurement, and $m_j$ identifies the variable/modality. Continuous value embedding and temporal encoding (e.g., sinusoidal positional encoding adapted for irregular time) will be used.
*   **Imputation-Aware Augmentations:** To generate positive pairs for contrastive learning, we apply stochastic augmentations to each input window $x$. Crucially, these augmentations are designed to mimic clinical data realities:
    *   **Sub-sampling/Temporal Jitter:** Randomly drop a fraction of observations or slightly perturb timestamps to simulate sampling variability.
    *   **Variable Masking:** Completely mask out one or more variable channels for a portion of the window's duration.
    *   **Observation Masking (Imputation Simulation):** Randomly mask a fraction ($p_{mask}$) of the *observed* values within the window, replacing them with a special 'mask' token. This forces the encoder to infer representations even when parts of the expected data are missing, promoting robustness to missingness patterns.
    *   **Gaussian Noise:** Add small Gaussian noise $N(0, \sigma^2)$ to observed values, with $\sigma$ calibrated based on known sensor noise or plausible physiological variation.
*   **Contrastive Loss:** We use the InfoNCE loss (Oord et al., 2018) to train the encoder $\phi$. Given a batch of time series windows $\{x_i\}_{i=1}^N$, we generate two augmented views $x_i'$ and $x_i''$ for each $x_i$. Let $z_i' = \phi(x_i')$ and $z_i'' = \phi(x_i'')$ be the encoded representations (e.g., the representation corresponding to the [CLS] token or mean-pooled output). The loss for anchor $z_i'$ is:
    $$
    \mathcal{L}_i = - \log \frac{\exp(\text{sim}(z_i', z_i'') / \tau)}{\sum_{j=1, k \in \{', ''\}}^N \mathbb{1}_{j \neq i \lor k \neq '} \exp(\text{sim}(z_i', z_j^k) / \tau)}
    $$
    where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{||\mathbf{u}|| ||\mathbf{v}||}$ is cosine similarity, and $\tau$ is a temperature hyperparameter. The total loss is the average over all anchors in the batch.

**2.2.2 Component 2: Uncertainty-Diversity Active Learning (AL)**
*   **Goal:** Select a batch of $B$ unlabeled windows from the pool $\mathcal{U}$ that are most informative for improving the model, given the current labeled pool $\mathcal{L}$.
*   **Process:** This is an iterative process:
    1.  Initialize: Train the encoder $\phi$ using SSL on all available data (labeled $\mathcal{L}_0$ and unlabeled $\mathcal{U}$). $\mathcal{L}_0$ might be empty or contain a small randomly selected seed set.
    2.  Proxy Task Training: Train a lightweight classification head $h$ (e.g., linear layer + softmax) on top of the frozen or lightly fine-tuned encoder $\phi$ using the current labeled set $\mathcal{L}_t$. This head predicts the downstream task (e.g., sepsis).
    3.  Acquisition Score Calculation: For each window $x_u \in \mathcal{U}$, compute its acquisition score $A(x_u)$.
    4.  Selection: Select the top $B$ samples $\mathcal{S} = \arg \max_{S \subset \mathcal{U}, |S|=B} \sum_{x_u \in S} A(x_u)$. (Approximated greedily or via batch selection methods).
    5.  Query Labels: Simulate clinician labeling by revealing the ground-truth labels for samples in $\mathcal{S}$.
    6.  Update Pools: $\mathcal{L}_{t+1} = \mathcal{L}_t \cup \mathcal{S}$, $\mathcal{U}_{t+1} = \mathcal{U} \setminus \mathcal{S}$.
    7.  Retrain/Fine-tune: Retrain or fine-tune the encoder $\phi$ and/or the head $h$ using $\mathcal{L}_{t+1}$. Optionally, the SSL pre-training can also be continued. Repeat from Step 2 until the labeling budget is exhausted.
*   **Acquisition Function:** $A(x_u)$ combines uncertainty and diversity:
    *   **Uncertainty ($U(x_u)$):** Measures how uncertain the current model $(h \circ \phi)$ is about the label of $x_u$. We will explore:
        *   *Predictive Entropy:* $H(p(y|x_u)) = -\sum_k p_k \log p_k$, where $p(y|x_u)$ is the probability distribution output by $h(\phi(x_u))$.
        *   *Bayesian Active Learning by Disagreement (BALD):* If using Bayesian approximations like MC-Dropout, $I(y; \omega | x_u, \mathcal{L}_t)$, measuring mutual information between prediction and model parameters.
    *   **Diversity ($D(x_u, \mathcal{S}_{\text{current}}, \mathcal{L}_t)$):** Measures how different $x_u$ is from already labeled samples and samples already selected in the current batch $\mathcal{S}_{\text{current}}$. We use distance in the embedding space: $D(x_u) = \min_{x_s \in \mathcal{S}_{\text{current}} \cup \mathcal{L}_t} ||\phi(x_u) - \phi(x_s)||_2^2$. This encourages exploration of the representation space.
    *   **Combination:** A common approach is a weighted combination or a sequential application (e.g., select top $k B$ samples by uncertainty, then use diversity to select $B$ from those $k B$). We will experiment with $A(x_u) = U(x_u)^\alpha \times D(x_u)^\beta$, tuning $\alpha, \beta$ via cross-validation on a held-out labeled set.

**2.2.3 Component 3: Prototype-Based Interpretability Layer**
*   **Goal:** Provide interpretable insights into the learned representations.
*   **Method:** We adapt ideas from Prototypical Networks (Snell et al., 2017) used in few-shot learning, applying them for interpretability. After the active learning phase, with the final encoder $\phi$ and labeled set $\mathcal{L}$, we define $K$ prototype vectors $\{\mathbf{p}_k\}_{k=1}^K$ in the embedding space $Z = \phi(X)$.
    *   **Prototype Learning:**
        1.  *Option 1 (Class-based):* If the downstream task has $C$ classes, we can compute $C$ prototypes, where each prototype $\mathbf{p}_c$ is the mean embedding of all labeled samples belonging to class $c$: $\mathbf{p}_c = \frac{1}{|\mathcal{L}_c|} \sum_{x_i \in \mathcal{L}_c} \phi(x_i)$. We can potentially learn more than one prototype per class (e.g., $K > C$) using clustering (k-means) within each class's embeddings.
        2.  *Option 2 (Learnable):* Introduce $K$ learnable prototype vectors $\{\mathbf{p}_k\}_{k=1}^K$. Train a final classification layer that computes distances to these prototypes. The probability of sample $x_i$ belonging to class $c$ (associated with a subset of prototypes $P_c$) could be modeled as:
            $$ p(y=c | x_i) = \frac{\sum_{\mathbf{p}_k \in P_c} \exp(-d(\phi(x_i), \mathbf{p}_k))}{\sum_{j=1}^K \exp(-d(\phi(x_i), \mathbf{p}_j))} $$
            where $d(\cdot, \cdot)$ is a distance function (e.g., squared Euclidean). The prototypes $\mathbf{p}_k$ and encoder $\phi$ (optionally) are fine-tuned to minimize cross-entropy loss on $\mathcal{L}$.
*   **Interpretation:**
    *   **Nearest Prototype:** For a new sample $x$, find its embedding $z = \phi(x)$ and identify the nearest prototype $\mathbf{p}_k$. This provides a classification based on similarity to learned archetypes ("This patient's state is closest to archetype $k$").
    *   **Prototype Visualization:** Visualize the characteristics of the learned prototypes. For each prototype $\mathbf{p}_k$, find the samples $x_i \in \mathcal{L}$ whose embeddings $\phi(x_i)$ are closest to $\mathbf{p}_k$. Analyze the average time series patterns (e.g., average vital signs, lab trajectories) of these representative samples. This helps clinicians assign semantic meaning to the prototypes (e.g., "stable," "early sepsis," "respiratory distress").
    *   **Feature Saliency:** To explain *why* a sample $x$ is mapped to a particular prototype $\mathbf{p}_k$ (or classified into a certain class via prototypes), generate saliency maps. We will use methods like:
        *   *Attention Weights:* If using a Transformer, visualize the attention scores, particularly in the final layers, to see which time points and variables contributed most to the final representation.
        *   *Gradient-based Methods:* Compute gradients of the similarity score $d(\phi(x), \mathbf{p}_k)$ (or the class probability) with respect to the input features $(v_j, t_j, m_j)$. Techniques like Integrated Gradients (Sundararajan et al., 2017) can provide feature attributions. These maps highlight which input values at which times were most influential.

### 2.3 Experimental Design and Validation
*   **Baselines:** We will compare Clin-ACT against several baselines:
    *   **Supervised Baseline:** Train the same encoder architecture (e.g., Transformer) directly on the downstream task using all available labels (if feasible) or the maximum label budget allowed.
    *   **Standard SSL + Linear Probe:** Train the encoder using a standard contrastive learning setup (e.g., SimCLR-style augmentations) on all data, then freeze the encoder and train a linear classifier using the same varying number of labels as Clin-ACT.
    *   **SSL + Random Sampling (RS) Active Learning:** Use the standard SSL encoder but acquire labels via random sampling instead of Clin-ACT's AL strategy.
    *   **SSL + Uncertainty-Only AL:** Use the standard SSL encoder and only the uncertainty part of the Clin-ACT AL acquisition function.
    *   **SSL + Diversity-Only AL:** Use the standard SSL encoder and only the diversity part of the Clin-ACT AL acquisition function.
    *   **Relevant Literature Baselines:** Adapt methods like STraTS (Tipirneni & Reddy, 2021) or SLAC-Time (Ghaderi et al., 2023) for supervised fine-tuning or active learning settings if possible.
*   **Downstream Task Evaluation:**
    *   **Task:** Pediatric sepsis prediction (binary classification, onset prediction within a future window, e.g., 4-12 hours).
    *   **Protocol:** For each method (Clin-ACT, baselines), simulate the active learning process starting with a small seed set (e.g., 1% of total potential labels) and incrementally adding batches ($B$ samples) of labels up to a predefined budget (e.g., 5%, 10%, 20%, 40% of total labels). At each budget level, evaluate the performance of a linear classifier trained on the representations learned by the respective method using the acquired labels. Plot performance curves (metric vs. % labels used). Use cross-validation (e.g., 5-fold) on patient splits to ensure robustness.
    *   **Metrics:** AUROC, AUPRC (crucial for imbalanced sepsis task), F1-score, Sensitivity (Recall), Specificity, Precision.
*   **Robustness Evaluation:** Evaluate performance degradation of Clin-ACT and baselines under artificially increased levels of missing data or injected noise/outliers in the test set.
*   **Interpretability Evaluation:**
    *   **Clinician Study:** Recruit 3-5 pediatric intensivists or fellows. Present them with case vignettes (patient time series windows) along with:
        *   Clin-ACT's prediction (e.g., sepsis risk).
        *   The closest prototype identified and its learned archetypal description (from prototype visualization).
        *   The feature saliency map highlighting influential data points.
    *   **Assessment:** Use a mixed-methods approach:
        *   *Quantitative:* Likert scale questionnaires assessing Understandability (is the explanation clear?), Clinical Plausibility (does the explanation make sense medically?), Trustworthiness (does the explanation increase confidence in the prediction?), Actionability (does the explanation provide useful information?).
        *   *Qualitative:* Semi-structured interviews using think-aloud protocols as clinicians review the explanations. Collect feedback on the perceived utility, limitations, and suggestions for improvement. Analyze interview transcripts using thematic analysis.

### 2.4 Evaluation Metrics
*   **Primary Performance Metrics:** AUROC, AUPRC, F1-Score vs. Number of Labels curve.
*   **Secondary Performance Metrics:** Sensitivity, Specificity, Precision at different label budgets.
*   **Robustness Metrics:** Performance drop (%) under increased missingness/noise.
*   **Interpretability Metrics:** Clinician ratings on Likert scales (Understandability, Plausibility, Trustworthiness, Actionability), qualitative feedback themes.
*   **Computational Metrics:** Training time, inference time.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes
1.  **A Fully Implemented Clin-ACT Framework:** An open-source implementation (where possible, respecting data privacy) of the proposed framework, including the imputation-aware contrastive encoder, the active learning module, and the prototype interpretability layer.
2.  **Demonstrated Label Efficiency:** Quantitative results showing that Clin-ACT achieves target downstream performance (e.g., sepsis prediction AUROC/AUPRC) using significantly fewer labels (expected >60% reduction) compared to random sampling strategies and potentially other AL baselines. We expect Clin-ACT to outperform standard SSL methods, especially in the low-label regime (e.g., <10% of total labels).
3.  **State-of-the-Art Performance:** Achievement of competitive or state-of-the-art performance on pediatric sepsis prediction benchmarks (like MIMIC-IV pediatric subset) under realistic label constraints. We anticipate a significant improvement (target +12%) in AUROC/AUPRC compared to baseline SSL methods using the same small label budget.
4.  **Robustness Validation:** Empirical evidence demonstrating that the representations learned by Clin-ACT are more robust to missing data and irregular sampling compared to baseline approaches that do not explicitly model these characteristics.
5.  **Clinically Relevant Interpretability:** Identification of meaningful clinical archetypes via the prototype layer, validated by clinicians. Positive feedback from pediatric clinicians indicating that the generated explanations (prototype similarity + saliency maps) are understandable, plausible, and increase trust in the model's predictions.
6.  **Dissemination:** Publication of findings in a leading machine learning, health informatics conference, or journal (e.g., NeurIPS, ICML, CHIL, JAMIA, Nature Medicine) and presentation at the workshop.

### 3.2 Impact
*   **Clinical Impact:** By drastically reducing the need for clinician annotation time, Clin-ACT can make the development of sophisticated ML models for pediatric critical care more feasible. Improved and trustworthy sepsis prediction models could lead to earlier interventions and better patient outcomes. The framework's interpretability features can foster clinician trust, facilitating the responsible integration of ML into clinical practice.
*   **Scientific Impact:** This research will contribute novel techniques to time series representation learning, particularly in handling challenges specific to clinical data (irregularity, missingness, label scarcity). The integration of SSL, AL, and interpretability offers a new paradigm for building practical and trustworthy ML systems in healthcare. The imputation-aware augmentations and prototype-based explanations are specific methodological contributions.
*   **Alignment with Workshop Goals:** This proposal directly addresses key themes of the workshop:
    *   **Labeling Challenges:** Tackled via active learning.
    *   **Missing Data/Irregularity:** Addressed by the model architecture and imputation-aware SSL.
    *   **Interpretability/Explainability:** Core component through the prototype layer and saliency.
    *   **Robustness:** Explicitly evaluated and targeted by the SSL design.
    *   **Actionable Clinical Practice:** Focus on sepsis prediction in pediatric ICU.
    *   **Minority Data Group:** Specifically targets pediatrics.
*   **Broader Applicability:** While focused on pediatric ICU data, the Clin-ACT methodology could be adapted to other healthcare domains (e.g., adult ICU, chronic disease monitoring, rare diseases) facing similar challenges of complex time series data and limited expert labels. It may also be relevant to non-medical time series applications where interpretability and label efficiency are crucial.

This research holds the promise of making advanced time series representation learning techniques more practical, trustworthy, and ultimately beneficial for patient care in resource-constrained and high-stakes clinical environments.