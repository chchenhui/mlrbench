# **Research Proposal**

## 1. Title

**Dynamic Modality Reliability Estimation for Trustworthy Multi-modal Medical Fusion**

## 2. Introduction

### 2.1 Background

Machine learning (ML) holds transformative potential for healthcare, demonstrating remarkable success in tasks ranging from medical image analysis to clinical outcome prediction. The increasing availability of diverse medical data modalities—such as Computed Tomography (CT), Magnetic Resonance Imaging (MRI), pathology slides, Electronic Health Records (EHRs), and genomic data—presents a significant opportunity. Multi-modal fusion methods, which integrate information from these diverse sources, promise more holistic patient assessments, leading to potentially improved diagnostic accuracy, prognosis prediction, and treatment planning compared to single-modality approaches.

However, despite promising results in controlled research settings, the translation of these multi-modal ML models into routine clinical practice remains slow. A primary barrier is the "trust gap": clinicians and patients often lack confidence in the reliability and robustness of ML predictions, particularly in high-stakes medical decisions. This distrust stems from several sources, including the "black-box" nature of complex models, concerns about generalization to new patient populations or data acquisition protocols, potential biases encoded in the data, and sensitivities to data quality issues.

A critical challenge specific to multi-modal fusion is the handling of *variable modality reliability*. In real-world clinical scenarios, data from different modalities can vary significantly in quality and relevance for a given patient or task. For instance, an MRI scan might be corrupted by motion artifacts, EHR data might be incomplete or contain errors, or a specific imaging modality might be less informative for a particular disease subtype. Current fusion strategies often implicitly assume uniform reliability across all modalities or employ static weighting schemes. This assumption is fragile; when one or more modalities are noisy, missing, biased, or simply uninformative, these models can produce overconfident yet erroneous predictions, potentially leading to incorrect clinical actions. As highlighted by recent works like MDA [1], DRIFA-Net [2], HEALNet [3], and DrFuse [4], addressing modality heterogeneity, missing data, noise, interpretability, and uncertainty is paramount. While these methods offer advancements in attention mechanisms, handling missingness, and basic uncertainty estimation (e.g., via Monte Carlo dropout), they often lack a dedicated mechanism to *dynamically assess the intrinsic reliability* of each modality *at inference time* based on the specific input data and integrate this assessment directly into the fusion process.

### 2.2 Research Objectives

This research aims to bridge this critical gap by developing a novel multi-modal fusion framework that explicitly models and leverages modality-specific reliability to enhance trustworthiness. The core objectives are:

1.  **Develop a Dynamic Modality Reliability Estimation Framework:** Design and implement a multi-modal fusion architecture that incorporates Bayesian principles to quantify the uncertainty associated with the features derived from each individual modality. This uncertainty will serve as a proxy for modality reliability.
2.  **Integrate Reliability into Fusion using Attention:** Develop a reliability-guided attention mechanism that dynamically adjusts the contribution of each modality to the final prediction based on its estimated reliability. Modalities deemed less reliable (higher uncertainty) will be down-weighted.
3.  **Enhance Reliability Assessment via Self-Supervision:** Introduce a self-supervised auxiliary task during training that explicitly teaches the model to recognize and quantify indicators of modality corruption (e.g., noise, missingness). This aims to improve the model's intrinsic ability to assess data quality.
4.  **Validate Robustness and Uncertainty Quantification:** Rigorously evaluate the proposed framework's robustness against various forms of simulated and potentially real-world modality degradation (e.g., varying levels of noise, missing data, domain shifts) using established medical datasets. Assess the quality of the uncertainty estimates produced by the model.
5.  **Improve Interpretability:** Demonstrate that the reliability estimates and attention weights provide meaningful insights into the model's decision-making process, indicating which modalities are trusted for a given prediction, thereby enhancing transparency.

### 2.3 Significance

This research directly addresses the critical need for trustworthy ML in healthcare, aligning with the core themes of the workshop. By dynamically assessing modality reliability, the proposed framework offers several significant contributions:

1.  **Enhanced Robustness:** The model is expected to be more resilient to real-world data imperfections like noise, artifacts, or incomplete data, leading to more stable and reliable predictions in clinical settings.
2.  **Improved Safety:** By providing uncertainty-aware predictions and down-weighting unreliable inputs, the framework can reduce the risk of overconfident errors, potentially flagging cases where human review is essential.
3.  **Increased Transparency:** The modality-specific reliability estimates and attention weights offer a degree of interpretability, allowing clinicians to understand *which* data sources the model relied upon, fostering trust and facilitating error analysis.
4.  **Contribution to Trustworthy AI:** This work contributes a novel technique to the broader field of trustworthy ML, specifically addressing uncertainty quantification and robustness in the context of multi-modal learning.
5.  **Potential Clinical Impact:** A successful framework could accelerate the adoption of multi-modal AI tools in clinical workflows for tasks like diagnosis, prognosis, and treatment response prediction, ultimately aiming for improved patient outcomes. It also provides a methodological step towards building more adaptive and context-aware medical AI systems.

## 3. Methodology

### 3.1 Overall Framework Architecture

We propose a multi-modal fusion framework, referred to as **D**ynamic **R**eliability-**A**ware **M**edical **F**usion **Net** (DRAM-Net). The architecture consists of the following key components (See conceptual flow below):

1.  **Input Modalities:** The framework accepts multiple input modalities ($X_1, X_2, ..., X_M$), where $M$ is the number of modalities (e.g., CT image, MRI image, EHR features).
2.  **Modality-Specific Encoders with Uncertainty Estimation:** Each modality $X_m$ is processed by a dedicated encoder $E_m$. Crucially, these encoders will incorporate mechanisms for uncertainty quantification, leveraging Bayesian Neural Networks (BNNs). We plan to explore practical BNN implementations like Monte Carlo (MC) Dropout [5] or Variational Inference (VI) [6].
    *   *MC Dropout:* Standard network layers (e.g., CNNs for images, Transformers or MLPs for EHRs) are used, but dropout is applied not only during training but also during inference. Multiple forward passes ($T$ times) with active dropout allow sampling from the approximate posterior distribution of the model weights.
    *   *Variational Inference:* Replace standard layers with variational layers where weights are distributions (e.g., Gaussian) rather than point estimates. The training objective includes minimizing the Kullback-Leibler (KL) divergence between the approximate posterior and the prior.
    *   From $E_m(X_m)$, we obtain not only a feature representation $f_m$ but also an uncertainty estimate $u_m$. For MC Dropout, $f_m = \frac{1}{T}\sum_{t=1}^{T} E_m(X_m; \theta_t)$ and $u_m = \text{Var}_{t=1..T}(E_m(X_m; \theta_t))$, representing predictive variance (a measure of total uncertainty). For VI, uncertainty can be derived from the variance of the approximate posterior predictive distribution.
3.  **Self-Supervised Reliability Assessment Module:** An auxiliary task is introduced during training. We synthetically corrupt input modalities (e.g., add Gaussian noise, apply random masking, simulate artifacts) with known types and levels. An auxiliary prediction head $H_{aux}$ takes the features $f_m$ (and potentially $u_m$) and predicts the type or level of corruption $c_m$ applied to the input $X_m$. The loss $L_{aux} = \sum_{m=1}^{M} L_{corruption}(H_{aux}(f_m), c_m)$ encourages the encoders $E_m$ to learn representations sensitive to data quality issues.
4.  **Reliability-Guided Attention Mechanism:** An attention module $A$ computes attention weights $\alpha_m$ for each modality based on both its feature representation $f_m$ and its uncertainty estimate $u_m$. High uncertainty should lead to lower attention. A potential formulation is:
    $$ z_m = W_f f_m + b_f $$
    $$ s_m = W_u \sigma(u_m) + b_u $$
    $$ e_m = v^T \tanh(z_m - \lambda s_m) $$
    $$ \alpha_m = \frac{\exp(e_m / \tau)}{\sum_{j=1}^{M} \exp(e_j / \tau)} $$
    where $W_f, b_f, W_u, b_u, v$ are learnable parameters, $\sigma(u_m)$ is a possibly scaled version of the uncertainty estimate (e.g., using sigmoid or log), $\lambda$ is a hyperparameter controlling the influence of uncertainty, and $\tau$ is a temperature parameter. The subtraction $z_m - \lambda s_m$ explicitly penalizes modalities with higher uncertainty. Other attention formulations incorporating uncertainty will also be explored.
5.  **Fusion and Prediction:** The modality features are combined using the attention weights to produce a fused representation $f_{fused} = \sum_{m=1}^{M} \alpha_m f_m$. This fused representation is then fed into a final prediction head $H_{pred}$ (e.g., an MLP) to produce the output $\hat{y}$ for the primary clinical task (e.g., disease classification, survival prediction).

### 3.2 Data Collection and Preprocessing

We plan to utilize publicly available multi-modal medical datasets to ensure reproducibility. Potential candidates include:

1.  **MIMIC-IV [7] and MIMIC-CXR [8]:** Combines EHR data (diagnoses, procedures, medications, lab results) with chest X-ray images. Suitable for tasks like mortality prediction or disease diagnosis.
2.  **The Cancer Genome Atlas (TCGA) [9]:** Provides multi-omic data (genomics, transcriptomics) along with histopathology images and clinical data for various cancer types. Suitable for survival analysis or cancer subtyping.
3.  **BraTS (Brain Tumor Segmentation Challenge) [10]:** Contains multi-parametric MRI scans (T1, T1ce, T2, FLAIR) for brain tumor segmentation. Suitable for evaluating robustness in imaging tasks.

Data preprocessing will follow standard practices for each modality:
*   **Images (CT/MRI/X-ray/Pathology):** Resizing, normalization (e.g., Z-score), potentially data augmentation (rotations, flips, intensity shifts).
*   **EHR Data:** Feature engineering (e.g., creating time-series representations, embedding categorical variables), handling missing values (using standard imputation techniques initially, although the model aims to handle missingness implicitly), normalization of numerical features.
*   **Genomic Data:** Standard pipelines for alignment, quantification, and normalization depending on the data type.

Crucially, for validation, we will programmatically introduce **controlled modality corruption**:
*   **Missing Modalities:** Randomly mask out entire modalities for a subset of samples during training and testing (at varying percentages).
*   **Noise Injection:** Add varying levels of Gaussian noise or modality-specific noise (e.g., simulating motion artifacts in MRI, salt-and-pepper noise in X-rays).
*   **Data Degradation:** Apply blurring, downsampling, or reduce feature density for EHRs.
*   **Domain Shift Simulation:** Potentially use data from different hospitals or scanners if available, or simulate shifts by altering data distributions (e.g., intensity shifts).

### 3.3 Algorithmic Steps & Training

The training process involves optimizing the parameters of the encoders $E_m$, the attention mechanism $A$, the auxiliary head $H_{aux}$, and the prediction head $H_{pred}$ end-to-end.

1.  **Input:** A batch of multi-modal samples $(X_1, ..., X_M, y)$, where $y$ is the ground truth for the primary task.
2.  **Corruption (for auxiliary task):** For a subset of samples/modalities in the batch, apply synthetic corruption $c_m$ to input $X_m$ to get $X'_m$. Other inputs remain $X_m$.
3.  **Forward Pass:**
    *   Process each (potentially corrupted) input $X'_m$ or $X_m$ through its respective encoder $E_m$. If using MC Dropout, perform $T$ forward passes to get $\{E_m(X'_m; \theta_t)\}_{t=1..T}$.
    *   Compute features $f_m$ (mean of embeddings over $T$ passes) and uncertainty $u_m$ (variance of embeddings over $T$ passes).
    *   Predict corruption level $\hat{c}_m = H_{aux}(f_m)$.
    *   Compute attention weights $\alpha_m = A(f_1, ..., f_M, u_1, ..., u_M)$.
    *   Compute fused representation $f_{fused} = \sum_{m=1}^{M} \alpha_m f_m$.
    *   Compute final prediction $\hat{y} = H_{pred}(f_{fused})$. If using MC Dropout for the predictor too, $\hat{y}$ can also represent a distribution mean, and overall predictive uncertainty can be estimated.
4.  **Loss Computation:**
    *   Primary Task Loss $L_{primary}$: Based on the task (e.g., Cross-Entropy for classification, Mean Squared Error for regression, Dice Loss for segmentation). Calculated between $\hat{y}$ and $y$.
    *   Auxiliary Task Loss $L_{aux}$: Loss for predicting the corruption type/level (e.g., Cross-Entropy if categorical, MSE if continuous). Calculated between $\hat{c}_m$ and $c_m$ for corrupted modalities.
    *   BNN Regularization Loss $L_{reg}$: If using VI, this includes the KL divergence term. If using concrete dropout or other BNN techniques, specific regularization terms may apply. For simple MC Dropout, this term might be zero or implicitly handled by weight decay.
    *   Total Loss: $L_{total} = L_{primary} + \beta L_{aux} + \gamma L_{reg}$, where $\beta$ and $\gamma$ are hyperparameters balancing the loss components.
5.  **Backward Pass and Optimization:** Compute gradients of $L_{total}$ w.r.t. all trainable parameters and update using an optimizer like Adam [11] or AdamW [12].
6.  **Inference:** During inference, corruption is not applied synthetically. Input data $X_m$ is processed through encoders $E_m$ (with $T$ passes for MC Dropout) to get $f_m$ and $u_m$. Attention weights $\alpha_m$ are computed based on reliability. The final prediction $\hat{y}$ is generated. The uncertainty $u_m$ and attention weights $\alpha_m$ are available as outputs for interpretability and confidence assessment.

### 3.4 Experimental Design and Validation

We will conduct a comprehensive evaluation to assess the performance and trustworthiness of DRAM-Net.

1.  **Baselines:** We will compare DRAM-Net against:
    *   **Single-modality models:** Models trained on each modality independently.
    *   **Simple Fusion:** Early fusion (concatenation of raw inputs, if feasible), late fusion (averaging predictions from single-modality models), intermediate fusion (concatenation of learned features before prediction head).
    *   **State-of-the-Art (SOTA) Multi-modal Fusion Methods:** Implementations or results from relevant papers identified in the literature review, such as attention-based fusion (e.g., similar to MDA [1] or DRIFA-Net [2] attention mechanisms but without dynamic reliability), and methods designed for missing data (e.g., HEALNet [3] principles).
    *   **Ablation Study:** Variants of DRAM-Net without the BNN component (using standard encoders), without the reliability-guided attention (e.g., standard attention), or without the self-supervised auxiliary task, to assess the contribution of each component.

2.  **Evaluation Scenarios:**
    *   **Standard Performance:** Evaluate on clean, complete datasets.
    *   **Robustness to Missing Data:** Systematically remove 10%, 30%, 50%, 70% of samples for one or more modalities and evaluate performance degradation.
    *   **Robustness to Noise/Corruption:** Introduce varying levels of synthetic noise/artifacts (as described in Sec 3.2) to one or more modalities and measure performance impact.
    *   **Robustness to Domain Shift:** If possible using data splits (e.g., different hospitals in MIMIC), evaluate performance on out-of-distribution data.

3.  **Evaluation Metrics:**
    *   **Task Performance:** Standard metrics relevant to the primary task (e.g., Accuracy, AUC-ROC, Precision, Recall, F1-score for classification; Mean Absolute Error, $R^2$ for regression; Dice Score, IoU for segmentation; Concordance Index for survival analysis).
    *   **Robustness:** Measure the relative drop in performance metrics under the different corruption/missingness scenarios compared to performance on clean data. Lower drop indicates higher robustness.
    *   **Uncertainty Quantification:**
        *   *Calibration:* Expected Calibration Error (ECE) to measure if the model's confidence scores reflect its actual accuracy.
        *   *Uncertainty-Error Correlation:* Correlation between the model's predictive uncertainty (derived from BNN outputs or variance across MC samples) and the prediction error. Higher correlation is desirable.
        *   *Selective Prediction:* Accuracy/AUC when rejecting predictions with uncertainty above a certain threshold (Risk-Coverage curves).
    *   **Interpretability:**
        *   *Attention Map Analysis:* Qualitatively analyze the attention weights $\alpha_m$. Verify if the model assigns lower weights to corrupted/missing/irrelevant modalities and higher weights to clean, informative ones. Compare attention maps between correct and incorrect predictions.
        *   *Modality Uncertainty Analysis:* Analyze the modality-specific uncertainty estimates $u_m$. Check if $u_m$ increases when modality $m$ is corrupted or noisy.

## 4. Expected Outcomes & Impact

### 4.1 Expected Outcomes

This research project is expected to deliver the following outcomes:

1.  **A Novel Multi-modal Fusion Framework (DRAM-Net):** The primary outcome will be the development and implementation of the DRAM-Net architecture, capable of dynamically estimating modality reliability via BNN uncertainty and leveraging this for robust fusion through reliability-guided attention, trained with a self-supervised corruption prediction task.
2.  **Demonstrated Robustness:** Empirical results showcasing DRAM-Net's superior robustness compared to baseline and SOTA methods when faced with missing modalities, noise, and other data corruptions commonly encountered in clinical practice. We expect to quantify performance degradation under these scenarios, showing DRAM-Net maintains higher accuracy/performance.
3.  **Effective Uncertainty Quantification:** Validation of the framework's ability to produce meaningful uncertainty estimates at both the modality level ($u_m$) and the final prediction level. This includes demonstrating good calibration (low ECE) and a strong correlation between uncertainty and prediction error.
4.  **Improved Interpretability:** Qualitative and potentially quantitative evidence showing that the modality uncertainties ($u_m$) and attention weights ($\alpha_m$) provide useful insights into the model's reasoning, highlighting trusted data sources and reflecting data quality issues.
5.  **Open Source Code and Models:** Release of the codebase for DRAM-Net and potentially pre-trained models (subject to data usage agreements) to facilitate reproducibility and further research by the community.
6.  **Publications:** Dissemination of the findings through publications in high-impact machine learning and medical imaging conferences (e.g., NeurIPS, ICML, MICCAI, IPMI) and potentially journals. Presentation at the Trustworthy Machine Learning for Healthcare Workshop would be an initial venue.

### 4.2 Impact

The successful completion of this research project is anticipated to have significant impact:

1.  **Clinical Impact:** By improving the reliability and robustness of multi-modal ML models, this work can pave the way for safer clinical decision support systems. Clinicians may gain more trust in AI tools that can explicitly indicate when their predictions are based on potentially unreliable data or exhibit high uncertainty. This could lead to more appropriate use of AI, improving diagnostic accuracy and patient management, especially in complex cases requiring integration of diverse data.
2.  **Scientific Impact:** This research contributes a novel methodology to the fields of multi-modal learning, trustworthy AI, and Bayesian deep learning. The concept of dynamically estimating and utilizing modality reliability based on data-driven uncertainty offers a new perspective compared to existing fusion techniques. It addresses key challenges identified in recent literature [1-4] concerning noise, missing data, and uncertainty. The proposed framework and the evaluation methodology could serve as a benchmark for future research on reliability-aware fusion.
3.  **Technological Impact:** The development of DRAM-Net provides a practical tool for researchers and practitioners working with multi-modal medical data. The techniques developed could potentially be adapted for other domains beyond healthcare where fusion of potentially unreliable sensor data is necessary (e.g., autonomous driving, robotics).
4.  **Alignment with Workshop Goals:** This research directly tackles several core topics of the Trustworthy Machine Learning for Healthcare Workshop, including generalization to out-of-distribution samples (via robustness), explainability (via attention and uncertainty), uncertainty estimation, debiasing from shortcuts (by focusing on reliable information), and multi-modal fusion. It aims to enhance the trust and confidence of stakeholders in medical AI, fostering its responsible adoption.

In conclusion, this research proposes a significant step towards building more trustworthy multi-modal AI systems for healthcare by explicitly addressing the challenge of variable modality reliability. Through a combination of Bayesian uncertainty estimation, reliability-guided attention, and self-supervised learning, we aim to create models that are not only accurate but also robust, transparent, and aware of their own limitations.

## References

[1] Fan, L., Ou, Y., Zheng, C., Dai, P., Kamishima, T., Ikebe, M., Suzuki, K., & Gong, X. (2024). *MDA: An Interpretable and Scalable Multi-Modal Fusion under Missing Modalities and Intrinsic Noise Conditions*. arXiv:2406.10569.

[2] Dhar, J., Zaidi, N., Haghighat, M., Goyal, P., Roy, S., Alavi, A., & Kumar, V. (2024). *Multimodal Fusion Learning with Dual Attention for Medical Imaging*. arXiv:2412.01248. (Note: Year adjusted based on likely publication cycle if arXiv date is Dec).

[3] Hemker, K., Simidjievski, N., & Jamnik, M. (2023). *HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data*. arXiv:2311.09115.

[4] Yao, W., Yin, K., Cheung, W. K., Liu, J., & Qin, J. (2024). *DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency*. arXiv:2403.06197.

[5] Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. Proceedings of the 33rd International Conference on Machine Learning (ICML).

[6] Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). *Weight Uncertainty in Neural Networks*. Proceedings of the 32nd International Conference on Machine Learning (ICML).

[7] Johnson, A. E. W., Bulgarelli, L., Pollard, T. J., Horng, S., Celi, L. A., & Mark, R. G. (2023). *MIMIC-IV* (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67.

[8] Johnson, A. E. W., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R., Lungren, M. P., Deng, C.-Y., Mark, R. G., & Horng, S. (2019). *MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports*. Scientific Data, 6(1), 317. https://doi.org/10.1038/s41597-019-0322-0.

[9] Cancer Genome Atlas Research Network (various years). Comprehensive genomic characterization defines human tumor types. *Nature* & other journals. (Data accessible via NIH National Cancer Institute Genomic Data Commons).

[10] Menze, B. H., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024.

[11] Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.

[12] Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. arXiv:1711.05101.