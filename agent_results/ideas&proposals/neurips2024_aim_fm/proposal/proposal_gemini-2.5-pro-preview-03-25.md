Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

## Research Proposal

**1. Title:** **Causal Reasoning Meets Explainable Medical Foundation Models**

**2. Introduction**

**Background:**
Foundation Models (FMs) have demonstrated remarkable capabilities across various domains, including natural language processing, computer vision, and audio understanding (Bommasani et al., 2021). Their potential to revolutionize industries is immense, with healthcare being a particularly promising, yet challenging, area. Medical Foundation Models (MFMs) offer the possibility of creating powerful AI-driven medical assistants capable of aiding in diagnosis, prognosis, treatment planning, and streamlining clinical workflows (Moor et al., 2023). This potential is especially critical given the global challenges in healthcare, such as high costs, overburdened medical professionals, and significant disparities in access to care, particularly in underserved regions (WHO, 2021). Developing effective, affordable, and professional MFMs could democratize access to medical expertise and significantly improve patient outcomes.

However, the adoption of FMs in high-stakes domains like healthcare faces significant hurdles. A primary concern is the inherent "black-box" nature of these large, complex models (Adadi & Berrada, 2018). Clinicians require transparency and interpretability to trust AI-driven recommendations, especially when patient well-being is at stake. Current explainability methods often rely on correlational techniques, such as attention maps or feature importance scores (e.g., SHAP, LIME), which highlight associations but may fail to capture the underlying causal mechanisms driving a prediction (Molnar, 2020). These associative explanations can be misleading or fragile, particularly when facing covariate shifts or confounding variables common in real-world medical data (Castro et al., 2020). Misinterpreting a correlation as causation can lead to flawed clinical decisions.

Furthermore, regulatory bodies are increasingly mandating transparency and robustness for AI systems deployed in critical sectors. For instance, the EU AI Act emphasizes the need for explainable and trustworthy AI, particularly in high-risk applications like medical diagnostics (European Commission, 2021). Addressing the explainability gap by moving beyond correlation towards causation is therefore not only a technical challenge but also a prerequisite for ethical deployment, regulatory compliance, and fostering trust between AI systems, clinicians, and patients. Recent work has begun exploring causal perspectives in deep learning and healthcare (Cheng et al., 2025; Zhang et al., 2023; Carloni, 2025; Shetty & Jordan, 2025), highlighting the timeliness and importance of integrating causal reasoning into the next generation of MFMs.

**Research Objectives:**
This research aims to bridge the gap between the capabilities of MFMs and the clinical need for trustworthy, interpretable AI by integrating causal reasoning principles directly into their framework. The primary objectives are:

1.  **Develop the Causal-MFM Framework:** To design and implement a novel framework, Causal-MFM, that explicitly integrates causal discovery and reasoning mechanisms within or alongside a large-scale multimodal MFM.
2.  **Implement Multimodal Causal Discovery:** To adapt and develop causal discovery algorithms capable of learning plausible causal structures (e.g., causal graphs) from heterogeneous medical data sources, including medical images (e.g., X-rays, CT scans), electronic health records (EHRs), clinical notes, and potentially biosignals, incorporating domain knowledge where available.
3.  **Design a Causal Explanation Module:** To create a module that leverages the learned causal structures to generate explanations that are not only faithful to the model's reasoning but also reflect underlying causal pathways. These explanations should be actionable and interpretable by clinicians (e.g., "Condition Y is predicted *because* biomarker X, known to causally influence Y, is elevated, as evidenced by feature Z in the image"). This includes generating counterfactual explanations answering "what-if" questions relevant to clinical interventions.
4.  **Evaluate Explainability, Robustness, and Utility:** To rigorously evaluate the Causal-MFM framework on benchmark medical tasks (e.g., radiology report generation, disease diagnosis/prognosis from EHRs). Evaluation will focus on the quality of explanations (faithfulness, clarity, plausibility assessed via clinician feedback), model robustness under data shifts, and overall task performance compared to baseline MFMs and standard explainability techniques.

**Significance:**
This research holds significant potential to advance the field of medical AI. By grounding MFM explanations in causal reasoning, we aim to:

*   **Enhance Trust and Adoption:** Provide clinicians with transparent, reliable, and actionable explanations, fostering greater trust in MFM predictions and facilitating their integration into clinical workflows.
*   **Improve Robustness:** Causal models are hypothesized to be more robust to distributional shifts than purely correlational models, potentially leading to more reliable MFM performance across different patient populations and clinical settings (Peters et al., 2017).
*   **Facilitate Scientific Discovery:** The causal structures learned from data could potentially offer new insights into disease mechanisms and treatment effects, complementing traditional medical research.
*   **Support Regulatory Compliance:** Develop MFMs that align better with emerging regulatory requirements for transparency and interpretability in high-risk AI systems.
*   **Advance Explainable AI (XAI):** Contribute novel methods for integrating causal reasoning into large foundation models, applicable beyond the medical domain.
*   **Promote Equitable Healthcare:** By improving the reliability and trustworthiness of MFMs, this work can contribute to the development of accessible and dependable AI-driven healthcare tools.

**3. Methodology**

**Overall Framework: Causal-MFM**
The proposed Causal-MFM framework will consist of three core components integrated around a base MFM (likely a Transformer-based architecture adapted for multimodal medical data): (1) A Multimodal Causal Discovery module, (2) The core MFM for prediction/generation tasks, and (3) A Causal Explanation Module. The integration aims to allow the causal knowledge to inform both the model's predictions (potentially through regularization or attention mechanisms) and the generation of explanations.

**Data Collection and Preparation:**
We will utilize publicly available, large-scale multimodal medical datasets such as MIMIC-IV (Johnson et al., 2023) (containing EHRs, clinical notes, waveforms) and CheXpert (Irvin et al., 2019) or MIMIC-CXR (Johnson et al., 2019) (containing chest X-rays and reports). If feasible through collaboration, we will seek access to curated institutional datasets for richer multimodality (e.g., including pathology or genomics data).

Data preprocessing will be crucial. This involves:
*   **Handling Missing Data:** Employing sophisticated imputation techniques suitable for medical data (e.g., multiple imputation, deep learning-based imputation) while being mindful of potential biases introduced.
*   **Feature Engineering:** Extracting relevant features from structured EHR data (labs, vitals), unstructured text (clinical notes, reports) using NLP techniques (e.g., medical entity recognition, embedding representations from models like BioBERT or ClinicalBERT), and medical images (using pre-trained CNNs or Vision Transformers).
*   **Normalization and Standardization:** Applying appropriate scaling techniques for numerical features.
*   **Addressing Data Scarcity/Imbalance:** Utilizing techniques like transfer learning, data augmentation (including potentially generative models for synthetic data – linking to workshop themes), and appropriate sampling strategies during training.

**Component 1: Multimodal Causal Discovery Module**
This module aims to learn a causal graph $G = (V, E)$, where $V$ represents a set of relevant medical variables (e.g., symptoms, diagnoses, lab results, image features, treatments, outcomes) and $E$ represents directed causal relationships between them.

*   **Algorithm Selection:** We will explore and adapt state-of-the-art causal discovery algorithms suitable for mixed data types (continuous, discrete, textual, image-based). Potential candidates include:
    *   Constraint-based methods (e.g., PC algorithm, FCI) adapted for observational medical data, potentially incorporating conditional independence tests suitable for mixed data types.
    *   Score-based methods (e.g., GES) that optimize a scoring function (like BIC) over the space of DAGs.
    *   Methods based on Structural Equation Models (SEMs) or Structural Causal Models (SCMs), including recent deep learning approaches like NOTEARS (Zheng et al., 2018) or gradient-based methods that can handle non-linear relationships.
    *   Methods specifically designed for integrating multiple data modalities or leveraging unlabeled data, potentially drawing inspiration from CInA (Zhang et al., 2023).
*   **Incorporating Domain Knowledge:** A key aspect will be integrating prior medical knowledge (e.g., known physiological pathways, established diagnostic criteria) as constraints to guide the discovery process, improving the plausibility and accuracy of the learned graph. This can be implemented by restricting the search space for edges or fixing certain known relationships.
*   **Handling Latent Confounders:** We will investigate techniques to account for unobserved confounding variables, which are prevalent in medical data, potentially using methods like FCI or leveraging instrumental variables if identifiable.
*   **Output:** The module will output a representation of the learned causal structure (e.g., an adjacency matrix for the graph, potentially with confidence scores for edges), which will inform the other components.

**Component 2: Core Medical Foundation Model**
We will leverage a pre-trained foundation model (e.g., a multimodal transformer architecture) and fine-tune it for specific medical tasks. The integration with the causal discovery module may occur in several ways:

*   **Causal Regularization:** Using the learned causal graph $G$ to regularize the MFM's learning process, encouraging the model to learn relationships consistent with the causal structure. For example, penalizing attention weights or feature contributions that contradict the graph.
*   **Causal Attention Mechanisms:** Modifying the attention mechanism within the transformer to prioritize information flow along paths identified in the causal graph, potentially inspired by CInA (Zhang et al., 2023).
*   **Graph-based Input Representations:** Incorporating graph neural networks (GNNs) operating on the learned causal graph $G$ to provide structured causal information as additional input to the MFM.

We will employ parameter-efficient fine-tuning (PEFT) techniques (e.g., LoRA, Adapters) to adapt the large MFM to specific tasks and datasets efficiently, addressing potential resource constraints (workshop theme).

**Component 3: Causal Explanation Module**
This module generates explanations grounded in the learned causal graph $G$ and the MFM's prediction for a specific instance.

*   **Causal Path Identification:** Tracing the paths in $G$ that connect relevant input features (e.g., identified symptoms, specific image regions linked to features) to the model's output (e.g., predicted diagnosis). The explanation would highlight these pathways.
*   **Counterfactual Explanation Generation:** Generating contrastive explanations by simulating interventions on the causal graph using techniques derived from SCMs and do-calculus (Pearl, 2009). For instance, "The diagnosis would likely change from D1 to D2 if Lab Value L was below threshold T, because L has a direct causal effect on the mechanism leading to D1." Formally, we estimate quantities like $P(Y | do(X=x'))$ compared to $P(Y | do(X=x))$. This requires estimating the underlying functional relationships in the causal model, potentially learned alongside the MFM. We can draw inspiration from methods like CausaLM (Shetty & Jordan, 2025) but extended to multimodal inputs.
*   **Textual Justification:** Synthesizing the identified causal paths and counterfactuals into natural language explanations understandable by clinicians, potentially using template-based or NLG techniques fine-tuned for causal explanations. Example: "Diagnosis X is likely due to [Symptom A] and [Image Finding B]. The causal graph indicates A directly influences X, while B influences X via intermediate factor C. If [Treatment T] were applied, which causally counteracts C, the probability of X would decrease."

**Experimental Design and Validation:**

*   **Tasks & Datasets:**
    *   *Task 1: Radiology Report Generation & Finding Prediction:* Using MIMIC-CXR or CheXpert. Predict findings (e.g., Cardiomegaly, Pneumonia) and generate corresponding reports. Evaluate prediction accuracy (AUC, F1) and report quality (ROUGE, BLEU, clinical accuracy).
    *   *Task 2: EHR-based Prognosis/Diagnosis:* Using MIMIC-IV. Predict outcomes like in-hospital mortality, length of stay, or specific diagnoses based on structured EHR data and clinical notes. Evaluate using AUC, F1, Accuracy.
*   **Baselines:**
    *   Standard MFM (e.g., fine-tuned multimodal transformer) without causal integration.
    *   MFM + Standard XAI: Applying post-hoc methods like SHAP, LIME, or Attention Visualization to the baseline MFM.
*   **Evaluation Metrics:**
    *   *Model Performance:* Standard task-specific metrics (AUC, F1, Accuracy, ROUGE, BLEU, etc.). We aim to demonstrate that incorporating causality maintains or minimally impacts predictive/generative performance.
    *   *Explanation Quality (Primary Focus):*
        *   *Faithfulness:* Assess how well explanations reflect the model's reasoning.
            *   *Feature Ablation:* Measure the change in model prediction probability when causally identified important features are perturbed or removed, compared to features identified by baseline XAI methods (inspired by Hooker et al., 2019).
            *   *Counterfactual Accuracy:* If ground truth or simulation allows, assess if the predicted outcomes under counterfactual scenarios match expectations.
            *   *Correlation with Internal Mechanics:* Analyze correlation between explanation scores and internal model representations (e.g., attention weights, gradients) along identified causal paths.
        *   *Plausibility & Clarity (Human Evaluation):* Engage board-certified clinicians (e.g., radiologists, internists relevant to the task) to rate explanations on Likert scales for:
            *   Clinical Plausibility: Does the explanation make sense based on medical knowledge?
            *   Clarity: Is the explanation easy to understand?
            *   Completeness: Does it capture the key driving factors?
            *   Actionability: Does it provide information that could inform subsequent decisions? (e.g., suggesting relevant follow-up tests based on causal links).
        *   *Robustness:* Evaluate model performance and explanation consistency when tested on out-of-distribution (OOD) data (e.g., data from a different hospital, demographic subgroup, or simulation with covariate shifts). Compare the stability of Causal-MFM explanations versus baseline XAI methods under these shifts.
*   **Clinical Collaboration:** We will collaborate closely with clinicians throughout the project: for incorporating domain knowledge into causal discovery, designing evaluation protocols for explanation quality, and interpreting the results in a clinical context. This addresses the critical challenge of clinical validation (Literature Challenge 5).

**Addressing Challenges:**
*   **Data Quality:** Utilize robust preprocessing and imputation; acknowledge limitations in causal discovery due to potential hidden bias or noise.
*   **Causal Complexity:** Start with identifiable sub-problems, leverage domain constraints, and focus on local causal structures relevant to specific predictions rather than a complete global graph initially.
*   **Interpretability vs. Performance:** Explicitly measure both and analyze the trade-off. Aim for methods that minimally impact performance while significantly boosting interpretabiliy and robustness.
*   **Generalizability:** Evaluate on diverse data splits and OOD settings. Explicitly test fairness across demographic groups.
*   **Clinical Validation:** Integrate clinician feedback iteratively throughout the development and evaluation process.

**4. Expected Outcomes & Impact**

**Expected Outcomes:**

1.  **A Novel Causal-MFM Framework:** A publicly available (code and potentially model weights) framework demonstrating the integration of causal discovery and reasoning with multimodal medical foundation models.
2.  **New Causal Explanation Methods:** Specific algorithms for generating faithful, plausible, and actionable causal explanations (path-based and counterfactual) from MFMs for medical tasks.
3.  **Benchmark Results:** Quantitative and qualitative evidence demonstrating the advantages of Causal-MFM over standard MFMs and baseline XAI methods in terms of explanation quality (faithfulness, clarity, plausibility), robustness to data shifts, potentially without significant loss in predictive performance.
4.  **Validated Causal Structures:** Insights into potential causal relationships within the studied medical domains (e.g., radiology findings, EHR variables related to outcomes), validated partially through domain knowledge and clinician feedback.
5.  **Guidelines for Trustworthy MFM Development:** Contribution towards best practices for building and evaluating explainable and robust MFMs for healthcare.

**Impact:**

*   **Scientific:** This research will push the boundaries of XAI by moving beyond correlational methods towards causality within large foundation models, particularly in the critical domain of healthcare. It will contribute to both the machine learning community (causal inference, foundation models, XAI) and the medical informatics community (clinical decision support, AI safety).
*   **Clinical:** By improving the transparency, trustworthiness, and robustness of MFMs, this work aims to pave the way for safer and more effective integration of AI into clinical practice. Clinicians equipped with reliable, causal explanations can make more informed decisions, potentially leading to improved diagnostic accuracy, better treatment planning, and enhanced patient outcomes.
*   **Societal & Regulatory:** The development of provably more interpretable and robust medical AI systems directly addresses societal concerns about the safety and reliability of AI in high-stakes applications. It provides a potential pathway to meet stringent regulatory requirements (like the EU AI Act), fostering public trust and enabling responsible innovation in AI for healthcare. Ultimately, this research contributes to the vision of AI augmenting human expertise to tackle complex healthcare challenges and improve global health equity.

---
**References:** (Note: Placeholder for full citations based on literature review and standard works)

*   Adadi, A., & Berrada, M. (2018). Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI). *IEEE Access*, 6, 52138-52160.
*   Bommasani, R., et al. (2021). On the Opportunities and Risks of Foundation Models. *arXiv preprint arXiv:2108.07258*.
*   Carloni, G. (2025). Human-aligned Deep Learning: Explainability, Causality, and Biological Inspiration. *arXiv preprint arXiv:2504.13717*.
*   Castro, D. C., et al. (2020). Causality matters in medical imaging. *Nature Communications*, 11(1), 3673.
*   Cheng, Y., et al. (2025). Causally-informed Deep Learning towards Explainable and Generalizable Outcomes Prediction in Critical Care. *arXiv preprint arXiv:2502.02109*.
*   European Commission. (2021). Proposal for a Regulation laying down harmonised rules on artificial intelligence (Artificial Intelligence Act).
*   Hooker, S., et al. (2019). A Benchmark for Interpretability Methods in Deep Neural Networks. *NeurIPS*.
*   Irvin, J., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI*.
*   Johnson, A. E., et al. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. *Scientific data*, 6(1), 317.
*   Johnson, A. E., et al. (2023). MIMIC-IV (version 2.2). *PhysioNet*.
*   Molnar, C. (2020). *Interpretable Machine Learning*. Leanpub.
*   Moor, M., et al. (2023). Foundation models for generalist medical artificial intelligence. *Nature*, 620(7973), 220-229.
*   Pearl, J. (2009). *Causality*. Cambridge university press.
*   Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT press.
*   Shetty, M., & Jordan, C. (2025). Quantifying Symptom Causality in Clinical Decision Making: An Exploration Using CausaLM. *arXiv preprint arXiv:2503.19394*.
*   WHO. (2021). *Global strategy on human resources for health: Workforce 2030*. World Health Organization.
*   Zhang, J., et al. (2023). Towards Causal Foundation Model: on Duality between Causal Inference and Attention. *arXiv preprint arXiv:2310.00809*.
*   Zheng, X., et al. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. *NeurIPS*.