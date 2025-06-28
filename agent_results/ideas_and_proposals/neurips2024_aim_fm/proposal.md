# Causal Reasoning Meets Explainable Medical Foundation Models  

## Introduction  

### Background  
Medical foundation models (MFMs) have demonstrated transformative potential in healthcare by enabling automated diagnosis, prognosis, and treatment planning. However, their adoption in clinical practice remains limited due to their "black-box" nature, which obscures decision-making pathways critical for high-stakes medical scenarios. Current explainability methods, such as attention maps or gradient-based visualizations, often capture spurious correlations rather than causal mechanisms, leading to unreliable interpretations. For instance, a model might associate a treatment with improved outcomes merely due to confounding factors like patient age, rather than genuine therapeutic effects. This gap undermines trust among clinicians, hinders regulatory compliance (e.g., under the EU AI Act), and limits the scalability of MFMs in precision medicine.  

### Research Objectives  
This proposal introduces **Causal-MFM**, a framework that integrates causal reasoning into MFMs to generate interpretable, action-aware explanations grounded in causal mechanisms. The primary objectives are:  
1. **Causal Discovery**: Learn multimodal causal graphs from heterogeneous medical data (e.g., imaging, electronic health records (EHRs), lab results) using domain-specific constraints.  
2. **Causal Explanation Module**: Embed causal Bayesian networks into the model architecture to produce counterfactual explanations (e.g., "What if drug dosage is increased?").  
3. **Validation**: Evaluate the faithfulness, robustness, and clinical utility of explanations through ablation studies and clinician feedback.  

### Significance  
By shifting from associative to causal explanations, Causal-MFM addresses critical challenges in healthcare AI:  
- **Trust**: Clinicians require causal justifications to validate AI-driven decisions, such as linking abnormal biomarkers to specific interventions.  
- **Robustness**: Causal models are inherently more resilient to covariate shifts (e.g., demographic changes) than correlational approaches.  
- **Regulatory Compliance**: Transparent causal pathways align with audit requirements for high-risk AI systems.  
- **Equity**: Identifying and mitigating causal biases in data (e.g., race-linked treatment disparities) fosters fairer outcomes.  

This work bridges the gap between cutting-edge causal inference research and clinical deployment, advancing the vision of trustworthy, human-aligned medical AI.  

---

## Methodology  

### Data Collection and Preprocessing  
**Datasets**:  
- **Multimodal Medical Data**:  
  - **Imaging**: Chest X-rays (MIMIC-CXR dataset), brain MRIs (BraTS), and corresponding radiology reports.  
  - **EHRs**: Structured lab results, medication histories, and diagnoses from the eICU Collaborative Research Database.  
  - **Sensor Data**: Physiological time-series (e.g., heart rate, blood oxygen) from ICU monitors.  
- **Curation**: Domain experts annotate causal relationships (e.g., "elevated troponin → myocardial infarction") to guide causal discovery.  

**Preprocessing**:  
- **Normalization**: Standardize imaging (e.g., lung segmentation in X-rays) and sensor data (e.g., resampling time-series).  
- **Missing Data**: Impute missing EHR values using multiple imputation by chained equations (MICE) or causal graph-aware techniques.  
- **Modality Alignment**: Use contrastive learning to align heterogeneous representations (e.g., text-image pairs in radiology reports).  

---

### Algorithmic Framework  

#### 1. Causal Discovery  
We learn causal graphs $ \mathcal{G} = (\mathbf{V}, \mathbf{E}) $, where nodes $ \mathbf{V} $ represent medical variables (e.g., symptoms, treatments) and edges $ \mathbf{E} $ denote causal relationships.  

**Steps**:  
- **Constraint-Based Learning**: Apply the PC algorithm with domain-specific constraints (e.g., temporal order of lab tests) to identify causal directionality.  
- **Neural Causal Discovery**: Train a variational autoencoder (VAE) to infer latent causal factors from high-dimensional imaging data.  
- **Counterfactual Augmentation**: Generate synthetic counterfactuals (e.g., "What if blood pressure were lower?") using structural causal models (SCMs):  
  $$
  Y = f(\text{do}(X), U),
  $$  
  where $ Y $ is an outcome, $ \text{do}(X) $ represents an intervention on variable $ X $, and $ U $ denotes unobserved confounders.  

#### 2. Causal Explanation Module  
We integrate causal graphs into a transformer-based MFM architecture (e.g., Med-PaLM) to generate causal explanations.  

**Design**:  
- **Causal Attention**: Modify self-attention to prioritize causally relevant tokens. For input features $ \mathbf{X} = \{x_1, ..., x_n\} $, compute attention weights $ \alpha_i $ using causal effect estimates:  
  $$
  \alpha_i = \text{softmax}(QK^T / \sqrt{d_k}) \cdot \mathcal{C}(x_i),
  $$  
  where $ \mathcal{C}(x_i) $ quantifies the causal strength of $ x_i $ on the outcome.  
- **Counterfactual Explanation Generator**: For a prediction $ \hat{Y} $, generate explanations via:  
  $$
  \text{Explain}(\hat{Y}) = \left\{x_i \in \mathbf{X} \mid \mathbb{E}[Y | \text{do}(x_i)] \neq \mathbb{E}[Y]\right\}.
  $$  

#### 3. Causal Bayesian Networks (CBNs)  
Embed CBNs into the model to formalize probabilistic causal relationships:  
$$
P(\mathbf{V}) = \prod_{i=1}^n P(V_i | \text{Pa}(V_i)),
$$  
where $ \text{Pa}(V_i) $ are the parents of node $ V_i $ in the causal graph. This enables interventions (e.g., simulating drug effects) and counterfactual queries.  

---

### Experimental Design  

#### 1. Baselines  
- **Associative Models**: Standard MFMs (e.g., BioViL, Med-PaLM) with attention-based explanations.  
- **Causal Competitors**: CInA (Zhang et al., 2023), CausaLM (Shetty et al., 2025).  

#### 2. Tasks  
- **Radiology Report Generation**: Predict diagnoses from X-rays and generate causal explanations (e.g., "Pleural effusion causes shortness of breath").  
- **EHR Prognosis**: Forecast 30-day readmission risk using causal pathways (e.g., "Uncontrolled diabetes → kidney failure").  

#### 3. Evaluation Metrics  
- **Quantitative**:  
  - **Faithfulness**: Measure sensitivity to input perturbations (e.g., ablation tests).  
  - **Robustness**: Evaluate performance under covariate shifts (e.g., new hospitals).  
  - **Accuracy**: AUC-ROC, F1-score.  
- **Qualitative**:  
  - **Clinician Feedback**: Survey 20+ clinicians on explanation clarity (Likert scale 1–5) and clinical relevance.  
  - **Counterfactual Validity**: Assess plausibility of generated counterfactuals via expert review.  

#### 4. Ablation Studies  
- **Causal Graph Quality**: Compare performance using ground-truth vs. learned graphs.  
- **Modality Contribution**: Test unimodal (e.g., imaging-only) vs. multimodal inputs.  

---

## Expected Outcomes & Impact  

### Expected Outcomes  
1. **Benchmark Improvements**:  
   - Achieve ≥15% higher clinician satisfaction scores on explanation clarity compared to associative baselines.  
   - Demonstrate ≥10% improvement in robustness metrics (e.g., AUC drop under covariate shift).  
2. **Technical Innovations**:  
   - Novel causal attention mechanisms that outperform standard explainability methods in faithfulness tests.  
   - Publicly release Causal-MFM code and causal graph datasets for reproducibility.  

### Scientific and Clinical Impact  
1. **Trust in AI**: Causal explanations will align with clinical reasoning, fostering adoption in high-risk settings (e.g., ICU decision-making).  
2. **Regulatory Alignment**: Provide audit-ready causal pathways to meet EU AI Act requirements for transparency.  
3. **Equity Advancements**: Identify and mitigate causal biases (e.g., race-linked treatment gaps) through interpretable graphs.  
4. **Foundation for Future Work**: Establish causal reasoning as a core component of MFMs, enabling applications in drug discovery and personalized treatment.  

### Long-Term Vision  
Causal-MFM aims to redefine medical AI by transforming opaque predictions into actionable, causally grounded insights. This work will catalyze collaborations between AI researchers and clinicians, accelerating the deployment of safe, equitable, and human-centric healthcare systems.  

--- 

This proposal directly addresses the challenges outlined in the literature review:  
- **Data Quality**: Domain-informed causal discovery mitigates biases in EHRs.  
- **Complexity**: Hybrid constraint-based/neural methods handle high-dimensional data.  
- **Interpretability-Performance Trade-off**: Empirical validation ensures causal modules do not harm accuracy.  
- **Clinical Adoption**: Clinician-in-the-loop evaluation ensures practical utility.  

By uniting causal inference and foundation models, Causal-MFM paves the way for the next generation of trustworthy medical AI.