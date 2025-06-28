Title  
Causal-MFM: Integrating Causal Reasoning into Explainable Medical Foundation Models  

1. Introduction  
Background  
Recent advances in large‐scale foundation models (FMs) have demonstrated remarkable capabilities in general language understanding, visual recognition, and multimodal reasoning.  However, when applying FMs to healthcare—where decisions can directly affect patient safety—clinicians require transparent, trustworthy, and actionable explanations.  Conventional post‐hoc explainability methods (e.g., saliency maps or attention visualization) often capture superficial correlations rather than true causal mechanisms, leading to unreliable or even misleading interpretations.  Moreover, regulations such as the EU AI Act demand that high‐risk AI systems provide audit‐ready, causally‐grounded justifications for their outputs.  

Literature Gaps  
• Causally‐informed outcome prediction models (Cheng et al., arXiv:2502.02109) have shown improved interpretability in critical care, but are not integrated into general MFMs.  
• Transformer‐based causal learning (Zhang et al., arXiv:2310.00809) highlights duality between attention and causal inference, yet lacks domain‐specific constraints for high‐stakes medical tasks.  
• Dissertations on human‐aligned deep learning (Carloni, arXiv:2504.13717) outline frameworks for causality in vision but do not scale to multimodal clinical data.  

Research Objectives  
1. Design a unified framework, Causal-MFM, that embeds causal discovery and counterfactual reasoning directly into the architecture of large multimodal medical foundation models.  
2. Develop a causal explanation module that generates action‐aware, clinician‐friendly textual and visual justifications.  
3. Validate the framework on diverse clinical tasks (radiology report generation, EHR‐based prognosis) under in‐distribution and out‐of‐distribution settings, measuring both predictive performance and explanation faithfulness.  

Significance  
By aligning MFMs with causal reasoning, Causal-MFM aims to:  
• Bridge the trust gap between AI and healthcare professionals.  
• Comply with emerging AI regulations requiring causal justifications.  
• Improve patient outcomes by enabling clinicians to understand “why” behind AI recommendations, not just “what.”  

2. Methodology  

2.1 Overview  
Causal-MFM consists of three key components:  
1. Multimodal Causal Discovery: Learn a causal graph from heterogeneous data.  
2. Causal-Augmented Foundation Model: Integrate the causal graph into a pretrained multimodal FM using graph neural networks (GNNs).  
3. Causal Explanation Module: Generate counterfactual explanations and natural language justifications.  

2.2 Data Collection & Preprocessing  
Datasets  
• MIMIC-IV (EHR records, demographics, lab results).  
• CheXpert (chest X-rays with labels).  
• Partner hospital data (de‐identified multimodal records: radiology images + structured EHR).  

Preprocessing  
• Missing data imputation via causal‐aware methods (e.g., using observed parents in the causal graph).  
• Normalization per modality (zero‐mean unit‐variance for numeric labs; image resizing/standardization).  
• Domain constraints: encode known clinical relationships (e.g., “AST and ALT elevations imply liver injury”) as prior edges in the causal graph.  

2.3 Multimodal Causal Discovery  
We denote the full feature vector as  
$$x = [x_{\text{img}},\,x_{\text{text}},\,x_{\text{lab}}]$$  
where $x_{\text{img}}$ are image features, $x_{\text{text}}$ are text embeddings, and $x_{\text{lab}}$ are lab‐value vectors.  

Objective: learn a directed acyclic graph (DAG) $G=(V,E)$ over $V=\{x_i\}$ such that edges capture direct causal influence.  

Algorithmic Steps  
1. Initialize graph with domain priors $G_0$.  
2. Estimate conditional independencies via kernel‐based tests on multimodal samples.  
3. Refine graph structure by maximizing a score  
   $$\text{Score}(G) = \ell(D\,|\,G) - \lambda\cdot\text{Complexity}(G),$$  
   where $\ell$ is the log‐likelihood of data under $G$ and $\lambda$ penalizes spurious edges.  
4. Perform constraint‐based search (e.g., PC‐algorithm variant) followed by score‐based refinement (e.g., GES).  

2.4 Causal-Augmented Foundation Model Architecture  
Base encoders  
• $f_{\text{img}}(x_{\text{img}})\in\mathbb{R}^d$ – vision transformer (pretrained).  
• $f_{\text{text}}(x_{\text{text}})\in\mathbb{R}^d$ – BERT‐style text encoder.  
• $f_{\text{lab}}(x_{\text{lab}})\in\mathbb{R}^d$ – feed‐forward network.  

Causal Graph Embedding  
Let $A\in\{0,1\}^{n\times n}$ be the adjacency matrix of learned DAG, $n=\lvert V\rvert$.  We embed causal structure using a GNN:  
$$h^{(0)}_i = \begin{cases} f_{\text{img}}(x_{\text{img},i}) & \text{if } i\in V_{\text{img}},\\ f_{\text{text}}(x_{\text{text},i}) & i\in V_{\text{text}},\\ f_{\text{lab}}(x_{\text{lab},i}) & i\in V_{\text{lab}},\end{cases}$$  
and for layers $\ell=0,\dots,L-1$:  
$$h_i^{(\ell+1)} = \sigma\Bigl(\sum_{j=1}^n A_{ij}\,W^{(\ell)}\,h_j^{(\ell)} + b^{(\ell)}\Bigr),$$  
where $W^{(\ell)},b^{(\ell)}$ are learnable parameters and $\sigma$ is a non‐linear activation (e.g., ReLU).  

Fusion & Prediction  
Concatenate final node embeddings:  
$$H = [h_1^{(L)};\dots;h_n^{(L)}]\in\mathbb{R}^{n\,d}.$$  
Pass through a task‐specific head for classification/regression:  
$$\hat y = \mathrm{Head}(H;\theta_{\text{head}}).$$  

2.5 Causal Explanation Module  
Goal: For a given input $x$ and prediction $\hat y$, generate a set of intervention–effect pairs and natural‐language justification.  

Counterfactual Effect Estimation  
We employ single‐feature interventions: for feature $x_i$, define $x^{(cf)}$ with $x_i$ set to a counterfactual value $x'_i$ (e.g., “normal” lab range), keeping others fixed.  Then  
$$\Delta_i = f(x^{(cf)}) - f(x)\,,$$  
quantifies the causal effect of $x_i$.  In classification we compute log‐odds change:  
$$\Delta_i = \log\frac{P(y=\text{pos}\mid do(x_i=x_i'))}{P(y=\text{pos}\mid do(x_i=x_i))}\,.$$  

Explanation Generation  
1. Select top‐$k$ features $\{i_1,\dots,i_k\}$ with largest $|\Delta_i|$.  
2. For each $i_j$, retrieve clinical concept $C_{i_j}$ (e.g., “elevated troponin”).  
3. Template‐based text generation:  
   “Intervention: If ${C_{i_j}}$ had been within normal range, the risk score would change by ${\Delta_{i_j}}$, suggesting a causal contribution of ${C_{i_j}}$ to the prediction.”  
4. Visual overlays: for imaging features, highlight regions corresponding to $i_j$ (using Grad‐CAM projected through causal attention weights).  

2.6 Training Procedure  
Pseudocode  
```
Input: Multimodal data D = {(x^(n), y^(n))}_n, domain priors G0
1. Learn causal graph G from D and G0 (Sec. 2.3)
2. Initialize encoders f_img, f_text, f_lab (from pretrained FM)
3. Initialize GNN parameters {W^(l),b^(l)} and Head(·;θ_head)
4. For epoch = 1 to N:
     For each minibatch B ⊂ D:
         1. Encode features h^(0) via f_*
         2. Perform GNN message passing to get H
         3. Compute predictions ŷ = Head(H)
         4. Compute loss L_task = CrossEntropy(ŷ,y)
         5. (Optional) Contrastive loss on interventions: 
            L_cf = ∑_i ||f(x) – f(do(x_i))||^2
         6. Total loss L = L_task + α L_cf
         7. Backpropagate and update all parameters
5. Return trained Causal‐MFM
```

2.7 Experimental Design & Evaluation Metrics  
Clinical Tasks  
• Radiology report generation (CheXpert images → structured report).  
• EHR prognosis (MIMIC‐IV lab/time‐series → multi‐label deterioration prediction).  

Baselines  
• Vanilla multimodal FM (no causal module).  
• Post‐hoc LIME/SHAP explanations.  
• Counterfactual explanation only (no graph integration).  

Evaluation Metrics  
1. Predictive performance: accuracy, AUROC, F1‐score for classification; BLEU/METEOR for report generation.  
2. Explanation faithfulness:  
   – Remove‐and‐Retrain (ROAR) test Δ in performance when top‐k explanations are removed.  
   – Ground‐truth causal intervention tests on synthetic validation subsets.  
3. Clinician‐rated relevance & clarity: 5‐point Likert scale over 50 cases with two radiologists/intensivists.  
4. Robustness under covariate shift: measure ΔAUROC when testing on out‐of-distribution site data.  
5. Fairness: demographic parity and equalized odds across age, gender, ethnicity subgroups.  

3. Expected Outcomes & Impact  

3.1 Anticipated Results  
• Predictive performance on par or superior (≤2% AUROC improvement) to strong multimodal baselines.  
• Explanation methods that outperform LIME/SHAP by >15% in ROAR faithfulness tests.  
• Clinician‐rated explanation clarity >4.0/5.0 on average, demonstrating high acceptability.  
• Reduced degradation (<5% relative) under covariate shift versus >10% in baselines.  
• Improved fairness metrics, narrowing group‐wise disparities by ≥10%.  

3.2 Scientific Contributions  
• A generalizable framework (Causal-MFM) that tightly integrates causal inference with multimodal foundation models in healthcare.  
• Novel causal graph embedding via GNNs, enabling end-to-end learning of prediction and explanation.  
• A comprehensive evaluation protocol combining quantitative robustness, faithfulness metrics, and qualitative clinician assessments.  

3.3 Clinical and Societal Impact  
• Enhanced trust and transparency of AI tools, fostering broader adoption in clinical workflows (e.g., decision support in radiology and ICU).  
• Regulatory compliance: generative causal justifications align with audit requirements in high‐risk settings.  
• Reduced diagnostic errors and improved patient outcomes by enabling clinicians to challenge and refine AI recommendations.  
• Democratization of expertise: by making causal drivers explicit, non‐specialist clinicians and health systems in resource‐limited settings can benefit from expert‐level reasoning.  

3.4 Future Directions  
• Extension to real‐time surgical assistance agents with causal planning modules.  
• Federated causal discovery to protect patient privacy across institutions.  
• Incorporation of interventional trial data for better external validity of causal graphs.  

By uniting causal discovery, multimodal representation learning, and counterfactual explanation, Causal-MFM aims to establish a new paradigm for trustworthy, robust, and interpretable medical foundation models—paving the way toward AI systems that clinicians can truly understand, trust, and integrate into patient care.