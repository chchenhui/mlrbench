Title  
Knowledge-Infused Evidential Graph Neural Networks for Interpretable and Uncertainty-Aware Clinical Diagnosis  

1. Introduction  
1.1 Background  
Machine learning (ML) has demonstrated enormous potential in automating clinical prediction tasks such as disease diagnosis, risk stratification, and treatment recommendation. However, most high-performance models in healthcare (e.g., deep neural networks) behave as “black boxes,” offering little insight into the reasoning behind their outputs. In clinical settings, opaque predictions undermine trust, slow regulatory approval, and raise patient-safety concerns. Two interrelated challenges must be addressed to accelerate adoption:  
• Interpretability. Clinicians require models that not only predict accurately but also provide human-comprehensible explanations grounded in established medical knowledge.  
• Uncertainty quantification. Robust diagnostic support must indicate when the model is uncertain—either due to insufficient evidence (aleatoric uncertainty) or lack of prior experience (epistemic uncertainty)—so clinicians can override or order further tests.  

Graph Neural Networks (GNNs) naturally accommodate relational medical knowledge (symptoms, diseases, tests, genes) via knowledge graphs (KGs). Recent advances in evidential deep learning and conformal prediction offer principled frameworks for uncertainty quantification. Yet, no existing approach jointly infuses structured medical knowledge, yields clinically aligned explanations, and provides rigorous uncertainty measures.  

1.2 Research Objectives  
This proposal aims to develop and validate a unified framework—Knowledge-Infused Evidential Graph Neural Networks (Ki-EGNN)—that:  
1. Embeds structured medical knowledge into a GNN architecture to guide message passing and align learned representations with clinical reasoning.  
2. Incorporates an evidential deep learning head within the GNN to quantify both aleatoric and epistemic uncertainties.  
3. Provides post-hoc conformal calibration, guaranteeing pre-specified coverage probabilities for diagnostic predictions.  
4. Extracts attention-based explanations tied to medically meaningful nodes and relations to facilitate clinician interpretation.  

1.3 Significance  
• Trustworthy AI. By combining interpretability and uncertainty quantification, Ki-EGNN can foster clinician confidence and satisfy regulatory requirements for medical AI.  
• Clinical alignment. Explanations grounded in standard medical ontologies (e.g., UMLS, SNOMED CT) bridge the gap between automated inference and human reasoning.  
• Safety. Reliable confidence intervals and out-of-distribution detection reduce the risk of overconfident errors in critical healthcare decisions.  

2. Methodology  
2.1 Data Collection and Preprocessing  
• Knowledge Graph Construction  
  – Sources: UMLS (Unified Medical Language System), SNOMED CT, MEDLINE co-occurrence networks, and disease–symptom test databases.  
  – Nodes: Clinical entities (diseases, symptoms, lab tests, genes, treatments).  
  – Edges: Typed relations (e.g., “symptom_of,” “test_for,” “gene_associated_with,” “treats”).  
  – Cleaning & Integration: Resolve entity synonyms, prune low-confidence edges, and enforce ontology consistency.  

• Patient Data  
  – Electronic Health Records (EHR): Longitudinal billing codes, lab test results, vital signs from MIMIC-IV.  
  – Imaging Features: Pre-extracted representations from chest X-rays or CT scans using a pretrained CNN (e.g., ResNet).  
  – Cohort: Adult ICU patients with primary diagnoses covering at least 10 ICD-10 categories.  

• Mapping Patient Records onto KG  
  – Entity Linking: Map EHR codes and imaging findings to corresponding nodes in the KG (e.g., ICD-10 E11 → “Type 2 diabetes mellitus”).  
  – Subgraph Extraction: For each patient, build a “patient graph” by including all nodes corresponding to observed entities and their k-hop neighbors (k=2), yielding subgraph $G_p=(V_p,E_p)$.  

2.2 Model Architecture  
Our Ki-EGNN model consists of three interdependent components: (1) a graph learning backbone, (2) an evidential deep learning head, and (3) a conformal calibration module.  

2.2.1 Graph Learning Backbone  
We employ a multi-layer Graph Attention Network (GAT) that propagates messages along edges weighted by learned attention scores that reflect clinical importance. Let $h_v^{(0)}\in \mathbb{R}^d$ be the initial embedding of node $v$ (obtained via an MLP on its feature vector). Then for layer $l=1,\dots,L$:  

Inline attention score:  
$$
\alpha_{uv}^{(l)} = \frac{\exp\big(\mathrm{LeakyReLU}\big(a^\top [W h_u^{(l-1)} \,\|\, W h_v^{(l-1)} \,\|\, e_{uv}]\big)\big)}{\sum_{w\in \mathcal{N}(v)} \exp\big(\mathrm{LeakyReLU}\big(a^\top [W h_w^{(l-1)} \,\|\, W h_v^{(l-1)} \,\|\, e_{wv}]\big)\big)},
$$

where  
• $W\in\mathbb{R}^{d'\times d}$ and $a\in\mathbb{R}^{2d'+p}$ are learnable parameters,  
• $e_{uv}\in\mathbb{R}^p$ encodes the edge type between $u$ and $v$,  
• $[\cdot\|\cdot\|\cdot]$ denotes vector concatenation.  

Node update:  
$$
h_v^{(l)} = \sigma\Big( \sum_{u\in \mathcal{N}(v)} \alpha_{uv}^{(l)}\, W h_u^{(l-1)} \Big),
$$

with activation $\sigma(\cdot)$ (e.g., ELU). After $L$ layers, we compute a patient-level representation by pooling over target nodes (e.g., disease nodes) or via a learnable readout:  
$$
h_p = \mathrm{READOUT}\big(\{h_v^{(L)} : v \in \mathcal{T}\}\big),
$$  
where $\mathcal{T}$ is the set of candidate disease nodes and READOUT can be mean, max, or attention-based pooling.  

2.2.2 Evidential Uncertainty Quantification  
Following the evidential deep learning paradigm, we interpret the model’s raw logits as “evidence” for each of $C$ diagnostic classes. Let $z \in \mathbb{R}^C$ be the output of a final MLP on $h_p$. We define nonnegative evidence vector $e = \mathrm{softplus}(z)$ and Dirichlet parameters  
$$
\alpha_i = e_i + 1,\quad i=1,\dots,C.
$$  
The predictive probability for class $i$ is  
$$
p_i = \frac{\alpha_i}{\alpha_0},\quad \alpha_0 = \sum_{j=1}^C \alpha_j.
$$  
Uncertainty is quantified as  
• Total uncertainty: $u_{\mathrm{total}} = \frac{C}{\alpha_0}$.  
• Aleatoric uncertainty: $u_{\mathrm{alea}} = \sum_{i=1}^C \frac{\alpha_i}{\alpha_0}\Big(1 - \frac{\alpha_i}{\alpha_0}\Big)\Big/\alpha_0$.  
• Epistemic uncertainty: $u_{\mathrm{epis}} = u_{\mathrm{total}} - u_{\mathrm{alea}}$.  

Loss function combines evidential classification loss and a KL-divergence prior regularization:  
$$
\mathcal{L}_{\mathrm{EDL}} = \sum_{i=1}^C y_i\big(\psi(\alpha_0) - \psi(\alpha_i)\big) + \lambda\,\mathrm{KL}\big(\mathrm{Dir}(\alpha)\,\|\,\mathrm{Dir}(\mathbf{1})\big),
$$  
where $y_i\in\{0,1\}$ is the one-hot ground truth, $\psi(\cdot)$ is the digamma function, and $\lambda$ controls strength of the regularizer.  

2.2.3 Conformal Calibration  
To guarantee finite-sample coverage, we apply split-conformal prediction on the evidential probabilities. We reserve a calibration set and compute nonconformity scores for each sample  
$$
s_j = 1 - p_{y_j}^{(j)},
$$  
where $p_{y_j}^{(j)}$ is the predicted probability of the true class. At significance level $\alpha$, the quantile $q_{1-\alpha}$ of $\{s_j\}$ yields a prediction set  
$$
\mathcal{C}(x) = \{ i : 1 - p_i(x) \le q_{1-\alpha} \},
$$  
ensuring $\Pr\{y\in \mathcal{C}(x)\} \ge 1-\alpha$.  

2.3 Explanation Mechanism  
Explanations are derived from attention weights $\alpha_{uv}^{(l)}$ aggregated along paths connecting observed symptoms to predicted diseases. For a prediction $\hat y$, we:  
1. Compute contribution score for each node $v$: $c_v = \sum_{l}\sum_{u\in \mathcal{N}(v)} \alpha_{uv}^{(l)}$.  
2. Rank the top-$k$ nodes and edges by $c_v$ and present subgraph glimpses to clinicians.  
3. Map them back to natural language explanations (e.g., “Elevated blood glucose contributed strongly to diabetes mellitus prediction”).  

2.4 Training and Hyperparameter Tuning  
• Split data into training (70%), calibration (10%), validation (10%), and test (10%) sets at the patient level.  
• Optimize $\mathcal{L}_{\mathrm{EDL}}$ plus graph regularization (e.g., Laplacian smoothing) using Adam.  
• Tune hyperparameters ($L$, hidden dimensions, $\lambda$, learning rate) by maximizing validation AUC and minimizing expected calibration error (ECE).  

2.5 Experimental Design  
We will evaluate Ki-EGNN on multiple diagnostic tasks:  
• Type 2 diabetes prediction from EHR+imaging.  
• Sepsis onset detection in ICU.  
• Pneumonia classification from chest X-rays coupled with clinical labs.  

Baselines:  
• Black-box models (ResNet, LSTM).  
• Standard GAT without knowledge infusion.  
• GAT + conformal prediction.  
• GAT + evidential head (no KG).  

We also design out-of-distribution (OOD) tests by withholding rare disease codes and assessing uncertainty responses.  

2.6 Evaluation Metrics  
Predictive performance: accuracy, F1-score, AUC-ROC.  
Calibration: expected calibration error (ECE), negative log-likelihood (NLL), reliability diagrams.  
Conformal coverage: empirical coverage vs. nominal $1-\alpha$.  
Uncertainty quality: separation between correct and incorrect predictions via uncertainty histogram and area under the sparseness–error curve.  
Interpretability:  
• Fidelity (agreement between explanations and model outputs).  
• Sparsity (number of nodes in explanation).  
• Clinician evaluation: human-subject study where physicians rate explanation usefulness on a Likert scale and compare against rule-based reasoning.  

3. Expected Outcomes & Impact  
3.1 Expected Outcomes  
• A unified Ki-EGNN architecture that outperforms baselines in diagnostic accuracy (AUC improvement ≥ 3%), yields well-calibrated uncertainty estimates (ECE ≤ 2%), and produces conformal sets meeting coverage guarantees (≥ 90% at $\alpha=0.1$).  
• Qualitative and quantitative evidence that attention-based explanations align with clinical knowledge (e.g., precision ≥ 0.8 in clinician evaluation).  
• Robust detection of OOD samples: epistemic uncertainty in withheld rare diseases is significantly higher than in-distribution.  
• An open-source implementation and benchmark suite for knowledge-infused, uncertainty-aware GNNs in healthcare.  

3.2 Broader Impact  
• Clinical Adoption: By providing transparent, knowledge-grounded explanations with rigorous uncertainty bounds, Ki-EGNN can accelerate regulatory approval and clinician acceptance of AI-driven diagnostics.  
• Patient Safety: Reliable uncertainty quantification will alert clinicians to ambiguous cases, reducing the risk of overconfident misdiagnoses.  
• Research Community: The proposed calibration and explanation framework can be extended to other domains requiring safe, interpretable ML (e.g., autonomous driving, finance).  
• Ethical AI: Grounding model reasoning in established medical ontologies mitigates hidden biases and fosters equitable healthcare delivery.  

In sum, this research will establish a principled foundation for interpretable, uncertainty-aware clinical decision support systems, bridging the gap between state-of-the-art graph learning and the rigorous demands of medical practice.