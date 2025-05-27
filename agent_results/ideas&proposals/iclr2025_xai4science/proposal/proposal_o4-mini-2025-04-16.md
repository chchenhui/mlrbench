Title  
Knowledge-Guided Self-Explainable Graph Neural Models for Biomedical Discovery

1. Introduction  
1.1 Background  
Machine learning has revolutionized biomedical research by enabling high-dimensional data analysis and predictive modeling of patient outcomes, drug responses, and disease subtypes. However, most high-capacity models (e.g., deep neural networks) act as black boxes, offering little insight into the biological mechanisms they exploit. In healthcare and life sciences, interpretability is not a luxury but a requirement: clinicians and biologists must trust model recommendations and understand underlying drivers before acting on them. Post-hoc explanation methods (e.g., saliency maps, LIME, SHAP) often lack fidelity, can be misleading, and do not guarantee consistency with domain knowledge. Conversely, purely interpretable models may sacrifice performance or fail to capture complex nonlinear interactions inherent in biological systems.

The “XAI4Science” workshop emphasizes the integration of interpretability and explainability (XAI) methods with scientific discovery across domains such as healthcare, climate, and materials science. In the biomedical arena, integrating structured domain knowledge (e.g., gene–gene interaction networks, pathway databases, pharmacokinetic ontologies) into model architectures promises both high predictive power and transparent, mechanistically meaningful outputs.  

1.2 Research Objectives  
This proposal develops and evaluates **Knowledge-Guided Self-Explainable Models** that:  
• Seamlessly embed biomedical ontologies into graph neural network (GNN) and additive model architectures.  
• Produce end-to-end interpretable representations aligned with known biological entities (genes, pathways, drugs).  
• Simultaneously optimize for predictive performance (e.g., survival time, drug synergy) and explanation quality.  
• Discover novel biological insights—such as disease subtypes, biomarkers, and therapeutic targets—that are validated in silico and, where feasible, confirmed in vitro.  

1.3 Significance  
Our approach bridges the gap between black-box ML and mechanistic biology by constructing models whose internal modules correspond to interpretable biological entities. This fosters trust and accelerates translational research, enabling:  
• **Actionable Insights.** Identification of biomarkers and drug targets that directly map to biological pathways.  
• **Precision Medicine.** Subpopulation-specific mechanistic explanations for treatment response.  
• **Responsible AI Adoption.** Transparent models that clinicians and researchers can audit, critique, and improve.  
• **Generalizable Framework.** A blueprint for other scientific domains requiring interpretable, knowledge-informed ML models.

2. Methodology  
2.1 Overview  
Our methodology consists of four stages: (1) data curation and graph construction; (2) design of self-explainable GNN/additive architectures integrating ontology modules; (3) joint optimization of predictive and interpretability objectives; (4) rigorous experimental evaluation—including wet-lab validation for selected findings.

2.2 Data Collection and Preprocessing  
• **Patient Cohorts & Omics Data.** We will use The Cancer Genome Atlas (TCGA) for gene-expression, mutation, copy-number, and clinical outcomes (survival times, treatment regimens).  
• **Drug Response Data.** Cell line sensitivity (e.g., Cancer Therapeutics Response Portal), drug synergy screens (e.g., NCI-ALMANAC), and patient response registries.  
• **Biomedical Ontologies & Networks.**  
  – Gene interaction graphs from STRING or BioGRID (nodes = genes/proteins, edges = physical or functional interactions).  
  – Pathway databases (KEGG, Reactome) structured as hypergraphs or factor graphs.  
  – Drug–target relationships from DrugBank and PharmGKB.

Preprocessing steps:  
1. Normalize gene-expression to log-TPM and standardize features.  
2. Filter low-variance genes and highly correlated features (retain top 5 000 genes by variance).  
3. Construct a heterogeneous graph \(G=(V,E)\) where  
   – \(V = V_{\text{genes}}\cup V_{\text{pathways}}\cup V_{\text{drugs}}\)  
   – Edges \(E\) include gene–gene interactions, gene–pathway memberships, drug–gene targets.  
4. Encode node features \(X\): gene nodes carry expression/mutation vectors; pathway nodes carry aggregated statistics; drug nodes carry fingerprint embeddings.

2.3 Model Architecture  
We propose a **modular GNN** with interpretable “concept bottleneck” layers and additive outputs. Each module explicitly corresponds to a biological entity or relation.

2.3.1 Graph Neural Encoder  
Let \(H^{(0)}=X\). For each GNN layer \(\ell=0,\dots,L-1\), we compute:  
$$
H^{(\ell+1)} = \sigma\bigl(
   \tilde{D}^{-\frac12}\tilde{A}\,\tilde{D}^{-\frac12}
   H^{(\ell)}W^{(\ell)}
\bigr)
$$  
where \(\tilde{A}=A+I\) is the adjacency with self-loops, \(\tilde{D}\) its degree matrix, and \(W^{(\ell)}\) learnable weights. To capture relation-type importance, we augment with **graph attention**:  
$$
\alpha_{ij}^{(\ell)} = 
\frac
{\exp\!\bigl(\mathrm{LeakyReLU}\bigl(a^{(\ell)\top}[W^{(\ell)}h_i^{(\ell)}\Vert W^{(\ell)}h_j^{(\ell)}]\bigr)\bigr)}
{\sum_{k\in\mathcal{N}(i)}\exp\!\bigl(\mathrm{LeakyReLU}(a^{(\ell)\top}[W^{(\ell)}h_i^{(\ell)}\Vert W^{(\ell)}h_k^{(\ell)}])\bigr)}.
$$  
Message passing then becomes  
$$
h_i^{(\ell+1)}
= \sigma\Bigl(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{(\ell)}W^{(\ell)}h_j^{(\ell)}\Bigr).
$$  
Attention weights \(\alpha_{ij}\) provide instance-specific explanations for edges.

2.3.2 Concept Bottleneck & Additive Explanation Layer  
After \(L\) GNN layers, we obtain final embeddings \(H^{(L)}\). We then group embeddings by module \(m\in\mathcal{M}\) (e.g., gene set, pathway), producing concept scores:  
$$
c_m = g_m\bigl(H_{\mathcal{V}_m}^{(L)}\bigr),
$$  
where \(g_m\) is a small interpretable function (e.g., linear or shallow MLP) operating only on nodes \(\mathcal{V}_m\). The final prediction \(y\) is additive:  
$$
\hat y = f(c_1,\dots,c_{|\mathcal{M}|})
= \sum_{m=1}^{|\mathcal{M}|}w_m\,c_m + b.
$$  
Weights \(w_m\) and biases \(b\) are learned; each \(c_m\) is a transparent concept corresponding to a known biological module.

2.3.3 Loss Functions  
We jointly optimize:  
$$
\mathcal{L} 
= \mathcal{L}_{\mathrm{pred}} 
+ \lambda_1\,\mathcal{L}_{\mathrm{know}}
+ \lambda_2\,\mathcal{L}_{\mathrm{spar}}
+ \lambda_3\,\mathcal{L}_{\mathrm{conc}}.
$$  
• \(\mathcal{L}_{\mathrm{pred}}\): predictive loss (e.g., cross-entropy for classification, Cox partial likelihood for survival analysis):  
$$
\mathcal{L}_{\mathrm{surv}}
= -\sum_{i:\delta_i=1}\Bigl[\hat h_i - \log\sum_{j:t_j\ge t_i}\exp(\hat h_j)\Bigr].
$$  
• \(\mathcal{L}_{\mathrm{know}}\): knowledge-consistency penalty that encourages attention weights \(\alpha_{ij}\) to align with known edge reliabilities \(r_{ij}\):  
$$
\mathcal{L}_{\mathrm{know}}
= \sum_{(i,j)\in E}\bigl|\alpha_{ij} - r_{ij}\bigr|.
$$  
• \(\mathcal{L}_{\mathrm{spar}}\): sparsity penalty on explainable concepts (\(\ell_1\)-norm on \(w_m\)).  
• \(\mathcal{L}_{\mathrm{conc}}\): concept-purity regularization that penalizes overlap across modules:  
$$
\mathcal{L}_{\mathrm{conc}}
= \sum_{m\neq m'}\bigl\|g_m(X)-g_{m'}(X)\bigr\|^2.
$$

2.4 Experimental Design  
2.4.1 Tasks and Baselines  
We evaluate on three tasks:  
1. **Survival Prediction.** Predict patient overall survival in TCGA cohorts.  
2. **Drug Response & Synergy.** Predict sensitivity of cancer cell lines or patient samples to single drugs and drug combinations.  
3. **Subtype Discovery.** Unsupervised clustering of patients into molecular subtypes with survival separation.  

Baselines include:  
• Black-box GNNs (no ontology, no concept layer).  
• Post-hoc explainers (GNNExplainer, Integrated Gradients).  
• Factor Graph Neural Networks (Ma & Zhang, 2019).  
• Concept Bottleneck Models without graph structure.

2.4.2 Evaluation Metrics  
• **Predictive Performance:**  
  – Classification: AUC-ROC, PR-AUC.  
  – Survival: Concordance index (\(C\)-index).  
  – Regression (drug response): RMSE, \(R^2\).  
• **Interpretability & Explanation Quality:**  
  – **Fidelity:** drop in performance when top-\(k\) concepts/edges are removed.  
  – **Sufficiency:** performance using only top-\(k\) concepts.  
  – **Alignment:** correlation between model-derived concept importance and known pathway relevance (Spearman’s \(\rho\)).  
  – **Stability:** Jaccard similarity of top-\(k\) concept sets across random seeds.  
• **Expert Assessment:** Domain experts rate explanations along clarity, actionability, and biological plausibility on a Likert scale.

2.4.3 Ablation Studies  
We will systematically ablate:  
• Knowledge regularization (\(\lambda_1=0\)).  
• Concept bottleneck layer.  
• Attention vs. uniform aggregation.  
• Sparsity and concept-purity terms.  

2.4.4 Wet-Lab Validation  
For a subset of top-ranked novel biomarkers/drug targets, we will collaborate with wet-lab partners to perform:  
• CRISPR/Cas9 knockdown of candidate genes in relevant cell lines.  
• Cell proliferation and apoptosis assays.  
• Combination index experiments for predicted synergistic drug pairs.

2.4.5 Implementation Details  
• Framework: PyTorch Geometric.  
• Hardware: NVIDIA A100 GPUs.  
• Hyperparameter search: grid search over \(\lambda\) weights, learning rate, hidden dimensions.  
• Cross-validation: nested 5-fold CV with held-out test sets.

3. Expected Outcomes & Impact  
3.1 Expected Outcomes  
• **State-of-the-Art Performance.** Our models will match or exceed black-box GNNs on survival and drug response benchmarks.  
• **High-Fidelity Explanations.** Quantitatively superior fidelity, sufficiency, and alignment scores compared to post-hoc methods.  
• **Novel Discoveries.** Identification of previously unreported gene modules, pathways, and drug combinations with potential clinical relevance—validated through literature survey and wet-lab experiments.  
• **Generalizable Toolkit.** Open-source library supporting user-supplied ontologies and concept definitions for other biomedical tasks.  

3.2 Scientific & Societal Impact  
• **Trustworthy AI in Healthcare.** Transparent models will increase clinician confidence in AI-driven recommendations.  
• **Accelerated Mechanistic Insight.** Embedding domain knowledge in models helps generate hypotheses that can be rapidly tested, closing the loop between computation and experiment.  
• **Precision Medicine Advances.** Mechanistic explanations enable patient stratification and tailored therapies.  
• **Framework for XAI4Science.** Demonstrates how self-explainable architectures can transform other scientific domains (e.g., climate modeling, materials discovery) by coupling predictive accuracy with human-interpretable structure.

4. Conclusion and Future Directions  
We propose a unified, self-explainable ML framework that integrates biomedical domain knowledge into GNN and additive architectures. By aligning model modules with interpretable entities—genes, pathways, drugs—we aim to deliver both high predictive performance and mechanistic insights, bridging the traditional divide between black-box ML and scientific understanding.  

Future work will explore:  
• **Multi-omics Integration.** Extending to metabolomics, proteomics, and imaging data in a unified graph.  
• **Interactive Explanations.** Human-in-the-loop refinement of concept definitions and weights.  
• **Transfer Learning.** Adapting learned knowledge graphs across cancer types or even other diseases.  
• **Broader XAI4Science Applications.** Customizing the framework for climate networks, materials design, and beyond.

By combining rigorous methodology, comprehensive evaluation, and wet-lab validation, this research will chart a path toward trustworthy, discovery-driven AI across the life sciences and other scientific fields.