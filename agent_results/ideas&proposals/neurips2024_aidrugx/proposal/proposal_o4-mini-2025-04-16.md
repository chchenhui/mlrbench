Title  
Multi-Modal Foundation Models for Predicting Therapeutic Outcomes in Cell and Gene Therapies  

Introduction  
Background  
Cell and gene therapies promise transformative treatments for genetic diseases, cancers, and degenerative disorders by directly editing or reprogramming cells. However, the underlying biological processes—from CRISPR‐mediated genome editing to downstream transcriptomic and phenotypic responses—are highly complex and context‐dependent. Traditional single‐modal AI approaches (e.g., sequence‐only or graph‐only models) struggle to capture cross‐modal interactions among genetic perturbations, molecular networks, and cellular phenotypes. As a result, predicting efficacy, safety (e.g., off‐target effects), and delivery efficiency remains a major bottleneck, requiring extensive wet‐lab screening and iterative experimentation.  

Research Objectives  
1. Develop a large–scale multi‐modal foundation model (MMFM) that jointly encodes genetic perturbations (e.g., CRISPR guide sequences), transcriptomic and proteomic readouts, and molecular interaction networks to predict therapeutic outcomes in cell and gene therapies.  
2. Design a hybrid architecture combining transformer‐based sequence encoders, graph neural networks (GNNs) for molecular interactions, and cross‐modal attention mechanisms to align perturbations with downstream effects.  
3. Pre‐train MMFM on extensive public datasets (e.g., DepMap CRISPR screens, GTEx expression profiles, PRIDE proteomics) under multi‐task objectives, then iteratively fine‐tune via active learning on lab‐generated perturbation–response pairs.  
4. Rigorously evaluate MMFM against state-of-the-art baselines (e.g., MAMMAL, scMMGPT, BioMedGPT) on downstream tasks: guide‐efficacy prediction, off‐ target risk estimation, cell-type‐specific editing outcomes, and delivery efficiency.  

Significance  
By integrating diverse biological modalities into a single predictive framework, MMFM will:  
• Accelerate target identification and guide design while reducing reliance on costly wet‐lab screens.  
• Improve safety by more accurately forecasting off‐target effects and adverse phenotypes.  
• Inform design of cell‐type‐specific delivery systems (e.g., lipid nanoparticles) by linking molecular properties to uptake and expression profiles.  
• Provide interpretable insights via attention maps and simple retrieval‐augmented explanations, facilitating regulatory compliance and biological discovery.  

Methodology  
1. Data Collection and Preprocessing  
Public Datasets  
• CRISPR Screens (DepMap): gene‐dependency scores across 1,000+ cancer cell lines, guide sequences, and genome‐wide on‐ and off‐target annotations.  
• Bulk and Single‐Cell Transcriptomics (GTEx, Human Cell Atlas): FPKM/RPKM values for 50+ tissues and 10M single cells; cell‐type annotations.  
• Proteomics (PRIDE): quantitative protein abundance profiles across diverse cell lines under perturbations.  
• Molecular Interaction Networks (BioGRID, STRING): protein–protein and gene–gene interaction graphs with edge confidence scores.  

Lab‐Generated Perturbation–Response Data  
• Targeted CRISPR perturbations in iPSC‐derived neurons and T cells (n≈10,000 perturbation–phenotype pairs).  
• Multi‐omic readouts: scRNA‐seq, CyTOF proteomics, high‐content imaging phenotypes (cell morphology, viability).  

Preprocessing Steps  
1. Sequence Tokenization: convert nucleotide sequences (guide RNAs, UTRs) into overlapping k-mers (k=6) and embed via learnable embeddings.  
2. Expression Normalization: log-transform and z-score bulk and single‐cell expression vectors.  
3. Graph Construction: nodes represent genes/proteins; edges weighted by interaction confidence; add self‐loops.  
4. Modality Alignment: ensure common gene namespaces (HGNC symbols) and map proteomic IDs to gene symbols.  

2. Model Architecture  
Overview  
MMFM comprises three modules: (a) Sequence Encoder $E_s$, (b) Graph Encoder $E_g$, and (c) Cross‐Modal Fusion with downstream Predictor $P$.  

2.1 Sequence Encoder ($E_s$)  
• Architecture: $L_s$‐layer transformer encoder.  
• Input: token sequence $x=(t_1,\dots,t_n)$ for guide RNA or UTR region.  
• Output: sequence embedding $h_s\in\mathbb{R}^{n\times d}$ and pooled vector $z_s\in\mathbb{R}^d$.  

Key equation (multi‐head self‐attention):  
$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr)V
$$  
where $Q = h_sW^Q$, $K = h_sW^K$, $V = h_sW^V$.  

2.2 Graph Encoder ($E_g$)  
• Architecture: $L_g$‐layer graph neural network (e.g., GraphSAGE or GAT).  
• Input: graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ with node features $X\in\mathbb{R}^{|\mathcal V|\times d}$ (baseline expression/protein abundance) and adjacency $A$.  
• Update rule (GraphSAGE example):  
$$
h_v^{(l+1)} = \sigma\bigl(W^{(l)}\cdot\mathrm{AGG}\bigl(\{h_u^{(l)}:u\in \mathcal{N}(v)\}\bigr)\bigr)
$$  
• Output: node embeddings $H_g\in\mathbb{R}^{|\mathcal V|\times d}$; pooled graph representation $z_g\in\mathbb{R}^d$.  

2.3 Cross‐Modal Fusion  
We align $z_s$ and $z_g$ via cross‐attention to produce joint representation $z_{sg}$. Let  
$$
Q = z_sW^Q,\quad K = z_gW^K,\quad V = z_gW^V.
$$  
Then  
$$
\mathrm{CrossAttn}(z_s,z_g) = \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d}}\Bigr)V,
$$  
and the fused vector:  
$$
z_{sg} = \mathrm{LayerNorm}(z_s + \mathrm{CrossAttn}(z_s,z_g)).
$$  

2.4 Predictor Head ($P$)  
Based on the fused embedding $z_{sg}$ and optionally cell‐type embedding $e_c$, we predict:  
• Editing efficacy (regression)  
• Off‐target risk score (classification/regression)  
• Delivery efficiency (regression)  
• Phenotypic profile similarity (contrastive ranking)  

For regression tasks:  
$$
\hat y = W_p z_{sg} + b_p,\quad \mathcal{L}_{\mathrm{MSE}} = \|y-\hat y\|^2.
$$  
For classification tasks:  
$$
\hat p = \mathrm{sigmoid}(W_p z_{sg} + b_p),\quad \mathcal{L}_{\mathrm{BCE}} = -[y\log\hat p + (1-y)\log(1-\hat p)].
$$  

3. Pre‐Training Strategy  
Multi‐Task Objectives  
• Masked Language Modeling (MLM) on sequence data: randomly mask 15% of tokens, predict via cross‐entropy.  
• Graph Link Prediction: mask 10% of edges, predict existence via binary cross‐entropy on node embeddings $H_g$.  
• Cross‐Modal Contrastive Alignment (InfoNCE):  
$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^N\log\frac{\exp(\mathrm{sim}(z_{s,i},z_{g,i})/\tau)}{\sum_{j=1}^N\exp(\mathrm{sim}(z_{s,i},z_{g,j})/\tau)},
$$  
where $\mathrm{sim}(u,v)=u^\top v/\|u\|\|v\|$, $\tau$ temperature.  

Optimization  
Train all modules jointly for $T_p$ epochs on combined public datasets using AdamW.  

4. Active Learning–Powered Fine‐Tuning  
1. Initialize MMFM with pre‐trained weights.  
2. For $k=1,\dots,K$ iterations:  
   a. Apply MMFM to an unlabeled candidate pool (CRISPR guides, target genes, cell types).  
   b. Select top‐$M$ most informative samples by highest predictive uncertainty (e.g., entropy for classification, variance in dropout‐based Bayesian approximation).  
   c. Conduct wet‐lab experiments to obtain ground‐truth readouts (scRNA‐seq, proteomics, phenotypes).  
   d. Fine‐tune MMFM on the expanded labeled set with a smaller learning rate.  
3. Stop when performance plateaus or resource budget exhausted.  

5. Experimental Design and Evaluation  
Benchmarks  
• MAMMAL (Shoshan et al., 2024) multi‐task FM  
• scMMGPT (Shi et al., 2025) single‐cell language model  
• BioMedGPT (Luo et al., 2023) multimodal transformer  
• Deep CNN/RNN off‐target predictors (2023)  
• GNN‐based drug response models (2024)  

Evaluation Tasks & Metrics  
1. Guide Efficacy Prediction (regression)  
   • Metrics: Pearson/Spearman correlation, MSE.  
2. Off‐Target Risk Classification  
   • Metrics: AUC‐ROC, AUPRC, F1 score.  
3. Delivery Efficiency Regression  
   • Metrics: MSE, relative error.  
4. Phenotypic Outcome Ranking  
   • Metrics: top‐K precision, normalized Discounted Cumulative Gain (nDCG).  

Cross‐Validation  
• Leave‐cell‐type‐out (generalization across cell types)  
• Cross‐study validation (train on DepMap, test on lab‐generated data)  
• k‐fold (k=5) random splits by guide sequences  

Ablation Studies  
• Remove cross‐modal attention (sequence‐only vs. graph‐only vs. full MMFM)  
• Vary pre‐training objectives (w/o contrastive, w/o MLM, etc.)  
• Test alternative GNN layers (GAT vs. GraphSAGE)  

Interpretability Analyses  
• Attention weight visualization to map sequence regions to network modules.  
• SHAP value estimation on input features (expression/protein levels).  
• Retrieval‐augmented generation: identify nearest training examples contributing to a given prediction.  

Computational Resources  
• Pre‐training on 8×A100 GPUs for 4 weeks; fine‐tuning on 4×V100 GPUs.  
• Estimated 25M parameters; mixed‐precision training (FP16) to reduce memory.  

Expected Outcomes & Impact  
Expected Outcomes  
• A unified MMFM achieving state‐of‐the‐art performance across all four evaluation tasks, exceeding baselines by 5–10% in correlation/AUC.  
• Robust generalization to unseen cell types and perturbations, demonstrated via cross‐validation.  
• Significant reduction (≥50%) in required wet‐lab experiments for guide selection through active learning.  
• Interpretable model outputs that reveal key sequence motifs, genes, and network modules driving therapeutic effects.  

Impact  
Scientific Impact  
• Establishes a new paradigm for integrating genomic, transcriptomic, proteomic, and phenotypic modalities in a single foundation model.  
• Provides mechanistic insights into therapy mechanisms via interpretable cross‐modal attention.  
• Advances methodology for data‐efficient fine‐tuning in biomedical domains through active learning.  

Translational Impact  
• Accelerates cell and gene therapy development by prioritizing high‐efficacy, low‐risk candidates, potentially saving millions in R&D.  
• Improves patient safety by more accurately forecasting off‐target and adverse phenotypes.  
• Guides design of delivery systems (e.g., lipid nanoparticles) tailored to cell‐type–specific uptake.  

Broader Impacts  
• Open‐source release of MMFM code, pre‐trained weights, and benchmark datasets under a permissive license, democratizing access.  
• Training workshops to enable adoption by academic and industry labs.  
• Fosters collaborations across AI, molecular biology, and translational medicine communities.  

In summary, this proposal will develop and validate a multi‐modal foundation model that coherently integrates diverse biological data to predict therapeutic outcomes in cell and gene therapies. By leveraging transformer‐GNN hybrids, cross‐modal attention, and active learning, we expect to deliver a highly accurate, interpretable, and resource‐efficient tool that accelerates the translation of novel modalities from bench to bedside.