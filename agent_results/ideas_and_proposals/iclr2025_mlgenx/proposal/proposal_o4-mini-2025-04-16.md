Title  
RegCircuitsFM: A Multi-Scale Attention and Graph-Inductive Foundation Model for Genomic Regulatory Circuit Discovery  

1. Introduction  
Background  
Understanding the structure and dynamics of gene regulatory networks (GRNs) is a foundational problem in molecular biology and drug discovery. Genes are regulated by a complex interplay of local sequence motifs (e.g., transcription factor binding sites), distal enhancers, chromatin modifications, and three‐dimensional genome architecture. Dysregulation of these circuits underlies many diseases, yet conventional computational methods—correlation-based network inference, simple deep-learning sequence models—often fail to capture long‐range dependencies, noisy measurements, multi‐modal signals, and context specificity simultaneously. As a result, critical disease mechanisms remain hidden and target identification for novel therapeutics stalls.  

Research Objectives  
We propose RegCircuitsFM, a foundation model that learns a unified representation of genomic regulatory circuits by integrating:  
• A multi‐scale attention encoder for local and global sequence features.  
• A graph neural induction module to explicitly model gene–gene and enhancer–gene interactions.  
• A perturbation prediction head for in silico simulation of genetic or chemical interventions.  

Our specific aims are:  
1. Pre‐train RegCircuitsFM on diverse public genomic datasets (ENCODE, Roadmap Epigenomics, GTEx).  
2. Fine‐tune on perturbation screens (CRISPR knockouts, drug treatments) to predict expression changes.  
3. Evaluate on held-out cell types, network reconstruction benchmarks, and target‐identification tasks.  

Significance  
By capturing both the “grammar” of gene regulation across cell types and the causal effects of perturbations, RegCircuitsFM will:  
– Accelerate target discovery by in silico screening of regulatory element perturbations.  
– Provide interpretable mechanistic insights via attention weights and graph edges.  
– Generalize across tissues and species, aiding translational research.  

Alignment with Workshop Themes  
This research addresses “Foundation models for genomics,” “Causal representation learning,” “Perturbation biology,” and “Multimodal representation learning,” bridging machine learning and genomics to push drug‐discovery frontiers.  

2. Literature Review and Related Work  
Quadratic Graph Attention Network (Q-GAT) (Zhang et al., 2023)  
• Introduces quadratic neurons in a dual‐attention GNN for GRN construction from expression data.  
• Demonstrates robustness to noise, but lacks sequence‐level modeling and perturbation forecasting.  

DiscoGen (Ke et al., 2023)  
• A neural‐network causal‐discovery method that denoises expression and leverages interventional data.  
• Strong in topology inference, but does not integrate sequence information or global context.  

GCBLANE (Ferrao et al., 2025)  
• A hybrid convolutional, BiLSTM, multi‐head attention, and GNN model for TF‐binding site prediction.  
• Excels at motif localization, yet focuses on local features and omits explicit graph induction.  

GATv2 for GRN analysis (Otal et al., 2024)  
• Uses Graph Attention v2 to model interactions from expression data, identifying key regulators.  
• Limited by single‐modal input (expression) and no perturbation prediction.  

Gaps and Opportunities  
1. No existing foundation model jointly learns sequence grammar, regulatory graph structure, and perturbation responses.  
2. Current models capture either local sequence signals or graph topology, but not both at multiple scales.  
3. Perturbation‐prediction modules remain underexplored in a unified pretrain/fine‐tune framework.  

3. Methodology  
3.1 Data Collection and Preprocessing  
Datasets  
– ENCODE & Roadmap Epigenomics: ChIP-seq (TFs, histone marks), DNase-seq, ATAC-seq.  
– GTEx: bulk RNA-seq across tissues.  
– High-resolution Hi-C: 3D contacts for candidate enhancer–promoter links.  
– CRISPR screens & drug‐perturbation RNA-seq from DepMap & LINCS.  

Preprocessing Steps  
1. Sequence Extraction: For each gene, extract a 10 kb promoter region and candidate enhancers ±1 Mb from TSS.  
2. Tokenization: One-hot encode nucleotides; apply k-mer embedding (k=6) to reduce sequence length.  
3. Graph Construction:  
   • Nodes: genes and enhancers.  
   • Edges: weighted by Hi-C contact frequency, co‐expression correlation, and eQTL associations.  
4. Normalization: Batch-norm and log-transform expression values; min‐max scale graph edge weights.  

3.2 Model Architecture  
Overall Structure  
RegCircuitsFM comprises three modules: Sequence Encoder, Graph Induction, and Perturbation Predictor.  

3.2.1 Multi‐Scale Sequence Encoder  
We embed the tokenized sequence into hidden vectors $x_i\in\mathbb{R}^d$ and apply $L$ layers of multi‐scale attention. Each layer computes:  
Inline attention formula:  
$\displaystyle
\text{MSAttn}(Q,K,V)=\sum_{s\in S} \mathrm{softmax}\Bigl(\frac{Q_s K_s^\top}{\sqrt{d_s}}\Bigr)V_s
$  
where $S=\{\text{local, regional, global}\}$ defines three window sizes (e.g. 64 bp, 512 bp, full sequence), and $d_s$ is the sub‐dimension. We use standard multi‐head decomposition on each scale.  

3.2.2 Graph Induction Module  
We treat nodes $v$ with features $h_v^{(0)}$ from the Sequence Encoder (for enhancers and promoters) or positional embedding (for genes). We then apply $M$ layers of Graph Attention v2:  
Block equation:  
$$
h_v^{(\ell+1)} = \bigg\Vert_{k=1}^H \sigma\Bigl(\sum_{u\in\mathcal{N}(v)} \alpha_{vu}^{(\ell,k)} W^{(\ell,k)} h_u^{(\ell)}\Bigr),
$$  
$$
\alpha_{vu}^{(\ell,k)} = \frac{\exp\bigl(\mathrm{LeakyReLU}(a^{(\ell,k)\top}[W^{(\ell,k)}h_v^{(\ell)}\Vert W^{(\ell,k)}h_u^{(\ell)}])\bigr)}{\sum_{u'\in\mathcal{N}(v)}\exp(\cdots)}.
$$  
Here $H$ is the number of attention heads, $\Vert$ denotes concatenation, and $\sigma$ is ELU.  

We simultaneously learn to update edge weights via a learned function $f_\theta(h_v,h_u)$ and re‐infer $\mathcal{N}(v)$ for link‐prediction tasks during pretraining.  

3.2.3 Perturbation Prediction Module  
Given an input graph $\mathcal{G}$ and a perturbation mask $p$ (e.g. knockout of enhancer $e$ or gene $g$), we zero‐out corresponding features and recompute node embeddings $\tilde h_v$. We then predict the log‐fold change in expression $\Delta y_g$ for each gene $g$ with:  
$$
\hat{\Delta y}_g = \mathrm{MLP}\bigl(\tilde h_g^{(M)}\bigr).
$$  

3.3 Pretraining and Fine-Tuning Objectives  
We jointly optimize a composite loss:  
$$
\mathcal{L} = \lambda_1\mathcal{L}_{\mathrm{MLM}} + \lambda_2\mathcal{L}_{\mathrm{Link}} + \lambda_3\mathcal{L}_{\mathrm{Perturb}}.
$$  
1. Masked Language Modeling (MLM): Mask 15% of nucleotides and predict them.  
2. Graph Reconstruction Loss: Randomly remove edges and predict their existence via cross-entropy.  
3. Perturbation Loss: On held-out CRISPR & drug data, minimize MSE$(\hat{\Delta y},\Delta y)$.  

Hyperparameters $\lambda_i$ are tuned by cross-validation.  

3.4 Experimental Design and Evaluation  
Baselines  
• Q-GAT, DiscoGen, GCBLANE, GATv2-only, Transformer-only.  

Tasks and Metrics  
1. GRN Reconstruction:  
   – AUC‐ROC & AUPRC on held-out edges.  
2. Perturbation Prediction:  
   – Pearson’s $r$ and MSE between predicted and observed $\Delta y$.  
3. Downstream Target Identification:  
   – Precision@k on CRISPR essentiality screens.  
4. Interpretability:  
   – Motif enrichment in high‐attention regions (Fisher’s exact test).  
5. Generalization:  
   – Zero‐shot transfer to new cell types (change in AUC).  
6. Computational Efficiency:  
   – Training time per epoch and memory footprint.  

Ablation Studies  
– Remove multi‐scale paths, graph induction, or perturbation head.  
– Assess performance drops to quantify component importance.  

Reproducibility  
All code, pretrained weights, and processed datasets will be released under a permissive open‐source license.  

4. Expected Outcomes & Impact  
4.1 Anticipated Technical Outcomes  
• A pre-trained foundation model achieving ≥10% lift in AUPRC for GRN reconstruction vs. state-of-the-art.  
• Perturbation prediction with Pearson’s $r>0.7$ on held-out drug‐screen datasets.  
• Robust cross-cell-type transfer with ≤5% drop in performance.  
• Interpretable attention maps overlapping >60% with known TF‐binding sites and enhancers.  
• Scalable pipeline capable of processing entire human chromosomes in parallel.  

4.2 Broader Scientific & Societal Impact  
Accelerating Drug Discovery  
By simulating genetic and chemical perturbations in silico, RegCircuitsFM will expedite the identification of novel therapeutic targets, reducing time and cost in early‐stage drug development.  

Elucidating Disease Mechanisms  
Interpretable insights into regulatory dependencies will shed light on the molecular underpinnings of complex diseases—cancer, neurodegeneration, immunological disorders.  

Enabling Foundation Models in Biology  
This work pioneers a template for large‐scale pretraining in genomics, promoting community adoption of foundation models that integrate sequence, graph, and perturbation modalities.  

Educational & Open‐Science Contribution  
All models, data splits, and evaluation scripts will be publicly available, fostering reproducibility and training the next generation of computational biologists.  

Ethical Considerations  
While empowering drug discovery, responsible guidelines will be issued to prevent misuse in harmful genome‐editing or dual‐use contexts.  

Alignment with Workshop Goals  
RegCircuitsFM addresses multiple workshop tracks—Foundation Models for Genomics, Causal Representation Learning, Perturbation Biology, and Graph Neural Networks—fostering interdisciplinary progress at the intersection of ML and genomics.