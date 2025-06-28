# Title  
**A Multiscale Foundation Model for Inferring Genomic Regulatory Circuits via Hybrid Attention-GNN Architecture**

---

## Introduction  
### Background  
Gene regulatory networks (GRNs) govern cellular function by encoding how genes interact with transcription factors, epigenetic factors, and enhancers. Despite their central role in disease mechanistics and drug discovery, GRNs remain poorly characterized due to three key challenges: (1) *noisy high-dimensional data* (Mardis, 2021), (2) *long-range dependencies* in genomic sequences (Liu et al., 2023), and (3) *context-specific regulation* across tissues and conditions (Mathelier et al., 2017). Existing methods struggle to model these dependencies robustly. For example, quadratic graph attention networks (Q-GAT) improve noise robustness but miss long-range regulatory patterns (Zhang et al., 2023), while DiscoGen handles interventional data but relies on shallow architectures lim-ited in scalability (Ke et al., 2023). Methodologies like GCBLANE (Ferrao et al., 2025) show promise in transcription factor binding site prediction through hybrid CNN-attention-GNN designs but lack dynamic modeling of perturbation effects.

The advent of foundation models—pre-trained on diverse genomic datasets—presents a transformative opportunity for GRN modeling. These models can learn universal "genomic grammars" across cell types and tissues, enabling in silico perturbation analysis for drug target discovery. However, no foundation model currently combines scale-specific attention mechanisms with graph neural networks (GNNs) for explicit regulatory circuit inference. This gap limits progress in identifying therapeutic targets in diseases like cancer, where context-specific gene regulation is paramount.

### Research Objectives  
This proposal aims to develop **RegulatoryCircuit-FOUNDATION**, a multiscale foundation model that:  
1. Integrates *multi-scale attention mechanisms* to capture local sequence features (promoters, enhancers) and global regulatory patterns (long-range chromatin interactions).  
2. Constructs *context-aware regulatory graphs* via GNNs to model gene-gene and enhancer-gene interactions.  
3. Implements a *perturbation prediction module* to simulate the downstream effects of genetic or pharmacologic interventions.  

### Significance  
By addressing critical limitations in existing GRN models, this work will:  
- Enable *in silico drug screening* by predicting downstream effects of perturbing targets.  
- Uncover *cell-type-specific regulatory dependencies* in diseases like Alzheimer’s and pancreatic cancer.  
- Serve as a reusable foundation model for tasks like CRISPR target discovery and synthetic promoter design.  

The proposed methodology builds on recent progress in biologically-informed machine learning, including GNN-based GRN analysis (Otal et al., 2024) and attention-driven gene expression modeling (Zhou et al., 2021).

---

## Methodology  
### Data Collection & Preprocessing  
#### Datasets  
- **Reference Genomes**: Human (hg38), mouse (mm10), annotated with gene body coordinates.  
- **Omics Data**:  
  - Epigenetic: ENCODE (ATAC-seq, ChIP-seq for TFs/histone marks), Roadmap Epigenomics (DNA methylation).  
  - Transcriptomic: GTEx (bulk RNA), Human Cell Landscape (single-cell RNA).  
  - Regulatory Interactions: HiChIP, EN-CODE enhancer-gene linking.  
- **Perturbation Data**: CRISPR screening from DepMap, drug response from LINCS.  

#### Preprocessing Pipeline  
1. For each cell type:  
   a. Tokenize sequences into 100-bp windows with 25-bp overlapping sliding window.  
   b. Annotate windows with epigenetic marks (binary), GC content (continuous), and positional embedding.  
   c. Construct *prior regulatory graphs* using HiChIP-loops and enhancer-promoter correlations from GTEx.  

### Model Architecture  
#### 1. Multi-Scale Attention Encoder  
The encoder processes genomic sequences and regulatory context through two pathways:  
- **Local Attention Path**:  
  - Input: $ X_{\text{local}} \in \mathbb{R}^{n \times d_{\text{local}}} $, where $ n $ is window count and $ d_{\text{local}} $ includes epigenetic marks and GC content.  
  - Operation: Multi-head attention ($ h $-heads) with query/key matrices $ Q_l, K_l, V_l $ (Vaswani et al., 2017):  
    $$ \text{MH}_l = \text{Concat}[\text{Attn}(Q_l^i, K_l^i, V_l^i)]W^O_l \quad \text{for } i=1,\dots,h $$  
  - Output: Local embeddings $ E_{\text{local}} \in \mathbb{R}^{n \times d} $.  

- **Global Attention Path**:  
  - Input: $ X_{\text{global}} \in \mathbb{R}^{g \times d_{\text{global}}} $, where $ g $ genes and $ d_{\text{global}} $ includes expression levels and methylation.  
  - Operation: Same as above, producing gene-level embeddings $ E_{\text{global}} \in \mathbb{R}^{g \times d} $.  

#### 2. Regulatory Graph Induction with Hybrid GNNs  
Given a prior adjacency matrix $ A \in \{0,1\}^{g \times g} $, refine regulatory edges using a hybrid GNN:  
- For each edge $ (u,v) $:  
  a. Concatenate $ E_{\text{global}}[u] $, $ E_{\text{global}}[v] $, and HiChIP interaction strength (if any).  
  b. Compute edge score via MLP:  
    $$ s_{u,v} = \sigma(W_{\text{GNN}}^{(2)} \cdot \text{ReLU}(W_{\text{GNN}}^{(1)} \cdot \text{Concat}(E_u, E_v, \text{HiChIP}_{u,v})) ) $$  
  c. Threshold scores to induce probabilistic adjacency matrix $ \hat{A} $.  

#### 3. Perturbation Prediction Module  
Given a perturbation $ p \in \mathbb{R}^g $ (e.g., CRISPR-KO gene $ k $), predict expression changes via nested attention:  
- Initialize perturbation vector as:  
  $$ p_i = \begin{cases}1 & i=k\\0 & \text{otherwise}\end{cases} $$  
- Update regulatory graph via:  
  $$ \Delta E = \sigma(W_{\Delta} \cdot \text{Attn}(E_{\text{global}}, \hat{A}, p)) $$  
- Output predicted expression change $ \Delta \text{Expr} = \text{MLP}_{\text{diff}}(\Delta E) $.  

### Training Protocol  
1. **Pre-training**:  
   - Loss: Weighted combination of epigenetic reconstruction $ \mathcal{L}_{\text{epi}} $, expression $ \mathcal{L}_{\text{expr}} $, and interaction recovery $ \mathcal{L}_{\text{int}} $.  
   - $$ \mathcal{L}_{\text{pretrain}} = \alpha \mathcal{L}_{\text{epi}} + \beta \mathcal{L}_{\text{expr}} + \gamma \mathcal{L}_{\text{int}} $$  
   - $\mathcal{L}_{\text{epi}}$: SVM-style reconstruction of methylation.  
   - $\mathcal{L}_{\text{int}}$: Binary cross-entropy on HiChIP edges.  

2. **Fine-tuning on CRISPR Data**:  
   - Predict knockdown target's downstream effects using DepMap screens.  

### Evaluation Metrics  
| Task                      | Metric                          | Baselines                      |  
|---------------------------|----------------------------------|----------------------------------|  
| Regulatory Graph Recovery | AUROC, AUPRC, FDR-corrected Jaccard | Q-GAT, GraphRNN+HiChIP           |  
| Perturbation Prediction   | Spearman rho, ΔExpr Correlation  | DiscoGen, TF-knockout CNN models |  
| Scalability               | Model memory & wall time         | scScope, Babel (GPU-only)        |  

### Experimental Design  
1. **Ablation Studies**:  
   - Remove local/global paths to measure their marginal utility.  
   - Compare GNN modules: GCN vs. GATv2 vs. graphSAGE.  

2. **Case Studies**:  
   - Predict tumor suppressor effects in TP53-mutant lines.  
   - Discover synthetic lethality in pancreatic cancer using OLAPAR-associated genes.  

3. **Interpretability Analysis**:  
   - Visualize attention heatmaps for CTCF binding sites.  
   - Rank perturbation importance using SHAP values.  

---

## Expected Outcomes & Impact  
### Technical Outcomes  
1. **State-of-the-Art GRN Accuracy**:  
   Achieve AUPRC ≥ 0.65 on HiChIP edge prediction (vs. 0.58 for GATv2) by leveraging multiscale attention.  

2. **Perturbation Simulation**:  
   Correlate predicted ΔExpr with DepMap CRISPR data (Spearman ≥ 0.35), enabling in silico target screening.  

3. **Model Reusability**:  
   Open-source release of foundation weights (modding-hub) and datasets (Zenodo) for tasks like scRNA perturbation prediction and enhancer design.  

### Biological Insight & Therapeutic Applications  
1. **Novel Regulatory Dependencies**:  
   Identify enhancer-gene interactions modulating immune checkpoints (e.g., PD1 enhancers in T-cells).  

2. **Drug Candidate Screening**:  
   Prioritize SMARCA2 and CREBBP for trial in small-cell lung cancer using perturbation experiments.  

3. **Clinical Translation**:  
   Partner with Genentech to validate predicted regulators in PSMAxBDT cell engagers.  

### Long-Term Impact  
This model will catalyze a paradigm shift in drug discovery by:  
- Reducing preclinical trial costs via *in silico target prioritization*.  
- Accelerating approval of RNA-based therapeutics through mechanistic validation.  
- Establishing genomic foundation models as essential tools for FDA regulatory review.  

--- 

(Supplementary references incorporated per workshop guidelines.)