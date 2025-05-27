# Title  
**Multi-Modal Foundation Models for Predicting Therapeutic Outcomes in Cell and Gene Therapies**  

---

# Introduction  

## Background  
Cell and gene therapies (CGTs) represent a paradigm shift in medicine, offering transformative potential for treating genetic disorders, cancer, and autoimmune diseases. Unlike traditional small-molecule or biologic drugs, CGTs operate by modifying genetic material (e.g., CRISPR-based editing), introducing RNA therapeutics, or manipulating cellular systems (e.g., CAR-T cells). However, their development is hindered by the complexity of modeling multi-scale biological interactions: from molecular-level perturbations (e.g., gene knockouts, RNA modifications) to systemic outcomes (e.g., cellular viability, immunogenicity, and delivery efficiency).  

Recent advancements in artificial intelligence (AI) have shown promise in addressing these challenges. For instance, foundation models (FMs) trained on large-scale biomedical datasets, such as MAMMAL (Shoshan et al., 2024) and BioMedGPT (Luo et al., 2023), demonstrate cross-modal integration of molecular and textual data. Similarly, single-cell models like scMMGPT (Shi et al., 2025) leverage transcriptomic data for zero-shot perturbation predictions. However, these approaches often neglect critical aspects of CGTs, such as dynamic molecular interactions, multi-modal readouts (e.g., transcriptomic, proteomic, phenotypic), and the integration of genetic perturbations with downstream therapeutic outcomes. This gap highlights the need for specialized FMs that explicitly model the causal relationships between therapeutic interventions and biological responses.  

## Research Objectives  
This study aims to develop a multi-modal foundation model that predicts therapeutic outcomes in CGTs by:  
1. Integrating genetic/molecular perturbation data (e.g., CRISPR screens, RNA modifications) with multi-omics readouts (transcriptomic, proteomic, phenotypic).  
2. Designing a hybrid architecture combining transformer-based encoders for sequence data (DNA/RNA) and graph neural networks (GNNs) for molecular interaction modeling.  
3. Introducing cross-modal attention mechanisms to align perturbations with biological effects.  
4. Leveraging pre-training on public datasets (DepMap, GTEx) followed by active learning with lab-generated perturbation-response pairs to refine predictions.  

## Significance  
This work addresses key bottlenecks in CGT development:  
- **Off-target effects**: Current CRISPR design tools (e.g., DeepCRISPR; Zhang et al., 2023) struggle to predict unintended genomic edits. Our model will enhance specificity by modeling chromatin accessibility and DNA repair pathways.  
- **Delivery efficiency**: Nanoparticle design for RNA delivery requires balancing stability and cell-type specificity, a challenge tackled by generative models for 3D molecular structures (Section 2.3 of task description).  
- **Costly validation**: Experimental screening of therapeutic candidates remains prohibitively expensive. By prioritizing high-efficacy targets (e.g., optimal guide RNAs), the model could reduce validation cycles by up to 50% (estimated from active learning studies, Tenenbaum et al., 2023).  

By unifying fragmented data modalities and modeling causal perturbation-response relationships, this FM will accelerate the translation of CGTs to clinical applications.  

---

# Methodology  

## 1. Data Collection and Preprocessing  

### Data Sources  
We will curate a multi-modal dataset spanning genetic perturbations, molecular readouts, and therapeutic outcomes:  
- **Perturbation data**: CRISPR screens (DepMap, CRISPRcleanR), antisense oligonucleotide designs (ENCODE), and RNA modification profiles (RNA-MethylationDB).  
- **Readouts**: Transcriptomic (GTEx, TCGA), proteomic (CPTAC), phenotypic (CellProfiler), and imaging data (Image Data Resource).  
- **Therapeutic metadata**: Delivery efficiency metrics (LNP databases), off-target scores (Guide-seq), and cell-type-specific regulatory elements (FACTORI).  

### Preprocessing  
- **Sequence tokenization**: DNA/RNA sequences will be tokenized into k-mers or learned embeddings using a BPE tokenizer (vocabulary size: 8,192).  
- **Graph construction**: Molecular interactions (protein-protein, gene-gene) will be represented as graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes $\mathcal{V}$ represent genes/proteins and edges $\mathcal{E}$ denote interactions (from STRING and BioGRID databases).  
- **Normalization**: Omics data (e.g., scRNA-seq counts) will be log-transformed and batch-corrected using Harmony (Korsunsky et al., 2019).  

## 2. Model Architecture  

### Hybrid Encoder Design  
The model combines **transformer encoders** for sequence data and **GNNs** for interaction networks (Figure 1):  

#### Sequence Encoder  
For DNA/RNA sequences $S = \{s_1, s_2, ..., s_N\}$, we use a transformer encoder:  
$$
\mathbf{H}^{(t)} = \text{Transformer}(\mathbf{E}_t(S)),
$$  
where $\mathbf{E}_t$ denotes token embeddings and $\mathbf{H}^{(t)} \in \mathbb{R}^{N \times d}$ represents contextualized sequence features.  

#### Interaction Encoder  
For a molecular graph $\mathcal{G}$, we employ a Graph Attention Network (GAT) to compute node embeddings:  
$$
\mathbf{h}_i^{(g)} = \sigma\left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right),
$$  
where $\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \, \|\, \mathbf{W}\mathbf{h}_j]))$ are attention coefficients, $\mathbf{W}$ is a learnable matrix, and $\|\|$ denotes concatenation.  

#### Cross-Modal Attention  
A cross-modal transformer aligns sequence and graph embeddings:  
$$
\mathbf{H}^{(a)} = \text{Attention}(\mathbf{Q}^{(t)}, \mathbf{K}^{(g)}, \mathbf{V}^{(g)}),
$$  
where $\mathbf{Q}^{(t)} = \mathbf{H}^{(t)}\mathbf{W}_Q$, $\mathbf{K}^{(g)} = \mathbf{H}^{(g)}\mathbf{W}_K$, and $\mathbf{V}^{(g)} = \mathbf{H}^{(g)}\mathbf{W}_V$ are learnable query, key, and value matrices.  

### Task-Specific Heads  
- **CRISPR guide design**: Predict on/off-target scores using a multi-layer perceptron (MLP) on $\mathbf{H}^{(a)}$.  
- **Delivery system optimization**: A diffusion model generates lipid nanoparticle (LNP) structures conditioned on $\mathbf{H}^{(a)}$.  
- **Therapeutic outcome classification**: Cell viability or toxicity predictions via a softmax layer.  

## 3. Training Strategy  

### Pre-Training  
- **Masked sequence modeling**: 15% of sequence tokens are masked, with the model trained to predict them.  
- **Contrastive learning**: Align graph embeddings $\mathbf{H}^{(g)}$ with matched omics readouts using InfoNCE loss:  
$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\mathbf{h}^{(g)} \cdot \mathbf{h}^{(o)}/\tau)}{\sum_{k=1}^K \exp(\mathbf{h}^{(g)} \cdot \mathbf{h}_k^{(o)}/\tau)},
$$  
where $\mathbf{h}^{(o)}$ is an omics embedding and $\tau$ is a temperature parameter.  

### Fine-Tuning with Active Learning  
To overcome limited annotated data (Challenge 3 in literature review), we adopt an active learning loop:  
1. Train the model on existing public data.  
2. Use the model to score a pool of candidate perturbations (e.g., CRISPR guides) by uncertainty metrics (e.g., entropy).  
3. Select top-$k$ uncertain samples for experimental validation (e.g., Guide-seq assays).  
4. Re-train the model with new labels.  

This iterative approach reduces experimental validation costs while improving generalization (Challenge 4).  

## 4. Experimental Design and Evaluation Metrics  

### Benchmarks  
We compare our model against state-of-the-art baselines:  
- **Single-modal models**: AlphaFold2 (protein structure), DeepCRISPR (off-target prediction).  
- **Multi-modal models**: MAMMAL, scMMGPT.  

### Metrics  
- **Perturbation prediction**: Area under the precision-recall curve (AUPRC) for off-target CRISPR sites.  
- **Delivery optimization**: Binding affinity $K_d$ (computed via molecular docking) and delivery efficiency $\eta = \frac{\text{RNA in target cells}}{\text{total RNA}}$.  
- **Generalization**: Zero-shot performance on unseen cell types (e.g., predicting hepatic LNP uptake using training data from HEK293 cells).  

### Ablation Studies  
- Impact of cross-modal attention (vs. late fusion).  
- GNN vs. MLP for interaction modeling.  

---

# Expected Outcomes & Impact  

## Expected Outcomes  
1. **First End-to-End FM for CGTs**: The model will predict optimal CRISPR targets, design LNPs for tissue-specific delivery, and quantify off-target risks (e.g., $ \text{Off-target score} = \frac{\sum_{i} \text{Guide}_i \cdot \text{Mismatch}_i}{\|\text{Guide}\|_2} $).  
2. **Improved Predictive Accuracy**: Based on MAMMAL’s benchmarking (Shoshan et al., 2024), we expect a 15–20% lift in AUPRC for off-target prediction and a 30% reduction in false positives.  
3. **Public Dataset and Model Release**: We will release a curated multi-modal CGT dataset (100,000+ samples) and pre-trained models.  

## Anticipated Impact  
1. **Accelerated CGT Development**: By prioritizing high-efficacy candidates, the model could reduce experimental validation cycles by 40% (Tenenbaum et al., 2023).  
2. **Safer Therapies**: Explicit modeling of DNA repair pathways and chromatin accessibility could reduce genotoxicity risks by enabling precise guide RNA design.  
3. **Foundation for Multi-Modal FMs**: The hybrid architecture (transformer + GNN) offers a blueprint for integrating diverse data in drug discovery, advancing the ML track’s goals (Section 1.2 of task description).  

## Future Directions  
- Extend to **peptide therapeutics** and **microbiome-based therapies** (as outlined in Section 1.1).  
- Integrate **reinforcement learning** to optimize multi-step therapeutic pipelines (e.g., simultaneous gene editing + delivery system design).  

--- 

This proposal bridges the gap between cutting-edge AI and the biological complexity of CGTs, aligning with both the application track (AI for DNA/RNA therapeutics) and the ML track (foundational models for multi-modal perturbations).