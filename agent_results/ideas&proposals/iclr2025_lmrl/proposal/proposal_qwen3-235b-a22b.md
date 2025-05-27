# Causal Graph-Contrast: A Multimodal Pretraining Framework for Cross-Scale Biological Representations

## Introduction

### Background  
Recent advances in representation learning for biological data have produced foundation models that excel at encoding single-modal datasets such as protein sequences (EsmFold, AlphaFold 2), cellular imaging (CellProfiler), or genomic profiling (GRO-seq). However, critical limitations persist in two dimensions: (1) **cross-scale causal alignment**—models cannot disentangle mechanistic relationships between molecular-level attributes (e.g., protein 3D structure) and cellular-scale phenotypes (e.g., morphological changes), and (2) **generalization to unseen perturbations**—representations often fail to extrapolate to novel biological contexts like unobserved drug dosages or gene knockouts. These gaps hinder transformative applications including *in silico* perturbation simulation, rational drug design, and predictive toxicology. The LMRL workshop’s emphasis on evaluating causal and generalizable representations aligns directly with overcoming these challenges.

### Research Objectives  
This research proposes **Causal Graph-Contrast**, a self-supervised graph-based framework that jointly pretrains molecular and cellular data to explicitly capture causal interactions across biological scales. The three core objectives are:
1. **Integrate heterogeneous biological data** into a unified graph framework encompassing atomic, molecular, and cellular modalities.
2. **Pretrain representations via contrastive and causal objectives** to ensure alignment across scales and robustness against distributional shifts.
3. **Benchmark generalization** on downstream tasks involving out-of-distribution (OOd) perturbations and zero-shot scenarios.

### Significance  
By bridging the gap between mechanistic biology and machine learning, this work directly addresses the LMRL workshop’s agenda in three ways:
1. **Multimodal foundation models**: Advances cross-modal integration of structured data (molecular graphs) with unstructured modalities (high-content imaging).
2. **Causal representation learning**: Introduces interventions as explicit signals to disentangle spurious correlations in observational data.
3. **Generalization metrics**: Proposes evaluation protocols for stress-testing embeddings on OOD biological perturbations, beyond traditional i.i.d. benchmarks. Success would enable *virtual cell* simulations where representations predict cellular responses to arbitrary molecular interventions, accelerating drug discovery and mechanistic discovery.

---

## Methodology

### 1. Data Integration: Heterogeneous Biological Graphs  
We construct a hierarchical graph $ \mathcal{G} = (\mathcal{V}, \mathcal{E}) $ spanning three biological scales:
- **Atomic scale**: Nodes $ v_i \in \mathcal{V}_{\text{atom}} $ represent atoms in small molecules or protein residues; edges $ e_{ij} \in \mathcal{E}_{\text{atom}} $ encode covalent/bond angles.
- **Molecular scale**: Nodes $ v_i \in \mathcal{V}_{\text{mol}} $ denote proteins or drug candidates; edges $ e_{ij} \in \mathcal{E}_{\text{mol}} $ include protein-protein interactions (PPIs) or ligand-target bindings.
- **Cellular scale**: Nodes $ v_i \in \mathcal{V}_{\text{cell}} $ represent subcellular regions (e.g., nuclear envelope, cytoplasm) extracted from high-content imaging via graph segmentation (e.g., Superpixel clustering); edges $ e_{ij} \in \mathcal{E}_{\text{cell}} $ model spatial proximity and organelle interactions.  

Inter-scale links $ e_{ij} \in \mathcal{E}_{\text{global}} $ connect protein-level features (e.g., phosphorylation sites) to cellular morphology (e.g., mitochondrial clustering). This structure enables modeling of multiscale causal relationships (e.g., how a ligand’s binding affinity affects organelle dynamics).

### 2. Pretraining Tasks  

#### a) Masked Node/Edge Recovery  
We train a graph neural network (GNN) to reconstruct corrupted subgraphs:
- **Masking procedure**: Randomly select 15% of atomic nodes $ v_i $ and 30% of edges $ e_{ij} $, replacing node features $ \mathbf{x}_i \in \mathbb{R}^{d_{\text{atom}}} $ with learnable tokens and removing edges.  
- **Reconstruction loss**: The GNN encoder $ f_{\theta} $ infers the original features $ \hat{\mathbf{x}}_i $ and adjacency matrix $ \hat{A} \in \{0,1\}^{N \times N} $ via a decoder head $ g_{\phi} $:  
$$
\mathcal{L}_{\text{MNE}} = \mathbb{E}_{\mathcal{G}}\left[ \sum_{v_i \in \mathcal{V}_{\text{masked}}} \text{CE}(\hat{\mathbf{x}}_i, \mathbf{x}_i) + \lambda_1 \cdot \sum_{e_{ij} \in \mathcal{E}_{\text{masked}}} \text{BCE}(\hat{a}_{ij}, a_{ij}) \right],
$$
where CE/BCE denote cross-entropy and binary cross-entropy, and $ \lambda_1 $ balances reconstruction weights. This task enforces local structure learning in molecular and cellular graphs.

#### b) Cross-Modal Contrastive Learning  
To align dissimilar modalities (e.g., small molecules vs. cellular phenotypes), we introduce an InfoNCE-style loss:
- **Positive/negative sampling**: For perturbation metadata $ \tau \in \mathbb{R}^k $ (e.g., drug dosage, gene knockout status), pair molecular graph $ \mathcal{G}_{\text{mol}} $ with its corresponding cellular graph $ \mathcal{G}_{\text{cell}} $ as positives. Negatives are sampled as unmatched graphs with mismatched $ \tau $.  
- **Representation projector**: Apply modality-specific projection heads $ h^{\text{mol}}, h^{\text{cell}} $ to global graph embeddings $ \mathbf{z}_{\text{mol}} = h^{\text{mol}}(f_{\theta}(\mathcal{G}_{\text{mol}})) $, $ \mathbf{z}_{\text{cell}} = h^{\text{cell}}(f_{\theta}(\mathcal{G}_{\text{cell}})) $.  
- **Contrastive objective**: Compute cosine similarity $ s(\mathbf{z}_{\text{mol}}, \mathbf{z}_{\text{cell}}) $ between all pairs in a batch:
$$
\mathcal{L}_{\text{XMC}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\mathbf{z}_{\text{mol}}^{(i)}, \mathbf{z}_{\text{cell}}^{(i)}) / \tau}}{\sum_{j=1}^N e^{s(\mathbf{z}_{\text{mol}}^{(i)}, \mathbf{z}_{\text{cell}}^{(j)}) / \tau}}.
$$
This pulls representations of perturbation-aligned multimodal samples closer together while repelling misaligned pairs.

#### c) Causal Intervention Modeling  
To disentangle causal from correlative signals, we model interventions $ do(\mathbf{x}) $ (Pearl, 2009) using structural causal models (SCMs):
- **Encoder-decoder framework**: Given a perturbation $ \tau $, predict the counterfactual cellular graph $ \mathcal{G}_{\text{cell}}' $ using a causal head $ c_{\psi} $:  
$$
\mathbf{z}_{\text{interv}} = c_{\psi}(f_{\theta}(\mathcal{G}_{\text{mol}}), \tau) \quad \text{(intervention embedding)}.
$$
- **Reconstruction loss**: Train $ c_{\psi} $ to minimize the discrepancy between predicted and observed cellular graphs:
$$
\mathcal{L}_{\text{CIM}} = \mathbb{E}_{\mathcal{G}, \tau}\left[ \|\mathbf{z}_{\text{interv}} - f_{\theta}(\mathcal{G}_{\text{cell}}')\|_2^2 + \lambda_2 \cdot \text{GNN Loss}(\hat{\mathcal{G}}', \mathcal{G}_{\text{cell}}') \right].
$$
Here, GNN Loss evaluates the topological consistency of the predicted cellular graph $ \hat{\mathcal{G}}' $ against observations. This task forces the model to encode invariant causal mechanisms rather than static correlations.

### 3. Experimental Design  

#### a) Datasets  
- **Molecular data**: ChEMBL (small molecules) and AlphaFoldDB (protein structures).  
- **Cellular imaging**: JUMP-CP benchmark, containing matched drug-cell phenotype pairs across 128 cell lines.  
- **Perturbation metadata**: RxRx3 for gene knockout-cell imaging links; LINCS L1000 for transcriptomic responses.

#### b) Baselines  
- **Unimodal models**: GIN, GraphSAGE (molecular), and ResNet50 (cell images).  
- **Multimodal models**: HyperGCL, MOGCL, Graph-ATM.  
- **Causal models**: CausalGAN, Causal InfoGAN adapted to graph domains.

#### c) Evaluation Metrics  
1. **Drug activity prediction (transfer learning)**: AUROC and AUPRC on DTC-IC50 for predicting compound efficacy.  
2. **Few-shot phenotype classification**: 5-shot accuracy using Prototypical Networks across 10 cell types in JUMP-CP.  
3. **OOd generalization**: Accuracy drop on high-dose (OOD) vs. low-dose (training) samples in RXR datasets, quantified via $ \Delta_{\text{OOD}} $.  
4. **Causal validity**: Structural Hamming distance (SHD) between learned and ground-truth intervention graphs in synthetic benchmarks.  

#### d) Ablation Studies  
- Disentangle contributions of $ \mathcal{L}_{\text{MNE}} $, $ \mathcal{L}_{\text{XMC}} $, and $ \mathcal{L}_{\text{CIM}} $ via ablation experiments.  
- Assess graph hierarchy depth (2 vs. 3 levels) and attention mechanisms on node classification performance.  

#### e) Implementation Details  
- Model uses 4-layer Graph Attention Networks (GATs) with edge features.  
- Training: Adam optimizer ($ \beta_1 = 0.9, \beta_2 = 0.999 $, initial LR = 3e-4), cosine decay over 200 epochs.  
- Hardware: 8 × NVIDIA A100 GPUs, distributed training with DDP.

---

## Expected Outcomes & Impact  

### Anticipated Results  
1. **Cross-scale causal embeddings**: Representations will explicitly link molecular attributes (e.g., binding sites) to cellular phenotypes (e.g., apoptosis markers), validated via improved SHD in synthetic perturbation benchmarks.  
2. **OOd generalization**: A 15–20% reduction in $ \Delta_{\text{OOD}} $ compared to HyperGCL and MOGCL on high-dose drug screens.  
3. **Downstream performance**: State-of-the-art AUROC (≥ 0.90) on DTC-IC50 and 5-shot accuracy ≥ 75% on JUMP-CP, surpassing Graph-ATM.  
4. **Biological interpretability**: Attention maps will highlight causal mediators (e.g., ERK phosphorylation for MEK inhibitor responses).

### Significance for LMRL Community  
This work directly addresses the 2025 LMRL workshop’s agenda by:
1. **Proposing novel benchmarks** for OOD generalization and causal validity in multimodal biological data.  
2. **Providing an open-framework** (Causal Graph-Contrast) for standardized evaluation of cross-scale representation learning.  
3. **Enabling *in silico* virtual labs** where pretraining captures mechanistic relationships between interventions and phenotypes, reducing experimental validation costs.  

### Long-Term Impact  
The framework will accelerate therapeutic discovery by enabling *virtual screening* of drug candidates and predicting patient-specific responses to perturbations. It will also serve as a blueprint for integrating causal reasoning into multimodal foundation models in biology, addressing the community’s call for interpretable, generalizable representations.

---

This proposal outlines a rigorous plan to tackle critical gaps in biological representation learning, directly responding to the LMRL workshop’s challenge of building models that generalize across scales, modalities, and perturbations.