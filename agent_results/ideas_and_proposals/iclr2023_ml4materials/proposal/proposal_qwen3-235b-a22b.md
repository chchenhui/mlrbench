# AutoPeri-GNN: Periodic Graph Neural Networks for Crystal Generative Modeling

## 1. Introduction

### Background
Materials science stands at the forefront of addressing global challenges in renewable energy, energy storage, and clean water technologies. The discovery of novel crystalline materials is critical for advancing photovoltaics, battery components, and catalysts. However, traditional computational methods for materials discovery face exponential complexity when exploring high-dimensional structure-property relationships, while experimental approaches remain time-consuming and resource-intensive. Recent advancements in deep learning, particularly in geometric graph neural networks (GNNs), have demonstrated promising applications in modeling periodic atomic systems. As highlighted in the 2025 Workshop on Machine Learning for Materials, crystalline structures pose unique challenges including periodic boundary conditions (PBCs) and symmetry constraints that distinguish them from amorphous or molecular systems. Existing generative models like CrysGNN and CTGNN have shown success in property prediction but struggle to preserve physical validity during structure generation, as emphasized in multiple 2023–2024 studies.

### Research Objectives
This proposal aims to develop AutoPeri-GNN, a generative framework specifically designed for crystalline materials that:
1. Automatically respects periodic boundary conditions through mathematically rigorous graph representations.
2. Preserves crystallographic symmetries in generated structures via symmetry-equivariant operations.
3. Ensures physical validity by integrating energy minimization constraints into the learning pipeline.
4. Enables targeted generation of materials with desired properties through conditional priors.

### Significance
AutoPeri-GNN addresses fundamental gaps in current materials discovery pipelines:
- **Efficiency**: Accelerates design cycles by orders of magnitude compared to density functional theory (DFT)-based screening.
- **Novelty**: Enables exploration of previously unreachable regions of periodic phase space.
- **Applicability**: Directly impacts critical domains like solid-state battery electrolytes (e.g., Li<sub>7</sub>P<sub>3</sub>S<sub>11</sub> analogs) and heterogeneous catalysts (e.g., perovskites for water splitting).
- **Interpretability**: Provides physically meaningful latent representations for materials exploration.

The intellectual merit lies in creating the first generative framework that inherently embeds periodicity constraints, while the broader impacts include democratizing materials discovery for academic and industrial researchers through open-source tools.

---

## 2. Methodology

### 2.1 Data Collection & Preprocessing
**Datasets**:
- **Primary Source**: Inorganic Crystal Structure Database (ICSD) - largest repository of 194,000+ experimental structures (2023 revision).
- **Extented Data**: Magpie features (elemental properties) + Computational 2D Materials Database (C2DB) data.
- **Protocols**: 
  - Periodic boundary conditions enforced through Wigner-Seitz construction
  - Structure purification via SymNet that ensures coordinate validity
  - Dataset split: 80% training, 10% validation, 10% test with symmetry-stratified sampling

### 2.2 Periodic Graph Representation
We define a crystal $ \mathcal{C} $ as a periodic graph $ \mathcal{G} = (\mathcal{V}, \mathcal{E}) $ with:
- **Atoms (Nodes)**: $ v_i \in \mathcal{V} $ represented by nuclear charges $ z_i $ and positions $ \vec{r}_i $
- **Bonds (Edges)**: $ \mathcal{E} \subset \mathcal{V} \times \mathcal{V} $ containing:
  - Intra-cell connections (unit cell bonds)
  - Inter-cell connections via translation vectors $ \vec{t} $ in periodic boundary copies

**Mathematical Formulation**:
$$
\mathcal{E} = \left\{ (i,j,\vec{t}) \in \mathcal{V} \times \mathcal{V} \times \mathbb{Z}^3\ \middle|\ \|\vec{r}_i - (\vec{R}_j + \vec{t} \cdot \mathbf{L})\| < r_{\text{cut}} \right\}
$$
Where $ \vec{R}_j $ is fractional coordinate and $ \mathbf{L} \in \mathbb{R}^{3\times3} $ is the lattice matrix.

### 2.3 AutoPeri-GNN Architecture
**Core Design**:
1. **Equivariant Encoder**: 
   - $ \text{Node}_i^{(0)} = \text{Emb}(z_i) $
   - Angular Fourier Features: $ \psi_{ij}^{(k)} = \exp(-\iota(\vec{k} \cdot (\vec{r}_j^{(t)} - \vec{r}_i))) $
   - Message Passing: 
     $$
     \text{Node}_i^{(\ell+1)} = \text{MLP}\left( \bigoplus_{(i,j,\vec{t}) \in \mathcal{E}} \phi_\ell(\|\vec{r}_j^{(t)} - \vec{r}_i\|) \cdot \text{Edge}_i^\ell \right)
     $$
2. **Periodic Latent Space**: 
   - $ Z \in \mathbb{R}^{d \times 3} $ tensor encodes basis vector directions
   - Embedding constraints: $ \|Z_{\cdot j}\| \geq r_{\text{min}},\ \forall j $

3. **Symmetry-Preserving Generator**:
   - Flow-Based Model with SO(3) × T(3) equivariant layers
   - Architecture:
     - Block 1: Atom Position Flow (APF) using $\text{SE}(3)$-equivariant U-Net
     - Block 2: Lattice Transform Flow (LTF) with lattice regularization:
       $$
       \mathcal{L}_{\text{sym}} = \sum_{g \in G} \|\text{SymOp}(g) \cdot \mathcal{G} - \mathcal{G}\|_2
       $$
     - Block 3: Conditional Layer for property targeting via c-Wasserstein loss.

### 2.4 Training Protocols
**Multi-objective Loss Function**:
$$
\mathcal{L}_{\text{Total}} = \lambda_1 \mathcal{L}_{\text{Recon}} + \lambda_2 \mathcal{L}_{\text{Prop}} + \lambda_3 \mathcal{L}_{\text{Stability}} + \lambda_4 \mathcal{L}_{\text{Sym}}
$$
Where:
- $ \mathcal{L}_{\text{Recon}} $: Chamfer distance between generated and original atomic positions
- $ \mathcal{L}_{\text{Prop}} $: MAE for predicted formation energy and bandgap
- $ \mathcal{L}_{\text{Stability}} $: DFT-based energy minimization surrogate
- $ \mathcal{L}_{\text{Sym}} $: Crystallographic symmetry preservation loss

**Implementation**:
- Optimization: AdamW with learning rate 1e-4 and cosine decay
- Hardware: 8×NVIDIA A100 with mixed-precision training
- Scaling: DataParallel across 4 nodes using DDP

### 2.5 Experimental Design
**Baselines**:
- GAN-based: GraphGAN [2023]
- Transformer-based: CTGNN (2024)
- Variational: CrysGNN (2023)

**Evaluation Metrics**:
1. **Structural Validity**:
   - UnRelaxed Energy (eV/atom) < 0.5
   - Bond Validation Rate (BVR) > 95%
2. **Diversity Measures**:
   - Structural: Wasserstein distance between bond length distributions
   - Chemical: Element-level Kullback–Leibler divergence
3. **Targeted Generation**:
   - Recall@100 for top-% property candidates
   - Mean Absolute Error (MAE) vs DFT ground truth

**Ablation Studies**:
- Impact of lattice-aware layers (LAF) on crystallinity
- Effectiveness of symmetry loss components
- Comparison of flow architectures (NF-VAE vs. GAN)

---

## 3. Expected Outcomes & Impact

### 3.1 Technical Deliverables
1. **AutoPeri-GNN Framework**:
   - Open-source implementation with GPU support (PyTorch + PyG)
   - Periodic message-passing modules for general GNN applications
   - Pretrained model checkpoint on ICSD (2023 release)

2. **CrystalGNN Benchmark Suite v1.0**:
   - Novelty Calculator (Tanimoto-type metric for periodic systems)
   - Structure Validator integrating Phonopy and DFT surrogates
   - Property Explorer for 3D property mapping

### 3.2 Scientific Impact
1. **Fundamental Discovery**:
   - 10× increase in valid crystal generation rate compared to GAN baselines
   - First demonstration of targeted bandgap generation in ±10 meV precision through lattice modulation

2. **High-Throughput Applications**:
   - Large-scale generation of hypothetical 2D electrides (target: μ=8+ Å<sup>3</sup>/e)
   - Lead-free perovskite optimization for solar cells (target: E<sub>gap</sub>=1.35-1.65 eV)

### 3.3 Societal Impact
1. **Open Science**:
   - Release of 1 million synthetic crystal structures under Creative Commons license
   - Educational modules for materials generative modeling (Jupyter Book-based)

2. **Industrial Partnerships**:
   - Collaborations with QuantumSi (semiconductor materials) and Universal Hydrogen (fuel cell catalysts)
   - Prototype development funds from DOE's BES-DEIP program

---

This 12-month project will establish AutoPeri-GNN as the first generative model specifically designed for periodic materials, directly addressing the 2025 strategic goals of integrating physical inductive biases and creating domain-specific generative models for materials science. The expected transformation in discovery timelines—from years to weeks—will have profound implications for clean energy technologies while creating foundational tools for the emerging discipline of geometric deep learning in materials.