**1. Title**  
AutoPeri-GNN: A Periodic-Aware Generative Framework for Crystal Structure Discovery via Equivariant Graph Neural Networks  

---

**2. Introduction**  

**Background**  
The discovery of novel crystalline materials is critical for advancing renewable energy technologies, energy storage systems, and catalysis. Traditional methods for materials discovery rely heavily on trial-and-error experimentation or computationally intensive density functional theory (DFT) simulations, which are slow and resource-heavy. Machine learning (ML) has shown promise in accelerating this process, but existing generative models, largely developed for molecules and proteins, fail to address materials-specific challenges. Crystalline materials exhibit **periodic boundary conditions** (PBCs) that necessitate specialized representations and symmetry-aware architectures, posing unique hurdles for ML models.  

**Research Objectives**  
This proposal aims to develop **AutoPeri-GNN**, a generative framework that explicitly models the periodic structure of crystalline materials. The objectives are:  
1. To design a periodic-aware graph neural network (GNN) architecture for encoding and generating valid crystal structures under PBCs.  
2. To integrate physical constraints (e.g., energy minimization, symmetry operations) as differentiable losses to ensure structural stability.  
3. To benchmark the framework against state-of-the-art models on property prediction, structural validity, and generation diversity.  

**Significance**  
Addressing the challenges of PBCs and physical validity in crystal generation will unlock new possibilities for rapid materials discovery. By automating the design of stable, property-targeted crystals, AutoPeri-GNN could accelerate the development of next-generation batteries, photovoltaics, and catalysts. The framework will also contribute open-source tools and datasets to the materials informatics community.  

---

**3. Methodology**  

**3.1 Data Collection and Preprocessing**  
- **Datasets**: Leverage the **Materials Project** database, which contains over 150,000 experimentally and computationally validated crystal structures with properties like formation energy, bandgap, and elastic tensors. Additional data from the **Open Quantum Materials Database (OQMD)** and **Crystallography Open Database (COD)** will be used.  
- **Preprocessing**:  
  - Represent crystals as **periodic graphs** with nodes (atoms) and edges (bonds) defined under PBCs. Edge connections account for interactions between atoms in neighboring unit cells.  
  - Normalize atom coordinates to fractional positions within the unit cell to preserve periodicity.  
  - Augment data using symmetry operations (e.g., space group transformations) to improve model robustness.  

**3.2 Model Architecture**  
AutoPeri-GNN comprises a **periodic-equivariant graph autoencoder** paired with a **flow-based generative model**.  

**3.2.1 Encoder: Periodic-Equivariant GNN**  
- **Graph Construction**: Each crystal is represented as a graph $G = (V, E, \mathbf{L})$, where nodes $v_i \in V$ correspond to atoms, edges $e_{ij} \in E$ represent atomic interactions across periodic boundaries, and $\mathbf{L} = [\mathbf{a}, \mathbf{b}, \mathbf{c}]$ denotes the lattice matrix of the unit cell.  
- **Equivariant Message Passing**: Inspired by E(n)-equivariant GNNs, the encoder updates node embeddings $\mathbf{h}_i$ and edge features $\mathbf{e}_{ij}$ using:  
  $$ 
  \mathbf{m}_{ij} = \phi_m\left(\mathbf{h}_i, \mathbf{h}_j, \|\mathbf{r}_j - \mathbf{r}_i + \mathbf{L}\mathbf{n}_{ij}\|^2\right), \quad \mathbf{h}_i' = \phi_h\left(\mathbf{h}_i, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}\right),
  $$  
  where $\mathbf{n}_{ij} \in \mathbb{Z}^3$ encodes the periodic shifts between atoms $i$ and $j$, and $\phi_m, \phi_h$ are MLPs.  
- **Latent Space**: The encoder outputs a latent vector $\mathbf{z} \in \mathbb{R}^d$ that encodes both atomic configurations and lattice parameters $\mathbf{L}$.  

**3.2.2 Decoder: Flow-Based Generation**  
The decoder uses a **continuous normalizing flow** to map latent vectors $\mathbf{z}$ to crystal structures:  
$$
\mathbf{z} = f_\theta(\mathbf{x}), \quad \log p_\theta(\mathbf{x}) = \log p(\mathbf{z}) + \int_0^1 \mathrm{Tr}\left(\frac{\partial f_\theta}{\partial \mathbf{x}(t)}\right) dt,
$$  
where $f_\theta$ is an invertible transformation conditioned on lattice parameters. Physical constraints (e.g., minimum bond lengths) are enforced via masked transformations.  

**3.2.3 Physical Constraints as Differentiable Losses**  
- **Energy Minimization**: A pretrained property predictor (e.g., formation energy estimator) provides a differentiable loss:  
  $$
  \mathcal{L}_{\text{energy}} = \mathbb{E}_{\mathbf{x} \sim p_\theta}\left[\hat{E}(\mathbf{x})\right],
  $$  
  where $\hat{E}$ is a GNN trained on DFT-computed energies.  
- **Symmetry Preservation**: Penalize deviations from space group symmetries using a symmetry distance metric:  
  $$
  \mathcal{L}_{\text{sym}} = \sum_{g \in \mathcal{G}} \|g(\mathbf{x}) - \mathbf{x}\|^2,
  $$  
  where $\mathcal{G}$ is the set of symmetry operations for the crystal’s space group.  

**3.3 Experimental Design**  
- **Baselines**: Compare against CrysGNN, CTGNN, and CGCNN for property prediction, and the GAN-based model from Liu et al. (2023) for generation.  
- **Evaluation Metrics**:  
  - **Validity**: Percentage of generated structures that satisfy crystallographic rules (e.g., Wyckoff positions).  
  - **Stability**: Formation energy (eV/atom) and phonon stability (via DFT validation).  
  - **Diversity**: Coverage of chemical space using metrics like COV and MMD.  
  - **Property Accuracy**: Mean absolute error (MAE) in predicting bandgap and elastic moduli.  
- **Training Protocol**:  
  - Train the autoencoder on 80% of the Materials Project data; use 20% for testing.  
  - Jointly optimize reconstruction loss ($\mathcal{L}_{\text{recon}}$), physical losses ($\mathcal{L}_{\text{energy}}, \mathcal{L}_{\text{sym}}$), and adversarial loss (if applicable).  

---

**4. Expected Outcomes & Impact**  

**Expected Outcomes**  
1. AutoPeri-GNN will generate crystalline structures with **>90% validity** (vs. <60% for existing models like Liu et al., 2023) by explicitly modeling PBCs.  
2. The framework will achieve **<0.1 eV/atom MAE** in formation energy prediction, outperforming CTGNN (0.15 eV/atom).  
3. Generated structures will exhibit **higher diversity** (COV >80%) while maintaining stability, enabling exploration of previously unreported crystal phases.  

**Impact**  
By solving the fundamental challenge of periodic boundary conditions in generative models, AutoPeri-GNN will drastically reduce the time and cost of materials discovery. The framework’s integration with high-throughput DFT workflows will enable rapid screening of candidates for renewable energy applications. Open-sourcing the model and datasets will foster collaboration across materials science and ML communities, accelerating progress toward global sustainability goals.  

---  

**5. Conclusion** (Summary of sections 3–4 for completeness)  
AutoPeri-GNN addresses the critical need for periodicity-aware generative models in computational materials science. Through its equivariant architecture and physics-informed training, the framework bridges the gap between geometric deep learning and crystallography, offering a robust tool for next-generation materials discovery. Successful implementation will pave the way for ML-driven breakthroughs in energy storage, catalysis, and beyond.