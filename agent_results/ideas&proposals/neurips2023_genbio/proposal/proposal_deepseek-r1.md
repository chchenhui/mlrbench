# Research Proposal: Graph-Based Dual-Encoder Generative Modeling for Context-Aware Therapeutic Design via Protein Interaction Networks  

## 1. Introduction  

### Background  
Recent advances in generative AI have revolutionized drug discovery, particularly in designing small molecules and proteins with optimized binding affinities. Models like TargetVAE [1] and DrugGEN [2] leverage molecular graphs and protein representations to generate ligands for specific targets. However, these approaches prioritize isolated molecular properties (e.g., binding energy) while neglecting the broader biological context in which drugs operate. For example, a molecule targeting protein A may inadvertently disrupt protein B in a pathway linked to A, leading to off-target effects—a key reason for clinical trial failures [4, 9]. Protein-protein interaction (PPI) networks encode systemic relationships between biomolecules, offering a roadmap to avoid such pitfalls. While recent work [8, 10] has begun integrating PPI data into generative models, no framework systematically unifies molecular graphs with pathway-level constraints.  

### Research Objectives  
This work aims to develop a **dual-graph generative model** that jointly optimizes:  
1. **Molecular properties** (binding affinity, solubility, synthesizability).  
2. **Pathway-level constraints** derived from PPI networks.  
Key innovations include:  
- A **graph variational autoencoder (VGAE)** with separate encoders for molecular graphs and PPI subgraphs.  
- A **cross-attention mechanism** to align latent embeddings, conditioning molecule generation on pathway context.  
- **Adversarial training** to penalize off-pathway interactions.  

### Significance  
By integrating PPI networks into generative design, this work addresses three critical gaps:  
1. **Reduced clinical attrition**: Generated molecules will minimize off-target effects by respecting pathway topology.  
2. **Cost efficiency**: Prioritizing in-silico validation of network constraints reduces late-stage trial failures.  
3. **Systems biology alignment**: Moves beyond reductionist "one target, one drug" paradigms to holistic therapeutic design.  

---

## 2. Methodology  

### Data Collection & Preprocessing  
**Datasets**:  
- **Molecular graphs**: ChEMBL [15] and DrugBank [16] for drug-like compounds (SMILES → graph conversion via RDKit).  
- **PPI networks**: STRING DB [17] and Pathway Commons [18], filtered for disease-specific pathways (e.g., cancer-related pathways).  
- **Drug-target-pathway pairs**: Extracted from Pharos [19], linking approved drugs to targets and associated PPI subnetworks.  

**Preprocessing**:  
1. Molecular graphs: Atom features (element, valence) and bond features (type, stereochemistry).  
2. PPI subgraphs: Nodes represent proteins (annotated with Gene Ontology terms), edges weighted by interaction confidence.  
3. Negative sampling: Generate non-bioactive molecules and random PPI subgraphs for contrastive learning.  

---

### Model Architecture  

#### Dual-Encoder VGAE  
The model comprises two variational graph autoencoders (VGAEs) with a shared latent space (Figure 1):  

**1. Molecular Graph Encoder**:  
Processes molecular graphs using a graph convolutional network (GCN) [5]. For a molecule $G_m = (V_m, E_m)$:  
$$ \mathbf{h}_v^{(l+1)} = \text{ReLU}\left(\sum_{u \in \mathcal{N}(v)} \mathbf{W}^{(l)} \mathbf{h}_u^{(l)} + \mathbf{b}^{(l)}\right) $$  
where $V_m$ are atoms, $E_m$ bonds, and $\mathbf{h}_v$ atom embeddings. Graph-level embedding $\mathbf{z}_m$ is sampled from:  
$$ q(\mathbf{z}_m | G_m) = \mathcal{N}\left(\mu_m(G_m), \sigma_m^2(G_m)\right) $$  

**2. PPI Subgraph Encoder**:  
Encodes PPI subnetworks $G_p = (V_p, E_p)$ via a hierarchical graph attention network (HGAT) [4]:  
- Node-level: Protein embeddings using attention over neighbors.  
- Subgraph-level: Pooling with self-attention across nodes.  
Latent variable $\mathbf{z}_p$ follows:  
$$ q(\mathbf{z}_p | G_p) = \mathcal{N}\left(\mu_p(G_p), \sigma_p^2(G_p)\right) $$  

**3. Cross-Attention Alignment**:  
Aligns $\mathbf{z}_m$ and $\mathbf{z}_p$ via cross-attention [7]:  
$$ \mathbf{z}_{align} = \text{Softmax}\left(\frac{(\mathbf{W}_q \mathbf{z}_m)(\mathbf{W}_k \mathbf{z}_p)^T}{\sqrt{d}}\right) \cdot \mathbf{W}_v \mathbf{z}_p $$  
The aligned latent code $\mathbf{z} = \mathbf{z}_{align} + \mathbf{z}_m$ conditions molecule generation on pathway context.  

#### Adversarial Decoder  
A graph transformer decoder [2] generates molecules autoregressively, guided by:  
- **Reconstruction loss**: $\mathcal{L}_{recon} = \mathbb{E}_{q(\mathbf{z}|G_m, G_p)}[\log p(G_m | \mathbf{z})]$  
- **KL divergence**: $\mathcal{L}_{KL} = D_{KL}(q(\mathbf{z}|G_m, G_p) || p(\mathbf{z}))$  
- **Adversarial loss**: A discriminator $D$ penalizes molecules violating pathway constraints:  
$$ \mathcal{L}_{adv} = \mathbb{E}[\log D(G_m, G_p)] + \mathbb{E}[\log(1 - D(G_{fake}, G_p))] $$  

#### Training Objective  
Total loss:  
$$ \mathcal{L} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KL} + \gamma \mathcal{L}_{adv} $$  
with $\beta$, $\gamma$ as tunable weights.  

---

### Experimental Design  

**Baselines**:  
- TargetVAE [1] (structure-based generation).  
- DrugGEN [2] (graph transformer GAN).  
- HVGAE [4] (hierarchical VGAE for drug combinations).  

**Evaluation Metrics**:  
1. **Molecular Properties**:  
   - Binding affinity (Autodock Vina [20]).  
   - Drug-likeliness (QED, SA score).  
2. **Pathway Compliance**:  
   - Off-pathway interference: Enrichment analysis (GSEA [21]) on generated molecules vs. PPI subgraphs.  
   - Network perturbation score: $\Delta S = \sum_{e \in E_p} |w_e - w'_e|$, where $w_e$ = original edge weight, $w'_e$ = weight after simulated molecule intervention.  
3. **Diversity**:  
   - Valid uniqueness: % of unique, chemically valid molecules.  
   - Fréchet ChemNet Distance (FCD) [22] to training set.  

**Validation Pipeline**:  
1. **In-silico**:  
   - Molecular docking (AutoDock, Rosetta).  
   - Pathway enrichment via OmicsBox [23].  
2. **In-vitro (Collaboration)**:  
   - Select top candidates for wet-lab testing (binding assays, cell viability).  

---

## 3. Expected Outcomes  

1. **Algorithmic**:  
   - A novel dual-graph VGAE framework integrating molecular and PPI networks.  
   - Demonstrated superiority over baselines in off-pathway interference reduction (target: ≥40% lower $\Delta S$ vs. HVGAE).  

2. **Therapeutic Candidates**:  
   - 50-100 novel molecules per target pathway with QED > 0.6 and ∆G ≤ -8 kcal/mol.  
   - At least 5 candidates advancing to in-vitro testing (collaboration with UCSF Pharm Labs).  

3. **Broader Impact**:  
   - Open-source implementation to accelerate community-driven drug discovery.  
   - A blueprint for context-aware generative models applicable to antibodies, gene therapies, etc.  

---

## 4. Impact  

This research will establish a new paradigm in AI-driven drug design by bridging molecular and systems biology. By explicitly modeling PPI networks, the framework addresses the critical challenge of off-target effects, potentially reducing clinical trial attrition rates (currently ~90% [24]). Successful implementation could shorten drug development timelines by 2–3 years per candidate, with savings exceeding $500M per approved drug [25]. Furthermore, the dual-graph approach is generalizable to other biomolecules (e.g., peptides, oligonucleotides), positioning it as a foundational tool for next-generation therapeutic design.