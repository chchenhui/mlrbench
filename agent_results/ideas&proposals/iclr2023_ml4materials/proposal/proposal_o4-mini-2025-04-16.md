Title  
AutoPeri-GNN: A Periodic Equivariant Graph‐Based Generative Framework for Crystalline Materials Discovery  

Introduction  
Background. The discovery of novel crystalline materials underpins advances in clean energy, catalysis, electronics and beyond. Traditional computational approaches—density functional theory (DFT), molecular dynamics and high‐throughput screening—often require substantial compute resources and human intervention. Machine learning (ML), particularly geometric deep learning, has transformed molecular and protein modeling, yet crystalline materials pose unique challenges. Unlike small molecules or proteins, crystals exist in the condensed phase under periodic boundary conditions (PBCs), and their fundamental “unit cell” representation requires careful treatment of lattice vectors, symmetry operations and infinite‐repetition constraints. Generative models that ignore PBCs often produce artifacts—broken lattices, spurious coordination environments or structures that violate space‐group symmetries.

Literature survey. Recent works (Self-Supervised Generative Models for Crystal Structures, Liu et al., 2023; CrysGNN, Das et al., 2023; CTGNN, Du et al., 2024; AnisoGNN, Hu & Latypov, 2024) have made significant progress in property prediction and generation of crystalline materials. However, existing generative frameworks often adopt either GAN‐based discriminators without explicit periodic losses or VAEs that do not fully encode lattice periodicity in latent spaces. Flow‐based methods under crystal symmetry groups remain largely unexplored. Key challenges include: (1) embedding PBCs into graph representations; (2) ensuring generated unit cells satisfy physical stability and energy minima; (3) leveraging symmetries and equivariance in invertible generative flows.  

Research objectives. We propose AutoPeri-GNN, a novel generative deep learning framework that:  
 • Represents crystals as graphs with explicit PBC handling and node features for atomic types, fractional coordinates and lattice vectors.  
 • Employs an E(3)-equivariant graph neural network encoder to map input crystals into a latent space structured as a toroidal manifold (to capture periodicity).  
 • Utilizes a symmetry-preserving normalizing flow as decoder, generating valid unit cells that respect target space groups and energy constraints.  
 • Incorporates differentiable physics‐inspired loss terms (e.g., formation energy, radial distribution fidelity) to bias generation toward stable crystals.  

Significance. AutoPeri-GNN will advance materials discovery by dramatically improving the quality and diversity of machine‐generated crystal structures under real‐world periodic constraints. Compared to prior generative models, it will accelerate screening of candidate materials with tailored properties (e.g., low formation energy, desired bandgap), reduce reliance on expensive DFT relaxations and open the door to automated design of complex inorganic crystals and alloys.

Methodology  
1. Data Collection and Preprocessing  
 a. Data sources: We will assemble a training corpus from the Materials Project (MP), Open Quantum Materials Database (OQMD) and Crystallography Open Database (COD), aggregating ~200 000 CIF files. Each entry includes atomic species, fractional coordinates $\{c_i\in[0,1)^3\}$, lattice vectors $L\in\mathbb{R}^{3\times3}$ and computed properties (formation energy $E_f$, bandgap $E_g$).  
 b. Preprocessing pipeline:  
  • Parse CIF files with pymatgen; normalize all crystals to primitive cells via symmetry analysis.  
  • Standardize lattice volumes to within a pre‐specified range (e.g., $[50, 500]\,\text{\AA}^3$) to avoid extreme cell shapes.  
  • Compute neighbor lists under PBCs: for each atom $i$, find neighbors $j$ in the central cell or periodic images such that $d_{ij}=\|\Delta\mathbf{r}_{ij}+L\mathbf{n}\|\le r_{\rm cut}$ (with $r_{\rm cut}=8\,\text{\AA}$ and $\mathbf{n}\in\{-1,0,1\}^3$).  
  • Node features: atomic number $Z_i$, one‐hot species embedding, fractional coordinate embedding via sine/cosine (for each dimension $k$):  
$$
s_{ik} = \bigl[\sin(2\pi c_{ik}),\;\cos(2\pi c_{ik})\bigr].
$$  
  • Edge features: radial basis expansion of periodic distance $d_{ij}$:  
$$
e_{ij}^{(m)} = \exp\Bigl(-\tfrac{(d_{ij}-\mu_m)^2}{2\sigma^2}\Bigr),
$$  
with $\{\mu_m\}$ fixed centers and $\sigma=0.5\,\text{\AA}$.  

2. Model Architecture  
2.1 Encoder: Periodic Equivariant GNN  
 We adopt an E(3)-equivariant message‐passing network (MPN) that updates atom embeddings $h_i\in\mathbb{R}^d$ and coordinates $\mathbf{r}_i$ while preserving rotational and translational symmetry under PBCs. At layer $l$:  
$$
\begin{aligned}
m_{ij}^{(l)} &= \phi_e\bigl(h_i^{(l)}, h_j^{(l)}, e_{ij}\bigr),\\
h_i^{(l+1)} &= h_i^{(l)} + \sum_{j\in\mathcal{N}(i)} m_{ij}^{(l)},\\
\mathbf{r}_i^{(l+1)} &= \mathbf{r}_i^{(l)} + \sum_{j\in\mathcal{N}(i)} \frac{\psi_r(h_i^{(l)}, h_j^{(l)})}{\|\Delta \mathbf{r}_{ij}\|^2}\,\Delta \mathbf{r}_{ij},
\end{aligned}
$$  
where $\phi_e,\psi_r$ are learned MLPs; $\Delta\mathbf{r}_{ij} = \bigl(\mathbf{r}_i - \mathbf{r}_j + L\mathbf{n}_{ij}\bigr)$ includes the periodic image offset minimizing distance. After $L$ layers, we pool node embeddings with a crystal‐level readout to obtain global summary $h_G\in\mathbb{R}^d$.  

2.2 Latent Representation with Periodicity  
 We split the latent code $z=(z_G,z_L)$, where $z_G\sim\mathcal{N}(\mu_G,\Sigma_G)$ captures global composition/structure and $z_L\sim\mathcal{T}^3$ (a 3D torus) encodes lattice vectors. Concretely,  
$$
q(z_G|\mathcal{G})=\mathcal{N}\bigl(\mu_G(h_G),\Sigma_G(h_G)\bigr),\quad
z_L = \mathrm{atan2}\bigl(s(L)\bigr)\in[0,2\pi)^3,
$$  
where $s(L)$ is a fixed mapping from lattice vectors to angles (via Gram matrix decomposition).  

2.3 Decoder: Symmetry‐Preserving Normalizing Flow  
 We design a flow model $f_{\theta}$ that maps base latent $u\sim\mathcal{N}(0,I)$ to $(z_G',z_L')$ and then to a crystal graph. The flow comprises $K$ coupling layers, each respecting space‐group operations $\mathcal{S}$ of common crystal families:  
$$
u^{(0)}=u,\quad u^{(k+1)} = \text{Coupling}_k\bigl(u^{(k)},\mathcal{S}\bigr),\quad
(z_G',z_L') = u^{(K)}.
$$  
We then decode $z_G'$ into atom counts and species probabilities via a deconvolution network, while $z_L'$ is mapped back to lattice vectors $L'=g_L(z_L')$. Atomic coordinates $\{\hat c_i\}$ and bond connectivity $\hat E$ are predicted by a transformer conditioned on $z_G',L'$.  

3. Loss Functions and Training Objective  
Our total loss is  
$$
\mathcal{L} = \lambda_{\rm VAE}\bigl(\mathcal{L}_{\rm rec} + \beta\,\mathrm{KL}(q(z_G|\mathcal{G})\|p(z_G))\bigr)
+ \lambda_{\rm flow}\,\mathcal{L}_{\rm flow}
+ \lambda_{\rm phys}\,\mathcal{L}_{\rm phys},
$$  
where:  
 • Reconstruction $\mathcal{L}_{\rm rec}$ measures graph edit distance and coordinate error:  
$$
\mathcal{L}_{\rm rec} = \sum_{i,j}\bigl\|e_{ij}-\hat e_{ij}\bigr\|^2 \;+\;\sum_i \|\mathbf{r}_i-\hat{\mathbf{r}}_i\|^2.
$$  
 • $\mathrm{KL}$ is the KL divergence between $q(z_G|\mathcal{G})$ and the standard Gaussian prior $p(z_G)$.  
 • Flow loss $\mathcal{L}_{\rm flow} = -\sum_k\log\bigl|\det \partial u^{(k+1)}/\partial u^{(k)}\bigr| - \log p(u)\,$ encourages high likelihood under the flow.  
 • Physical loss $\mathcal{L}_{\rm phys}$ enforces energy and structural stability:  
   – Formation‐energy penalty:  
   $$
   L_E = \bigl|E_f(\hat{\mathcal{G}})-E_f(\mathcal{G})\bigr|,
   $$  
   computed via a pre‐trained energy surrogate network.  
   – Radial distribution fidelity:  
   $$
   L_{\rm RDF} = \sum_{r}\bigl|g_{\mathcal{G}}(r)-g_{\hat{\mathcal{G}}}(r)\bigr|,
   $$  
   where $g(r)$ is the pair distribution function under PBCs.  
Hyperparameters $\{\lambda,\beta\}$ are selected by grid search on a validation split.

4. Experimental Design and Evaluation  
4.1 Baselines  
 • CGVAE (Crystal Graph VAE) without explicit PBC losses.  
 • GAN‐based crystal generators (Liu et al., 2023).  
 • Simple VAE + decoder ignoring lattice periodicity.  

4.2 Datasets and Splits  
 We will use a train/validation/test split of 70%/15%/15% on MP + OQMD combined, ensuring no overlapping compositions.  

4.3 Metrics  
 Generative quality will be measured by:  
  • Validity: fraction of outputs that satisfy charge neutrality and coordination rules.  
  • Uniqueness & Novelty: proportion of generated crystals not in the training set and not duplicates.  
  • Coverage: the fraction of test‐set compositions represented.  
  • Distribution matching: Frechét Crystal Distance (FCD) computed on formation energy and lattice distributions.  
  • DFT validation: for a random subset of 100 generated structures, run single‐point DFT to assess formation energy error and structural relaxation success.  

4.4 Ablation Studies  
 We will systematically disable: (1) periodic coordinate embeddings, (2) equivariant layers, (3) physical loss terms, to quantify their individual contributions to generative performance.  

Expected Outcomes & Impact  
1. High‐Quality Crystal Generation. We anticipate AutoPeri-GNN will achieve >95% validity and >80% novelty while reducing FCD by 30% relative to state‐of‐the‐art crystal VAEs. The inclusion of physics‐inspired losses will lead to a 50% reduction in DFT relaxation failures on generated structures.  
2. Accelerated Materials Discovery. By integrating energy surrogates into the generation loop, AutoPeri-GNN will enable “goal‐directed” sampling: e.g., targeting low formation energy or specific bandgaps. We expect a 10× speed‐up over combinatorial DFT searches for candidate identification.  
3. Open‐Source Release. We will release code, pre‐trained models and generated datasets under an MIT license, facilitating community benchmarking and downstream applications.  
4. Scientific Impact. AutoPeri-GNN’s explicit treatment of PBCs and equivariant flows will provide a blueprint for generative modeling across other periodic systems—polymers, surfaces and nanoporous frameworks—broadening the scope of ML‐driven materials design.  
5. Long‐Term Vision. By coupling AutoPeri-GNN with automated synthesis and high‐throughput characterization platforms, we foresee fully closed‐loop discovery workflows, driving accelerated innovation in energy materials, catalysis and beyond.  

This research plan addresses the critical gap in crystalline generative modeling by embedding periodic boundary conditions and physical priors into a unified equivariant GNN plus flow framework. Successful implementation will empower materials scientists with an autonomous, scalable generator of novel, physically sound crystal structures.