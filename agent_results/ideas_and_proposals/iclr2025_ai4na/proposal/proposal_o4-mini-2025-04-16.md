1. Title  
DiffuNA: Diffusion‐Powered Generative Design of RNA Therapeutics

2. Introduction  
2.1 Background  
RNA molecules—aptamers, ribozymes, siRNAs—have emerged as versatile therapeutic agents due to their ability to bind targets with high specificity, catalyze reactions, or modulate gene expression. However, de novo design of functional RNA typically relies on high‐throughput screening and experimental trial‐and‐error, incurring substantial time and cost. Recent advances in AI for small molecules and proteins (e.g., SE(3)‐equivariant diffusion models) demonstrate the promise of generative frameworks to navigate large chemical or sequence‐structure spaces. In parallel, methods such as RiboDiffusion and trRosettaRNA have improved our ability to predict RNA structure or invert structure to sequence, but a unified generative framework for joint sequence‐structure design of novel RNA therapeutics remains lacking.

2.2 Research Objectives  
This proposal aims to develop DiffuNA, a 3D graph‐based diffusion model that:  
• Learns the joint distribution of RNA sequence, secondary and tertiary structure from public datasets (PDB, SHAPE reactivity).  
• Enables conditional generation of RNA molecules tailored to user‐specified binding pockets or structural scaffolds.  
• Integrates a reinforcement‐learning (RL) refinement loop to optimize folding stability and binding affinity in silico.  

2.3 Significance  
By coupling a principled generative diffusion backbone with physics‐informed RL rewards, DiffuNA promises:  
• Acceleration of lead discovery for RNA therapeutics.  
• Exploration of novel sequence‐structure motifs beyond known families.  
• Transferable methodology applicable to diverse RNA targets and potential extension to DNA or protein–RNA complexes.  

3. Methodology  
3.1 Data Collection and Preprocessing  
• Structural Data: Extract high‐resolution RNA 3D structures (resolution < 3.0 Å) from the PDB and RNAcentral. For each RNA chain, record atomic coordinates, base‐pair annotations, and tertiary contacts.  
• Reactivity Profiles: Incorporate SHAPE reactivity from databases such as RMDB to inform flexible vs. rigid regions.  
• Pocket Templates: Curate a small‐molecule or protein‐RNA binding pocket library from PDB complexes to serve as conditional scaffolds.  
• Preprocessing Pipeline:  
  1. Parse each RNA into a graph $G=(V,E)$ where $V$ are nucleotides and $E$ includes (i) sequential edges, (ii) annotated base‐pairs, and (iii) proximity edges ($\|x_i-x_j\|<5\,$Å).  
  2. Represent each node $i$ by features $(s_i,\rho_i)$: one‐hot encoding $s_i\in\{A,C,G,U\}$ and SHAPE reactivity $\rho_i$. Store $3$D coordinates $x_i\in\mathbb{R}^3$.  
  3. Split into train/validation/test sets by family to evaluate generalization to novel folds.

3.2 Diffusion Model Architecture  
3.2.1 Forward Diffusion on Graphs  
We treat the continuous geometry and discrete sequence jointly. Let $z^0=(X^0,S^0)$ be the original data, with $X^0\in\mathbb{R}^{n\times3}$ coordinates and $S^0\in\{0,1\}^{n\times4}$ one‐hot sequence. We define a noise schedule $\{\beta_t\}_{t=1}^T$ and a forward process:  
$$q(X^t\mid X^{t-1})=\mathcal{N}\bigl(X^t;\sqrt{1-\beta_t}\,X^{t-1},\,\beta_t I\bigr)$$  
For the discrete sequence, we apply multinomial diffusion:  
$$q(S^t\mid S^{t-1})=(1-\alpha_t)S^{t-1}+\alpha_t\text{Uniform}$$  
where $\alpha_t$ is a small corruption probability schedule.

3.2.2 Reverse Denoising with SE(3)‐Equivariant Graph Neural Network  
We parameterize the reverse model $p_\theta(z^{t-1}\mid z^t)$ via an SE(3)‐equivariant GNN that jointly predicts:  
• A vector field $\mu_\theta(X^t,t)$ for coordinate denoising.  
• A categorical distribution $\pi_\theta(S^{t-1}\mid z^t)$ for sequence denoising.  

At each step:  
1. Encode $z^t$ with a layered GNN that mixes message passing on edges $E$ and updates scalar/node features and relative positions in an equivariant manner (e.g., by using spherical harmonics or invariant point attention).  
2. Predict denoising targets $(\hat\epsilon_X,\hat S)$, i.e., geometry noise and original one‐hot.  
3. Obtain parameters:  
   $$\mu_\theta(X^t,t) = \frac{1}{\sqrt{1-\beta_t}}\Bigl(X^t - \beta_t\,\hat\epsilon_X\Bigr),\quad 
     \pi_\theta(S^{t-1}\mid z^t)=\mathrm{softmax}(\mathrm{MLP}(\mathrm{gnn\_features}))$$  

3.2.3 Training Objective  
The loss is a weighted sum of continuous and discrete denoising losses:  
$$\mathcal{L}=\mathbb{E}_{t,X^0,S^0,\epsilon_S,\epsilon_X}\Bigl[\|\epsilon_X - \hat\epsilon_X\|^2 + \lambda\,\mathrm{CE}(S^0,\hat S)\Bigr]$$  
where $\mathrm{CE}$ is cross‐entropy and $\lambda$ balances geometry vs. sequence.

3.3 Conditioning on Binding Pockets or Scaffolds  
At inference, users supply a pocket template $P$ (either coordinates or a learned embedding). We encode $P$ via the same GNN backbone and cross‐attention into the denoising network at each timestep, enabling conditional generation.  

3.4 Reinforcement Learning‐Based Refinement  
To bias generated candidates towards high stability and affinity, we embed an RL loop during sampling:  
• State $s$ is the partially denoised sample after $t=0$.  
• Action $a$ selects perturbed $z^0$ from a shortlist of $k$ samples.  
• Reward $R(s)$ combines:  
  1. Folding stability score $F(s)$ from a pretrained folding predictor (e.g., trRosettaRNA‐based).  
  2. Binding affinity surrogate $B(s,P)$ from a light‐weight docking network.  
  $$R(s)=\lambda_1\,F(s)+\lambda_2\,B(s,P)$$  
We apply Proximal Policy Optimization (PPO) to learn a policy $\pi_\phi(a\mid s)$ that resamples or perturbs candidates to maximize expected reward.  

3.5 Experimental Design and Evaluation  
Benchmarks:  
- Thrombin‐binding aptamers (PDB IDs: 4DII, 4DIL).  
- Hammerhead ribozymes (Rfam family RF00008).  

Baselines: RiboDiffusion, DiffSBDD (adapted for RNA), an MLP‐VAE on secondary structure, a conditional Transformer.  

Metrics:  
• Sequence novelty: percent identity to nearest train example.  
• Structural accuracy: TM‐score of generated fold vs. experimentally determined (for validation set).  
• Predicted binding affinity: docking score via Autodock‐Vina.  
• Folding stability: predicted free energy $\Delta G$.  
• Diversity: pairwise RMSD among top‐N candidates.  

Ablation studies:  
1. Remove RL loop (pure diffusion).  
2. Omit sequence diffusion (structure‐only).  
3. Vary diffusion steps $T\in\{50,100,200\}$.  
4. Compare soft vs. hard conditioning on pocket.

3.6 Implementation Details  
• Framework: PyTorch + e3nn for equivariant layers.  
• Training: 8 × NVIDIA A100 GPUs, batch size 16, $T=100$ steps, AdamW optimizer, learning rate $10^{-4}$.  
• Runtime: ~3 weeks for full training. Code and pretrained weights to be publicly released.

4. Expected Outcomes & Impact  
4.1 Expected Outcomes  
• A validated generative platform (DiffuNA) capable of producing novel RNA sequences that fold into high‐affinity 3D structures for user‐specified pockets.  
• Demonstrated improvement over state‐of‐the‐art in aptamer and ribozyme benchmarks: higher novelty (>30% IDR), better predicted docking scores (≥1 kcal/mol gain), and stability.  
• Open‐source library and pretrained models for the AI4NA community.

4.2 Scientific Impact  
• Establishes a new paradigm for joint sequence‐structure RNA design via diffusion.  
• Provides insights into sequence–structure landscapes of functional RNAs.  
• Bridges AI and nucleic acid research, motivating further cross‐disciplinary work at AI4NA.

4.3 Practical Impact  
• Accelerates lead‐candidate generation in RNA therapeutics, reducing wet‐lab cycles.  
• Enables rapid prototyping of aptamers for diagnostic assays and ribozymes for gene editing.  
• Forms the foundation for future extensions to multi‐modal design (RNA–protein complexes, DNA nanostructures).

5. Timeline & Milestones  
Month 1–2: Data curation, graph‐pipeline development, initial GNN architecture prototyping  
Month 3–5: Training the unconditional DiffuNA model; baseline evaluations  
Month 6–7: Implement conditional pocket encoding; refine sampling loop  
Month 8–9: Integrate RL module; tune reward weights $(\lambda_1,\lambda_2)$  
Month 10–11: Benchmarking vs. baselines; ablation studies  
Month 12: Manuscript preparation, code documentation, workshop presentation  

In summary, DiffuNA will harness the expressive power of diffusion‐based generative modeling and reinforcement learning to transform the landscape of RNA therapeutic design, offering a scalable and generalizable platform for the AI4NA community.