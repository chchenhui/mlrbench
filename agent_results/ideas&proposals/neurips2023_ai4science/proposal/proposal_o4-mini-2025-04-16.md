Title  
Physics-Informed Reinforcement Learning with Molecular-Dynamics Surrogates for De Novo Drug Design  

Introduction  
Background and Motivation  
De novo molecular generation driven by deep learning has revolutionized early‐stage drug discovery by exploring chemical space far beyond traditional medicinal chemistry heuristics. Transformer‐ and graph-neural-network based generative models, often trained with reinforcement learning (RL), can optimize chemical properties such as solubility, drug-likeness (QED) and scaffold novelty. However, purely data-driven approaches frequently neglect the underlying physics of molecular stability and dynamic behavior, resulting in high attrition rates when candidates are synthesized and tested in vitro or in silico with full molecular dynamics (MD). The need to ensure both chemical validity and physical plausibility is acute: failed candidates represent wasted resources, extended timelines, and missed therapeutic opportunities.  

Research Objectives  
This proposal aims to develop a physics-informed RL framework that couples a graph-based molecular generator with a lightweight MD surrogate model for rapid, on-the-fly evaluation of physical stability, free-energy differences, and binding-affinity proxies. The central research objectives are:  
1. Design and train an MD surrogate network that predicts key physical metrics (e.g., folding stability, binding energy) with high fidelity to all-atom simulations, reducing per-molecule evaluation time by ≥90%.  
2. Integrate the surrogate into an RL loop—where the agent’s reward balances chemical objectives (e.g., QED, synthetic accessibility) and surrogate-predicted physical plausibility—to guide generation towards molecules both chemically valid and physically stable.  
3. Demonstrate that the physics-informed framework yields a higher proportion of synthesizable, thermodynamically stable candidates compared to state-of-the-art RL baselines, reducing costly full MD re-simulation cycles by 30–50%.  

Significance  
By grounding generative models in physical reality, this research bridges a major gap between AI-driven design and experimental validation, accelerating the hit-to-lead stage in drug discovery. The MD surrogate approach is broadly applicable to structural biology challenges, high-throughput screening, and other scientific domains requiring rapid, physics-constrained generation.  

Methodology  
1. Overview of Framework  
We define an RL environment $\mathcal{E}$ in which an agent (policy network $\pi_\theta$) generates molecular graphs $G$ and receives a composite reward $R(G)$. The environment consists of two components:  
– A chemical evaluator computing standard metrics: validity, quantitative estimate of drug-likeness (QED), synthetic‐accessibility (SA) score.  
– An MD surrogate model $\mathcal{M}_\phi$ predicting physical scores:  
  • Stability score $S_{\rm stab}$ (e.g., surrogate free energy of folding)  
  • Binding-affinity proxy $B_{\rm aff}$ (e.g., predicted docking energy)  

The overall reward is  
$$  
R(G) \;=\; w_{\rm chem}\,R_{\rm chem}(G)\;+\;w_{\rm phys}\,R_{\rm phys}(G)\;,  
$$  
where  
$$  
R_{\rm chem}(G)=\alpha_1\;\mathrm{QED}(G)\;-\;\alpha_2\;\mathrm{SA}(G)\,,  
\quad  
R_{\rm phys}(G)=\beta_1\,S_{\rm stab}(G)\;+\;\beta_2\,B_{\rm aff}(G)\,.  
$$  

2. MD Surrogate Model  
2.1 Architecture  
We adopt a graph-neural-network (GNN) based surrogate  
$$  
\mathcal{M}_\phi(G)\;=\;\mathrm{MLP}\bigl( \mathrm{AGG}\{h_v\,|\,v\in V(G)\}\bigr)\,,  
$$  
where $h_v$ are learned atom representations and AGG is a permutation-invariant aggregator (sum or mean). The surrogate outputs $(\hat S_{\rm stab},\,\hat B_{\rm aff})$.  

2.2 Training Data Collection  
We curate a dataset of $N\approx 50{,}000$ drug-like molecules from ZINC and PDB complexes. For each molecule or protein–ligand complex we run short all-atom MD simulations (2–5 ns) to obtain ground-truth stability $S_{\rm stab}^\ast$ (e.g., folding ΔG) and docking scores $B_{\rm aff}^\ast$.  

2.3 Loss Function  
We train $\phi$ by minimizing mean-squared error plus regularization:  
$$  
\mathcal{L}(\phi)=\frac{1}{N}\sum_{i=1}^N\Bigl[(\hat S_{\rm stab}^i-S_{\rm stab}^{\ast,i})^2 + (\hat B_{\rm aff}^i-B_{\rm aff}^{\ast,i})^2\Bigr] + \lambda\|\phi\|_2^2\,.  
$$  

3. Reinforcement Learning Agent  
3.1 Policy Network  
The policy $\pi_\theta(a_t|s_t)$ is implemented as a graph-transformer that grows molecules atom-by-atom or bond-by-bond. The state $s_t$ is the current partial graph; actions $a_t$ add atoms, bonds or terminate.  

3.2 Objective and Gradient  
We maximize the expected composite return  
$$  
J(\theta)=\mathbb{E}_{G\sim\pi_\theta}[R(G)]  
$$  
via policy gradients:  
$$  
\nabla_\theta J = \mathbb{E}_{G\sim\pi_\theta}\Bigl[\sum_{t=1}^T \nabla_\theta\log\pi_\theta(a_t|s_t)\,(R(G)-b(s_t))\Bigr]\,,  
$$  
where $b(s_t)$ is a learned baseline to reduce variance.  

3.3 Adaptive Reward Balancing  
To avoid domination by either chemical or physical terms, we adapt weights $w_{\rm chem},w_{\rm phys}$ during training by monitoring their contribution variance and normalizing to unit range:  
$$  
w_{\rm chem}\leftarrow \frac{1}{\sigma[R_{\rm chem}]}\,,\quad  
w_{\rm phys}\leftarrow \frac{1}{\sigma[R_{\rm phys}]}\,.  
$$  

4. Full Algorithm  
Algorithm 1: Physics-Informed RL for Molecular Generation  
1. Pretrain surrogate $\mathcal{M}_\phi$ on $(G_i,S_i^\ast,B_i^\ast)$  
2. Initialize policy $\pi_\theta$, baseline $b$, weights $(w_{\rm chem},w_{\rm phys})$  
3. For each training iteration:  
   a. Generate batch of $M$ molecules $\{G^{(j)}\}$ via $\pi_\theta$  
   b. For each $G^{(j)}$:  
      i. Compute $R_{\rm chem},\,\hat S_{\rm stab},\,\hat B_{\rm aff}$  
      ii. Compute total $R^{(j)}$  
   c. Update $\theta$ via policy-gradient using $\{R^{(j)}\}$ and baseline $b$  
   d. Update $b(s_t)$ by regression to observed returns  
   e. Adapt $(w_{\rm chem},w_{\rm phys})$ by normalizing reward variances  
4. Periodically fine-tune $\mathcal{M}_\phi$ with new full-MD results on promising candidates  

5. Experimental Design and Validation  
5.1 Baselines  
– Mol-AIR (Park et al., 2024)  
– Transformer-RL (Xu et al., 2023)  
– Graph-RL with static physical constraints (arXiv:2312.04567)  

5.2 Datasets  
– ZINC15 for pretraining generative models (~1 M molecules)  
– PDB complexes (~100 K protein–ligand pairs) for surrogate training  
– Held-out test set of 5 K molecules from ChEMBL for evaluation  

5.3 Evaluation Metrics  
Chemical metrics:  
  • Validity (% chemically valid SMILES/graphs)  
  • Uniqueness (% distinct)  
  • Novelty (% not in training set)  
  • QED and SA distributions  
Physical metrics (surrogate & full MD):  
  • Fraction of candidates with $\hat S_{\rm stab}>S_\mathrm{thresh}$  
  • Predicted binding energy $\hat B_{\rm aff}$ below threshold  
  • Post-hoc MD validation of top 200 candidates: RMSD stability, ΔG calculation  
Efficiency metrics:  
  • Wall-clock time per molecule generation + evaluation  
  • Number of full-MD calls reduced vs. naïve MD-in-loop  

5.4 Ablation Studies  
– Effect of surrogate fidelity: train $\mathcal{M}_\phi$ with varying MD simulation lengths  
– Reward term ablations: chemical‐only, physical‐only, unbalanced vs. adaptive  
– Surrogate vs. direct MD in reward loop  

5.5 Implementation Details  
– Hardware: NVIDIA A100 GPUs, 64-core CPUs for MD data generation  
– Software: PyTorch Geometric for GNNs, OpenMM for MD, RDKit for cheminformatics  
– Hyperparameters: learning rate $1\mathrm{e}{-4}$ for $\theta$, $1\mathrm{e}{-3}$ for $\phi$, batch size $M=64$, temperature annealing for exploration  

Expected Outcomes & Impact  
We anticipate the proposed physics-informed RL framework will:  
1. Achieve a 30–50% reduction in full MD simulations needed per viable candidate by leveraging the surrogate model, lowering computational cost and accelerating screening cycles.  
2. Increase the fraction of synthesizable, thermodynamically stable molecules by ≥25% relative to chemical-only RL baselines, as validated by post-hoc full MD.  
3. Demonstrate improved hit rates in in silico binding assays, yielding lead candidates with predicted binding affinities on par with known actives.  

Impact on Science and Drug Discovery  
By incorporating physical insights directly into generative AI, this work advances the science of scientific discovery itself—moving beyond black-box molecular generation toward methods that respect fundamental physical laws. Practically, drug discovery pipelines will benefit from shorter design cycles, reduced attrition in hit-to-lead stages, and a generalizable framework adaptable to protein–protein interfaces, materials design, and other domains demanding both chemical creativity and physical rigor. The open-source release of code, pretrained models, and curated MD datasets will foster community adoption and further innovation at the intersection of AI and the physical sciences.