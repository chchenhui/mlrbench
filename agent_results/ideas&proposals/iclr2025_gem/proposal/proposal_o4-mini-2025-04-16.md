Title:  
Iterative Generative Design with Active Learning for Optimized Antibody Affinity Maturation  

Introduction  
Background  
The engineering of high-affinity antibodies is a cornerstone of modern therapeutic development, diagnostics, and fundamental immunology research. Traditional affinity maturation relies on directed evolution techniques—such as random mutagenesis, phage or yeast display, and screening—that often require screening tens of thousands of variants in the wet lab, resulting in high time and resource costs. In parallel, generative machine-learning models (e.g., ProteinMPNN, ESM-IF) have demonstrated the ability to propose novel antibody sequences with favorable properties in silico. However, purely computational pipelines face two major limitations: (1) predictive inaccuracies when extrapolating beyond training distributions, and (2) the prohibitive cost of validating large numbers of candidates in the laboratory.  

Research Objectives  
We propose a closed-loop framework—Iterative Generative and Active Learning (IGAL)—that tightly integrates a generative sequence model with an active learning acquisition strategy to guide experimental affinity maturation. The specific objectives are:  
1. Develop a generative model $G_\theta$ that proposes antibody variants conditioned on a parent sequence.  
2. Train a predictive model $f_\phi$ to estimate binding affinity from sequence.  
3. Design an acquisition function $a(s)$ to select a small, informative batch of candidate variants for wet-lab testing.  
4. Implement an iterative loop that: (i) generates candidates, (ii) selects the optimal subset via active learning, (iii) performs wet-lab affinity measurements, and (iv) updates both models.  
5. Validate the framework on two therapeutic‐relevant antigen–antibody systems, comparing efficiency and performance against baselines (random mutagenesis, pure generative sampling, existing Bayesian optimization).  

Significance  
By fusing generative modeling with experiment-guided active learning, IGAL aims to (a) reduce experimental burden by focusing on the most informative variants, (b) refine both generative and predictive models using real affinity data, and (c) accelerate the discovery of antibodies with sub-nanomolar binding affinities. Successful demonstration will set a new paradigm for ML–wet-lab integration in biomolecular design, with broad applications in therapeutics and beyond.  

Methodology  
1. Data Collection and Preprocessing  
• Initial dataset $\mathcal D_0$: Collect publicly available antibody–antigen complex data from SAbDab and internal yeast-display ΔΔG measurements (∼5,000 sequences with measured affinities).  
• Parent sequences: Select two well-characterized antibodies (e.g., against PD-1 and VEGF) as seeds for affinity maturation.  
• Sequence encoding: Represent each antibody variable region as one-hot encoded amino acids or via embeddings from a pre-trained protein language model (e.g., ESM-2).  
• Affinity normalization: Convert dissociation constants $K_D$ to $\Delta G_{\mathrm{bind}} = RT\ln K_D$ for regression stability.  

2. Generative Model Design  
• Base architecture: Start from a pre-trained ProteinMPNN or ESM-IF model, denoted $G_{\theta_0}$.  
• Conditional generation: Fine-tune $G_\theta$ to generate variants conditioned on parent sequence $s_0$. At iteration $t$, the model samples sequences $s\sim G_\theta(\cdot\,|\,s_0)$.  
• Fine-tuning objective: Given the growing dataset $\mathcal D_t=\{(s_i,y_i)\}_{i=1}^{N_t}$, update $\theta$ by maximizing:  
  $$
  \mathcal L_{\mathrm{gen}}(\theta)\;=\;-\sum_{i=1}^{N_t} w(y_i)\,\log p_\theta(s_i\,|\,s_0)\,,
  $$  
  where $w(y_i)=1+\alpha\,(y_i-\bar y)$ emphasizes high-affinity examples ($\alpha>0$) and $\bar y$ is the mean affinity in $\mathcal D_t$.  

3. Predictive Model Design  
• Architecture: Use a regression head $f_\phi$ on top of frozen or fine-tuned ESM embeddings, predicting normalized binding free energy $\hat y = f_\phi(s)$.  
• Loss function: Mean squared error on $\mathcal D_t$:  
  $$  
  \mathcal L_{\mathrm{pred}}(\phi)\;=\;\frac{1}{N_t}\sum_{i=1}^{N_t}\bigl(f_\phi(s_i)-y_i \bigr)^2\,.  
  $$  
• Uncertainty estimation: Use Monte Carlo dropout or deep ensembles to estimate predictive variance $\sigma^2(s)$ for active learning.  

4. Active Learning Framework  
• Acquisition function: We adopt a Gaussian-process–inspired upper confidence bound (UCB)  
  $$  
  a(s)\;=\;\mu(s)\;+\;\kappa\,\sigma(s)\,,  
  $$  
  where $\mu(s)=f_\phi(s)$ and $\kappa>0$ balances exploration (via $\sigma$) and exploitation (via $\mu$).  
• Batch selection: From $M$ candidates $S_t=\{s_j\}_{j=1}^M$, compute $a(s_j)$ and select the top $b$ sequences $B_t=\arg\max_{s\in S_t}a(s)$ for wet-lab testing.  

5. Iterative Optimization Algorithm  
We formalize the closed-loop as follows:  

Algorithm IGAL  
Input: Parent sequence $s_0$, initial dataset $\mathcal D_0$, generative model $G_{\theta_0}$, predictor $f_{\phi_0}$, acquisition weight $\kappa$, candidate pool size $M$, batch size $b$, total rounds $T$.  
For $t=1,\dots,T$:  
  1. Candidate generation: Sample $S_t=\{s_j\}_{j=1}^M\sim G_{\theta_{t-1}}(\cdot\,|\,s_0)$.  
  2. Acquisition scoring: For each $s_j\in S_t$, compute $\mu_j=f_{\phi_{t-1}}(s_j)$ and $\sigma_j$. Then $a_j=\mu_j+\kappa\,\sigma_j$.  
  3. Sequence selection: $B_t\leftarrow$ top $b$ sequences by $a_j$.  
  4. Wet-lab measurement: Express $B_t$ in yeast display; measure binding constants $K_D$ via surface plasmon resonance (SPR); convert to $\Delta G_{\mathrm{bind}}$.  
  5. Dataset update: $\mathcal D_t\leftarrow \mathcal D_{t-1}\cup\{(s_i,y_i)\mid s_i\in B_t\}$.  
  6. Model update:  
     • Fine-tune predictor: $\phi_t = \arg\min_\phi \mathcal L_{\mathrm{pred}}(\phi;\mathcal D_t)$.  
     • Fine-tune generator: $\theta_t = \arg\min_\theta \mathcal L_{\mathrm{gen}}(\theta;\mathcal D_t)$.  
End For  
Output: Top sequences in $\bigcup_{t=1}^T B_t$ ranked by measured affinity.  

Hyperparameters and implementation details:  
• Candidate pool $M=5{,}000$, batch $b=50$, rounds $T=5$.  
• Learning rates: $1\!\times\!10^{-4}$ for $f_\phi$, $5\!\times\!10^{-5}$ for $G_\theta$.  
• Optimizer: AdamW with weight decay 1e-2.  
• Compute: NVIDIA A100 GPUs for model fine-tuning; standard yeast-display platform and Biacore SPR instrument for binding assays.  

6. Experimental Validation  
Case studies on two antigen systems (e.g., PD-1 and VEGF):  
• Round-by-round tracking of median, mean, and best measured affinities.  
• Expression yield and developability assays (e.g., thermal stability, aggregation propensity) on top variants.  
• Comparative controls:  
  – Random mutagenesis: measure affinity improvements when selecting uniformly from $S_t$.  
  – Pure generative sampling: top $b$ by $\mu(s)$ only (no uncertainty).  
  – CloneBO (Bayesian optimization baseline).  

7. Evaluation Metrics  
• Affinity improvement: fold-change in $K_D$ relative to $s_0$.  
• Experimental efficiency: number of wet-lab measurements required to reach an affinity threshold.  
• Model performance: correlation ($r$), root mean squared error (RMSE) between predicted and measured $\Delta G$.  
• Diversity: mean pairwise sequence identity among selected $B_t$.  
• Statistical significance: paired $t$-tests comparing IGAL vs. baselines.  

Expected Outcomes & Impact  
Anticipated Outcomes  
• Affinity gain: Achieve >10-fold improvement in binding affinity (e.g., from 100 nM to sub-10 nM $K_D$) within five rounds, using only $250$ wet-lab measurements.  
• Experimental savings: Demonstrate ≥50% reduction in experimental budget compared to random or purely greedy generative approaches.  
• Model advancement: Improved predictive accuracy (RMSE < 0.5 kcal/mol; $r>0.8$) and generative fidelity (maintaining natural antibody motif statistics).  
• Generalizability: Replicable success across two antigen systems, showing robustness to different epitope features and antibody families.  

Broader Impact  
• Accelerated antibody discovery: By tightly coupling generative proposals with active learning–guided experiments, IGAL can shorten the lead optimization phase from months to weeks, dramatically lowering costs in pharmaceutical R&D.  
• Framework extensibility: The closed-loop design is agnostic to target type and can be applied to enzymes, peptide ligands, or small-molecule binding proteins.  
• Community benefit: All code, trained models, and anonymized affinity data will be released under an open-source license, promoting reproducibility and further innovation in ML-driven biomolecular design.  
• Workshop relevance: IGAL exemplifies the synergy of generative ML and experimental validation, directly addressing the GEM workshop’s goal of bridging computational and experimental perspectives.  

In sum, this proposal outlines a rigorous, mathematically grounded, and experimentally validated approach to antibody affinity maturation that leverages the strengths of generative machine learning and active learning to deliver high-affinity candidates with maximal efficiency.