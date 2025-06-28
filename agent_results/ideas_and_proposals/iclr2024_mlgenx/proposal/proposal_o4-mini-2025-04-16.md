1. Title  
ActiveCausalOmics: Active Learning-Driven Causal Discovery in Multi-Omics Perturbation Data  

2. Introduction  
Background  
Understanding the causal mechanisms that drive disease phenotypes is a cornerstone of modern drug discovery. Traditional observational genomics studies often expose correlations without revealing underlying causal structure, leading to irreproducible findings and late‐stage failures in clinical trials. Recent high-throughput perturbation platforms (e.g., CRISPR knockouts, RNA interference) coupled with multi-omics readouts (single-cell RNA-seq, bulk transcriptomics, proteomics, spatial omics) provide unprecedented opportunities to interrogate gene–protein–phenotype relationships under controlled interventions. However, naïvely scaling perturbation experiments is cost-prohibitive, and existing causal-inference methods struggle with high dimensionality, data sparsity, and the need for interpretable models that biologists can trust.  

In parallel, advances in causal representation learning and active learning have laid the groundwork for iterative experimental design: one can maintain a posterior over candidate causal graphs, propose the next most informative perturbation, observe the outcome, and update beliefs. By integrating structured variational autoencoders (SVAEs) with Bayesian causal-graph posteriors and an active‐learning acquisition function, we can both compress multi-omics into interpretable latent mechanisms and accelerate the identification of high-confidence causal edges.  

Research Objectives  
This proposal aims to develop and validate ActiveCausalOmics, a principled framework that:  
1. Learns latent causal representations from heterogeneous omics modalities via a structured VAE whose prior is governed by a causal graph.  
2. Infers causal edges through interventional and observational data using counterfactual reasoning and Bayesian model averaging.  
3. Actively designs perturbation experiments to maximize information gain about the causal graph, subject to cost constraints.  

Significance  
By tightly coupling causal‐representation learning with active experimental design, ActiveCausalOmics promises to:  
– Reduce the number of required perturbation experiments by 30–50% compared to random or heuristic designs.  
– Deliver interpretable gene–protein–phenotype networks with quantified uncertainty, facilitating target prioritization.  
– Generalize across synthetic benchmarks and real datasets (e.g., LINCS L1000, pooled CRISPR screens), thereby accelerating hypothesis‐driven drug discovery.  

3. Methodology  
3.1 Data Collection and Preprocessing  
We will leverage both synthetic and real perturbation datasets:  
• Synthetic Benchmark: We will build upon existing network simulators (e.g., GeneNetWeaver) to generate multi-omics data—gene expression $X^{(RNA)}$, protein abundance $X^{(prot)}$, and spatial features $X^{(spat)}$—given a ground-truth causal directed acyclic graph (DAG) $G^*$. Perturbations consist of single-node interventions.  
• Real Datasets:  
  – LINCS L1000: bulk mRNA profiles under small-molecule and genetic perturbations.  
  – Pooled CRISPR Screens: single-cell RNA-seq profiles post gene knockout.  
  – Spatial proteomics: tissue‐section immunofluorescence under targeted perturbations.  

Preprocessing steps include: normalization (e.g., TPM or CPM for RNA, median‐ratio normalization for proteomics), batch‐effect correction (e.g., Combat), and outlier removal. All modalities are aligned by sample and intervention labels.  

3.2 Structured Variational Autoencoder with Causal Graph Prior  
We denote the observed data for $N$ samples by $\{x_i\}_{i=1}^N$, where $x_i = \{x_i^{(m)}\}_{m=1}^M$ are $M$ omics modalities. We introduce latent vectors $z_i \in \mathbb{R}^d$ that capture underlying biological mechanisms. A causal graph $G$ on $d$ latent nodes encodes structural equations.  

Generative model  
$$
p(x_i, z_i \mid G)
= p(z_i \mid G)\prod_{m=1}^M p\bigl(x_i^{(m)}\mid z_i^{(m)}\bigr).
$$  
Here $p(z_i\mid G)$ factorizes according to the DAG $G$:  
$$
p(z_i\mid G) = \prod_{j=1}^d p\bigl(z_{i,j}\mid z_{i,\mathrm{pa}(j)}\bigr),
$$  
where $\mathrm{pa}(j)$ are the parents of node $j$ in $G$. We assume linear Gaussian structural equations for interpretability:  
$$
z_{i,j} = \sum_{k\in\mathrm{pa}(j)} W_{kj}\,z_{i,k} + \varepsilon_{i,j},\quad
\varepsilon_{i,j}\sim\mathcal{N}(0,\sigma_j^2).
$$  

Decoder networks $p(x^{(m)}\mid z^{(m)})$ are modality-specific neural decoders (e.g., negative‐binomial for count data, Gaussian for continuous proteomics).  

Variational Inference  
We approximate the posterior with $q_\phi(z_i\mid x_i)$, a structured encoder that outputs the mean and variance of a multivariate Gaussian. The evidence lower bound (ELBO) for a given graph $G$ is:  
$$
\mathrm{ELBO}(G;\phi,\theta)
= \sum_{i=1}^N \mathbb{E}_{q_\phi(z_i\mid x_i)}\Bigl[\sum_{m=1}^M \log p_\theta\bigl(x_i^{(m)}\mid z_i^{(m)}\bigr)\Bigr]
- \sum_{i=1}^N D_{KL}\bigl(q_\phi(z_i\mid x_i)\,\|\,p(z_i\mid G)\bigr).
$$  
We learn $\phi,\theta$ by maximizing $\mathrm{ELBO}$ conditioned on $G$.  

3.3 Bayesian Causal-Graph Posterior and Active Learning  
We maintain a posterior distribution over graphs $p(G\mid D)$ given data $D$. Using a score‐based approach with a graph prior $p(G)$ (e.g., uniform or sparsity‐encouraging), we approximate  
$$
p(G\mid D)\propto p(D\mid G)\,p(G)\approx \exp\bigl(\mathrm{ELBO}(G)\bigr)\,p(G).
$$  
Interventions $I$ are single‐node knockouts or overexpressions. For each candidate perturbation $i\in\{1,\dots,d\}$, we compute the expected information gain (EIG):  
$$
\mathrm{EIG}(i) 
= H\bigl[p(G\mid D)\bigr]
- \mathbb{E}_{y_i\sim p(y_i\mid D,I=i)}\bigl[\,H\bigl[p(G\mid D\cup\{(i,y_i)\})\bigr]\bigr].
$$  
We select the next intervention by  
$$
i^*=\arg\max_{i}\Bigl\{\mathrm{EIG}(i)-\lambda\,C(i)\Bigr\},
$$  
where $C(i)$ is the experimental cost and $\lambda$ trades off cost vs. information. We approximate both $p(y_i\mid D,I=i)$ and the posterior entropy via Monte Carlo sampling of $(G,z,x)$.  

3.4 Algorithmic Steps  
Algorithm: ActiveCausalOmics  
Input: Dataset $D_0$ (initial observational + any existing interventions), intervention budget $B$, cost function $C(\cdot)$  
1. Initialize posterior $p(G\mid D_0)$ by scoring top‐$K$ graphs via ELBO.  
2. For $t=1$ to $T$ (until budget exhausted):  
   a. For each candidate $i$, estimate $\mathrm{EIG}(i)$ by sampling $G\sim p(G\mid D_{t-1})$ and $y_i\sim p(x\mid G,I=i)$.  
   b. Select $i^* = \arg\max_i \{\mathrm{EIG}(i)-\lambda C(i)\}$.  
   c. Perform perturbation $I=i^*$ in wet lab, observe $y_{i^*}$.  
   d. Update dataset $D_t = D_{t-1}\cup \{(i^*,y_{i^*})\}$.  
   e. Recompute $p(G\mid D_t)$ via ELBO scoring.  
3. Return posterior‐mean graph $\hat G$ and latent encoder/decoder parameters.  

3.5 Experimental Design and Evaluation Metrics  
We will evaluate ActiveCausalOmics on both synthetic and real benchmarks:  

Synthetic Experiments  
– Graph recovery: measure structural Hamming distance (SHD) and edge‐precision/recall between $\hat G$ and $G^*$.  
– Sample efficiency: number of interventions required to reach a target SHD.  
– Ablations: remove active learning (random interventions), remove multimodal integration, remove uncertainty quantification.  

Real Data Experiments  
– Evaluate on LINCS L1000: compare discovered gene–gene causal edges to known interactions from KEGG/Reactome; compute precision@K.  
– CRISPR validation: for top‐$k$ predicted causal regulators of a phenotype, perform focused knockouts and assess phenotype change.  
– Predictive performance: held‐out log‐likelihood of omics modalities, mean squared error for continuous outputs, negative log‐likelihood for counts.  

Uncertainty Calibration  
– Compute Brier score and calibration plots for edge existence probabilities.  
– Compare Bayesian credible intervals against bootstrapped CIs from baseline methods (e.g., PC, GES).  

Implementation Details  
– Model implemented in PyTorch; graph search parallelized across GPUs.  
– Hyperparameters ($\beta$ in ELBO, cost penalty $\lambda$, latent dimension $d$) tuned via grid search on validation synthetic data.  
– Code and datasets released under open‐source license.  

4. Expected Outcomes & Impact  
Expected Outcomes  
1. A unified software toolkit, ActiveCausalOmics, that integrates structured VAE learning, Bayesian causal‐graph inference, and active experimental design.  
2. Demonstrated improvements on synthetic benchmarks: 40–60% fewer interventions to achieve comparable graph recovery (SHD < 5) versus baselines.  
3. Validated causal networks on real perturbation datasets, with precision@50 exceeding 0.7 for known pathways.  
4. Empirical evidence that active learning reduces wet‐lab cost by prioritizing high‐information experiments.  
5. Well‐calibrated uncertainty estimates for inferred edges, aiding target prioritization.  

Impact  
ActiveCausalOmics will empower biologists and drug‐discovery teams to:  
– Rapidly home in on high‐confidence causal targets, reducing trial‐and-error cycles.  
– Allocate experimental resources more efficiently via informed perturbation selection.  
– Interpret complex multi-omics interactions through transparent latent mechanisms.  
– Generalize the approach across disease areas, from oncology to neurodegeneration.  

In the long term, this work will contribute to a paradigm shift in genomics research: moving from large‐scale, hypothesis‐free screens to a closed‐loop, hypothesis-driven discovery process that is both data-efficient and causally grounded. By bridging the gap between machine learning, causal inference, and experimental biology, ActiveCausalOmics stands to accelerate the translation from genomic insights to viable drug targets.