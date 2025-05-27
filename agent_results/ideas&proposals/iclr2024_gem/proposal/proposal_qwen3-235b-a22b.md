# Adaptive Design Space Exploration for Protein Engineering Using Generative Models and Bayesian Optimization

## Introduction

### Background
Protein engineering stands at the forefront of biomolecular innovation, with engineered proteins enabling breakthroughs in therapeutics, industrial catalysis, and sustainable materials. However, the exponential growth of sequence space—where even a moderate-length protein of 100 amino acids spans $20^{100}$ possibilities—renders brute-force experimental exploration infeasible. While generative machine learning (ML) models, such as variational autoencoders (VAEs) and diffusion networks, have demonstrated promise in silico [3, 6], their translation to wet-lab validation remains limited. This disconnect stems from three key challenges: **high false-positive rates** in computational predictions [1], **inefficient allocation of experimental resources**, and **static design frameworks** that fail to incorporate iterative feedback [7].

Recent advancements in adaptive experimental design [5, 8] and closed-loop learning systems [7, 10] offer a paradigm to address these limitations. By integrating real-time experimental feedback into computational pipelines, such frameworks can dynamically refine exploration of sequence space, balancing exploitation of known functional motifs with exploration of novel regions. For instance, Calvanese et al. (2025) demonstrated that likelihood-based reintegration of experimental data could reduce false positives by 40% in RNA design [1], while Martinez et al. (2025) showed that adaptive design strategies could accelerate protein fitness optimization [10]. Despite this progress, no existing method systematically combines the **generative diversity** of VAEs with the **precision of Bayesian optimization (BO)** to iteratively sculpt experimental campaigns.

### Research Objectives
This proposal aims to develop, validate, and benchmark an adaptive design space exploration framework that:
1. **Generates diverse protein candidates** using a VAE conditioned on structural and functional priors.
2. **Selects optimal candidates** for experimental validation via BO-driven acquisition functions that quantify both predicted fitness and uncertainty.
3. **Refines the model iteratively** through closed-loop feedback, updating latent representations and fitness predictors with new experimental data.
4. **Quantifies cost-performance trade-offs**, targeting an 80% reduction in experimental burden relative to conventional screening while maintaining discovery rates.

### Significance
By bridging computational and experimental workflows, this framework will:
- **Reduce experimental costs** through targeted sampling of high-utility sequences.
- **Accelerate protein discovery** by focusing resources on regions of latent space most likely to yield functional variants.
- **Provide interpretability** in design decisions via uncertainty quantification and latent space visualization.
- **Establish a benchmark** for adaptive ML in biomolecular engineering, addressing GEM workshop priorities for datasets, adaptive design, and cross-disciplinary integration.

---

## Methodology

### 1. Data Collection and Preprocessing
#### Datasets
The framework will leverage:
- **Sequence-structure databases**: Pfam [PFAM] and PDB [PDB] for evolutionary and structural motifs.
- **Functional annotations**: Enzymatic activity data from BRENDA for conditioning generative models.
- **High-throughput experimental datasets**: Mutational scans of proteins like beta-lactamase [BLAT_ECOLX] and fluorescent proteins [P2FP_ECHGR].

#### Preprocessing
- **MSA generation**: Align sequences using HHBlits to extract coevolutionary features.
- **Embeddings**: Initialize VAE decoders with ESM-2 [esm] language model weights for structural priors.
- **Oracle construction**: Define a fitness function $f(\mathbf{x})$ mapping sequences $\mathbf{x}$ to experimental activity scores (e.g., fluorescence intensity).

### 2. Generative Modeling with VAEs
#### Architecture
A VAE will learn a latent representation $\mathbf{z} \in \mathbb{R}^d$ of sequences $\mathbf{x} \in \{1,\dots,20\}^{L}$ (length $L$, 20 amino acids). The model maximizes the evidence lower bound (ELBO):
$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\left[\log p_{\theta}(\mathbf{x}|\mathbf{z})\right] - D_{\text{KL}}\left(q_{\phi}(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z})\right),
$$
where $q_{\phi}$ is the encoder and $p_{\theta}$ is the decoder. The latent space will be regularized using a hierarchical prior $p(\mathbf{z})$ informed by structural annotations.

#### Conditional Generation
To prioritize functional sequences, the decoder is conditioned on fitness predictions $\hat{y} = \mu(\mathbf{z})$ from a Gaussian process (GP) regressor:
$$
p_{\theta}(\mathbf{x}|\mathbf{z}, \hat{y}) = \text{Softmax}\left(\text{MLP}([\mathbf{z}; \hat{y}])\right).
$$
This ensures that sampling occurs in regions of latent space associated with high predicted fitness.

### 3. Adaptive Bayesian Optimization (BO)
#### Acquisition Function
At each iteration $t$, BO selects sequences $\mathbf{x}^{(t)}$ for experimental validation using an acquisition function $\alpha(\mathbf{z})$ that balances exploration and exploitation:
$$
\mathbf{z}^{(t)} = \arg\max_{\mathbf{z}} \left[ \mu_t(\mathbf{z}) + \kappa \sigma_t(\mathbf{z}) \right],
$$
where $\mu_t$ and $\sigma_t$ are the mean and standard deviation of the GP posterior at iteration $t$, and $\kappa$ controls exploration intensity [5].

#### Uncertainty-Aware Diversity Selection
To prevent redundancy, selected sequences are diversified using a **determinantal point process (DPP)** kernel [9]:
$$
P(S) \propto \det(\mathbf{K}_S), \quad \mathbf{K}_{ij} = \exp\left(-\frac{\|\mathbf{z}_i - \mathbf{z}_j\|^2}{2\ell^2}\right),
$$
where $S$ is a subset of $k$ sequences and $\ell$ is a length scale. This maximizes diversity in latent space while prioritizing high-acquisition candidates.

### 4. Experimental Validation and Feedback Loop
#### Wet-Lab Pipeline
1. **High-throughput screening**: Synthesize top-$k$ candidates (e.g., $k=100$) using chip-based DNA synthesis and express in *E. coli*.
2. **Activity assays**: Quantify fitness $y_i$ via fluorescence-activated cell sorting (FACS) for fluorescent proteins or MIC assays for antibiotic resistance enzymes.

#### Model Update
New data $\mathcal{D}_t = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ are used to:
- Refine the GP posterior: $p(\mu_{t+1}, \sigma_{t+1}) \leftarrow \text{GP}(\mathcal{D}_{1:t})$
- Retrain the VAE with weighted maximum likelihood, emphasizing sequences $\mathbf{x}_i$ with $y_i > \tau$ (threshold $\tau$).

### 5. Evaluation Metrics
#### Primary Metrics
- **Success rate**: Fraction of validated sequences exceeding a fitness threshold.
- **Cost reduction**: $\frac{N_{\text{traditional}} - N_{\text{adaptive}}}{N_{\text{traditional}}} \times 100$, where $N$ is the number of experiments required.
- **Active learning efficiency**: Area under the precision-recall curve (AUPRC) for discovering high-fitness variants.

#### Baselines for Comparison
- **Random sampling**: Uniform sampling from sequence space.
- **Static ML**: VAE + GP without iterative feedback [6].
- **IsEM-Pro**: State-of-the-art method for diverse sequence generation [4].

#### Reproducibility
Code and trained models will be released under an MIT license. Experiments will use PyTorch [torch] and BoTorch [botorch], with hyperparameters provided in Appendix A.

---

## Expected Outcomes & Impact

### Quantitative Outcomes
1. **Cost Reduction**: The framework is projected to reduce experimental costs by **80%** compared to random sampling. For a target of 1,000 functional sequences, adaptive design would require ~200 experiments vs. 1,000 with traditional screens.
2. **Accelerated Discovery**: AUPRC is expected to improve by **30–50%** relative to static ML baselines, reflecting more efficient navigation of latent space.
3. **Generalization**: The method will enable discovery of functional variants in low-homology proteins (e.g., novel folds), where existing methods struggle due to data scarcity [2].

### Qualitative Impact
1. **Methodological Innovation**: The combination of VAEs, DPP-driven diversity, and BO creates a blueprint for adaptive design in RNA engineering, small-molecule discovery, and synthetic biology.
2. **Cross-Disciplinary Bridge**: By aligning computational predictions with experimental constraints, the framework addresses GEM workshop priorities for "biological problems ready for ML" and "adaptive experimental design" [GEM Topics].
3. **Open-Source Tools**: A modular PyTorch implementation will lower the barrier for adopting adaptive ML in academic and industrial labs.

### Long-Term Vision
This work directly addresses two grand challenges from the literature [1, 10]:
- **Closing the simulation-to-reality gap**: Experimental feedback will iteratively calibrate latent spaces, reducing false positives from 45% to <10% as demonstrated by Calvanese et al. [1].
- **Democratizing protein engineering**: By reducing experimental burdens, the framework empowers smaller labs to tackle complex design tasks without massive HTS infrastructure.

Future work will explore multi-objective optimization (e.g., balancing solubility and activity) and integration of AlphaFold2 [alphafold] into the fitness oracle. Ultimately, adaptive design space exploration could enable real-time, AI-guided biomolecular engineering pipelines with minimal human intervention.

--- 

**Appendix A: Hyperparameters**  
- VAE: Latent dimension $d=128$, KL divergence weight $\beta=0.1$.  
- GP: ARD kernel with ESM embedding-based length scales.  
- BO: $\kappa=2.0$, batch size $k=100$.  
- DPP: $\ell=0.5$ for latent space kernel.  

**Appendix B: Ethics Statement**  
This work adheres to biocontainment guidelines for synthetic proteins. No human subject data is involved.