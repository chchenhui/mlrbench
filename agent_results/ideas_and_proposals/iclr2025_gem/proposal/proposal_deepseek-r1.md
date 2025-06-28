# Closed-loop Iterative Generative Design with Active Learning for High-Efficiency Antibody Affinity Maturation  

## 1. Introduction  
### Background  
Antibody affinity maturation—the process of enhancing binding strength to a target antigen—is critical for developing effective therapeutics for cancer, infectious diseases, and autoimmune disorders. Traditional approaches rely on error-prone PCR and phage/yeast display screening, requiring thousands of variants to be synthesized and tested experimentally. While generative machine learning (ML) models like *ProteinMPNN* and *ESM-IF* offer accelerated *in silico* design of antibody variants, most efforts remain siloed in computational pipelines, with limited feedback from wet-lab experiments. Current ML methods prioritize static benchmark performance (e.g., perplexity, recovery rates) but fail to address the *experimental bottleneck*: validating millions of generated sequences is cost-prohibitive.  

Recent work (e.g., Gessner et al., 2024; Amin et al., 2024) demonstrates that active learning (AL) can strategically prioritize high-value candidates for experimental testing, reducing resource consumption. However, existing frameworks lack tight integration between generative models, predictive scoring, and iterative experimental validation.  

### Research Objectives  
This work aims to establish a **closed-loop system** that synergizes:  
1. A **generative model** for *de novo* antibody sequence design,  
2. A **predictive model** for binding affinity estimation,  
3. An **active learning framework** to guide wet-lab experimentation,  
4. **Iterative feedback** from experiments to refine both models.  

Key objectives include:  
- Developing a robust AL strategy balancing exploration (novel sequence discovery) and exploitation (affinity optimization).  
- Validating the framework through yeast display and surface plasmon resonance (SPR) experiments.  
- Providing guidelines for ML-driven biological design prioritizing real-world applicability.  

### Significance  
This work bridges the gap between *in silico* generative design and experimental biology by:  
- **Reducing experimental costs** through targeted screening of high-potential candidates,  
- **Accelerating therapeutic development** via rapid iteration between ML and wet-lab workflows,  
- Establishing **benchmarks** for evaluating ML models in real-world antibody design workflows.  
The proposed framework aligns with the GEM workshop’s mission to foster collaboration between computational and experimental domains, with direct implications for drug discovery and synthetic biology.  

---

## 2. Methodology  
The methodology comprises three iterative phases (Figure 1): **(1)** Initial training of generative and predictive models, **(2)** Active learning-guided experimental cycles, and **(3)** Model updating and validation.  

### 2.1 Phase 1: Initial Model Training  
**Generative Model:**  
- **Architecture:** Use a pre-trained protein language model (*ESM-IF* or *IgDiff*) fine-tuned on antibody sequences from the *Structural Antibody Database (SAbDab)*. The model generates mutant variants by conditioning on the parent antibody’s complementarity-determining regions (CDRs).  
- **Input:** Parent sequence (heavy/light chain) and structural descriptors (e.g., paratope residues).  
- **Output:** Novel antibody sequences with mutations focused on CDR-H3/L3 regions.  
- **Loss Function:** Masked language modeling loss with regularization to preserve structural stability:  
  $$ \mathcal{L}_{\text{gen}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ -\log P(x_{\text{masked}} | x_{\text{unmasked}}) \right] + \alpha \cdot \text{KL}(P_{\text{current}} || P_{\text{pretrained}}), $$  
  where $\alpha$ controls divergence from the pre-trained model.  

**Predictive Model:**  
- **Architecture:** A graph neural network (GNN) trained on antibody-antigen structural complexes (e.g., from *PDB*) to predict binding affinity ($\Delta \Delta G$ or $K_D$).  
- **Input:** Antibody-antigen 3D structure (or predicted via *AlphaFold-Multimer*) and sequence features.  
- **Output:** Predicted binding affinity and uncertainty estimates (e.g., Monte Carlo dropout).  
- **Loss Function:** Mean squared error (MSE) with uncertainty calibration:  
  $$ \mathcal{L}_{\text{pred}} = \frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 + \beta \cdot \sigma_i^2, $$  
  where $\sigma_i^2$ is the predictive variance.  

**Synthetic Data Augmentation:**  
To address data scarcity, generate *in silico* mutants using Rosetta *ddG_monomer* and molecular dynamics simulations (GROMACS). An adversarial training regime (similar to AffinityFlow; Chen et al., 2025) refines the predictive model using both real and synthetic data.  

### 2.2 Phase 2: Iterative Active Learning Cycle  
**Step 1: Candidate Generation**  
The generative model proposes $N$ variants (e.g., $N=500$) through temperature-controlled sampling.  

**Step 2: Acquisition Function**  
A batch of $M$ variants (e.g., $M=20$) is selected for experimental testing using a hybrid acquisition strategy:  
$$ a(s) = \lambda_1 \cdot \underbrace{\sigma(s)}_{\text{Uncertainty}} + \lambda_2 \cdot \underbrace{\text{EI}(s)}_{\text{Expected Improvement}} + \lambda_3 \cdot \underbrace{\text{Sim}(s, s_{\text{parent}})}_{\text{Sequence Similarity}}, $$  
where $EI(s) = \mathbb{E}[\max(y(s) - y_{\text{best}}, 0)]$ and $\lambda_1 + \lambda_2 + \lambda_3 = 1$.  

**Step 3: Wet-Lab Validation**  
Selected variants undergo:  
- **Yeast Display Screening:** High-throughput binding affinity measurement.  
- **SPR/BLI Validation:** Quantify kinetic parameters ($K_D$, $k_{\text{on}}$, $k_{\text{off}}$) for top candidates.  

**Step 4: Model Updating**  
- The predictive model is retrained with new data, emphasizing hard examples via focal loss.  
- The generative model is fine-tuned using reinforcement learning, rewarding sequences with high experimental affinity:  
  $$ \mathcal{L}_{\text{RL}} = -\mathbb{E}_{s \sim \pi} \left[ R(s) \cdot \log P(s) \right], $$  
  where $R(s)$ is the affine-transformed binding score.  

### 2.3 Phase 3: Experimental Design & Evaluation  
**Case Study:**  
Validate the framework using a well-characterized antibody-antigen pair (e.g., anti-HER2 trastuzumab).  

**Baselines:** Compare against:  
- **Random Sampling:** Testing randomly selected variants.  
- **Static ML:** Using the initial generative model without active learning.  
- **Bayesian Optimization (CloneBO; Amin et al., 2024).**  

**Metrics:**  
- **In-Silico:**  
  - Affinity prediction RMSE and Spearman’s $\rho$.  
  - Sequence diversity (Hamming distance from parent, HDF score).  
- **Wet-Lab:**  
  - Success rate (% of variants with $K_D < 10$ nM).  
  - Experimental cost reduction vs. random screening.  

**Statistical Analysis:**  
- Compare Pareto frontiers of affinity vs. experimental budget across methods.  
- Use two-tailed t-tests to assess significance ($p < 0.05$) in affinity improvements.  

---

## 3. Expected Outcomes & Impact  
### Expected Outcomes  
1. **High-Affinity Antibodies with Fewer Experiments:** The framework will identify variants with sub-nanomolar $K_D$ values in ≤5 experimental cycles, reducing screening costs by >60% compared to random approaches.  
2. **Improved Model Generalization:** The iterative feedback loop will enhance predictive accuracy (RMSE < 0.8 kcal/mol) and sequence quality (recovery rate >40% in CDR-H3).  
3. **Benchmark Dataset:** Release a curated dataset of antibody variants with experimental $K_D$, $k_{\text{on}}$, and $k_{\text{off}}$ measurements to foster community progress.  

### Broader Impact  
- **Therapeutic Development:** Accelerate the design of antibodies for undruggable targets (e.g., GPCRs) by integrating ML with high-throughput experimentation.  
- **ML Community:** Provide a blueprint for closed-loop active learning in biological design, emphasizing real-world applicability over static benchmarks.  
- **Workshop Alignment:** Directly addresses GEM’s goal of uniting computational and experimental efforts, with potential for fast-tracking in *Nature Biotechnology*.  

By closing the loop between generative models and wet-lab validation, this work will establish a new paradigm for efficient, scalable biomolecular design.