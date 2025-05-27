# Causal Discovery through Multi-Omics Perturbation Analysis via Active Learning in Causal Graphical Models  

## 1. Introduction  

### Background  
The identification of causal relationships in genomics is pivotal for understanding disease mechanisms and developing targeted therapies. Traditional approaches rely on observational data, which are confounded by environmental and genetic variables, leading to spurious correlations that fail to reproduce in clinical trials. While perturbation experiments (e.g., CRISPR, RNAi) generate interventional data critical for causal inference, their high cost and technical complexity necessitate strategic experimental design. Concurrently, the proliferation of multimodal omics datasets—from single-cell transcriptomics to spatially resolved proteomics—offers unprecedented opportunities to model the causal interplay between genes, proteins, and phenotypes. However, integrating these datasets into interpretable causal frameworks remains challenging due to high dimensionality, data heterogeneity, and uncertainty in inferred relationships.  

### Research Objectives  
This research aims to:  
1. Develop a causal representation learning framework to infer latent causal structures from multimodal omics data.  
2. Design an active learning strategy to prioritize perturbation experiments that maximize causal graph confidence.  
3. Validate the approach on synthetic and real-world datasets to demonstrate improved accuracy and interpretability over existing methods.  

### Significance  
By integrating causal graphical models with active learning, this work will address key limitations in genomics research: (1) poor reproducibility of correlation-based methods, (2) inefficient allocation of experimental resources, and (3) lack of interpretability in black-box models. The proposed framework will accelerate drug target discovery by generating robust, causally validated hypotheses, reducing the reliance on costly trial-and-error approaches.  

---

## 2. Methodology  

### Research Design  
The framework comprises three interconnected modules:  
1. **Latent Causal Representation Learning**  
2. **Causal Graph Inference via Counterfactual Reasoning**  
3. **Active Experimental Design for Perturbation Selection**  

#### 2.1 Data Collection and Preprocessing  
- **Data Sources**:  
  - *Synthetic Data*: Simulate gene regulatory networks with ground-truth causal edges, perturbing nodes to generate observational and interventional datasets.  
  - *Real Data*: Use the LINCS L1000 dataset (gene expression under CRISPR perturbations) and single-cell multi-omics datasets (e.g., CITE-seq for paired RNA and protein measurements).  
- **Preprocessing**: Normalize data per modality (e.g., log-transform RNA counts), embed spatial omics via graph representations, and align multimodal datasets using shared features.  

#### 2.2 Latent Causal Representation Learning  
A structured variational autoencoder (VAE) is trained to learn low-dimensional latent variables $z$ from multimodal inputs $x$ (e.g., gene expression, protein levels). The VAE architecture includes:  
- **Encoder**: $q_\phi(z | x)$ maps input $x$ to a distribution over latent variables.  
- **Decoder**: $p_\theta(x | z)$ reconstructs $x$ from $z$.  
- **Causal Adjacency Matrix**: A learnable matrix $A \in \mathbb{R}^{d \times d}$ encodes causal relationships between latent variables.  

**Loss Function**:  
The evidence lower bound (ELBO) incorporates reconstruction loss, KL divergence, and a sparsity penalty on $A$:  
$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \parallel p(z)) + \lambda \cdot \|A\|_1,
$$  
where $\beta$ and $\lambda$ control regularization strength.  

#### 2.3 Causal Graph Inference  
Given interventional data $\mathcal{D}_{\text{int}}$ from perturbations, counterfactual queries are used to estimate causal effects. Let $do(z_i = \zeta)$ denote an intervention on latent variable $z_i$. The causal effect on target $z_j$ is:  
$$
\text{Effect}(z_j \mid do(z_i = \zeta)) = \mathbb{E}[z_j \mid do(z_i = \zeta)] - \mathbb{E}[z_j].
$$  
The causal adjacency matrix $A$ is updated via gradient descent to minimize the discrepancy between predicted and observed counterfactuals.  

#### 2.4 Active Learning for Perturbation Design  
An acquisition function $\alpha(z_i)$ quantifies the expected information gain from perturbing $z_i$:  
$$
\alpha(z_i) = H(\hat{A}_{z_i}) - \mathbb{E}_{\zeta \sim p(\zeta)}[H(\hat{A}_{z_i} \mid do(z_i = \zeta))],
$$  
where $H$ is the entropy over the estimated causal edges involving $z_i$. Experiments are prioritized to maximize $\alpha(z_i)$, iteratively refining $A$ until convergence.  

#### 2.5 Experimental Validation  
- **Baselines**: Compare against correlation-based methods (e.g., WGCNA), causal discovery algorithms (PC, FCI), and state-of-the-art models (e.g., CausalGPS [1], GENIE3).  
- **Metrics**:  
  - *Accuracy*: AUROC, F1 score for edge detection.  
  - *Interpretability*: Sparsity and modularity of inferred networks.  
  - *Efficiency*: Number of experiments required to achieve 90% causal confidence.  
- **Validation Steps**:  
  1. Train models on synthetic data with known ground truth.  
  2. Fine-tune on real multi-omics datasets, using held-out perturbations for testing.  
  3. Perform ablation studies to assess contributions of active learning and multimodal integration.  

---

## 3. Expected Outcomes & Impact  

### Expected Outcomes  
1. **Algorithmic Contributions**:  
   - A structured VAE framework for learning latent causal representations from multimodal omics data.  
   - An active learning strategy that reduces the number of required perturbation experiments by ≥30% compared to random selection.  
2. **Empirical Results**:  
   - Improved causal graph accuracy (AUROC >0.85 vs. <0.7 for baselines) on synthetic and LINCS datasets.  
   - Identified causal drivers of disease phenotypes (e.g., oncogenes) validated through literature and CRISPR screens.  
3. **Software Tools**: Open-source implementation of the framework for modular integration with genomics pipelines.  

### Broader Impact  
This work will advance precision medicine by enabling hypothesis-driven drug discovery. By quantifying uncertainty and prioritizing high-impact experiments, it will reduce costs and accelerate the translation of genomic insights into therapies. The interpretable causal networks will enhance collaboration between computational biologists and wet-lab researchers, fostering a deeper understanding of disease mechanisms.  

---  

**References**  
[1] López, R. et al. (2022). Learning Causal Representations of Single Cells. *NeurIPS*.  
[2] Wu, M. et al. (2024). Identifying Perturbation Targets through Causal Differential Networks. *ICML*.  
[3] Johnson, M. et al. (2024). Uncertainty Quantification in Causal Graphical Models. *Bioinformatics*.