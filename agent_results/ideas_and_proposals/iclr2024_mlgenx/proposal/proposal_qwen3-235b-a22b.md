# Causal Discovery through Multi-Omics Perturbation Analysis via Active Learning in Causal Graphical Models  

## Introduction  

### Background  
The integration of machine learning (ML) with genomics has emerged as a transformative approach for drug discovery, particularly in identifying causal relationships between molecular entities (genes, proteins) and phenotypic outcomes. Despite advancements in high-throughput omics technologies—such as single-cell RNA sequencing (scRNA-seq), proteomics, and spatial transcriptomics—our ability to infer causality from observational data remains limited. Current methods predominantly rely on correlational analyses, which are prone to confounding variables and fail to generalize across experimental conditions. This gap contributes to the high attrition rate of drug candidates in clinical trials, where non-causal targets are often validated through costly, iterative experimentation.  

Recent breakthroughs in perturbation biology (e.g., CRISPR-Cas9, RNAi) and multimodal data generation offer unprecedented opportunities to map causal networks. However, the high dimensionality of omics data, coupled with experimental costs, necessitates efficient strategies for perturbation design and causal inference. Active learning—a paradigm that iteratively selects the most informative experiments—has shown promise in reducing redundancy while improving causal discovery. Concurrently, advances in causal graphical models, variational inference, and uncertainty quantification provide theoretical foundations for learning interpretable latent representations.  

### Research Objectives  
This proposal aims to develop a unified framework for causal discovery in genomics by addressing three core objectives:  
1. **Latent Causal Representation Learning**: Integrate multimodal omics data (scRNA-seq, proteomics, spatial omics) into a structured latent space that disentangles causal factors using variational autoencoders (VAEs) with graph-constrained priors.  
2. **Counterfactual Causal Inference**: Estimate interventional effects through counterfactual reasoning over the learned causal graph, enabling robust identification of direct and indirect perturbation targets.  
3. **Active Learning for Experimental Design**: Optimize perturbation experiments by quantifying uncertainty in causal estimates and prioritizing interventions that maximize information gain.  

### Significance  
By bridging causal ML and experimental genomics, this work addresses critical bottlenecks in drug target identification. The proposed framework will:  
- Reduce reliance on trial-and-error experimentation by prioritizing high-impact perturbations.  
- Enable interpretable causal networks that link molecular mechanisms to phenotypes, enhancing reproducibility.  
- Provide open-source tools for uncertainty-aware causal discovery, accelerating hypothesis-driven research in gene therapy, RNA-based drugs, and personalized medicine.  

---

## Methodology  

### Research Design Overview  
The framework combines three pillars (Figure 1):  
1. **Structured Latent Causal Model**: A multimodal VAE learns disentangled representations from heterogeneous data, regularized by a causal graph.  
2. **Counterfactual Inference**: Structural causal models (SCMs) quantify perturbation effects using do-calculus and potential outcomes.  
3. **Active Learning Loop**: Bayesian experimental design selects perturbations that minimize uncertainty in causal effect estimates.  

![Framework Overview](https://via.placeholder.com/600x300?text=Framework+Overview)  
*Figure 1: Workflow integrating multimodal data, causal VAEs, counterfactual inference, and active learning.*  

---

### Data Collection and Preprocessing  
**Datasets**:  
- **Synthetic Benchmarks**: Generated using the *GENIE3* simulator with controlled causal graphs and confounding.  
- **Real-World Data**:  
  - **LINCS L1000**: Gene expression profiles under small-molecule perturbations.  
  - **CRISPR Screen Datasets**: Genome-wide knockout effects from the Broad Institute’s Achilles project.  
  - **Single-Cell Multi-Omics**: Paired scRNA-seq and proteomics data (e.g., CITE-seq) from cancer cell lines.  
  - **Spatial Omics**: Visium spatial transcriptomics datasets with matched histology.  

**Preprocessing**:  
- Normalize modalities separately (e.g., log-transformation for RNA, arcsinh for proteins).  
- Batch effect correction via Harmony or MNN.  
- Construct prior knowledge graphs using Pathway Commons and STRING for protein interactions.  

---

### Algorithmic Components  

#### 1. Structured Variational Autoencoder for Latent Causal Learning  
We model multimodal data as observations $ \mathbf{X} = \{\mathbf{X}^{(1)}, \dots, \mathbf{X}^{(M)}\} $ across $ M $ modalities, with latent causal variables $ \mathbf{Z} $. The encoder-decoder architecture enforces a causal graph $ \mathcal{G} $ over $ \mathbf{Z} $:  

**Encoder**:  
$$
q_{\phi}(\mathbf{Z} \mid \mathbf{X}) = \prod_{i=1}^N \mathcal{N}(\mathbf{z}_i \mid \mu_{\phi}(\mathbf{X}_i), \text{diag}(\sigma_{\phi}^2(\mathbf{X}_i)))
$$  
**Decoder**:  
$$
p_{\theta}(\mathbf{X} \mid \mathbf{Z}, \mathcal{G}) = \prod_{m=1}^M p_{\theta_m}(\mathbf{X}^{(m)} \mid \mathbf{Z})
$$  
**Causal Regularization**:  
The latent variables $ \mathbf{Z} $ are structured by a graph $ \mathcal{G} $, parameterized via a graph Laplacian matrix $ \mathbf{L} $. The VAE loss incorporates a causal constraint term:  
$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_{\phi}}[\log p_{\theta}(\mathbf{X} \mid \mathbf{Z})] - \beta \cdot \text{KL}(q_{\phi}(\mathbf{Z}) \parallel p(\mathbf{Z} \mid \mathcal{G})),
$$  
where $ \beta $ controls the strength of causal regularization.  

#### 2. Counterfactual Causal Inference  
Given a perturbation $ \text{do}(Z_j = z_j') $, we estimate the causal effect on target $ Z_k $ using counterfactuals:  
$$
\text{ATE}(Z_j \rightarrow Z_k) = \mathbb{E}[Z_k \mid \text{do}(Z_j = z_j')] - \mathbb{E}[Z_k \mid \text{do}(Z_j = z_j)].
$$  
This is approximated via the backdoor adjustment formula using the learned graph $ \mathcal{G} $:  
$$
p(Z_k \mid \text{do}(Z_j)) = \sum_{\mathbf{C} \in \text{pa}(Z_j)} p(Z_k \mid Z_j, \mathbf{C}) p(\mathbf{C}),
$$  
where $ \text{pa}(Z_j) $ denotes parents of $ Z_j $ in $ \mathcal{G} $.  

#### 3. Active Learning for Perturbation Design  
At iteration $ t $, the model selects a perturbation $ a_t \in \mathcal{A} $ (e.g., gene knockout) to minimize posterior uncertainty over $ \mathcal{G} $. The acquisition function uses mutual information:  
$$
a_t = \arg\max_{a \in \mathcal{A}} \mathcal{I}(\mathcal{G}; Y \mid a, \mathcal{D}_t),
$$  
where $ \mathcal{D}_t $ is data up to round $ t $, and $ Y $ is the experimental outcome. Approximating this, we use Bayesian neural networks to estimate predictive entropy:  
$$
a_t = \arg\max_{a} \left[ H(Y \mid \mathcal{D}_t) - \mathbb{E}_{\mathcal{G} \sim p(\cdot \mid \mathcal{D}_t)} [H(Y \mid \mathcal{G}, a)] \right].
$$  

---

### Experimental Design and Evaluation  

#### Baselines  
- **Correlation-Based Methods**: WGCNA, Pearson/Spearman networks.  
- **Causal Discovery**: PC-algorithm, GES, NOTEARS.  
- **Deep Learning**: DAG-GNN, Causal VAE (Louizos et al.), CRN (Bica et al.).  

#### Synthetic Evaluation  
- **Metrics**: Structural Hamming Distance (SHD), True Causal Edge Recovery (AUC-ROC), Counterfactual Accuracy.  
- **Scenarios**: Varying confounding strength, missing modalities, spatial heterogeneity.  

#### Real-Data Validation  
- **CRISPR Screens**: Predict knockout effects on gene expression and cell viability.  
- **LINCS**: Forecast drug-induced transcriptional responses.  
- **Spatial Omics**: Infer signaling pathways driving tumor microenvironment interactions.  

#### Uncertainty Quantification  
- **Calibration Metrics**: Expected Calibration Error (ECE), Predictive Entropy.  
- **Robustness**: Sensitivity to missing data and batch effects.  

---

## Expected Outcomes & Impact  

### Anticipated Results  
1. **Interpretable Causal Networks**: Discovery of latent causal drivers in multi-omics datasets, validated against gold-standard pathways (e.g., KEGG, Reactome).  
2. **Efficient Experimental Design**: Reduction in required perturbations by ≥40% compared to random or greedy baselines, measured by ATE estimation accuracy.  
3. **Uncertainty-Aware Inference**: Bayesian credible intervals for causal effects, enabling risk-stratified target prioritization.  

### Scientific and Practical Impact  
- **Drug Discovery**: Accelerate target validation by focusing on high-confidence causal relationships, reducing preclinical trial costs.  
- **Methodological Advancements**: Novel integration of active learning with causal graphical VAEs, advancing ML for hypothesis generation in biology.  
- **Open-Source Tools**: Release of a Python library (*CausalGen*) with pre-trained models and benchmark datasets.  

### Long-Term Vision  
This work lays the foundation for closed-loop discovery systems where ML models iteratively design, execute, and refine biological experiments. By unifying causal inference, multimodal integration, and active learning, we aim to transform genomics into a precision engineering discipline for next-generation therapeutics.  

--- 

This proposal directly addresses key challenges in genomics: high dimensionality via structured latent models, multimodal integration through graph-regularized VAEs, interpretability via counterfactual explanations, and experimental efficiency through active learning. The synthesis of these elements represents a paradigm shift toward hypothesis-driven, causality-based drug development.