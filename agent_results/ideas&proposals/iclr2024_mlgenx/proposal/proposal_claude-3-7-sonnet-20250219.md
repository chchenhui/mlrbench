# ACTIVE-CMAP: Active Learning for Causal Multi-omics Analysis in Perturbation Biology

## 1. Introduction

### 1.1 Background

Understanding the causal mechanisms underlying diseases represents one of the most significant challenges in modern drug discovery. Despite the enormous growth in genomic and multi-omics data, our ability to identify causal relationships in complex biological systems remains limited. This limitation directly impacts drug development processes, with approximately 90% of drug candidates failing in clinical trials, often due to insufficient understanding of disease mechanisms (Cook et al., 2014). The traditional correlation-based approaches to analyzing biological data frequently lead to spurious associations rather than causal insights, resulting in poor reproducibility and costly failures in translational research.

Recent technological advances have created unprecedented opportunities for addressing these challenges. High-throughput perturbation technologies, including CRISPR-Cas9 gene editing, RNA interference (RNAi), and small molecule libraries, now enable systematic manipulation of biological systems at scale. Concurrently, multi-modal omics platforms can capture comprehensive molecular profiles spanning the genome, transcriptome, proteome, and beyond. These complementary capabilities create a fertile ground for causal inference approaches that can learn from both observational and interventional data.

However, a critical gap remains in efficiently integrating these experimental capabilities with computational methods that can extract causal insights. Current approaches often fail to: (1) effectively combine information across multiple omics modalities; (2) incorporate uncertainty quantification necessary for confident causal claims; (3) guide experimental design to maximize information gain while minimizing experimental burden; and (4) produce interpretable models that can inform actionable hypotheses for drug development.

### 1.2 Research Objectives

The proposed research aims to develop a novel computational framework, ACTIVE-CMAP (Active Causal Multi-omics Analysis in Perturbation biology), that addresses these limitations through the integration of causal graphical models with active learning for perturbation biology. Specifically, we aim to:

1. Develop a structured multi-omics representation learning approach that captures latent causal factors across different molecular modalities (genomics, transcriptomics, proteomics).

2. Design an uncertainty-aware causal discovery method that can effectively integrate observational and interventional data to infer causal relationships.

3. Implement an active learning framework to guide experimental design by selecting optimal perturbations that maximize information gain about causal structures.

4. Validate the framework on both synthetic benchmarks and real multi-omics perturbation datasets, with a focus on applications in drug target discovery.

### 1.3 Significance and Innovation

This research offers several innovative contributions with significant potential impact:

First, while most existing methods focus on single data modalities, our approach integrates multiple omics layers within a unified causal framework, enabling more comprehensive biological insights. Second, the active learning component introduces a novel paradigm for experimental design in perturbation biology, potentially reducing experimental costs while maximizing causal information gain. Third, our uncertainty quantification methods will provide confidence estimates for inferred causal relationships, addressing a critical need for reliability in translational applications.

The potential applications of this work include: (1) more accurate identification of drug targets with causal links to disease phenotypes; (2) prediction of off-target effects of potential therapeutics; (3) personalized treatment recommendations based on patient-specific causal networks; and (4) design of combination therapies targeting multiple causal factors. Ultimately, this research could help reduce the high failure rate in drug development by providing more robust, causally-grounded biological insights.

## 2. Methodology

### 2.1 Overall Framework

The ACTIVE-CMAP framework consists of four interconnected components (Figure 1): (1) a multi-modal representation learning module that extracts latent causal factors from heterogeneous omics data; (2) a causal discovery module that infers causal relationships from both observational and interventional data; (3) an active learning module that optimizes the selection of perturbation experiments; and (4) an uncertainty quantification module that estimates confidence in causal inferences. These components operate in an iterative pipeline, with each cycle refining the causal model through new experiments.

### 2.2 Multi-Modal Representation Learning

#### 2.2.1 Structured Variational Autoencoder for Multi-Omics Data

We will develop a structured variational autoencoder (SVAE) to learn latent representations that capture shared causal factors across multiple omics modalities. For each sample $i$, we observe multi-omics data $\mathbf{X}_i = \{\mathbf{x}_i^{(1)}, \mathbf{x}_i^{(2)}, ..., \mathbf{x}_i^{(M)}\}$, where $\mathbf{x}_i^{(m)}$ represents data from modality $m$ (e.g., RNA-seq, proteomics).

The SVAE model assumes that these observations are generated from a set of latent variables $\mathbf{Z}_i = \{\mathbf{z}_i^{shared}, \mathbf{z}_i^{(1)}, \mathbf{z}_i^{(2)}, ..., \mathbf{z}_i^{(M)}\}$, where $\mathbf{z}_i^{shared}$ captures shared information across modalities and $\mathbf{z}_i^{(m)}$ captures modality-specific factors.

The generative model is defined as:

$$p(\mathbf{X}_i, \mathbf{Z}_i) = p(\mathbf{z}_i^{shared}) \prod_{m=1}^{M} p(\mathbf{z}_i^{(m)}|\mathbf{z}_i^{shared}) p(\mathbf{x}_i^{(m)}|\mathbf{z}_i^{shared}, \mathbf{z}_i^{(m)})$$

We implement this using neural networks as follows:
- For the prior: $p(\mathbf{z}_i^{shared}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$
- For modality-specific latents: $p(\mathbf{z}_i^{(m)}|\mathbf{z}_i^{shared}) = \mathcal{N}(\mu_{\theta}^{(m)}(\mathbf{z}_i^{shared}), \sigma_{\theta}^{(m)}(\mathbf{z}_i^{shared}))$
- For observations: $p(\mathbf{x}_i^{(m)}|\mathbf{z}_i^{shared}, \mathbf{z}_i^{(m)}) = f_{\theta}^{(m)}(\mathbf{z}_i^{shared}, \mathbf{z}_i^{(m)})$

where $f_{\theta}^{(m)}$ is a modality-specific decoder network with appropriate distribution (e.g., Gaussian for continuous data, negative binomial for count data).

#### 2.2.2 Sparse Mechanism Shift Modeling for Perturbation Effects

To model the effects of perturbations, we extend the SVAE with a sparse mechanism shift approach. For data collected under perturbation $j$, we model:

$$p(\mathbf{X}_i^j, \mathbf{Z}_i^j) = p(\mathbf{z}_i^{shared}) \prod_{m=1}^{M} p(\mathbf{z}_i^{(m)}|\mathbf{z}_i^{shared}, \mathbf{I}_j) p(\mathbf{x}_i^{(m)}|\mathbf{z}_i^{shared}, \mathbf{z}_i^{(m)})$$

where $\mathbf{I}_j$ represents the intervention information. Following Lopez et al. (2022), we model perturbations as stochastic interventions targeting sparse subsets of latent variables:

$$p(\mathbf{z}_i^{(m)}|\mathbf{z}_i^{shared}, \mathbf{I}_j) = \prod_{k=1}^{d_m} [(1-m_{jk}) \cdot p(z_{ik}^{(m)}|\mathbf{z}_i^{shared}) + m_{jk} \cdot p(z_{ik}^{(m)}|\mathbf{I}_j)]$$

where $m_{jk} \in \{0,1\}$ indicates whether dimension $k$ in modality $m$ is affected by perturbation $j$, and $p(z_{ik}^{(m)}|\mathbf{I}_j)$ is the intervention distribution.

### 2.3 Causal Discovery Module

#### 2.3.1 Causal Graph Structure Learning

We represent the causal structure as a directed acyclic graph (DAG) $\mathcal{G} = (\mathbf{V}, \mathbf{E})$, where vertices $\mathbf{V}$ correspond to latent variables $\mathbf{Z}$ and edges $\mathbf{E}$ represent causal relationships. To learn the graph structure, we adopt a score-based approach with a continuous optimization of the adjacency matrix $\mathbf{A}$:

$$\min_{\mathbf{A}} \mathcal{L}_{score}(\mathbf{A}, \mathbf{Z}) + \lambda \cdot h(\mathbf{A})$$

where $\mathcal{L}_{score}$ is a scoring function measuring how well the graph explains the data, and $h(\mathbf{A})$ is a differentiable constraint ensuring acyclicity, given by:

$$h(\mathbf{A}) = \text{tr}(e^{\mathbf{A} \circ \mathbf{A}}) - d$$

where $\circ$ denotes element-wise product, $d$ is the dimension of $\mathbf{A}$, and $\text{tr}$ is the matrix trace.

For the scoring function, we use a combination of observational and interventional log-likelihoods:

$$\mathcal{L}_{score}(\mathbf{A}, \mathbf{Z}) = \mathcal{L}_{obs}(\mathbf{A}, \mathbf{Z}^{obs}) + \alpha \sum_{j=1}^{J} \mathcal{L}_{int}(\mathbf{A}, \mathbf{Z}^j, \mathbf{I}_j)$$

where $\mathbf{Z}^{obs}$ represents observational data, $\mathbf{Z}^j$ represents data under perturbation $j$, and $\alpha$ controls the weight of interventional data.

#### 2.3.2 Counterfactual Reasoning for Causal Effect Estimation

Given the learned causal graph, we estimate causal effects using counterfactual reasoning. For any two variables $Z_i$ and $Z_j$, the average causal effect (ACE) is defined as:

$$\text{ACE}(Z_i \rightarrow Z_j) = \mathbb{E}[Z_j | do(Z_i = z_i + \delta)] - \mathbb{E}[Z_j | do(Z_i = z_i)]$$

We implement this using the learned model by:
1. Abduction: Infer the posterior distribution over exogenous variables $U$ given observed variables $\mathbf{Z}$
2. Action: Modify the structural equations to reflect the intervention $do(Z_i = z_i)$
3. Prediction: Use the modified model to predict $Z_j$

This process allows us to estimate causal effects even for variables that have not been directly perturbed in experiments.

### 2.4 Active Learning Module

#### 2.4.1 Information-Theoretic Perturbation Selection

The active learning component selects perturbations that maximize information gain about the causal structure. For each candidate perturbation $I_c$, we compute the expected information gain:

$$\text{EIG}(I_c) = \mathbb{E}_{p(\mathbf{Z}^c|I_c)} [H(p(\mathbf{A}|\mathcal{D})) - H(p(\mathbf{A}|\mathcal{D} \cup \{\mathbf{Z}^c, I_c\}))]$$

where $H$ denotes entropy, $p(\mathbf{A}|\mathcal{D})$ is the posterior distribution over graph structures given current data $\mathcal{D}$, and $\mathbf{Z}^c$ is the predicted outcome of perturbation $I_c$.

Since exact computation of EIG is intractable, we approximate it using a Monte Carlo approach:

$$\text{EIG}(I_c) \approx \frac{1}{S} \sum_{s=1}^{S} \log \frac{p(\mathbf{A}^{(s)}|\mathcal{D} \cup \{\mathbf{Z}^c, I_c\})}{p(\mathbf{A}^{(s)}|\mathcal{D})}$$

where $\mathbf{A}^{(s)}$ are samples from the posterior $p(\mathbf{A}|\mathcal{D})$.

#### 2.4.2 Experimental Budget Optimization

Given a budget constraint of $B$ perturbation experiments, we formulate a batch selection problem:

$$\max_{C \subset \{1,2,...,K\}, |C| \leq B} \text{EIG}(I_C)$$

where $I_C$ represents the set of selected perturbations. Since this is a submodular optimization problem, we employ a greedy algorithm that iteratively selects the perturbation with the highest marginal information gain.

### 2.5 Uncertainty Quantification Module

#### 2.5.1 Bayesian Inference for Causal Graphs

To quantify uncertainty in causal relationships, we implement a Bayesian inference approach for the adjacency matrix $\mathbf{A}$. We approximate the posterior distribution $p(\mathbf{A}|\mathcal{D})$ using variational inference:

$$p(\mathbf{A}|\mathcal{D}) \approx q_{\phi}(\mathbf{A}) = \prod_{i,j} \text{Bernoulli}(A_{ij}|\sigma(f_{\phi}(i,j)))$$

where $f_{\phi}$ is a neural network parameterized by $\phi$ that outputs logits for each potential edge, and $\sigma$ is the sigmoid function.

The variational parameters are optimized by minimizing:

$$\mathcal{L}_{VI}(\phi) = \mathbb{E}_{q_{\phi}(\mathbf{A})}[\mathcal{L}_{score}(\mathbf{A}, \mathbf{Z})] + \lambda \cdot \mathbb{E}_{q_{\phi}(\mathbf{A})}[h(\mathbf{A})] + \text{KL}(q_{\phi}(\mathbf{A}) || p(\mathbf{A}))$$

where $p(\mathbf{A})$ is a sparsity-inducing prior.

#### 2.5.2 Edge Confidence and Causal Effect Uncertainty

For each potential causal relationship, we compute confidence scores:

$$\text{Conf}(Z_i \rightarrow Z_j) = \mathbb{P}_{q_{\phi}}(A_{ij} = 1) = \sigma(f_{\phi}(i,j))$$

Additionally, we quantify uncertainty in causal effect estimates by sampling from the posterior:

$$\text{ACE}_s(Z_i \rightarrow Z_j) = \text{ACE}(Z_i \rightarrow Z_j | \mathbf{A}^{(s)})$$

where $\mathbf{A}^{(s)} \sim q_{\phi}(\mathbf{A})$. From these samples, we compute 95% credible intervals and coefficient of variation to assess the reliability of causal effect estimates.

### 2.6 Experimental Design and Validation

#### 2.6.1 Synthetic Data Benchmarks

We will evaluate our method on synthetic datasets with known ground truth causal structures:

1. Generate synthetic multi-omics data using a ground truth causal graph with realistic biological characteristics
2. Simulate perturbation experiments by modifying the data generating process
3. Apply ACTIVE-CMAP and baseline methods to infer causal relationships
4. Evaluate performance using metrics such as:
   - Structural Hamming Distance (SHD) between inferred and true graphs
   - Area Under the Precision-Recall Curve (AUPRC) for edge prediction
   - Mean Squared Error (MSE) for causal effect estimation
   - Number of experiments required to achieve a specified accuracy threshold

#### 2.6.2 Real-World Datasets

We will validate our approach on multiple real-world datasets:

1. **LINCS L1000 dataset**: Contains gene expression profiles after perturbation with diverse small molecules and genetic reagents
2. **Perturb-seq data**: Single-cell RNA-seq with CRISPR perturbations
3. **Multi-omics cancer datasets**: Such as TCGA with gene expression, methylation, and proteomics data

For these datasets, we will:
1. Split data into training, validation, and test sets
2. Apply ACTIVE-CMAP to learn causal relationships
3. Validate predictions against known biological pathways from databases like KEGG
4. Perform prospective validation of selected high-confidence predictions through targeted experiments

#### 2.6.3 Case Study: Drug Target Discovery

As a focused application, we will apply our framework to identify potential drug targets for a specific disease context:

1. Select disease-relevant datasets (e.g., inflammatory bowel disease, neurodegenerative disorders)
2. Apply ACTIVE-CMAP to identify causal factors with high confidence
3. Validate predicted targets against existing literature and drug databases
4. Design validation experiments for novel predictions
5. Evaluate the framework's ability to prioritize targets with higher likelihood of successful drug development

## 3. Expected Outcomes & Impact

### 3.1 Immediate Outcomes

The successful completion of this research will yield several concrete outcomes:

1. A novel computational framework (ACTIVE-CMAP) integrating causal inference, multi-omics representation learning, and active learning for experimental design in perturbation biology.

2. Open-source software implementing this framework, with comprehensive documentation and tutorials to facilitate adoption by the scientific community.

3. Benchmark datasets for evaluating causal inference methods in multi-omics contexts, including synthetic data with ground truth and preprocessed real-world datasets.

4. A comprehensive evaluation of the framework's performance across diverse biological contexts, providing insights into its strengths and limitations.

5. Interpretable causal graphs linking genes, proteins, and phenotypes in specific disease contexts, with quantified uncertainty for each relationship.

### 3.2 Scientific Impact

This research will advance scientific understanding in several key areas:

1. **Methodological advances in causal inference**: The integration of representation learning with causal discovery and active learning represents a significant advance over current approaches, potentially establishing new standards for causal inference in complex biological systems.

2. **Improved understanding of biological mechanisms**: By distinguishing correlation from causation, our approach will help clarify the actual drivers of disease processes, potentially resolving conflicting findings in the literature.

3. **More efficient experimental design**: The active learning component will demonstrate how computational approaches can guide experimental design, reducing the cost and time required for biological discovery.

4. **Enhanced interpretability of multi-omics data**: Our structured representation learning approach will provide new insights into how different molecular layers interact, advancing systems biology understanding.

### 3.3 Practical Impact

Beyond scientific advancements, our research holds potential for significant practical impact:

1. **Accelerated drug target discovery**: By identifying causal factors with high confidence, our approach could reduce reliance on trial-and-error approaches in target selection, potentially increasing success rates in drug development.

2. **Reduced experimental costs**: The active learning framework will optimize experimental resources by prioritizing the most informative perturbations, potentially saving millions in research costs.

3. **More robust translational research**: By quantifying uncertainty in causal relationships, our approach will help prioritize findings most likely to translate successfully to clinical applications.

4. **New collaborative paradigms**: The framework establishes a data-driven cycle between computational prediction and experimental validation, potentially fostering closer collaboration between computational and experimental researchers.

### 3.4 Challenges and Mitigation Strategies

We anticipate several challenges in implementing our proposed framework:

1. **Computational scalability**: High-dimensional omics data presents computational challenges. We will address this through dimensionality reduction techniques, efficient implementation, and parallelization.

2. **Model identifiability**: Ensuring that latent representations capture true causal factors is challenging. We will incorporate domain knowledge constraints and validate with ground truth experiments.

3. **Experimental validation costs**: Full validation of all predictions may be prohibitively expensive. We will prioritize validation based on uncertainty quantification and potential impact.

4. **Integration of heterogeneous data types**: Multi-omics data varies in scale, sparsity, and noise characteristics. Our modality-specific encoders and decoders will be designed to address these challenges.

Despite these challenges, the potential benefits of accurate causal inference in genomics justify the ambitious nature of this research. By bridging machine learning with genomics and perturbation biology, ACTIVE-CMAP has the potential to transform our understanding of disease mechanisms and accelerate the development of effective therapies.