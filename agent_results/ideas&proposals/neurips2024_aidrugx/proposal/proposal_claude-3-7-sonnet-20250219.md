# Multi-Modal Foundation Model for Predicting Therapeutic Outcomes in Cell and Gene Therapies

## 1. Introduction

### Background

Cell and gene therapies represent a paradigm shift in modern medicine, offering unprecedented potential to treat previously intractable diseases by modifying, replacing, or augmenting genetic material within target cells. Despite remarkable clinical successes, such as CAR-T cell therapies for cancer and gene therapies for rare genetic disorders, the development and optimization of these advanced therapeutic modalities continue to face significant challenges. Current approaches to therapeutic design often rely on empirical testing and iterative optimization, which are time-consuming, resource-intensive, and frequently yield unpredictable outcomes.

The complexity of cellular responses to genetic perturbations arises from the intricate network of molecular interactions spanning multiple biological scales and modalities. A single genetic modification can trigger cascading effects across transcriptomic, proteomic, and phenotypic levels, making it challenging to predict therapeutic efficacy, safety profiles, and optimal delivery mechanisms. Traditional computational approaches typically focus on individual biological domains (e.g., gene expression or protein structure prediction), failing to capture the cross-modal interactions that ultimately determine therapeutic outcomes.

Recent advances in artificial intelligence, particularly in foundation models, have demonstrated remarkable capabilities in integrating and learning from diverse data types. Models such as MAMMAL (Shoshan et al., 2024) have shown that unified architectures can effectively handle multiple biological modalities, while specialized approaches like scMMGPT (Shi et al., 2025) have successfully bridged single-cell RNA sequencing data with textual information. Additionally, BioMedGPT (Luo et al., 2023) has established the potential of generative pre-trained transformers to connect biological entities with natural language understanding. However, there remains a critical gap in developing comprehensive models that specifically address the unique challenges of cell and gene therapies by integrating perturbation inputs with multi-modal biological responses.

### Research Objectives

The primary aim of this research is to develop a multi-modal foundation model (MMFM) that integrates genetic/molecular perturbation data with diverse biological readouts to predict therapeutic outcomes in cell and gene therapies. Specifically, we seek to:

1. Design a novel architecture that effectively encodes and aligns perturbation data (e.g., CRISPR edits, gene knockdowns) with multi-modal readouts (transcriptomic, proteomic, and phenotypic responses).

2. Develop pre-training strategies that leverage existing large-scale publicly available datasets to learn generalizable biological representations.

3. Implement an active learning framework that enables efficient fine-tuning through targeted experimental validation.

4. Create interpretable prediction mechanisms that provide actionable insights for optimizing therapeutic design.

5. Validate the model's predictive performance across multiple therapeutic applications, including CRISPR guide design, cell-type-specific delivery optimization, and off-target effect prediction.

### Significance

The proposed research addresses critical bottlenecks in the development pipeline of cell and gene therapies. By accurately predicting the outcomes of genetic perturbations, our model has the potential to:

1. Accelerate therapeutic development by reducing the need for extensive experimental screening and validation cycles.

2. Enhance safety profiles by better predicting and minimizing off-target effects and unintended cellular responses.

3. Improve therapeutic efficacy through optimized design of genetic constructs and delivery systems targeted to specific cell types.

4. Enable personalized medicine approaches by accounting for individual genetic backgrounds in therapeutic response predictions.

5. Provide mechanistic insights into the molecular pathways influenced by genetic modifications, potentially uncovering novel therapeutic targets.

The successful implementation of this model could dramatically reduce the time and cost associated with bringing new cell and gene therapies to clinical trials, ultimately expanding patient access to these transformative treatment modalities.

## 2. Methodology

### 2.1 Data Collection and Preprocessing

Our approach leverages both existing public datasets and targeted experimental data generation to create a comprehensive training and validation resource for the foundation model.

#### 2.1.1 Public Dataset Integration

We will compile and integrate the following public datasets:

1. **Perturbation Data**:
   - DepMap CRISPR screens (20,000+ genes across 1,000+ cell lines)
   - Perturb-seq datasets containing single-cell gene expression profiles following CRISPR perturbations
   - The LINCS L1000 dataset encompassing cellular responses to various genetic and chemical perturbations

2. **Multi-Modal Readouts**:
   - Gene expression data from GTEx and TCGA
   - Proteomic data from CPTAC and the Human Protein Atlas
   - Cell imaging datasets from the Human Protein Atlas and Cell Painting datasets
   - Functional annotations from Gene Ontology and KEGG pathways

#### 2.1.2 Data Standardization

To address the heterogeneity in data formats and experimental protocols, we will implement the following standardization procedures:

1. **Perturbation Representation**: Genetic perturbations will be encoded as a combination of:
   - Target sequence features (k-mer representations of the target DNA/RNA)
   - Perturbation type (e.g., knockout, knockdown, activation)
   - Delivery method metadata (e.g., viral vector, lipid nanoparticle)

2. **Multi-Modal Readout Standardization**:
   - Transcriptomic data will be normalized using TPM/FPKM with batch effect correction
   - Proteomic data will undergo intensity normalization and missing value imputation
   - Phenotypic data will be standardized through feature extraction from cellular images

3. **Cross-Dataset Alignment**:
   - Cell line and tissue type ontologies will be harmonized across datasets
   - Time points will be standardized into categorical bins (immediate, early, late response)
   - A common feature space will be established for each modality to enable cross-dataset learning

### 2.2 Model Architecture

We propose a novel multi-modal foundation model architecture that effectively encodes, aligns, and generates predictions across diverse biological data types (Figure 1).

#### 2.2.1 Encoder Modules

Our architecture incorporates specialized encoders for each data modality:

1. **Sequence Encoder**: A transformer-based encoder for processing genetic sequences, including:
   - DNA/RNA target sequences
   - Guide RNA sequences for CRISPR applications
   - mRNA coding sequences for therapeutic design

   This encoder uses a hierarchical attention mechanism described by:

   $$\text{SeqEmb}(X) = \text{MultiHead}(\text{LayerNorm}(X + \text{Attention}(X)))$$

   Where $X$ represents the input sequence tokens and the attention mechanism is defined as:

   $$\text{Attention}(X) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

2. **Graph Neural Network (GNN)**: For encoding molecular structures and protein-protein interaction networks:

   $$h_v^{(l+1)} = \text{Update}\left(h_v^{(l)}, \text{Aggregate}\left(\{h_u^{(l)}: u \in \mathcal{N}(v)\}\right)\right)$$

   Where $h_v^{(l)}$ represents the node features at layer $l$, and $\mathcal{N}(v)$ denotes the neighbors of node $v$.

3. **Cell State Encoder**: A specialized encoder for cell-type-specific features and cellular context:

   $$C = \text{MLP}(\text{concat}(E_{tissue}, E_{celltype}, E_{state}))$$

   Where $E_{tissue}$, $E_{celltype}$, and $E_{state}$ represent embeddings of tissue type, cell type, and cellular state features.

#### 2.2.2 Cross-Modal Integration

To effectively align information across modalities, we implement a cross-modal attention mechanism:

$$\text{CrossAttn}(Q_i, K_j, V_j) = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$$

Where $Q_i$, $K_j$, and $V_j$ are the query, key, and value projections from modalities $i$ and $j$, respectively.

The integrated representation is formed through:

$$Z = \sum_{i=1}^{M} \sum_{j=1}^{M} \alpha_{ij} \cdot \text{CrossAttn}(Q_i, K_j, V_j)$$

Where $\alpha_{ij}$ are learnable weights determining the importance of each cross-modal interaction, and $M$ is the total number of modalities.

#### 2.2.3 Decoder Architecture

The decoder component generates predictions for different therapeutic outcomes:

1. **Transcriptomic Response Prediction**:
   $$\hat{Y}_{trans} = \text{TransDecoder}(Z)$$

2. **Proteomic Response Prediction**:
   $$\hat{Y}_{prot} = \text{ProtDecoder}(Z)$$

3. **Phenotypic Outcome Prediction**:
   $$\hat{Y}_{pheno} = \text{PhenoDecoder}(Z)$$

4. **Safety Profile Prediction**:
   $$\hat{Y}_{safety} = \text{SafetyDecoder}(Z)$$

Each decoder employs a task-specific architecture optimized for the corresponding output modality.

### 2.3 Training Strategy

Our training approach follows a multi-stage process to effectively leverage both public and experimental data.

#### 2.3.1 Pre-training

We will implement a self-supervised pre-training strategy:

1. **Masked Modality Modeling**: Randomly mask input features from one modality and train the model to predict these from the remaining modalities:

   $$\mathcal{L}_{MMM} = \sum_{i=1}^{M} \mathbb{E}_{X_i \sim \mathcal{D}} \left[ \| \hat{X}_i - X_i \|^2 \right]$$

   Where $\hat{X}_i$ represents the reconstructed features for modality $i$.

2. **Contrastive Learning**: Train the model to align representations of the same biological entity across different modalities:

   $$\mathcal{L}_{CL} = -\log \frac{\exp(s(z_i, z_j^+)/\tau)}{\sum_{k=1}^{N} \exp(s(z_i, z_j^k)/\tau)}$$

   Where $s(·,·)$ is a similarity function, $z_i$ and $z_j^+$ are positive pairs, and $\tau$ is a temperature parameter.

3. **Modality Translation**: Train the model to translate between different biological modalities:

   $$\mathcal{L}_{MT} = \sum_{i=1}^{M} \sum_{j=1}^{M} \mathbb{E}_{X_i \sim \mathcal{D}} \left[ \| f_{i \rightarrow j}(X_i) - X_j \|^2 \right]$$

   Where $f_{i \rightarrow j}$ is a translation function from modality $i$ to modality $j$.

The combined pre-training objective is:

$$\mathcal{L}_{pretrain} = \lambda_1 \mathcal{L}_{MMM} + \lambda_2 \mathcal{L}_{CL} + \lambda_3 \mathcal{L}_{MT}$$

Where $\lambda_1$, $\lambda_2$, and $\lambda_3$ are hyperparameters controlling the contribution of each loss component.

#### 2.3.2 Fine-tuning with Active Learning

To efficiently utilize experimental resources, we implement an active learning framework:

1. **Uncertainty Sampling**: Select perturbation candidates with the highest prediction uncertainty:

   $$x^* = \arg\max_x \mathcal{U}(x)$$

   Where $\mathcal{U}(x)$ represents the model's uncertainty for a candidate $x$, calculated as:

   $$\mathcal{U}(x) = \frac{1}{T} \sum_{t=1}^{T} \| f_\theta^{(t)}(x) - \bar{f}_\theta(x) \|^2$$

   With $f_\theta^{(t)}$ representing model predictions with dropout applied, and $\bar{f}_\theta$ the average prediction.

2. **Diversity Sampling**: Ensure selected candidates cover diverse regions of the perturbation space:

   $$S^* = \arg\max_{S \subset \mathcal{X}, |S|=k} \sum_{x_i, x_j \in S, i \neq j} d(x_i, x_j)$$

   Where $d(·,·)$ is a distance function in the perturbation embedding space.

3. **Model Update**: Periodically retrain the model with newly acquired experimental data:

   $$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t, \mathcal{D}_t \cup \mathcal{D}_{new})$$

   Where $\mathcal{D}_t$ is the existing dataset, $\mathcal{D}_{new}$ is the newly acquired data, and $\eta$ is the learning rate.

### 2.4 Experimental Validation

We will validate our model through a series of experimental studies focusing on key applications in cell and gene therapies.

#### 2.4.1 CRISPR Guide RNA Design Optimization

1. **Experimental Setup**:
   - Design 100 guide RNAs targeting 10 therapeutically relevant genes
   - Evaluate on-target efficiency and off-target effects using GUIDE-seq or CIRCLE-seq
   - Measure transcriptomic responses using RNA-seq
   - Assess phenotypic outcomes through cellular assays

2. **Evaluation Metrics**:
   - On-target efficiency prediction accuracy (Pearson's r)
   - Off-target site prediction (AUROC, AUPRC)
   - Transcriptomic response prediction accuracy (cosine similarity)
   - Phenotypic outcome prediction accuracy (F1-score)

#### 2.4.2 Cell-Type-Specific Delivery Optimization

1. **Experimental Setup**:
   - Test 50 lipid nanoparticle formulations with varying compositions
   - Evaluate delivery efficiency across 5 cell types using fluorescently labeled mRNA
   - Measure cell-type-specific expression and immunogenicity profiles

2. **Evaluation Metrics**:
   - Delivery efficiency prediction accuracy (RMSE)
   - Cell-type specificity prediction (AUROC)
   - Expression level prediction accuracy (Pearson's r)
   - Immunogenicity prediction accuracy (F1-score)

#### 2.4.3 mRNA Therapeutic Design

1. **Experimental Setup**:
   - Design 50 mRNA variants with different UTR and codon optimizations
   - Evaluate translation efficiency, stability, and immunogenicity
   - Measure protein expression levels and duration

2. **Evaluation Metrics**:
   - Translation efficiency prediction accuracy (Pearson's r)
   - Stability prediction accuracy (RMSE)
   - Protein expression prediction accuracy (R²)
   - Duration of expression prediction (RMSE)

### 2.5 Interpretability Analysis

To ensure the model provides actionable insights, we will implement the following interpretability methods:

1. **Attention Visualization**: Extract and visualize attention weights to identify important features:

   $$\text{Importance}(x_i) = \sum_{j} \text{Attention}(x_i, x_j)$$

2. **Integrated Gradients**: Attribute predictions to input features:

   $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha \times (x - x'))}{\partial x_i} d\alpha$$

   Where $x'$ is a baseline input and $f$ is the model prediction function.

3. **Feature Ablation Studies**: Systematically remove or perturb features to assess their impact on predictions:

   $$\text{Impact}(x_i) = |f(x) - f(x_{-i})|$$

   Where $x_{-i}$ represents the input with feature $i$ removed or perturbed.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

The successful implementation of this research is expected to yield several significant outcomes:

1. **A Comprehensive Multi-Modal Foundation Model**: We anticipate developing a state-of-the-art foundation model capable of integrating diverse biological data types to predict therapeutic outcomes in cell and gene therapies. This model will serve as a valuable resource for the wider research community and can be continually refined as new data becomes available.

2. **Improved Therapeutic Design Tools**: Our research will produce specialized tools for optimizing various aspects of therapeutic design, including:
   - CRISPR guide RNA design with enhanced on-target efficiency and minimized off-target effects
   - Cell-type-specific delivery systems tailored to therapeutic requirements
   - mRNA therapeutic designs with optimized stability, translation efficiency, and immunogenicity profiles

3. **Predictive Biomarkers**: Through the analysis of model predictions and interpretability studies, we expect to identify novel biomarkers that predict therapeutic responses across different genetic backgrounds and cellular contexts.

4. **Reduced Experimental Iteration Cycles**: The active learning framework will demonstrate significant reductions in the number of experimental iterations required to optimize therapeutic designs, potentially cutting development time by 30-50%.

5. **Mechanistic Insights**: The interpretability components of our model will reveal new mechanistic insights into how genetic perturbations propagate through cellular networks to produce therapeutic effects, potentially identifying new therapeutic targets and strategies.

6. **Open-Source Software and Datasets**: We will release our model architecture, pre-trained weights, and curated datasets to the research community, facilitating further innovation in computational approaches to cell and gene therapy development.

### 3.2 Impact

The anticipated impact of this research spans scientific, clinical, and societal dimensions:

#### Scientific Impact

This work will advance the field of AI for drug discovery by demonstrating how complex biological systems can be modeled across multiple modalities. The integration of perturbation inputs with diverse biological readouts represents a novel approach that extends beyond current foundation models, which typically focus on individual modalities or simpler prediction tasks. The methodological advances in multi-modal learning and active learning could influence approaches in other domains of biological research.

#### Clinical Impact

The acceleration of therapeutic design and optimization processes enabled by our model could significantly reduce the time and cost associated with bringing new cell and gene therapies to clinical trials. By improving the prediction of therapeutic efficacy and safety profiles, our approach has the potential to increase success rates in clinical development, ultimately delivering more effective treatments to patients with fewer adverse effects.

#### Societal Impact

Cell and gene therapies currently face significant accessibility challenges due to their complex manufacturing processes and high development costs. By streamlining the development pipeline, our research could contribute to reducing the overall cost of these therapies, potentially expanding access to transformative treatments for previously untreatable conditions. Additionally, the improved safety profiles predicted by our model could enhance public confidence in these novel therapeutic modalities.

In conclusion, the proposed multi-modal foundation model represents a significant step forward in applying artificial intelligence to the development of advanced therapeutic modalities. By bridging the gap between genetic perturbations and their complex biological outcomes, our approach addresses critical bottlenecks in the current development pipeline for cell and gene therapies, potentially accelerating the translation of these promising technologies into clinical practice.