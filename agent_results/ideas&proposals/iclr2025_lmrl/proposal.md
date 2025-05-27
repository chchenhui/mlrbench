# Causal Graph-Contrast: A Multimodal Pretraining Framework for Cross-Scale Biological Representations

## 1. Introduction

Recent years have witnessed unprecedented growth in the availability of large-scale biological datasets across various modalities, including DNA and RNA sequencing, protein sequences and 3D structures, mass spectrometry, and high-content cell imaging (Rozenblatt-Rosen et al., 2021; Fay et al., 2023; Chandrasekaran et al., 2023). This data explosion has catalyzed the development of foundation models aimed at extracting meaningful representations from high-dimensional biological data. While these models have shown remarkable success within individual modalities, they often fail to capture the complex causal relationships that exist across different biological scales—from molecular structures to cellular phenotypes.

The ability to learn representations that bridge these scales is crucial for advancing our understanding of biological systems and developing more effective therapeutic interventions. Current approaches typically focus on single-modality embeddings (e.g., protein language models or cell image encoders) but struggle to model how molecular perturbations propagate through biological systems to produce observable phenotypic changes. This limitation severely constrains their utility for in-silico simulation of cellular responses to novel compounds, rational drug design, and accurate phenotype prediction under unseen conditions.

### Research Objectives

This research proposal introduces Causal Graph-Contrast, a novel self-supervised pretraining framework designed to learn meaningful representations that span multiple biological scales while capturing causal relationships. Our specific objectives are to:

1. Develop a unified graph-based representation that integrates molecular structures (small molecules, proteins) with cellular morphology features derived from high-content imaging.

2. Design self-supervised pretraining tasks that explicitly model causal relationships between molecular interventions and cellular responses.

3. Create evaluation methodologies that rigorously assess the learned representations' ability to generalize to unseen perturbations and transfer to downstream tasks.

4. Demonstrate the utility of the framework for in-silico simulation of cellular responses to novel molecular interventions.

### Significance

The proposed research addresses a critical gap in current biological representation learning approaches by explicitly modeling cross-scale causal relationships. Successfully bridging molecular and cellular scales through causal modeling would represent a significant advancement with several important implications:

1. **Enhanced drug discovery**: By accurately predicting cellular responses to novel compounds, we can accelerate the identification of promising therapeutic candidates and reduce reliance on costly experimental screening.

2. **Improved understanding of disease mechanisms**: Cross-scale representations can reveal how molecular aberrations manifest as cellular phenotypes in disease states, providing insights into pathological processes.

3. **More efficient experimental design**: Causal representations enable scientists to prioritize the most informative experiments, reducing the resources needed for biological discovery.

4. **Foundation for in-silico cellular simulation**: This work contributes to the long-term goal of developing comprehensive virtual cell models (Bunne et al., 2024) capable of simulating cellular function and behavior under various conditions.

## 2. Methodology

Our proposed Causal Graph-Contrast framework consists of three main components: (1) a multimodal data integration architecture, (2) a suite of self-supervised pretraining tasks, and (3) a comprehensive evaluation strategy. We describe each component in detail below.

### 2.1 Multimodal Data Integration Architecture

#### 2.1.1 Heterogeneous Graph Construction

We represent biological entities and their relationships using a heterogeneous graph structure $G = (V, E, \mathcal{T}_V, \mathcal{T}_E)$, where $V$ is the set of nodes, $E$ is the set of edges, $\mathcal{T}_V$ is the set of node types, and $\mathcal{T}_E$ is the set of edge types. Our graph incorporates the following key components:

1. **Molecular subgraphs**: For small molecules, nodes represent atoms with edges representing chemical bonds. For proteins, nodes represent amino acids or protein domains with edges representing spatial proximity or functional interactions.

2. **Cellular morphology subgraphs**: Derived from high-content imaging data, nodes represent cellular regions or organelles with edges representing spatial adjacency or functional relationships.

3. **Cross-scale edges**: Connect molecular entities to cellular components based on known interactions (e.g., drug targets) or co-occurrence patterns in experimental data.

Formally, we define the node set $V = V_M \cup V_C$, where $V_M$ comprises molecular nodes and $V_C$ comprises cellular nodes. The edge set $E = E_M \cup E_C \cup E_{MC}$ includes molecular edges $E_M$, cellular edges $E_C$, and cross-scale edges $E_{MC}$.

#### 2.1.2 Node and Edge Features

Each node and edge in the graph is associated with a feature vector:

- **Molecular node features**: For atoms, features include atomic number, formal charge, hybridization state, etc. For protein nodes, features include amino acid properties, secondary structure information, and evolutionary conservation scores.

- **Cellular node features**: Derived from cell imaging data, include morphological descriptors, texture features, and intensity measurements.

- **Edge features**: Include bond types for molecular graphs, distance measures for spatial relationships, and confidence scores for predicted interactions.

#### 2.1.3 Graph Neural Network Architecture

We employ a heterogeneous graph transformer (HGT) architecture (Hu et al., 2020) to process the multimodal graph. The HGT layer is defined as:

$$\mathbf{h}_i^{(l+1)} = \sum_{t \in \mathcal{T}_V} \sum_{j \in \mathcal{N}_i^t} \alpha_{i,j}^t \cdot \mathbf{W}_t^V \mathbf{h}_j^{(l)}$$

where $\mathbf{h}_i^{(l)}$ is the feature vector of node $i$ at layer $l$, $\mathcal{N}_i^t$ is the set of neighbors of node $i$ with type $t$, $\alpha_{i,j}^t$ is the attention coefficient, and $\mathbf{W}_t^V$ is a type-specific transformation matrix.

The attention mechanism is defined as:

$$\alpha_{i,j}^t = \frac{\exp\left(\mathbf{a}_{\tau(i),t}^T \cdot \text{LeakyReLU}\left(\mathbf{W}_{\tau(i),t}^Q \mathbf{h}_i^{(l)} \| \mathbf{W}_t^K \mathbf{h}_j^{(l)}\right)\right)}{\sum_{k \in \mathcal{N}_i^t} \exp\left(\mathbf{a}_{\tau(i),t}^T \cdot \text{LeakyReLU}\left(\mathbf{W}_{\tau(i),t}^Q \mathbf{h}_i^{(l)} \| \mathbf{W}_t^K \mathbf{h}_k^{(l)}\right)\right)}$$

where $\tau(i)$ denotes the type of node $i$, $\mathbf{a}_{\tau(i),t}$ is a learnable attention vector, and $\|$ represents concatenation.

### 2.2 Self-Supervised Pretraining Tasks

We propose three complementary pretraining tasks designed to learn meaningful cross-scale representations:

#### 2.2.1 Masked Node/Edge Recovery

This task involves randomly masking node and edge features, then training the model to recover the masked attributes. Formally, given a graph $G$, we create a corrupted version $\tilde{G}$ by masking a subset of nodes $V_{\text{mask}} \subset V$ and edges $E_{\text{mask}} \subset E$. The objective is to minimize:

$$\mathcal{L}_{\text{mask}} = \sum_{i \in V_{\text{mask}}} \|\mathbf{h}_i - f_{\text{node}}(\mathbf{z}_i)\|^2 + \sum_{(i,j) \in E_{\text{mask}}} \|\mathbf{e}_{ij} - f_{\text{edge}}(\mathbf{z}_i, \mathbf{z}_j)\|^2$$

where $\mathbf{h}_i$ is the original feature vector of node $i$, $\mathbf{e}_{ij}$ is the original feature vector of edge $(i,j)$, $\mathbf{z}_i$ is the encoded representation of node $i$ in the corrupted graph, and $f_{\text{node}}$ and $f_{\text{edge}}$ are projection heads for node and edge feature recovery, respectively.

#### 2.2.2 Cross-Modal Contrastive Learning

We employ contrastive learning to align representations of corresponding molecular and cellular entities. For each molecular-cellular pair $(m, c)$ that are known to interact (positive pairs), we sample negative pairs $(m, c')$ where $c'$ is a cellular entity not known to interact with $m$.

The contrastive loss is defined as:

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_m, \mathbf{z}_c) / \tau)}{\sum_{c' \in \mathcal{B}} \exp(\text{sim}(\mathbf{z}_m, \mathbf{z}_{c'}) / \tau)}$$

where $\mathbf{z}_m$ and $\mathbf{z}_c$ are the encoded representations of molecular entity $m$ and cellular entity $c$, respectively, $\text{sim}(\cdot, \cdot)$ is the cosine similarity, $\tau$ is a temperature parameter, and $\mathcal{B}$ is the batch of cellular entities including both positive and negative examples.

#### 2.2.3 Causal Intervention Modeling

This task explicitly models the causal relationship between molecular interventions and cellular responses. We leverage perturbation metadata (e.g., drug dosages, gene knockouts) to construct counterfactual examples.

Let $G = (V, E, \mathbf{X})$ be a graph with node features $\mathbf{X}$, and let $I$ be an intervention that modifies a subset of molecular nodes. We denote the graph after intervention as $G_I = (V, E, \mathbf{X}_I)$. The causal intervention task involves predicting the cellular node features in $G_I$ given the features in $G$ and the intervention $I$.

The loss function is:

$$\mathcal{L}_{\text{causal}} = \sum_{i \in V_C} \|\mathbf{x}_{I,i} - f_{\text{causal}}(\mathbf{z}_i, \mathbf{z}_I)\|^2$$

where $\mathbf{x}_{I,i}$ is the feature vector of cellular node $i$ after intervention, $\mathbf{z}_i$ is the encoded representation of node $i$ before intervention, and $\mathbf{z}_I$ is a representation of the intervention.

#### 2.2.4 Combined Pretraining Objective

The overall pretraining objective combines the three tasks:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{mask}} + \lambda_2 \mathcal{L}_{\text{contrast}} + \lambda_3 \mathcal{L}_{\text{causal}}$$

where $\lambda_1$, $\lambda_2$, and $\lambda_3$ are hyperparameters controlling the relative importance of each task.

### 2.3 Implementation Details

#### 2.3.1 Datasets

We will leverage the following datasets for pretraining:

1. **JUMP-CP**: A large-scale dataset of cellular images with corresponding molecular perturbations (Chandrasekaran et al., 2023).

2. **RxRx3**: A dataset of cellular responses to various compound treatments (Fay et al., 2023).

3. **Human Cell Atlas**: Single-cell transcriptomics data providing molecular signatures of different cell types (Rozenblatt-Rosen et al., 2021).

4. **PubChem and ChEMBL**: Chemical structure databases providing molecular graphs for small molecules.

5. **PDB and AlphaFold DB**: Protein structure databases providing 3D coordinates for protein domain modeling.

#### 2.3.2 Data Preprocessing

1. **Molecular Graph Construction**: For small molecules, we use RDKit to convert SMILES strings to molecular graphs. For proteins, we use BioPython to extract backbone and side chain information from PDB files.

2. **Cell Morphology Graph Construction**: We apply cell segmentation algorithms to high-content images, then construct graphs based on cellular compartments and their spatial relationships.

3. **Cross-Scale Edge Definition**: We use known drug-target interactions from DrugBank and other databases to establish initial cross-scale edges, supplemented with co-occurrence patterns mined from experimental data.

#### 2.3.3 Training Procedure

We employ a staged training approach:

1. **Initial pretraining**: Train on the masked node/edge recovery task alone to learn basic feature representations.

2. **Cross-modal alignment**: Introduce the contrastive learning objective to align molecular and cellular representations.

3. **Causal fine-tuning**: Incorporate the causal intervention modeling task to refine the representations.

We use the Adam optimizer with a learning rate of $10^{-4}$ and a batch size of 256. Training is performed on 8 NVIDIA A100 GPUs with gradient accumulation to effectively handle larger batch sizes.

### 2.4 Evaluation Strategy

We design a comprehensive evaluation strategy to assess the quality and utility of the learned representations:

#### 2.4.1 Zero-Shot and Few-Shot Transfer

We evaluate the representations on downstream tasks with limited labeled data:

1. **Drug activity prediction**: Predict whether a compound will show activity against a specific target or pathway.

2. **Compound toxicity prediction**: Predict cellular toxicity of novel compounds.

3. **Cell type classification**: Classify cells based on their morphological features.

For each task, we assess performance with varying numbers of labeled examples (0, 1, 5, 10, and 20 per class).

#### 2.4.2 Out-of-Distribution Generalization

We evaluate the model's ability to generalize to unseen perturbations:

1. **Novel compound classes**: Test performance on structurally distinct compounds not seen during training.

2. **Novel cellular contexts**: Test performance on cell types or conditions not seen during training.

3. **Dose-response relationships**: Assess the model's ability to capture non-linear dose-dependent effects.

#### 2.4.3 Causal Consistency Evaluation

We assess the causal consistency of the learned representations:

1. **Intervention consistency**: Verify that interventions on the same target produce consistent cellular responses regardless of the specific intervention.

2. **Counterfactual accuracy**: Measure the model's accuracy in predicting cellular responses to counterfactual interventions.

3. **Do-calculus tests**: Apply Pearl's do-calculus to verify that the model captures causal rather than merely correlational relationships.

#### 2.4.4 Evaluation Metrics

For classification tasks:
- Accuracy, F1-score, AUROC, AUPRC

For regression tasks:
- Mean squared error (MSE), Pearson correlation, Spearman correlation

For causal consistency:
- Intervention consistency score (ICS): $\text{ICS} = \frac{1}{|I|} \sum_{i \in I} \text{sim}(f(G, i), f(G, i'))$, where $i$ and $i'$ are interventions targeting the same molecular entity, and $f(G, i)$ is the predicted cellular response.

- Counterfactual accuracy: Measured as the correlation between predicted and actual cellular responses to held-out interventions.

## 3. Expected Outcomes & Impact

### 3.1 Technical Outcomes

1. **Cross-scale embeddings**: A unified representation framework that captures relationships between molecular structures and cellular phenotypes.

2. **Causal intervention model**: A model capable of predicting cellular responses to novel molecular interventions with improved accuracy compared to existing methods.

3. **Open-source software**: A comprehensive software package implementing the Causal Graph-Contrast framework, making it accessible to the broader research community.

4. **Benchmark datasets**: Curated datasets for evaluating cross-scale biological representations, facilitating future research in this area.

### 3.2 Scientific Impact

1. **Mechanistic insights**: The learned representations will reveal mechanistic relationships between molecular interventions and cellular responses, potentially uncovering novel biological pathways.

2. **Improved drug discovery**: By accurately predicting cellular responses to novel compounds, our framework can accelerate the identification of promising therapeutic candidates and reduce late-stage failures.

3. **Personalized medicine**: The ability to model cellular responses in different genetic contexts can contribute to more personalized therapeutic approaches.

4. **Foundation for in-silico biology**: This work represents a step toward the long-term goal of developing comprehensive virtual cell models capable of simulating cellular function and behavior under various conditions.

### 3.3 Broader Impact

1. **Reduced experimental burden**: By enabling more accurate in-silico predictions, our framework can reduce the need for costly and time-consuming wet-lab experiments.

2. **Educational applications**: The visualizable cross-scale representations can serve as educational tools for understanding complex biological systems.

3. **Interdisciplinary collaboration**: This work bridges machine learning and biology, fostering collaboration between these fields and potentially leading to novel insights in both domains.

The Causal Graph-Contrast framework represents a significant advancement in biological representation learning, addressing the critical need for models that capture cross-scale causal relationships. By explicitly modeling how molecular perturbations propagate to cellular phenotypes, this framework will enable more accurate prediction of biological responses to novel interventions, accelerating discovery across multiple domains of biology and medicine.