# **Research Proposal: Causal Graph-Contrast - Unifying Molecular and Cellular Representations Across Scales with Causal Inference**

## 1. Introduction

### 1.1 Background
The field of biological representation learning is undergoing a rapid transformation, driven by the deluge of large-scale datasets spanning genomics, proteomics, transcriptomics, and high-content imaging (Rozenblatt-Rosen et al., 2021; Fay et al., 2023). The Learning Meaningful Representations of Life (LMRL) community recognizes the critical need to develop methods that not only process these vast datasets but also extract representations that are truly "meaningful"—capable of capturing underlying biological mechanisms, generalizing across diverse conditions, and powering downstream predictive and generative tasks (Chandrasekaran et al., 2023). While recent "foundation models" have shown promise within specific modalities (e.g., protein language models, image-based cell embedding models), a significant challenge remains: effectively integrating information across different biological scales and modalities to understand complex causal relationships.

Biological systems are inherently multiscale and interconnected. The function of a cell, its response to stimuli, and ultimately the phenotype of an organism arise from intricate interactions starting at the molecular level (e.g., drug-protein binding, gene regulation) and propagating through cellular structures and processes. Current models often struggle to bridge this micro-to-macro gap. They may generate powerful embeddings for molecules *or* cells in isolation, but fail to capture the causal link between a specific molecular perturbation (like a drug treatment) and its resulting cellular phenotype (observed via imaging). This limitation hinders progress towards key goals such as accurate *in silico* simulation of biological processes, prediction of drug efficacy and toxicity under novel conditions, rational drug design, and the ambitious vision of a comprehensive "virtual cell" (Bunne et al., 2024).

Recent advances in multimodal learning, particularly using contrastive approaches, have shown success in aligning representations from different data types, such as molecular graphs and text descriptions (Lacombe et al., 2023), multi-omics data (Rajadhyaksha & Chitkara, 2023), or knowledge graphs with text (Dang et al., 2025). Graph contrastive learning techniques have also been enhanced to incorporate multi-modal attributes and topological features (Saifuddin et al., 2025; Chen et al., 2024). Furthermore, integrating multimodal data, such as histology images and spatial context, has improved tasks like gene expression prediction (Min et al., 2024), and similar techniques fuse image and non-image data in medical contexts (Ding et al., 2024). However, many existing multimodal frameworks focus primarily on correlation rather than causation. Concurrently, causal representation learning is emerging as a crucial direction for biology (Sun et al., 2024; Tejada-Lapuerta et al., 2023), aiming to disentangle causal factors from spurious correlations, which is essential for robust generalization, especially under interventions or perturbations.

This proposal introduces **Causal Graph-Contrast**, a novel self-supervised pretraining framework designed to learn unified, causally-informed representations that span molecular and cellular scales. By explicitly modeling the relationship between molecular entities (small molecules, proteins) represented as graphs and cellular states (derived from high-content imaging) also represented as graphs, and incorporating causal reasoning based on perturbation metadata, we aim to create embeddings that capture the mechanistic links driving cellular responses. This approach directly addresses the LMRL workshop's key questions regarding the data, models, and algorithms needed for meaningful representations and their evaluation, particularly focusing on multimodality, multiscale connections, causality, and generalization under perturbations.

### 1.2 Research Objectives
The primary goal of this research is to develop and rigorously evaluate the Causal Graph-Contrast framework for learning meaningful cross-scale biological representations.

The specific objectives are:

1.  **Develop a Heterogeneous Graph Representation:** Design and implement a method for constructing unified graphs that integrate molecular structure information (e.g., from small molecules or proteins) with quantitative cellular features derived from high-content microscopy images (e.g., cell morphology networks). This involves defining appropriate node types, features, and edges that meaningfully connect the molecular and cellular scales based on experimental context (e.g., perturbation experiments).
2.  **Design and Implement a Multitask Self-Supervised Pretraining Framework:** Develop the Causal Graph-Contrast pretraining strategy comprising three core components:
    *   Intra-modal representation learning via masked node/edge feature reconstruction to capture fine-grained details within molecular and cellular graphs.
    *   Cross-modal contrastive learning to align the representations of corresponding molecular perturbations and resulting cellular states.
    *   Causal intervention modeling to explicitly incorporate information about experimental interventions (e.g., drug identity, dosage, gene knockouts) to disentangle causal effects from correlations and improve generalization to unseen perturbations.
3.  **Evaluate Representation Quality and Generalization:** Systematically evaluate the learned representations on a diverse set of downstream tasks, focusing on:
    *   **Out-of-Distribution (OOD) Generalization:** Assess the model's ability to predict cellular responses or molecular properties under conditions (e.g., novel molecules, concentrations, genetic backgrounds) not seen during pretraining.
    *   **Transfer Learning:** Measure the performance improvement gained by using the pre-trained embeddings (either frozen or fine-tuned) on tasks like drug activity prediction, mechanism of action (MoA) classification, and cell phenotype prediction.
    *   **Few-Shot Learning:** Evaluate the data efficiency of the learned representations by assessing performance on downstream tasks with limited labeled samples.
4.  **Benchmark and Ablate:** Compare the performance of Causal Graph-Contrast against relevant baseline methods, including single-modality pretraining, non-causal multimodal contrastive learning, and potentially other state-of-the-art biological foundation models. Conduct ablation studies to quantify the contribution of each component of the proposed framework (masking, contrastive alignment, causal modeling).

### 1.3 Significance
This research holds significant potential to advance the field of biological representation learning and its applications.

*   **Addressing Key LMRL Challenges:** It directly tackles core themes of the LMRL workshop, including multimodal learning (molecules and cell images), multiscale modeling (linking molecular structure to cellular phenotype), incorporating causality, improving generalization (especially under perturbation), and developing robust evaluation strategies.
*   **Bridging the Molecular-Cellular Gap:** By creating representations that explicitly link molecular perturbations to cellular outcomes, this work can provide deeper insights into biological mechanisms, moving beyond simple correlations often captured by existing models.
*   **Enhancing *In Silico* Prediction and Simulation:** The causally-informed, cross-scale embeddings are expected to significantly improve the accuracy and reliability of predicting cellular responses to novel drugs or genetic interventions, paving the way for more powerful *in silico* models and simulations, contributing towards the concept of a "virtual cell".
*   **Accelerating Drug Discovery:** Improved prediction of drug efficacy, toxicity, and MoA, particularly under OOD conditions, can streamline the drug discovery and development pipeline, reducing costs and timelines.
*   **Methodological Innovation:** The proposed Causal Graph-Contrast framework, integrating graph neural networks, contrastive learning, and causal inference principles for cross-scale biological data, represents a novel methodological contribution with potential applicability beyond the specific molecular-cellular context studied here.
*   **Benchmark Development:** The rigorous evaluation protocol, focusing on OOD generalization and causal aspects, will contribute valuable benchmarks and insights for assessing future representation learning models in biology.

## 2. Methodology

This section details the proposed research plan, including data acquisition and processing, the Causal Graph-Contrast framework architecture and pretraining tasks, and the experimental design for validation.

### 2.1 Data Collection and Preparation

We will primarily leverage large-scale, publicly available datasets that pair molecular perturbations with high-content cellular imaging data. Key candidates include:

*   **JUMP-Cell Painting Consortium (JUMP-CP):** This dataset provides Cell Painting images for ~140,000 genetic and chemical perturbations (including ~120,000 compounds) across diverse cell types (U2OS initial release). It offers rich morphological profiles (~7,000 features per cell) linked to specific perturbagens (molecule SMILES strings, gene identifiers). This is ideal as it provides the necessary molecule-perturbation-phenotype linkage.
*   **RxRx Datasets (e.g., RxRx1, RxRx3):** These datasets contain high-content fluorescence microscopy images of cells under various small molecule treatments, often across multiple cell types and experimental conditions. They provide well-structured image data and associated metadata.

**Data Processing Steps:**

1.  **Molecular Graph Construction:** For small molecule perturbations (e.g., from JUMP-CP), we will convert SMILES strings into molecular graphs using libraries like RDKit. Nodes will represent atoms, and edges will represent chemical bonds. Node features will include atom type, charge, chirality, hybridization, etc. Edge features will encode bond type (single, double, etc.). For protein targets (if applicable, e.g., gene knockouts affecting a protein), we might use protein structure graphs (from PDB if available) or simplified sequence-based graph representations. Let $G_m = (V_m, E_m, X_m, W_m)$ denote a molecular graph, where $V_m$ are atoms, $E_m$ are bonds, $X_m$ are atom features, and $W_m$ are bond features.
2.  **Cellular Graph Construction:** This is a crucial step where we represent populations of cells or single-cell morphological states as graphs.
    *   **Image Segmentation:** Use pre-computed features from datasets like JUMP-CP or apply segmentation algorithms (e.g., CellProfiler, StarDist, U-Net) to identify individual cells and nuclei in the microscopy images.
    *   **Feature Extraction:** Extract morphological, intensity, and texture features for each cell using standard toolkits (e.g., CellProfiler features, or features learned by a pre-trained image encoder like a ResNet or Vision Transformer). Let $x_c \in \mathbb{R}^d$ be the feature vector for a single cell.
    *   **Graph Definition:** Construct a graph $G_c = (V_c, E_c, X_c, W_c)$ representing the cellular state under a specific perturbation. Several strategies are possible:
        *   *Population Graph:* Nodes $V_c$ could represent clusters of morphologically similar cells within the population, or perhaps statistical summaries (mean, variance) of features across the population. Edges $E_c$ could represent relationships between these clusters or summaries.
        *   *Cell Neighbourhood Graph:* Nodes $V_c$ represent individual cells. Edges $E_c$ connect spatially neighbouring cells (e.g., within a certain radius or k-nearest neighbours based on centroids). Node features $X_c$ are the morphological vectors $x_c$. Edge features $W_c$ could include distance or relative orientation.
        *   *Single-Cell Graph (if resolution allows):* Nodes could represent sub-cellular components (nucleus, cytoplasm, organelles) with edges representing spatial adjacency. Features would correspond to component morphology/texture. (This is more challenging and depends heavily on image quality/resolution).
    *   We will initially focus on the Cell Neighbourhood Graph approach, as it retains single-cell variability while capturing spatial context.
3.  **Heterogeneous Graph Construction & Pairing:** We need to link molecular graphs $G_m$ with their corresponding cellular graphs $G_c$. The core linkage comes from the experimental metadata: a specific molecule $m$ applied at concentration $z$ results in cell state $c$. We can represent the full dataset as a collection of triplets $(G_m, I, G_c)$, where $I$ represents the intervention context (e.g., $I = (\text{molecule_id}, \text{concentration}, \text{cell_type}, \text{treatment_duration}))$. For pretraining, positive pairs will be $(G_m, G_c)$ derived from the same experiment $(G_m, I, G_c)$, while negative pairs will involve $(G_m, G_{c'})$ where $G_{c'}$ comes from a different intervention $I'$.

### 2.2 Causal Graph-Contrast Framework

The framework employs Graph Neural Networks (GNNs) to process both molecular and cellular graphs, learning embeddings that are refined through self-supervised tasks.

**Architecture:**

*   **Molecular Encoder ($f_m$):** A GNN (e.g., Graph Convolutional Network - GCN, Graph Attention Network - GAT, Message Passing Neural Network - MPNN) that operates on molecular graphs $G_m$ to produce node embeddings $H_m = f_m(G_m) \in \mathbb{R}^{|V_m| \times d_{emb}}$ and potentially a graph-level embedding $h_m = \text{Pool}(H_m) \in \mathbb{R}^{d_{emb}}$.
*   **Cellular Encoder ($f_c$):** A potentially different GNN architecture suitable for the structure of $G_c$ (e.g., GAT might be effective for cell neighbourhood graphs) to produce node embeddings $H_c = f_c(G_c) \in \mathbb{R}^{|V_c| \times d_{emb}}$ and a graph-level embedding $h_c = \text{Pool}(H_c) \in \mathbb{R}^{d_{emb}}$. We will explore parameter sharing between $f_m$ and $f_c$ where appropriate.
*   **Projection Heads ($g_m, g_c$):** Non-linear projection heads (e.g., small MLPs) applied to the graph-level embeddings ($z_m = g_m(h_m)$, $z_c = g_c(h_c)$) before the contrastive loss, a common practice in contrastive learning.

**Self-Supervised Pretraining Tasks:**

1.  **Task 1: Masked Feature Reconstruction (Intra-modal Learning):** To ensure encoders capture fine-grained local information.
    *   Randomly mask a fraction of node features in $G_m$ and $G_c$.
    *   Use the GNN output node embeddings $H_m, H_c$ to predict the original masked features using simple linear or MLP decoders.
    *   Loss function: Combines losses for both modalities, e.g., using mean squared error (MSE) for continuous features or cross-entropy for categorical features.
        $$L_{mask} = \mathbb{E}_{G_m, G_c} \left[ \sum_{v \in \text{Masked}(V_m)} \text{Loss}(Decoder_m(H_m[v]), X_m[v]) + \sum_{v' \in \text{Masked}(V_c)} \text{Loss}(Decoder_c(H_c[v']), X_c[v']) \right]$$

2.  **Task 2: Cross-Modal Contrastive Learning (Alignment):** To align representations of causally linked molecule-cell pairs.
    *   Given a batch of $N$ molecule-cell pairs $(G_{m_i}, G_{c_i})$ derived from interventions $(G_{m_i}, I_i, G_{c_i})$.
    *   Obtain projected graph-level embeddings $z_{m_i} = g_m(h_{m_i})$ and $z_{c_i} = g_c(h_{c_i})$.
    *   Use the InfoNCE loss to pull representations of corresponding pairs $(z_{m_i}, z_{c_i})$ together and push apart non-corresponding pairs $(z_{m_i}, z_{c_j})$ for $i \neq j$.
        $$L_{contrast} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(z_{m_i}, z_{c_i}) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_{m_i}, z_{c_j}) / \tau)} + \log \frac{\exp(\text{sim}(z_{c_i}, z_{m_i}) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_{c_i}, z_{m_j}) / \tau)} \right]$$
    where $\text{sim}(u, v) = u^T v / (||u|| ||v||)$ is cosine similarity and $\tau$ is a temperature hyperparameter.

3.  **Task 3: Causal Intervention Modeling (Disentanglement & Generalization):** To explicitly leverage intervention information $I$ and encourage the model to learn causal mechanisms rather than just correlations. We propose incorporating intervention context and predicting intervention effects.
    *   **Intervention Context:** Represent the intervention $I = (\text{molecule_id}, \text{concentration}, \text{cell_type}, ...)$ as a vector $v_I$. This vector can be concatenated to node features or integrated within the GNN message passing steps (e.g., using FiLM layers) for both $f_m$ and $f_c$, conditioning the embeddings on the specific experimental setup.
    *   **Predicting Intervention Outcomes:** Introduce an auxiliary task to predict the cellular state embedding $h_c$ given the molecular embedding $h_m$ and the intervention details $v_I$. This forces the model to learn the transformation induced by the intervention.
        $$L_{causal\_pred} = \mathbb{E}_{(G_m, I, G_c)} [ || h_c - \text{Predictor}(h_m, v_I) ||^2 ]$$
        where Predictor is an MLP or transformer-based module.
    *   **(Alternative/Complementary) Invariance Regularization:** If data includes experiments across different "environments" (e.g., different cell lines, batches, labs treated as potentially confounding factors $\epsilon$), we could adapt ideas from invariant risk minimization (IRM) or similar techniques. The goal would be to learn representations $h_m, h_c$ such that a predictor of cellular state from molecular state ($h_c \approx P(h_m, I)$) performs consistently across environments $\epsilon$. This encourages learning stable, causal relationships. $L_{causal\_inv} = \text{Variance}_{\epsilon \in \text{Environments}} (\text{Risk}_{\text{data} \sim \epsilon} [P(h_m, I), h_c])$. This requires careful definition of environments.

**Overall Pretraining Objective:**
The final loss is a weighted sum of the individual task losses:
$$L_{total} = L_{mask} + \lambda_{con} L_{contrast} + \lambda_{caus} L_{causal}$$
where $\lambda_{con}$ and $\lambda_{caus}$ are hyperparameters balancing the contributions of the tasks. We will perform hyperparameter tuning for GNN architectures, embedding dimensions, $\tau$, and loss weights $\lambda$.

### 2.3 Experimental Design and Evaluation

**Datasets and Splitting:**
*   Use JUMP-CP and/or RxRx datasets.
*   **Train/Validation Split:** Standard random split (e.g., 80%/20%) of perturbations for model development and hyperparameter tuning.
*   **Test Splits:**
    *   **IID (In-Distribution):** Held-out set of perturbations similar to the training set (e.g., known molecules at seen concentration ranges).
    *   **OOD (Out-of-Distribution):** Carefully designed splits to test generalization:
        *   *Unseen Molecules:* Test on perturbations involving molecules whose scaffolds/classes were not present in training.
        *   *Unseen Concentrations:* Test on concentrations outside the range seen during training.
        *   *Unseen Cell Types/Contexts:* If data permits, test on cell types or experimental batches held out from training (relevant for causal invariance).
        *   *Unseen Combinations:* Test on combinations of known molecules/genes not seen together during training (if combination perturbation data is available).

**Baselines:** To demonstrate the effectiveness of Causal Graph-Contrast, we will compare against:
*   **Single-Modality Baselines:**
    *   Molecule-only: Pre-trained molecular GNN (e.g., using masked atom prediction or graph contrastive learning like GraphCL) evaluated on downstream molecular tasks.
    *   Cell-only: Pre-trained cell image/graph encoder (e.g., CNN on images, GNN on cell graphs with contrastive learning on perturbations) evaluated on downstream cellular tasks.
*   **Multimodal (Non-Causal) Baselines:**
    *   Simple Concatenation: Concatenate embeddings from separate molecule and cell encoders.
    *   Standard Cross-Modal Contrastive Learning: Implement our framework *without* Task 3 ($L_{causal}$), similar to Lacombe et al. (2023) or Rajadhyaksha & Chitkara (2023) adapted to molecule-cell graphs.
*   **Existing Foundation Models (if applicable):** Compare against publicly available relevant foundation models pre-trained on similar data types, if feasible to adapt for comparison on our tasks.

**Downstream Evaluation Tasks:**

1.  **OOD Cellular Response Prediction:** Given $G_m$ and intervention $I$ from an OOD test set, predict properties of $G_c$ (e.g., predict the graph embedding $h_c$, or predict specific morphological profiles/phenotypes derived from $G_c$). Use metrics like MSE (for embedding prediction) or Accuracy/F1/AUC (for phenotype classification).
2.  **Drug Activity Prediction (Transfer Learning):** Use the pre-trained molecular embeddings $h_m$ (potentially combined with $h_c$ if cellular context is relevant for the assay) as input features for predicting biological activity in standard benchmark datasets (e.g., MoleculeNet assays, ChEMBL activity data). Evaluate using AUC-ROC, AUPRC, RMSE.
3.  **Mechanism of Action (MoA) Classification (Transfer Learning):** Use the learned embeddings (primarily $h_c$, perhaps informed by $h_m$) to classify the MoA of compounds based on their cellular profiles (e.g., using JUMP-CP MoA annotations). Evaluate using classification accuracy, F1-score.
4.  **Few-Shot Learning:** Evaluate performances on tasks 2 and 3 using only a small number (e.g., 1, 5, 10) of labeled examples per class/task for fine-tuning or k-NN classification on embeddings.

**Intrinsic Evaluation:**
*   **Embedding Visualization:** Use t-SNE or UMAP to visualize $h_m$ and $h_c$ embeddings, colored by molecule class, MoA, perturbation type, concentration, etc., to qualitatively assess separation and structure.
*   **Disentanglement Metrics:** If possible, adapt metrics to quantify how well factors like molecule identity, concentration, and cell state are disentangled in the latent space.
*   **Ablation Studies:** Systematically remove each loss component ($L_{mask}, L_{contrast}, L_{causal}$) and architectural element (e.g., causal conditioning) to quantify its specific contribution to performance on downstream tasks, especially OOD generalization. We will also ablate different cellular graph construction methods.

**Evaluation Metrics:**
*   **Regression Tasks:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared ($R^2$).
*   **Classification Tasks:** Accuracy, F1-Score (macro/micro), Area Under the Receiver Operating Characteristic Curve (AUC-ROC), Area Under the Precision-Recall Curve (AUPRC).
*   **Clustering/Similarity:** Silhouette Score, qualitative assessment via visualization.
*   **Generalization Gap:** Difference in performance between IID and OOD test sets.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

We anticipate the following key outcomes from this research:

1.  **A Novel Pretraining Framework:** The successful development and implementation of the Causal Graph-Contrast framework, providing a publicly available codebase and methodology for learning cross-scale, causally-informed biological representations from multimodal data.
2.  **High-Quality Biological Representations:** Generation of embeddings ($h_m, h_c$) that effectively capture both molecular properties and cellular states, while crucially encoding the causal relationship between them under perturbation. These representations are expected to cluster biologically related entities meaningfully (e.g., drugs with similar MoA inducing similar cellular graph embeddings).
3.  **Improved Generalization Performance:** Demonstration, through rigorous OOD testing, that Causal Graph-Contrast significantly surpasses baseline models in predicting biological outcomes under novel conditions (unseen molecules, concentrations, contexts). This improvement will stem directly from incorporating causal intervention modeling alongside multimodal contrastive learning.
4.  **Enhanced Transferability:** Evidence that the pre-trained representations provide substantial performance gains when transferred to various downstream tasks (drug activity prediction, MoA classification) with minimal fine-tuning, particularly in low-data regimes (few-shot learning).
5.  **Quantitative Validation of Causal Modeling:** Ablation studies will quantify the specific benefits derived from the causal intervention modeling component ($L_{causal}$), highlighting its importance for robust biological representation learning.
6.  **Insights into Cross-Scale Mechanisms:** The learned representations and the model's predictive capabilities may offer novel biological insights into how molecular features drive specific cellular morphological changes, potentially highlighting key pathways or structural determinants.
7.  **Benchmarks for Future Research:** Establishment of a clear experimental setup, including dataset processing, OOD splits, downstream tasks, and evaluation metrics, serving as a benchmark for future research in cross-scale causal representation learning in biology.

### 3.2 Impact

The successful completion of this research project is expected to have a multi-faceted impact:

*   **Scientific Advancement:** This work will push the boundaries of representation learning in biology by integrating multimodality, multiscale relationships, and causal inference. It offers a concrete step towards building more mechanistic and predictive computational models of biological systems, contributing to the long-term vision of *in silico* biology and the "virtual cell." By focusing on causality, it addresses a critical limitation of many current AI models in biology, enabling more reliable scientific discovery and hypothesis generation.
*   **Accelerating Biomedical Research:** The framework and learned representations can be directly applied to accelerate drug discovery and development. Improved prediction of *in vitro* responses, MoA prediction, and off-target effects for novel compounds under diverse conditions can significantly de-risk and streamline the pre-clinical pipeline. It can also aid in understanding disease mechanisms by modeling how genetic or environmental perturbations manifest at the cellular level.
*   **Contribution to the AIxBio Community:** Aligned with the goals of the LMRL workshop, this research provides novel methods for learning representations across modalities and scales, proposes rigorous evaluation strategies focusing on generalization and causality, and addresses the need for models that capture meaningful biological interactions. The findings, code, and potentially pre-trained models will be shared with the community, fostering further research and collaboration.
*   **Methodological Influence:** The integration of graph contrastive learning with causal modeling techniques for heterogeneous, cross-scale data offers a methodological template that could inspire similar approaches in other scientific domains where understanding causal interactions across scales and modalities is crucial (e.g., materials science, climate modeling, neuroscience).

In conclusion, Causal Graph-Contrast represents a significant step towards building more powerful, reliable, and interpretable AI models for biology. By tackling the critical challenge of learning causal, cross-scale representations, this research promises substantial scientific and translational impact, directly contributing to the core themes and goals of the LMRL community.