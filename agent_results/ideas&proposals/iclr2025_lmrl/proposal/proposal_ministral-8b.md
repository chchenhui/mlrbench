# Title: Causal Graph-Contrast: A Multimodal Pretraining Framework for Cross-Scale Biological Representations

## Introduction

The field of representation learning for biological data has witnessed remarkable growth, driven by the availability of large-scale public datasets spanning genomics, proteomics, cell imaging, and more. Current foundation models excel at single-modality embeddings but struggle to capture cross-scale interactions and generalize under unseen perturbations. Bridging this gap is critical for in-silico perturbation simulation, rational drug design, and accurate phenotype prediction. This research proposal aims to address these challenges by introducing Causal Graph-Contrast, a self-supervised pretraining framework that unifies molecular graphs with cellular graphs extracted from high-content imaging.

### Research Objectives

1. **Data Integration**: Develop a method to construct heterogeneous graphs linking atom-level nodes, protein domains, and cell-subgraph regions.
2. **Pretraining Tasks**: Design self-supervised pretraining tasks, including masked node/edge recovery, cross-modal contrastive learning, and causal intervention modeling.
3. **Evaluation Metrics**: Assess generalization on out-of-distribution perturbations, transfer to drug activity prediction, and few-shot phenotype classification.
4. **Expected Outcomes**: Generate embeddings that capture mechanistic links across scales, enabling robust in-silico simulation of cellular responses and accelerating biologically informed model design.

### Significance

This research is significant because it addresses the critical need for cross-scale, cross-modal biological representations that generalize well to unseen perturbations. By leveraging causal graph contrastive learning, we aim to develop a framework that not only captures complex biological interactions but also provides interpretable insights into these interactions. This work has the potential to advance the field of computational biology and contribute to real-world applications such as drug discovery and personalized medicine.

## Methodology

### Data Integration

To integrate heterogeneous biological data modalities, we construct a unified graph representation that links atom-level nodes, protein domains, and cell-subgraph regions. This involves the following steps:

1. **Molecular Graph Construction**: Represent small molecules and proteins as molecular graphs, where nodes correspond to atoms or amino acids, and edges represent chemical bonds or domain interactions.
2. **Cellular Graph Construction**: Extract cell-subgraph regions from high-content imaging data, where nodes represent cell regions or organelles, and edges represent morphological or functional connections.
3. **Graph Integration**: Merge the molecular and cellular graphs into a single heterogeneous graph, where inter-modal edges link corresponding entities (e.g., a specific protein domain in a cell).

### Pretraining Tasks

We propose three self-supervised pretraining tasks to learn meaningful representations:

#### 1. Masked Node/Edge Recovery

This task involves masking a subset of nodes or edges in the graph and training the model to predict the original graph structure. The objective is to learn local chemistry and cell-morphology features. The loss function for masked node recovery is defined as:

$$
\mathcal{L}_{\text{masked node}} = \sum_{i=1}^{N} \sum_{j=1}^{M} \left( y_{ij} - \hat{y}_{ij} \right)^2
$$

where \(N\) is the number of nodes, \(M\) is the number of edges, \(y_{ij}\) is the original edge label, and \(\hat{y}_{ij}\) is the predicted edge label.

#### 2. Cross-Modal Contrastive Learning

This task pulls together corresponding molecule–cell pairs (e.g., known perturbations) and pushes apart unrelated samples. The contrastive loss function is defined as:

$$
\mathcal{L}_{\text{contrastive}} = \sum_{i=1}^{K} \left( \frac{1}{2} \left( 1 - \text{sim}(x_i, x_i^+) \right) + \frac{1}{2} \left( 1 - \text{sim}(x_i, x_i^-) \right) \right)
$$

where \(K\) is the number of positive and negative pairs, \(\text{sim}(x_i, x_i^+)\) is the similarity between the positive pair, and \(\text{sim}(x_i, x_i^-) is the similarity between the negative pair.

#### 3. Causal Intervention Modeling

This task leverages perturbation metadata (e.g., drug dosages, gene knockouts) to disentangle causal from correlative signals. The causal intervention loss function is defined as:

$$
\mathcal{L}_{\text{causal}} = \sum_{i=1}^{L} \left( \frac{1}{2} \left( 1 - \text{sim}(x_i, x_i^+) \right) + \frac{1}{2} \left( 1 - \text{sim}(x_i, x_i^-) \right) \right)
$$

where \(L\) is the number of causal interventions, and \(\text{sim}(x_i, x_i^+)\) and \(\text{sim}(x_i, x_i^-) are the similarity between the positive and negative interventions, respectively.

### Evaluation Metrics

To assess the generalization and utility of the learned representations, we propose the following evaluation metrics:

1. **Out-of-Distribution Perturbation Generalization**: Evaluate the model's ability to generalize to unseen perturbations by measuring its performance on a held-out set of perturbations.
2. **Drug Activity Prediction**: Assess the model's transferability to drug activity prediction tasks by measuring its performance on a separate drug activity dataset.
3. **Few-Shot Phenotype Classification**: Evaluate the model's ability to classify phenotypes with few training examples by measuring its performance on a few-shot phenotype classification task.

## Expected Outcomes & Impact

### Expected Outcomes

1. **Cross-Scale Representations**: Generate embeddings that capture mechanistic links across scales, from molecular structure to cellular phenotype.
2. **Robust In-Silico Simulation**: Enable robust in-silico simulation of cellular responses to perturbations.
3. **Biologically Informed Model Design**: Accelerate biologically informed model design by providing interpretable insights into biological mechanisms.

### Impact

This research has the potential to significantly advance the field of computational biology by developing a framework that can capture complex biological interactions and provide interpretable insights into these interactions. By enabling robust in-silico simulation of cellular responses and accelerating biologically informed model design, this work has the potential to contribute to real-world applications such as drug discovery and personalized medicine. Furthermore, by fostering collaboration between AI and biology researchers, this research can help to establish open-source standardization of datasets and evaluation metrics for benchmarking new methods in the growing field of representation learning for biological data.

## Conclusion

In conclusion, Causal Graph-Contrast is a novel self-supervised pretraining framework that unifies molecular graphs with cellular graphs extracted from high-content imaging. By leveraging causal graph contrastive learning, this framework aims to generate cross-scale, cross-modal biological representations that generalize well to unseen perturbations. Through this research, we aim to advance the field of computational biology and contribute to real-world applications in drug discovery and personalized medicine.