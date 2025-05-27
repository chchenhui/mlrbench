Title: Causal Graph-Contrast: A Multimodal Pretraining Framework for Cross-Scale Biological Representations

Motivation:  
Current foundation models excel at single-modality embeddings (e.g., protein sequences or cell images) but struggle to capture causal, cross-scale interactions—from molecular structure to cellular phenotype—and to generalize under unseen perturbations. Bridging this gap is critical for in-silico perturbation simulation, rational drug design, and accurate phenotype prediction.

Main Idea:  
We propose Causal Graph-Contrast, a self-supervised pretraining framework that unifies molecular graphs (e.g., small molecules, protein structures) with cellular graphs extracted from high-content imaging (e.g., cell morphology networks).  
1. Data Integration: Construct heterogeneous graphs linking atom-level nodes, protein domains, and cell-subgraph regions.  
2. Pretraining Tasks:  
  a. Masked Node/Edge Recovery to learn local chemistry and cell-morphology features.  
  b. Cross-Modal Contrastive Learning that pulls together corresponding molecule–cell pairs (e.g., known perturbations) and pushes apart unrelated samples.  
  c. Causal Intervention Modeling using perturbation metadata (e.g., drug dosages, gene knockouts) to disentangle causal from correlative signals.  
3. Evaluation Metrics: Assess generalization on out-of-distribution perturbations, transfer to drug activity prediction, and few-shot phenotype classification.  
Expected outcomes include embeddings that capture mechanistic links across scales, enabling robust in-silico simulation of cellular responses and accelerating biologically informed model design.