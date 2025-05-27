**Title:** Causal Foundation Models for Target Discovery in Genomics via Multimodal Perturbation Learning  

**Motivation:**  
Drug discovery failures often stem from a lack of mechanistic understanding of disease biology. While existing machine learning models excel at pattern recognition in omics data, they struggle to distinguish correlation from causation, limiting their utility in identifying viable drug targets. Integrating causal reasoning with foundation models could bridge this gap, enabling the prediction of intervention effects and robust target prioritization.  

**Main Idea:**  
We propose developing a **causal foundation model** trained on multimodal perturbation datasets (e.g., CRISPR screens, drug-response profiles, single-cell RNA-seq) to learn causal representations of gene-disease relationships. The model will combine graph neural networks (GNNs) with structural causal modeling to encode biological knowledge (e.g., protein interactions, pathways) and infer latent causal mechanisms from observational and interventional data. Key innovations include:  
1. A hybrid architecture that integrates GNNs with counterfactual reasoning to model long-range dependencies and perturbation effects.  
2. Self-supervised pre-training on heterogeneous omics data, followed by fine-tuning on disease-specific perturbation datasets.  
3. Active learning to prioritize experiments that validate predicted causal targets, reducing validation costs.  
Expected outcomes include improved accuracy in predicting drug targets, mechanistic insights into disease pathways, and generalizable models across tissues and modalities. This work could accelerate target discovery for complex diseases and inform rational design of RNA-based or gene therapies, directly addressing the workshop’s focus on bridging ML and genomics for drug innovation.