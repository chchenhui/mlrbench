**Title: Cross-Scale Consistency Benchmarking for Biological Representation Learning**  

**Motivation:**  
Current biological foundation models lack rigorous evaluation of their ability to capture hierarchical, cross-scale relationships (e.g., molecular → cellular → tissue interactions), limiting their utility in simulating complex biological systems. Existing metrics focus on task-specific performance rather than biologically grounded, scale-aware consistency, hindering progress toward generalizable "virtual cell" models.  

**Main Idea:**  
This work proposes a benchmark framework to train and evaluate multiscale biological representations using datasets that integrate multiple layers (e.g., protein structures, single-cell RNA-seq, and microscopy images). Models are tested via:  
1. **Consistency tasks:** Predict perturbations at one scale (e.g., gene knockout) from another (e.g., cell morphology changes), enforced via contrastive learning.  
2. **Latent space analysis:** Quantify alignment between learned embeddings and known biological hierarchies (e.g., gene ontology) using graph-based metrics.  
3. **Downstream generalization:** Measure performance on unseen tasks across scales (e.g., drug effect prediction from protein interactions).  

Expected outcomes include open-source datasets, a unified evaluation toolkit, and insights into architectural choices (e.g., graph transformers vs. diffusion models) for cross-scale reasoning. This framework would standardize progress toward foundation models that robustly encode biological causality and hierarchy.