Title:
Causal Graph-Contrast: A Multimodal Pretraining Framework for Cross-Scale Biological Representations

1. Introduction  
Background  
Representation learning for biological data has advanced rapidly in the past two years, driven by the availability of large-scale genomics, proteomics, and cell-imaging datasets (e.g., RxRx3, JUMP-CP, Human Cell Atlas). Foundation models trained on single modalities—such as masked language models for protein sequences or convolutional networks for cell images—yield powerful embeddings but often fail to capture the causal, cross-scale interactions that underlie cellular phenotype. Bridging molecular structure (atoms, residues) with cellular morphology (subcellular compartments, tissue architecture) is critical to enable in silico perturbation simulation, rational drug design, and robust phenotype prediction.

Despite progress in multimodal contrastive learning (Lacombe et al., 2023; Rajadhyaksha & Chitkara, 2023) and causal representation learning (Sun et al., 2024; Tejada-Lapuerta et al., 2023), current methods seldom integrate causal intervention metadata (e.g., drug dosage, gene knockouts) into self-supervised pretraining. As a result, embeddings capture mostly correlative patterns and degrade under out-of-distribution (OOD) perturbations.  

Research Objectives  
This proposal aims to develop and validate Causal Graph-Contrast, a self-supervised pretraining framework that:  
• Unifies molecular and cellular data in a single heterogeneous graph capturing atom-to-cell-region interactions.  
• Leverages contrastive and masked recovery tasks to learn rich local and global features.  
• Incorporates causal intervention modeling to disentangle causal signals from confounders.  

Significance  
By fusing cross-scale information and explicitly modeling interventions, our approach will yield embeddings that generalize to unseen perturbations and improve downstream tasks such as drug activity prediction, phenotype classification, and in silico simulation of cellular responses. This positions Causal Graph-Contrast as a foundational step toward AI-driven virtual cell simulators.

2. Methodology  
2.1 Data Collection  
Datasets  
• Molecular Graphs: Public small-molecule libraries (ChEMBL) and protein 3D structures (PDB).  
• Morphological Profiles: High-content cell imaging from JUMP-CP and RxRx3, with single-cell segmentations and feature embeddings.  
• Perturbation Metadata: Drug dosages, gene knockouts, and treatment timepoints associated with paired molecule–cell data.

Data Preprocessing  
• Molecules are converted into attributed graphs $G^{\rm mol}=(V^{\rm mol},E^{\rm mol},X^{\rm mol})$, where nodes $v\in V^{\rm mol}$ carry atom types and edges $e\in E^{\rm mol}$ carry bond orders.  
• Cell images are segmented into regions of interest (ROIs) and represented as cell graphs $G^{\rm cell}=(V^{\rm cell},E^{\rm cell},X^{\rm cell})$, where ROIs form nodes with morphological feature vectors and edges encode spatial adjacency.

2.2 Heterogeneous Graph Construction  
We construct a unified graph $G=(V,E)$ by linking $G^{\rm mol}$ and $G^{\rm cell}$ via “perturbation edges.” For each molecule–cell pair known to interact, we add edges between a special molecule-cloud node $v_0^{\rm mol}$ and cell-cloud node $u_0^{\rm cell}$. This yields  
• $V = V^{\rm mol} \cup V^{\rm cell} \cup \{v_0^{\rm mol},u_0^{\rm cell}\}$  
• $E = E^{\rm mol} \cup E^{\rm cell} \cup \{(v_0^{\rm mol},u_0^{\rm cell})\}$  

2.3 Graph Encoders  
We define two parallel Graph Neural Network (GNN) encoders:  
• Molecular encoder $f_G: G^{\rm mol}\to \mathbb{R}^d$  
• Cellular encoder $f_C: G^{\rm cell}\to \mathbb{R}^d$  

Each encoder is a stack of graph‐attention layers. The cloud nodes $v_0^{\rm mol}$ and $u_0^{\rm cell}$ aggregate global features across modalities. The final cloud embeddings are denoted  
$$
z^{\rm mol} = f_G\bigl(G^{\rm mol}\bigr),\quad
z^{\rm cell} = f_C\bigl(G^{\rm cell}\bigr).
$$  

2.4 Pretraining Tasks  
2.4.1 Masked Node/Edge Recovery  
We randomly mask a subset $V_{\rm mask}\subset V^{\rm mol}\cup V^{\rm cell}$ and/or $E_{\rm mask}\subset E^{\rm mol}\cup E^{\rm cell}$ and train the encoders to predict original attributes. Let $\hat{X}_v$ be the predicted attribute distribution for node $v$. We minimize:  
$$
L_{\rm mask} = -\sum_{v\in V_{\rm mask}}\sum_{k=1}^{K_v} y_{v,k}\,\log \hat{X}_{v,k}
\;+\;
-\sum_{e\in E_{\rm mask}}\sum_{\ell=1}^{L_e} y_{e,\ell}\,\log \hat{X}_{e,\ell},
$$  
where $K_v$ and $L_e$ are the sizes of classification heads.

2.4.2 Cross-Modal Contrastive Learning  
We form positive pairs $(z_i^{\rm mol},z_i^{\rm cell})$ for aligned molecule–cell samples and treat other pairs as negatives. Define cosine similarity $\mathrm{sim}(u,v)=u^\top v/\|u\|\|v\|$ and temperature $\tau$. The InfoNCE loss is:  
$$
L_{\rm NCE} = -\sum_{i=1}^N 
\log\frac{\exp\bigl(\mathrm{sim}(z_i^{\rm mol},z_i^{\rm cell})/\tau\bigr)}
{\sum_{j=1}^N \exp\bigl(\mathrm{sim}(z_i^{\rm mol},z_j^{\rm cell})/\tau\bigr)}.
$$  

2.4.3 Causal Intervention Modeling  
To disentangle causal from spurious correlations, we incorporate perturbation metadata $p_i$ (e.g., dosage, gene knockout). We define an intervention encoder $f_I$ that conditions on $p_i$:  
$$
z_i^{\rm int} = f_I\bigl(z_i^{\rm mol},z_i^{\rm cell},p_i\bigr).
$$  
We enforce that $z_i^{\rm int}$ captures causal shifts by minimizing a regret‐style loss between intervened and observational embeddings:  
$$
L_{\rm causal} = \sum_{i=1}^N \bigl\|z_i^{\rm int} - z_i^{\rm cell}\bigr\|_2^2
\quad\text{s.t.}\quad
\mathrm{Cov}(z^{\rm int}, z^{\rm mol})\;\text{low for non‐causal paths}.
$$  
In practice we implement this via a Hilbert–Schmidt Independence Criterion (HSIC) penalty to reduce dependence on purely correlative channels.

2.5 Total Loss  
The overall pretraining objective is  
$$
L = \lambda_1\,L_{\rm mask} + \lambda_2\,L_{\rm NCE} + \lambda_3\,L_{\rm causal},
$$  
with hyperparameters $\lambda_1,\lambda_2,\lambda_3$.

2.6 Algorithmic Steps  
Algorithm 1 Pretraining Causal Graph-Contrast  
1. Initialize GNN encoders $f_G,f_C,f_I$.  
2. For each mini‐batch of $N$ molecule–cell pairs:  
   a. Construct heterogeneous graphs $G_i$.  
   b. Sample masks $V_{\rm mask},E_{\rm mask}$.  
   c. Compute $z_i^{\rm mol}=f_G(G_i^{\rm mol})$, $z_i^{\rm cell}=f_C(G_i^{\rm cell})$.  
   d. Compute masked recovery loss $L_{\rm mask}$.  
   e. Compute contrastive loss $L_{\rm NCE}$ over the batch.  
   f. Compute $z_i^{\rm int}=f_I(z_i^{\rm mol},z_i^{\rm cell},p_i)$ and $L_{\rm causal}$.  
   g. Backpropagate total loss $L$ and update parameters.

2.7 Experimental Design  
Datasets & Splits  
• Pretraining: ∼1M molecule–cell pairs drawn from JUMP-CP & ChEMBL with 80/10/10 train/val/test splits, ensuring held‐out perturbation types in test.  
• Downstream Tasks:  
  – Drug activity prediction on a ChEMBL benchmark (binary bioactivity).  
  – Few‐shot phenotype classification on RxRx3 (10–50 labeled examples per class).  
  – OOD generalization: evaluate on unseen gene knockouts.  

Baselines  
• Single-modality pretrained GNNs (no contrastive or causal tasks).  
• Multimodal contrastive without causal loss (Rajadhyaksha & Chitkara, 2023).  
• Causal representation learning without cross-modal contrastive (Sun et al., 2024).

Training Protocol  
• Optimizer: AdamW, lr=1e-4, weight decay=1e-5.  
• Batch size: 256 pairs.  
• Temperature $\tau=0.07$, HSIC weight $\lambda_3=0.1$.  
• Early stopping on validation contrastive loss.

Evaluation Metrics  
• Linear probing accuracy: train a linear classifier on frozen embeddings.  
• ROC‐AUC and PR‐AUC for drug activity tasks.  
• Few-shot classification accuracy and F1‐score.  
• Silhouette Coefficient and clustering ARI for embedding quality.  
• OOD robustness gap: difference in accuracy between in‐distribution and OOD perturbations.

Ablation Studies  
• Remove $L_{\rm causal}$ to assess causal modeling benefit.  
• Replace hypergraph with simple bipartite graph to test data‐integration necessity.  
• Vary masking ratio and contrastive temperature.

3. Expected Outcomes & Impact  
Expected Outcomes  
• Embeddings that capture mechanistic links across scales: we anticipate a ≥5% absolute improvement in linear‐probe accuracy on drug activity prediction over state-of-the-art multimodal contrastive baselines.  
• OOD Generalization: by integrating causal interventions, our model should halve the OOD accuracy gap compared to non-causal frameworks.  
• Richer Biological Interpretability: HSIC-guided causal disentanglement will yield dimensions in embedding space that correlate with known biological pathways (validated via enrichment analysis).  
• Few-Shot Phenotype Prediction: we anticipate improvements of 10–15% in low‐shot classification of cell phenotypes under novel perturbations.

Broader Impact  
Causal Graph-Contrast will provide the AIxBio community with:  
• A generalizable pretraining recipe for cross‐scale representation learning, enabling virtual experiments that predict cell behavior under novel drugs or genetic edits.  
• Open‐source code, pretrained weights, and standardized benchmarks (data splits, evaluation scripts) to foster reproducible research.  
• A pathway toward AI‐powered virtual cell simulators that can accelerate drug discovery, reduce wet‐lab costs, and reveal causal mechanisms in cellular systems.  

In summary, our proposed framework addresses two critical challenges in biological representation learning—integrating heterogeneous data modalities and disentangling causal from correlative signals—paving the way for more reliable, interpretable, and generalizable foundation models of living systems.