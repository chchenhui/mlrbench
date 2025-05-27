1. Title  
Permutation-Equivariant Contrastive Embeddings over Neural Network Weight Spaces for Efficient Model Zoo Retrieval

2. Introduction  
Background  
Over the last few years, the number of publicly available pre-trained neural network models has exploded—Hugging Face alone now hosts over one million models. Practitioners no longer ask “Which architecture should I train?” but rather “Which of the million existing networks should I fine-tune?” Current discovery methods rely on metadata (task tags, model family, training dataset), which often fails to capture deep functional similarities encoded in raw weight tensors. As a result, redundant training and wasted compute resources remain common.  

A recent line of work treats neural network weights as a new data modality, characterized by unique symmetries (permutations, scalings), high dimensionality, and non-Euclidean structure. Early works in this space include flow-based generative modeling of weight spaces (Erdogan 2025), meta-classification of weight vectors (Eilertsen et al. 2020), and permutation-invariant embeddings for retrieval (2024). Contrastive learning tailored to weights has shown promise (2023, 2025), but existing methods either ignore rich layer-wise graph structure or fail to achieve full symmetry equivariance.  

Objectives  
This proposal aims to develop a unified, scalable framework for embedding entire neural networks into a low-dimensional space that:  
• Respects layer-wise symmetries—neuron permutations and filter scalings—via equivariant graph encoding.  
• Leverages contrastive learning on symmetry-preserving augmentations to cluster functionally similar models.  
• Enables high-precision, k-NN retrieval of pre-trained networks for rapid transfer learning.  

Significance  
A robust weight-space retrieval system will:  
• Dramatically reduce redundant training by surfacing networks that already implement desired functions.  
• Lower computational cost and carbon footprint of deep learning research.  
• Democratize access to the best models for downstream tasks across vision, NLP, scientific computing, and beyond.  
• Establish a principled foundation for future work in weight synthesis, model merging, and automatic architecture search.

3. Methodology  
3.1 Data Collection and Preprocessing  
– Model Zoo Curation: We will download ~50 000 – 100 000 pre-trained models from Hugging Face covering classification (ImageNet variants, CIFAR), segmentation (Cityscapes), vision transformers, MLPs, ResNets, and lightweight architectures (MobileNet, EfficientNet).  
– Standardization: For each model, extract weight tensors $\{W^1,\ldots,W^L\}$ and bias vectors $\{b^1,\ldots,b^L\}$. Normalize each tensor by its Frobenius norm:  
  $$\widetilde W^l = \frac{W^l}{\|W^l\|_F},\quad \widetilde b^l = \frac{b^l}{\|b^l\|_2}\,. $$  
– Graph Construction: Represent each layer $l$ as a directed bipartite graph $G^l=(V^l_{\text{in}}\cup V^l_{\text{out}},E^l)$:  
  • $V^l_{\text{in}}$ nodes correspond to input neurons, $V^l_{\text{out}}$ to output neurons.  
  • Each edge $(i\to j)\in E^l$ carries feature $A^l_{ij}=\widetilde W^l_{ij}$.  
  • Node features $h_i^{(0)}$ initialize as zero vectors or incorporate bias terms.  

3.2 Permutation-Equivariant Graph Encoder  
We adopt a shared Graph Neural Network (GNN) architecture $\Phi_\theta$ that guarantees equivariance to neuron permutations and scaling. For layer $l$, define node embeddings $h_i^{(t)}$ at message-passing step $t$:  
  $$m_i^{(t+1)} = \sum_{j\in N(i)} \frac{A^l_{ji}}{\sqrt{\deg(i)\,\deg(j)}}\,\phi_\theta\bigl(h_j^{(t)}\bigr)\,, $$
  $$h_i^{(t+1)} = \sigma\Bigl(\psi_\theta\bigl(h_i^{(t)}\bigr) + m_i^{(t+1)}\Bigr)\,, $$
where $\phi_\theta,\psi_\theta$ are learnable MLPs and $\sigma$ is ReLU. After $T$ steps, apply a readout:  
  $$z^l = \text{READOUT}\bigl(\{h_i^{(T)}\mid i\in V^l_{\text{in}}\cup V^l_{\text{out}}\}\bigr)\,, $$
using mean-pooling.  

3.3 Model-Level Aggregation  
Aggregate layer embeddings $\{z^1,\dots,z^L\}$ into a global model vector:  
  $$z = \text{Agg}\bigl(\{z^l\}_{l=1}^L\bigr)\,, $$
where Agg can be a secondary Transformer or a simple MLP that concatenates or sums projections of each $z^l$. This two-stage encoding shares weights across layers, enabling the network to generalize across architectures of varying depth.  

3.4 Contrastive Learning with Symmetry-Preserving Augmentations  
We generate positive pairs by applying two random symmetry transformations to each model:  
  – Neuron permutation $P$: permute rows and columns of each $\widetilde W^l$, updating node identities accordingly.  
  – Filter scaling $S$: sample per-neuron scales $s_i\sim\text{LogNormal}(0,\sigma^2)$, apply $\widetilde W^l \leftarrow D_{\text{out}}\,\widetilde W^l\,D_{\text{in}}^{-1}$.  

Let $z_i$ and $z_i^+$ be embeddings of two augmentations of the same model. For a batch of $N$ models and their augmentations, define the NT-Xent loss  
  $$\mathcal{L}_{\mathrm{contrastive}} = -\sum_{i=1}^N \log \frac{\exp\!\bigl(\mathrm{sim}(z_i,z_i^+)/\tau\bigr)}{\sum_{j=1}^N \exp\!\bigl(\mathrm{sim}(z_i,z_j^+)/\tau\bigr)}\,, $$
where $\mathrm{sim}(u,v)=u^\top v/\|u\|\|v\|$ and $\tau$ is a temperature hyperparameter.  

Optionally, if each model $i$ has a known downstream performance score $y_i$ (e.g., top-1 accuracy on a held-out dataset), we add  
  $$\mathcal{L}_{\mathrm{sup}} = \frac1N\sum_{i=1}^N \bigl\|g_\phi(z_i)-y_i\bigr\|_2^2\,, $$
where $g_\phi$ is a small MLP regressor. The full objective is  
  $$\mathcal{L} = \mathcal{L}_{\mathrm{contrastive}} + \lambda\,\mathcal{L}_{\mathrm{sup}}\,. $$

3.5 Training Details  
• Optimizer: AdamW, initial learning rate $1\mathrm{e}{-4}$, weight decay $1\mathrm{e}{-5}$.  
• Batch size: 256 models, each with two augmentations.  
• Epochs: 200 with cosine-annealing learning rate schedule.  
• Negative Sampling: in-batch negatives suffice; we will also explore a memory bank of size 65 536 for larger batches.  

3.6 Experimental Validation and Evaluation Metrics  
Baselines  
– Metadata search (tags, architecture names).  
– Embedding methods from “Permutation-Invariant Neural Network Embeddings” (2024).  
– Contrastive weight space learning (2023, 2025).  

Retrieval Task  
Given a query model $q$, retrieve the top-$k$ pre-trained models $\{r_1,\dots,r_k\}$ whose embeddings are nearest to $z_q$. Evaluate:  
• Precision@k (P@1, P@5, P@10) based on whether retrieved models share the same task (e.g., classification on same dataset).  
• Mean Reciprocal Rank (MRR).  

Clustering Coherence  
Cluster embeddings using k-means (with known number of tasks). Report Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).  

Transfer Learning Efficiency  
For a held-out downstream dataset (e.g., CIFAR-100, Pascal VOC), compare:  
• Fine-tuning from the top-1 retrieved model vs. (i) random model in zoo, (ii) best metadata match.  
• Measure initial few-shot performance (after 10 epochs) and final top-1 accuracy after 100 epochs.  

Ablation Studies  
• Remove permutation equivariance (use raw adjacency).  
• Remove scaling invariance.  
• Replace our GNN backbone with Geom-GCN (Pei et al. 2020) or GraphSAGE.  
• Vary depth $T$ of message passing.  

Scalability Analysis  
Measure embedding time per model and memory footprint for 50 000 models. Extrapolate to 1 million.  

4. Expected Outcomes & Impact  
We anticipate the following key outcomes:  
1. A principled, publicly-released PyTorch library implementing permutation-equivariant, contrastive weight encoders that scale to tens of thousands of models.  
2. Empirical demonstration of significant retrieval gains—e.g., Precision@5 improvements of 10–20 points over metadata baselines, 5–10 points over prior embeddings.  
3. Transfer learning speedups: top-1 retrieved models should reach target accuracy 2× faster than random or metadata-matched baselines.  
4. A comprehensive ablation report quantifying the importance of each symmetry and architectural choice.  
5. A held-out benchmark suite (“Model Zoo Retrieval Challenge”) for the community to evaluate future weight-space methods.  

Broader Impact  
By transforming the way practitioners discover pre-trained models, this research will:  
– Cut down redundant compute and energy consumption in deep learning.  
– Accelerate progress in domains that lack abundant compute resources.  
– Lay groundwork for automated model merging, weight interpolation, and generative weight modeling.  
– Stimulate interdisciplinary research at the intersection of representation learning, graph theory, and deep learning theory.  

5. References  
[1] Ege Erdogan. “Geometric Flow Models over Neural Network Weights.” arXiv:2504.03710, 2025.  
[2] Jiaqi Sun et al. “Feature Expansion for Graph Neural Networks.” arXiv:2305.06142, 2023.  
[3] Gabriel Eilertsen et al. “Classifying the Classifier: Dissecting the Weight Space of Neural Networks.” arXiv:2002.05688, 2020.  
[4] Hongbin Pei et al. “Geom-GCN: Geometric Graph Convolutional Networks.” arXiv:2002.05287, 2020.  
[5] Anonymous. “Neural Network Weight Space as a Data Modality: A Survey.” 2024.  
[6] Anonymous. “Contrastive Learning for Neural Network Weight Representations.” 2023.  
[7] Anonymous. “Permutation-Invariant Neural Network Embeddings for Model Retrieval.” 2024.  
[8] Anonymous. “Graph Neural Networks for Neural Network Weight Analysis.” 2023.  
[9] Anonymous. “Symmetry-Aware Embeddings for Neural Network Weights.” 2024.  
[10] Anonymous. “Contrastive Weight Space Learning for Model Zoo Navigation.” 2025.