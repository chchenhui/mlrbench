Title:
Permutation-Equivariant Graph Embeddings of Neural Weights for Model Retrieval and Synthesis

1. Introduction  
Background  
The explosive growth of publicly available pre-trained neural networks—now exceeding one million models on repositories like Hugging Face—has created both an opportunity and a challenge. On one hand, vast model zoos hold rich information about architectures, training dynamics, and domain-specific solutions. On the other hand, harnessing this resource requires new methods to compare, retrieve, and synthesize weights across different architectures and training runs. A central obstacle is the inherent weight-space symmetries: neuron permutations within layers and arbitrary layer-wise rescalings leave model function unchanged, yet defeat naive comparison or interpolation of raw weight tensors.

Research Objectives  
We propose to learn a symmetry-aware embedding of entire neural weights that is:  
• Invariant to neuron permutations and scaling, yet expressive enough to distinguish architecture, task specialization, and training quality.  
• Capable of supporting downstream tasks such as fast model retrieval, zero-shot performance prediction, and principled model merging without expensive fine-tuning.  
• Scalable to real-world model zoos containing thousands of diverse architectures and size regimes.

Specifically, our objectives are:  
1. To design a hierarchical graph neural network (GNN) that processes each layer’s weight matrix as a fully connected graph and assembles layer embeddings into a global model embedding.  
2. To train this architecture with a contrastive learning objective that enforces invariance under random permutations and rescalings of neurons, while preserving distinctions between different models.  
3. To validate the learned embedding on three downstream tasks: model retrieval, weight synthesis via embedding interpolation, and zero-shot accuracy prediction.  

Significance  
An effective embedding of neural weights will unlock weight-space as a first-class data modality. Researchers could search for relevant pre-trained models with simple nearest-neighbor lookups, dramatically reducing compute and carbon footprints. Weight interpolation in embedding space promises hybrid model initializations that combine strengths of multiple pre-trained networks. Zero-shot performance prediction will guide efficient architecture and hyperparameter search. Together, these advances will democratize large-scale model reuse and synthesis, accelerating progress across vision, language, scientific computing, and beyond.

2. Related Work  
Permutation-Equivariant Representations  
Graph Neural Networks for Learning Equivariant Representations (Kofinas et al., 2024) introduced computational-graph encodings of network parameters, enabling GNNs and transformers that respect permutation symmetry. SpeqNets (Morris et al., 2022) and Subgraph Permutation Equivariant Networks (Mitton & Murray-Smith, 2021) address sparsity and scalability in permutation-equivariant architectures. E(n)-Equivariant GNNs (Satorras et al., 2021) extend equivariance to geometric transformations, though our focus is on purely combinatorial symmetries in fully connected layers.

Contrastive Self-Supervised Learning  
Contrastive Language–Image Pre-training (CLIP, Radford et al., 2021) demonstrated the power of InfoNCE objectives for cross-modal embedding. Self-supervised learning surveys (2024) highlight the importance of careful positive/negative pair selection to shape embedding geometry. Our work adapts these principles to weight graphs, defining positives via randomized permutations and scalings.

Model Retrieval and Merging  
Recent model merging and “model soups” approaches perform heuristic weight averaging or selective merging but lack a principled embedding framework. Meta-learning networks and hyper-networks have been proposed for weight generation, but rarely enforce symmetry invariance explicitly. Our contribution lies in learning an embedding space tailored for retrieval and generative interpolation of weights.

3. Methodology  
3.1. Data Collection and Preprocessing  
• Model Zoo Construction: We will curate a dataset of $N\approx50\,000$ pre-trained networks from Hugging Face, covering classification (ResNets, VGG, EfficientNets), transformer encoders, and implicit neural representations (INRs such as NeRF decoders). We ensure variety in depth, width, and training datasets (CIFAR-10/100, ImageNet, COCO, ShapeNet).  
• Layer Graph Representation: Each layer $\ell$ with weight matrix $W^\ell\in\mathbb{R}^{n_{\ell+1}\times n_{\ell}}$ and bias vector $b^\ell\in\mathbb{R}^{n_{\ell+1}}$ is mapped to a fully connected directed graph $G^\ell=(V^\ell,E^\ell)$:  
  – Nodes $V^\ell=\{v_i\}_{i=1}^{n_{\ell+1}}$ correspond to output neurons.  
  – Node features $x_i^\ell = \big\{\|W^\ell_{i,:}\|_2,\; |b^\ell_i|,\; n_\ell,\; n_{\ell+1}\big\}$ capture incoming weight norm, bias magnitude, and layer dimensions.  
  – Edges $(v_j \to v_i)$ carry feature $e_{ji}^\ell = W^\ell_{i,j}$, the scalar weight.

3.2. Hierarchical GNN Architecture  
Our architecture has two stages: intra-layer GNNs produce layer embeddings, and an inter-layer transformer aggregates these into a global code.

3.2.1. Intra-Layer Message Passing  
For each layer $\ell$, we initialize node embeddings $h_i^{(0),\ell} = \mathrm{MLP}_\mathrm{init}(x_i^\ell)$. We perform $T$ message-passing steps:  
$$
m_i^{(t),\ell} \;=\; \sum_{j=1}^{n_\ell}\psi\big(h_i^{(t),\ell}, h_j^{(t),\ell}, e_{ji}^\ell\big),\quad
h_i^{(t+1),\ell} \;=\; \phi\big(h_i^{(t),\ell}, m_i^{(t),\ell}\big),
$$  
where $\psi$ and $\phi$ are feed-forward networks shared across layers. This update is permutation equivariant since permutations of indices permute both node features and messages consistently.

After $T$ steps, we pool node embeddings to produce layer code $z^\ell = \mathrm{POOL}\big(\{h_i^{(T),\ell}\}_{i=1}^{n_{\ell+1}}\big)$ using a symmetric function such as mean or sum.

3.2.2. Inter-Layer Aggregation  
We collect the sequence of layer embeddings $\{z^1,\dots,z^L\}$ for an $L$-layer network. A standard Transformer encoder processes this sequence, preserving layer order:  
$$
\hat Z = \mathrm{Transformer}\big([z^1,\dots,z^L]\big),\quad
z_\mathrm{global} = \mathrm{MeanPool}\big(\hat Z\big),
$$  
yielding a fixed‐dimensional global embedding $z_\mathrm{global}\in\mathbb{R}^d$. This embedding is invariant to intra-layer neuron permutations and scales, and captures inter-layer structure.

3.3. Contrastive Learning Objective  
To enforce invariance, we construct positive pairs $(z_i,z_i^+)$ by applying random layer-wise permutations $P^\ell$ and diagonal scalings $S^\ell$ to weights and biases:  
$$
W_i^{\ell,+} = P^\ell\,W_i^\ell\,S^\ell,\quad
b_i^{\ell,+} = s^\ell\odot b_i^\ell,
$$  
then re-embed to get $z_i^+$. Negatives $\{z_j^-\}_{j\neq i}$ come from other distinct models. We use the InfoNCE loss for a batch of size $B$:  
$$
\mathcal{L}_\mathrm{contrast} = -\frac{1}{B}\sum_{i=1}^B \log\frac{\exp\big(\mathrm{sim}(z_i, z_i^+)/\tau\big)}%
{\exp\big(\mathrm{sim}(z_i, z_i^+)/\tau\big) + \sum_{j=1, j\neq i}^B \exp\big(\mathrm{sim}(z_i, z_j^-)/\tau\big)},
$$  
where $\mathrm{sim}(u,v)=u^\top v/\|u\|\|v\|$ is cosine similarity and $\tau>0$ is a temperature hyperparameter.

3.4. Downstream Tasks and Experimental Design  
We evaluate the embedding on three tasks:

3.4.1. Model Retrieval  
• Task: Given a query model’s embedding $z_q$, retrieve the top-$k$ nearest neighbor embeddings in the model zoo.  
• Evaluation Metrics: Recall@k (percentage of retrieved models matching query architecture or task), Mean Reciprocal Rank (MRR), and retrieval latency.  
• Baselines: Raw weight vector Euclidean distance, PCA on flattened weights, and prior GNN approaches without equivariance.

3.4.2. Zero-Shot Performance Prediction  
• Task: Predict a model’s validation accuracy $a$ on its original task from its embedding $z$.  
• Method: Train a regression head $\hat a = \mathrm{MLP}_\mathrm{reg}(z)$ on a held-out set.  
• Metrics: $R^2$ coefficient of determination, mean squared error $\mathrm{MSE} = \frac1N\sum(a-\hat a)^2$, and Spearman rank correlation to assess monotonicity.

3.4.3. Embedding-Based Model Merging  
• Task: Given two models $A,B$ with embeddings $z_A,z_B$, generate a hybrid initialization for fine-tuning on a new task.  
• Method: Interpolate embeddings $z_\alpha = (1-\alpha)z_A + \alpha z_B$ for $\alpha\in[0,1]$, then decode to weights via a small hyper-network $\mathcal{H}$ trained to invert embeddings in a pre-training phase:  
$$
\tilde W^\ell(\alpha),\,\tilde b^\ell(\alpha) = \mathcal{H}^\ell\big(z_\alpha\big).
$$  
• Metrics: Final fine-tuned accuracy versus baselines (random init, linear weight interpolation), convergence speed (epochs to reach 90% accuracy), and compute cost.

3.5. Implementation Details  
• Hyperparameters: Node and edge MPNN hidden size 128, message-passing steps $T=3$, global embedding dimension $d=256$, temperature $\tau=0.1$, batch size $B=128$.  
• Training: Adam optimizer with learning rate $1\mathrm{e}{-4}$, cosine learning-rate decay over 200 epochs on 8 GPUs.  
• Scalability: We will compare full-graph MPNNs to subgraph-based (SPEN) and sparsity-aware (SpeqNet) variants, measuring GPU memory and throughput.

4. Expected Outcomes & Impact  
Expected Outcomes  
• Symmetry-Aware Embedding Model: A publicly released PyTorch implementation of the hierarchical permutation-equivariant GNN and transformer encoder, along with pre-trained weights for embedding generation.  
• Retrieval and Prediction Benchmarks: Detailed evaluation of model retrieval accuracy, zero-shot performance regression, and embedding-based model merging against strong baselines, demonstrating:  
  – Recall@10 > 90% for architecture identification across 20 classes.  
  – $R^2>0.85$ for zero-shot accuracy prediction on held-out models.  
  – 25% reduction in fine-tuning epochs for merged initializations compared to scratch.  
• Dataset and Leaderboard: A curated subset of $50\,000$ pre-trained models with metadata and embedding vectors, enabling future research.

Broader Impact  
• Democratizing Model Reuse: By indexing models via compact embeddings, practitioners can instantly locate suitable pre-trained networks for transfer learning, lowering compute and energy barriers.  
• Enabling Weight-Space Search: The embedding space can serve as a substrate for weight-space optimization, meta-learning, and neural architecture search, fostering novel algorithms that navigate the model zoo efficiently.  
• Foundations for Weight Synthesis: Decoding embeddings into weights via hyper-networks opens new directions for generative model synthesis and continual learning without direct exposure to raw weight tensors.  
• Cross-Domain Applications: While our primary focus is vision models and INRs, the methodology readily extends to language models, graph networks, and dynamical systems, advancing disciplines from NLP to computational physics.  

In summary, our proposal establishes a principled and scalable framework for treating neural network weights as a new data modality. By enforcing permutation and scaling invariance through graph-based encoders and contrastive learning, we pave the way for efficient retrieval, analysis, and synthesis of models in large-scale zoos. This research will catalyze interdisciplinary collaboration and accelerate progress in both theoretical and applied machine learning.