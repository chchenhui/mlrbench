1. Title  
Contrastive Multi-Modal Alignment for Unified Material Representations  

2. Introduction  

Background  
Advances in artificial intelligence have begun to revolutionize materials discovery by enabling high-throughput screening, inverse design, and automated characterization. Nevertheless, most AI-driven approaches treat each data modality—atomic structure, synthesis protocol text, and characterization images—in isolation. Structural GNNs excel at predicting properties from graphs but ignore synthesis history; NLP models capture synthesis recipes but cannot directly infer structural or property relationships; computer-vision models interpret micrographs without contextualizing atomic arrangements or protocols. This siloed approach overlooks the rich, cross-modal correlations that govern material performance in real‐world applications.  

The recent success of foundation models (e.g., CLIP for vision‐language, large language models for text) underscores the power of contrastive multi-modal pre-training to learn unified embeddings that transfer efficiently to downstream tasks. In materials science, an analogous foundation model would unify heterogeneous data—graphs, text, images—into a shared latent space, yielding representations that capture the interplay between structure, synthesis, and function.  

Research Objectives  
• Develop a contrastive learning framework that aligns graph-, text-, and image-based encodings of the same material in a shared latent space.  
• Demonstrate that the unified embeddings improve performance on downstream tasks—property prediction, synthesis recommendation, and defect identification—relative to single-modality baselines.  
• Evaluate generalization across material classes (crystalline, amorphous, nanomaterials) and assess the interpretability of learned embeddings.  

Significance  
By unifying multi-modal data, this research:  
• Lays the groundwork for a materials “foundation model,” addressing the AI4Mat-ICLR-2025 theme of building foundational models for materials science.  
• Advances next-generation representations by incorporating structural, procedural, and visual information, thus tackling the challenge of multi-modal integration.  
• Enables holistic material understanding, accelerating discovery pipelines and guiding experimental planning.  

3. Methodology  

3.1 Data Collection and Preprocessing  
We will assemble a large, diverse dataset comprising:  
• Atomic structures from the Materials Project and OQMD (∼1M crystal and molecular graphs).  
• Synthesis protocols scraped from published experimental papers (∼500K text descriptions).  
• Characterization images (X-ray diffraction (XRD), scanning electron microscopy (SEM)) linked to known structures (∼200K images).  

Preprocessing steps:  
1. Structure → Graph: Represent each material as a graph $G=(V,E)$ where nodes $v\in V$ carry atomic features (atomic number, electronegativity, coordination number) and edges $e\in E$ encode bonding or near-neighbor distances. We normalize interatomic distances and encode them via radial basis functions.  
2. Text → Tokens: Tokenize synthesis protocols using a subword tokenizer (e.g., Byte-Pair Encoding) and construct input sequences of maximum length 512, embedding tokens with learned vectors.  
3. Images → Arrays: Standardize SEM/XRD image sizes (e.g., $224\times224$) and normalize pixel intensities.  

3.2 Multi-Modal Encoders  
We define three modality-specific encoders:  

• Structural Encoder ($f_g$): A message-passing Graph Neural Network (MP-GNN) with $L$ layers. At layer $\ell$, node features $h_i^{(\ell)}$ update as  
$$
h_i^{(\ell+1)} = \mathrm{ReLU}\Big(W_1^{(\ell)} h_i^{(\ell)} + \sum_{j\in\mathcal{N}(i)} W_2^{(\ell)} h_j^{(\ell)} + b^{(\ell)}\Big).
$$  
After $L$ layers, we apply a readout (mean pooling) to obtain a structural embedding $h^g\in\mathbb{R}^d$.  

• Text Encoder ($f_t$): A Transformer encoder with $M$ self-attention layers. Given token embeddings $x_1,\dots,x_T$, at layer $m$:  
$$
\begin{aligned}
\tilde{x}_i &= x_i + \sum_{j=1}^T \alpha_{ij}\,x_j,\quad \alpha_{ij} = \frac{\exp\big((x_iW_Q)(x_jW_K)^\top/\sqrt{d_k}\big)}{\sum_{k=1}^T\exp\big((x_iW_Q)(x_kW_K)^\top/\sqrt{d_k}\big)},\\
x_i' &= \mathrm{LayerNorm}\big(\tilde{x}_i W_V\big),\quad x_i^{(m+1)} = \mathrm{FFN}(x_i').
\end{aligned}
$$  
We extract the [CLS] token representation $h^t\in\mathbb{R}^d$.  

• Image Encoder ($f_v$): A convolutional neural network (e.g., ResNet‐50) truncated before the classification head. Given image $I$, the final feature map is average-pooled to yield $h^v\in\mathbb{R}^d$.  

All encoders project to the same embedding dimension $d$ and are followed by $\ell_2$ normalization.  

3.3 Contrastive Alignment Framework  
We align paired embeddings of the same material across modalities using a pairwise InfoNCE loss. For a training batch of $N$ materials, let $\{(h_i^g,h_i^t,h_i^v)\}_{i=1}^N$ be the normalized embeddings. Define the cosine similarity $\mathrm{sim}(u,v)=u^\top v$. The pairwise losses are:  
$$
\mathcal{L}_{g,t}=-\frac{1}{N}\sum_{i=1}^N\log\frac{\exp\big(\mathrm{sim}(h_i^g,h_i^t)/\tau\big)}{\sum_{j=1}^N\exp\big(\mathrm{sim}(h_i^g,h_j^t)/\tau\big)},
$$  
and analogously $\mathcal{L}_{g,v}$, $\mathcal{L}_{t,v}$. The total loss is  
$$
\mathcal{L}_{\mathrm{contrast}} = \lambda_{gt}\,\mathcal{L}_{g,t} + \lambda_{gv}\,\mathcal{L}_{g,v} + \lambda_{tv}\,\mathcal{L}_{t,v},
$$  
where $\lambda_{*}$ weight each pair. We set $\lambda_{gt}=\lambda_{gv}=\lambda_{tv}=1$ initially and tune via validation. Temperature $\tau$ is a learnable scalar.  

3.4 Algorithmic Steps  
1. Initialize encoder parameters $\theta_g,\theta_t,\theta_v$.  
2. For each epoch:  
   a. Sample a minibatch of $N$ materials with complete triplets.  
   b. Compute embeddings $h_i^g=f_g(G_i;\theta_g)$, $h_i^t=f_t(T_i;\theta_t)$, $h_i^v=f_v(V_i;\theta_v)$.  
   c. Normalize embeddings: $\bar h = h / \|h\|_2$.  
   d. Compute pairwise contrastive losses $\mathcal{L}_{g,t},\mathcal{L}_{g,v},\mathcal{L}_{t,v}$.  
   e. Backpropagate $\nabla_{\theta}\mathcal{L}_{\mathrm{contrast}}$ and update parameters using AdamW.  
3. Optionally incorporate curriculum learning: begin with two-modal alignment (graph–text), then add image alignment in later epochs.  

3.5 Downstream Fine-Tuning and Evaluation  
After pre-training, we fine-tune the unified encoder for three tasks:  

• Property Prediction: Attach a 2-layer MLP to $h^g$ (or to the fused embedding) to predict continuous properties (e.g., band gap). Loss: mean squared error (MSE).  

• Synthesis Recommendation: Given a target property vector $p^*$, retrieve top-$k$ synthesis texts by nearest-neighbor search in the joint space. Evaluate Recall@k and nDCG.  

• Defect Identification: Classify SEM images of defects using a classification head on $h^v$. Loss: cross-entropy; evaluate accuracy and F1 score.  

Ablation studies will compare:  
– Single-modality baselines (GNN only, Transformer only, CNN only).  
– Dual-modality contrastive models (graph–text, graph–image, text–image).  
– Full tri-modal model.  

3.6 Experimental Design and Evaluation Metrics  
We will split data into train/val/test (80/10/10) ensuring no material overlaps across splits. Key metrics:  
• Regression: MAE, RMSE, Pearson and Spearman correlation.  
• Classification: accuracy, precision, recall, F1, ROC-AUC.  
• Retrieval: Recall@k (k=1,5,10), nDCG.  
• Embedding quality: Silhouette score and Davies–Bouldin index on clustering of material classes.  
• Computational cost: GPU hours, peak memory usage.  

Hyperparameters (to be tuned via grid search) include embedding size $d\in\{128,256,512\}$, batch size $N\in\{64,128,256\}$, learning rate $\{10^{-4},5\times10^{-4},10^{-5}\}$, and number of GNN/Transformer layers.  

4. Expected Outcomes & Impact  

Expected Outcomes  
• A pre-trained multi-modal encoder that produces unified material embeddings capturing correlations across structure, synthesis, and characterization.  
• Empirical evidence that the tri-modal model outperforms single- and dual-modality baselines on property prediction (e.g., 10–20% reduction in MAE), synthesis recommendation (e.g., +15% Recall@5), and defect classification (e.g., +5% F1).  
• Demonstrated generalization to unseen material classes, showcasing the model’s ability to transfer knowledge across domains (crystalline → amorphous, nano → bulk).  
• Qualitative interpretability analyses: visualization (t-SNE/UMAP) of embeddings revealing clustering by property or synthesis route; attention weights in the text encoder highlighting critical synthesis steps.  

Scientific and Societal Impact  
• Foundation Model Catalyst: This work establishes a proof-of-concept for a materials science foundation model, aligning with the workshop’s first theme. It offers a blueprint for scaling to billions of examples, paving the way for truly general-purpose materials embeddings.  
• Next-Gen Representations: By integrating graph, text, and image modalities, we address the workshop’s second theme of novel representations. The unified embeddings can serve as plug-and-play features for diverse downstream tasks beyond those studied here (e.g., phase prediction, catalysis screening).  
• Accelerated Discovery: Improved property predictions and synthesis recommendations will shorten the experimental design loop, reducing cost and environmental impact by minimizing trial-and-error iterations.  
• Collaborative Platform: The release of the pre-trained model and code fosters an inclusive community resource, enabling both AI researchers and material scientists to build upon our work.  

Timeline and Milestones  
• Months 1–3: Data curation and preprocessing pipeline; baseline single-modality models.  
• Months 4–8: Implementation of multi-modal encoders and contrastive training; initial validation.  
• Months 9–12: Ablation studies; fine-tuning on downstream tasks; hyperparameter optimization.  
• Months 13–15: Interpretability analyses; robustness evaluations on out-of-distribution materials.  
• Months 16–18: Manuscript preparation; open-source release of code, pre-trained weights, and datasets.  

In sum, this proposal outlines a comprehensive plan to develop and validate a contrastive multi-modal alignment framework for materials science. By leveraging structural, textual, and visual data, we aim to create a unified representation that advances both foundational AI research and practical materials discovery.