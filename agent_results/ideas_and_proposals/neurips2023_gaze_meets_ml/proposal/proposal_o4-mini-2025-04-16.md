1. Title  
Gaze-Guided Self-Supervised Feature Prioritization for Unsupervised Medical Image Analysis

2. Introduction  
Background  
Medical imaging (e.g., chest X-rays, CT scans, MRIs) is a cornerstone of modern diagnostics, yet accurate interpretation often depends on expensive, expert-annotated data. Self-supervised learning (SSL) methods such as SimCLR and MoCo have achieved remarkable results in computer vision by exploiting large volumes of unlabeled images. However, they lack a built-in mechanism to prioritize clinically relevant regions, leading to features that may not align with the diagnostic reasoning of radiologists. Eye-tracking studies reveal that expert clinicians naturally fixate on diagnostically critical areas (e.g., lung lesions, nodules), producing spatially localized “gaze heatmaps” that implicitly encode diagnostic attention. Leveraging this signal as a form of free, weak supervision promises to bridge the gap between generic visual representations and medically meaningful features.

Recent work—Medical contrastive Gaze Image Pre-training (McGIP) and FocusContrast—shows the potential of using gaze data to improve contrastive pre‐training; GazeGNN further integrates raw eye‐tracking into graph representations. Yet challenges remain: (1) available gaze datasets are limited in size and modality; (2) inter‐reader variability complicates modeling; (3) existing frameworks often treat gaze only as a pair-selection signal, not as a continuous attention prior that can dynamically steer feature learning.  

Research Objectives  
We propose a Gaze-Guided Self-Supervised Feature Prioritization framework (“GazePrior”) that:  
  a. Integrates radiologist gaze as a continuous attention prior within convolutional or transformer backbones, dynamically weighting patch embeddings.  
  b. Employs a gaze-weighted contrastive loss to align embeddings of gaze-attended regions across augmentations.  
  c. Demonstrates improved anomaly detection and disease classification without manual labels, matching or exceeding state-of-the-art SSL and semi-supervised baselines.  
  d. Delivers interpretable attention maps that closely correlate with expert gaze, fostering trust and transparency in clinical AI systems.  

Significance  
By harnessing gaze as a free, annotation-light supervisory signal, GazePrior will:  
  • Reduce the reliance on costly pixel-level or bounding-box annotations.  
  • Improve model generalization in low-data regimes and across modalities.  
  • Enhance interpretability by aligning model attention with expert visual behavior.  
  • Establish a new paradigm for embedding human cognitive signals into SSL, with applications in radiology, pathology, and beyond.

3. Methodology  
3.1 Overview  
GazePrior consists of two core modules: (1) a Gaze-Guided Attention Module (GGAM) that injects continuous gaze priors into feature extraction, and (2) a Gaze-Weighted Contrastive Loss (GWCL) that enforces embedding similarity for gaze-attended regions across stochastic augmentations. Figure 1 sketches the training pipeline.  

3.2 Data Collection and Preprocessing  
Dataset  
  • Imaging: We will use the MIMIC-CXR dataset (Chest X-rays, n≈370 k) for unlabeled pre-training and CheXpert (n≈224 k) for downstream evaluation.  
  • Gaze: Leverage existing eye-tracking corpora (e.g., McGIP’s radiologist gaze on 50 k X-rays) and collect an additional set of 5 board-certified radiologists reading 5 k images each using wearable eye-trackers (Tobii Pro Glasses 3), yielding ∼25 k gaze sessions.  

Preprocessing  
  1. Resize each image to 224×224 pixels; normalize intensities per Imagenet mean/std.  
  2. Generate per-image gaze heatmaps $H(x,y)$ by placing a Gaussian kernel (σ=5px) at each fixation coordinate and normalizing so that $\sum_{x,y}H(x,y)=1$.  
  3. Partition each image into $P=M×M$ non-overlapping patches (e.g., $M=14$ ⇒ 14×14 patches of size 16×16). Compute patch-wise gaze weight  
     $$g_i = \sum_{(x,y)\in\text{patch}_i} H(x,y)\,,\quad i=1,\dots,P$$  
     and normalize $\tilde g_i = \frac{g_i}{\max_j g_j}\in[0,1]$.  

3.3 Gaze-Guided Attention Module (GGAM)  
We adapt a convolutional backbone (e.g., ResNet-50) or Vision Transformer (ViT-Base) by injecting a gating mechanism over patch embeddings. Let $z_i\in\mathbb R^d$ be the feature vector for patch $i$ at a chosen intermediate layer. We compute a gaze gating scalar:  
  $$\alpha_i = \sigma\bigl(w_g\,\tilde g_i + b_g\bigr)\in(0,1)$$  
where $(w_g,b_g)$ are learnable parameters and $\sigma$ is the sigmoid function. The gated embedding is  
  $$z_i' = \alpha_i\,z_i\,. $$  
For CNNs, $z_i$ is obtained by spatially pooling the feature map over the patch region; for ViTs, $z_i$ is the patch token embedding. The gated embeddings $\{z_i'\}$ are concatenated and passed through the remaining network layers.

3.4 Gaze-Weighted Contrastive Loss (GWCL)  
We generate two stochastic augmentations $(t,t')$ of an input image $X$ (e.g., random crop, flip, Gaussian blur) and compute gated embeddings $\{z_i'\}$ and $\{z_{i'}'\}$ for each view. We treat patch embeddings from the same spatial index $i$ as positive pairs weighted by gaze similarity, and all other patch pairs as negatives. Define the per-pair similarity  
  $$\mathrm{sim}(z_i',z_{j'}') = \frac{z_i'^\top z_{j'}'}{\|z_i'\|\|z_{j'}'\|}\,. $$  
Let $w_{ij'} = \tilde g_i \cdot \tilde g_{j'}$ be the product of normalized gaze weights. The GWCL for a single anchor patch $i$ is  
  $$
  \mathcal{L}_i = - \sum_{j'} w_{ij'}\;\log 
     \frac{\exp\bigl(\mathrm{sim}(z_i',z_{j'}')/\tau\bigr)}
          {\sum_{k'} \exp\bigl(\mathrm{sim}(z_i',z_{k'}')/\tau\bigr)}\,,
  $$
and the total loss over all patches and batch size $N$ is  
  $$
  \mathcal{L}_{\mathrm{GWCL}} 
   = \frac{1}{N\,P}\sum_{n=1}^N\sum_{i=1}^P \mathcal{L}_i^{(n)}\,.
  $$  
Here $\tau$ is a temperature hyperparameter. This encourages embeddings of gaze-attended regions to align across views, with stronger gaze regions contributing more heavily.

Overall objective  
  $$\mathcal{L} = \mathcal{L}_{\mathrm{GWCL}} + \lambda\,\mathcal{L}_{\mathrm{proj}}\,, $$  
where $\mathcal{L}_{\mathrm{proj}}$ is an $L_2$ regularizer on projection‐head weights and $\lambda$ balances regularization.

3.5 Training Protocol  
  • Optimizer: AdamW with initial learning rate 1e-4, weight decay 1e-5.  
  • Batch size: 256 images (GPU memory permitting); accumulate gradients if needed.  
  • Epochs: 200 for pre-training.  
  • Learning rate schedule: cosine decay with 10-epoch warm-up.  
  • Hyperparameters: $\tau=0.1$, $\lambda=1e^{-4}$, $w_g$ initialized to 1.  

3.6 Downstream Evaluation  
After self-supervised pre-training, we freeze the backbone and train a linear classifier on CheXpert labels (14 disease classes). Metrics:  
  • Area under the ROC curve (AUC) for each disease.  
  • Sensitivity and specificity at clinically relevant thresholds.  
  • Macro-average F1 score.  

Unsupervised anomaly detection  
We fit a one-class SVM or Gaussian Mixture Model on the frozen embeddings of normal images; test on images with abnormalities. Evaluate using AUC‐ROC for anomaly detection.

Interpretability metrics  
  • Rank correlation (Spearman’s ρ) between $\tilde g_i$ and the model’s learned attention map per test image.  
  • Earth Mover’s Distance (EMD) between gaze heatmaps and model saliency maps (Grad-CAM or attention rollout).

Ablation studies  
  1. Remove GGAM (i.e., $\alpha_i=1$ ∀i).  
  2. Set $w_{ij'}=1/P$ (i.e., ignore gaze weights in GWCL).  
  3. Vary $\lambda\in\{0,10^{-5},10^{-3}\}$.  
  4. Compare CNN backbone vs. ViT backbone.  
  5. Compare with baselines: SimCLR, MoCo, McGIP, FocusContrast, GazeGNN.

4. Expected Outcomes & Impact  
4.1 Improved Unsupervised Feature Learning  
We anticipate GazePrior will yield higher downstream classification AUCs (≥2–3% gain) over SimCLR/MoCo and outperform existing gaze-based SSL methods by better leveraging continuous gaze priors. Anomaly detection AUC is expected to exceed 0.90 on chest X-ray datasets, demonstrating robust unsupervised identification of pathologies.

4.2 Enhanced Interpretability and Trust  
By generating attention maps that closely mirror radiologist gaze (Spearman’s ρ≥0.7, EMD↓20%), GazePrior will provide clinicians with intuitive visual explanations, fostering trust in AI-driven diagnostics.

4.3 Reduced Annotation Burden  
In low-label regimes (using only 1% labeled data for fine-tuning), we expect GazePrior to match or exceed the performance of fully supervised models trained on 10× more labels, highlighting the practical value of gaze as a free supervisory signal.

4.4 Broader Impacts  
  • Methodology generalizes to other modalities (CT, MRI) and specialties (pathology slides).  
  • Framework paves the way for integrating other physiological signals (EEG, heart rate) into SSL.  
  • Promotes ethical AI by aligning model focus with expert intent and clarifying decision processes.

5. Timeline & Milestones  
Months 1–3  
  • Collect additional eye-tracking data; preprocess images and gaze heatmaps; implement GGAM.  
Months 4–6  
  • Develop GWCL; integrate into training pipeline; run initial pre-training.  
Months 7–9  
  • Conduct downstream evaluations on CheXpert; perform anomaly detection experiments.  
Months 10–12  
  • Complete ablation studies; refine hyperparameters; submit results for publication; prepare open-source code release.

By systematically embedding radiologist gaze into self-supervised feature learning, this project will advance both medical AI performance and interpretability, offering a cost-efficient path toward clinically aligned, trustworthy diagnostic systems.