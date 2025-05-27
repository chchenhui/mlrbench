1. Title  
Clin-ACT: Clinician-in-the-Loop Active Contrastive Learning for Robust and Interpretable Pediatric ICU Time Series Representation  

2. Introduction  
Background  
Pediatric intensive care units (PICUs) continuously record multivariate vital signs (heart rate, blood pressure, oxygen saturation), laboratory results (blood gases, electrolyte panels) and therapies (medications, ventilator settings). These time series are high-dimensional, irregularly sampled and frequently contain missing values or outliers due to varying monitoring schedules or sensor artifacts. Moreover, clinically meaningful labels (e.g., sepsis onset) are scarce because clinician annotation is time-consuming and expertise-limited. Conventional supervised models require large labeled cohorts and imputation heuristics, both of which can degrade performance and clinician trust.  

Self-supervised and contrastive methods have recently shown promise in learning robust representations without heavy reliance on labels. Yet most ignore missingness patterns, irregular sampling, and fail to provide interpretable insights. Active learning can reduce labeling burden, but classical criteria (uncertainty alone) may oversample redundant examples in highly correlated ICU signals. Prototype-based explanation modules can offer interpretability, yet have not been integrated with contrastive representation learning in critical care settings.  

Research Objectives  
This proposal introduces Clin-ACT, a unified framework that:  
• Learns robust embeddings of pediatric ICU time series by combining imputation-aware contrastive self-supervision with data augmentations tailored to irregular sampling and outliers.  
• Employs a hybrid uncertainty-diversity active learning criterion to select the most informative windows for clinician annotation, aiming to reduce labeling effort by at least 60%.  
• Integrates a lightweight prototype module that maps learned embeddings to clinical archetypes and generates feature saliency maps, thereby enhancing interpretability and clinician trust.  

Significance  
Clin-ACT will bridge the gap between representation learning research and clinical practice by delivering label-efficient, robust, and interpretable embeddings for downstream tasks such as early sepsis detection. By focusing on pediatric critical care—a minority, high-risk population often under-represented in large datasets—this work addresses a pressing need for trustworthy ML tools. Successful validation of Clin-ACT could lead to more rapid and reliable clinical decision support systems, improved patient outcomes, and higher adoption rates among healthcare professionals.  

3. Methodology  
Overview  
Clin-ACT comprises three core modules: (A) Imputation-Aware Contrastive Encoder, (B) Active Learning Query Strategy, and (C) Prototype-Based Interpretability Layer. Figure 1 (conceptual) shows the data flow.  

A. Imputation-Aware Contrastive Encoder  
Data Preprocessing  
• Windowing: Each patient’s multivariate time series $X \in \mathbb{R}^{T\times d}$ (where $T$ is total time points and $d$ the number of channels) is split into overlapping windows of length $w$ with stride $s$.  
• Missingness Mask: For each window, construct a binary mask $M\in\{0,1\}^{w\times d}$ indicating observed entries.  

Augmentation Strategies  
1. Temporal Cropping and Time-Warping: Randomly crop a subwindow of length $\alpha w$ ($\alpha\in[0.7,1.0]$) and rescale time.  
2. Masked Channel Dropout: Randomly zero out $p\%$ of channels per window, simulating sensor dropout.  
3. Noise Injection & Outlier Simulation: Add Gaussian noise $\mathcal{N}(0,\sigma^2)$ to observed values; randomly amplify a small subset (1–2%) by factor $\beta>1$ to mimic outliers.  
4. Mask Perturbation: Randomly flip 5–10% of mask bits to simulate false missingness or imputation errors.  

Encoder Architecture  
We adopt a Transformer encoder with continuous-value embedding (inspired by STraTS). For each window, the input is the triplet sequence $\{(t_i, x_i, m_i)\}_{i=1}^n$, where $t_i$ is timestamp, $x_i\in\mathbb{R}^d$ is observation, and $m_i\in\{0,1\}^d$ the mask. Each triplet is linearly mapped to a joint embedding:  
$$z_i = W_t\,\phi(t_i) + W_x(x_i\odot m_i) + W_m m_i + b$$  
where $\phi(t_i)$ applies Fourier features, and $W_t,W_x,W_m,b$ are learnable. The Transformer stack yields a fixed-length representation $h\in\mathbb{R}^k$ via mean pooling over tokens.  

Contrastive Objective  
We generate two augmentations $(\tilde X_i,\tilde M_i)$ and $(\tilde X_j,\tilde M_j)$ from the same original window to produce embeddings $(h_i,h_j)$. We minimize the InfoNCE loss:  
$$
\mathcal{L}_{\text{contrast}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(h_i,h_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{I}_{[k\neq i]}\exp(\text{sim}(h_i,h_k)/\tau)},
$$  
where $\text{sim}(u,v)=u^\top v/\|u\|\|v\|$ and $\tau$ a temperature hyperparameter.  

Regularization  
To respect missingness patterns, add a consistency loss between original mask $M$ and reconstructed mask prediction $\hat M=\sigma(f_{\text{mask}}(h))$:  
$$
\mathcal{L}_{\text{mask}} = \|M - \hat M\|_F^2.
$$  

B. Active Learning Query Strategy  
After initial contrastive pretraining on unlabeled windows, we select a small batch for labeling under budget constraint $B$. For each unlabeled embedding $h_i$ we compute:  
1. Uncertainty Score $u_i = 1 - \max_{c} p(y=c\mid h_i)$ from a lightweight classifier $f_{\text{uncert}}$ trained on labelled pool.  
2. Diversity Score $d_i = \min_{j\in S} (1 - \text{sim}(h_i,h_j))$, where $S$ is the current labelled set.  

We define a combined acquisition function:  
$$
a_i = \lambda\,\frac{u_i - \min(u)}{\max(u)-\min(u)} + (1-\lambda)\,\frac{d_i - \min(d)}{\max(d)-\min(d)},
$$  
and select top-$B$ windows by $a_i$. Here $\lambda$ balances uncertainty vs. diversity. We hypothesize that $\lambda=0.6$ will yield maximal label efficiency based on pilot tuning.  

Procedure:  
Algorithm: Clin-ACT Active Loop  
1. Pretrain encoder with $\mathcal{L}_{\text{contrast}} + \gamma \mathcal{L}_{\text{mask}}$ on unlabeled data.  
2. Initialize labelled set $S$ with $k_0$ randomly chosen windows.  
3. Repeat until annotation budget exhausted:  
   a. Train a downstream classifier $f_{\text{uncert}}$ on $S$.  
   b. Compute $u_i,d_i,a_i$ for all unlabeled $i$.  
   c. Query top-$B$ windows, obtain clinician labels, update $S$.  

C. Prototype-Based Interpretability Layer  
We learn $P$ prototypes $\{c_p\}_{p=1}^P$ in the embedding space by minimizing:  
$$
\mathcal{L}_{\text{proto}} = \frac{1}{|S|}\sum_{i\in S}\min_{p}\|h_i - c_p\|_2^2 + \eta \sum_{p\neq q}\exp(-\|c_p - c_q\|_2^2),
$$  
encouraging coverage and separation. Each prototype is associated with a clinical archetype (e.g., “Stable”, “Hypotensive”, “Septic”).  

To explain individual predictions, we compute feature saliency via gradient-based method:  
$$
\text{Saliency}(x_t) = \left|\frac{\partial \text{sim}(h,c_{p^*})}{\partial x_t}\right|,
$$  
where $c_{p^*}$ is the nearest prototype to $h$. We visualize saliency across time and channels to indicate which measurements drove assignment to a clinical archetype.  

D. Experimental Design  
Datasets:  
• Primary: Internal PICU dataset covering 3,200 admissions, 24 channels, up to 10 days per admission.  
• External Validation: Publicly available pediatric sepsis dataset (e.g., 2023 PhysioNet challenge).  

Downstream Tasks:  
1. Sepsis onset detection (binary classification)  
2. Mortality prediction (binary)  

Baselines:  
• Supervised Transformer with zero self-supervision + mean imputation  
• SLAC-Time (self-supervised clustering)  
• MM-NCL (multi-modal contrastive learning)  
• STraTS (self-supervised transformer)  

Evaluation Metrics:  
• AUROC, AUPRC for classification tasks  
• Label Efficiency: performance vs. number of labeled windows  
• Annotation Reduction: percent labels saved relative to random sampling  
• Interpretability:  
   – Faithfulness: drop-in-performance when top-k salient features removed  
   – Sparsity: average number of non-zero saliency entries per explanation  
• Clinician Satisfaction Survey: Likert scale on trust and usefulness (n=10 pediatric intensivists)  

Implementation Details:  
• Pretraining: 200 epochs, batch size 512 windows, learning rate 1e-4  
• Active loop: budget $B=100$ windows per iteration, total budget 1,000 windows (≈60% label reduction vs. full)  
• Prototypes: $P=10$, prototype learning interleaved every 10 active iterations  
• Hardware: NVIDIA A100 GPUs, PyTorch framework  

4. Expected Outcomes & Impact  
Expected Outcomes  
1. Representation Quality: Clin-ACT embeddings will achieve at least +12% relative improvement in AUROC for pediatric sepsis detection compared to the best baseline, under equal label budgets.  
2. Label Efficiency: Active selection with the proposed uncertainty-diversity criterion will reduce clinician labeling effort by at least 60% to reach target performance.  
3. Interpretability: Prototype explanations and saliency maps will yield high faithfulness (>0.8 drop-in-accuracy when salient features masked) and clinician satisfaction scores averaging ≥4/5 in trust and usefulness.  
4. Robustness to Missingness: By design, Clin-ACT will demonstrate stable performance (±2% AUROC variance) under simulated increased missingness (up to 30% additional random drops).  

Impact  
Clin-ACT stands to deliver actionable, trustworthy time series representations for pediatric critical care, facilitating earlier and more accurate detection of sepsis and other adverse events. With substantially reduced labeling requirements, the framework is scalable to other minority clinical cohorts (e.g., neonatal ICU, rare diseases). The prototype layer offers transparent, case-specific explanations that can be directly reviewed by clinicians, fostering adoption in real-world settings. Ultimately, this work can accelerate the integration of advanced ML into pediatric healthcare, improving patient outcomes while respecting clinician time constraints and fostering trust in AI-driven decision support.