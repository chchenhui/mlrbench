Title  
Learning Invariant Feature Spaces for Cross-Domain Representational Alignment in Biological and Artificial Intelligences  

1. Introduction  
Background  
Intelligent systems—whether biological (e.g., human or non-human primate cortex) or artificial (e.g., deep neural networks)—encode high-dimensional representations of sensory inputs, language, and other modalities. Comparing these representations can reveal shared computational strategies, guide the design of more interpretable and brain-compatible AI, and shed light on the principles governing learning and perception across species and architectures. Yet existing alignment metrics (e.g., RSA, CKA, Procrustes) often fail to generalize across modalities (e.g., fMRI vs. activations), scales (voxel responses vs. hundreds-of-thousand-dimensional embeddings), and structures (temporal sequences vs. static feature maps).  

Research Objectives  
This proposal aims to develop a unified framework that learns an invariant feature space in which representations from disparate domains become directly comparable. Specifically, we will:  
• Design a dual-encoder architecture (one encoder per domain) with joint adversarial and contrastive objectives to project domain-specific representations into a shared latent space.  
• Introduce domain-agnostic alignment metrics computed in the shared space, capturing functional equivalence rather than modality artifacts.  
• Validate the framework on multiple cross-domain pairs (primate visual cortex vs. vision models; human fMRI during reading vs. large language models).  
• Quantify how alignment scores predict behavioral congruence (e.g., task performance, error patterns) and explore interventions (e.g., co-training with neural data) to systematically steer alignment.  

Significance  
By producing a robust, scalable, and domain-agnostic alignment metric, this project will:  
– Advance fundamental understanding of shared computation in biological and artificial systems.  
– Provide tools for neuroscientists to test computational theories against deep model activations.  
– Offer engineers principled methods to incorporate neural constraints into AI training, potentially improving generalization and interpretability.  

2. Related Work  
Domain-Adaptation and Contrastive Learning  
• Yadav et al. (2023, CDA) introduced a two-stage adversarial + contrastive domain-adaptation method that aligns feature distributions across labeled source and unlabeled target domains, achieving state-of-the-art in vision tasks.  
• Thota & Leontidis (2021) extended contrastive learning to unsupervised domain adaptation by mining positives/negatives without target labels, reducing false negatives.  
• Wang et al. (2021, CDCL) employed pseudo-label clustering plus contrastive objectives to adapt across visual domains.  
• Liu et al. (2021) applied patch-wise contrastive losses to semantic segmentation, demonstrating the benefit of local feature alignment.  

Challenges in Cross-Domain Representational Alignment  
1. Data Modality Differences: Neuroimaging (fMRI, EEG, multi-unit recordings) vs. network activations have different statistics, dimensionalities, and noise properties.  
2. Class-Conditional Shift: Feature distributions shift non-uniformly across classes or stimuli, leading to ambiguous alignment near decision boundaries.  
3. Unlabeled Target Data: Neural recordings under naturalistic stimuli often lack explicit labels or are sparsely annotated.  
4. False Negatives in Contrastive Training: Random sampling may treat semantically similar points as negatives, hindering learning.  
5. Scalability and Generalization: Methods must scale to millions of model activations and tens of thousands of neural measurements while generalizing across tasks and modalities.  

Our approach builds on these insights by combining adversarial domain confusion with a supervised or self-supervised contrastive loss, augmented by careful sampling and pseudo-labeling strategies to mitigate false negatives.  

3. Methodology  

3.1 Overview  
We propose a Dual-Encoder Invariant Feature Space (DEIFS) that consists of:  
• Domain Encoders E₁, E₂ parameterized by θ₁, θ₂ (e.g., two separate Transformer- or CNN-based encoders).  
• A shared latent space ℝ^d.  
• A domain discriminator Dϕ mapping z∈ℝ^d to a scalar, trained adversarially.  
• A contrastive module to enforce that semantically equivalent samples across domains become neighbors in latent space.  

Given datasets {x₁^i} from Domain 1 (e.g., fMRI responses to images or read text) and {x₂^j} from Domain 2 (e.g., CNN activations or LLM hidden states), we learn θ₁, θ₂, ϕ by optimizing  

L_total = L_contrast + λ_adv L_adv + λ_task L_task.  

3.2 Model Architectures  
Encoders  
• E₁: For neuroimaging data, a lightweight MLP or Transformer that maps preprocessed responses x₁∈ℝ^{n₁} to z₁∈ℝ^d.  
• E₂: For model activations, a projection head (e.g., two-layer MLP) mapping x₂∈ℝ^{n₂} to z₂∈ℝ^d.  

Domain Discriminator  
• D : ℝ^d→[0,1], a two-layer MLP trained to distinguish z’s from Domain 1 vs. Domain 2.  

3.3 Loss Functions  

3.3.1 Contrastive Loss  
We adopt a supervised (if labels available) or self-supervised contrastive loss. Let P be a set of matched pairs (i,j) where x₁^i and x₂^j are known to be semantically equivalent (e.g., same stimulus). Let N(i) be a set of negatives sampled from both domains, avoiding false negatives by excluding high-similarity candidates via a nearest-neighbor filter. For each positive pair (i,j):  

$$
L_{con}^{(i,j)} = -\log \frac{\exp\big(\mathrm{sim}(z₁^i,z₂^j)/τ\big)}
{\sum_{(i',j')\in P} \exp\big(\mathrm{sim}(z₁^i,z₂^{j'})/τ\big)\;+\;\sum_{k\in N(i)} \exp\big(\mathrm{sim}(z₁^i,z₂^k)/τ\big)} ,
$$  

where sim(u,v)=u^T v/(\|u\|\|v\|) and τ is a temperature hyperparameter. We symmetrize over both directions.  

3.3.2 Adversarial Domain Confusion  
To encourage E₁, E₂ to produce domain-invariant embeddings, we train Dϕ to minimize:  

$$
L_{D} = -\mathbb{E}_{x₁}\big[ \log D(E₁(x₁))\big] \;-\;\mathbb{E}_{x₂}\big[ \log(1 - D(E₂(x₂)))\big].
$$  

Simultaneously, we train encoders to maximize the entropy of D’s predictions (via a gradient-reversal layer), yielding:  

$$
L_{adv} = -\mathbb{E}_{x₁}\big[ \log(1 - D(E₁(x₁)))\big] \;-\;\mathbb{E}_{x₂}\big[ \log D(E₂(x₂))\big].
$$  

3.3.3 Task-Oriented Loss (Optional)  
If domain-specific labels y₁, y₂ exist (e.g., class labels for images), we include cross-entropy or regression losses L_task for each encoder to preserve task performance.  

3.4 Training Algorithm  
1. Preprocess inputs:  
   • Domain 1: Normalize neural responses, apply PCA/whitening to size n₁′.  
   • Domain 2: Extract activations from a chosen layer, optionally reduce dimension via PCA.  
2. Initialize θ₁, θ₂, ϕ randomly.  
3. For each mini-batch:  
   a) Sample matched (i,j) pairs and negatives N(i).  
   b) Compute z₁^i=E₁(x₁^i), z₂^j=E₂(x₂^j).  
   c) Update ϕ by descending ∇_ϕ L_D.  
   d) Update θ₁, θ₂ by descending ∇_{θ₁,θ₂} (L_contrast + λ_adv L_adv + λ_task).  
4. Repeat until convergence.  

3.5 Experimental Design  
Datasets  
• Vision:  
   – Biological: IT cortex recordings or fMRI data for a large set of object images (e.g., from BrainScore benchmarks).  
   – Artificial: Activations from ResNet, Transformer-based vision models on the same image set.  
• Language:  
   – Biological: fMRI/MEG responses from human subjects reading sentences from a standardized corpus (e.g., stories, GLUE stimuli).  
   – Artificial: Hidden states from GPT-style LLMs for the same text.  

Baselines  
• RSA (Representational Similarity Analysis).  
• CKA (Centered Kernel Alignment):  

$$
\mathrm{CKA}(Z₁,Z₂)=\frac{\|Z₁^T Z₂\|_F^2}{\|Z₁^T Z₁\|_F\;\|Z₂^T Z₂\|_F}.
$$  

• Linear Procrustes distance.  

Evaluation Metrics  
• Alignment Score: Spearman correlation between flattened RDMs:  

$$
RDM₁(i,j)=1-\mathrm{sim}(z₁^i,z₁^j),\quad RDM₂(i,j)=1-\mathrm{sim}(z₂^i,z₂^j),
$$  
$$
\text{AlignScore}=\mathrm{Spearman}\big(\mathrm{vec}(RDM₁),\mathrm{vec}(RDM₂)\big).
$$  

• Behavioral Congruence: Correlation between error patterns or response times across domain 1 and domain 2 models on held-out stimuli.  
• Generalization: Alignment maintained under domain shifts (e.g., new image sets or out-of-distribution text).  
• Ablations: Vary λ_adv, λ_contrast, choice of negatives, encoder depth.  

4. Expected Outcomes & Impact  

4.1 Expected Outcomes  
1. A trained DEIFS model yielding a shared latent space in which fMRI and model activations for the same stimulus are adjacent, while dissimilar stimuli are separated.  
2. A domain-agnostic alignment metric (AlignScore) that outperforms RSA, CKA, and Procrustes in correlating with behavioral congruence (e.g., Spearman ρ above baselines by ≥0.1).  
3. Empirical insights into which features (e.g., edge detectors in vision, syntactic vs. semantic patterns in language) are most conserved across domains, revealed by latent-space clustering and feature-importance analyses.  
4. Demonstration that co-training a vision or language model with a small amount of neural data, via our adversarial-contrastive objective, improves both alignment and out-of-distribution generalization by 2–5%.  

4.2 Broader Impact  
• Neuroscience: Provides a quantitative tool for testing theories of cortical representation against deep model features, accelerating discovery of universal coding principles.  
• Cognitive Science: Helps link human behavior (e.g., reading times, recognition accuracy) with latent computations in AI, fostering more integrated theories of cognition.  
• AI/ML Engineering: Introduces a plug-and-play module for neural-guided training that can be embedded into existing architectures, offering a path to more human-aligned and interpretable systems.  
• Ethical Considerations: By improving alignment between AI and human representations, the work may support development of safer systems whose internal reasoning better matches human expectations. We will engage with ethicists to monitor potential misuse (e.g., over-reliance on neural proxies) and ensure transparency.  

5. Timeline & Milestones  
Month 1–3: Data collection, preprocessing pipelines, baseline implementations.  
Month 4–6: Develop and debug DEIFS architecture; preliminary experiments on vision data.  
Month 7–9: Extend to language domain; optimize hyperparameters; run ablations.  
Month 10–12: Behavioral congruence studies; write and disseminate findings; prepare workshop submission.  

By the end of this project, we will deliver (i) open-source code for DEIFS, (ii) standardized benchmarks and alignment metrics, and (iii) empirical validations across vision and language, paving the way for a new generation of cross-domain representational alignment methods.