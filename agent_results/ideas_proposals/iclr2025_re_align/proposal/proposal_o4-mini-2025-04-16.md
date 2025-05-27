Title  
Prototypical Contrastive Alignment for Brain–DNN Representations  

1. Introduction  
Background  
Representational alignment—the degree to which two intelligent systems encode sensory inputs in comparable internal feature spaces—lies at the heart of efforts to relate artificial neural networks (DNNs) to biological brains. Metrics such as Representational Similarity Analysis (RSA) and Centered Kernel Alignment (CKA) have been widely used to quantify post-hoc correspondence between DNN activations and neural recordings (e.g., fMRI, EEG, electrophysiology). However, these approaches suffer from three key limitations:  
• Lack of interpretable anchors: RSA and CKA yield scalar similarity scores but do not reveal the semantic axes along which two systems agree or disagree.  
• Poor generalizability: Metrics tuned to image classification networks may fail on auditory or language models, and metrics computed on one brain area often do not transfer to another.  
• No direct intervention: Post-hoc alignment fails to inform model training, leaving open the question of how to steer DNNs toward brain-like representations.  

Research Objectives  
We propose a framework—Prototypical Contrastive Alignment (PCA*)—that addresses these gaps by (1) extracting a compact library of semantically meaningful “anchor” vectors (prototypes) jointly derived from brain and DNN activations, (2) using these prototypes both as an interpretable alignment metric and as target vectors in a contrastive loss to steer DNN learning, and (3) evaluating the impact of this regularizer on neural predictivity, task performance, and behavioral alignment. Our specific objectives are:  
1. Prototype Extraction: Develop a joint clustering method that yields a small set of shared prototypes capturing the principal semantic dimensions present in both brain and DNN response spaces.  
2. Prototypical Contrastive Loss: Formulate and implement a loss function that simultaneously aligns DNN activations to brain-derived prototypes and maintains intra-prototype discrimination.  
3. Experimental Validation: Quantify improvements in representational alignment (RSA, CKA), neural predictivity (encoding-model performance), task transfer, and behavioral alignment across multiple domains (vision, audition).  
4. Interpretability and Intervention: Demonstrate that prototypes serve as interpretable anchors enabling targeted modification of DNN representations (e.g., increasing alignment on specific semantic axes).  

Significance  
By uniting metric development with active intervention, our approach goes beyond descriptive analysis of representational alignment and moves toward prescriptive model design. The resulting prototype library provides neuroscientists with compact semantic landmarks for comparing brain areas, and the prototypical contrastive loss gives engineers a tool to steer DNNs toward more brain-like computation. This has implications for cognitive neuroscience (revealing shared and divergent representational strategies), for machine learning (enhanced robustness and transfer), and for human–machine interaction (improved behavioral alignment and interpretability).  

2. Methodology  

2.1 Overview  
Our pipeline consists of two sequential stages:  
Stage 1: Joint Prototype Extraction  
Stage 2: Prototype-Based DNN Fine-Tuning  

2.2 Data Collection  
Stimuli  
We select $N$ naturalistic stimuli (e.g., 5,000 images from the Natural Scenes Dataset (NSD) and/or 2,000 short audio clips from a balanced environmental sound corpus).  

Neural Recordings  
• Vision: fMRI responses from $S$ human subjects viewing the $N$ images (e.g., NSD, 7T scans).  
• Audition (optional): Electrophysiology or MEG/EEG responses from $S'$ subjects listening to the $N$ clips.  
Preprocessing  
• DNN activations: For each stimulus, extract layerwise activations $\{z_i^\ell\}_{i=1}^N$ from a backbone model (e.g., ResNet-50 or Vision Transformer).  
• Brain responses: Preprocess fMRI data (motion correction, normalization) and extract voxel-wise response vectors $r_i$. Reduce dimensionality via PCA to $d_r$. Similarly reduce DNN activations to $d_z$ (e.g., $d_r,d_z\approx 128$).  

2.3 Stage 1—Joint Clustering for Prototype Extraction  
Objective  
Derive $K$ prototype vectors that capture the principal semantic axes shared by brain and DNN representations.  

Algorithm  
1. Concatenate reduced representations:  
   For each stimulus $i$, form the joint vector $u_i=[\,r_i\;;\;z_i\,]\in\mathbb{R}^{d_r + d_z}$.  
2. Dimensionality reduction (optional): Apply CCA or joint PCA to $\{u_i\}_{i=1}^N$ to obtain $\tilde u_i\in\mathbb{R}^{d'}$ with $d'\ll d_r+d_z$.  
3. Clustering: Run $K$-means on $\{\tilde u_i\}$ to obtain cluster centroids $\{p_k\in\mathbb{R}^{d'}\}_{k=1}^K$.  
4. Prototype projection: Project each centroid back into brain and DNN subspaces via inverse PCA/CCA transforms to yield $\{p^r_k\in\mathbb{R}^{d_r}\}$ and $\{p^z_k\in\mathbb{R}^{d_z}\}$.  

Interpretation  
Each prototype $k$ corresponds to a compact semantic concept jointly represented in both domains (e.g., “animal face” or “mechanical texture”).  

2.4 Stage 2—Prototypical Contrastive Fine-Tuning  
We fine-tune the DNN (parameters $\theta$) using a loss that aligns its layerwise activations to the brain-derived prototypes $p^z_k$.  

Notation  
Let $f_\theta(x_i)\in\mathbb{R}^{d_z}$ be the DNN activation for stimulus $x_i$. Let $c(i)\in\{1,\dots,K\}$ be the cluster assignment of $i$.  

Prototypical Contrastive Loss  
We adopt an InfoNCE-style objective that pulls $f_\theta(x_i)$ toward its matching prototype $p^z_{c(i)}$ and repels it from the other $K-1$ prototypes:  
$$
\mathcal{L}_{\mathrm{proto}}(x_i;\theta)\;=\;-\,\log 
\frac{\exp\bigl(\mathrm{sim}(f_\theta(x_i),\,p^z_{c(i)})/\tau\bigr)}
{\sum_{k=1}^K \exp\bigl(\mathrm{sim}(f_\theta(x_i),\,p^z_k)/\tau\bigr)}\,,
$$  
where $\mathrm{sim}(u,v)=u^\top v/\|u\|\|v\|$ is cosine similarity and $\tau>0$ is a temperature hyperparameter.  

Overall Training Objective  
We combine the prototypical contrastive loss with the original task loss (e.g., cross-entropy $\mathcal{L}_{\mathrm{CE}}$ for classification):  
$$
\mathcal{L}(\theta)\;=\;\mathcal{L}_{\mathrm{CE}}(\theta)\;+\;\lambda\,\mathbb{E}_{i\in[N]}\bigl[\mathcal{L}_{\mathrm{proto}}(x_i;\theta)\bigr]\,,
$$  
where $\lambda\ge0$ balances task performance against alignment.  

Algorithmic Steps  
1. Pretrain or load a backbone model $f_\theta$.  
2. Extract DNN and brain activations on the stimulus set.  
3. Compute prototypes $\{p_k^z\}$ via Stage 1.  
4. For each training iteration:  
   a. Sample minibatch $\{x_i,y_i\}$.  
   b. Compute activations $f_\theta(x_i)$ and the cross-entropy loss $\mathcal{L}_{\mathrm{CE}}$.  
   c. Compute prototypical loss $\mathcal{L}_{\mathrm{proto}}$ using Eq. above.  
   d. Update $\theta\leftarrow\theta - \eta\nabla_\theta(\mathcal{L}_{\mathrm{CE}} + \lambda\,\mathcal{L}_{\mathrm{proto}})$.  

2.5 Experimental Design and Validation  
Baselines  
• Vanilla training (no alignment regularizer).  
• Additive RSA/CKA regularization: maximizse layerwise RSA/CKA with brain data (Sucholutsky et al., 2023).  
• Prior prototypical contrastive methods (Li et al., 2020; Johnson & Lee, 2023).  

Datasets and Domains  
• Vision: ImageNet classification with NSD fMRI prototypes.  
• Audition (optional): Audio classification with MEG/EEG prototypes.  

Evaluation Metrics  
1. Representational Alignment  
   – RSA score between model layers and brain data on held-out stimuli.  
   – CKA similarly computed.  
2. Neural Predictivity  
   – Fit linear encoding models to predict brain responses from $f_\theta(x)$; report cross-validated $R^2$.  
3. Task Transfer  
   – Downstream classification and detection tasks (e.g., CIFAR-10, COCO).  
4. Behavioral Alignment  
   – Compare feature-importance maps (e.g., Grad-CAM) to human attention/eye-tracking patterns via correlation.  
5. Prototype Interpretability  
   – Human subject studies: label top-activated stimuli per prototype and measure labeling agreement.  

Statistical Analysis  
• For each metric, perform repeated experiments ($M=5$ seeds).  
• Report mean±SE and conduct paired t-tests between PCA and baselines (Bonferroni correction across metrics).  

Implementation Details  
• Prototype count $K\in\{10,20,50\}$ tuned via validation.  
• Temperature $\tau\in\{0.05,0.1,0.2\}$, alignment weight $\lambda\in[0,1]$.  
• Optimizer: AdamW with learning rate schedule.  
• Compute prototypes every $E=5$ epochs to adapt to evolving representations.  

3. Expected Outcomes & Impact  

3.1 Expected Outcomes  
• Improved Alignment Metrics: We anticipate that PCA will yield significantly higher RSA and CKA scores compared to vanilla and prior methods (p < .01).  
• Enhanced Neural Predictivity: Encoding models built on PCA-fine-tuned DNNs should explain a larger fraction of variance in fMRI/EEG responses (Δ$R^2$ > 5%).  
• Robust Task Performance: We expect negligible loss (≤1%) in primary task accuracy despite the alignment regularizer, and improved transfer learning performance on held-out tasks.  
• Semantic Interpretability: Prototype-based anchors will align with human-interpretable categories (behavioral labeling accuracy ≥80%).  
• Tunable Representation Steering: By adjusting $\lambda$, we can systematically increase or decrease alignment on specific prototypes, demonstrating direct intervention capability.  

3.2 Broader Impact  
Neuroscience  
The prototype library offers neuroscientists a compact, semantically meaningful scaffold for comparing representations across brain areas, subjects, and modalities.  

Machine Learning  
Our prototypical contrastive loss provides a general-purpose regularizer to inject external alignment priors (brain, language, semantics) into DNNs, fostering more robust, interpretable, and human-aligned models.  

Cognitive Science  
Interpretable prototypes shed light on which latent dimensions underlie both human cognition and DNN processing, informing theories of concept representation and category structure.  

Ethical and Societal Considerations  
By enabling controlled alignment of AI systems to human neural representations, our method points toward safer, more trustworthy AI that respects human cognitive biases and priorities. Conversely, understanding how to decrease alignment may reveal vulnerabilities and guide the development of systems robust to adversarial manipulation.  

3.3 Contribution to the Re-Align Community and Hackathon  
We will release:  
• A standardized codebase for prototypical clustering and contrastive fine-tuning.  
• Precomputed prototype libraries for common vision and audition datasets.  
• Benchmarks and scripts to reproduce all alignment metrics (RSA, CKA, neural predictivity).  
This resource will facilitate community efforts in the Re-Align hackathon to compare alignment metrics under a common framework, advancing consensus on best practices.  

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––  
*PCA: Prototypical Contrastive Alignment