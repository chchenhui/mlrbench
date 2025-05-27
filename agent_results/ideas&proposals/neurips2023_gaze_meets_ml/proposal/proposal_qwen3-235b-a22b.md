# Self-Supervised Feature Prioritization in Medical Imaging via Eye Gaze Patterns  

## Introduction  

### Background  
Medical imaging plays a pivotal role in disease diagnosis, yet unsupervised feature learning in this domain faces significant challenges due to the scarcity of labeled data and the high cost of expert annotations. Radiologists’ diagnostic workflows inherently involve selective visual attention, where gaze patterns naturally highlight regions of clinical relevance, such as lesions or abnormalities. These gaze signals offer a rich, untapped source of weak supervision for training machine learning models. Recent advances in eye-tracking technology and the availability of large-scale gaze datasets (e.g., chest X-ray gaze data) have enabled the development of gaze-guided contrastive learning frameworks, such as Medical contrastive Gaze Image Pre-training (McGIP) and FocusContrast. These methods leverage gaze patterns to construct positive/negative sample pairs or attention-aware augmentations, improving downstream task performance without manual labels. However, existing approaches often rely on predefined gaze heatmap thresholds or auxiliary networks to integrate gaze data, limiting their adaptability to diverse imaging modalities and clinical workflows.  

### Research Objectives  
This research proposes a novel self-supervised framework that directly incorporates radiologists’ gaze patterns into the feature learning process of convolutional neural networks (CNNs) or vision transformers (ViTs). The primary objectives are:  
1. **Gaze-Driven Contrastive Learning**: Develop a contrastive loss function that explicitly contrasts gaze-attended regions (positive samples) against non-attended regions (negative samples) to prioritize diagnostically relevant features.  
2. **Architecture Integration**: Design a modular architecture that seamlessly integrates gaze data into both CNN and transformer backbones, enabling end-to-end training without preprocessing gaze signals into binary masks.  
3. **Validation on Clinical Tasks**: Evaluate the framework’s efficacy in unsupervised anomaly detection (e.g., pneumonia in chest X-rays) and few-shot classification, benchmarking performance against state-of-the-art gaze-agnostic and gaze-guided models.  
4. **Interpretability Analysis**: Quantify the alignment between model attention maps and radiologists’ gaze patterns to enhance trust in AI-driven diagnostics.  

### Significance  
This work addresses critical gaps in medical AI by:  
- **Reducing Annotation Burden**: Eliminating the need for pixel-level or image-level labels through gaze-guided self-supervision.  
- **Improving Generalization**: Enhancing model robustness in low-data regimes by mimicking expert visual reasoning.  
- **Enabling Interpretable AI**: Producing attention maps that mirror clinician workflows, fostering clinician-AI collaboration.  
- **Advancing Gaze-ML Integration**: Proposing a scalable framework applicable to diverse imaging modalities (e.g., retinal scans, mammography).  

---

## Methodology  

### Data Collection and Preprocessing  
**Datasets**:  
- **OpenI and MIMIC-CXR**: Publicly available chest X-ray datasets with associated radiologist gaze data (fixation points and dwell times).  
- **NIH ChestX-ray14**: For cross-dataset generalization testing, using weak outcome labels (e.g., "pneumonia") as proxies for evaluation.  
- **Private Retinal Imaging Dataset**: To validate modality-agnostic applicability (with institutional review board approval).  

**Gaze Preprocessing**:  
Raw gaze coordinates $(x_t, y_t)$ are converted into spatiotemporal heatmaps $G \in \mathbb{R}^{H \times W}$ via kernel density estimation:  
$$
G(i,j) = \sum_{t=1}^T K_\sigma(i - x_t, j - y_t),
$$  
where $K_\sigma$ is a Gaussian kernel with bandwidth $\sigma$. Heatmaps are normalized to $[0,1]$ and resized to match input image dimensions.  

### Model Architecture  
The framework integrates gaze data into a dual-branch Siamese network (Figure 1) with:  
1. **Image Encoder**: A pretrained ViT or ResNet-50 backbone $f_\theta$ mapping images $X$ to embeddings $z \in \mathbb{R}^d$.  
2. **Gaze-Guided Attention Module**: A lightweight convolutional network $A_\phi$ that transforms gaze heatmaps $G$ into spatial attention weights $\alpha \in \mathbb{R}^{H' \times W'}$, where $H', W'$ are feature map dimensions.  
3. **Contrastive Loss Head**: Computes similarity between attended and non-attended regions using a temperature-scaled cross-entropy loss.  

**Attention-Guided Feature Modulation**:  
Given an image $X$ and its gaze heatmap $G$, the attended feature map $F_{\text{att}}$ is computed as:  
$$
F_{\text{att}} = \alpha(G) \odot f_\theta(X),
$$  
where $\odot$ denotes element-wise multiplication. Non-attended regions are sampled from $F_{\text{non-att}} = (1 - \alpha(G)) \odot f_\theta(X)$.  

### Algorithmic Steps  
1. **Self-Supervised Pretraining**:  
   - Input pairs: Augmented views of the same image with identical gaze heatmaps.  
   - Contrastive loss:  
     $$
     \mathcal{L}_{\text{cont}} = -\log \frac{\exp(\text{sim}(F_{\text{att}}^+, F_{\text{att}}^-)/\tau)}{\sum_{k=1}^K \exp(\text{sim}(F_{\text{att}}^+, F_{\text{non-att}}^{(k)})/\tau)},
     $$  
     where $\text{sim}(\cdot)$ is cosine similarity, $\tau$ is a temperature parameter, and $K$ is the number of negative samples.  
   - Reconstruction loss (optional):  
     $$
     \mathcal{L}_{\text{recon}} = \|X - \text{Decoder}(F_{\text{att}})\|_2^2.
     $$  
   - Total loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{cont}} + \lambda_2 \mathcal{L}_{\text{recon}}$.  

2. **Fine-Tuning for Downstream Tasks**:  
   - Freeze $f_\theta$ and train a classifier head on limited labeled data.  
   - For unsupervised anomaly detection, cluster embeddings $z$ using DBSCAN or Gaussian Mixture Models.  

### Experimental Design  
**Baselines**:  
- **Gaze-Agnostic**: MoCo, SimCLR, and DINO pretrained without gaze guidance.  
- **Gaze-Guided**: McGIP, FocusContrast, and GazeGNN.  

**Evaluation Metrics**:  
- **Quantitative**: AUC-ROC, sensitivity/specificity at optimal thresholds, F1-score.  
- **Interpretability**: Pearson correlation between model attention maps and ground-truth gaze heatmaps.  
- **Robustness**: Performance under low-data regimes (1%, 5%, 10% labeled samples).  

**Ablation Studies**:  
- Impact of $\lambda_1/\lambda_2$ ratios and $\tau$ values.  
- Comparison of CNN vs. ViT backbones.  
- Sensitivity to gaze data quality (e.g., noise injection).  

**Ethical Considerations**:  
- Data anonymization and compliance with HIPAA/GDPR regulations.  
- Auditing for bias in gaze patterns across radiologist demographics.  

---

## Expected Outcomes & Impact  

### Technical Advancements  
1. **Improved Anomaly Detection**: We anticipate a ≥5% increase in AUC-ROC over gaze-agnostic baselines in unsupervised pneumonia detection on MIMIC-CXR, validated via bootstrapping ($p < 0.05$).  
2. **Interpretable Attention**: Model-derived attention maps will correlate strongly with radiologists’ gaze patterns ($r > 0.7$), enhancing trust in AI decisions.  
3. **Few-Shot Learning**: The framework will achieve >85% accuracy in few-shot classification tasks with ≤10% labeled data, outperforming McGIP by ≥8%.  

### Clinical and Societal Impact  
- **Cost Reduction**: Eliminating manual annotations could save ~\$2.5M annually in radiology AI development costs (based on NIH grant estimates).  
- **Global Health Equity**: Enabling accurate diagnostics in resource-limited settings where labeled medical data is scarce.  
- **Cross-Disciplinary Synergy**: Bridging cognitive science and ML by formalizing gaze as a proxy for expert reasoning.  

### Addressing Literature Challenges  
- **Data Scarcity**: Transfer learning from chest X-rays to retinal imaging will demonstrate cross-modality generalization.  
- **Variability**: Attention modules will adaptively weigh gaze signals, mitigating individual differences.  
- **Privacy**: Anonymized gaze data and federated learning protocols will safeguard sensitive information.  

This work will advance gaze-assisted ML by providing a scalable, self-supervised paradigm that aligns AI with human expertise, ultimately fostering safer and more efficient clinical AI systems.