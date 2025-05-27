# Research Proposal: Gaze-Guided Contrastive Learning for Self-Supervised Feature Prioritization in Medical Imaging  

## 1. Title  
**Aligning AI with Radiological Expertise: Self-Supervised Feature Prioritization via Multi-Scale Gaze Contrastive Learning**  

---

## 2. Introduction  
### Background  
Medical imaging plays a critical role in clinical diagnostics, yet supervised deep learning models remain constrained by the scarcity of annotated data. Unsupervised methods, while promising, often fail to prioritize clinically relevant regions without explicit labels. Meanwhile, radiologists naturally focus on diagnostically significant areas during image interpretation, as reflected in their eye gaze patterns. These gaze trajectories offer a cost-effective, annotation-free signal that encodes expert reasoning. Recent advances in eye-tracking technology and self-supervised learning (SSL) frameworks, such as contrastive methods, present an opportunity to bridge this gap by aligning model feature prioritization with human attention.  

### Research Objectives  
This proposal aims to develop **GazeCL**, a gaze-guided contrastive learning framework that:  
1. Leverages radiologists’ eye-tracking data to guide SSL in medical imaging.  
2. Prioritizes features in diagnostically critical regions without manual annotations.  
3. Improves generalization in low-data regimes and enhances model interpretability.  

### Significance  
- **Annotation Efficiency**: Reduces reliance on costly labeled datasets by exploiting gaze as weak supervision.  
- **Human-AI Alignment**: Ensures AI models focus on regions validated by clinical expertise, fostering trust.  
- **Clinical Impact**: Enhances anomaly detection and classification accuracy, particularly in resource-constrained settings.  

---

## 3. Methodology  
### 3.1 Data Collection and Preprocessing  
- **Datasets**: Utilize publicly available chest X-ray datasets (e.g., CheXpert, MIMIC-CXR) paired with gaze data from radiologists diagnosing pathologies.  
- **Gaze Heatmap Generation**: Convert raw gaze coordinates into spatial heatmaps using Gaussian filtering:  
  $$
  G(x,y) = \sum_{i=1}^{N} \exp\left(-\frac{(x - x_i)^2 + (y - y_i)^2}{2\sigma^2}\right),
  $$
  where $(x_i, y_i)$ are gaze fixation points and $\sigma$ controls spatial spread.  
- **Augmentation**: Apply gaze-preserving augmentations (e.g., mild rotations, translations) to avoid distorting critical regions, as per findings in arXiv:2501.02451.  

### 3.2 Model Architecture  
**Backbone**: Vision Transformer (ViT) with patch-based embedding for global context capture.  
**Key Components**:  
1. **Gaze-Guided Attention Layer**: Modulates patch embeddings using attention weights derived from gaze heatmaps:  
   $$
   \alpha_i = \text{Sigmoid}(W \cdot z_i + b), \quad \alpha_i \propto G(p_i),
   $$
   where $z_i$ is the patch embedding and $G(p_i)$ is the gaze intensity at patch $p_i$.  
2. **Projection Head**: Maps attended embeddings to a latent space for contrastive learning.  

### 3.3 Gaze-Guided Contrastive Learning  
- **Positive/Negative Sampling**: For each image, extract two views:  
  - **Gazed View**: Patches with intensity above threshold $\gamma$ in $G$.  
  - **Non-Gazed View**: Randomly sampled patches below $\gamma$.  
- **Loss Function**: Maximize similarity between gazed views while distancing non-gazed regions:  
  $$
  \mathcal{L}_{\text{GazeCL}} = -\log \frac{\sum_{j \in \mathcal{P}_i} \exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \in \mathcal{P}_i \cup \mathcal{N}_i} \exp(\text{sim}(z_i, z_k)/\tau)},
  $$
  where $\mathcal{P}_i$ and $\mathcal{N}_i$ are positive (gazed) and negative (non-gazed) patches for anchor $i$, $\tau$ is a temperature parameter, and $\text{sim}$ denotes cosine similarity.  

### 3.4 Experimental Design  
- **Baselines**: Compare against state-of-the-art gaze-guided models (McGIP, FocusContrast, GazeGNN) and SSL methods (SimCLR, MoCo).  
- **Tasks**:  
  1. **Pathology Classification**: Accuracy, AUC-ROC on chest X-ray datasets.  
  2. **Anomaly Localization**: Dice score between model attention maps and ground-truth lesion masks.  
  3. **Attention Alignment**: Normalized Scanpath Saliency (NSS), Similarity Index (SIM) against radiologists’ gaze heatmaps.  
- **Low-Data Regime Evaluation**: Train on 1%, 10%, and 50% labeled data subsets after SSL pretraining.  

---

## 4. Expected Outcomes & Impact  
### Expected Outcomes  
1. **Improved Diagnostic Accuracy**: GazeCL will achieve ≥5% higher AUC-ROC in pathology classification compared to non-gaze SSL baselines.  
2. **Enhanced Interpretability**: Model attention maps will exhibit ≥0.7 NSS correlation with expert gaze patterns.  
3. **Data Efficiency**: Pretraining with GazeCL will reduce labeled data requirements by 30–50% for downstream tasks.  

### Broader Impact  
- **Clinical Workflow Integration**: Enables AI systems to mirror radiologists’ reasoning, improving adoption in hospitals.  
- **Resource Optimization**: Reduces dependency on annotated data, democratizing access to high-quality diagnostic tools.  
- **Research Community**: Establishes a blueprint for leveraging gaze in SSL across domains like autonomous driving and AR/VR.  

---

## 5. Conclusion  
By aligning self-supervised feature learning with radiologists’ gaze patterns, GazeCL bridges the gap between human expertise and machine intelligence. This framework promises to advance medical AI systems toward greater accuracy, interpretability, and clinical trust, while fostering interdisciplinary collaboration in gaze-guided machine learning.