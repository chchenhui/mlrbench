# Self-Supervised Feature Prioritization in Medical Imaging via Gaze-Guided Attention Transfer

## 1. Introduction

Medical image analysis has witnessed remarkable advancements with the integration of deep learning techniques, particularly in tasks such as disease classification, segmentation, and anomaly detection. However, these approaches often face significant challenges, including the requirement for large annotated datasets, the "black-box" nature of model decision-making, and the difficulty in capturing the nuanced domain expertise of medical professionals. Furthermore, supervised learning approaches in healthcare often struggle to generalize to novel conditions or populations due to distribution shifts and the inherent complexity of medical data.

The diagnostic process employed by experienced radiologists represents a highly refined form of visual attention and reasoning. During image assessment, radiologists naturally direct their gaze toward diagnostically significant regions, embodying years of clinical training and domain knowledge. These eye movement patterns provide a rich, untapped source of information about the visual reasoning process underlying expert diagnosis. Recent research has shown that radiologists' gaze patterns correlate strongly with areas of clinical importance, suggesting that these patterns could serve as implicit annotations for guiding machine learning models.

This research proposal introduces a novel self-supervised learning framework that leverages eye-tracking data collected from radiologists as they examine medical images. Rather than requiring explicit manual annotations of pathologies or regions of interest, our approach utilizes the natural visual attention patterns of experts as a form of weak supervision to guide the feature learning process. The proposed method, Gaze-Guided Attention Transfer (GazAT), employs contrastive learning principles to train models that prioritize features in regions attended by experts while creating representations that align with clinical reasoning.

The significance of this research lies in its potential to address several critical challenges in medical image analysis:

1. **Reduction in annotation burden**: By utilizing implicit supervision from eye-tracking data, the approach reduces the need for time-consuming and costly manual annotations by medical experts.

2. **Alignment with clinical reasoning**: The model learns to focus on regions deemed important by medical professionals, potentially improving the clinical relevance of its predictions.

3. **Improved interpretability**: By training models to attend to regions similar to those prioritized by radiologists, the resulting models may produce more interpretable and trustworthy predictions.

4. **Transfer learning capabilities**: The self-supervised nature of the approach allows pre-training on large unlabeled datasets augmented with eye-tracking data, with potential benefits for downstream tasks with limited labeled data.

This research draws upon recent advances in contrastive learning, attention mechanisms, and the emerging field of gaze-guided machine learning. Building on prior work such as Medical contrastive Gaze Image Pre-training (McGIP) and FocusContrast, we propose a more comprehensive framework that not only leverages gaze data for view generation but introduces novel mechanisms for attention transfer and multi-level feature alignment between human visual attention and model-learned representations.

## 2. Methodology

### 2.1 Overview

The proposed GazAT framework integrates radiologists' eye-tracking data into a self-supervised learning paradigm for medical image analysis. The core innovation lies in a multi-stage approach that (1) transforms raw gaze data into attention maps, (2) utilizes these maps to guide contrastive learning, and (3) incorporates attention transfer mechanisms to align model-generated attention with human visual focus patterns. The overall pipeline is illustrated in Figure 1 (conceptual diagram).

### 2.2 Data Collection and Preprocessing

#### 2.2.1 Dataset Requirements

The framework requires two types of data:
1. **Medical images**: Unlabeled medical images (e.g., chest X-rays, CT scans, or MRIs)
2. **Eye-tracking data**: Recorded gaze patterns from radiologists examining these images

We will utilize existing public datasets where available, such as the REFLACX dataset, which contains chest X-rays with corresponding eye-tracking data from multiple radiologists. For modalities where such datasets are not available, we propose a limited data collection phase involving 5-10 radiologists examining 100-200 images each, with appropriate ethical approvals and consent.

#### 2.2.2 Gaze-to-Attention Conversion

Raw eye-tracking data typically consists of fixation points, saccades, and temporal information. We transform this data into continuous attention maps using the following process:

1. **Temporal aggregation**: For each image $I$, we aggregate fixation points $F = \{f_1, f_2, ..., f_n\}$ where each fixation $f_i = (x_i, y_i, d_i)$ includes spatial coordinates and duration.

2. **Attention map generation**: We generate a continuous attention map $A_g$ using a Gaussian kernel centered at each fixation point, weighted by duration:

$$A_g(x, y) = \sum_{i=1}^{n} d_i \cdot \exp\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma^2}\right)$$

where $\sigma$ controls the spread of attention around each fixation point.

3. **Normalization**: The attention map is normalized to the range [0,1]:

$$A_g^{norm}(x, y) = \frac{A_g(x, y) - \min(A_g)}{\max(A_g) - \min(A_g)}$$

4. **Expert aggregation**: For images with multiple radiologists' data, we compute the mean attention map or employ a weighted aggregation based on radiologist experience.

### 2.3 Self-Supervised Learning Architecture

#### 2.3.1 Network Architecture

Our framework employs a dual-branch architecture:
1. **Backbone network**: A convolutional neural network (e.g., ResNet) or vision transformer (e.g., ViT) that extracts features from medical images.
2. **Projection head**: A multilayer perceptron that maps features to a lower-dimensional space for contrastive learning.
3. **Attention module**: A dedicated module that generates attention maps from intermediate network features.

#### 2.3.2 Gaze-Guided Contrastive Learning

We extend traditional contrastive learning by incorporating gaze information:

1. **Augmentation strategy**: For each image $I$, we generate two views:
   - $I_a$: Standard augmentations (random crop, rotation, intensity shifts)
   - $I_g$: Gaze-guided augmentation that preserves high-attention regions

The gaze-guided augmentation employs an attention-preserving transform:

$$I_g = T_g(I, A_g^{norm})$$

where $T_g$ applies stronger augmentations to low-attention regions while preserving high-attention regions:

$$T_g(I, A) = A \odot I + (1-A) \odot T_{strong}(I)$$

where $\odot$ represents element-wise multiplication and $T_{strong}$ represents strong augmentations.

2. **Contrastive loss**: We employ InfoNCE loss to pull together representations of differently augmented views of the same image while pushing apart representations of different images:

$$\mathcal{L}_{contrast} = -\log\frac{\exp(sim(z_a, z_g)/\tau)}{\sum_{k=1}^{2N}\mathbbm{1}_{k\neq a}\exp(sim(z_a, z_k)/\tau)}$$

where $z_a$ and $z_g$ are the projected representations of $I_a$ and $I_g$ respectively, $sim(\cdot,\cdot)$ is the cosine similarity, and $\tau$ is a temperature parameter.

#### 2.3.3 Attention Transfer Mechanism

To align the model's internal attention with radiologists' gaze patterns, we introduce an attention transfer loss:

1. **Model attention extraction**: From intermediate layers of the backbone, we extract attention maps $A_m^l$ at multiple levels $l$:
   - For CNNs: Using Grad-CAM or similar techniques
   - For ViTs: Directly from attention matrices

2. **Multi-scale attention alignment**: We align the model's attention maps with the gaze-derived attention at multiple scales:

$$\mathcal{L}_{attn} = \sum_{l} \lambda_l \cdot D(A_m^l, A_g^{norm})$$

where $D$ is a distance function (e.g., KL divergence or L2 distance) and $\lambda_l$ are layer-specific weights.

3. **Regional contrastive loss**: We introduce a region-based contrastive loss that encourages the model to learn similar representations for regions with high gaze attention across different images:

$$\mathcal{L}_{regional} = -\log\frac{\exp(sim(r_{high}, r_{high}^+)/\tau)}{\exp(sim(r_{high}, r_{high}^+)/\tau) + \sum_{j}\exp(sim(r_{high}, r_{low}^j)/\tau)}$$

where $r_{high}$ represents features from high-attention regions, $r_{high}^+$ represents features from high-attention regions in the positive pair, and $r_{low}^j$ represents features from low-attention regions in negative examples.

#### 2.3.4 Combined Training Objective

The overall training objective combines the above losses:

$$\mathcal{L}_{total} = \mathcal{L}_{contrast} + \alpha \cdot \mathcal{L}_{attn} + \beta \cdot \mathcal{L}_{regional}$$

where $\alpha$ and $\beta$ are hyperparameters that balance the contribution of each loss component.

### 2.4 Implementation Details

The training process consists of two phases:

1. **Pre-training phase**: The model is trained on the unlabeled dataset with eye-tracking data using the combined loss function.

2. **Fine-tuning phase**: The pre-trained backbone is fine-tuned for specific downstream tasks such as classification or segmentation using limited labeled data.

Key implementation details include:
- Backbone: ResNet-50 or Vision Transformer (ViT-B/16)
- Projection head: 2-layer MLP with 256-dimensional output
- Batch size: 128
- Optimizer: AdamW with learning rate 0.0001
- Training schedule: 200 epochs for pre-training, 50 epochs for fine-tuning
- Augmentations: Random crop (0.8-1.0), rotation (±10°), brightness/contrast adjustments (±0.2)

### 2.5 Experimental Design

#### 2.5.1 Datasets

We will evaluate our approach on multiple medical imaging datasets:

1. **REFLACX**: Chest X-rays with eye-tracking data from radiologists
2. **RSNA Pneumonia Detection**: Chest X-rays with pneumonia annotations
3. **ChestX-ray14**: Chest X-rays with 14 disease labels
4. **MIMIC-CXR**: Large-scale chest X-ray dataset with reports

For datasets without eye-tracking data, we will use transfer learning from models pre-trained on REFLACX.

#### 2.5.2 Baseline Methods

We will compare GazAT against several baseline approaches:
1. Supervised learning with full annotations
2. Standard self-supervised methods (SimCLR, BYOL, MoCo)
3. Existing gaze-guided methods (McGIP, FocusContrast)
4. Random attention maps (ablation)

#### 2.5.3 Evaluation Metrics

We will evaluate the approach using the following metrics:

1. **Representation quality**:
   - Linear probing accuracy on downstream tasks
   - Few-shot learning performance (1%, 10% of labels)
   - t-SNE visualization of learned embeddings

2. **Attention alignment**:
   - Normalized Scanpath Saliency (NSS)
   - Area Under ROC Curve (AUC)
   - Pearson correlation between model attention and gaze maps

3. **Downstream task performance**:
   - Classification: Accuracy, F1-score, AUC-ROC
   - Segmentation: Dice coefficient, IoU
   - Anomaly detection: Precision-recall curves

4. **Interpretability metrics**:
   - Pointing game accuracy (localization of pathologies)
   - Radiologist agreement with model attention (user study)

#### 2.5.4 Ablation Studies

We will conduct comprehensive ablation studies to analyze:
1. Impact of different components (attention transfer, regional contrastive loss)
2. Sensitivity to the number of radiologists' gaze data
3. Effect of different attention map generation methods
4. Comparison of different backbone architectures
5. Impact of layer-specific attention transfer weights

## 3. Expected Outcomes & Impact

### 3.1 Technical Outcomes

The successful implementation of the proposed GazAT framework is expected to yield several important technical outcomes:

1. **Improved representation learning**: The proposed approach will likely produce feature representations that better align with radiologists' visual reasoning patterns, capturing clinically relevant features without explicit supervision.

2. **Enhanced transfer learning**: Pre-trained models using our gaze-guided approach should demonstrate stronger transfer to downstream tasks with limited labeled data, potentially achieving 10-15% improvement in accuracy with just 1-10% of labeled data compared to standard self-supervised methods.

3. **Interpretable attention mechanisms**: By aligning model attention with radiologists' gaze patterns, the resulting models should produce more interpretable attention maps that highlight clinically relevant regions, making the decision-making process more transparent.

4. **Domain knowledge encoding**: The framework provides a novel mechanism for implicitly encoding domain knowledge (through experts' visual attention) into deep learning models without requiring explicit rule formulation.

### 3.2 Clinical and Practical Impact

Beyond technical advancements, the research has potential for significant clinical and practical impact:

1. **Reduced annotation burden**: By leveraging eye-tracking data as implicit annotations, the approach could significantly reduce the time and cost associated with creating large labeled datasets for medical imaging, making AI more accessible in healthcare settings.

2. **AI-assisted diagnosis**: The developed models could serve as assistive tools for radiologists, highlighting regions of potential concern that align with expert attention patterns, potentially reducing oversight errors and improving diagnostic efficiency.

3. **Training and education**: The gaze-attention alignment could be used as an educational tool for training junior radiologists, highlighting the visual search patterns of experienced practitioners.

4. **Generalizable methodology**: While focused on radiology, the framework could be extended to other medical imaging domains (e.g., pathology, dermatology) and potentially to non-medical domains where expert visual attention provides valuable signals.

### 3.3 Research Contributions

The proposed research will make several novel contributions to the field:

1. **Methodological innovations**: The multi-level attention transfer and region-based contrastive learning mechanisms represent novel approaches to incorporating human visual attention into self-supervised learning frameworks.

2. **Bridge between perception and AI**: The work helps bridge the gap between human perceptual processes and machine learning by establishing explicit connections between radiologists' visual attention and model-learned features.

3. **Interdisciplinary impact**: By combining elements from computer vision, cognitive science, and medical imaging, the research promotes cross-disciplinary approaches to solving complex healthcare problems.

### 3.4 Limitations and Future Directions

While promising, we acknowledge several potential limitations that will guide future research:

1. **Scalability challenges**: Collecting eye-tracking data at scale remains challenging, and future work will need to explore methods for synthetic gaze data generation or transfer from limited gaze datasets.

2. **Variability in expert gaze**: Individual differences in radiologists' gaze patterns may introduce variability, and methods for addressing this heterogeneity will require further investigation.

3. **Privacy considerations**: The use of eye-tracking data raises privacy concerns that must be carefully addressed in any practical implementation.

4. **Extension to 3D imaging**: While initially focused on 2D imaging, future work should extend the framework to 3D modalities like CT and MRI, which present additional challenges for gaze data collection and representation.

In conclusion, the proposed GazAT framework represents a significant step toward creating more human-aligned and interpretable AI systems for medical imaging, with potential benefits for clinical practice, education, and research. By learning from the implicit knowledge embedded in radiologists' visual attention, we can develop models that not only perform well but do so in ways that align with and complement human expertise.