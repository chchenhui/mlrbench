# Self-Supervised Feature Prioritization in Medical Imaging via Eye Gaze Patterns

## 1. Title

Self-Supervised Feature Prioritization in Medical Imaging via Eye Gaze Patterns

## 2. Introduction

### Background

Medical imaging, a cornerstone of modern diagnostics, is increasingly leveraging machine learning to improve accuracy and efficiency. However, the reliance on extensive labeled datasets poses a significant challenge, particularly in low-resource settings. Traditional supervised learning methods require expensive and time-consuming manual annotations, which can be prohibitive. Eye-tracking technology offers a promising solution by providing an indirect yet powerful signal of human attention, which can be used to guide feature learning in medical imaging. This research aims to develop a self-supervised framework that utilizes radiologists' eye-tracking data to prioritize clinically relevant features in convolutional or transformer-based networks, thereby enhancing unsupervised learning in medical imaging.

### Research Objectives

The primary objectives of this research are:
1. To develop a self-supervised learning framework that incorporates eye-tracking data to guide feature importance in medical imaging.
2. To evaluate the effectiveness of this approach in improving anomaly detection accuracy and generating interpretable attention maps.
3. To assess the generalizability and scalability of the proposed method across different medical imaging modalities and datasets.

### Significance

The proposed approach has the potential to revolutionize unsupervised learning in medical imaging by:
- Eliminating the need for costly manual annotations.
- Enabling AI systems to generalize better in low-data regimes.
- Enhancing trust in medical diagnostics by aligning model behavior with clinician workflows.
- Providing interpretable attention maps that mirror expert focus, aiding in the explainability of AI decisions.

## 3. Methodology

### Data Collection

The proposed method will utilize large-scale eye-tracking datasets from radiologists, specifically focusing on chest X-rays. These datasets will include gaze heatmaps that capture the areas of interest during diagnosis. The datasets will be anonymized to ensure data privacy and comply with ethical guidelines.

### Framework Overview

The framework comprises two main components:
1. **Gaze Data Preprocessing**: This involves converting gaze heatmaps into a format suitable for training the neural network.
2. **Self-Supervised Learning**: This involves training a convolutional or transformer-based network to contrast gaze-attended image regions against non-attended ones, enforcing similarity in embeddings for regions fixated during diagnoses.

### Algorithmic Steps

#### Step 1: Gaze Data Preprocessing

Gaze heatmaps will be preprocessed to extract regions of interest (ROIs) that radiologists fixate upon during diagnosis. This involves:
1. **Heatmap Normalization**: Normalizing the gaze heatmaps to ensure uniform distribution.
2. **ROI Extraction**: Extracting ROIs based on a threshold value that defines significant fixations.

#### Step 2: Contrastive Learning

The network will be trained using contrastive learning, where images with similar gaze patterns are treated as positive pairs, and those with different patterns are treated as negative pairs. The key components of this step are:
1. **Image Embedding**: Extracting image embeddings using a convolutional or transformer-based network.
2. **Contrastive Loss**: Calculating contrastive loss to enforce similarity in embeddings for gaze-attended regions:
   \[
   L_{contrastive} = \sum_{i=1}^{N} \sum_{j=1}^{M} \left[ y_{ij} \cdot \text{sim}(f(x_i), f(x_j)) + (1 - y_{ij}) \cdot \text{max}(0, m - \text{sim}(f(x_i), f(x_j))) \right]
   \]
   where \( f(x) \) is the embedding function, \( y_{ij} \) is the label indicating whether \( x_i \) and \( x_j \) are positive pairs, \( \text{sim}(f(x_i), f(x_j)) \) is the similarity function (e.g., cosine similarity), and \( m \) is the margin.

#### Step 3: Auxiliary Contrastive Loss

To further enhance the learning of gaze-attended regions, an auxiliary contrastive loss will be applied, which contrasts the embeddings of gaze-attended regions with non-attended regions:
\[
L_{auxiliary} = \sum_{i=1}^{N} \sum_{j=1}^{M} \left[ y_{ij} \cdot \text{sim}(f(x_i), f(x_j)) + (1 - y_{ij}) \cdot \text{max}(0, m - \text{sim}(f(x_i), f(x_j))) \right]
\]
where \( y_{ij} \) is the label indicating whether \( x_i \) and \( x_j \) are gaze-attended pairs.

### Experimental Design

The experimental design will involve the following steps:
1. **Dataset Preparation**: Preparing the dataset by splitting it into training, validation, and test sets.
2. **Model Training**: Training the network using the contrastive and auxiliary contrastive losses.
3. **Evaluation Metrics**: Evaluating the model using accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC) for anomaly detection tasks. Additionally, generating and analyzing attention maps to assess interpretability.

### Validation

To validate the method, we will compare the performance of the gaze-guided model with baseline models that do not incorporate gaze data. The comparison will be conducted on multiple medical imaging datasets to assess the generalizability of the proposed approach.

## 4. Expected Outcomes & Impact

### Expected Outcomes

The expected outcomes of this research include:
1. **Improved Anomaly Detection Accuracy**: The proposed method is expected to enhance the accuracy of anomaly detection in unsupervised settings by leveraging radiologists' gaze patterns.
2. **Interpretable Attention Maps**: The model will generate attention maps that mirror expert focus, providing insights into the regions of interest for diagnosis.
3. **Generalizability and Scalability**: The method will be evaluated across different medical imaging modalities and datasets to assess its generalizability and scalability.

### Impact

The impact of this research is expected to be significant in several ways:
- **Enhanced AI in Medical Imaging**: By aligning AI with expert reasoning, the proposed method can improve the performance and trustworthiness of AI systems in medical diagnostics.
- **Reduced Data Annotation Costs**: Eliminating the need for manual annotations can significantly reduce the costs and time associated with developing AI models for medical imaging.
- **Improved Explainability**: The interpretable attention maps generated by the model can enhance the explainability of AI decisions, aiding in the adoption and acceptance of AI in healthcare.
- **Advancements in Unsupervised Learning**: The proposed method can contribute to the development of new techniques for unsupervised learning in medical imaging, enabling AI systems to generalize better in low-data regimes.

## Conclusion

This research aims to develop a self-supervised learning framework that utilizes eye-tracking data to guide feature importance in medical imaging. By leveraging radiologists' gaze patterns, the proposed method has the potential to enhance unsupervised learning, improve anomaly detection accuracy, and generate interpretable attention maps. The expected outcomes and impact of this research can significantly advance the field of medical imaging and AI, contributing to more accurate, efficient, and trustworthy diagnostics.