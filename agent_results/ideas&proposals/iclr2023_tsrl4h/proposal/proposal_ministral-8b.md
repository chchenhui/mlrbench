# Clin-ACT – Clinician-in-the-Loop Active Contrastive Learning for Pediatric Time Series

## Introduction

Time series data plays a pivotal role in healthcare, particularly in diagnosing diseases, predicting disease progression, and monitoring patient conditions. However, the application of machine learning techniques to time series data in healthcare is still limited due to several challenges, including the high dimensionality of the data, irregular sampling, missing values, and the scarcity of labeled data. These challenges are particularly pronounced in pediatric intensive care units (ICU), where time series data are often high-dimensional, irregular, and sparsely labeled due to expert time constraints. Standard self-supervised representations often ignore missingness patterns and lack clinician trust, making it difficult to support downstream tasks such as sepsis prediction effectively.

To address these challenges, this research proposal introduces Clin-ACT, a novel approach that combines contrastive self-supervision with active learning and a prototype-based interpretability layer. The primary goal of Clin-ACT is to generate label-efficient, robust, and interpretable embeddings that support downstream tasks while minimizing the annotation burden on clinicians. By merging active contrastive learning with human feedback and prototype explanations, Clin-ACT aims to deliver actionable, trustworthy time series representations in critical low-data healthcare settings.

## Methodology

### Overview

Clin-ACT consists of three main components: (1) an encoder trained with imputation-aware augmentations, (2) an active learning module that flags informative windows for clinician annotation, and (3) a prototype-based interpretability layer that maps learned embeddings to clinical archetypes and generates feature saliency maps.

### Encoder Training

The encoder is trained using contrastive self-supervision with imputation-aware augmentations tailored to the irregular sampling and outliers present in pediatric ICU time series data. The augmentations include:

- **Time Warping**: To handle irregular sampling, time warping is applied to create augmented sequences by shifting the time axis.
- **Outlier Imputation**: Outliers are imputed using a robust statistical method, such as the median or trimmed mean, to reduce their impact on the representation learning process.

The contrastive loss used for training is the Noise Contrastive Estimation (NCE) loss, which maximizes the agreement between positive pairs (augmented sequences) and minimizes the agreement between negative pairs (randomly sampled sequences). The loss function is defined as:

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i, z_i^+))}{\exp(\text{sim}(z_i, z_i^+)) + \sum_{j \neq i} \exp(\text{sim}(z_i, z_j^-))}
$$

where \(z_i\) and \(z_i^+\) are the embeddings of the original and augmented sequences, respectively, and \(z_j^-\) are the embeddings of negative samples.

### Active Learning

The active learning module uses an uncertainty-diversity criterion to flag the most informative windows for clinician annotation. The criterion is based on the entropy of the predicted class probabilities and the diversity of the embeddings within each window. The entropy \(H\) of a window \(W\) is calculated as:

$$
H(W) = -\sum_{c} p(c|W) \log p(c|W)
$$

where \(p(c|W)\) is the probability distribution of the predicted class for window \(W\). The diversity \(D\) of a window \(W\) is calculated as the mean pairwise cosine similarity between the embeddings within the window:

$$
D(W) = \frac{1}{|W|^2} \sum_{i,j \in W} \text{cos}(z_i, z_j)
$$

The uncertainty-diversity score \(U\) for a window \(W\) is:

$$
U(W) = H(W) \times D(W)
$$

Windows with high uncertainty-diversity scores are flagged for clinician annotation, reducing the number of labels required by an estimated 60%.

### Prototype-based Interpretability

The prototype-based interpretability layer maps the learned embeddings to clinical archetypes using a lightweight prototype module. The module consists of a set of prototype vectors, each corresponding to a specific clinical condition or phenotype. The prototypes are initialized using a clustering algorithm, such as k-means, and updated during training using the embeddings of the labeled data. The similarity between an embedding \(z\) and a prototype \(p\) is calculated using the cosine similarity:

$$
\text{sim}(z, p) = \frac{z \cdot p}{\|z\| \|p\|}
$$

The prototype module generates feature saliency maps by highlighting the features that contribute most to the similarity between an embedding and its corresponding prototype. The saliency map is calculated as:

$$
S(z, p) = \frac{\partial \text{sim}(z, p)}{\partial z}
$$

where \(\partial \text{sim}(z, p) / \partial z\) is the gradient of the similarity score with respect to the embedding \(z\). The saliency map provides clinicians with transparent insights into what drives each representation, enhancing trust in the model's outputs.

### Experimental Design

Clin-ACT will be validated on pediatric ICU vital signs and lab series data, with a focus on sepsis prediction. The dataset will be randomly split into training, validation, and test sets, with 70% of the data used for training, 15% for validation, and 15% for testing. The performance of Clin-ACT will be evaluated using the following metrics:

- **Accuracy**: The proportion of correctly predicted sepsis cases.
- **Precision, Recall, and F1-Score**: To evaluate the model's performance on the minority class (sepsis).
- **Clinician Satisfaction**: A survey will be administered to clinicians to assess their satisfaction with the interpretability of the model's outputs.

### Evaluation Metrics

The performance of Clin-ACT will be evaluated using the following metrics:

- **Accuracy**: The proportion of correctly predicted sepsis cases.
- **Precision, Recall, and F1-Score**: To evaluate the model's performance on the minority class (sepsis).
- **Clinician Satisfaction**: A survey will be administered to clinicians to assess their satisfaction with the interpretability of the model's outputs.

## Expected Outcomes & Impact

### Technical Outcomes

The primary technical outcome of this research is the development of Clin-ACT, a novel approach that combines contrastive self-supervision, active learning, and prototype-based interpretability for pediatric ICU time series data. Clin-ACT addresses the challenges of handling missing and irregular data, limited labeled data, and the need for interpretable and trustworthy outputs in critical low-data healthcare settings. The technical outcomes of this research include:

- A robust encoder architecture that handles irregular sampling and outliers in pediatric ICU time series data.
- An active learning module that efficiently reduces the annotation burden on clinicians by flagging the most informative windows for annotation.
- A prototype-based interpretability layer that provides clinicians with transparent insights into the model's outputs.

### Clinical Impact

The clinical impact of this research is the development of actionable, trustworthy time series representations that support downstream tasks such as sepsis prediction in pediatric ICU settings. By merging active contrastive learning with human feedback and prototype explanations, Clin-ACT aims to deliver clinically relevant insights that enhance decision-making and improve patient outcomes. The clinical impact of this research includes:

- Improved sepsis prediction performance, with an estimated accuracy improvement of +12% compared to standard baselines.
- Enhanced clinician trust in the model's outputs, leading to increased adoption and effective decision-making.
- Reduced annotation burden on clinicians, enabling more efficient use of their time and expertise.

### Societal Impact

The societal impact of this research is the development of a label-efficient, robust, and interpretable embedding approach that supports downstream tasks in critical low-data healthcare settings. By addressing the challenges of handling missing and irregular data, limited labeled data, and the need for interpretable and trustworthy outputs, Clin-ACT has the potential to improve healthcare outcomes and reduce the burden on clinicians. The societal impact of this research includes:

- Improved healthcare outcomes for pediatric patients, particularly those with sepsis, through more accurate and timely diagnosis and treatment.
- Reduced healthcare costs associated with misdiagnosis and delayed treatment.
- Enhanced clinician satisfaction and efficiency, leading to better patient care and reduced burnout.

In conclusion, Clin-ACT represents a significant step forward in the development of time series representation learning approaches for healthcare applications. By combining contrastive self-supervision, active learning, and prototype-based interpretability, Clin-ACT addresses the unique challenges of pediatric ICU time series data and delivers actionable, trustworthy time series representations that support downstream tasks and enhance decision-making in critical low-data healthcare settings.