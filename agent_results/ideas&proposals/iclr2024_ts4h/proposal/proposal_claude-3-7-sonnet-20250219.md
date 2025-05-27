# Adaptive Continuous-Time Masked Autoencoder for Multi-Modal Health Signals: A Self-Supervised Foundation Model for Irregular Time Series Data

## 1. Introduction

Healthcare systems generate vast amounts of time series data from diverse sources, including electronic health records (EHR), physiological monitors (ECG, EEG), and wearable devices. These data streams hold immense potential for improving patient outcomes through early disease detection, personalized treatment, and continuous health monitoring. However, healthcare time series present unique challenges that have hindered the development of effective machine learning models:

1. **Irregular sampling and missing data**: Health data is often collected at variable intervals with gaps and missing values due to equipment malfunctions, patient non-compliance, or clinical limitations.

2. **Multi-modal misalignment**: Different data sources (e.g., lab tests, vital signs, wearable sensors) operate on different timescales and frequently lack temporal alignment.

3. **High dimensionality and noise**: Health signals contain numerous features with varying levels of noise and relevance.

4. **Cross-institutional variability**: Data collection protocols and equipment specifications differ across healthcare institutions, complicating model generalization.

Current approaches to these challenges rely heavily on data preprocessing techniques such as imputation and resampling, which introduce assumptions that may not reflect the underlying physiological processes. While recent advances in transformers and masked autoencoders have shown promise in handling sequential data, they typically assume regular sampling intervals and complete data, limiting their direct applicability to healthcare time series.

### Research Objectives

This research proposes the Adaptive Continuous-Time Masked Autoencoder (CT-MAE), a self-supervised foundation model specifically designed to address the challenges of multi-modal health time series. Our objectives are to:

1. Develop a continuous-time transformer architecture that natively handles irregular sampling intervals without requiring imputation or resampling.

2. Design a multi-modal masking strategy that encourages the model to learn robust representations by reconstructing missing segments jointly across different data sources.

3. Incorporate learnable temporal kernels that capture the varying time dependencies in health data across different timescales.

4. Create a foundation model that can be efficiently fine-tuned for various downstream tasks such as disease prediction, anomaly detection, and treatment recommendation.

### Significance

The proposed CT-MAE addresses a critical gap in healthcare AI by providing a unified framework for learning from irregular, multi-modal time series. Unlike previous approaches that treat irregularity as a preprocessing concern, our model embraces the inherent temporal structure of health data. This paradigm shift offers several potential benefits:

1. **Improved predictive performance**: By preserving the original temporal information and leveraging complementary signals across modalities, CT-MAE can potentially extract more meaningful patterns for clinical prediction tasks.

2. **Uncertainty quantification**: The model's ability to process data in its natural form facilitates more reliable uncertainty estimates, crucial for clinical decision support.

3. **Interpretability**: The attention mechanisms across time and modalities provide insights into which signals and timepoints contribute most to predictions.

4. **Scalability**: The self-supervised pretraining approach enables learning from large unlabeled datasets, addressing the common scarcity of labeled health data.

Successful development of CT-MAE would represent a significant advancement in time series modeling for healthcare, potentially enabling more accurate, reliable, and interpretable AI systems that can operate effectively in real-world clinical settings with incomplete and irregular data.

## 2. Methodology

### 2.1 Overview of Adaptive Continuous-Time Masked Autoencoder (CT-MAE)

The proposed CT-MAE is a self-supervised learning framework designed to learn robust representations from irregular, multi-modal health time series data. The architecture consists of three main components: (1) a continuous-time encoder that processes irregularly sampled inputs, (2) a strategic masking mechanism that operates across time and modalities, and (3) a multi-modal decoder that reconstructs the masked segments. Figure 1 illustrates the overall architecture of CT-MAE.

### 2.2 Data Representation

We consider a collection of multi-modal time series data from $N$ patients. For each patient $i$, we have $M$ different modalities, where each modality $m$ consists of a sequence of observations $\{(t_{i,m,j}, \mathbf{x}_{i,m,j})\}_{j=1}^{T_{i,m}}$. Here, $t_{i,m,j}$ represents the timestamp of the $j$-th observation in modality $m$ for patient $i$, and $\mathbf{x}_{i,m,j} \in \mathbb{R}^{d_m}$ is the corresponding feature vector with dimension $d_m$. The number of observations $T_{i,m}$ and the timestamps can vary across patients and modalities, reflecting the irregularity of health data.

### 2.3 Continuous-Time Encoder

The encoder transforms the irregular multi-modal time series into a continuous-time representation. For each modality $m$, we first project the input features using a modality-specific linear projection:

$$\mathbf{v}_{i,m,j} = \mathbf{W}_m \mathbf{x}_{i,m,j} + \mathbf{b}_m$$

where $\mathbf{W}_m \in \mathbb{R}^{d_{model} \times d_m}$ and $\mathbf{b}_m \in \mathbb{R}^{d_{model}}$ are learnable parameters.

To encode the temporal information, we represent each timestamp using a set of learnable temporal basis functions. Specifically, we define a set of $K$ Gaussian-process-inspired basis functions:

$$\phi_k(t) = \exp\left(-\frac{(t - \mu_k)^2}{2\sigma_k^2}\right)$$

where $\mu_k$ and $\sigma_k$ are learnable parameters that determine the center and width of each basis function. The temporal encoding for timestamp $t_{i,m,j}$ is computed as:

$$\mathbf{τ}_{i,m,j} = \sum_{k=1}^{K} \phi_k(t_{i,m,j}) \mathbf{w}_k$$

where $\mathbf{w}_k \in \mathbb{R}^{d_{model}}$ are learnable weight vectors. This approach allows the model to capture temporal patterns at multiple scales and adapt to the irregular sampling in health data.

The input representation for the $j$-th observation of modality $m$ for patient $i$ is then:

$$\mathbf{h}_{i,m,j}^{(0)} = \mathbf{v}_{i,m,j} + \mathbf{τ}_{i,m,j} + \mathbf{e}_m$$

where $\mathbf{e}_m \in \mathbb{R}^{d_{model}}$ is a modality-specific embedding vector.

### 2.4 Continuous-Time Transformer

The continuous-time transformer processes the input representations using a multi-layer transformer architecture with modifications to handle irregular time intervals. For each layer $l$, we compute:

$$\mathbf{h}_{i,m,j}^{(l)} = \text{TransformerLayer}(\mathbf{h}_{i,m,j}^{(l-1)}, \{\mathbf{h}_{i,m',j'}^{(l-1)}, t_{i,m',j'}\}_{m',j'})$$

The key innovation in our transformer is the continuous-time attention mechanism that incorporates temporal distances between observations. For observations at times $t_a$ and $t_b$, we define a time-decay function:

$$\alpha(t_a, t_b) = \exp(-\lambda |t_a - t_b|)$$

where $\lambda$ is a learnable parameter. The attention weights are computed as:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{T}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \odot \mathbf{A}\right)\mathbf{V}$$

where $\mathbf{A}_{ab} = \alpha(t_a, t_b)$ is the matrix of time-decay factors, and $\odot$ denotes element-wise multiplication.

To handle multi-modal data, we implement cross-modal attention where queries from one modality can attend to keys and values from all modalities, enabling the model to leverage complementary information across data sources.

### 2.5 Masking Strategy

We employ a strategic masking approach that operates at three levels:

1. **Temporal masking**: We randomly mask continuous segments of time for each modality independently, with a probability $p_t$.

2. **Feature masking**: Within each modality, we randomly mask individual features with a probability $p_f$.

3. **Modality masking**: We completely mask certain modalities with a probability $p_m$, encouraging the model to learn cross-modal relationships.

For a given patient $i$, the masked input sequence for modality $m$ is denoted as $\{\mathbf{h}_{i,m,j}^{(mask)}\}$, where masked values are replaced with a learnable [MASK] token.

### 2.6 Multi-Modal Decoder

The decoder aims to reconstruct the original values of the masked segments by leveraging information from the unmasked portions across all modalities. We employ a transformer-based decoder with cross-modal attention mechanisms:

$$\mathbf{z}_{i,m,j} = \text{DecoderLayer}(\mathbf{h}_{i,m,j}^{(mask)}, \{\mathbf{h}_{i,m',j'}^{(L)}\}_{m',j'})$$

where $\mathbf{h}_{i,m',j'}^{(L)}$ are the outputs from the final layer of the encoder for all unmasked observations.

The decoder output is then passed through modality-specific projection layers to predict the original values:

$$\hat{\mathbf{x}}_{i,m,j} = \mathbf{W}_m^{dec} \mathbf{z}_{i,m,j} + \mathbf{b}_m^{dec}$$

### 2.7 Training Objective

The model is trained to minimize the reconstruction loss for the masked values:

$$\mathcal{L}_{rec} = \sum_{i=1}^{N} \sum_{m=1}^{M} \sum_{j \in \mathcal{M}_{i,m}} \mathcal{L}_m(\mathbf{x}_{i,m,j}, \hat{\mathbf{x}}_{i,m,j})$$

where $\mathcal{M}_{i,m}$ is the set of masked indices for patient $i$ and modality $m$, and $\mathcal{L}_m$ is a modality-specific loss function (e.g., MSE for continuous variables, cross-entropy for categorical variables).

To encourage the model to learn uncertainty estimates, we additionally predict the variance of the reconstruction for continuous variables:

$$\hat{\sigma}_{i,m,j}^2 = \text{softplus}(\mathbf{W}_m^{var} \mathbf{z}_{i,m,j} + \mathbf{b}_m^{var})$$

The loss function for continuous variables becomes:

$$\mathcal{L}_m(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{2}\left(\frac{(\mathbf{x} - \hat{\mathbf{x}})^2}{\hat{\sigma}^2} + \log(\hat{\sigma}^2)\right)$$

### 2.8 Fine-tuning for Downstream Tasks

After pretraining, the encoder can be fine-tuned for various downstream tasks with minimal modifications. For classification or regression tasks, we add a task-specific head that processes the encoded representations:

$$\mathbf{y}_i = \text{TaskHead}(\{\mathbf{h}_{i,m,j}^{(L)}\}_{m,j})$$

The task head can be designed based on the specific requirements of the downstream task, such as pooling operations followed by MLP layers for classification, or autoregressive components for forecasting.

### 2.9 Experimental Design

#### 2.9.1 Datasets

We will evaluate CT-MAE on three diverse healthcare datasets:

1. **MIMIC-IV**: A large-scale EHR dataset containing vital signs, laboratory measurements, medications, and clinical notes from intensive care units.

2. **PhysioNet Challenge 2019**: A multi-modal dataset for early prediction of sepsis, including vital signs and laboratory measurements collected at irregular intervals.

3. **PPMI Wearable Dataset**: A dataset of Parkinson's disease patients with wearable sensor data (accelerometer, gyroscope) and clinical assessments.

#### 2.9.2 Pretraining Setup

For pretraining, we will use all available time series data without requiring labels. We will experiment with different masking probabilities ($p_t$, $p_f$, $p_m$) and masking strategies to determine the optimal configuration. The model will be trained using the Adam optimizer with learning rate warm-up and decay.

#### 2.9.3 Downstream Tasks

We will evaluate the fine-tuned CT-MAE on the following tasks:

1. **Disease prediction**: Early detection of clinical deterioration, sepsis prediction, and mortality prediction.

2. **Anomaly detection**: Identification of abnormal patterns in ECG signals and vital signs.

3. **Treatment recommendation**: Suggestion of appropriate interventions based on patient trajectories.

4. **Phenotype discovery**: Unsupervised identification of patient subgroups with similar characteristics.

#### 2.9.4 Baseline Methods

We will compare CT-MAE against the following baselines:

1. Traditional time series models (ARIMA, Prophet)
2. Deep learning models for regular time series (RNN, LSTM, GRU)
3. Specialized models for irregular time series (GRU-D, IP-Nets)
4. Self-supervised approaches (Temporal Neighborhood Coding, bioFAME)
5. Multi-modal fusion methods (Multi-modal Attention Networks)

#### 2.9.5 Evaluation Metrics

For each downstream task, we will use task-specific evaluation metrics:

1. **Classification tasks**: AUROC, AUPRC, F1-score, sensitivity, specificity
2. **Regression tasks**: RMSE, MAE, R²
3. **Forecasting tasks**: Horizon-specific RMSE, calibration metrics
4. **Imputation tasks**: Imputation error (RMSE, MAE)

Additionally, we will evaluate:

1. **Calibration**: Expected Calibration Error (ECE) to assess the reliability of uncertainty estimates
2. **Robustness**: Performance under varying levels of missingness and noise
3. **Interpretability**: Qualitative assessment of attention maps
4. **Computational efficiency**: Training time, inference time, and memory requirements

## 3. Expected Outcomes & Impact

### 3.1 Expected Technical Contributions

1. **Novel continuous-time representation learning framework**: CT-MAE will provide a principled approach to learning from irregular, multi-modal health time series without requiring imputation or resampling, addressing a fundamental limitation of existing methods.

2. **Adaptive temporal kernel learning**: The learnable temporal basis functions will capture complex temporal dependencies at multiple scales, potentially revealing novel insights about the progression of health conditions.

3. **Uncertainty-aware multi-modal fusion**: By jointly modeling multiple data sources and providing calibrated uncertainty estimates, CT-MAE will enable more reliable clinical predictions, especially in scenarios with missing data.

4. **Foundation model for healthcare time series**: The pretrained CT-MAE will serve as a versatile foundation for various downstream tasks, reducing the need for large labeled datasets and specialized model architectures for each task.

### 3.2 Expected Clinical Impact

1. **Improved early warning systems**: The ability to process and interpret irregular, multi-modal data in real-time could enhance early warning systems for clinical deterioration, potentially saving lives through timely interventions.

2. **Personalized treatment recommendations**: By capturing patient-specific temporal patterns across modalities, CT-MAE could inform more personalized treatment decisions that account for individual trajectories and responses.

3. **Continuous remote monitoring**: The model's robustness to missing data and irregular sampling makes it well-suited for remote patient monitoring applications, where data collection may be inconsistent.

4. **Enhanced clinical decision support**: The interpretable attention mechanisms and uncertainty estimates will provide clinicians with transparent insights into the model's predictions, facilitating more informed decision-making.

### 3.3 Expected Research Directions

The development of CT-MAE will open up several promising research directions:

1. **Causal inference in irregular time series**: Extending the model to capture causal relationships between variables over time could lead to more actionable insights about treatment effects.

2. **Hierarchical temporal modeling**: Incorporating hierarchical structures in the temporal representations could better capture both short-term fluctuations and long-term trends in health data.

3. **Federated learning across institutions**: The foundation model approach could facilitate privacy-preserving collaborative learning across healthcare institutions without sharing raw patient data.

4. **Integration with clinical workflows**: Research on effective ways to deploy the model in clinical settings and present its outputs to healthcare providers would maximize real-world impact.

### 3.4 Broader Impact

Beyond the immediate applications in healthcare, the methodological advances in CT-MAE could benefit other domains with irregular, multi-modal time series data, such as environmental monitoring, industrial systems, and financial forecasting. The focus on interpretability and uncertainty quantification also aligns with the growing emphasis on responsible AI deployment in high-stakes settings.

By addressing the fundamental challenges of healthcare time series analysis, CT-MAE has the potential to bridge the gap between theoretical advances in machine learning and practical applications in clinical care, ultimately contributing to more effective, equitable, and patient-centered healthcare systems.