# Adaptive Continuous-Time Masked Autoencoder for Multi-Modal Health Signals

## 1. Introduction

### Background  
Time series data in healthcare—ranging from Electronic Health Records (EHR) to wearable sensors—encode critical insights into patient health but are notoriously challenging to model. Key obstacles include **irregular sampling intervals**, **missing data**, and **cross-modal misalignment** (e.g., asynchronous ECG, lab results, and accelerometry). For instance, EHR labs may be collected days apart, while wearables report heart rate every minute. Traditional models like LSTMs or conventional Transformers struggle with these irregularities, relying on imputation or discretization that distort temporal dynamics. This gap limits the deployment of AI systems for critical tasks like sepsis forecasting, where timely intervention relies on synthesizing heterogeneous signals.

Recent advances in masked autoencoders (MAE) and continuous-time modeling offer new hope. Papers like **bioFAME** (Liu et al., 2023) and **Time-Series Transformers** (Qian et al., 2021) demonstrate progress in frequency-aware multimodal pretraining and irregular data handling. However, no existing framework unifies three critical features: (1) joint handling of **missing values and irregular timestamps** without imputation, (2) **cross-modal reconstruction** via temporal attention, and (3) **scalable self-supervision** for multimodal health signals.

---

### Research Objectives  
We propose **Adaptive Continuous-Time Masked Autoencoder (CT-MAE)**, a foundation model for health time series with the following goals:  
1. **Robustness to Missingness**: Mask both values *and timestamps* during self-supervised pretraining to explicitly model uncertainty caused by irregular sampling (e.g., missing EHR labs).  
2. **Multimodal Fusion**: Use cross-modal attention in the decoder to leverage complementary signals (e.g., ECG and wearables) for reconstructing missing segments.  
3. **Temporal Continuity**: Encode timestamps via learnable temporal kernels (e.g., Gaussian process basis functions) to capture variable intervals natively.  
4. **Clinical Deployability**: Evaluate task-specific fine-tuning for downstream tasks (e.g., arrhythmia detection, sepsis forecasting) with emphasis on explainability (via attention maps) and uncertainty quantification.  

---

### Significance  
CT-MAE addresses key challenges highlighted in the literature (Zhou et al., 2025; Morrill et al., 2020):  
- **Handling Irregular Data**: Eliminates reliance on imputation, a critical barrier to real-world deployment where missingness patterns vary across patient populations.  
- **Interpretability**: Attention mechanisms reveal which modalities (e.g., ECG vs. oxygen saturation) most strongly influence predictions, aligning with clinical workflows.  
- **Scalability**: Lightweight decoders and masking strategies reduce computational complexity, enabling near-real-time analysis.  

Successful development could redefine how clinicians use AI: For example, a CT-MAE model trained on wearable data might trigger alerts for subtle arrhythmias by integrating intermittent ECG snippets with continuous heart-rate variability trends.

---

## 2. Methodology

### 2.1 Data Collection & Preprocessing  
We will pretrain CT-MAE on two large, multi-modal datasets:  
1. **MIMIC-III** (EHR + ICU waveforms) for clinical tasks like sepsis forecasting.  
2. **Sleep Heart Health Study (SHHS)** for wearable+ECG analysis.  

**Processing Steps**:  
- Standardize modalities (e.g., z-score for vitals, raw ECG amplitudes).  
- Introduce synthetic **temporal misalignments** (e.g., randomly drop 30% of EHR entries) to simulate real-world missingness.  

---

### 2.2 Model Architecture  
CT-MAE extends the MAE paradigm (He et al., 2021) with continuous-time components:

#### **Temporal Kernel Encoding**  
Each timestamp $t_i$ is embedded via temporal feature maps:  
$$
\phi(t_i) = \big[\cos(\omega_1 t_i), \sin(\omega_1 t_i), \dots, \cos(\omega_d t_i), \sin(\omega_d t_i)\big],
$$
where $\omega_k$ are learnable frequency bases inspired by Gaussian processes (Morrill et al., 2020). This allows continuous-time extrapolation without imputation.

#### **Masking Strategy**  
- **Continuous Masking**: For a given modality, mask intervals $[t_i, t_j]$ with duration sampled from a log-normal distribution.  
- **Multi-Modal Masking**: Apply independent masks per modality to enforce cross-modal dependency learning.  

#### **Encoder**  
A **continuous-time Transformer** computes attention weights as:  
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} \odot \exp\left(-\lambda \Delta T\right)\right)V,
$$  
where $\Delta T_{ij} = |t_i - t_j|$ is the time gap between events, and $\lambda$ modulates temporal decay. Missing timestamps are treated as a latent variables integrated over during training.

#### **Decoder**  
A cross-modal attention module reconstructs masked tokens:  
$$
\hat{x}_i = \text{DecAttn}(E, M; \Theta),
$$  
where $E$ is the encoder’s output, $M$ is the mask matrix, and $\text{DecAttn}$ learns to combine multimodal context (e.g., using wearable trends to reconstruct a missing ECG segment).

---

### 2.3 Training Protocol  
- **Pretext Task**: Reconstruct masked values with a loss:  
  $$
  \mathcal{L} = \sum_{i=1}^N \|\hat{x}_i - x_i\|^2 + \beta \cdot \|\hat{t}_i - t_i\|^2,
  $$  
  where $\beta$ balances timestamp recovery.  
- **Optimization**: AdamW optimizer, cosine LR decay, masking ratio = 40\% initially (increasing to 60\% during curriculum learning).  

---

### 2.4 Evaluation Metrics  
| **Task**              | **Metric**                | **Baseline Models**                |  
|-----------------------|---------------------------|-------------------------------------|
| Arrhythmia Detection  | AUC-ROC, F1 Score         | ResNet, BioFAME (Liu et al., 2023)  |  
| Sepsis Forecasting    | Brier Score, AUROC        | T-LSTM, Time-Series Transformer     |  
| Imputation Accuracy   | MAE, RMSE                 | M3GP (Yoo et al., 2020), GNN-IM     |  

**Ablation Studies**:  
- Effect of temporal kernel variants (Fourier vs. RBF bases).  
- Comparison of masking strategies vs. traditional imputation (e.g., linear/polynomial).  

---

### 2.5 Experimental Design  
1. **Pretraining**: Conduct on 10,000 patients from MIMIC-III + SHHS.  
2. **Fine-Tuning**: Transfer to downstream tasks:  
   - **Task 1**: 12-hour sepsis prediction in ICU (MIMIC-III).  
   - **Task 2**: Atrial fibrillation detection (SHHS ECG).  
3. **Robustness Testing**: Evaluate AUC degradation when 50% of EHR entries are masked.  

---

## 3. Expected Outcomes & Impact  

### 3.1 Technical Advances  
We anticipate three novel contributions:  
1. **First Continuous-Time MAE for Health**: CT-MAE will outperform discrete-time models in handling missingness (e.g., ≤ 0.05 MAE error on EHR imputation, vs. ≥ 0.08 for GNN baselines).  
2. **Cross-Modal Reconstruction**: ECG masked segments reconstructed using watch-derived heart rate trends will show ≥ 0.98 correlation in sinus rhythm.  
3. **Uncertainty Quantification**: For irregular ECG data, CT-MAE’s uncertainty metrics (e.g., entropy of reconstructions) will correlate with diagnostic ambiguity ($r > 0.75$ vs. clinician labels).  

---

### 3.2 Clinical & Societal Impact  
1. **Scalable Pretraining**: A single CT-MAE foundation model pretrained on 100K patients could outperform 10 task-specific models (e.g., sepsis, COPD, arrhythmia) with 50% fewer parameters.  
2. **Wearable-Driven Early Warning Systems**: CT-MEA could predict hypoglycemia in diabetic patients by fusing intermittent CGM readings, accelerometer data, and sleep stages.  
3. **Explainability for AI Adoption**: Attention maps highlighting critical time points (e.g., an oxygen desaturation dip 24 hours before sepsis onset) would align with clinician intuition, fostering trust.  

---

### 3.3 Long-Term Vision  
CT-MAE could become a unifying framework for health time series:  
- **Deployment-Ready**: Lightweight decoder variants would enable edge computing on wearables.  
- **Generalization**: Transfer to low-resource settings by fine-tuning on smaller datasets (e.g., neonatal EHR).  
- **Public Release**: Open-source code and pretrained weights for EHR and wearables would accelerate research in resource-limited institutions.  

By explicitly modeling irregularity and multimodal interplay, CT-MAE aims to bring health time series AI closer to the clinical frontline.