Title  
Adaptive Continuous-Time Masked Autoencoder for Multi-Modal Health Signals  

Introduction  
Background. Health‐related time series—from Electronic Health Record (EHR) laboratory measurements and vital‐sign streams to continuous wearable sensor recordings—offer unprecedented insight into patient state and disease progression. However, these data pose unique challenges: sampling is irregular, missingness is pervasive and non‐random, modalities (e.g. EHR vs. ECG vs. accelerometry) are often misaligned, label acquisition is expensive and noisy, and real‐world deployment demands robust uncertainty estimates and interpretability. Traditional sequence models (RNNs, 1D‐CNNs) require imputation or regularization heuristics that distort temporal information; recent Transformer‐based approaches (e.g. Time‐Series Transformer for Irregularly Sampled Data¹) improve modeling of variable intervals but struggle with multi‐modal fusion and large‐scale self‐supervision. Masked Autoencoders (MAE)² have revolutionized vision and language by enabling efficient self‐supervised pretraining via random masking and reconstruction, yet existing MAE variants for biosignals (e.g. bioFAME³, C-MELT⁴, MMAE-ECG⁵) do not address irregular sampling natively or extend to multiple modalities.

Objectives. We propose CT-MAE, a Continuous-Time Masked Autoencoder tailored to multi-modal health time series. The primary research questions are:  
1. Can learnable continuous‐time embeddings based on temporal kernels capture irregular intervals more faithfully than fixed‐grid positional encodings?  
2. Does random masking of both values and timestamps across modalities enable robust foundation representations that transfer to clinical downstream tasks (e.g. sepsis forecasting, arrhythmia detection) under high missingness?  
3. Can cross‐modal attention in the decoder yield interpretable alignments and calibrated uncertainty estimates critical for clinical decision support?

Significance. CT-MAE aims to deliver a self-supervised foundation model that (1) natively handles irregular sampling without ad‐hoc imputation, (2) fuses complementary signals across EHR, ECG, and wearable channels, and (3) produces calibrated predictions with interpretable attention maps. Success will advance the deployment of reliable, generalizable health AI systems and enable data‐efficient adaptation to diverse clinical settings.

Methodology  
Overview. Our approach comprises: (A) data collection and preprocessing; (B) model architecture (encoder, masking strategy, decoder); (C) self-supervised pretraining objectives; (D) fine-tuning on downstream tasks; (E) experimental design and evaluation.

A. Data Collection and Preprocessing  
• Datasets:  
  – MIMIC-III⁶: ICU EHR time series (labs, vitals).  
  – PhysioNet Challenge ECG (e.g. arrhythmia detection¹⁰).  
  – Wearable sensor cohort (e.g. accelerometer from the Cleveland HeartLab study).  
  – Multi-site extension using eICU database for cross-institution generalization.  

• Preprocessing Steps:  
  1. Timestamp alignment: convert all sources to UTC and rescale to hours since admission or study start.  
  2. Normalization: per‐channel z-score normalization using training‐set statistics.  
  3. Channel selection: select a harmonized set of 20 EHR labs/vitals, 1-lead ECG waveform, 3-axis accelerometry.  
  4. Segmentation: split each patient record into overlapping windows of length $T_\text{max}=48$ h with maximum sampling rate 1 Hz for ECG, 0.1 Hz for EHR, 10 Hz for accelerometry.  
  5. Missingness indicator matrix $M\in\{0,1\}^{T_\text{max}\times C}$ (channels $C$) to preserve mask information.  

B. Model Architecture  
1. Continuous-Time Encoder  
   We embed each observation $(t_i, x_i)\in\mathbb{R}\times\mathbb{R}^C$ using a learnable temporal‐kernel feature map. Let $\Delta t_i = t_i - t_{i-1}$. Define $K$ Gaussian‐process basis functions:  
   $$\phi(\Delta t_i) = \bigl[\exp\bigl(-\tfrac{(\Delta t_i - \mu_k)^2}{2\sigma_k^2}\bigr)\bigr]_{k=1}^K\in\mathbb{R}^K,$$  
   where $\{\mu_k,\sigma_k\}$ are learnable. The input embedding at position $i$ is:  
   $$e_i = W_x x_i + W_\phi \phi(\Delta t_i) + b_e,$$  
   with $W_x\in\mathbb{R}^{d\times C}$, $W_\phi\in\mathbb{R}^{d\times K}$. A continuous‐time Transformer encoder with $L$ layers applies masked multi‐head self‐attention and feed-forward networks exactly on the irregularly spaced sequence $\{e_i\}$.  

2. Masking Strategy  
   We define a per-modality mask for both values and timestamps:  
   – Mask fraction $\rho\in[0,1]$ controls the number of tokens masked.  
   – Two masking types: value‐mask (replace $x_i$ with a learnable token) and time‐mask (drop $\phi(\Delta t_i)$).  
   Algorithmic steps:  
   ```
   for each window:
     sample mask indices I ⊂ {1,…,N} of size ⌊ρN⌋
     for each i in I:
       with prob. p_val: x_i ← [MASK]_val
       with prob. p_time: φ(Δt_i) ← [MASK]_time
   ```  
   We set $p_{val}=0.8,\;p_{time}=0.2$ by default.

3. Cross-Modal Decoder  
   The decoder reconstructs masked segments using cross‐modal attention between encoder outputs of different channels. Let $h_i^{(l)}\in\mathbb{R}^d$ denote the encoder representation at layer $l$. The decoder layer performs:  
   $$\tilde{h}_i^{(l+1)} = \mathrm{FFN}\!\Bigl(\mathrm{CMHA}(h_i^{(l)},\,\{h_j^{(l)}\}_{j\neq i})\Bigr),$$  
   where CMHA is a multi-head attention that attends across all channel embeddings for the same timestamp window. Finally, a linear head $W_{\text{out}}$ projects $\tilde{h}_i^{(L)}$ back to the original channel space.

C. Pretraining Objectives  
1. Reconstruction Loss  
   We minimize the mean squared error between reconstructed $\hat{x}_i$ and true $x_i$ on masked positions:  
   $$\mathcal{L}_{\mathrm{rec}} = \frac{1}{|\mathcal{I}|}\sum_{i\in \mathcal{I}}\|x_i - \hat{x}_i\|_2^2.$$  

2. Time Consistency Loss  
   To encourage consistency in temporal embeddings, we add:  
   $$\mathcal{L}_{\mathrm{time}} = \frac{1}{N} \sum_{i=2}^N \|\phi(\Delta t_i) - \phi(\widehat{\Delta t}_i)\|_2^2,$$  
   where $\widehat{\Delta t}_i$ is the reconstructed interval decoder output.  

3. Total Loss  
   $$\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{time}}\,\mathcal{L}_{\mathrm{time}},$$  
   with $\lambda_{\mathrm{time}}=0.1$.

D. Fine-Tuning for Downstream Tasks  
1. Task Heads  
   – Sepsis forecasting: a binary classification head over 6-h or 12-h horizons. Loss is cross-entropy $\mathcal{L}_{\mathrm{CE}}$.  
   – Arrhythmia detection: multi-class classification head with softmax, cross-entropy loss.  

2. Adaptation Strategy  
   We freeze encoder and decoder layers and insert lightweight adapter modules (two‐layer MLP) at each Transformer layer. Only adapter weights and task head are updated during fine-tuning, enabling parameter‐efficient transfer.

E. Experimental Design and Evaluation  
1. Baselines  
   – GRU-D⁷ (stateful RNN with decay).  
   – Time-Series Transformer⁶ with imputation.  
   – bioFAME³ and MMAE-ECG⁵ (single‐modality MAEs).  
   – Multimodal Vision-Language MAE⁴ adapted to time series (M³AE-TS).  

2. Data Splits and Protocol  
   – Within‐site evaluation: 70/10/20 train/val/test split on each dataset.  
   – Cross‐site generalization: train on MIMIC-III, test on eICU.  
   – Missingness stress tests: induce additional random missing rates 10%, 30%, 50%.  

3. Metrics  
   – Classification: AUROC, AUPRC, accuracy, F1‐score.  
   – Forecasting: RMSE, MAE, Pearson correlation.  
   – Calibration: Expected Calibration Error (ECE).  
   – Computational: parameter count, FLOPs, latency (ms per window).  

4. Ablation Studies  
   – No temporal kernel (replace $\phi(\Delta t)$ with linear embedding).  
   – No cross‐modal attention in decoder (separate per‐modality decoders).  
   – Vary mask ratio $\rho\in\{0.1,0.3,0.5,0.7\}$.  
   – Adapter vs. full fine-tuning.  

5. Statistical Analysis  
   Each experiment is repeated over 5 random seeds. We report mean±std and perform paired t‐tests to assess significance ($p<0.05$).

Expected Outcomes & Impact  
We hypothesize that CT-MAE will:  
1. Achieve superior reconstruction under missingness: reduce $\mathcal{L}_{\mathrm{rec}}$ by 20–30% relative to baselines at 50% mask rates.  
2. Improve downstream classification: increase AUROC by 5–10% on sepsis and arrhythmia tasks compared to Time-Series Transformer and GRU-D under both within-site and cross-site settings.  
3. Yield well‐calibrated predictions: lower ECE by 10–15% through native uncertainty estimation via multi‐head attention distributions.  
4. Demonstrate interpretability: attention maps highlighting clinically relevant features (e.g. rising lactate levels, R-peak irregularities) will align with expert annotations.  
5. Maintain computational efficiency: runtime comparable to standard Transformers, with ~10% overhead for temporal kernels.

Impact on Healthcare and Machine Learning. By delivering a foundation model that natively addresses irregular sampling, multi-modal fusion, and missing data, CT-MAE will:  
• Accelerate development of reliable predictive tools for early warning (e.g. sepsis onset), reducing patient morbidity and mortality.  
• Provide a transferable backbone for diverse tasks—emotion recognition, fall risk prediction, dynamic treatment recommendation—through light-weight fine-tuning.  
• Advance research on self-supervised learning for irregular time series, offering open-source code, pretrained checkpoints, and new masking/attention strategies.  
• Inform best practices for clinical deployment by demonstrating model calibration, interpretability, and cross-institution generalization.

In sum, CT-MAE promises to bridge the gap between cutting-edge time‐series models and the stringent demands of healthcare, fostering safer, more transparent, and more effective AI-driven patient care.