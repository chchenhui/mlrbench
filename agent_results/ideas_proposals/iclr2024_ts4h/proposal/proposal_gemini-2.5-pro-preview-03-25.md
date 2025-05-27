Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** Adaptive Continuous-Time Masked Autoencoder for Robust Multi-Modal Health Signal Representation Learning

**2. Introduction**

**2.1. Background**
Time series data are fundamental to modern healthcare, capturing dynamic physiological and behavioral processes through sources like Electronic Health Records (EHR), electrocardiograms (ECG), electroencephalograms (EEG), wearable sensors (e.g., accelerometers, photoplethysmography), and audio recordings. These data streams hold immense potential for advancing healthcare through early disease detection, personalized treatment recommendations, patient monitoring, and understanding health behaviors [Task Description]. However, realizing this potential is significantly hindered by the inherent complexities of health time series. Unlike data from domains like finance or engineering, clinical time series are notoriously challenging: measurements are often sampled irregularly, leading to variable time intervals between observations; missing values are pervasive due to sensor drop-off, intermittent monitoring, or data entry practices; data are frequently multi-modal, requiring the integration of heterogeneous signals (e.g., sparse lab values from EHR, dense waveforms from ECG, sporadic sensor readings from wearables); labels for supervised tasks can be scarce, noisy, or delayed; and data distributions can shift over time due to changes in patient state, treatment protocols, or sensor technology [Task Description, Lit Review Challenge 1, 2]. Furthermore, deploying machine learning models in high-stakes clinical settings necessitates robustness, interpretability, and reliable uncertainty quantification [Task Description, Lit Review Challenge 3, 5].

Existing machine learning approaches often struggle to adequately address these cumulative challenges. Traditional recurrent neural networks (RNNs) can have difficulties with long-range dependencies and irregular sampling. While Transformers have shown promise for sequence modeling, standard architectures typically assume regularly sampled data and require imputation or ad-hoc modifications to handle missingness and irregularity, potentially introducing bias or discarding valuable temporal information [6]. Masked Autoencoders (MAEs) have emerged as a powerful self-supervised learning paradigm, particularly in vision [5] and more recently adapted for biosignals [1, 2] and multi-modal medical data [3, 4]. These models learn rich representations by reconstructing randomly masked portions of the input. However, existing MAE variants often operate on tokenized sequences assuming fixed time steps or require modality-specific preprocessing (e.g., heartbeat segmentation [1]), limiting their direct applicability to raw, irregularly sampled, multi-modal health time series. Models like the Time-Series Transformer [6] handle irregularity but may not be inherently designed for multi-modal fusion or the self-supervised MAE framework. There remains a critical need for a foundational model that can natively handle the temporal irregularities and multi-modal nature of health data within a robust self-supervised learning framework.

**2.2. Research Objectives**
This research aims to develop and evaluate a novel self-supervised foundation model, the **Continuous-Time Masked Autoencoder (CT-MAE)**, specifically designed to address the unique challenges of multi-modal, irregularly sampled health time series. Our primary objectives are:

1.  **Develop the CT-MAE Architecture:** Design and implement a novel MAE architecture that incorporates continuous-time modeling principles to handle irregular sampling intervals natively, without requiring explicit imputation. This involves integrating learnable temporal kernels into the input encoding stage.
2.  **Enable Multi-Modal Fusion:** Extend the CT-MAE framework to jointly process and reconstruct signals from diverse health data modalities (e.g., EHR, ECG, wearables), leveraging cross-modal attention mechanisms in the decoder to capture complementary information.
3.  **Implement Advanced Masking Strategy:** Design and evaluate a masking strategy that randomly masks not only the observation values but also their associated timestamps across different modalities, forcing the model to learn robust temporal and cross-modal dependencies.
4.  **Pre-train a Foundational Health Time Series Model:** Pre-train the CT-MAE model on large-scale, multi-modal health datasets (e.g., combining EHR and waveform data from sources like MIMIC-IV and PhysioNet) in a self-supervised manner to learn generalizable representations.
5.  **Evaluate Downstream Task Performance:** Evaluate the effectiveness of the pre-trained CT-MAE representations by fine-tuning the model on various downstream clinical tasks, such as sepsis forecasting from EHR data and arrhythmia detection from ECG signals. Compare performance against state-of-the-art baselines, particularly under conditions of high data irregularity and missingness.
6.  **Assess Robustness, Interpretability, and Uncertainty:** Quantify the model's robustness to missing data and sampling rate variations. Investigate the model's interpretability using attention maps and assess the calibration of its predictive uncertainty.

**2.3. Significance**
This research directly addresses the critical gap between current time series modeling capabilities and the requirements for practical deployment in healthcare [Task Description]. By developing CT-MAE, we aim to make the following significant contributions:

*   **Methodological Innovation:** Proposes a novel architecture combining continuous-time modeling with the power of masked autoencoding for self-supervised learning on complex health data. This approach natively handles irregularity and multi-modality, overcoming limitations of existing methods.
*   **Foundation Model for Health Time Series:** Contributes to the growing field of foundation models in healthcare [Task Description Theme] by providing a pre-trained model capable of adapting to various downstream tasks with lightweight fine-tuning, potentially democratizing access to powerful time series analysis tools.
*   **Improved Robustness and Reliability:** Addresses key challenges like irregular sampling and missing data [Lit Review Challenge 1], aiming for models that are more robust and perform reliably in real-world clinical environments. The focus on uncertainty estimation enhances trustworthiness.
*   **Enhanced Clinical Decision Support:** By improving the accuracy of tasks like early disease prediction (e.g., sepsis) and event detection (e.g., arrhythmia), this work has the potential to lead to more timely interventions and better patient outcomes.
*   **Advancing Multi-Modal Learning:** Provides a principled framework for fusing information from disparate health data sources, unlocking synergistic insights that may not be apparent from individual modalities.

This research aligns perfectly with the workshop themes of **Foundation Models** and addressing challenges in **Behavioral Health** (as wearable data often reflects behavior) and general health time series, tackling topics like representation learning, novel architectures, classification/forecasting, handling missing/irregular data, multi-modal models, and interpretability [Task Description Topics].

**3. Methodology**

**3.1. Data Representation and Preprocessing**
We consider multi-modal health time series data. For a given patient, the data consists of multiple sequences, one for each modality $m \in \{1, ..., M\}$. Each modality sequence is represented as a set of observations $S_m = \{ (t_{m,i}, x_{m,i}) \}_{i=1}^{N_m}$, where $t_{m,i} \in \mathbb{R}^+$ is the timestamp of the $i$-th observation for modality $m$, and $x_{m,i} \in \mathbb{R}^{d_m}$ is the corresponding value (e.g., a scalar lab value, a vector of ECG lead measurements, accelerometer readings). Note that the timestamps $t_{m,i}$ are irregular, and the number of observations $N_m$ can vary significantly across modalities and patients.

Preprocessing will involve:
*   Normalization: Standardizing continuous features (e.g., lab values, sensor readings) per feature, potentially using statistics derived from the training set. Categorical features (e.g., EHR codes) will be embedded.
*   Time Alignment: While the model handles irregular timestamps, a common time reference/origin will be established for each patient record (e.g., time of admission). Timestamps $t_{m,i}$ will represent time elapsed since this origin.
*   Data Sources: We plan to utilize publicly available datasets such as MIMIC-IV (containing EHR data, including labs, vitals, and potentially linked waveforms) and PhysioNet databases (e.g., PhysioNet/CinC Challenge 2012, 2017, 2020 for ECG, vital signs data). Combining these sources will allow for large-scale multi-modal pre-training.

**3.2. CT-MAE Architecture**
The proposed CT-MAE follows the encoder-decoder paradigm inspired by [5], but adapted for continuous-time, multi-modal data.

**3.2.1. Continuous-Time Input Embedding**
For each observation $(t_{m,i}, x_{m,i})$, we need to encode both the value and its precise timing. We propose using learnable temporal basis functions to embed the timestamp $t_{m,i}$. Similar to ideas in Neural Processes or Gaussian Process kernels [7], we can represent the time dimension using a set of basis functions $\phi_k: \mathbb{R}^+ \to \mathbb{R}$. The temporal embedding for timestamp $t_{m,i}$ can be computed as $e_{time}(t_{m,i}) = [\phi_1(t_{m,i}), ..., \phi_K(t_{m,i})] \in \mathbb{R}^K$. Examples of basis functions include Gaussian Radial Basis Functions (RBFs) centered at learnable locations, or spline bases. The value $x_{m,i}$ is embedded using a modality-specific linear layer or MLP, $e_{value}(x_{m,i}) \in \mathbb{R}^{D_{emb}}$. The final input embedding for an observation is the combination (e.g., concatenation or addition) of its value and temporal embeddings, potentially with a modality embedding added:
$$ z_{m,i} = \text{Combine}(e_{value}(x_{m,i}), e_{time}(t_{m,i})) + e_{modality}(m) \in \mathbb{R}^{D_{model}} $$

**3.2.2. Masking Strategy**
Following the MAE principle [5], we randomly mask a significant portion (e.g., 50-75%) of the input observations *before* feeding them to the encoder. Crucially, our masking strategy operates on the $(t, x)$ pairs. For a chosen observation $(t_{m,i}, x_{m,i})$, we can either mask the value $x_{m,i}$, the timestamp $t_{m,i}$, or both. We hypothesize that masking timestamps, in addition to values, will force the model to better learn the underlying temporal dynamics and relationships between irregularly spaced points. Masking is applied independently across different modalities but within the same patient record. Let $\mathcal{V}$ denote the set of visible (unmasked) observation indices, and $\mathcal{H}$ the set of hidden (masked) indices across all modalities for a patient.

**3.2.3. Continuous-Time Transformer Encoder**
The encoder processes only the visible embeddings $\{ z_{m,i} | (m,i) \in \mathcal{V} \}$. We adapt the Time-Series Transformer architecture [6] or similar continuous-time attention mechanisms. These models typically modify the standard self-attention mechanism to account for the time differences between observations. For instance, the attention score between observation $i$ (query) and observation $j$ (key) can be modulated by their time difference $\Delta t = t_i - t_j$:
$$ \text{Attention}(Q_i, K_j, V_j) = \text{softmax}\left( \frac{ (W_Q z_i)^T (W_K z_j) + f(\Delta t) }{\sqrt{d_k}} \right) (W_V z_j) $$
where $W_Q, W_K, W_V$ are projection matrices, $d_k$ is the key dimension, and $f(\Delta t)$ is a function capturing the temporal relationship (e.g., learned or fixed function of time difference). The encoder consists of multiple layers of such continuous-time self-attention and feed-forward networks, producing context-aware representations $h_{m,i}$ for the visible tokens.

**3.2.4. Cross-Modal Decoder**
The decoder's task is to reconstruct the masked observations (both values $x_{m,j}$ and timestamps $t_{m,j}$ for $(m,j) \in \mathcal{H}$) using the encoded representations of the visible tokens $\{h_{m,i} | (m,i) \in \mathcal{V}\}$ and special mask tokens representing the positions of masked observations. The decoder employs cross-attention mechanisms where the mask tokens act as queries, attending to the encoded sequence of visible tokens $\{h\}$. Crucially, this attention operates across all modalities present in the visible set, allowing information from one modality (e.g., ECG) to aid in the reconstruction of another (e.g., EHR vitals).
Let $M_{m,j}$ be the mask token corresponding to the masked observation $(t_{m,j}, x_{m,j})$. The decoder predicts the original value $\hat{x}_{m,j}$ and potentially the timestamp $\hat{t}_{m,j}$ (or time difference from a reference point) based on the mask token's representation after passing through decoder layers incorporating cross-attention:
$$ \hat{x}_{m,j}, \hat{t}_{m,j} = \text{DecoderHead}(\text{Decoder}(M_{m,j}, \{h_{m',i'}\}_{(m',i') \in \mathcal{V}})) $$
The decoder is typically lighter weight than the encoder [5].

**3.2.5. Loss Function**
The model is trained by minimizing the reconstruction error on the masked observations. The loss function is defined over the masked set $\mathcal{H}$. For continuous values (e.g., lab results, sensor readings), Mean Squared Error (MSE) is appropriate. For categorical values (e.g., diagnosis codes), Cross-Entropy loss can be used. If reconstructing timestamps, an MSE or L1 loss on the predicted time values or time differences could be applied. The total loss is a weighted sum over the masked elements and modalities:
$$ \mathcal{L}_{recon} = \sum_{(m,j) \in \mathcal{H}} [ \lambda_{val} \mathcal{L}_{value}(x_{m,j}, \hat{x}_{m,j}) + \lambda_{time} \mathcal{L}_{time}(t_{m,j}, \hat{t}_{m,j}) ] $$
where $\lambda_{val}$ and $\lambda_{time}$ are hyperparameters balancing the reconstruction priorities.

**3.3. Experimental Design**

**3.3.1. Pre-training Phase**
*   **Dataset:** Utilize large-scale multi-modal datasets like MIMIC-IV (potentially linking ICU stays with waveform data like MIMIC-III Waveform Database) or a combination of MIMIC-IV for EHR and PhysioNet databases (e.g., ECG-Cardiology database, Icentia 11-lead Arrhythmia database) for waveform data. Construct patient records containing time series from available modalities.
*   **Procedure:** Train the CT-MAE model using the self-supervised reconstruction objective described above on the combined dataset. Explore different masking ratios (e.g., 50%, 75%) and strategies (value masking vs. value+timestamp masking).

**3.3.2. Fine-tuning and Evaluation Phase**
*   **Downstream Tasks:**
    *   *Sepsis Forecasting:* Binary classification task using EHR data (e.g., from MIMIC-IV). Predict onset of sepsis within a future time window based on patient history up to a certain point.
    *   *Arrhythmia Detection:* Multi-class classification task using ECG data (e.g., from PhysioNet/CinC Challenge 2017 or 2020). Classify short ECG segments based on the presence and type of arrhythmia.
    *   *Patient Outcome Prediction:* E.g., predicting in-hospital mortality or length-of-stay based on multi-modal data from the first 24/48 hours of an ICU stay.
*   **Fine-tuning Strategy:** Attach a task-specific head (e.g., a linear layer or small MLP) to the pre-trained CT-MAE encoder (or a subset of its layers). Fine-tune either the head only (linear probing) or the entire model end-to-end with a smaller learning rate on the labeled dataset for the specific task.
*   **Baselines:** Compare CT-MAE against:
    *   Standard Transformers [Vaswani et al., 2017] with imputation (e.g., forward-fill, mean imputation) and time embeddings.
    *   Time-Series Transformer (TST) [6] trained end-to-end or pre-trained with a similar MAE objective if possible.
    *   RNN-based models (LSTMs, GRUs) with imputation.
    *   Simpler self-supervised methods (e.g., contrastive learning [8, 10]) adapted for multi-modal irregular data.
    *   Recent specialized MAE models for biosignals [1, 2] adapted to the specific task and data modality, if feasible.
*   **Evaluation Metrics:**
    *   Classification: Area Under the Receiver Operating Characteristic Curve (AUC-ROC), Area Under the Precision-Recall Curve (AUPRC), F1-score, Accuracy.
    *   Reconstruction (during pre-training analysis): Mean Squared Error (MSE), Mean Absolute Error (MAE).
    *   Uncertainty Calibration: Expected Calibration Error (ECE).
    *   Interpretability: Analyze attention weights qualitatively or quantitatively (e.g., using attention rollout) to understand which past observations or modalities contribute most to predictions.
*   **Ablation Studies:** Systematically evaluate the contribution of key components:
    *   Continuous-time embedding vs. simple time discretization/binning.
    *   Cross-modal attention decoder vs. independent modality decoders.
    *   Value+timestamp masking vs. value-only masking.
    *   Impact of different temporal basis functions ($\phi_k$).
*   **Robustness Analysis:** Evaluate fine-tuned model performance on test sets with artificially increased levels of missing data (randomly dropping observations) or varying sampling frequencies/irregularity patterns. Compare robustness against baselines.

**4. Expected Outcomes & Impact**

**4.1. Expected Outcomes**
1.  **A Novel CT-MAE Architecture:** A fully implemented and validated novel deep learning architecture (CT-MAE) specifically tailored for irregularly sampled, multi-modal health time series.
2.  **Pre-trained Foundation Model:** A publicly released (code and potentially weights, subject to data usage agreements) pre-trained CT-MAE model on large-scale health data, serving as a foundation for various downstream health time series tasks.
3.  **State-of-the-Art Performance:** Demonstrated superior or competitive performance of the fine-tuned CT-MAE model on benchmark clinical prediction tasks (sepsis, arrhythmia detection, outcome prediction) compared to existing baselines, particularly under challenging data conditions (irregularity, missingness).
4.  **Improved Robustness:** Quantitative evidence showing that CT-MAE exhibits greater robustness to missing data and sampling irregularities compared to baseline models.
5.  **Insights into Model Behavior:** Analysis of model interpretability (e.g., via attention maps showing cross-modal dependencies) and calibrated uncertainty estimates, enhancing trust and understanding of the model's predictions.
6.  **Publications and Dissemination:** High-quality publications detailing the methodology, results, and findings in leading machine learning and healthcare informatics venues (including the Time Series for Health Workshop).

**4.2. Impact**
This research holds significant potential for impact:

*   **Scientific Impact:** Advances the field of time series representation learning, particularly for challenging real-world domains like healthcare. It contributes a novel method integrating continuous-time modeling, multi-modal fusion, and self-supervised learning via masked autoencoding. It provides a concrete implementation advancing the idea of Foundation Models for health time series.
*   **Clinical Impact:** By enabling more accurate and robust analysis of complex health data, CT-MAE could lead to improved clinical decision support systems for early diagnosis, risk stratification, and personalized treatment. Its ability to handle raw, irregular data reduces the need for complex preprocessing or imputation, potentially streamlining clinical workflow integration. The focus on interpretability and uncertainty aids clinical adoption.
*   **Societal Impact:** Ultimately contributes to improving patient care and outcomes by unlocking deeper insights from ubiquitous health data sources. By potentially releasing a pre-trained model, it could accelerate research and development in computational health across the community.

This work directly addresses the call for innovative methods, preliminary results, and relevance to healthcare challenges within the Time Series for Health Workshop. It aligns with the key themes and topics of interest, offering a promising direction for bridging the gap towards deployable and impactful machine learning systems in healthcare.

**5. References**

1.  Zhou, Y., Yang, Y., Gan, J., Li, X., Yuan, J., & Zhao, W. (2025). Multi-scale Masked Autoencoder for Electrocardiogram Anomaly Detection. *arXiv preprint arXiv:2502.05494*.
2.  Liu, R., Zippi, E. L., Pouransari, H., Sandino, C., Nie, J., Goh, H., Azemi, E., & Moin, A. (2023). Frequency-Aware Masked Autoencoders for Multimodal Pretraining on Biosignals. *arXiv preprint arXiv:2309.05927*.
3.  Pham, M., Saeed, A., & Ma, D. (2024). C-MELT: Contrastive Enhanced Masked Auto-Encoders for ECG-Language Pre-Training. *arXiv preprint arXiv:2410.02131*.
4.  Chen, Z., Du, Y., Hu, J., Liu, Y., Li, G., Wan, X., & Chang, T.-H. (2022). Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training. *arXiv preprint arXiv:2209.07098*.
5.  He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). Masked autoencoders are scalable vision learners. *arXiv preprint arXiv:2111.06377*.
6.  Qian, Z., Zhang, Y., Li, Y., Djolonga, J., & Günnemann, S. (2021). Time-Series Transformer for Irregularly Sampled Data. *arXiv preprint arXiv:2106.14112*.
7.  Morrill, J., Hensman, J., & Solin, A. (2020). Continuous-Time Models for Stochastic Processes. *arXiv preprint arXiv:2006.06820*.
8.  Liu, Y., Cui, Z., Hong, X., & Yang, J. (2021). Self-Supervised Learning for ECG-Based Emotion Recognition. *arXiv preprint arXiv:2104.01666*.
9.  You, J., Ying, R., & Leskovec, J. (2020). Handling Missing Data with Graph Neural Networks. *arXiv preprint arXiv:2006.07572*.
10. Zhang, Y., Yang, Q., & Zhang, W. (2019). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. *arXiv preprint arXiv:1905.10437*.
11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All you Need. *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.

*(Note: Citations from the provided literature review are included. Additional standard citations like Vaswani et al. are added for context.)*

---