1. Title  
Dynamic Reliability‐Aware Multi‐Modal Fusion for Trustworthy Clinical Decision Support  

2. Introduction  
Background. Modern healthcare increasingly relies on multi‐modal data—radiology scans (CT, MRI), electronic health records (EHRs), pathology images, genomics—to form a holistic view of patient state. Deep learning fusion methods have shown promise in diagnosis, prognosis, and treatment planning. Yet in practice, clinical deployment remains limited due to trust gaps: some modalities may be noisy (e.g., low‐dose CT), missing (e.g., incomplete EHR fields), or out‐of‐distribution (e.g., scans from new scanner types). When a fused model treats all modalities equally, it can become overconfident even on corrupted inputs, potentially leading to harmful clinical decisions.  

Research Objectives. We propose a methodology that (1) dynamically estimates each modality’s reliability at inference time, (2) uses reliability‐aware attention to weight fusion, (3) produces well‐calibrated predictive uncertainties, and (4) offers interpretable modality‐weight maps to flag low‐confidence cases. The three core objectives are:  
• Develop a Bayesian multi‐modal fusion framework that yields per‐modality uncertainty estimates.  
• Introduce a self‐supervised Auxiliary Corruption Detection (ACD) task to train the network to recognize and quantify modality degradation.  
• Validate the approach on real‐world benchmarks with both simulated and genuine modality corruption, demonstrating improvements in robustness, calibration, and interpretability.  

Significance. By accurately quantifying and leveraging modality‐specific reliability, our approach will reduce overconfidence, improve safety margins in clinical decision support, and foster clinician trust. It will also serve as a foundation for regulatory‐compliant AI systems that can explicitly signal when a recommendation is uncertain due to data quality issues.  

3. Methodology  

3.1 Data Collection and Preprocessing  
• Datasets  
  – MIMIC‐CXR (chest X‐rays + EHRs) and MIMIC‐IV for clinical outcomes.  
  – TCGA multi‐modal cancer datasets (histopathology + genomics + MRI).  
  – In‐house multi‐center CT/MRI/EHR cohort (subject to IRB approval).  
• Preprocessing  
  – Imaging: standardize voxel spacing, intensity clipping, z‐score normalization.  
  – EHRs: extract time‐series of vitals, labs; embed categorical fields; pad/truncate sequences.  
  – Synchronization: align modalities by patient ID and time; create modality masks for missing data.  
• Corruption Simulation  
  – Noise injections: Gaussian noise $\mathcal{N}(0,\sigma^2)$ on images; random feature masks for EHR.  
  – Resolution degradation: down‐sampling / JPEG compression on images.  
  – Missingness: randomly drop one or more modalities per sample with probability $p_{\mathrm{miss}}$.  

3.2 Model Architecture  
The model consists of $M$ modality‐specific Bayesian backbones, a reliability estimation module, and an attention‐based fusion head (Figure 1).  

3.2.1 Modality Backbones as Bayesian Neural Networks  
For modality $m\in\{1,\dots,M\}$, let $x_m$ be the preprocessed input. We define a Bayesian neural network with parameters $w_m$ and approximate posterior $q(w_m\mid\theta_m)$. Using Monte Carlo Dropout [Gal and Ghahramani, 2016], each forward pass samples $w_m^{(t)}\sim q(w_m)$ to produce features  
$$h_m^{(t)} = f_m(x_m;w_m^{(t)})\,,\quad t=1,\dots,T\,,$$  
and predictive distribution  
$$p(y\mid x_m) \approx \frac{1}{T}\sum_{t=1}^T p(y\mid h_m^{(t)})\,. $$  
The total predictive variance (uncertainty) per modality is  
$$U_m = \underbrace{\frac{1}{T}\sum_{t}(p(y\mid h_m^{(t)}) - \bar p)^2}_{\text{aleatoric + epistemic}}\,,\quad \bar p=\frac{1}{T}\sum_t p(y\mid h_m^{(t)})\,. $$  

3.2.2 Reliability Weight Computation  
We convert $U_m$ into a reliability score $r_m$ via  
$$r_m = \frac{\exp(-\alpha\,U_m)}{\sum_{k=1}^M \exp(-\alpha\,U_k)}\,, $$  
where $\alpha>0$ controls sensitivity. A high $U_m$ (uncertain) yields low $r_m$.  

3.2.3 Attention‐Weighted Fusion  
Each backbone outputs a representation vector $h_m=\bar h_m=\frac1T\sum_t h_m^{(t)}$. The fused representation is  
$$H = \sum_{m=1}^M r_m\,h_m\,. $$  
$H$ is passed through a classification head $g(H)$ to predict clinical labels (disease classes, survival risk, etc.).  

3.2.4 Self‐Supervised Auxiliary Corruption Detection (ACD)  
To teach the model to detect modality degradation, we introduce binary corruption indicators $c_m\in\{0,1\}$ applied at random during training. The network also predicts $\hat c_m = \sigma(u_m^\top h_m + b_m)$ via a small corruption‐detection head. We use the auxiliary loss  
$$L_\mathrm{aux} = -\sum_{m=1}^M \bigl[c_m\log \hat c_m + (1-c_m)\log(1-\hat c_m)\bigr]\,. $$  
By optimizing this alongside the main task, the backbones learn features sensitive to corruption, improving $U_m$ estimation.  

3.3 Training Objective  
The overall loss for a batch of $N$ samples is  
$$L = \underbrace{\frac{1}{N}\sum_{i=1}^N \ell\bigl(g(H^{(i)}), y^{(i)}\bigr)}_{L_{\rm main}\text{ (e.g.\ cross‐entropy)}} + \lambda\,L_\mathrm{aux}\,, $$  
where $\lambda$ balances classification vs.\ auxiliary tasks. We optimize $L$ via Adam [Kingma & Ba, 2015], learning rate scheduling, and early stopping on a held‐out validation set.  

3.4 Experimental Design and Evaluation  
Dataset Splits  
• In‐distribution (ID): 70/15/15 train/val/test stratified by hospital/site.  
• Out‐of‐distribution (OOD): test on held‐out hospital(s) or patients from unseen scanning protocols.  

Baselines  
• Early fusion: concatenation of raw modality features, single deterministic network.  
• Late fusion: independent predictions aggregated by mean or majority vote.  
• MDA [Fan et al., 2024], DRIFA‐Net [Dhar et al., 2024], HEALNet [Hemker et al., 2023], DrFuse [Yao et al., 2024].  

Metrics  
• Discrimination: accuracy, AUC‐ROC, F1‐score.  
• Calibration: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score.  
• Uncertainty Quality:  
  – Negative log‐likelihood (NLL) on test data.  
  – Area Under the Risk‐Coverage Curve (AURC) for selective classification.  
• Robustness: performance drop (%) under simulated corruption (varying $\sigma$, $p_{\mathrm{miss}}$).  
• Interpretability: correlation between $r_m$ and known input quality metrics; qualitative heatmaps of $r_m$.  

Ablation Studies  
• Remove ACD auxiliary loss ($\lambda=0$).  
• Fixed equal weights $r_m=1/M$ to test reliability weighting.  
• Vary $\alpha$ sensitivity parameter.  

Statistical Validation  
Report mean±std over five random seeds; use paired t‐tests to assess significance at $p<0.05$.  

4. Expected Outcomes & Impact  
We anticipate that our Dynamic Reliability‐Aware Fusion (DRAF) framework will:  
1. Increase Robustness. On corrupted or missing modalities, DRAF should exhibit smaller performance degradation (e.g., <5% drop in AUC) compared to baseline fusion methods (>10% drop).  
2. Improve Calibration. By leveraging Bayesian uncertainty and reliability weighting, DRAF will reduce ECE by at least 30% relative to deterministic fusion.  
3. Enhance Detect‐Refuse Behavior. With uncertainty‐driven thresholds, the model can abstain on low‐confidence cases, ensuring higher precision on accepted samples.  
4. Provide Interpretability. Clinicians can inspect modality weights $r_m$, understanding which data sources drove each prediction and when to seek additional diagnostics.  

Broader Impact. This research directly addresses the trustworthiness pillars—uncertainty estimation, robustness to distribution shifts, interpretability, and fairness (by avoiding overreliance on biased modalities)—crucial for real‐world clinical AI adoption. We will open‐source our code, pretrained models, and a new benchmark suite for reliability‐aware multi‐modal fusion. This will propel further advances in safe, transparent, and effective AI tools for healthcare, ultimately leading to improved patient outcomes and clinician confidence.  

5. References  
[1] Lin Fan et al. “MDA: An Interpretable and Scalable Multi‐Modal Fusion under Missing Modalities and Intrinsic Noise Conditions.” arXiv:2406.10569, 2024.  
[2] Joy Dhar et al. “Multimodal Fusion Learning with Dual Attention for Medical Imaging.” arXiv:2412.01248, 2024.  
[3] Konstantin Hemker et al. “HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data.” arXiv:2311.09115, 2023.  
[4] Wenfang Yao et al. “DrFuse: Learning Disentangled Representation for Clinical Multi‐Modal Fusion with Missing Modality and Modal Inconsistency.” arXiv:2403.06197, 2024.  
[5] Yarin Gal and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” ICML, 2016.  
[6] Diederik P. Kingma and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” ICLR, 2015.