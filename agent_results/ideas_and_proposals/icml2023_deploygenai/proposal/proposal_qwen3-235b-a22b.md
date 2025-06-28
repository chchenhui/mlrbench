# SAFEGEN: Interpretable Safety Checks for Generative Medical Imaging

## 1. Introduction  
### Background  
Generative models have transformed biomedical imaging by enabling realistic synthetic data generation for training diagnostic systems and improving data imbalance. However, their deployment in high-stakes domains like healthcare remains risky. Current challenges include ensuring generated images are free from artifacts that could mislead clinicians or downstream AI systems, and providing clinicians with explainable assurances about image safety. Existing quality checks often rely on manual inspection or simplistic metrics like Frechet Inception Distance (FID), which lack both granularity in identifying spatial anomalies and interpretable rationale for clinicians.

### Research Objectives  
This proposal introduces **SAFEGEN**, a framework for automated, interpretable safety checks in medical generative imaging. Our goals are:  
1. To train an anomaly detection module that flags plausibly unreal regions in synthetic images.  
2. To integrate interpretability techniques (e.g., Grad-CAM, SHAP) to visualize features contributing to anomaly scores, enabling clinician verification.  
3. To validate SAFEGEN’s ability to correlate machine-detected artifacts with expert radiologist assessments through quantitative and qualitative evaluation.  

### Significance  
SAFEGEN addresses two critical workshop themes:  
- **Safety & Interpretability:** Fills the gap in post-hoc evaluation tools for generative medical imaging, combining robust anomaly detection (via contrastive learning) with human-understandable feedback.  
- **Real-World Deployment:** Enables iterative improvement of generative models by pinpointing failure modes (e.g., chemically unrealistic tissues in CT), accelerating adoption in clinical workflows.  

**Literature Context:** While PHANES (2023) reconstructs "pseudo-healthy" images for anomaly detection and THOR (2024) adapts diffusion models for robust medical analysis, none provide spatial interpretability for synthetic images. SAFEGEN closes this gap by unifying recent advances in interpretable GANs (e.g., medXGAN 2022) with fine-grained anomaly detection (e.g., DIA 2023).

---

## 2. Methodology  

### 2.1 Data Collection & Preprocessing  
**Target Modalities:** Synthetic medical images generated via diffusion models or GANs (e.g., CT scans of lungs, brain MRIs, retinal fundus images).  
**Real-World Datasets:** Utilize publicly available, de-identified datasets (BraTS for brain MRIs, NIH ChestX-ray14 for CXRs) and collaborate with hospitals for proprietary scans.  
**Preprocessing:**  
- Normalize intensity values (e.g., windowing in CT scans: Hounsfield Units ∈ [-1000, 1000]).  
- Split 3D volumes into 512×512 axial slices for 2D analysis.  
- Anonymize metadata (e.g., age, pathology labels) to preserve privacy.  

### 2.2 Model Architecture  
**Stage 1: Anomaly Detection**  
Train a **diffusion-based contrastive model** following DIA (2023) but adapted for synthetic image evaluation:  
1. Train a denoising diffusion model on real scans ($x^{(real)}$) to estimate the data distribution.  
   - Forward process: $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon$, where $t=1\dots T$.  
   - Reverse process: Predict noise $\epsilon_\theta(x_t, t)$ via UNet with attention.  
2. Use the trained model to reconstruct synthetic images ($x^{(syn)}$), computing pixel-wise residuals $r = |x^{(syn)} - \hat{x}^{(syn)}|$.  
3. Identify anomalous regions via top-$k$ residuals exceeding a threshold derived from real-image training statistics.  

**Stage 2: Interpretability Fine-Tuning**  
1. Train a **Grad-CAM** model to highlight discriminative regions. Let $f^c$ be the final CNN layer before classification head. Compute weights via activation gradients:  
   $$\alpha_c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial f^c_{ij}},$$  
   where $y^c$ is class score for realism (real/synthetic). Final heatmap: $\mu_f^c + \sigma\cdot\alpha_c \cdot f^c_{ij}$.  
2. Integrate SHAP for feature attribution by comparing synthetic snapshots against a reference dataset of 1000 real scans, using a resnet50 extractor.  

### 2.3 Experimental Design  
**Training Protocol:**  
- **Dataset Splitting:** 70% training, 15% validation (for hyperparameter tuning), 15% test.  
- **Evaluation Cohorts:** Mine synthetic images for 4 artifact types: (1) anatomically impossible juxtapositions (e.g., displaced organs), (2) noise/artiface patterns (e.g., motion blur in MRI), (3) intensity extremes (e.g., hyperattenuated calcifications), (4) subtle physiological inconsistencies (e.g., unrealistic lung nodule growth).  
- **Baselines:** Compare against PHANES (2023), MITS-GAN (2024), and vanilla diffusion-based detectors.  

**Metrics:**  
- **Quantitative:** AUROC for anomaly detection; Dice coefficient between heatmaps and ground-truth artifact masks (when available).  
- **Human Evaluation:** Deploy a 5-point Likert scale survey with 10 radiologists to rate heatmap relevance (e.g., "Is region A indicative of synthetic distortion?").  
- **Statistical Validity:** Wilcoxon signed-rank tests for inter-annotator consistency; bootstrap CI for ROCs.  

**Implementation Details:**  
- Compute: Train on 8×RTX 6000 Ada GPUs for 200 epochs; mixed-precision training (AMP).  
- Code Libraries: MONAI (2023) for 3D preprocessing, Donovan (2024) for diffusion models.  

---

## 3. Expected Outcomes & Impact  

**Primary Outputs:**  
1. **Benchmarked Framework:** Release SAFEGEN as an open-source toolkit (PyTorch + MONAI-compatible) for reproducible evaluation.  
2. **Quantitative Performance:** Demonstrate AUROC ≥ 0.89 on synthetic CXRs for artifact detection, outperforming SOTA models by ≥5%.  
3. **Interpretability Correlation:** Achieve Spearman’s $\rho > 0.80$ between heatmap-identified regions and radiologists’ annotations on 3/4 artifact categories.  

**Implications for Deployment:**  
- **Clinical Trust:** Provide visualize justification to clinicians vetting synthetic data for training implanted AI (e.g., FDA-regulated systems).  
- **Model Auditing:** Enable generative model developers to conduct root-cause analysis on flaws (e.g., "why does this CT subplanes) generate implausible rib structures?").  
- Policy Alignment: Addresses AI Act’s transparency requirements for high-risk healthcare applications by documenting model audit trails.  

---

## 4. Broader Impact & Future Directions  

**Interdisciplinary Advancement:** SAFEGEN’s framework can transition beyond radiology to other generative challenges:  
- Pathological image generation (e.g., validating spatial consistency in synthetic histopathology slides).  
- Counterfactual explanations in diagnostic systems (e.g., "Modify region B in CXR to make pneumonia less probable").  

**Limitations to Address:**  
- Generalization to 3D volumes: Extend Grad-CAM to operate volumetrically.  
- Artifact definition ground-truth: Crowdsourced labeling from AI radiology boards to mitigate expert bias.  

---

## 5. Timeline & Milestones  

| Phase                | Duration | Tasks                                      |  
|----------------------|----------|--------------------------------------------|  
| Dataset Curation     | Month 1  | Secure data access; ETL pipeline creation  |  
| Model Development    | Months 2-6 | Train diffusion contrastive net, Grad-CAM |  
| Pilot Evaluation     | Months 7-9 | Radiologist surveys on small CXR cohort    |  
| Optimization         | Months 10-11 | Hyperparameter tuning on 3D MRIs         |  
| Final Deployment     | Month 12 | Launch software, submit EMBC for validation |  

**Ethics Statement:** Adhere to HIPAA guidelines by discarding patient identifiers. Synthetic data sharing will align with the Montreal Cognitive Unlearning Pledge to prevent memorization.  

---  

By bridging the critical gaps in interpretable safety checks, SAFEGEN offers a systematic, deployable solution for high-stakes generative AI in healthcare, advancing the workshop’s mission of cross-domain, real-world impact.