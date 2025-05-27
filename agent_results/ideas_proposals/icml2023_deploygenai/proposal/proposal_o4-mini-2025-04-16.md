1. Title  
SAFEGEN: An Interpretable Safety‐Check Framework for Generative Medical Imaging  

2. Introduction  
Background  
The recent success of generative models—GANs, VAEs and diffusion models—in natural language and computer vision has spurred interest in their application to medical imaging. Synthetic scans can augment scarce datasets, de-identify patient data, and enable data‐hungry AI systems in domains such as CT, MRI and X-ray analysis. However, unvetted synthetic images can contain unrealistic anatomical structures, subtle artifacts or distributional shifts that mislead downstream diagnostic models or clinicians, risking patient safety.

Research Objectives  
This project aims to develop SAFEGEN, a modular framework that automatically assesses the safety and realism of generated medical images while providing interpretable, region-level feedback. The specific objectives are:  
1. Design an unsupervised anomaly‐detection module trained on real medical images to assign both global and pixel‐wise anomaly scores to synthetic scans.  
2. Integrate interpretability methods (Grad-CAM and Shapley value approximations) to highlight image regions driving anomalous scores.  
3. Validate SAFEGEN on multiple modalities (CT, MRI) using both generated healthy-pathology counterfactuals and known artifact injections.  
4. Conduct a human-facing evaluation involving radiologists to measure detection accuracy, trust and usability.

Significance  
SAFEGEN will fill a crucial gap in deploying generative AI for high-stakes medical applications by:  
• Reducing reliance on slow, manual visual inspection of synthetic data.  
• Offering fine-grained explanations that help developers diagnose failure modes.  
• Enabling clinicians to verify synthetic scans before use in training or decision support, thereby increasing trust and safety.  
• Providing a template for interpretable safety checks in other domains (e.g., autonomous vehicles, finance).

3. Methodology  

3.1 Data Collection and Preprocessing  
• Real medical‐image corpus: We will assemble publicly available datasets—ADNI (brain MRI), BraTS (glioma MRI), LIDC-IDRI (lung CT)—and partner with a clinical institution for additional CT/MRI scans under IRB approval. All images are resampled to a common voxel resolution and intensity–normalized to [0,1].  
• Synthetic image generation: Two state-of-the-art generative pipelines will produce synthetic scans:  
  – GAN‐based: StyleGAN2‐CT trained on lung CT volumes.  
  – Diffusion‐based: DDPM (Denoising Diffusion Probabilistic Model) for brain MRIs.  
• Artifact injection for evaluation: We will inject controlled anomalies (e.g., Gaussian noise, motion blur, ring artifacts, tissue dropout) into a subset of synthetic images, producing ground-truth anomaly masks.  

3.2 SAFEGEN Framework Overview  
SAFEGEN consists of two coupled modules:  

3.2.1 Anomaly Detection Module  
We train an autoencoder‐based detector on only real images to learn the manifold of healthy anatomy. Let $E_\theta: \mathcal{X}\to\mathcal{Z}$ and $G_\psi: \mathcal{Z}\to\mathcal{X}$ denote the encoder and decoder. We optimize the reconstruction loss  
$$  
\mathcal{L}_{\mathrm{rec}}(\theta,\psi)=\mathbb{E}_{x\sim P_{\mathrm{real}}}\big\|x - G_\psi(E_\theta(x))\big\|_2^2\,.  
$$  
At test time, given a synthetic image $x$, we compute:  
• Global anomaly score:  
$$  
A_{\mathrm{global}}(x)=\big\|x - G_\psi(E_\theta(x))\big\|_2\,.  
$$  
• Pixel‐wise anomaly map:  
$$  
A_{\mathrm{map}}(x)_{i,j}=\big|\,x_{i,j} - [G_\psi(E_\theta(x))]_{i,j}\big|\,.  
$$  
To capture multi‐scale anomalies, we may extend to feature‐space distances:  
$$  
A_{\mathrm{ms}}(x)=\sum_{\ell=1}^L w_\ell\big\|F_\ell(x)-F_\ell(G_\psi(E_\theta(x)))\big\|_2\,,  
$$  
where $F_\ell(\cdot)$ is the $\ell$-th layer activation of a pretrained feature extractor (e.g., VGG) and $w_\ell$ are learned weights.

3.2.2 Interpretability Module  
We augment anomaly detection with two complementary interpretability methods:  

a) Grad-CAM on the reconstruction error network  
Given the scalar global score $A_{\mathrm{global}}(x)$, we backpropagate gradients to the decoder’s last convolutional feature maps $\{M^k\}$ and compute:  
$$  
\alpha^k = \frac{1}{Z}\sum_{i,j}\frac{\partial A_{\mathrm{global}}(x)}{\partial M^k_{i,j}}\,,  
\quad  
\text{GradCAM}(x)=\mathrm{ReLU}\Big(\sum_k \alpha^k M^k\Big)\,  
$$  
where $Z$ is a normalization constant.

b) SHAP‐based pixel attributions  
To approximate Shapley values $\phi_{i,j}$ for each pixel we use the Deep SHAP framework. For pixel $p=(i,j)$,  
$$  
\phi_p = \sum_{S\subseteq P\setminus\{p\}}\frac{|S|!(|P|-|S|-1)!}{|P|!}\Big[f(x_{S\cup\{p\}})-f(x_S)\Big]\,,  
$$  
where $f$ is a lightweight anomaly‐score predictor and $x_S$ denotes the image with only pixels in $S$ present.

3.3 Algorithmic Pipeline  

Algorithm SAFEGEN_Evaluate(x):  
1. recon ← Gψ(Eθ(x))  
2. A_glob ← ∥x – recon∥₂  
3. A_map ← |x – recon|  ⊳ pixel map  
4. HC ← GradCAM(x)        ⊳ highlight regions by gradient  
5. HS ← DeepSHAP(x)       ⊳ Shapley map  
6. CombinedHeatmap ← normalize(α₁·A_map + α₂·HC + α₃·|HS|)  
7. Mask ← CombinedHeatmap > τ  ⊳ thresholding with τ set via validation  
8. return {A_glob, CombinedHeatmap, Mask}  

Hyperparameters α₁, α₂, α₃ and threshold τ will be chosen to maximize detection‐interpretability trade‐off on a held-out validation set.

3.4 Experimental Design  

Datasets & Splits  
• Training: 70% of real images for autoencoder training.  
• Validation: 10% real + synthetic for hyperparameter tuning.  
• Test: 20% synthetic images (clean + artifact injections) and 200 real clinical cases with known pathologies.

Baselines  
• Pure reconstruction–based anomaly detection (no interpretability).  
• DIA (Shi et al. 2023) fine-grained anomaly detector.  
• PHANES (Bercea et al. 2023) pseudo-healthy mask method.

Evaluation Metrics  
1. Detection performance: AUC-ROC, precision, recall, F1-score on synthetic artifact detection.  
2. Region‐level accuracy: IoU and Dice between Mask and ground-truth artifact masks.  
3. Interpretability quality:  
   – Pointing Game accuracy: fraction of top-K pixels in CombinedHeatmap that overlap ground truth anomalies.  
   – Insertion/Deletion metrics: impact of gradually adding/deleting pixels in order of attribution.  
4. Clinical human study:  
   – Participants: 10 board-certified radiologists.  
   – Task: classify 100 synthetic images as “safe” or “unsafe” (with/without SAFEGEN assistance).  
   – Outcomes: classification accuracy, decision time, trust rating (Likert scale).  

Statistical Analysis  
We will perform paired t-tests (p<0.05) to compare SAFEGEN against baselines on detection metrics. Human–AI studies will use repeated measures ANOVA to assess improvements in accuracy and trust.

3.5 Implementation Details  
• Autoencoder architecture: 5-layer U-Net with residual skip‐connections; latent dimension 256.  
• Optimizer: Adam, learning rate 1e-4, batch size 16, 100 epochs.  
• Hardware: NVIDIA A100 GPUs.  
• Codebase: PyTorch, integration with MONAI for medical‐image handling.  

4. Expected Outcomes & Impact  

Expected Outcomes  
• SAFEGEN will achieve >95% AUC in detecting injected artifacts, outperforming pure reconstruction baselines by >10%.  
• Combined interpretable heatmaps will yield IoU >0.70 with ground‐truth anomaly masks, enabling precise localization.  
• Radiologist study will show a statistically significant increase in correct “unsafe” classifications (Δaccuracy >15%) and reduced decision time (–20%) when using SAFEGEN.  
• Public release of code, pretrained SAFEGEN models, and a synthetic artifact dataset for community benchmarking.

Broader Impact  
• Trustworthy Deployment: By surfacing why a synthetic image is flagged unsafe, SAFEGEN fosters clinician trust and regulatory acceptance of generative AI in healthcare.  
• Risk Mitigation: Automatic anomaly checks prevent use of flawed synthetic data in downstream training, reducing risk of misdiagnosis by AI systems.  
• Generalizability: The SAFEGEN paradigm—unsupervised manifold learning + interpretability—can extend to other high-stakes domains (e.g., autonomous driving, financial fraud detection).  
• Community Resource: Open‐source tools and datasets will catalyze further research on interpretable safety checks for generative models.

In summary, SAFEGEN offers an end-to-end, interpretable solution to a critical barrier in deploying generative AI for medical imaging. By fusing anomaly detection with human‐facing explanations, it paves the way for safer, more transparent synthetic data generation in healthcare and beyond.