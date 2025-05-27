Title  
Dynamic Adversarial Training for Robust Generative AI Watermarking  

1. Introduction  
1.1 Background  
In the era of rapid advances in generative models, invisible watermarking has emerged as a critical tool for authenticating AI-generated content, attributing provenance, and deterring misuse. Recent works such as InvisMark (Xu et al., 2024) and Certifiable Robust Image Watermark (Jiang et al., 2024) demonstrate that imperceptible marks can be embedded into high-resolution images while resisting many common distortions. However, most existing methods rely on static embedding schemes or one-off adversarial examples. Attackers can reverse-engineer the watermarking process or apply novel distortions—cropping, noise, inpainting, style transfer—to degrade or eliminate the watermark. The arms race between watermark embedder and attacker thus demands a paradigm that anticipates evolving threats.  

1.2 Research Objectives  
This proposal aims to develop a Dynamic Adversarial Training (DAT) framework that co‐trains a watermark embedder and a suite of adversarial removal models in a zero‐sum game. The key objectives are:  
- O1: Design a generator network G that embeds multi‐bit watermarks into images with high imperceptibility (PSNR ≥ 50 dB, SSIM ≥ 0.995) and minimal impact on downstream perception (CLIP similarity ≥ 0.98).  
- O2: Construct an ensemble of adversarial attack models A = {A₁,…,Aₖ} that simulate both traditional distortions (e.g., crop, noise, JPEG) and learned removal strategies.  
- O3: Formulate a minimax optimization  
   $$\min_{G}\max_{A\in\mathcal{A}}\; \mathcal{L}_{\text{detect}}(G,A)\;+\;\lambda\,\mathcal{L}_{\text{perc}}(G)\;-\;\mu\,\mathcal{L}_{\text{adv}}(A,G)$$  
  that yields watermarks robust to unseen attacks.  
- O4: Empirically validate DAT on standard benchmarks (e.g., W-Bench) and compare against state-of-the-art methods: InvisMark, VINE, Spread Them Apart, and Certifiable Smoothing.  

1.3 Significance  
By dynamically adapting the watermark to the current strongest adversary, DAT promises:  
- Robustness to novel, unforeseen attacks—closing the generalization gap identified in recent surveys.  
- A unified training recipe that scales to high-resolution images without per‐instance fine‐tuning.  
- A blueprint for industry deployment in media, publishing, and IP enforcement—addressing the policy and regulation themes of the GenAI Watermarking workshop.  

2. Methodology  
2.1 Overview  
Our DAT framework comprises three modules: (i) a watermark embedder $G_\theta$, (ii) a watermark detector $D_\phi$, and (iii) an adversarial attack ensemble $\{A_{\omega_i}\}_{i=1}^k$. We alternate updates: strengthening $D$ and $G$ against the current adversaries, then evolving adversaries to exploit the latest embedder. Figure 1 (not shown) diagrams this loop.  

2.2 Data Collection  
- Source images: We will use the FFHQ (70k faces at 1024×1024) and ImageNet (1.2M images) datasets.  
- Preprocessing: Standard normalization to [–1,1]; random horizontal flips.  
- Watermark payload: 256-bit binary strings sampled uniformly for each image.  

2.3 Model Architectures  
2.3.1 Watermark Embedder $G_\theta$  
- A U-Net backbone with skip connections.  
- Input: clean image $x$ and payload $b\in\{0,1\}^{256}$.  
- Embedding head: A fully connected layer reshapes $b$ into a spatial map, concatenated with the U-Net’s bottleneck features.  
- Output: watermarked image $\tilde x = G_\theta(x,b)$.  

2.3.2 Watermark Detector $D_\phi$  
- A ResNet-50 encoder that takes $\hat x$ (possibly attacked) and predicts $\hat b\in[0,1]^{256}$.  
- Detection loss:  
   $$\mathcal{L}_{\text{detect}}(G,D,A) = \mathbb{E}_{x,b}\Big[\sum_{j=1}^{256} \text{BCE}\big(\hat b_j,\,b_j\big)\Big]\quad\text{where}\;\hat b = D_\phi\big(A(G_\theta(x,b))\big).$$  

2.3.3 Adversarial Attack Ensemble $\{A_{\omega_i}\}$  
- Fixed “classical” attacks:  
  • Gaussian noise with $\sigma\in[0.01,0.1]$  
  • JPEG compression, quality $q\in[10,90]$  
  • Random crop & resize: crop ratio 0.6–0.9  
  • Inpainting via partial masks (10–30% region) using a pretrained diffusion inpaint model.  
- Learned removers $A_{\omega_i}$: lightweight convolutional nets trained to minimize detector accuracy:  
   $$\min_{\omega_i}\;-\mathcal{L}_{\text{detect}}\big(G,D,A_{\omega_i}\big)\;+\;\alpha_{\text{adv}}\|\tilde x - A_{\omega_i}(\tilde x)\|_2^2.$$  

2.4 Loss Functions  
- Perceptual fidelity: use a combination of pixel‐wise MSE and CLIP perceptual loss:  
   $$\mathcal{L}_{\text{perc}}(G) = \mathbb{E}_x\big[\|G_\theta(x,b)-x\|_2^2\big] + \beta\,\mathbb{E}_x\big[1 - \mathrm{sim}_{\mathrm{CLIP}}(G_\theta(x,b),x)\big].$$  
- Adversarial payoff (for the generator):  
   $$\mathcal{L}_{\text{adv}}(G,A) \;=\; \mathbb{E}_{x,b}\Big[\sum_{j=1}^{256}\text{BCE}\big(D_\phi\big(A(G_\theta(x,b))\big)_j,\,b_j\big)\Big].$$  
- Overall minimax objective:  
   $$
     \min_{\theta,\phi}\;\max_{i=1,\dots,k}\;\Big[\mathcal{L}_{\text{detect}}(G,D,A_{\omega_i}) \;+\;\lambda\,\mathcal{L}_{\text{perc}}(G)\Big]
     \;-\;\mu\,\sum_{i=1}^k\mathcal{L}_{\text{adv}}(G,A_{\omega_i}).
   $$  
  Hyperparameters: $\lambda=1.0,\;\beta=0.01,\;\mu=0.5,\;\alpha_{\text{adv}}=0.1$.  

2.5 Training Algorithm  
Algorithm 1: Dynamic Adversarial Training Loop  

1. Initialize $\theta,\phi,\{\omega_i\}_{i=1}^k$ randomly.  
2. For each epoch:  
   a. Sample batch $(x^{(n)},b^{(n)})_{n=1}^N$.  
   b. Update adversarial weights $\{\omega_i\}$ by gradient ascent on $-\mathcal{L}_{\text{detect}}$.  
   c. Update detector $D_\phi$ by gradient descent on $\mathcal{L}_{\text{detect}}$.  
   d. Update embedder $G_\theta$ by gradient descent on $\max_i\mathcal{L}_{\text{detect}} + \lambda\mathcal{L}_{\text{perc}} - \mu\sum_i\mathcal{L}_{\text{adv}}$.  
3. Periodically (every 5 epochs) expand the classical attack set by random parameter sampling.  
4. Stop when validation detection accuracy under all $A_i$ converges.  

2.6 Experimental Design  
2.6.1 Baselines  
- InvisMark (Xu et al., 2024)  
- VINE (Lu et al., 2024)  
- Spread Them Apart (Pautov et al., 2025)  
- Certifiable Smoothing (Jiang et al., 2024)  

2.6.2 Evaluation Metrics  
- Bit detection accuracy under attack: proportion of bits correctly recovered.  
- Bit error rate (BER): average Hamming distance over 256 bits.  
- PSNR & SSIM between $x$ and $\tilde x$.  
- CLIP similarity: $\mathrm{sim}_{\mathrm{CLIP}}(\tilde x,x)$.  
- Robustness to unseen attacks: test on style transfer and generative inpainting from external models.  
- Computational overhead: inference time (ms) per 1024×1024 image.  

2.6.3 Protocol  
- Train DAT on FFHQ-ImageNet mix for 200 epochs, batch size 16, using Adam ($\eta=10^{-4}$).  
- Evaluate on a held‐out 5k image set under:  
   • Standard distortions (20 combinations of crop, noise, JPEG).  
   • Learned removal via unseen architectures.  
   • Black-box oracle attacks using API queries (for generalization).  
- Report mean ± std over three runs.  

3. Expected Outcomes & Impact  
3.1 Technical Outcomes  
- A generative watermark embedder that achieves ≥ 95% detection accuracy under aggressive distortions (e.g., 30% crop + σ=0.05 noise + Q=30 JPEG).  
- Imperceptibility metrics: PSNR ≥ 50 dB, SSIM ≥ 0.995, CLIP similarity ≥ 0.98.  
- Empirical demonstration of generalization: ≥ 90% detection accuracy on five unseen attack families (e.g., style transfer, random inpainting).  
- Ablation studies quantifying the contribution of each adversary and each loss term.  

3.2 Broader Impacts  
- Industry Deployment: The DAT framework can be integrated into existing content‐generation pipelines (e.g., Adobe Firefly, DALL·E) to embed robust watermarks at inference time, satisfying media integrity requirements.  
- Policy & Regulation: Provides quantitative benchmarks for regulators to specify minimum robustness standards.  
- Ethical Use: Enables trustworthy tracing of AI‐generated content provenance, deterring misuse (e.g., deepfakes, misinformation).  

3.3 Dissemination  
- Open-source release of code and pretrained models on GitHub under MIT license.  
- Public benchmark suite (DAT-Bench) augmenting W-Bench with new adversarial scenarios.  
- Publications targeting top venues: NeurIPS, ICCV, and a dedicated GenAI Watermarking workshop paper.  

4. Timeline  
- Months 1–3: Data pipeline, implement embedder and detector.  
- Months 4–6: Build classical and learned adversaries; preliminary joint training.  
- Months 7–9: Full dynamic adversarial training; hyperparameter tuning.  
- Months 10–12: Extensive evaluation against baselines, ablation studies, manuscript preparation.  

In summary, this proposal presents a novel Dynamic Adversarial Training framework that anticipates and adapts to evolving watermark removal strategies. By framing watermark embedding as a minimax game and leveraging both classical distortions and learned adversaries, we expect to achieve state-of-the-art robustness while preserving content quality—advancing the science and practice of trusted AI content provenance.