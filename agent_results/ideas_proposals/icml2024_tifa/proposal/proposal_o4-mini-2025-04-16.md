Title  
Cross-Modal Latent Space Watermarking for Robust Provenance Tracing in Multi-Modal Generative AI  

1. Introduction  
Background  
Recent advances in Multi-Modal Generative Models (MMGMs)—capable of synthesizing text, images, audio and video—have unlocked unprecedented creative and analytic capabilities. Models such as Stable Diffusion (image), Sora (video), Latte (audio+text) and large multi-modal LLMs (e.g., Llava, QwenVL) can jointly process and generate across modalities. While such Foundation Models (MFMs) and AI Agents promise to accelerate scientific discovery and creative workflows, they also introduce new vectors for misinformation, deepfakes and unauthorized content reuse. A critical element of building trustworthy MFMs is the ability to verifiably trace the provenance of generated artifacts, distinguishing them from real‐world content and identifying the specific model, version or user session that produced them.  

Research Objectives  
This project aims to design and validate a unified watermarking framework that:  
1. Embeds a compact, cryptographically verifiable identifier into the *latent representations* of MMGMs, *before* generation, so that any output in text, image, audio or video carries the watermark.  
2. Ensures high imperceptibility (minimal distortion of utility or perceptual quality) and high robustness (reliable decoding under compression, cropping, re-encoding and adversarial removal attempts).  
3. Scales to large-scale generative systems with minimal computational overhead and integrates seamlessly into existing diffusion‐ and transformer-based architectures.  
4. Provides cross-modal decoding: a single extraction mechanism that can recover the watermark from any modality (e.g., decode a video watermark from a single frame).  

Significance  
By enabling verifiable content provenance across modalities, our framework will:  
– Deter malicious misuse of generative AI (misinformation, unauthorized impersonation).  
– Facilitate model accountability, copyright enforcement and regulatory compliance.  
– Lay the technical groundwork for industry standards in multi-modal watermarking.  

2. Methodology  

2.1 Overview of the Watermarking Framework  
We propose to wrap each generative model \(G_m\) (for modality \(m\in\{\text{text, image, audio, video}\}\)) with a lightweight *Embedder* \(E_m\) that injects an \(L\)-bit watermark vector \(w\in\{0,1\}^L\) into the latent code \(z\), and a shared *Decoder* \(D\) that, given a (possibly transformed) output \(x'\), recovers \(w\). The data flow is:  
1. Sample or encode content‐latent \(z\sim P_{Z_m}\).  
2. Compute watermarked latent:  
   \[
     z' = E_m(z, w)\,,
   \]  
3. Generate content:  
   \[
     x = G_m(z')\in\mathcal{X}_m\,.
   \]  
4. After post-processing (augmentation) \(x' = T(x)\), decode:  
   \[
     \hat w = D(x')\,.
   \]  

2.2 Latent Embedding Network  
For each modality \(m\):  
– Let the unwatermarked latent be \(z\in\mathbb{R}^d\).  
– We design \(E_m\) as a shallow neural network that outputs a *perturbation* \(\Delta z\in\mathbb{R}^d\) given \((z,w)\). We set  
  \[
    z' = z + \lambda_m\,\Delta z\,,\quad \Delta z = \text{MLP}_m\bigl([z; \,\Psi(w)]\bigr),
  \]  
  where \(\Psi:\{0,1\}^L\to\mathbb{R}^{p}\) maps the binary watermark to an embedding of size \(p\), and \(\lambda_m\) controls the imperceptibility–robustness tradeoff.  
– Embedding networks for image/video latents can mirror the U-Net architecture in diffusion models; for text latents, \(E_{\text{text}}\) injects signals into token embeddings or cross-attention key vectors; for audio, \(E_{\text{audio}}\) perturbs spectrogram latents.  

2.3 Decoder Network  
We employ a unified decoder \(D\) that takes raw or transformed content \(x'\) and outputs bit-wise probabilities \(\hat w\in[0,1]^L\). \(D\) is a small, modality-agnostic network:  
– For images/video, \(D\) uses a CNN + global pooling.  
– For audio, \(D\) uses a 1D-CNN over spectrogram.  
– For text, \(D\) employs a transformer encoder over tokens.  
Weights are shared across modalities to encourage a universal decoding strategy.  

2.4 Training Objective  
We jointly train \(\{E_m\}_m\) and \(D\) (while freezing \(G_m\)) by minimizing, for each modality \(m\), the following composite loss:  
\[
  \mathcal{L}_m = 
    \underbrace{\mathbb{E}_{z,w,x}[\ell_{\text{task}}(G_m(z'),\,x)]}_{\text{generation fidelity}}
  + \beta\,\mathbb{E}_{z,w,T}[\ell_{\text{decode}}\bigl(D(T(G_m(z'))),\,w\bigr)]
  + \gamma\,\|\Delta z\|_2^2\,,
\]  
where:  
– \(\ell_{\text{task}}\) is a reconstruction or perceptual loss (e.g., \(\ell_1\), VGG-feature loss, cross-entropy for text).  
– \(\ell_{\text{decode}}\) is a binary cross-entropy over bits of \(w\).  
– \(T\sim\mathcal{D}_{\text{aug}}\) is a random augmentation (compression, cropping, format conversion, additive noise, spectral resampling for audio, paraphrasing for text).  
– \(\gamma\) regularizes the embedding magnitude for imperceptibility.  
Hyper-parameters \(\beta,\gamma,\lambda_m\) are tuned per modality.  

2.5 Robustness via Adversarial Augmentation  
To resist adversarial removal, we incorporate *adversarial training* in latent space: for each example, we solve  
\[
  \max_{\|\delta\|\le \epsilon}
    \ell_{\text{decode}}\bigl(D\bigl(T(G_m(z'+\delta))\bigr),\,w\bigr),
\]  
using PGD on \(\delta\), and include this adversarial loss term in \(\mathcal{L}_m\). This min–max formulation enhances the scheme’s resilience to gradient-based watermark removal.  

2.6 Cross-Modal Code Design  
We adopt a 64-bit session code \(w\), augmented by a BCH error-correcting code to yield a 128-bit ECC code \(w_\text{ECC}\). This improves bit-recovery under severe distortions. A cryptographic hash of \((\text{model\_ID}\,\|\,\text{version}\,\|\,\text{session\_salt})\) seeds \(w\), ensuring each generation has a unique identifier yet can be publicly verified if the secret key is released.  

2.7 Experimental Design  
Data Collection  
– Image: MS COCO, OpenImages.  
– Video: MSR-VTT.  
– Audio: LibriSpeech subsets.  
– Text: Wikipedia + Common Crawl prompts for LLM output.  

Baselines  
– InvisMark (Xu et al., 2024), GenPTW (Gan et al., 2025), VLPMarker (Tang et al., 2023), Fernandez (2025).  

Evaluation Metrics  
1. Imperceptibility  
   – Image/Video: PSNR, SSIM, FID.  
   – Audio: PESQ, STOI.  
   – Text: Perplexity, BLEU.  
2. Watermark Recovery  
   – Bit-error rate (BER), decoding accuracy.  
   – Robustness under augmentations: JPEG (Q10–Q95), MPEG-4 re-encode, MP3 64-320 kbps, cropping ≥ 30%, resampling (8→16 kHz).  
   – Adversarial removal success rate vs. output quality (∆PSNR).  
3. Computational Overhead  
   – Additional parameters (%) and generation latency increase.  

Ablations  
– Effect of \(\lambda_m\), \(\beta\), code length \(L\).  
– Joint vs. modality-specific decoder.  
– With vs. without adversarial training.  

Implementation  
– PyTorch, Hugging Face Diffusers, Accelerate.  
– GPUs: NVIDIA A100 (8 × 40 GB), 4–6 weeks total training time.  

3. Expected Outcomes & Impact  

3.1 Technical Deliverables  
– A *Cross-Modal Latent Watermarking* library implementing \(\{E_m,D\}\) for text, image, audio, video pipelines.  
– Trained checkpoints for Stable Diffusion-based image, Sora video, Whisper audio, and GPT-derivatives.  
– A public benchmark suite and evaluation scripts measuring imperceptibility, robustness and adversarial resistance.  
– A white-paper and open-source code to seed standardization efforts in AI watermarking.  

3.2 Scientific Contributions  
– Demonstration that latent-space watermarking can be unified across disparate modalities with a single decoding network.  
– A novel adversarial training regime for watermark robustness against sophisticated removal attacks.  
– Quantitative trade-off curves between imperceptibility, robustness and overhead across four modalities.  

3.3 Societal and Regulatory Impact  
– Provides a practical tool for platforms (social media, news agencies) to tag and track AI-generated content, mitigating misinformation.  
– Empowers copyright holders and regulatory bodies with verifiable provenance, enabling accountability and legal recourse.  
– Lays groundwork for multi-modal watermarking standards, aligned with forthcoming AI governance frameworks.  

4. References  
[1] Xu, R., et al. “InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance.” arXiv:2411.07795, 2024.  
[2] Gan, Z., et al. “GenPTW: In-Generation Image Watermarking for Provenance Tracing and Tamper Localization.” arXiv:2504.19567, 2025.  
[3] Singhi, S., et al. “Provenance Detection for AI-Generated Images: Combining Perceptual Hashing, Homomorphic Encryption, and AI Detection Models.” arXiv:2503.11195, 2025.  
[4] Jiang, Z., et al. “Watermark-based Attribution of AI-Generated Content.” arXiv:2404.04254, 2024.  
[5] Fernandez, P. “Watermarking across Modalities for Content Tracing and Generative AI.” arXiv:2502.05215, 2025.  
[6] Desu, A., et al. “Generative Models are Self-Watermarked: Declaring Model Authentication through Re-Generation.” arXiv:2402.16889, 2024.  
[7] Jiang, Z., et al. “Evading Watermark based Detection of AI-Generated Content.” arXiv:2305.03807, 2023.  
[8] Zhang, H., et al. “Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models.” Cryptology ePrint Archive: 2023/1776, 2023.  
[9] Zhong, H., et al. “Copyright Protection and Accountability of Generative AI: Attack, Watermarking and Attribution.” arXiv:2303.09272, 2023.  
[10] Tang, Y., et al. “Watermarking Vision-Language Pre-trained Models for Multi-modal Embedding as a Service.” arXiv:2311.05863, 2023.