1. Title  
“StegaDiff: A Diffusion-Based Steganographic Watermarking Framework for Verifiable Text-to-Speech Synthesis”  

2. Introduction  
2.1 Background  
Generative text-to-speech (TTS) models based on diffusion processes have recently achieved human-level audio quality and expressiveness. At the same time, malicious audio deepfakes—fabricated speech that convincingly mimics real speakers—pose serious threats to trust in journalism, legal evidence, and personal identity. Although post-hoc detection methods (e.g., self-supervised WavLM classifiers (Guo et al., 2023), FakeSound detectors (Xie et al., 2024)) achieve impressive results, they suffer from false positives in noisy environments and are rarely integrated into the generative pipeline itself. Steganographic watermarking—embedding imperceptible, robust codes into synthetic audio—offers a proactive way to trace and authenticate model outputs. Prior work on audio watermarking (Liu et al., 2025; San Roman et al., 2024) demonstrates feasibility, yet no end-to-end framework exists that jointly trains a diffusion-based TTS model with watermark embedding and extraction.  

2.2 Research Objectives  
This project aims to design, implement, and evaluate an end-to-end steganographic watermarking framework, “StegaDiff,” integrated directly into a diffusion-based TTS model. Our specific objectives are:  
• Develop a conditioning mechanism that embeds a secret, content-specific watermark code $w\in\{0,1\}^k$ into the denoising process with imperceptible audio distortion (<1 dB SNR loss).  
• Design a differentiable watermark extractor network that, given arbitrary audio (possibly corrupted), recovers $w$ with ≥ 98 % accuracy.  
• Demonstrate zero-shot detection capability: the extractor should generalize to unseen generative models and distortions.  
• Establish standardized metrics and benchmarks on public TTS datasets (VCTK, FS2) for reproducible evaluation of audio watermarking.  

2.3 Significance  
By embedding provenance information (e.g., author ID, timestamp, prompt hash) at synthesis time, StegaDiff provides an accountable foundation for responsible AI deployment in media, legal, and consumer applications. It mitigates disinformation risks, protects voice privacy, and establishes a community benchmark for future research into secure generative audio.  

3. Methodology  
3.1 Data Collection and Preprocessing  
We will use two public corpora:  
• VCTK (44 kHz, 109 speakers)  
• FS2 (LJSpeech-style single-speaker dataset)  
Preprocessing steps:  
1. Resample to 24 kHz and normalize amplitude.  
2. Extract phoneme sequences via a pretrained phonemizer.  
3. Compute mel-spectrograms with 80 bands, 50 ms window, 12.5 ms hop.  
4. Split each dataset into 90 % train / 10 % test.  

3.2 Model Architecture  
We build on a diffusion TTS backbone (e.g. Grad-TTS or DiffWave):  
• Forward process: $$q(z_{i}|z_{i-1}) = \mathcal{N}(\sqrt{1-\beta_i}\,z_{i-1}, \beta_i I)\,,\quad i=1\ldots T$$  
• Reverse denoising network $\epsilon_\theta(z_i,i,t)$ conditioned on text embedding $t$.  

Steganographic conditioning: we introduce a learnable watermark embedding network $\phi_w:\{0,1\}^k\!\rightarrow\!\mathbb{R}^d$. At each UNet block, we concatenate $\phi_w(w)$ with the text context:  
  $$h^{(l)} = \mathrm{Block}^{(l)}\bigl(h^{(l-1)},\;[\mathrm{Enc}(t);\;\phi_w(w)]\bigr)\,. $$  
This ensures the secret code modulates the denoising trajectory $z_T\!\to\!z_0^{\rm wm}$.  

3.3 Loss Functions  
We jointly minimize three losses:  
1. Diffusion reconstruction loss:  
   $$\mathcal{L}_{\rm diff} = \mathbb{E}_{z_i,t,w}\Bigl[\bigl\|\epsilon_\theta(z_i,i,t,w)-\epsilon\bigr\|_2^2\Bigr]\,. $$  
2. Watermark extraction loss: given extractor network $D_\psi$, which maps waveforms to $\hat w\in[0,1]^k$,  
   $$\mathcal{L}_{\rm wm} = -\sum_{j=1}^k \bigl[w_j\log \hat w_j + (1-w_j)\log(1-\hat w_j)\bigr]\,. $$  
3. Imperceptibility loss (psychoacoustic-weighted spectrogram difference):  
   $$\mathcal{L}_{\rm imp} = \tfrac{1}{N}\big\|M\odot\bigl(S(z_0)-S(z_0^{\rm wm})\bigr)\big\|_2^2, $$  
   where $S(\cdot)$ computes mel-spectrograms, and $M$ is a masking matrix derived from a psychoacoustic model.  

Total loss:  
   $$\mathcal{L} = \mathcal{L}_{\rm diff} \;+\;\lambda_{\rm wm}\mathcal{L}_{\rm wm}\;+\;\lambda_{\rm imp}\mathcal{L}_{\rm imp}\,. $$  
Hyperparameters $\lambda_{\rm wm},\lambda_{\rm imp}$ balance detection accuracy vs. audio quality.  

3.4 Watermark Extraction Network  
The extractor $D_\psi$ is a 1D-CNN followed by multi-head attention layers that output $\hat w$. During training, we apply random augmentations (MP3 compression @128 kbps, AWGN at SNR=10 dB, reverberation) to $z_0^{\rm wm}$ before feeding to $D_\psi$. This encourages robustness.  

3.5 Training Procedure  
Algorithm (pseudo‐code):  
```
for each minibatch:
  sample prompts t, watermark codes w
  sample clean audio x ~ TTS(t)
  sample noise ε ~ N(0,I)
  compute z_i via forward diffusion
  predict ε̂ = ε_θ(z_i,i,t,w)
  compute L_diff, L_wm, L_imp
  L_total = L_diff + λ_wm L_wm + λ_imp L_imp
  backprop L_total to update θ, φ_w, ψ
```
We use Adam optimizer, lr=1e-4, batch size=16, T=50 diffusion steps. Code length $k=64$ bits. Training on 4 GPUs for 200k steps.  

3.6 Experimental Design  
We propose the following experiments:  
A. Baseline Comparison  
   • Compare StegaDiff to XAttnMark (Liu et al., 2025), AudioSeal (San Roman et al., 2024), and Diffusion-based watermarking (Chen et al., 2024). Metrics: detection accuracy, bit error rate (BER), PESQ, STOI, SNR.  
B. Robustness Tests  
   • Evaluate on audio after compression, noise, equalization, room impulse responses. Track detection accuracy vs. distortion severity.  
C. Zero-Shot Generalization  
   • Train extractor on one TTS model variant; test on a different architecture to assess whether $D_\psi$ generalizes to unseen generators.  
D. Ablation Studies  
   • Vary $\lambda_{\rm wm},\lambda_{\rm imp}$ to map the imperceptibility–robustness trade-off.  
   • Test injection at all UNet blocks vs. only early/late stages.  
E. Human Listening Tests  
   • Conduct MUSHRA‐style listening tests (n=20) to confirm imperceptibility of watermarks, ensuring scores within 5 points of non-watermarked audio.  

3.7 Evaluation Metrics  
• Detection accuracy (%) and bit error rate (BER).  
• Audio quality: PESQ, STOI, and SNR degradation (dB).  
• Speech intelligibility: word error rate (WER) via a pretrained ASR.  
• Robustness curves: detection accuracy vs. distortion type/severity.  
• Listening test scores (MUSHRA).  

4. Expected Outcomes & Impact  
4.1 Expected Outcomes  
• A unified TTS synthesis framework with embedded steganographic watermarks achieving ≥ 98 % detection accuracy under clean and distorted conditions, with <1 dB SNR loss.  
• Demonstration of zero-shot watermark detection across model variants, validating generalization.  
• A public benchmark suite and open-source code for reproducible evaluation of audio watermarking in generative speech.  

4.2 Impact  
This research will establish the first end-to-end, verifiable TTS pipeline that proactively embeds accountability into synthetic speech. Media organizations can authenticate audio sources, legal systems gain a technical means to verify evidence, and responsible AI developers obtain a standardized toolkit to prevent misuse. By open-sourcing StegaDiff and our benchmark protocols, we anticipate broad adoption in both academia and industry, laying the groundwork for trustworthy generative audio technologies.