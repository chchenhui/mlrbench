# Steganographic Watermarking for Verifiable Synthesis in Text-to-Speech Generative Models  

## Introduction  

The proliferation of generative adversarial networks (GANs) and diffusion-based text-to-speech (TTS) systems has enabled the creation of synthetic speech with near-human fidelity. However, this technological advancement poses existential risks to media trustworthiness, personal identity security, and legal integrity. Audio deepfakes are increasingly challenging to distinguish from real speech, with studies like FakeSound (2024) demonstrating that human listeners perform worse than machine detectors in identifying synthetic speech. Current detection frameworks such as AudioSeal (2024) achieve ~96% classification accuracy but suffer from high false positives when generalized to unseen generative models or post-processing attacks. This gap has created a critical need for proactive, verifiable synthesis methods that embed traceable provenance directly into the generation pipeline.  

Our research addresses this need through XAttnWatermark, a steganographic watermarking framework designed specifically for TTS systems. Unlike post-hoc detection methods, our approach integrates watermark encoding into denoising diffusion processes, embedding imperceptible identifiers during audio generation. The watermarks contain metadata such as source input prompts, origin timestamps, and generator identifiers, enabling forensic tracing of synthetic speech provenance while preserving audio clarity. Formally, let $ f_{\theta}(\cdot) $ denote a diffusion-based TTS model with parameters $ \theta $. Given text input $ x \in \mathcal{X} $ and secret code $ w \in \{0,1\}^k $, our framework generates $ y \in \mathbb{R}^T $ satisfying:  
$$
\left\| y - \text{TTS}_{\theta}(x; w) \right\|_2^2 \leq \epsilon \quad \text{and} \quad \text{PSNR}(y, \text{TTS}_{\theta}(x)) \geq 40 \text{dB},
$$
where $ \epsilon $ controls embedding distortion and PSNR ensures perceptual transparency. This builds upon XAttnMark (2025), which achieved 42.3 dB PSNR with 98% robust watermark retrieval, but differs in being tightly integrated with the generation pipeline rather than applied as a post-processor.  

This work addresses four critical challenges identified in recent literature (FakeSound 2024; Benchmarking Watermarking Methods 2024): (1) balancing watermark imperceptibility with robustness against real-world perturbations, (2) achieving integration with state-of-the-art TTS systems without degrading synthesis quality, (3) enabling zero-shot detection of unseen generative models through watermark-robust encoders, and (4) providing standardized benchmarks through comprehensive evaluation. By embedding provenance directly during synthesis, our framework avoids the limitations of reactive detection methods, offering a paradigm shift toward accountable AI. Potential impacts include enhanced trust in news media through audio authentication, legal enforceability of synthetic identity usage, and reduced voice cloning misuse in virtual communication platforms.  

## Methodology  

### Data Collection & Preprocessing  

Our evaluation will use the VCTK-Corpus (108 speakers, 44 kHz sampling) and the LJSpeech dataset (13,100 utterances) for initial training, augmented with synthetic speech from TTS benchmarks like FS2 (FastSpeech2). Audio will be preprocessed as 64-dimensional log-Mel spectrograms at 22.05 kHz, with phase reconstruction using Griffin-Lim. For adversarial testing, we will synthesize 20,000 utterances from 10 diverse TTS models (WaveNet, Tacotron, DiffSpeech) and apply transformations like pitch shifting, noise addition, and low-pass filtering based on ITU-T Rec. P.800.  

### Model Design & Training  

1. **Watermarked Audio Synthesis**:  
We extend the DiffSpeech pipeline with cross-attention watermarking layers. Let $ \mathcal{T}(x) = [h_1, ..., h_T] \in \mathbb{R}^{d \times T} $ represent text encoder outputs. The watermark code $ w \in \mathbb{R}^k $ is expanded to $ \hat{w} \in \mathbb{R}^{d \times T} $ via a positional embedding matrix $ P \in \mathbb{R}^{T \times k} $. Latent noise trajectories $ \mathbf{z}_t $ are updated using:  
$$
\mathbf{z}_t = \alpha_t \mathbf{z}_0 + \beta_t \epsilon \quad \text{with} \quad \epsilon \sim \mathcal{N}(0, I),  
$$
where the denoising network incorporates watermark information through conditional self-attention:  
$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{(QW^Q)(\hat{w}W^{\hat{w}} + KW^K)^T}{\sqrt{d_k}} \right) VW^V.
$$
This formulation adapts the cross-attention mechanism from XAttnMark (2025) to diffusion processes, enabling watermark-aware denoising.  

2. **Differentiable Watermark Extraction**:  
A U-Net-based detector $ D_{\phi} $ is trained to reconstruct $ \hat{w} = D_{\phi}(y) $ from watermarked audio. The architecture combines convolutional recurrent networks with phase-aware spectral reconstruction. Loss function components include:  
- Perceptual loss: $ \mathcal{L}_{\text{percept}} = \| y_{\text{clean}} - y_{\text{watermarked}} \|_1 $  
- Robustness loss: $ \mathcal{L}_{\text{robust}} = 1 - \text{SSIM}(w, D_{\phi}(y)) $  
- Adversarial loss: $ \mathcal{L}_{\text{adv}} = \log(1 - D_{\text{critic}}(D_{\phi}(y))) $  

3. **Watermark-Robust Encoder Training**:  
For zero-shot detection, we train a Provenance-ResNet101 encoder $ E_{\psi} $? using metric learning. Embedding pairs $ (E_{\psi}(y_{\text{real}}), E_{\psi}(y_{\text{generated}})) $ are trained with triplet loss $ \mathcal{L}_{\text{triplet}} = \max\left( \|a - p\|^2 - \|a - n\|^2 + \margin, 0 \right) $, where anchor $ a $, positive $ p $, and negative $ n $ represent clean, watermarked, and altered synthetic audio samples.  

### Training Protocol  

The generator-discriminator framework follows a staged training schedule:  
1. Pretrain TTS base model on VCTK-LJSpeech corpus (300 epochs, batch size=64, AdamW).  
2. Freeze TTS and train detector $ D_{\phi} $? with synthetic watermarked audio (100 epochs, learning rate=1e-4, cosine decay).  
3. Joint fine-tuning of TTS+detector with differentiable loss:  
$$
\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{perceptual}} + \beta \cdot \mathcal{L}_{\text{robustness}}(\hat{w}, w) + \gamma \cdot \mathcal{L}_{\text{adversarial}} 
$$
where weights are determined by grid search.  

4. Train encoder $ E_{\psi} $? using 80% of the DeepSpeech2 adversarial validation set.  

### Evaluation Metrics  

1. **Watermark Detection**:  
- Accuracy: $ \frac{\text{True Positive + True Negative}}{\text{Total Samples}} $?  
- Bit Error Rate $ \text{BER} = \frac{1}{k} \sum_{i=1}^k |w_i - \hat{w}_i| $  
- Mean Square Error $ \text{MSE} = \frac{1}{T} \sum_{t=1}^T (y_t - \hat{y}_t)^2 $?  

2. **Audio Quality**:  
- Perceptual Evaluation of Speech Quality $ \text{PESQ} $  
- Fréchet Audio Distance $ \text{FAD}_{\text{GT}} $? for naturalness  
- Short-Time Objective Intelligibility $ \text{STOI} $  

3. **Robustness Testing**:  
- Attack scenarios follow ITU-T Rec. P.800: MP3 encoding (128kbps), resampling (16kHz), noise addition (SNR=15dB), and low-pass filtering (<8kHz).  

4. **Benchmark Comparison**:  
- Against XAttnMark (2025), AudioSeal (2024), and ProvenanceResNet (2023) on:  
- Detection accuracy after attacks  
- Embedding capacity (bits/second)  
- Synthesis speed overhead ($ \Delta $ms/utterance)  

Statistical significance will be assessed using bootstrap sampling (95% confidence intervals) and the Benjamini-Hochberg correction for multiple comparisons.  

## Expected Outcomes & Impact  

This research is expected to achieve three transformative outcomes:  

**1. Technical Advancements**:  
- Demonstrate ≥98% watermark detection accuracy on the ASVspoof 2021 DF test set with <1dB audio distortion, as measured by PESQ scores ≥4.0 and STOI ≥0.97.  
- Show ≥23% improvement in robustness against compression (MP3 128kbps) compared to AudioSeal (2024) through enhanced perceptual loss functions.  
- Enable ≤30-minute zero-shot detection of novel TTS models using watermark-robust encoders with ≥88% accuracy.  

**2. Framework Standardization**:  
- Release an open-source toolkit integrating our watermark encoder with HuggingFace's 🎭 TTS pipeline (including Tacotron2, FastSpeech2, VITS). This will standardize provenance tracking across leading TTS frameworks.  
- Propose a new benchmark evaluation (XAttnWatermark-Bench) featuring 100,000 watermarked audio samples across 20 attacks (resampling, noise, recompression, re-encoder/decoding), following the structure of the FakeSound (2024) evaluation.  

**3. Societal and Legal Impact**:  
- Enable forensic verification of audio provenance for journalism, reducing the spread of synthetic voice misinformation.  
- Support legal compliance through embeddable voice rights information, addressing requirements like the EU AI Act's transparency obligations for deepfakes.  
- Facilitate the creation of verifiable voice banking platforms where users maintain ownership proof of synthetic identities.  

This work directly addresses the ethical and technical gaps identified in the recent literature (Ethical Deployment of AI-Generated Speech, 2024). By making watermarking integral to the synthesis process, we shift from reactive detection to proactive authentication, creating a safer ecosystem for synthetic speech. The proposed framework will be evaluated through a public demo at NeurIPS 2024, showcasing verification capabilities on real-world scenarios including political speech deepfakes, AI voice cloning in customer service, and synthetic podcast generation.  

Beyond audio forensics, this research has broader implications for multimodal generative systems. The integration of steganographic techniques within diffusion processes could inspire similar approaches in text-to-music, video-to-speech, and spatial AR/VR audio generation, aligning with the core themes of the Audio Imagination workshop.