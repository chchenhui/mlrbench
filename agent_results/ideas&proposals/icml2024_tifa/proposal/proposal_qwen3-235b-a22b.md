# Cross-Modal Watermarking for Verifiable AI-Generated Content Provenance  

## Introduction  

### Background  
The rapid advancement of Multi-modal Generative Models (MMGMs) such as Sora, Stable Diffusion, and Latte has enabled the creation of highly realistic synthetic content across text, images, video, and audio. While these models unlock transformative applications, their ability to generate indistinguishable content from reality poses risks including misinformation, intellectual property disputes, and erosion of public trust. A critical challenge lies in robustly identifying and tracing AI-generated content to its source model or context. Existing watermarking techniques focus primarily on single-modal content (e.g., images) and falter when applied to cross-modal outputs (e.g., text-to-video) or under adversarial manipulations.  

### Research Objectives  
This proposal aims to design, implement, and evaluate a cross-modal watermarking framework to address the following limitations in current methods:  
1. **Unified Embedding**: Embed watermarks into latent representations of **multiple modalities** (text, image, video, audio) during generation, enabling consistent provenance tracking despite modality transformations.  
2. **Robustness**: Ensure watermark persistence under realistic post-processing (compression, cropping, format changes) and evade removal through adversarial perturbations.  
3. **Scalability**: Integrate watermarking into MMGM pipelines without compromising generation quality or inference efficiency.  
4. **Verifiability**: Develop decoders that extract watermarks from **partial or degraded content**, linking outputs to model versions, user sessions, or contextual prompts.  

### Significance  
A reliable cross-modal watermarking solution will:  
- **Combat Misinformation**: Enable fact-checkers to verify content authenticity in multi-modal media.  
- **Ensure Accountability**: Link AI-generated content to creators or systems for legal and ethical compliance.  
- **Support Regulatory Frameworks**: Align with emerging policies (e.g., EU AI Act) mandating AI output disclosure.  
- **Advance AI Safety**: Provide a foundation for auditing and governing large-scale MMGM deployments.  

---

## Methodology  

### Data Collection and Datasets  
To train and evaluate our framework:  
1. **MMGM-generated Data**: Use diverse models like Stable Diffusion (text-to-image), Sora (text-to-video), and MusicLM (text-to-audio) to synthesize 100,000+ cross-modal samples spanning languages, styles, and content domains.  
2. **Real-world Media**: Acquire unwatermarked LAION-400M images, Kinetics-700 videos, and Common Voice audio for robustness benchmarking.  
3. **Post-Processing Pipeline**: Apply 12 common transformations per sample: JPEG compression (Q=50–90), Gaussian blur (σ=1–5), cropping (center/edge), chroma subsampling (4:2:0, 4:4:4), MP3 encoding (128–320kbps), and text token recombination (e.g., synonym swaps).  

### Algorithmic Framework  
Our approach embeds watermarks in **latent spaces** before content generation, leveraging the forward/inverse latent diffusion formulation of modern MMGMs.  

#### Watermark Embedding  
Let $ \mathcal{T}(\cdot) $ denote a modality-specific encoder mapping content to a latent space $ \mathbf{Z} \in \mathbb{R}^{d \times n} $, where $ d $ is dimensionality and $ n $ the sequence length. During generation, we inject a watermark code $ \mathbf{w}_t \in \{0,1\}^k $ specific to:  
- **Model Identifier**: A hash of the MMGM’s training checkpoint (e.g., `Sora-v1.0`).  
- **Session Context**: User query metadata or timestamp for fine-grained tracing.  

The watermarked latent is computed as:  
$$
\mathbf{Z}' = \mathcal{T}(x) + \alpha \cdot \mathbf{W} \cdot \text{Expand}(\mathbf{w}_t),
$$
where $ \mathbf{W} \in \mathbb{R}^{d \times k} $ is a learnable watermark projection matrix, $ \alpha $ controls embedding strength ($ \alpha \in [0.01, 0.1] $), and $ \text{Expand}(\cdot) $ reshapes $ \mathbf{w}_t $ to match $ \mathbf{Z} $’s spatial/temporal structure (e.g., replicating tokens or tiling spectrogram subbands).  

#### Modality-Specific Injection  
- **Text**: Modify hidden states in the attention layer during token generation:  
  $$
  \mathbf{H}_i' = \mathbf{H}_i + \alpha \cdot \mathbf{W} \cdot \text{GELU}(\mathbf{W}_b \mathbf{w}_t),
  $$
  where $ \mathbf{W}_b \in \mathbb{R}^{k \times k} $ projects $ \mathbf{w}_t $ into a bottleneck representation.  
- **Image/Video**: Integrate with latent diffusion’s noise prediction step (denoising step $ t $):  
  For a denoised latent $ \mathbf{Z}_t $, compute:  
  $$
  \mathbf{Z}_t' = \mathbf{Z}_t + \alpha \cdot \mathbf{W} \cdot \text{FFT}(\mathbf{w}_t),
  $$
  where FFT transforms the watermark into frequency space to resist spatial transformations.  
- **Audio**: Encode watermarks into spectrogram magnitude subbands using spectral masking techniques inspired by Wavenet.  

#### Watermark Extraction  
Given watermarked content $ x' $, the decoder $ \mathcal{D}(\cdot) $ - a lightweight Transformer-based network - processes $ \mathbf{Z}' = \mathcal{T}(x') $ to reconstruct $ \mathbf{w}_t $:  
$$
\hat{\mathbf{w}}_t = \sigma(\mathbf{W}_d \cdot \text{MLP}(\mathbf{Z}') + \mathbf{b}_d),
$$
where $ \sigma $ is the sigmoid function. Binary watermarks are thresholded at 0.5.  

### Training Protocol  
1. **Multi-Task Loss Function**:  
   - **Perceptual-preserving Loss**: $ \mathcal{L}_{\text{FID}} $ between $ x $ and $ x' $ to maintain visual/textual fidelity.  
   - **Watermark Reconstruction Loss**: Binary cross-entropy $ \mathcal{L}_{\text{BCE}}(\mathbf{w}_t, \hat{\mathbf{w}}_t) $.  
   - **Adversarial Loss**: Discriminator $ \mathcal{E} $ trained to distinguish real/watermarked samples, minimizing:  
     $$
     \mathcal{L}_{\text{GAN}} = \mathbb{E}_{x'}[\log \mathcal{E}(x')] + \mathbb{E}_x[\log(1 - \mathcal{E}(x))].
     $$
   - **Total Loss**:  
     $$
     \mathcal{L} = \lambda_1 \mathcal{L}_{\text{FID}} + \lambda_2 \mathcal{L}_{\text{BCE}} + \lambda_3 \mathcal{L}_{\text{GAN}},
     $$
     with weights $ \lambda_1=1, \lambda_2=5, \lambda_3=0.01 $.  

2. **Robustness Augmentation**: During training, simulate adversarial attacks using projected gradient descent (PGD), adding small perturbations $ \delta $ to $ \mathbf{Z} $:  
   $$
   \delta = \arg\min_{\|\delta'\|_\infty \leq \epsilon} \mathcal{L}_{\text{BCE}}(\mathbf{w}_t, \mathcal{D}(\mathbf{Z}+\delta')).
   $$

### Experimental Design and Evaluation  

#### Baselines  
Compare against:  
- Single-modal methods: InvisMark (image), MusicFM (audio), BERTWatermark (text).  
- Cross-modal baselines: VLPMarker, GenPTW+ LSTM for multi-modal.  

#### Metrics  
1. **Bit Accuracy (BA)**: Proportion of watermark bits correctly extracted.  
2. **Minimum Bit Accuracy (Min BA)**: For asymmetric watermarks (e.g., 1s feared more than 0s).  
3. **Perceptual Quality**: Learned Perceptual Image Patch Similarity (LPIPS), Flesch-Kincaid (text), MUSHRA (audio).  
4. **Robustness**: BA degradation after transformations (e.g., BA from 98% to 72% post-cropping).  
5. **False Positive Rate (FPR)**: On real data, probability of incorrectly detecting a watermark.  

#### Statistical Analysis  
- **Cross-Modal Resilience Test**: Measure BA for watermarks embedded in text but extracted from image/video outputs (e.g., prompt-to-video scenarios).  
- **Attack Resistance**: Evaluate defeat rates of recent attacks [7] on our method versus baselines.  
- **Ablation Study**: Analyze impact of $ \alpha $, transform domains (spatial vs. frequency), and attention mechanisms in the decoder.  

#### Implementation Details  
- **Framework**: PyTorch, HuggingFace Transformers, fairseq.  
- **Model Sizes**: Use base-sized MMGMs (e.g., Stable Diffusion 2.1) for reproducibility.  
- **Ethics**: Apply data filtering for NSFW content and bias mitigation in watermarks.  

---

## Expected Outcomes & Impact  

### Technical Outcomes  
1. **Cross-Modal Watermarking Framework**: First implementation of a unified watermarking method tested across text, image, video, and audio, with average BA >90% and LPIPS <0.03.  
2. **Benchmark Dataset**: Release of **Watermark-X**, a dataset of 200,000+ cross-modal samples with post-processing metadata and ground-truth watermarks.  
3. **Open Source Tools**: Publish libraries for embedder/extractor integration into MMGM pipelines under permissive licenses.  

### Scientific Impact  
1. **Robustness Frontier**: Demonstrate practical existence of watermarking schemes in regimes where [8] predicted impossibility by carefully balancing $ \alpha $, architecture depth, and adversarial training.  
2. **Latent Space Navigation**: Reveal dependencies between watermark persistence and MMGM training dynamics (e.g., diffusion step sensitivity).  

### Societal and Regulatory Impact  
1. **Media Accountability**: Equip platforms like Meta and Alibaba Cloud with open-source tools for verifying MMGM-generated content in feeds (e.g., YouTube Shorts watermarked videos).  
2. **Policy Compliance**: Provide boilerplate APIs for conforming to the EU AI Act Article 52’s synthetic media disclosure requirements.  
3. **Public Awareness**: Collaborate with INJECT project to develop browser plugins that visualize watermarks in social media content.  

### Commercialization  
Pursue non-exclusive licensing partnerships with MMGM-as-a-service providers (e.g., AWS Bedrock, Azure AI) to institutionalize content provenance in APIs.  

By addressing the **cross-modal and robustness gap**, this work will anchor the next generation of trustworthy MMGMs, mitigating existential threats to information ecosystems while enabling ethical innovation.