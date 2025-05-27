# Cross-Modal Watermarking for Robust AI Content Provenance in Multimodal Foundation Models

## Introduction

The proliferation of Multi-modal Foundation Models (MFMs) and AI Agents has revolutionized content creation across various modalities, including text, images, audio, and video. Models such as Sora, Stable Diffusion, and Llava can generate highly realistic content that is increasingly indistinguishable from human-created content. While these technological advancements offer unprecedented creative possibilities, they simultaneously raise significant concerns about misinformation, deepfakes, and the erosion of trust in digital content. As these models become more sophisticated and accessible, the ability to verify the provenance of AI-generated content has emerged as a critical challenge for ensuring trustworthy AI systems.

Current watermarking approaches primarily focus on single-modality scenarios (e.g., image-only or text-only watermarking) and often fail when content crosses modality boundaries—a common occurrence in modern MFMs that routinely convert between text, images, audio, and video. For instance, when a text prompt generates a video through models like Sora, traditional watermarking methods cannot reliably track this cross-modal generation process. Furthermore, existing watermarking techniques typically struggle with robustness against common transformations like compression, cropping, and format conversions that AI-generated content often undergoes in real-world scenarios.

This research aims to address these critical gaps by developing a novel Cross-Modal Watermarking (CMW) framework specifically designed for MFMs. Our approach embeds watermarks directly in the shared latent space representations that underlie various modalities in foundation models. By positioning the watermarking mechanism at this fundamental level, we can ensure that the provenance information persists regardless of the output modality or subsequent transformations. This advancement will enable reliable tracking of content back to its source model, version, and potentially even the specific generation session or prompt.

The significance of this research extends beyond academic interest. As regulatory frameworks increasingly demand transparency and accountability for AI-generated content, robust watermarking has become essential for responsible AI development and deployment. Our proposed framework addresses a fundamental technical challenge in establishing trustworthy AI systems while providing practical tools for content verification in an era of synthetic media. By enabling reliable provenance tracking across modalities, this research contributes to combating misinformation, protecting intellectual property, and fostering greater trust in AI technologies.

## Methodology

Our proposed Cross-Modal Watermarking (CMW) framework consists of four main components: (1) Unified Latent Space Watermarking, (2) Robust Watermark Embedding, (3) Cross-Modal Watermark Detection, and (4) Provenance Verification System. Below, we detail each component with specific technical approaches and evaluation methodologies.

### 1. Unified Latent Space Watermarking

We propose embedding watermarks directly into the shared latent representations that underlie the generation process in MFMs. These latent spaces serve as a "common language" across modalities in foundation models, making them ideal insertion points for watermarks that can persist regardless of the output modality.

**Watermark Structure and Encoding:**  
We will encode a provenance identifier $W$ consisting of:
- Model identifier: $\boldsymbol{M_{\text{id}}}$
- Model version: $\boldsymbol{V_{\text{id}}}$
- Generation timestamp: $\boldsymbol{T_{\text{id}}}$
- Optional session/prompt hash: $\boldsymbol{H_{\text{prompt}}}$

The complete watermark payload is defined as:
$$\boldsymbol{W} = \text{Encode}(\boldsymbol{M_{\text{id}}} \oplus \boldsymbol{V_{\text{id}}} \oplus \boldsymbol{T_{\text{id}}} \oplus \boldsymbol{H_{\text{prompt}}})$$

where $\oplus$ represents concatenation, and $\text{Encode}(\cdot)$ transforms the identifier into a binary sequence using error-correction coding (e.g., Reed-Solomon or BCH codes) to ensure robustness against bit errors.

The watermark is then encrypted using a model-specific secret key $K$:
$$\boldsymbol{W_{\text{encrypted}}} = \text{Encrypt}(\boldsymbol{W}, K)$$

This encryption prevents unauthorized watermark manipulation while enabling authorized verification.

### 2. Robust Watermark Embedding

The embedding process integrates the watermark into the latent space representations during the generative process. We propose two embedding approaches:

**Additive Latent Space Embedding:**  
For diffusion-based models, we modify the denoising process by adding the watermark signal to the latent representation at multiple denoising steps:

$$\boldsymbol{z_t'} = \boldsymbol{z_t} + \alpha \cdot \boldsymbol{G}(\boldsymbol{W_{\text{encrypted}}}, \boldsymbol{z_t})$$

where $\boldsymbol{z_t}$ is the latent representation at timestep $t$, $\alpha$ is a strength parameter controlling watermark intensity, and $\boldsymbol{G}(\cdot,\cdot)$ is a mapping function that transforms the watermark into a pattern compatible with the latent space dimensionality.

**Attention-based Embedding:**  
For transformer-based models, we inject the watermark through attention mechanisms:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \boldsymbol{B}(\boldsymbol{W_{\text{encrypted}}})\right)V$$

where $\boldsymbol{B}(\cdot)$ is a function that converts the watermark into a bias matrix compatible with the attention computation.

**Perceptual Constraints:**  
To ensure watermark imperceptibility, we impose perceptual constraints during embedding:

$$\mathcal{L}_{\text{perceptual}} = \lambda_1 \cdot \mathcal{L}_{\text{LPIPS}}(\boldsymbol{x}, \boldsymbol{x_w}) + \lambda_2 \cdot \mathcal{L}_{\text{SSIM}}(\boldsymbol{x}, \boldsymbol{x_w})$$

where $\boldsymbol{x}$ is the unwatermarked output, $\boldsymbol{x_w}$ is the watermarked output, $\mathcal{L}_{\text{LPIPS}}$ measures perceptual similarity, $\mathcal{L}_{\text{SSIM}}$ measures structural similarity, and $\lambda_1, \lambda_2$ are weighting parameters.

### 3. Cross-Modal Watermark Detection

Our detection framework aims to recover the watermark from content across different modalities. The detection process varies by modality:

**Image/Video Detector:**  
For visual content, we employ a CNN-based detector $D_{\text{visual}}$ that extracts features at multiple scales to identify watermark patterns:

$$\boldsymbol{\hat{W}_{\text{visual}}} = D_{\text{visual}}(\boldsymbol{x_w})$$

**Audio Detector:**  
For audio content, we utilize a spectrogram-based approach combined with time-frequency analysis:

$$\boldsymbol{\hat{W}_{\text{audio}}} = D_{\text{audio}}(\boldsymbol{x_w})$$

**Text Detector:**  
For textual content, we analyze linguistic patterns and statistical features:

$$\boldsymbol{\hat{W}_{\text{text}}} = D_{\text{text}}(\boldsymbol{x_w})$$

**Unified Cross-Modal Detection:**  
We combine these modality-specific detectors into a unified framework using a fusion mechanism:

$$\boldsymbol{\hat{W}} = F(\boldsymbol{\hat{W}_{\text{visual}}}, \boldsymbol{\hat{W}_{\text{audio}}}, \boldsymbol{\hat{W}_{\text{text}}})$$

where $F(\cdot)$ is a fusion function that prioritizes the most reliable detection signals across available modalities.

Once the encrypted watermark is extracted, it is decrypted using the model's public verification key:

$$\boldsymbol{\hat{W}_{\text{decrypted}}} = \text{Decrypt}(\boldsymbol{\hat{W}}, K_{\text{public}})$$

Error correction codes are then applied to recover from potential bit errors:

$$\boldsymbol{W_{\text{recovered}}} = \text{ErrorCorrect}(\boldsymbol{\hat{W}_{\text{decrypted}}})$$

### 4. Provenance Verification System

We will develop a complete verification system that enables:

1. **Content Authentication:** Verifying if content originated from a specific model
2. **Source Identification:** Determining which model generated the content
3. **Tampering Detection:** Identifying if content has been modified since generation
4. **Confidence Scoring:** Providing a reliability score for detection results

The verification process follows this algorithm:

```
Algorithm: Cross-Modal Provenance Verification
Input: Content x of any modality (image, video, audio, text)
Output: Provenance information and confidence score

1. Detect modality type of x
2. Apply appropriate detector D_modality to extract W_hat
3. Decrypt W_hat to obtain W_decrypted
4. Apply error correction to obtain W_recovered
5. Parse W_recovered to extract M_id, V_id, T_id, H_prompt
6. Verify extracted information against database of known models
7. Calculate confidence score based on bit error rate and detection strength
8. Return provenance information and confidence score
```

### Experimental Design and Evaluation

We will evaluate the CMW framework through comprehensive experiments focusing on four key aspects:

**1. Watermark Fidelity Assessment:**
- Perceptual metrics: PSNR, SSIM, LPIPS for images/videos
- PESQ, STOI for audio
- Human evaluation studies with n=200 participants comparing watermarked vs. non-watermarked content

**2. Cross-Modal Detection Performance:**
- We will test detection across all modality combinations (e.g., text-to-image, image-to-video, etc.)
- Metrics: Detection accuracy, precision, recall, F1-score
- Bit error rate (BER) across recovered watermarks

**3. Robustness Against Transformations:**
We will test watermark persistence against:
- Compression (JPEG, H.264, MP3, etc.)
- Scaling and cropping (25%, 50%, 75% of original size)
- Format conversions
- Partial content extraction
- Color/audio adjustments
- Style transfer and other artistic filters

**4. Security Against Removal Attacks:**
- White-box attacks (attacker knows watermarking algorithm)
- Black-box attacks (attacker doesn't know algorithm details)
- Adversarial perturbation attacks
- Watermark overwriting attempts

**Dataset Creation:**
We will create a diverse evaluation dataset containing:
- 10,000 text-to-image generations
- 5,000 text-to-video generations
- 5,000 image-to-image transformations
- 3,000 text-to-audio generations
- Multiple model sources (Stable Diffusion, Sora, Llava, etc.)

**Implementation Details:**
- We will integrate our watermarking framework with three popular MFMs:
  1. Stable Diffusion (image generation)
  2. A video generation model (e.g., Sora-like architecture)
  3. A multimodal LLM (e.g., Llava-like architecture)
- Implementation will be in PyTorch with accelerated inference support

**Evaluation Metrics:**
1. Bit Accuracy Rate (BAR): Percentage of correctly recovered watermark bits
2. Watermark Recovery Rate (WRR): Percentage of samples where complete watermark is recovered
3. False Positive Rate (FPR): Rate of detecting watermarks in unwatermarked content
4. False Negative Rate (FNR): Rate of failing to detect watermarks in watermarked content
5. Content Quality Impact (CQI): Difference in quality metrics between watermarked and original content

**Comparative Baseline:**
We will compare our approach against:
- Current state-of-the-art single-modal watermarking methods
- Existing cross-modal tracking approaches
- No-watermark baseline

## Expected Outcomes & Impact

Our research on Cross-Modal Watermarking is expected to yield several significant outcomes with broad impacts on the field of trustworthy AI and content authentication:

**1. Technical Advancements**

We anticipate developing the first robust cross-modal watermarking framework specifically designed for MFMs. This framework will:
- Achieve >95% watermark recovery rates across modality transitions (e.g., text-to-image-to-video)
- Maintain watermark persistence against common transformations with >90% bit accuracy
- Demonstrate negligible impact on output quality (PSNR difference <0.5dB, SSIM difference <0.01)
- Enable reliable provenance tracking even for partial content extraction

These technical capabilities will establish new benchmarks for watermarking in multimodal AI systems and provide a foundation for future research in this domain.

**2. Societal and Regulatory Impact**

The development of reliable content provenance tools addresses critical societal challenges:
- Mitigating the spread of AI-generated misinformation by enabling content verification
- Supporting regulatory compliance with emerging AI transparency requirements
- Fostering public trust in digital content by enabling verification of content origins
- Providing forensic tools for identifying the source of potentially harmful AI-generated content

As regulatory frameworks like the EU AI Act increasingly demand transparency and accountability for AI-generated content, our framework provides a technical solution to these requirements.

**3. Industry Standards and Adoption**

Our research aims to contribute to the development of industry standards for AI content provenance:
- The proposed watermarking protocol could serve as a foundation for standardization efforts
- The open-source implementation will encourage adoption across the AI industry
- The modular design allows integration with various MFMs and content verification systems

By providing both technical specifications and reference implementations, we aim to accelerate the adoption of robust watermarking practices in commercial AI systems.

**4. Research Extensions**

This work opens several promising research directions:
- Extending the framework to additional modalities beyond text, image, audio, and video
- Exploring integration with blockchain-based verification systems for immutable provenance records
- Investigating privacy-preserving watermarking that can verify model origin without revealing user identity
- Developing adaptive watermarking techniques that optimize the embedding strategy based on content type

These extensions represent valuable opportunities for future research collaborations and follow-up studies.

**5. Limitations and Ethical Considerations**

We acknowledge potential limitations and ethical considerations:
- Watermarking systems inevitably involve a tradeoff between robustness and imperceptibility
- Determined adversaries with sufficient computational resources may still be able to remove watermarks
- There are legitimate privacy concerns regarding content tracking that must be carefully addressed
- Watermarking should not be seen as a complete solution to AI content challenges but rather as one component of a broader trustworthy AI framework

By transparently addressing these limitations, we aim to contribute to responsible development and deployment of AI provenance technologies.

In conclusion, our Cross-Modal Watermarking framework represents a significant advancement in the field of trustworthy AI, providing a robust technical solution to the critical challenge of content provenance in the era of increasingly powerful multimodal foundation models. This research directly addresses a pressing need for reliable methods to verify AI-generated content across modalities, supporting broader efforts to ensure responsible AI development and deployment.