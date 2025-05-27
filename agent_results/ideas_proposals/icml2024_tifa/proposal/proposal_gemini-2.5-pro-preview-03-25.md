Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

## **1. Title:** **LatentMark: A Unified Framework for Robust Cross-Modal Watermarking in Multi-modal Generative Models for Verifiable Content Provenance**

## **2. Introduction**

**2.1 Background**

The rapid advancement of Multi-modal Foundation Models (MFMs), encompassing Multi-modal Large Language Models (MLLMs) like LLaVA and QwenVL, and particularly Multi-modal Generative Models (MMGMs) like Sora and Stable Diffusion, marks a significant leap in artificial intelligence capabilities. These models can generate highly realistic and complex content across various modalities (text, images, audio, video) from diverse inputs, offering tremendous potential for creativity, communication, and automation. However, this power introduces unprecedented challenges related to trust and safety, aligning directly with the goals of the Trustworthy Multi-modal Foundation Models and AI Agents (TiFA) initiative.

The ability of MMGMs to generate synthetic content indistinguishable from reality poses serious risks, including the proliferation of deepfakes, sophisticated misinformation campaigns, intellectual property theft, and erosion of public trust in digital media (Jiang et al., 2023; Zhong et al., 2023). Establishing the provenance of AI-generated content – reliably identifying *whether* content was machine-generated and *which* model or entity generated it – is crucial for accountability, regulation, and mitigation of these harms.

Digital watermarking has emerged as a promising technique for embedding identifiable information directly into generated content. Recent research has explored watermarking for AI-generated images (Xu et al., 2024; Gan et al., 2025), text, and audio (Fernandez, 2025), and even for attributing content to specific users (Jiang et al., 2024). Techniques like InvisMark (Xu et al., 2024) achieve high imperceptibility and robustness for images, while GenPTW (Gan et al., 2025) integrates watermarking into the generation process of diffusion models. Others explore perceptual hashing combined with cryptography (Singhi et al., 2025) or leverage inherent model fingerprints (Desu et al., 2024). Watermarking has also been explored for protecting vision-language models themselves (Tang et al., 2023).

However, existing approaches face significant limitations, particularly in the context of advanced MMGMs capable of *cross-modal generation* (e.g., text-to-video, image-to-audio). Current methods are often modality-specific, meaning a watermark embedded for image generation might not be applicable or detectable if the same model generates video or audio. Furthermore, robustness against common content manipulations (compression, resizing, format changes) remains a major hurdle (Xu et al., 2024), and security against dedicated adversarial removal attacks is a fundamental challenge, with some theoretical work suggesting the impossibility of perfectly robust watermarks under certain assumptions (Jiang et al., 2023; Zhang et al., 2023).

**2.2 Research Gap and Proposed Idea**

There is a critical need for a unified watermarking framework specifically designed for the cross-modal capabilities of modern MMGMs. Current solutions lack a mechanism to embed a consistent, verifiable identifier that persists regardless of the output modality generated from a common seed or latent representation. Moreover, achieving practical robustness and security in this multi-modal context requires novel approaches.

This research proposes **LatentMark**, a novel watermarking framework designed to embed a unique and verifiable identifier directly into the shared *latent space* of an MMGM *before* the modality-specific decoding or generation process begins. The core idea is that by watermarking the common internal representation, the mark will subtly manifest in *any* resulting output modality (image, video, audio, text, etc.). This latent watermark will encode information such as the originating model's identity, version, potentially a session or user ID, and a timestamp. Crucially, the embedding and decoding mechanisms will be designed for resilience against standard post-processing manipulations and evaluated against potential adversarial removal attacks.

**2.3 Research Objectives**

This research aims to address the identified gaps with the following specific objectives:

1.  **Design and Develop the LatentMark Framework:** Create a conceptual and algorithmic framework for embedding watermarks into the shared latent space(s) of MMGMs. This includes defining the watermark structure, the embedding mechanism that minimally impacts generative quality, and ensuring theoretical cross-modal consistency.
2.  **Implement Robust Watermark Embedding and Decoding:** Develop concrete algorithms for injecting the watermark into latent representations (e.g., diffusion model latents, transformer embeddings) and robust decoders capable of reliably extracting the watermark from various output modalities (images, video frames, audio segments), even when the content is degraded or partially available. Error correction codes will be explored to enhance robustness.
3.  **Evaluate Cross-Modal Performance and Robustness:** Empirically validate the framework's effectiveness across different modalities generated from the same watermarked latent space. Systematically evaluate the watermark's imperceptibility (visual/audible quality) and robustness against a comprehensive suite of standard manipulations (compression, noise, cropping, format conversion) and modality-specific transformations (e.g., video codec changes, audio filtering).
4.  **Analyze Security and Scalability:** Assess the security of the LatentMark scheme against known watermark removal attacks and adaptive adversarial attacks. Analyze the computational overhead (latency, memory) introduced by the watermarking process during both model training/fine-tuning and inference.

**2.4 Significance**

This research directly contributes to the TiFA goals by enhancing the trustworthiness and accountability of advanced AI systems. A successful LatentMark framework would provide a crucial technical tool for:

*   **Combating Misinformation:** Enabling reliable identification of AI-generated content and its source, helping platforms and users discern synthetic media.
*   **Ensuring Accountability:** Providing verifiable provenance linking generated content back to specific models or potentially usage sessions, supporting responsible AI deployment and investigation of misuse.
*   **Facilitating Regulation and Policy:** Offering a technical mechanism that can support emerging regulations regarding disclosure of AI-generated content.
*   **Protecting Creators and IP:** Assisting in identifying unauthorized use of generative models trained on copyrighted data or used to mimic artistic styles.
*   **Advancing MFM Safety Research:** Providing a foundational technique for content tracing that can be integrated with other safety measures, contributing robust identifiers as called for in the TiFA topics.

By addressing the unique challenges of cross-modal generation, this work aims to provide a more future-proof solution for content provenance in the rapidly evolving landscape of multi-modal AI.

## **3. Methodology**

**3.1 Research Design**

This research will employ a constructive research design, involving the development and empirical evaluation of the LatentMark framework. The methodology comprises four main phases: framework design, implementation, evaluation, and security analysis. We will focus on MMGMs based on latent diffusion models (LDMs) and transformer architectures due to their prevalence and capability for multi-modal generation, but the principles aim for broader applicability.

**3.2 Data Collection and Preparation**

We will utilize publicly available large-scale datasets standardly used for training and evaluating MMGMs. These include:

*   **Image-Text:** LAION-5B, COCO Captions.
*   **Video-Text:** WebVid-10M, VATEX.
*   **Audio-Text:** AudioCaps, VGG-Sound.

These datasets will be used to fine-tune or train components of the watermarking system and for generating watermarked content for evaluation. We will primarily work with pre-trained open-source MMGMs (or components thereof) that allow modification of their architecture or fine-tuning, such as variants of Stable Diffusion adapted for video/audio, or potentially transformer-based models like LLaVA if exploring MLLM-based generation aspects.

**3.3 LatentMark Framework Design**

**3.3.1 Watermark Structure:**
The watermark $w$ will be a binary string of length $L$ bits (e.g., $L=64$ or $128$). It will encode:
*   Model Identifier ($ID_{model}$)
*   Model Version ($V_{model}$)
*   Timestamp ($T$)
*   Optional Session/User Identifier ($ID_{session/user}$) - subject to privacy considerations.
*   Error Correction Codes (ECC): Redundant bits added using codes like BCH or Reed-Solomon to allow recovery from partial corruption. $w = ECC_{encode}(Payload)$.

**3.3.2 Latent Space Embedding:**
Let $z$ be the latent representation in the MMGM (e.g., the output of the text encoder used as conditioning in an LDM, or intermediate representations in a transformer). The embedding function $f_{embed}$ will inject the watermark $w$ into $z$ to produce the watermarked latent $z_w$.

$$ z_w = f_{embed}(z, w, \theta_{embed}) $$

where $\theta_{embed}$ are the parameters of the embedding module. We will explore two primary approaches for $f_{embed}$:

*   **Additive Injection:** $z_w = z + \alpha \cdot E(w)$, where $E(w)$ is an embedding layer projecting the watermark bits into the latent space dimension, and $\alpha$ is a scaling factor controlling watermark strength. This is simple but might be less robust.
*   **Learnable Embedding Network:** A small neural network (e.g., MLP or convolutional layers) takes $z$ and $w$ as input and outputs $z_w$. This network can be trained jointly with the MMGM or as a separate module to optimize for robustness and imperceptibility. $z_w = Net_{embed}(z, w; \theta_{embed})$.

The embedding needs to occur at a stage *before* modality-specific processing. In LDMs, this could be applied to the conditioning information (e.g., CLIP text embeddings) or the initial noise tensor $z_T$. In autoregressive transformer models, it could modify the shared context embeddings.

**3.3.3 Cross-Modal Consistency:**
The core principle is that the *same* $z_w$ (or watermarked conditioning) is used as input for generating content in different modalities. If the MMGM uses shared latent representations or conditioning mechanisms that influence generation across modalities, embedding the watermark there ensures its conceptual presence regardless of the final output type $x_{modality}$ (where $modality \in \{image, video, audio, ...\}$).

$$ x_{modality} = Decoder_{modality}(z_w, c) $$

where $c$ is any other conditioning information.

**3.4 Watermark Decoding**

For each modality, a corresponding decoder function $g_{decode, modality}$ will be designed to extract the watermark estimate $\hat{w}$ from the generated content $x_{modality}$.

$$ \hat{w}_{payload} = ECC_{decode}(g_{decode, modality}(x_{modality}, \theta_{decode, modality})) $$

The decoder architecture will depend on the modality:

*   **Image/Video:** Convolutional Neural Network (CNN) based decoders, potentially trained to be robust to spatial transformations and compression artifacts. Similar architectures to InvisMark or GenPTW decoders but potentially adapted for latent embedding characteristics.
*   **Audio:** Decoders based on CNNs applied to spectrograms or waveform-based networks (e.g., 1D CNNs or transformers).
*   **Text:** More challenging; potentially analyze statistical properties, word embeddings distributions, or require specific generation constraints (less desirable). Focus will initially be on non-textual modalities.

The decoders might share parameters or leverage a universal architecture trained on multi-modal data to improve efficiency and cross-modal robustness. They might require access to side information (e.g., model ID to select the correct decoding key/parameters if necessary).

**3.5 Training Strategy**

If using learnable components ($Net_{embed}$, $g_{decode}$), they need to be trained. We propose an end-to-end training approach (or fine-tuning of the MMGM along with the watermark components) minimizing a composite loss function $\mathcal{L}_{total}$:

$$ \mathcal{L}_{total} = \lambda_{gen} \mathcal{L}_{gen} + \lambda_{imp} \mathcal{L}_{imp} + \lambda_{dec} \mathcal{L}_{dec} + \lambda_{rob} \mathcal{L}_{rob} $$

*   $\mathcal{L}_{gen}$: The original MMGM generation loss (e.g., diffusion loss, reconstruction loss) to ensure content quality is maintained. Applied between original $x$ and generated $x'$ from non-watermarked $z$.
*   $\mathcal{L}_{imp}$: Imperceptibility loss, measuring the difference between content generated from $z$ and $z_w$.
    *   Image/Video: $||x_{image} - x'_{image}||_2^2$ or perceptual losses like LPIPS.
    *   Audio: Mean Squared Error on spectrograms or waveform difference.
*   $\mathcal{L}_{dec}$: Watermark decoding loss, ensuring accurate extraction. Typically Binary Cross-Entropy (BCE) between the original watermark $w$ and the decoded estimate $\hat{w}$ after passing through the decoder $g_{decode}$.
    $$ \mathcal{L}_{dec} = BCE(w, g_{decode}( Decoder_{modality}(z_w) ) ) $$
*   $\mathcal{L}_{rob}$: Robustness loss, encouraging the watermark to survive transformations. The decoder loss is calculated on *transformed* versions of the generated content $T(x_{modality})$.
    $$ \mathcal{L}_{rob} = BCE(w, g_{decode}( T(Decoder_{modality}(z_w)) ) ) $$
    where $T$ is a randomly sampled transformation from a predefined set (e.g., JPEG compression, noise addition, cropping).

The hyperparameters $\lambda_{gen}, \lambda_{imp}, \lambda_{dec}, \lambda_{rob}$ balance the trade-offs between generative quality, imperceptibility, and watermark robustness.

**3.6 Experimental Design and Evaluation**

**3.6.1 Baselines:**
We will compare LatentMark against state-of-the-art modality-specific watermarking methods:
*   Image: InvisMark (Xu et al., 2024), GenPTW (Gan et al., 2025).
*   Multi-modal (if applicable/reproducible): Methods from Fernandez (2025).
*   No Watermark: As a control.

**3.6.2 Evaluation Setup:**
*   Generate datasets of watermarked content across multiple modalities (e.g., generate images, short video clips, audio samples) using MMGMs embedded with LatentMark and baseline methods.
*   Apply a standard battery of transformations simulating real-world distortions:
    *   Images: JPEG compression (various quality factors), resizing, cropping, Gaussian noise, blurring.
    *   Video: H.264/H.265 compression (various bitrates), frame dropping, resizing, spatial/temporal cropping.
    *   Audio: MP3/AAC compression, noise addition, resampling, filtering.
    *   Cross-format conversions where applicable.

**3.6.3 Evaluation Metrics:**

*   **Imperceptibility:**
    *   Image/Video: PSNR, SSIM, LPIPS. Human evaluation studies (e.g., A/B testing, Mean Opinion Score - MOS).
    *   Audio: Signal-to-Noise Ratio (SNR), Perceptual Evaluation of Speech Quality (PESQ) for speech, Objective Difference Grade (ODG) for general audio. Human MOS.
*   **Robustness:**
    *   Watermark Bit Accuracy (WBA) or Bit Error Rate (BER) after applying each transformation. Measured as the percentage of correctly recovered payload bits.
    *   Detection Rate: Accuracy of detecting the presence/absence of a watermark.
*   **Cross-Modal Consistency:**
    *   Generate content in modalities A and B from the *same* $z_w$. Measure WBA for both $x_A$ and $x_B$. Compare the extracted watermarks ($\hat{w}_A$ vs $\hat{w}_B$) to verify they match the original $w$.
*   **Payload Capacity:** Maximum number of bits ($L$) embeddable while maintaining acceptable imperceptibility and robustness thresholds.
*   **Computational Overhead:**
    *   Increase in MMGM training time (if fine-tuning).
    *   Increase in inference latency (ms per image/video frame/audio second).
    *   Increase in model size (MB).
    *   Decoding time.
*   **Security:**
    *   Evaluate against known watermark removal attacks (e.g., StirMark for images, diffusion-based purification, intentionally adding noise).
    *   Develop adaptive attacks targeting the specific LatentMark mechanism (e.g., gradient-based attacks to minimize decoder output while preserving quality). Evaluate Attack Success Rate vs. Quality Degradation (e.g., measure PSNR/SSIM/LPIPS of attacked content). Acknowledge theoretical limits (Zhang et al., 2023) and focus on practical security thresholds.

## **4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**

1.  **A Novel Latent-Space Cross-Modal Watermarking Framework (LatentMark):** A fully specified architecture and set of algorithms for embedding and decoding watermarks within the latent space of MMGMs, designed for cross-modal consistency.
2.  **Implementation and Models:** Open-source implementation of the LatentMark embedding and decoding modules, potentially integrated with one or more existing open-source MMGMs. Pre-trained watermarking components and potentially sample watermarked MMGMs.
3.  **Comprehensive Evaluation Results:** Rigorous benchmarking results quantifying LatentMark's performance across multiple modalities in terms of imperceptibility, robustness against a wide range of manipulations, payload capacity, computational overhead, and cross-modal consistency. Comparison against state-of-the-art baselines.
4.  **Security Analysis Report:** An assessment of the framework's vulnerability to existing and novel watermark removal attacks, providing insights into its practical security limits and potential countermeasures. This will explicitly address the challenges raised by works like Zhang et al. (2023) by focusing on the *practical difficulty* and *quality trade-offs* involved in removing the watermark.
5.  **Publications and Dissemination:** High-quality publications in leading AI/ML security and multimedia conferences (e.g., NeurIPS, ICML, CVPR, ACM MM, IEEE S&P) and journals. Presentations and potential tutorials to disseminate the findings.

**4.2 Impact**

This research is expected to have significant impact aligned with the TiFA program's objectives:

*   **Enhanced Trustworthiness of MFMs:** By providing a reliable method for content provenance, LatentMark will contribute directly to building more trustworthy MFMs and AI Agents. Knowing the origin of content is a cornerstone of trust.
*   **Mitigation of Malicious Use:** The ability to trace generated content back to its source model can deter malicious actors and aid in the investigation of deepfakes, misinformation, and other harms facilitated by MMGMs.
*   **Support for AI Governance and Regulation:** LatentMark offers a concrete technical solution that can underpin policies requiring disclosure and traceability of AI-generated media, providing regulators with a verifiable mechanism.
*   **Advancement of Watermarking Science:** This work pushes the boundaries of digital watermarking by tackling the complex challenges of cross-modal generation and latent space manipulation, potentially inspiring new research directions in robust representation learning and multi-modal security.
*   **Strengthening the AI Safety Ecosystem:** By providing a practical tool for identifying AI-generated content, this research contributes a vital component to the broader ecosystem of AI safety and alignment techniques, complementing efforts in detection, monitoring, and control.

While acknowledging the theoretical impossibility of *perfectly* secure watermarking against all powerful adversaries (Zhang et al., 2023), LatentMark aims to raise the bar significantly, making watermark removal computationally expensive and/or result in noticeable quality degradation, thus providing a strong *practical* defense for verifying content provenance in the age of advanced multi-modal generative models. This contribution is crucial for fostering responsible innovation and deployment of these powerful technologies.

---