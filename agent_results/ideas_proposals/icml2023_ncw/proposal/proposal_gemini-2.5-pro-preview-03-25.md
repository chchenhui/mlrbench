# Research Proposal

## 1. Title

**FlowCodec: Continuous-Flow Neural Compression with Information-Bottleneck Guarantees**

## 2. Introduction

### 2.1 Background

The exponential growth of digital data across diverse domains, from high-resolution imagery and video streams to large-scale scientific simulations and the parameters of massive foundation models, has created an unprecedented need for efficient data compression techniques. Classical compression algorithms (e.g., JPEG, H.264) have been highly optimized but are reaching performance plateaus, particularly for complex data modalities and very low bitrates where perceptual quality is paramount.

In recent years, learned neural compression has emerged as a powerful paradigm, leveraging deep generative models to achieve state-of-the-art performance on various data types like images, videos, and audio. These methods typically employ an autoencoder architecture where an encoder maps the input data to a latent representation, which is then quantized and entropy-coded for transmission or storage. A decoder reconstructs the data from the quantized latents. Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) combined with quantization have been particularly successful.

However, the reliance on discrete quantization introduces significant challenges. Firstly, quantization is inherently non-differentiable, breaking the end-to-end differentiability desirable for gradient-based optimization. This necessitates the use of proxy functions or straight-through estimators during training, which can lead to suboptimal performance and training instability. Secondly, the discrete nature of the latent space complicates theoretical analysis, making it difficult to derive tight bounds on rate-distortion (RD) performance or establish clear connections to information-theoretic principles like the Information Bottleneck (IB). Thirdly, quantization inevitably introduces information loss that might be suboptimal, potentially limiting reconstruction fidelity, especially for fine details and textures.

Normalizing Flows (NFs) offer a compelling alternative. NFs are a class of deep generative models that learn an invertible transformation (a diffeomorphism) between a simple base distribution (e.g., Gaussian) and a complex target distribution. Their key advantages include exact likelihood computation, efficient sampling, and, crucially, invertibility with a tractable Jacobian determinant. These properties make them suitable for density estimation, generative modeling, and potentially compression. Furthermore, the Information Bottleneck (IB) principle provides a theoretical framework for optimal representation learning, seeking representations that retain maximal relevant information about the input while being maximally compressive regarding the input itself.

### 2.2 Problem Statement

The core problem addressed by this research is the limitation imposed by discrete quantization in state-of-the-art learned neural compression systems. Specifically, quantization:
1.  **Hinders End-to-End Optimization:** Non-differentiability requires approximations, potentially leading to suboptimal RD trade-offs.
2.  **Complicates Theoretical Analysis:** Makes it difficult to establish rigorous connections between the learned model, information theory, and achievable RD limits.
3.  **May Introduce Irreversible Information Loss:** Discretization can discard subtle information that might be crucial for high-fidelity reconstruction, especially at low bitrates.
4.  **Offers Coarse Rate Control:** Rate control is typically achieved by adjusting quantization step sizes or entropy coding parameters, which may not offer the smooth, fine-grained control desirable in many applications.

### 2.3 Proposed Solution: FlowCodec

We propose **FlowCodec**, a novel neural compression framework that circumvents quantization by leveraging continuous normalizing flows and explicitly incorporating the Information Bottleneck principle. The core idea is to replace the quantizer with a continuous latent space modeled by normalizing flows.

FlowCodec consists of an encoder and a decoder, both potentially incorporating flow-based architectures. The encoder maps an input $x$ to a continuous latent variable $z$ whose conditional distribution $q(z|x)$ is modeled tractably, often using a flow transformation applied to a base distribution whose parameters are predicted from $x$. Instead of quantizing $z$, we operate directly in the continuous domain. To control the information content (and thus the bitrate) of the latent representation, we introduce an explicit information bottleneck via a KL divergence penalty in the training objective. The objective function takes the form of a rate-distortion Lagrangian:

$$L = \mathbb{E}_{p(x)} \mathbb{E}_{q(z|x)} [d(x, \hat{x})] + \beta \cdot \mathbb{E}_{p(x)} [KL(q(z|x) || p(z))]$$

where:
*   $x$ is the input data drawn from distribution $p(x)$.
*   $\hat{x} = g_\phi(z)$ is the reconstruction produced by the decoder $g_\phi$ from the latent variable $z$.
*   $z \sim q(z|x)$ is sampled from the conditional latent distribution modeled by the encoder $f_\theta$.
*   $d(x, \hat{x})$ is a distortion measure (e.g., Mean Squared Error (MSE), MS-SSIM).
*   $p(z)$ is a prior distribution over the latent space, typically chosen to be simple (e.g., standard Gaussian) or a flexible distribution also modeled by a normalizing flow.
*   $KL(q(z|x) || p(z))$ is the Kullback-Leibler divergence between the conditional latent distribution and the prior. This term acts as a regularizer, encouraging $q(z|x)$ to stay close to the prior $p(z)$, effectively limiting the amount of information $z$ carries about $x$.
*   $\beta$ is a Lagrange multiplier that balances the trade-off between distortion (reconstruction quality) and rate (information content/bitrate).

This formulation results in a fully differentiable pipeline trainable end-to-end. The KL divergence term provides a theoretically grounded proxy for the coding rate, directly linked to the IB principle [10]. By varying $\beta$, we can smoothly navigate the rate-distortion curve. We hypothesize that avoiding explicit quantization will allow FlowCodec to achieve sharper reconstructions and potentially better RD performance, particularly in regimes where fine details matter. The continuous nature and tractable likelihoods also pave the way for deriving theoretical performance bounds and extending the framework to related tasks like joint source-channel coding (JSCC).

### 2.4 Research Objectives

The primary objectives of this research are:

1.  **Develop the FlowCodec Framework:** Design and implement the encoder-decoder architecture based on normalizing flows, carefully defining the conditional latent distribution $q(z|x)$ and the prior $p(z)$. Investigate suitable NF architectures (e.g., coupling flows [RealNVP, Glow], autoregressive flows [MAF], continuous flows [FFJORD, OT-Flow [8]]) for this task.
2.  **Implement and Optimize the Training Procedure:** Implement the Lagrangian objective function $L$ and develop robust training protocols, including strategies for selecting and potentially annealing the trade-off parameter $\beta$. Explore optimization techniques suitable for deep flow-based models.
3.  **Theoretically Analyze FlowCodec:** Derive theoretical upper bounds on the achievable rate based on the KL divergence term. Investigate the relationship between $\beta$, the KL divergence, the actual bitrate achievable with practical entropy coding on discretized latents (if needed post-training), and information-theoretic RD limits, potentially leveraging variational inference or f-divergence perspectives.
4.  **Empirically Evaluate FlowCodec:** Benchmark FlowCodec against state-of-the-art neural compression methods (both VQ-based and potentially other flow-based methods like [9]) on standard image and potentially video datasets (e.g., Kodak, CLIC, Tecnick, Vimeo90K). Evaluate performance using standard metrics: Rate-Distortion curves (PSNR vs. bpp, MS-SSIM vs. bpp), computational complexity (encoding/decoding latency), and qualitative visual assessment.
5.  **Explore Extensions:** Investigate the potential of FlowCodec for joint source-channel coding (JSCC) by composing the encoder flow with a flow representing channel noise, enabling end-to-end optimization for noisy channels.

### 2.5 Significance

This research holds significant potential for advancing the field of neural compression and its intersection with information theory:

*   **Improved Performance:** FlowCodec may achieve superior rate-distortion performance, especially regarding perceptual quality and reconstruction of fine details, by avoiding information loss inherent in quantization.
*   **Theoretical Grounding:** Provides a compression framework with stronger connections to information theory (IB principle) and allows for more rigorous theoretical analysis, potentially leading to performance guarantees or tighter bounds.
*   **End-to-End Differentiability:** Simplifies training and allows for more effective optimization compared to methods relying on quantization proxies.
*   **Smooth Rate Control:** Offers fine-grained control over the RD trade-off via the continuous parameter $\beta$.
*   **New Capabilities:** The continuous and differentiable nature opens possibilities for novel applications, such as differentiable compression layers within larger systems or robust JSCC schemes.
*   **Addressing Workshop Themes:** Directly contributes to the workshop's focus on improving learned compression, exploring theoretical limits, integrating information-theoretic principles, and potentially accelerating inference (due to potentially simpler decoding paths compared to complex entropy models).

## 3. Methodology

### 3.1 Theoretical Framework and Model Architecture

**FlowCodec Architecture:**
The proposed FlowCodec employs an encoder-decoder structure.

*   **Encoder ($f_\theta$):** The encoder's role is to define the conditional distribution $q(z|x)$. It will likely consist of a convolutional neural network (CNN) backbone to extract features from the input $x$, followed by a conditional normalizing flow. The CNN output will parameterize the base distribution and/or the transformations within the flow. Given input $x$, the encoder defines a transformation $T_{\theta, x}: \mathcal{U} \to \mathcal{Z}$ such that if $u \sim p_U(u)$ (a simple base distribution, e.g., Gaussian $\mathcal{N}(0, I)$), then $z = T_{\theta, x}(u)$ follows the desired $q(z|x)$. The log-density is tractable:
    $$\log q(z|x) = \log p_U(T_{\theta, x}^{-1}(z)) + \log |\det J_{T_{\theta, x}^{-1}}(z)|$$
    Specific flow architectures like RealNVP, Glow, or residual flows conditioned on $x$ will be investigated. We may draw inspiration from recent advances like entropy-informed shuffling [5] or optimal transport regularization [8] to enhance expressivity and efficiency.

*   **Latent Space ($\mathcal{Z}$):** This is a continuous D-dimensional Euclidean space, $\mathcal{Z} \subseteq \mathbb{R}^D$. No explicit quantization is performed during training or inference.

*   **Prior ($p(z)$):** The prior distribution over the latent space. We will start with a simple standard multivariate Gaussian $p(z) = \mathcal{N}(0, I)$. We will also explore using a more flexible prior modeled by another (unconditional) normalizing flow trained jointly or pre-trained on representative latent samples. This allows the model to learn a more efficient latent structure.

*   **Decoder ($g_\phi$):** The decoder maps the continuous latent variable $z$ back to the reconstructed data space $\hat{x}$. It will likely be structured symmetrically to the encoder's CNN part, using transposed convolutions or similar upsampling layers. $g_\phi$ is trained to minimize the expected distortion $d(x, g_\phi(z))$ where $z \sim q(z|x)$.

**Objective Function:**
The training objective is the rate-distortion Lagrangian:
$$L(\theta, \phi; \beta) = \mathbb{E}_{p(x)} \mathbb{E}_{q(z|x)} [d(x, g_\phi(z))] + \beta \cdot \mathbb{E}_{p(x)} [KL(q(z|x) || p(z))]$$
The distortion $d(x, \hat{x})$ will primarily be MSE or MS-SSIM for images/videos. The KL divergence term is computed using the tractable densities of $q(z|x)$ and $p(z)$:
$$KL(q(z|x) || p(z)) = \int q(z|x) \log \frac{q(z|x)}{p(z)} dz = \mathbb{E}_{q(z|x)} [\log q(z|x) - \log p(z)]$$
In practice, both expectations (over $p(x)$ and $q(z|x)$) will be estimated using Monte Carlo sampling with mini-batches during training.

**"Dequantization Noise" Aspect:**
The original idea mentions injecting Gaussian noise similar to dequantization noise. In our flow-based setup, this stochasticity is naturally handled by the definition of $q(z|x)$, which models a distribution rather than a deterministic mapping. If the encoder deterministically maps $x$ to some features $y=h_\theta(x)$, we could define $q(z|x)$ as $z = y + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$. In this case, the KL term becomes $KL(\mathcal{N}(y, \sigma^2 I) || p(z))$. Alternatively, the stochasticity arises directly from sampling the base noise $u$ which is then transformed by the conditional flow $T_{\theta, x}$. We will primarily explore the latter, more general formulation.

### 3.2 Training Procedure

*   **Optimization:** The parameters $\theta$ and $\phi$ of the encoder and decoder will be optimized jointly by minimizing the objective $L$ using stochastic gradient descent (SGD) or variants like Adam/AdamW.
*   **Rate-Distortion Trade-off:** The parameter $\beta$ controls the emphasis between minimizing distortion and minimizing the KL divergence (rate). To obtain models operating at different points on the RD curve, we will train separate models for a range of $\beta$ values (e.g., logarithmically spaced). Alternatively, we might explore techniques for training a single model that can adapt to different $\beta$ values at inference time, although this is more challenging.
*   **Numerical Stability:** Training deep normalizing flows can be challenging. We will employ techniques like gradient clipping, careful initialization, activation normalization, and potentially use more stable flow architectures (e.g., residual flows).
*   **Bitrate Estimation:** The KL divergence term $R_{KL} = KL(q(z|x) || p(z))$ serves as a proxy for the bitrate during training. For actual compression, the continuous latent $z$ would need to be discretized finely and then entropy coded (e.g., using Arithmetic Coding based on the learned prior $p(z)$ or a more sophisticated entropy model). However, the primary evaluation will focus on the theoretical rate $R_{KL}$ versus distortion, acknowledging that practical coding incurs overhead. We will also investigate methods for directly estimating the achievable bitrate from $q(z|x)$ and $p(z)$ using techniques related to variational inference lower bounds (ELBO) on mutual information, potentially linking $\beta$ to bounds derived from f-divergences as suggested.

### 3.3 Data Collection and Preparation

*   **Datasets:** We will use standard benchmark datasets for image compression:
    *   Kodak Lossless True Color Image Suite (24 images)
    *   Tecnick Dataset (subset, e.g., 100 images)
    *   CLIC (Challenge on Learned Image Compression) professional validation dataset (e.g., CLIC 2020).
    For video compression (if explored):
    *   UVG dataset
    *   Vimeo90K septuplet dataset for training.
*   **Preprocessing:** Images will be normalized (e.g., to [0, 1] or [-1, 1]). For training, images might be randomly cropped to fixed-size patches (e.g., 256x256) to form mini-batches. Videos will be processed frame by frame or using small temporal windows.

### 3.4 Experimental Design and Validation

1.  **Baseline Methods:** We will compare FlowCodec against:
    *   Classical codecs: JPEG, JPEG2000 (for images), H.265/HEVC (for video reference).
    *   State-of-the-art learned compression methods using VQ: Methods based on Ballé et al. (e.g., hyperpriors), Minnen et al. (context-adaptive entropy models), Cheng et al. (attention). Specific implementations from CompressAI library will be used.
    *   Existing flow-based compression methods: Primarily "Lossy Image Compression with Normalizing Flows" [9] to highlight the contribution of the explicit IB formulation and potential architectural differences.

2.  **Evaluation Metrics:**
    *   **Rate:** Bits per pixel (bpp) for images, or kilobits per second (kbps) for videos. Rate will be primarily measured by the KL divergence term $R_{KL}$. We will also report estimated achievable bitrates via discretization and standard entropy coding simulation if feasible, to connect $R_{KL}$ to practical rates.
    *   **Distortion:** Peak Signal-to-Noise Ratio (PSNR) and Multiscale Structural Similarity Index Measure (MS-SSIM).
    *   **Visualization:** Side-by-side comparisons of reconstructed images/videos at similar bitrates. Residual maps (difference between original and reconstruction).
    *   **Computational Cost:** Encoding and decoding time (latency) on specific hardware (CPU/GPU). Model size (number of parameters).

3.  **Rate-Distortion Curves:** For each method, we will generate RD curves by plotting distortion (PSNR/MS-SSIM) against rate (bpp/kbps) for multiple operating points (achieved by varying $\beta$ for FlowCodec, or quality parameters for other codecs).

4.  **Ablation Studies:** To understand the contribution of different components:
    *   Impact of NF architecture choice (e.g., RealNVP vs. FFJORD).
    *   Impact of prior complexity (Standard Gaussian vs. learned flow prior).
    *   Effect of the dimensionality of the latent space $D$.
    *   Sensitivity analysis with respect to hyperparameters (e.g., learning rate, batch size).

5.  **Joint Source-Channel Coding (JSCC) Exploration:**
    *   Define a simple parametric noise channel model (e.g., Additive White Gaussian Noise - AWGN channel) $p(z'|z)$.
    *   Potentially model the channel using another flow $c_\psi$. The received latent is $z' = c_\psi(z, \nu)$, where $\nu$ is channel noise.
    *   Modify the objective to optimize for end-to-end reconstruction quality under channel noise:
        $$L_{JSCC} = \mathbb{E}_{p(x)} \mathbb{E}_{q(z|x)} \mathbb{E}_{p(z'|z)} [d(x, g_\phi(z'))] + \beta \cdot \mathbb{E}_{p(x)} [KL(q(z|x) || p(z))]$$
        (Note: the rate term might need modification in JSCC).
    *   Evaluate robustness to varying channel conditions (e.g., different Signal-to-Noise Ratios - SNRs). Compare against separate source and channel coding.

## 4. Expected Outcomes & Impact

### 4.1 Expected Outcomes

1.  **A Novel Continuous Neural Compression Model (FlowCodec):** A complete implementation of the FlowCodec framework leveraging normalizing flows and the Information Bottleneck principle, capable of compressing images and potentially videos.
2.  **State-of-the-Art RD Performance:** Empirical results demonstrating that FlowCodec achieves competitive, and potentially superior, rate-distortion performance compared to established VQ-based neural compressors, particularly regarding perceptual quality (sharpness, texture preservation) at various bitrates.
3.  **Theoretical Insights:** Derivation and analysis of the connection between the KL divergence term, the parameter $\beta$, achievable coding rates, and theoretical RD limits. Validation of the IB principle's effectiveness as a regularization mechanism for compression.
4.  **Efficient Implementation:** Benchmarking of computational latency and model complexity, aiming for practical applicability comparable to or better than some existing learned methods (potentially lower latency due to avoiding complex discrete entropy coding steps during decoding).
5.  **Demonstration of Extensibility:** Initial results showcasing the viability of extending FlowCodec to JSCC scenarios, highlighting the benefits of the continuous, differentiable latent space for end-to-end optimization over noisy channels.
6.  **Open Source Contribution:** Release of the FlowCodec implementation and trained models to facilitate reproducibility and further research.

### 4.2 Impact

This research is expected to have a significant impact on the fields of machine learning, data compression, and information theory:

*   **Advancement in Neural Compression:** Provides a powerful, theoretically motivated alternative to quantization-based methods, potentially pushing the state-of-the-art in learned compression, especially for applications demanding high perceptual fidelity.
*   **Bridging Theory and Practice:** Strengthens the connection between information-theoretic principles (like IB) and practical deep learning models for compression, fulfilling a key goal of the workshop. Offers a framework where theoretical bounds are more directly related to the model's objective function.
*   **Enabling New Research Directions:** The end-to-end differentiability and continuous nature of FlowCodec can facilitate its integration into larger differentiable systems (e.g., in downstream analysis tasks, or learning representations directly optimized for compression). It provides a natural platform for exploring continuous JSCC.
*   **Improved Understanding:** Contributes to a better understanding of the role of continuous representations and information constraints in generative models and representation learning, aligning with the workshop's interest in understanding learning via compression principles.
*   **Potential for Real-World Applications:** If proven efficient and effective, FlowCodec could influence the design of next-generation compression systems for images, video, scientific data, and potentially model compression, where fine-grained rate control and high fidelity are crucial.

By addressing the fundamental limitations of quantization and grounding neural compression more firmly in information theory via normalizing flows and the Information Bottleneck, FlowCodec promises to catalyze progress towards more efficient, robust, and theoretically understood information processing systems.

## 5. References

[1] Ho, M., Zhao, X., & Wandelt, B. (2023). Information-Ordered Bottlenecks for Adaptive Semantic Compression. *arXiv preprint arXiv:2305.11213*.

[2] Butakov, I., Tolmachev, A., Malanchuk, S., Neopryatnaya, A., Frolov, A., & Andreev, K. (2023). Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression. *arXiv preprint arXiv:2305.08013*.

[3] Scagliotti, A., & Farinelli, S. (2023). Normalizing Flows as Approximations of Optimal Transport Maps via Linear-Control Neural ODEs. *arXiv preprint arXiv:2311.01404*.

[4] Orozco, R., Louboutin, M., Siahkoohi, A., Rizzuti, G., van Leeuwen, T., & Herrmann, F. (2023). Amortized Normalizing Flows for Transcranial Ultrasound with Uncertainty Quantification. *arXiv preprint arXiv:2303.03478*.

[5] Chen, W., Du, S., Li, S., Zeng, D., & Paisley, J. (2024). Entropy-Informed Weighting Channel Normalizing Flow. *arXiv preprint arXiv:2407.04958*.

[6] Raveri, M., Doux, C., & Pandey, S. (2024). Understanding Posterior Projection Effects with Normalizing Flows. *arXiv preprint arXiv:2409.09101*.

[7] Wang, S., Chen, J., Li, C., Zhu, J., & Zhang, B. (2022). Fast Lossless Neural Compression with Integer-Only Discrete Flows. *arXiv preprint arXiv:2206.08869*.

[8] Onken, D., Fung, S. W., Li, X., & Ruthotto, L. (2020). OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport. *arXiv preprint arXiv:2006.00104*.

[9] Helminger, L., Djelouah, A., Gross, M., & Schroers, C. (2020). Lossy Image Compression with Normalizing Flows. *arXiv preprint arXiv:2008.10486*.

[10] Ardizzone, L., Mackowiak, R., Rother, C., & Köthe, U. (2020). Training Normalizing Flows with the Information Bottleneck for Competitive Generative Classification. *arXiv preprint arXiv:2001.06448*.

*(Additional relevant foundational papers on neural compression, normalizing flows, and information bottleneck might be cited in a full paper, e.g., works by Ballé et al., Tishby et al., Dinh et al., Kingma et al., Rezende & Mohamed)*.