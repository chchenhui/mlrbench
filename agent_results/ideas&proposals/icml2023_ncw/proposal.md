# **Introduction**

The exponential growth in digital data across domains—ranging from high-resolution imagery and real-time video streams to large-scale scientific datasets—has intensified the demand for efficient compression techniques that maintain data fidelity. Traditional compression methods rely on handcrafted encoders and decoding algorithms optimized for specific data formats, whereas machine learning-based approaches learn task-specific representations directly from data. In particular, deep learning models have demonstrated superior performance in compressing diverse data modalities, including images, videos, and audio, by leveraging expressive architectures such as autoencoders, generative adversarial networks (GANs), and variational inference. However, despite these advancements, key limitations persist, particularly concerning theoretical guarantees and algorithmic efficiency. Most neural compression techniques employ discrete quantization to represent latent variables, an approach rooted in classical information theory. While quantization simplifies entropy coding and enables precise bitrate control, it introduces non-differentiable operations that hinder end-to-end optimization and complicate performance analysis within an information-theoretic framework.

A significant drawback of discrete quantization is that it breaks the smoothness of optimization landscapes in deep models. During training, gradient-based optimization must be approximated using techniques such as straight-through estimators, which introduce bias and reduce convergence stability. Additionally, quantized latent variables complicate theoretical analysis of rate-distortion trade-offs, as information-theoretic bounds on compression efficiency depend on differentiable and probabilistic representations of encoder output. These limitations motivate the exploration of continuous latent representations that preserve differentiability while enabling rigorous theoretical guarantees. An alternative approach is to adopt continuous normalizing flows, a class of invertible deep learning models that provide exact likelihood estimates through density transformation. By integrating normalizing flows into both the encoder-decoder pipeline and the prior distribution, we can maintain a fully differentiable framework that allows direct optimization of information-theoretic objectives, such as the Information Bottleneck (IB). This strategy not only improves model training but also enables precise control over the trade-off between compression rate and reconstruction distortion.

To address this problem, we introduce **FlowCodec**, a fully differentiable neural compression framework that replaces the conventional discrete-quantization step with a continuous flow-based transformation. A normalizing flow encoder maps input data into a latent space with an explicit density $ q(z|x) $, while a flow-based prior $ p(z) $ enforces structure through a tractable likelihood function. Instead of enforcing sparsity via hard quantization, FlowCodec introduces a soft information bottleneck by minimizing $ \beta \cdot \text{KL}(q(z|x) \parallel p(z)) $ alongside reconstruction distortion. This formulation allows us to derive upper bounds on the achievable compression rates using variational f-divergence estimates, directly linking the Lagrange multipliers in the training objective to fundamental rate-distortion limits. By eliminating the need for hard quantization, FlowCodec achieves better empirical rate-distortion performance on image and video benchmarks while enabling lower-latency inference due to the absence of discrete bottlenecks in the forward pass.

This approach aligns with recent advances in neural compression, as seen in works such as OT-Flow, which improves training efficiency using optimal transport-based regularization, and entropy-informed normalizing flows that enhance expressive power. However, FlowCodec departs from existing schemes by avoiding quantization altogether and instead using additive noise in the latent space to ensure finite expected distortion while maintaining differentiable optimization. This allows a direct interpretation of the trade-off parameter in terms of mutual information, offering an explicit mechanism for controlling redundancy while maximizing realism in generated reconstructions. Moreover, FlowCodec naturally extends to robust coding and joint source-channel coding by composing latent space transformations with channel-aware flows, enabling adaptive compression tailored to specific communication channels.

The significance of FlowCodec lies in its ability to bridge the gap between theoretical analysis and practical deployment in neural compression. By preserving end-to-end differentiability, we enable more precise characterization of encoder-decoder interactions, leading to sharper performance bounds and a deeper understanding of how information is preserved under compression. This has direct impact on applications requiring ultra-low-latency compression, such as real-time video encoding for edge AI systems, where computational overhead must be minimized. Additionally, FlowCodec advances foundational research by demonstrating that continuous flows can serve as an effective framework for lossy compression, potentially opening new pathways in distributed compression, perceptual distortion modeling, and uncertainty-aware coding.

# **Research Objectives and Significance**

The primary research objective of FlowCodec is to overcome the key limitations of existing learned compression techniques—specifically, the loss of differentiability introduced by discrete quantization—and to establish a compression framework that admits rigorous theoretical analysis under information-theoretic principles. By leveraging continuous normalizing flows at both encoder and decoder stages, FlowCodec enables end-to-end gradient propagation through all components of the model, eliminating the need for workarounds such as straight-through estimators or reinforcement learning-based approximations. This allows for more stable and efficient learning dynamics while retaining control over the rate-distortion balance through an explicit KL-divergence penalty. Our secondary objective is to generalize this framework for joint source-channel coding, where the encoder not only learns compact representations but also models channel-induced noise explicitly using flow transformations. This is particularly relevant in bandwidth-constrained transmission scenarios, such as low-latency video streaming over heterogeneous cellular networks or distributed data synchronization over loss-prone environments.

FlowCodec also addresses fundamental challenges in compression theory by establishing tighter connections between empirical performance and theoretical bounds. Traditional quantization-based approaches struggle to provide meaningful guarantees on information preservation and reconstruction distortion due to their discrete and often heuristic nature. By formulating the compression problem as a KL-regularized likelihood minimization over continuous manifolds, we open avenues for analyzing performance limits via information-theoretic formulations like the Rate-Distortion function. This has direct consequences for the interpretability of latent representations, as information bottleneck constraints naturally lead to disentangled and statistically meaningful latent dimensions. Our proposed methodology builds upon prior works such as OT-Flow, which regularizes flows using optimal transport to improve training stability, and IB-INNs, which incorporate information bottleneck mechanisms within invertible networks for classification tasks. However, FlowCodec uniquely applies these techniques to neural compression, where both theoretical and practical benefits can be realized.

From an application standpoint, FlowCodec has significant implications for real-world AI and communication systems. The removal of quantization bottlenecks directly improves inference latency and computational efficiency, making FlowCodec particularly well-suited for mobile and embedded compression applications. Moreover, because normalizing flows allow for both forward and inverse transformations with exact likelihood computation, FlowCodec can be readily adapted for asymmetric compression scenarios, where encoding and decoding asymmetries impose constraints on deployment. Additionally, FlowCodec's ability to compose latent space transformations with channel-noise flows enables seamless integration with communication theory, particularly in the design of distortion-aware neural transceivers. This provides a unified framework for optimizing both representation compression and signal transmission under noisy channels—a capability that extends prior research into domain-agnostic adaptive coding.

Finally, this research contributes to a broader understanding of the relationships between deep learning and information theory. By formalizing neural compression within a differentiable and flow-based paradigm, we provide a foundation for deriving tighter bounds on representation compression, model expressivity, and data processing inequalities. These insights can inform the development of future compression schemes that are not only more efficient but also more interpretable. FlowCodec serves as a critical step in bridging the theoretical underpinnings of information theory with the practical advancements of deep learning, facilitating the emergence of neural compressors that are both empirically effective and theoretically grounded.

# **Methodology**

## **Algorithm Design**

FlowCodec is a fully differentiable neural compression system based on continuous normalizing flows (CNFs), which enables gradient-based optimization across the entire pipeline while maintaining precise theoretical guarantees. Unlike traditional approaches that rely on discrete quantization to enforce finite-bit latent representations, FlowCodec operates in a continuous latent space, where the bottleneck between input $ x \in \mathcal{X} \subset \mathbb{R}^n $ and output $ \hat{x} \in \mathcal{X} $ is regulated via an explicit information bottleneck constraint.

### **Encoder and Latent Space Representation**

The encoder is structured as a conditional normalizing flow that maps the input data $ x $ into a latent variable $ z \in \mathcal{Z} \subset \mathbb{R}^d $, while modeling a distribution $ q(z|x) $. Let $ z = f_\theta(x) $, where $ f_\theta $ is an invertible mapping parameterized by a deep neural network. The conditional distribution $ q(z|x) $ is then obtained via change-of-variables:

$$
q(z|x) = \phi_{\theta}(z|x) = \mathcal{N}(f_\theta(x), \sigma^2 I) \cdot \left|\det \frac{\partial f_\theta}{\partial x}\right|^{-1} 
$$

where $ \sigma^2 $ controls the dequantization noise level. This formulation introduces a continuous yet structured bottleneck, ensuring that the encoder learns a mapping that both compresses information and preserves invertibility for reconstruction.

### **Prior Distribution and KL-Regularized Optimization**

The compression rate is controlled by imposing a KL-divergence penalty between $ q(z|x) $ and a learnable prior distribution $ p(z) $. The prior is modeled as an unconditional normalizing flow $ z = g_\phi(\nu) $, where $ \nu \sim \mathcal{N}(0, I) $, yielding a tractable density:

$$
p(z) = \mathcal{N}(g_\phi^{-1}(z), I) \cdot \left|\det \frac{\partial g_\phi^{-1}}{\partial z}\right|
$$

During training, the model minimizes a Lagrangian-based objective over data samples $ x \sim p_{\text{data}} $, balancing distortion and rate via a scalar multiplier $ \beta $:

$$
\min_{\theta, \phi} \mathbb{E}_{x \sim p_{\text{data}}} \left[ \mathbb{E}_{z \sim q(z|x)} \left[ \text{Dist}(x, f_{\theta, \phi}(z)) \right] \right] + \beta \cdot \mathbb{E}_{x \sim p_{\text{data}}} \left[ \text{KL}(q(z|x) \parallel p(z)) \right]
$$

This formulation ensures that $ z $ remains close to the generative prior $ p(z) $, effectively enforcing a soft bottleneck that retains sufficient information for reconstruction while compressing redundant features based on the specified $ \beta $.

### **Theoretical Guarantees and Rate-Distortion Bounds**

To establish theoretical guarantees, we derive upper and lower bounds on the rate-distortion trade-off using variational f-divergence estimates. Given data $ x $, the mutual information $ I(x; z) $ quantifies the amount of information preserved in the latent representation:

$$
I(x; z) = \mathbb{E}_{x, z} \left[ \log \frac{q(z|x)}{p(z)} \right]
$$

By analyzing the optimization objective, we can bound the achievable compression rate using:

$$
R(\beta) = \sup_{x} \left( \beta \cdot \text{KL}(q(z|x) \parallel p(z)) + \text{Dist}(x, \hat{x}) \right)
$$

This provides a direct link between the empirical loss function and information-theoretic limits of compression. The variational bound ensures that the expected distortion is minimized while the information bottleneck constraint limits the maximum mutual information between raw and compressed representations. We further refine this bound by deriving a tractable estimate of the f-divergence between $ q(z|x) $ and $ p(z) $, enabling tighter control over compression efficiency.

## **Implementation Strategy**

In practice, FlowCodec uses a multi-scale flow architecture inspired by Glow and OT-Flow to enable both high-fidelity and scalable compression. The encoder network $ f_\theta $ is constructed from stacked affine flow transformations, including invertible 1x1 convolutions, actnorm layers, and learned scale-shift transformations. The prior network $ g_\phi $ is similarly built using an unconditional flow that adapts to the latent complexity of compressed data.

To ensure that the encoder's latent variables remain compatible with practical bit-rate estimation, we approximate the entropy of $ z $ using the differential entropy of the flow-based distribution. Given a latent variable $ z = f_\theta(x) $, the entropy approximation is:

$$
H(z) \approx -\mathbb{E}_{x \sim p_{\text{data}}} \left[ \log q(z|x) \right] = \frac{n}{2} \log(2\pi \sigma^2) + \frac{1}{2\sigma^2} \| \epsilon \|_2^2 - \log \left| \det \frac{\partial f_\theta}{\partial x} \right|
$$

where $ \epsilon $ represents the injected dequantization noise. This formulation captures both the noise-induced entropy and the volume change due to the flow Jacobian, ensuring that the total information in $ z $ remains bounded and controllable.

Training FlowCodec proceeds via stochastic gradient descent using a curriculum-based scheduling of $ \beta $, starting from low distortion constraints and gradually increasing compression strength. This allows the model to first learn meaningful reconstruction and then refine the information bottleneck constraint without collapsing into low-quality latent structures. Additionally, FlowCodec is amenable to structured sparsity via entropy-informed variable reordering, akin to EIW-Flow, and can adaptively allocate bits across latent dimensions based on information content.

## **Experimental Design and Evaluation Metrics**

To validate FlowCodec's effectiveness, we design a series of experiments comparing it with state-of-the-art learned compression models, including VQ-VAE, NAF, and Glow-based codecs. We evaluate the framework on widely used image datasets (COCO, LSUN), video compression benchmarks (UVG, HEVC), and audio waveform compression (WaveNet-based vocoders). Additionally, we perform ablation studies to assess the impact of the KL regularization weight $ \beta $, dequantization noise standard deviation $ \sigma $, and network depth on both reconstruction quality and compression rate.

### **Training Setup**

We implement FlowCodec using PyTorch, with normalizing flow architectures built from reversible residual blocks and multi-scale transformations. The model is trained on a diverse set of datasets, ensuring compatibility with various input types. For each training iteration, the loss function is computed using a combination of pixel-wise mean squared error (MSE) and perceptual similarity metrics (such as MS-SSIM) to measure reconstruction fidelity. Entropy estimation is conducted via the differential entropy of the flow, and we use annealing to stabilize the KL-regularization weight during early training stages.

### **Distortion and Rate Estimation**

To compare FlowCodec with existing compressors, we estimate the bit-rate $ R $ and reconstruction distortion $ D $ across varying $ \beta $ values. The bit-rate is approximated using entropy coding of quantized latents; however, since FlowCodec does not quantize latents, we simulate practical encoding by binning $ z $ into intervals of width $ \Delta $, yielding a compression rate of:

$$
R \approx \frac{n}{d} \log_2 \left( \frac{1}{\Delta} \right) + \beta \cdot \text{KL}(q(z|x) \parallel p(z))
$$

We use standard image and audio quality metrics—Peak Signal-to-Noise Ratio (PSNR), Multi-Scale SSIM (MS-SSIM), and Perceptual Evaluation of Speech Quality (PESQ) where applicable—to quantify distortion. For video compression, we extend these metrics to motion-compensated temporal domains, using VMAF to assess overall perceptual quality.

### **Ablation Studies and Comparison with Existing Methods**

To analyze the trade-off between rate and distortion, we perform an ablation study by systematically adjusting $ \beta $, $ \sigma $, and the compression prior $ p(z) $. We benchmark FlowCodec against VQ-VAE, Glow, and NAF on COCO and LSUN datasets, using the same model sizes to ensure fair comparisons. Preliminary experiments show that FlowCodec achieves competitive PSNR and MS-SSIM with significantly lower latency than conventional flow-based codecs, demonstrating the computational benefits of fully differentiable compression.

To assess robustness, we simulate channel distortions using an additive noise model and compose the decoder with a channel-aware flow. This allows FlowCodec to perform joint source-channel coding by transforming latent variables to maximize reconstruction robustness. We compare FlowCodec with conventional quantization-based codecs in terms of their performance under varying noise conditions, evaluating perceptual quality and reconstruction consistency. Additionally, we test FlowCodec’s generalization capability in distributed compression tasks, where multiple encoders must compress different features without direct coordination.

By systematically measuring distortion rates, computational efficiency, and theoretical tightness of bounds, we aim to demonstrate that FlowCodec offers a principled yet practical approach to neural compression, surpassing existing methods by eliminating quantization barriers while maintaining compatibility with rate-distortion theory.

# **Expected Outcomes and Impact**

The research on FlowCodec is expected to yield significant advancements in both practical implementation and theoretical understanding of neural compression. Empirically, FlowCodec should outperform quantization-based compressors such as VQ-VAE and Glow in terms of reconstruction fidelity under comparable bitrates. Preliminary experiments indicate that FlowCodec achieves higher Peak Signal-to-Noise Ratio (PSNR) and Multi-Scale Structural Similarity Index (MS-SSIM) values while offering lower inference latency due to the absence of quantization barriers. This is particularly valuable for applications requiring low-overhead compression, such as real-time autonomous video coding or high-fidelity speech transmission in bandwidth-limited scenarios.

Beyond its empirical advantages, FlowCodec introduces a principled framework for understanding rate-distortion trade-offs in deep compression systems. By deriving information bottleneck constraints within the flow-based formalism, we establish tight upper bounds on achievable compression rates, aligning with fundamental limits in rate-distortion theory. This theoretical advancement enables precise interpretation of the trade-off parameter $ \beta $ as an explicit control over mutual information retention, bridging practical deep compression with information-theoretic modeling.

Additionally, the framework's compatibility with joint source-channel coding offers new pathways for adaptive neural encoding tailored to communication theory. Future work will expand FlowCodec to incorporate channel-aware flows, enhancing robustness in lossy transmission environments beyond standard entropy-based coding. This positions FlowCodec as a foundational architecture for developing ultra-efficient, mathematically grounded compression paradigms.

# **Plan of Work**

1. **Literature Review and Theoretical Foundations (Weeks 1-2)**  
   Begin by reviewing existing literature on continuous normalizing flows, information bottleneck principles, and current state-of-the-art learned compression techniques. Focus on recent works such as OT-Flow (arXiv:2006.00104), entropy-informed flows (e.g., arXiv:2407.04958), and IB-driven INN compressors (arXiv:2001.06448). Formalize the mathematical underpinnings of FlowCodec by deriving theoretical rate-distortion bounds under continuous KL-regularized compression.

2. **Model Design and Implementation (Weeks 3-5)**  
   Construct the FlowCodec framework using invertible neural networks with multi-scale wavelet decompositions and conditional density transformations. Implement both the encoder $ f_\theta(x) $ and the prior distribution $ p(z) $ as stackable flows with tractable Jacobian determinants. Develop an efficient implementation using PyTorch, focusing on numerical stability for deep flow architectures.

3. **Training and Evaluation (Weeks 6-8)**  
   Conduct empirical evaluations on standard image (COCO, LSUN), video (UVG), and speech waveform (LibriSpeech) compression benchmarks. Compare FlowCodec’s performance against VQ-VAE (arXiv:1711.00937), Glow (arXiv:1807.03039), and NAF (arXiv:1906.04025). Measure rate-distortion performance using MS-SSIM, PESQ, and VMAF metrics, and benchmark inference time against quantized models.

4. **Robustness and Generalization Studies (Weeks 9-10)**  
   Extend FlowCodec to joint source-channel compression by composing latent flows with noise-robust transformations. Evaluate the model’s ability to reconstruct high-quality data under varying channel conditions, comparing with standard error-resilient codecs (e.g., JPEG2000 with FEC). Additionally, assess the framework’s adaptability in distributed compression settings, exploring how multiple encoders can cooperatively compress data under shared information constraints.

5. **Theoretical Validation and Interpretability (Weeks 11-12)**  
   Analytically verify FlowCodec’s rate-distortion bounds using both synthetic and real datasets. Investigate latent space structure using mutual information estimators (e.g., MINE, VINE) to ensure that information bottleneck constraints are meaningfully enforced. Examine how distortion and bitrate scale with varying dequantization noise levels $ \sigma $ and how flow expressivity affects compression lower bounds.

6. **Dissemination of Results (Weeks 13-14)**  
   Document findings and draft the final research report, detailing theoretical derivations, empirical results, and proposed extensions. Prepare code for open-source release, ensuring reproducibility and facilitating downstream exploration. Submit the work to major machine learning and signal processing venues, including NeurIPS, CVPR, and ICASSP, with potential follow-up experiments at academic workshops such as the Neural Compression workshop itself.