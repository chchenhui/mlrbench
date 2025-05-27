# Neural Distributed Compression of Correlated Sources via Mutual Information Regularization

## 1. Introduction

The increasing deployment of distributed sensing systems, edge computing devices, and federated learning frameworks has created an unprecedented demand for efficient compression techniques that can handle correlated data from multiple sources. Traditional approaches to distributed compression have primarily relied on principles from the Slepian-Wolf theorem, which provides theoretical bounds for lossless compression of correlated sources. However, these classical methods often struggle with complex, high-dimensional correlations found in real-world data and typically require explicit knowledge of the joint distribution between sources.

Recent advancements in neural compression have demonstrated remarkable success in modeling intricate dependencies within high-dimensional data, achieving state-of-the-art performance for single-source compression tasks. Neural approaches, particularly those based on variational autoencoders (VAEs), have shown exceptional capability in learning compact latent representations that capture the underlying structure of the data. However, these methods have not been fully adapted to distributed settings where multiple correlated sources must be compressed independently while accounting for their mutual information.

This research proposal aims to bridge this critical gap by developing a novel mutual information (MI)-regularized neural framework for distributed compression of correlated continuous sources. The proposed approach leverages the representational power of neural networks while incorporating information-theoretic principles to optimize compression efficiency in distributed settings. Unlike traditional methods that rely on explicit quantization schemes, our approach utilizes continuous latent spaces that are explicitly trained to maximize the mutual information between correlated sources while minimizing reconstruction error.

The significance of this research lies in its potential to:
1. Establish theoretical connections between neural compression techniques and classical information-theoretic bounds
2. Enable efficient distributed compression for complex, high-dimensional data without requiring explicit knowledge of joint distributions
3. Provide a flexible framework that can adapt to varying degrees of correlation between sources
4. Support emerging applications in multi-sensor systems, federated learning, and decentralized IoT networks where bandwidth constraints and privacy considerations limit direct data sharing

By developing both the theoretical foundations and practical implementations of MI-regularized neural distributed compression, this research will contribute to the advancement of efficient, scalable information-processing systems that can operate under the constraints of decentralized architectures.

## 2. Methodology

Our proposed methodology consists of three primary components: (1) the neural distributed compression architecture, (2) mutual information regularization, and (3) theoretical analysis and experimental validation. Each component is detailed below.

### 2.1 Neural Distributed Compression Architecture

We consider a scenario with $K$ distributed sources, each generating data samples $x_1, x_2, \ldots, x_K$ that are correlated but must be encoded independently. For each source $k$, we design a neural encoder-decoder pair $(E_k, D_k)$ following a VAE-like structure:

The encoder $E_k$ maps input $x_k$ to a latent distribution:
$$E_k(x_k) = q_{\phi_k}(z_k|x_k)$$

where $\phi_k$ represents the encoder parameters and $q_{\phi_k}(z_k|x_k)$ is typically parameterized as a Gaussian distribution:
$$q_{\phi_k}(z_k|x_k) = \mathcal{N}(z_k|\mu_{\phi_k}(x_k), \sigma_{\phi_k}^2(x_k))$$

The functions $\mu_{\phi_k}$ and $\sigma_{\phi_k}^2$ are implemented as neural networks that output the mean and variance of the latent distribution, respectively.

The decoder $D_k$ reconstructs the input from the latent representation:
$$\hat{x}_k = D_k(z_k) = g_{\theta_k}(z_k)$$

where $\theta_k$ represents the decoder parameters and $g_{\theta_k}$ is a neural network mapping from the latent space back to the input space.

To enable efficient compression, we incorporate an entropy model $p_{\psi_k}(z_k)$ that approximates the marginal distribution of the latent variables. This model is used to estimate the coding cost of transmitting the latent representation and is typically implemented as a hierarchical prior or a normalizing flow.

### 2.2 Mutual Information Regularization

The key innovation in our approach is the incorporation of mutual information regularization to explicitly account for correlations between sources. We propose a multi-objective loss function that balances reconstruction quality, coding efficiency, and mutual information maximization:

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^K \mathcal{L}_{\text{recon}}^k + \lambda_1 \sum_{k=1}^K \mathcal{L}_{\text{rate}}^k - \lambda_2 \sum_{i \neq j} \mathcal{L}_{\text{MI}}^{i,j}$$

The reconstruction loss $\mathcal{L}_{\text{recon}}^k$ measures the distortion between the original input and its reconstruction:
$$\mathcal{L}_{\text{recon}}^k = d(x_k, \hat{x}_k)$$
where $d(\cdot, \cdot)$ is an appropriate distortion metric (e.g., mean squared error for continuous data or cross-entropy for discrete data).

The rate loss $\mathcal{L}_{\text{rate}}^k$ approximates the coding cost of the latent representation:
$$\mathcal{L}_{\text{rate}}^k = -\mathbb{E}_{q_{\phi_k}(z_k|x_k)}[\log p_{\psi_k}(z_k)]$$

The mutual information term $\mathcal{L}_{\text{MI}}^{i,j}$ encourages the latent codes of different sources to capture shared information:
$$\mathcal{L}_{\text{MI}}^{i,j} = I(z_i; z_j)$$

Since mutual information is challenging to compute directly, we employ a variational estimator based on the InfoNCE bound:

$$\mathcal{L}_{\text{MI}}^{i,j} \approx \mathbb{E}_{p(x_i, x_j)}\left[\log \frac{e^{f(z_i, z_j)}}{\frac{1}{N}\sum_{n=1}^N e^{f(z_i, z_j^n)}}\right]$$

where $f(z_i, z_j)$ is a neural network that estimates the compatibility between latent representations, and $z_j^n$ are negative samples drawn from the marginal distribution.

The hyperparameters $\lambda_1$ and $\lambda_2$ control the trade-off between reconstruction quality, coding efficiency, and mutual information maximization. These parameters can be adjusted to target specific rate-distortion operating points.

### 2.3 Compression Protocol and Implementation

The practical implementation of our framework involves the following steps:

1. **Training Phase**:
   - Collect paired samples from all sources: $(x_1, x_2, \ldots, x_K)$
   - Train the encoders, decoders, entropy models, and MI estimator jointly using the multi-objective loss
   - Save the trained models on respective encoding devices

2. **Encoding Phase** (performed independently at each source):
   - Encode input $x_k$ to obtain latent distribution parameters $\mu_{\phi_k}(x_k)$ and $\sigma_{\phi_k}^2(x_k)$
   - Sample a latent code $z_k \sim q_{\phi_k}(z_k|x_k)$
   - Quantize the latent code: $\hat{z}_k = \text{round}(z_k)$
   - Entropy code $\hat{z}_k$ using the learned entropy model $p_{\psi_k}(z_k)$
   - Transmit the encoded bitstream to the decoder

3. **Decoding Phase**:
   - Entropy decode the received bitstreams to obtain $\hat{z}_1, \hat{z}_2, \ldots, \hat{z}_K$
   - Apply a joint decoder that leverages the correlation between latent codes:
     $$\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_K = J_\theta(\hat{z}_1, \hat{z}_2, \ldots, \hat{z}_K)$$
   where $J_\theta$ is a joint decoder that can be implemented as a separate neural network or as a composition of the individual decoders with an optional fusion module

To address the non-differentiability of quantization during training, we employ the straight-through estimator (STE) or additive uniform noise to approximate the quantization process:
$$\tilde{z}_k = z_k + u, \quad u \sim \mathcal{U}(-0.5, 0.5)$$

### 2.4 Theoretical Analysis

We will establish theoretical connections between our MI-regularized approach and classical distributed source coding bounds. Specifically, we will analyze:

1. The relationship between the mutual information regularization strength $\lambda_2$ and the achievable rate-distortion performance
2. The conditions under which our approach can achieve the Slepian-Wolf bounds for lossless compression
3. The rate-distortion trade-offs in the lossy compression setting, connecting to the Wyner-Ziv and Berger-Tung bounds

For example, in the two-source case, we will derive bounds on the sum rate:
$$R_1 + R_2 \geq H(X_1, X_2) - I(X_1, X_2; \hat{X}_1, \hat{X}_2) + I(X_1; X_2) - I(Z_1; Z_2)$$

where the final term $I(Z_1; Z_2)$ corresponds to our mutual information regularization term.

### 2.5 Experimental Design

We will evaluate our proposed method on multiple datasets that feature different types of correlations:

1. **Multi-view Imagery**:
   - NYU Depth Dataset V2: RGB and depth image pairs
   - KITTI Stereo: Left and right stereo image pairs
   - These datasets will test the method's ability to exploit spatial correlations between different views of the same scene

2. **Multi-modal Sensor Data**:
   - Intel Berkeley Research Lab: Temperature, humidity, and light readings from distributed sensors
   - UCI Gas Sensor Array Dataset: Readings from multiple chemical sensors in varying conditions
   - These datasets will test the method's performance on low-dimensional but highly correlated time-series data

3. **Synthetic Data**:
   - Gaussian sources with known correlation structures
   - Non-Gaussian sources with complex dependencies
   - These controlled experiments will allow us to verify theoretical predictions and analyze performance across different correlation regimes

For each dataset, we will implement the following baselines:
1. Independent neural compression (with no mutual information regularization)
2. Classical distributed compression methods based on explicit Slepian-Wolf coding
3. State-of-the-art neural distributed compression methods from recent literature

Evaluation metrics will include:
1. **Rate-Distortion Performance**:
   - Bits per pixel (bpp) or bits per sample for the compressed representation
   - Distortion metrics appropriate to the data type (PSNR, MS-SSIM for images; MSE for sensor data)
   - Rate-distortion curves comparing our method to baselines

2. **Computational Efficiency**:
   - Encoding/decoding time
   - Model size and complexity

3. **Information-Theoretic Metrics**:
   - Estimated mutual information between latent representations
   - Comparison to theoretical bounds
   - Analysis of the information bottleneck trade-off

We will also conduct ablation studies to analyze:
1. The impact of varying the mutual information regularization strength $\lambda_2$
2. The effect of different mutual information estimators
3. The influence of the latent code dimensionality
4. The performance with different entropy model architectures

## 3. Expected Outcomes & Impact

This research is expected to yield several significant outcomes with broad impact on the fields of neural compression, distributed systems, and information theory:

### 3.1 Methodological Advances

1. **Novel Neural Architecture**: We will develop a new neural framework for distributed compression that explicitly incorporates mutual information regularization, providing a flexible approach that adapts to varying correlation structures without requiring explicit joint distribution modeling.

2. **Improved Compression Performance**: Our method is expected to achieve superior rate-distortion performance compared to traditional distributed compression techniques, particularly for complex, high-dimensional data with intricate correlation patterns.

3. **Theoretical Insights**: We will establish connections between neural compression methods and classical information-theoretic bounds, providing a deeper understanding of how mutual information regularization influences achievable rate regions in distributed settings.

4. **Practical Implementation**: We will deliver practical algorithms and implementations that can be deployed in real-world distributed sensing systems, with code and pre-trained models made available to the research community.

### 3.2 Applications and Impact

1. **Distributed Sensing Systems**: Our approach will enable more efficient data collection and transmission in multi-sensor environments, such as environmental monitoring, industrial IoT, and smart infrastructure, where bandwidth is limited and sensors capture correlated information.

2. **Federated Learning**: The principles developed in this research can be applied to compress model updates in federated learning settings, reducing communication overhead while preserving information about common patterns across distributed datasets.

3. **Edge Computing**: Our method will support edge computing architectures where multiple devices must process and transmit correlated observations under bandwidth constraints, enabling more intelligent data filtering and aggregation.

4. **Multi-modal Learning**: The mutual information regularization approach can inform better multi-modal representation learning, where different modalities (e.g., vision, audio, text) contain complementary and correlated information.

### 3.3 Long-term Scientific Impact

1. **Bridging Neural Compression and Information Theory**: This research will contribute to the ongoing effort to integrate deep learning approaches with classical information-theoretic principles, potentially leading to new theoretical frameworks that accommodate both perspectives.

2. **Advancing Distributed Information Processing**: By addressing fundamental questions about how to optimally compress correlated information in distributed settings, this work will contribute to broader understanding of distributed information processing systems.

3. **New Research Directions**: The combination of mutual information regularization with neural compression may inspire new approaches to related problems, such as distributed sensing, collaborative inference, and privacy-preserving data sharing.

In summary, this research aims to develop a theoretically grounded yet practically effective approach to distributed neural compression that explicitly accounts for correlations between sources through mutual information regularization. The expected outcomes will advance the state of the art in neural compression methods for distributed settings, with applications spanning from IoT sensor networks to federated learning systems. By bridging neural network approaches with information-theoretic principles, this work will contribute to the fundamental understanding of distributed information processing while enabling more efficient practical systems.