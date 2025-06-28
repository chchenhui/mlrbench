# **Neural Distributed Compression of Correlated Sources via Mutual Information Regularization**

## **Introduction**

### **Background**
The exponential growth of data in distributed systems—from edge sensor networks to federated learning frameworks—has intensified the need for efficient compression techniques that respect both computational constraints and theoretical limits. Classical distributed source coding (DSC) theory, epitomized by the Slepian-Wolf theorem, provides foundational guarantees for lossless compression of correlated sources. However, these results assume discrete data and explicit quantization, which struggle with high-dimensional, noisy, or nonlinearly correlated real-world data like sensor measurements or imagery. Recent advances in neural compression, such as deep variational autoencoders (VAEs) and conditional vector quantization, have demonstrated remarkable empirical performance but lack rigorous theoretical connections to DSC limits. Notably, existing neural DSC methods (e.g., [1, 4]) often prioritize architectural innovations over principled information-theoretic constraints, leading to ad hoc solutions that generalize poorly or lack performance guarantees.

### **Research Objectives**
This work proposes a neural distributed compression framework that explicitly regularizes mutual information (MI) between latent representations of correlated sources to align with theoretical rate-distortion bounds. The key objectives are:
1. **Theoretical Alignment**: Derive a continuous extension of Slepian-Wolf bounds for high-dimensional data, linking MI regularization strength to compression rate.
2. **Algorithmic Design**: Develop a VAE-based encoder-decoder architecture where encoders learn correlation-aware latent spaces through MI maximization.
3. **Empirical Validation**: Demonstrate improved compression rates and distortion guarantees on multi-view imagery and multi-sensor datasets compared to classical and neural baselines.

### **Significance**
By bridging neural compression and information theory, this work aims to:
- Enable scalable, efficient compression for distributed IoT systems and federated learning, where bandwidth is scarce.
- Provide theoretical insights into how MI regularization governs the trade-off between coding rate and reconstruction fidelity in distributed settings.
- Advance understanding of deep learning’s role in approaching or exceeding classical DSC limits for complex data.

## **Methodology**

### **Framework Overview**
The framework (Figure 1) comprises $N$ independent encoders $\{E_i\}_{i=1}^N$ and a joint decoder $D$, where $N$ is the number of correlated sources. Each encoder maps its source $X_i \in \mathcal{X}_i$ to a latent vector $Z_i \in \mathcal{Z}_i$, and the decoder reconstructs the original sources using the concatenated latent vectors $Z = [Z_1, Z_2, \ldots, Z_N]$. The core innovation lies in regularizing the encoders to maximize MI between latent variables $\{Z_i\}$, ensuring that each $Z_i$ captures globally significant correlations rather than isolated source-specific features.

---

### **Objective Function**
The training objective combines VAE-based reconstruction with MI regularization:
$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^N \underbrace{\mathbb{E}_{q_{\phi_i}(Z_i|X_i)}[\log p_{\theta_i}(X_i|Z_i)]}_{\text{(a) Reconstruction loss}} + \underbrace{D_{\text{KL}}\left(q_{\phi_i}(Z_i|X_i) \parallel p(Z_i)\right)}_{\text{(b) Latent prior matching}} - \underbrace{\beta_{\text{MI}} \cdot I(Z_1; Z_2; \ldots; Z_N)}_{\text{(c) MI regularization}},
$$
where:
- $q_{\phi_i}(Z_i|X_i)$ is the encoder’s variational posterior with parameters $\phi_i$.
- $p_{\theta_i}(X_i|Z_i)$ is the decoder’s likelihood model.
- $p(Z_i)$ is the latent prior (e.g., standard normal).
- $I(Z_1; Z_2; \ldots; Z_N)$ is the multi-way mutual information among latents.
- $\beta_{\text{MI}}$ controls regularization strength.

For pairwise correlation (e.g., $N=2$), the MI term simplifies to $I(Z_1; Z_2)$, which can be estimated via the InfoNCE loss [10]:
$$
I(Z_1; Z_2) \geq \log \frac{1}{K} \sum_{k=1}^K \frac{p(Z_1^{(k)}, Z_2^{(k)})}{p_{\text{marg}}(Z_1^{(k)}) \cdot p_{\text{marg}}(Z_2^{(k)})},
$$
where $K$ is the number of negative samples.

---

### **Algorithmic Steps**
1. **Encoder Training**:
   - For each source $X_i$, train encoder $E_i$ using the reparameterization trick:
     $$
     Z_i = \mu_{\phi_i}(X_i) + \epsilon \cdot \sigma_{\phi_i}(X_i), \quad \epsilon \sim \mathcal{N}(0, I).
     $$
   - Optimize the encoder’s weights via gradient ascent on $I(Z_1; Z_2)$ and descent on $\mathcal{L}_{\text{total}}$.

2. **Decoder Training**:
   - Train $D$ to minimize the reconstruction loss $-\log p_{\theta}(X|Z)$ using the aggregated latent $Z$.

3. **Alternating Optimization**:
   - Use alternating updates: Fix $D$ and update $E_i$; then fix $E_i$ and update $D$. Stabilizes co-learning of MI-regularized latents.

---

### **Theoretical Analysis**
Let $R$ denote the total compression rate (bits per dimension) and $D$ the average distortion $\mathbb{E}[\|X - \hat{X}\|^2]$. The framework’s achievable $(R, D)$ trade-off satisfies:
$$
R \geq I(X_1, X_2; Z_1, Z_2) - \beta_{\text{MI}} \cdot I(Z_1; Z_2),
$$
where the first term quantifies the information preserved between sources and latents, and the second term penalizes redundancy reduction via $\beta_{\text{MI}}$. As $\beta_{\text{MI}} \to 0$, the bound approaches the classical VAE rate-distortion limit. For maximized $I(Z_1; Z_2)$, mutual redundancy between sources is minimized, approaching Slepian-Wolf efficiency.

---

### **Experimental Design**
#### **Datasets**
1. **Multi-view Imagery**: Middlebury stereo dataset with left-right image pairs.
2. **Multi-sensor Data**: PhysioNet’s multi-channel ECG dataset.

#### **Baselines**
1. Classical: Distributed quantization [1].
2. Neural: Variational binning [1], cross-attention alignment [2], distributed VQ-VAE [4].

#### **Metrics**
- **Rate**: Bits per dimension (BPD).
- **Distortion**: PSNR, FID (images), and RMSE (sensor data).
- **Efficiency**: Training time and inference latency.

#### **Ablation Studies**
- Vary $\beta_{\text{MI}}$ to study the rate-distortion trade-off.
- Replace InfoNCE with alternative MI estimators (e.g., MINE [7]).

#### **Evaluation Protocol**
- Train on 80% of data, validate on 10%, test on 10%.
- Use 5-fold cross-validation to report mean ± standard deviation.

---

## **Expected Outcomes & Impact**

### **Performance Gains**
We anticipate the MI-regularized framework will reduce coding rate by 15–20% compared to distributed VQ-VAE [4] while maintaining PSNR within 3 dB of original data. For example:
- On Middlebury, achieve ≥30 dB PSNR at 0.4 BPD vs. 0.5 BPD for [4].
- On ECG, compress 12-lead signals at ≤2 Mbps (4x less than MP3 audio) with ≤2% RMSE.

### **Theoretical Advances**
The derived rate-distortion bound will provide the first explicit connection between MI regularization and Slepian-Wolf-like efficiency for continuous sources. By varying $\beta_{\text{MI}}$, the model will empirically demonstrate a Pareto frontier trading off redundancy elimination (via $I(Z_i; Z_j)$) and distortion.

### **Broader Impact**
1. **Distributed IoT Systems**: Enable energy-efficient compression for multi-sensor edge devices with minimal coordination.
2. **Federated Learning**: Reduce uplink communication costs by compressing client gradients while preserving task-relevant information.
3. **Information Theory**: Establish MI as a foundational tool for neural compression beyond heuristic dimensionality reduction.

This work aligns with the workshop’s mission to unite machine learning, compression, and information theory. By grounding neural architectures in theoretical DSC bounds, we aim to catalyze scalable, principled AI systems for decentralized data ecosystems.

---

**References**  
[1] Ozyilkan et al., "Neural Distributed Compressor Discovers Binning" (2023).  
[2] Mital et al., "Neural Distributed Image Compression with Cross-Attention" (2022).  
[4] Whang et al., "Neural Distributed Source Coding" (2021).  
[7] Poole et al., "Variational Inference for Deep Probabilistic Canonical Correlation Analysis" (2016).  
[10] Nguyen et al., "MI Maximization for Distributed Representation Learning" (2023).