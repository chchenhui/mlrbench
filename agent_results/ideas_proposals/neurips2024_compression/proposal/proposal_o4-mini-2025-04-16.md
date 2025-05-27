Title  
Mutual Information–Regularized Neural Framework for Distributed Compression of Correlated Continuous Sources  

1. Introduction  
Background  
Distributed compression of correlated sources is a cornerstone of information theory, epitomized by the Slepian–Wolf and Wyner–Ziv theorems, which characterize the theoretical limits of lossless and lossy coding with side information. Classical schemes rely on carefully designed quantizers and binning strategies, but they struggle to model high‐dimensional, complex dependencies found in modern multi‐sensor and multi‐view data. Concurrently, deep generative models such as variational autoencoders (VAEs) and vector‐quantized VAEs (VQ‐VAEs) have demonstrated remarkable capacity to model intricate distributions and deliver state‐of‐the‐art rates in single‐source compression. Bridging these strands, recent neural distributed coding works (e.g., Neural Distributed Compressor Discovers Binning, Neural Distributed Image Compression with Cross‐Attention) have shown promising empirical performance but often lack rigorous theoretical grounding, especially in continuous‐valued, multi‐source settings.  

Research Objectives  
This proposal aims to develop, analyze, and validate a mutual information–regularized neural compression framework for distributed coding of correlated continuous sources. We focus on:  
• Designing independent encoders and a joint decoder based on VAEs that replace explicit quantization with continuous latent representations.  
• Introducing a mutual information (MI) regularizer between latent codes to encourage exploitation of inter‐source correlations in encoding.  
• Establishing theoretical connections between the MI regularization weight and achievable rate–distortion bounds under the Slepian–Wolf region.  
• Conducting thorough experiments on multi‐view imagery and synthetic multi‐sensor data to validate the advantages in rate–distortion performance and computational efficiency.  

Significance  
Success will yield a scalable, neural‐based distributed compression architecture with rigorously grounded rate–distortion trade‐offs. Potential applications include resource‐constrained IoT networks, federated learning systems with communication budgets, and low‐bandwidth real‐time video streaming. Theoretically linking MI regularization to classical limits strengthens the bridge between machine learning and information theory.  

2. Methodology  
2.1 Problem Formulation  
We consider $N$ correlated continuous sources $X_1,\dots,X_N$ with joint density $p(x_1,\dots,x_N)$. Each source $X_i$ is observed at an encoder $E_i$ that produces a latent code $Z_i\in\mathbb{R}^d$. A central decoder $D$ receives all codes $\{Z_i\}$ and reconstructs $\{\hat X_i\}$. The goal is to minimize the sum‐rate under a sum‐distortion constraint.  

2.2 Model Architecture  
Encoders: For each $i$, define an encoder network  
$$q_{\phi_i}(z_i\mid x_i)\,,\quad z_i\in\mathbb{R}^d$$  
parameterized by $\phi_i$. Decoders: A joint decoder network  
$$p_\theta(\hat x_1,\ldots,\hat x_N\mid z_1,\ldots,z_N)$$  
parameterized by $\theta$, splits into per‐source reconstructions for distortion measurement.  

2.3 Objective Function  
We combine a distortion term with an MI regularization between every pair of latent codes. Denoting a distortion measure $d(x,\hat x)$ (e.g. MSE), the total loss over a batch is:  
$$  
\mathcal{L}(\phi,\theta,\psi)  
= \sum_{i=1}^N \mathbb{E}_{p(x_i)}\mathbb{E}_{q_{\phi_i}(z_i\mid x_i)}[\,d(x_i,\hat x_i)\,]  
\;+\;\lambda\sum_{i<j} I_{\psi}(Z_i;Z_j)\,.  
$$  
Here $\lambda>0$ controls correlation exploitation, and $I_\psi(Z_i;Z_j)$ is a variational lower bound estimator (e.g., MINE).  

Mutual Information Estimation  
We adopt the MINE estimator: define a critic network $T_\psi(z_i,z_j)\colon\mathbb{R}^{2d}\to\mathbb{R}$. The Donsker‐Varadhan lower bound gives:  
$$  
I(Z_i;Z_j) \ge  
\mathbb{E}_{p(z_i,z_j)}[T_\psi]\;-\;\log\mathbb{E}_{p(z_i)p(z_j)}[e^{T_\psi}]\,.  
$$  
We sample $(z_i,z_j)\sim q_{\phi_i}(z_i\mid x_i)q_{\phi_j}(z_j\mid x_j)$ using paired data $(x_i,x_j)$ for the joint term, and by shuffling batches for the product term.  

2.4 Training Algorithm  
1. Sample a minibatch $\{(x_i^{(n)})_{i=1}^N\}_{n=1}^B$ from the training set.  
2. For each $i$ and $n$, sample $z_i^{(n)}\sim q_{\phi_i}(z_i\mid x_i^{(n)})$ using the reparameterization trick.  
3. Compute reconstructions $\hat x_i^{(n)} = D_i(z_1^{(n)},\dots,z_N^{(n)};\theta)$.  
4. Estimate MI terms $I_\psi(Z_i;Z_j)$ using the critic $T_\psi$ on joint vs. marginal samples.  
5. Compute $\mathcal{L}$ and take gradient steps on $(\phi_1,\dots,\phi_N,\theta,\psi)$ using Adam.  
6. Repeat until convergence.  

2.5 Theoretical Analysis  
We will derive bounds relating the MI regularizer to classical rate–distortion functions. Recall the single‐source bound:  
$$R(D) = \inf_{p(z|x):\,\mathbb{E}[d(X,\hat X)]\le D}I(X;Z)\,. $$  
For two sources under Slepian–Wolf, the achievable region in the lossless case satisfies  
$$R_1 \ge H(X_1\mid X_2),\quad R_2 \ge H(X_2\mid X_1),\quad R_1+R_2\ge H(X_1,X_2)\,. $$  
We will show that enforcing a finite $\lambda\,I(Z_1;Z_2)$ induces an implicit binning structure that approaches these bounds in the limit. By relating the penalty weight $\lambda$ to a Lagrange multiplier in the constrained optimization of distributed rate–distortion, we will formalize how tuning $\lambda$ trades off sum‐rate vs. distortion.  

2.6 Experimental Design and Evaluation  
Datasets  
• Multi‐View Imagery: KITTI Stereo, Cityscapes stereo pairs.  
• Synthetic Gaussian Sensor Network: $N$ sources with joint Gaussian distribution and tunable correlation coefficient $\rho$.  
Preprocessing: Normalize to $[0,1]$, crop, and batch.  

Baselines  
• Classical Slepian–Wolf with scalar quantization + ideal entropy coding.  
• Independent VAE compression without MI regularization ($\lambda=0$).  
• VQ‐VAE–based distributed code (Neural Distributed Source Coding).  

Metrics  
• Rate–Distortion Curves: Bits per pixel (bpp) vs. PSNR (dB), MS‐SSIM.  
• Achieved Sum‐Rate vs. Theoretical Lower Bound (for Gaussian case).  
• Computational Cost: encoding/decoding latency, parameter count.  
• Latent MI: measured via a separate MI estimator for analysis.  

Protocol  
1. For each dataset and baseline, train models at multiple target rates (by adjusting $\lambda$ or quantization bits).  
2. Plot RD curves and compute area‐under‐curve (AUC) gains.  
3. On synthetic Gaussian data, sweep $\rho\in\{0.2,0.5,0.8\}$, measure sum‐rate vs. theoretical Slepian–Wolf lower bound.  
4. Perform ablation: compare MI estimators (MINE vs. InfoNCE), latent dimensionalities $d\in\{32,64,128\}$.  
5. Evaluate generalization: train on one correlation level, test on others; train on Cityscapes, test on KITTI.  

3. Expected Outcomes & Impact  
Expected Outcomes  
• A neural distributed compression architecture that consistently outperforms classical Slepian–Wolf schemes and unregularized VAEs in RD performance, particularly at high correlation regimes.  
• Empirical confirmation that the MI regularization weight $\lambda$ provides a practical knob aligning performance with theoretical Slepian–Wolf limits.  
• Theoretical bounds and proofs demonstrating how MI regularization induces binning‐like behavior and approaches optimal rate–distortion trade‐offs in the continuous‐valued setting.  
• Insights into estimator choices (MINE vs. InfoNCE), encoder/decoder capacities, and hyperparameter regimes relevant for real‐world deployment.  

Impact  
Bridging the gap between deep generative modeling and classical information theory, this work will:  
• Provide a scalable solution for distributed compression in IoT, sensor fusion, and federated learning, reducing communication overhead without sacrificing fidelity.  
• Offer theoretical foundations that demystify recent neural distributed coding methods, facilitating principled model design and hyperparameter tuning.  
• Stimulate further interdisciplinary research by connecting MI estimation techniques with rate–distortion theory in multi‐source contexts.  
• Lay the groundwork for extensions to joint source–channel coding, adaptive real‐time streaming, and privacy‐preserving compression in distributed AI systems.  

By marrying mutual information regularization with neural compression, we anticipate this research will catalyze the next generation of efficient, theoretically grounded, distributed information‐processing systems.