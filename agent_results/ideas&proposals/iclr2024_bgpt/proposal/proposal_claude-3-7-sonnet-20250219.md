# Dynamic Curvature-Aware Optimization: Bridging Theory and Practice at the Edge of Stability

## 1. Introduction

Deep learning has witnessed remarkable empirical success across a wide range of applications, yet significant gaps persist between practical implementations and theoretical understanding. One of the most prominent disparities concerns optimization dynamics during neural network training. Recent theoretical studies have identified the "Edge of Stability" (EoS) phenomenon (Cohen et al., 2021), where gradient-based optimizers operate just beyond the stability threshold defined by the maximum eigenvalue of the Hessian. This observation challenges classical optimization theory which often assumes smoothness and convexity of the loss landscape.

Traditional optimizers like SGD, Adam, and their variants are primarily designed based on first-order information, largely ignoring the rich curvature information contained in the loss landscape. While second-order methods theoretically offer advantages in navigating complex loss surfaces, their prohibitive computational costs have limited practical adoption in deep learning. The disconnect between theory and practice becomes particularly apparent when considering non-smooth regions, sharp minima, and the implicit regularization effects observed during training.

The empirical evidence of the EoS phenomenon suggests that the optimization process in deep neural networks follows more complex dynamics than previously theorized. As demonstrated by Cohen et al. (2022) and Song and Yun (2023), gradient descent exhibits a characteristic pattern where the maximum eigenvalue of the Hessian hovers just above a stability threshold determined by the learning rate. This behavior results in non-monotonic training trajectories that nevertheless achieve consistent long-term loss reduction. Adaptive methods like Adam demonstrate similar patterns, albeit with preconditioned gradients creating an "Adaptive Edge of Stability" (Cohen et al., 2022).

This research aims to bridge this theory-practice gap by developing a Dynamic Curvature-Aware Optimizer (DCAO) that explicitly leverages second-order information while maintaining computational efficiency. By periodically probing the local curvature spectrum and dynamically adjusting optimization hyperparameters, DCAO seeks to stabilize training in high-curvature regions while accelerating convergence in more favorable landscape regions. Unlike existing approaches that either ignore curvature information entirely or apply it uniformly, our method adapts its behavior based on spectral properties of the loss landscape revealed during training.

The significance of this research is threefold. First, it operationalizes theoretical insights about non-smooth optimization and the EoS phenomenon into a practical algorithm that improves training dynamics. Second, it provides a framework for understanding how curvature information can be efficiently leveraged in large-scale deep learning. Third, it narrows the gap between optimization theory and practice by empirically validating theoretical predictions about convergence behavior in non-smooth settings.

## 2. Methodology

### 2.1 Overview of the Dynamic Curvature-Aware Optimizer (DCAO)

The proposed DCAO algorithm periodically estimates local curvature information using efficient low-rank approximations of the Hessian matrix. This information guides adaptive adjustment of key optimization hyperparameters, including learning rate, momentum, and weight decay, to achieve stable and efficient convergence. The algorithm operates in two alternating phases:

1. **Standard optimization steps**: The majority of updates follow a momentum-based update rule similar to traditional optimizers.
2. **Curvature probing and adaptation**: At regular intervals, the algorithm performs curvature estimation and adjusts hyperparameters accordingly.

### 2.2 Hessian Approximation via Stochastic Lanczos Iteration

The Hessian matrix $H \in \mathbb{R}^{d \times d}$ of a neural network with $d$ parameters is prohibitively large to compute and store explicitly. Instead, we employ the stochastic Lanczos quadrature method to efficiently approximate the top-$k$ eigenvalues and corresponding eigenvectors of the Hessian.

The algorithm proceeds as follows:

1. Select a random unit vector $v_0 \in \mathbb{R}^d$.
2. Apply the Lanczos iteration to construct an orthogonal basis $\{v_0, v_1, ..., v_{m-1}\}$ for the Krylov subspace $\mathcal{K}_m(H, v_0) = \text{span}\{v_0, Hv_0, H^2v_0, ..., H^{m-1}v_0\}$, which yields a tridiagonal matrix $T_m$ that represents the projection of $H$ onto this subspace.
3. Compute the eigendecomposition of $T_m$ to obtain approximations of the extreme eigenvalues of $H$.

The Lanczos iteration can be implemented efficiently using Hessian-vector products without forming the Hessian matrix explicitly:

$$Hv \approx \frac{\nabla f(w + \epsilon v) - \nabla f(w)}{\epsilon}$$

where $f$ is the loss function, $w$ represents the current parameters, and $\epsilon$ is a small perturbation.

To reduce computational overhead, we perform this estimation on mini-batches of data rather than the full dataset and repeat the process with multiple random starting vectors $v_0$ to improve accuracy.

### 2.3 Curvature Metrics

From the Lanczos approximation, we derive the following curvature metrics:

1. **Spectral radius** ($\rho$): The magnitude of the largest eigenvalue $|\lambda_1|$, indicating the maximum curvature.
2. **Spectral gap** ($\Delta$): The difference between the largest and second-largest eigenvalues, $|\lambda_1 - \lambda_2|$, providing information about the conditioning of the loss landscape.
3. **Spectral density**: The distribution of the top-$k$ eigenvalues, providing insights into the overall curvature profile.
4. **Negative eigenvalue ratio** ($\nu$): The proportion of negative eigenvalues among the top-$k$, indicating saddle point proximity.

### 2.4 Dynamic Hyperparameter Adjustment

Based on the curvature metrics, DCAO dynamically adjusts the following hyperparameters:

1. **Learning rate adjustment**: The base learning rate $\eta_t$ is adjusted according to:

$$\eta_t = \eta_{\text{base}} \cdot \min\left(1, \frac{\beta}{\rho_t}\right) \cdot \sqrt{1 + \gamma \Delta_t}$$

where $\beta$ is the stability threshold (typically 2.0 for gradient descent), $\rho_t$ is the current spectral radius, and $\gamma$ is a scaling factor for the spectral gap contribution. This formulation reduces the learning rate when approaching highly curved regions while allowing for acceleration in well-conditioned regions.

2. **Momentum adjustment**: The momentum parameter $\mu_t$ is adjusted based on the spectral gap:

$$\mu_t = \mu_{\text{base}} + \alpha(1 - \exp(-\kappa \Delta_t))$$

where $\mu_{\text{base}}$ is the base momentum value, $\alpha$ controls the maximum momentum increase, and $\kappa$ scales the impact of the spectral gap. This allows increased momentum in directions with consistent curvature.

3. **Weight decay adjustment**: The weight decay parameter $\lambda_t$ is adjusted based on the negative eigenvalue ratio:

$$\lambda_t = \lambda_{\text{base}} \cdot (1 + \phi \nu_t)$$

where $\lambda_{\text{base}}$ is the base weight decay and $\phi$ controls the scaling. This increases regularization near saddle points.

### 2.5 DCAO Algorithm

The complete DCAO algorithm can be formalized as follows:

**Input**: Initial parameters $w_0$, base learning rate $\eta_{\text{base}}$, base momentum $\mu_{\text{base}}$, base weight decay $\lambda_{\text{base}}$, curvature estimation frequency $K$, number of eigenvalues to estimate $k$, stability threshold $\beta$, and hyperparameters $\gamma$, $\alpha$, $\kappa$, $\phi$.

**Output**: Trained parameters $w_T$

**Algorithm**:
1. Initialize momentum buffer $m_0 = 0$
2. For $t = 0, 1, 2, ..., T-1$ do:
   1. Sample mini-batch $\mathcal{B}_t$ and compute gradient $g_t = \nabla f(w_t, \mathcal{B}_t)$
   2. If $t \mod K = 0$ then:
      1. Estimate top-$k$ eigenvalues $\{\lambda_1, \lambda_2, ..., \lambda_k\}$ and eigenvectors using stochastic Lanczos iteration
      2. Compute curvature metrics: $\rho_t = |\lambda_1|$, $\Delta_t = |\lambda_1 - \lambda_2|$, $\nu_t = \frac{1}{k}|\{i: \lambda_i < 0\}|$
      3. Adjust hyperparameters:
         - $\eta_t = \eta_{\text{base}} \cdot \min\left(1, \frac{\beta}{\rho_t}\right) \cdot \sqrt{1 + \gamma \Delta_t}$
         - $\mu_t = \mu_{\text{base}} + \alpha(1 - \exp(-\kappa \Delta_t))$
         - $\lambda_t = \lambda_{\text{base}} \cdot (1 + \phi \nu_t)$
   3. Else:
      1. Use previously computed $\eta_t$, $\mu_t$, $\lambda_t$
   4. Update momentum: $m_t = \mu_t m_{t-1} + g_t$
   5. Update parameters: $w_{t+1} = w_t - \eta_t m_t - \eta_t \lambda_t w_t$
3. Return $w_T$

### 2.6 Theoretical Analysis

We analyze the convergence properties of DCAO in both smooth and non-smooth settings. For smooth functions, we establish the following convergence result:

**Theorem 1**: Let $f$ be an $L$-smooth function and assume the spectral radius estimation has error bound $|\hat{\rho}_t - \rho_t| \leq \epsilon_{\rho}$. If the learning rate satisfies $\eta_t \leq \frac{\beta}{\rho_t + \epsilon_{\rho}}$ for all $t$, then DCAO with momentum $\mu_t \in [0, 1)$ converges to a stationary point with rate $O(1/T)$ for non-convex objectives.

For the Edge of Stability regime, we build on the self-stabilization theory of Damian et al. (2022) to analyze the implicit regularization effect:

**Theorem 2**: Under the EoS dynamics, DCAO implicitly performs projected gradient descent onto the manifold $\mathcal{M} = \{w : \lambda_{\max}(H(w)) \leq \beta/\eta_{\text{base}}\}$, with additional acceleration along directions with large spectral gaps.

### 2.7 Experimental Design

We will evaluate DCAO on the following tasks and architectures:

1. **Image classification**:
   - CIFAR-10 and CIFAR-100 using ResNet-18, ResNet-50, and Vision Transformer (ViT)
   - ImageNet using ResNet-50 and EfficientNet-B0

2. **Natural language processing**:
   - GLUE benchmark tasks using BERT-base
   - Machine translation on WMT14 using Transformer architectures
   - Fine-tuning GPT-2 on language modeling tasks

3. **Reinforcement learning**:
   - Deep Q-Network (DQN) on Atari games
   - Proximal Policy Optimization (PPO) on MuJoCo environments

For each task, we will compare DCAO against the following baselines:
- SGD with momentum
- Adam
- AdamW
- Adabelief
- Shampoo (as a representative second-order method)

We will conduct extensive hyperparameter sweeps for baseline methods to ensure fair comparison. For DCAO, we will fix the curvature estimation frequency $K$ based on computational budget considerations (typically 10-100 iterations) and the number of eigenvalues $k$ to a small value (5-20) to maintain efficiency.

### 2.8 Evaluation Metrics

We will evaluate the optimizers on the following metrics:

1. **Convergence speed**: Training and validation loss vs. iterations/wall-clock time
2. **Final performance**: Test accuracy, F1-score, BLEU score, or cumulative reward depending on the task
3. **Training stability**: Variance of the loss across multiple runs with different seeds
4. **Generalization**: Difference between training and test performance
5. **Robustness to hyperparameters**: Sensitivity to initial learning rate and other hyperparameters
6. **Computational overhead**: Additional time/memory required compared to first-order methods

Additionally, we will track and visualize the following metrics to gain insights into the optimization dynamics:

1. Maximum eigenvalue (spectral radius) of the Hessian throughout training
2. Spectral gap evolution
3. Learning rate, momentum, and weight decay adjustments
4. Correlation between curvature metrics and optimization performance

## 3. Expected Outcomes & Impact

### 3.1 Expected Scientific Outcomes

1. **Improved optimization dynamics**: We expect DCAO to demonstrate faster convergence and better final performance than standard first-order methods, particularly for complex architectures and tasks where curvature information provides significant advantages. By adapting to the local geometry of the loss landscape, DCAO should navigate challenging regions more effectively.

2. **Enhanced stability in the EoS regime**: A key anticipated outcome is more stable training at the Edge of Stability. Rather than oscillating unpredictably around the stability threshold, DCAO should maintain controlled progress through high-curvature regions by intelligently adjusting its hyperparameters.

3. **Better generalization properties**: By incorporating curvature information, DCAO is expected to find broader minima with better generalization properties. The dynamic weight decay adjustment should further enhance regularization, particularly near saddle points.

4. **Reduced sensitivity to initial hyperparameters**: The adaptive nature of DCAO should make it less sensitive to initial hyperparameter choices compared to static optimizers, potentially eliminating the need for extensive hyperparameter tuning.

5. **Empirical validation of theoretical insights**: The experimental results will provide empirical validation of theoretical predictions about optimization dynamics in the EoS regime, contributing to a better understanding of deep learning optimization.

### 3.2 Broader Impact

1. **Bridging theory and practice**: This research directly addresses the gap between optimization theory and deep learning practice by operationalizing theoretical insights about curvature, stability, and convergence into a practical algorithm. The findings will contribute to a more unified understanding of deep learning optimization.

2. **Practical efficiency gains**: By improving training stability and convergence speed, DCAO has the potential to reduce computational resources required for model training, making deep learning more accessible and environmentally sustainable.

3. **Foundation for future optimizers**: The framework for efficient curvature estimation and dynamic hyperparameter adjustment provides a foundation for developing more sophisticated optimization methods that leverage higher-order information without prohibitive computational costs.

4. **Enhanced model quality**: Improved optimization may lead to better-performing models across various domains, potentially enabling advances in computer vision, natural language processing, and reinforcement learning applications.

5. **Educational impact**: The insights gained from this research will enhance our understanding of deep learning optimization, benefiting educational curricula and training materials for researchers and practitioners.

### 3.3 Potential Challenges and Mitigations

1. **Computational overhead**: While stochastic Lanczos iteration is relatively efficient, curvature estimation still adds computational cost. We will carefully balance the frequency of curvature probing to minimize overhead while maintaining benefits. We will also explore GPU-optimized implementations and mixed-precision approaches.

2. **Noisy eigenvalue estimates**: Stochastic approximations of eigenvalues may be noisy, potentially leading to unstable hyperparameter adjustments. We will implement smoothing techniques, such as exponential moving averages of curvature metrics, to mitigate this issue.

3. **Scalability to very large models**: For extremely large models like GPT-3, even periodic curvature estimation may be challenging. We will investigate hierarchical approaches that estimate curvature on subsets of parameters or model components.

4. **Domain-specific tuning**: Different domains and architectures may benefit from different hyperparameter adjustment strategies. We will analyze domain-specific patterns and develop guidelines for adapting DCAO to various settings.

In conclusion, the Dynamic Curvature-Aware Optimizer represents a promising approach to bridge the gap between optimization theory and deep learning practice. By explicitly incorporating curvature information into the optimization process while maintaining computational efficiency, DCAO has the potential to improve training dynamics, enhance generalization, and advance our theoretical understanding of deep learning optimization.