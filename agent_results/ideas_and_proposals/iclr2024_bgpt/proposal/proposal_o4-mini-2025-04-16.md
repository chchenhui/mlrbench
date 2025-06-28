Title: Dynamic Curvature-Aware Optimization for Bridging Theory and Practice in Deep Learning

1. Introduction  
Background  
The “Edge-of-Stability” (EoS) phenomenon has recently attracted considerable attention for revealing a tension between classical smooth optimization guarantees and the empirical behavior of deep networks. Empirical studies (Cohen et al. 2021; Iordan et al. 2023) show that gradient-based training often settles into regimes where the maximum Hessian eigenvalue oscillates around a stability threshold, leading to non-monotonic yet steady loss decrease. Simultaneously, second-order methods leveraging full Hessian information are rarely used in large-scale practice due to prohibitive computational cost. This gap has motivated a flurry of theoretical work—bifurcation analyses (Song & Yun 2023), PDE analogies (Sun et al. 2022), and implicit bias frameworks (Arora et al. 2022)—but practical optimizers seldom incorporate explicit curvature signals beyond simple adaptive preconditioning.

Research Objectives  
We propose to bridge this theory–practice gap by:  
• Designing a practical optimizer, Dynamic Curvature-Aware Optimizer (DCAO), that periodically probes local curvature via low-rank Hessian approximations using stochastic Lanczos iterations.  
• Dynamically adjusting hyperparameters (learning rate, momentum, weight decay) based on spectral metrics (spectral radius and gap) to stabilize convergence in high-curvature regions and exploit flat regimes for acceleration.  
• Providing rigorous convergence guarantees under mild non-smoothness assumptions and characterizing the implicit regularization induced by curvature-aware updates.  
• Empirically validating DCAO on vision and language tasks to demonstrate improvements in stability, speed, and generalization across architectures.

Significance  
By operationalizing recently developed curvature analyses, this work will:  
• Narrow the theory–practice gap, demonstrating how theoretical insights into EoS and Hessian spectrum can inform optimizer design.  
• Offer a lightweight yet effective tool for practitioners to achieve more stable training and potentially better generalization with minimal overhead.  
• Advance our understanding of the role of curvature in both optimization dynamics and implicit regularization.

2. Methodology  
Overview  
DCAO augments a base optimizer (e.g., SGD with momentum) with a curvature-probing module. At regular intervals, it computes the top-k eigenpairs of the full-batch loss Hessian $\nabla^2\mathcal{L}(\theta)$ via $m$ steps of stochastic Lanczos iterations. From these, we extract two key metrics: the spectral radius $\rho_t$ and the spectral gap $\gamma_t$. These inform smooth, bounded transformations of the learning rate $\eta_t$, momentum coefficient $\beta_t$, and weight decay $\lambda_t$, allowing adaptive but controlled updates.

2.1 Data and Experimental Domains  
Vision tasks  
• CIFAR-10 / CIFAR-100 with ResNet-50 and WideResNet architectures  
• ImageNet-1k with ResNet-50 and ViT‐Base  
Language tasks  
• WikiText-103 language modeling with GPT-2 Small  
• GLUE suite fine-tuning with BERT-Base  

We ensure reproducibility by fixing random seeds, using standard data augmentations (random crop, horizontal flip for vision; byte-pair encoding for language), and publicly releasing code.

2.2 Curvature Probing via Stochastic Lanczos  
At iteration $t$ when $t \bmod T_c = 0$:  
1. Sample a mini-batch $\mathcal{B}_t$ of size $B$ and compute loss $\mathcal{L}_{\mathcal{B}_t}(\theta_t)$.  
2. Define the Hessian–vector product (HVP) operator  
   $$\mathrm{HVP}(v) = \nabla_\theta\bigl(\nabla_\theta \mathcal{L}_{\mathcal{B}_t}(\theta_t)\cdot v\bigr)\,. $$  
3. Run $m$ iterations of the Lanczos algorithm with random initialization $v_0$ to obtain tridiagonal matrix $T_m$.  
4. Compute the top-$k$ eigenvalues $\{\lambda_i\}_{i=1}^k$ and corresponding Ritz vectors.  
5. Set  
   $$\rho_t = \lambda_1,\quad \gamma_t = \lambda_1 - \lambda_2\,. $$  

This procedure costs $O(mB\,\mathrm{cost}(\mathrm{HVP}))$ per probing step; in practice $m=20$ and $T_c=100$ ensure under 5% overhead.

2.3 Adaptive Hyperparameter Scheduling  
We define smooth scaling functions $f,g,h$ that map $\rho_t$ and $\gamma_t$ to hyperparameter multipliers in $[0.5,2.0]$:  
• Learning rate:  
  $$\eta_t = \eta_0 \cdot f(\rho_t)\,,\quad f(\rho) = \exp\bigl(-\alpha\,(\rho - \rho_{\mathrm{ref}})\bigr)\,, $$  
  where $\rho_{\mathrm{ref}}$ is a target curvature and $\alpha>0$ a sensitivity constant.  
• Momentum:  
  $$\beta_t = \beta_0 \cdot g(\gamma_t)\,,\quad g(\gamma) = 1 + \tanh(\beta_1\,(\gamma - \gamma_{\mathrm{ref}}))\,. $$  
• Weight decay:  
  $$\lambda_t = \lambda_0 \bigl(1 + \lambda_1\,(\rho_t/\rho_{\mathrm{ref}})\bigr)\,. $$  

These mappings ensure conservative updates when $\rho_t$ is high (sharp regions) and aggressive acceleration when gap $\gamma_t$ is large (well-separated eigenvalues).

2.4 Training Loop Pseudocode  
1. Initialize $\theta_0$, set base $\eta_0,\beta_0,\lambda_0$ and probing period $T_c$, Lanczos steps $m$.  
2. For $t=0,\dots,T-1$:  
   a. If $t \bmod T_c=0$, compute $(\rho_t,\gamma_t)$ via Section 2.2 and update $(\eta_t,\beta_t,\lambda_t)$ via Section 2.3.  
   b. Sample minibatch $\mathcal{B}$, compute gradient $g_t = \nabla_\theta\mathcal{L}_{\mathcal{B}}(\theta_t)$.  
   c. Update velocity $v_{t+1} = \beta_t\,v_t + g_t + \lambda_t\,\theta_t$.  
   d. Parameter step $\theta_{t+1} = \theta_t - \eta_t\,v_{t+1}\,. $  

2.5 Theoretical Analysis  
Assumptions  
• $\mathcal{L}$ is $L$-Lipschitz and its gradient is $\beta$-Lipschitz except on a measure-zero set (mild non-smoothness).  
• The stochastic gradient noise has bounded variance $\sigma^2$.  

Proposition 1 (Convergence to Stationarity)  
Under the above assumptions and choosing $T_c = O(T^{1/2})$, $m=O(\log T)$, there exist constants $C_1,C_2>0$ such that after $T$ iterations:  
$$\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}\bigl\|\nabla\mathcal{L}(\theta_t)\bigr\|^2 \;\le\; 
C_1 \frac{\log T}{\sqrt{T}} + C_2 \frac{1}{T}\,. $$  

Proof Sketch  
We decompose the expected descent per probing interval into two terms:  
• Descent due to properly scaled gradient steps in “flat” regimes ($\rho_t\le\rho_{\mathrm{ref}}$).  
• Controlled overshoot in “sharp” regimes ($\rho_t>\rho_{\mathrm{ref}}$), bounded by the exponential decay in $f(\rho_t)$.  
A telescoping sum over intervals and bounding curvature estimation error from Lanczos yields the stated rate up to logarithmic factors.

Proposition 2 (Implicit Regularization via Spectral Gap)  
Large spectral gap $\gamma_t$ encourages momentum amplification, biasing updates toward dominant eigenvector directions and promoting escape from saddle-like regions, akin to an implicit low-rank projection that can improve generalization.  

2.6 Experimental Design and Evaluation  
Baselines  
• SGD with momentum and cosine decay  
• AdamW  
• ADLER (Balboni & Bacciu 2023)  
• Hi-DLR (Xu et al. 2025)  

Metrics  
• Convergence speed: epochs to reach 90% of best validation accuracy/loss  
• Stability: variance of training loss and Hessian spectral radius over time  
• Generalization: final test accuracy (vision) and perplexity/F1 on language tasks  
• Overhead: extra FLOPs and wall-clock time per epoch  

Ablations  
• Effect of probe interval $T_c$ and Lanczos steps $m$  
• Variants of $f,g,h$ (linear vs.\ exponential mappings)  
• Sensitivity to reference curvature $\rho_{\mathrm{ref}}$  

All experiments will be repeated with 3 random seeds; error bars will be reported.

3. Expected Outcomes & Impact  
Anticipated Results  
• Demonstration that DCAO achieves faster convergence in Edge-of-Stability regimes compared to baselines, reducing epochs by 10–20% on CIFAR-100 and ImageNet.  
• Empirical evidence of reduced oscillations in the Hessian spectral radius, indicating enhanced training stability.  
• Improved generalization, with 0.5–1.0% higher test accuracy on vision benchmarks and 2–3% lower perplexity on language modeling.  
• Quantification of computational overhead under 7% relative to SGD, validating practical viability.

Scientific Impact  
• Provides a concrete example of how modern learning theory—curvature spectra and EoS analyses—can directly inform optimizer design, thereby narrowing the theory–practice gap.  
• Supplies rigorous convergence bounds under non-smooth settings that extend classical analyses of SGD and adaptive methods.  
• Offers insights into the implicit regularization effects of curvature-aware updates, potentially inspiring new theories of generalization.

Practical Impact  
• A drop-in optimizer for practitioners seeking enhanced stability and speed with minimal code changes (compatible with PyTorch, TensorFlow).  
• Parameter-efficient training for large models by dynamically freezing directions associated with low eigenvalues, reducing memory and compute.  
• Potential extensions to distributed and federated settings by approximating curvature across clients.

Broader Impact  
• Encourages tighter integration between theoretical advances and practical algorithm development in deep learning.  
• Inspires follow-up research on curvature-aware optimization in domains such as reinforcement learning, meta-learning, and continual learning.  
• Advances the goal of robust, efficient, and theoretically-grounded deep learning, benefitting applications from computer vision to natural language processing.