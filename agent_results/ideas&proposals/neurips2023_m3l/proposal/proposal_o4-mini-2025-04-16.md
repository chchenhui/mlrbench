Title  
Dynamical Insights into Edge of Stability Optimization for Large‐Scale Deep Learning  

1. Introduction  

Background  
Over the past decade, deep learning has driven breakthroughs across computer vision, natural language processing, and reinforcement learning.  However, training modern neural networks—often comprising billions or even trillions of parameters—remains more an empirical art than a principled science.  Classical convergence theory for gradient‐based methods prescribes small step sizes to guarantee monotonic descent, yet practitioners routinely train state-of-the-art vision and language models with very large learning rates and high levels of stochastic noise.  Recent empirical work by Cohen et al. (2021) shows that full-batch gradient descent often operates at the so‐called Edge of Stability (EoS), where the top Hessian eigenvalue $\lambda_{\max}$ hovers just above $2/\eta$ (with $\eta$ the step size), yielding non-monotonic short-term loss behavior but long‐term convergence.  Arora et al. (2022) and subsequent analyses have begun to uncover the implicit bias induced by EoS dynamics, but a comprehensive theoretical framework remains elusive.

Research Objectives  
  • Develop a continuous‐time mathematical framework—via stochastic differential equations (SDEs)—that captures the interaction between large learning rates, gradient noise, and curvature in nonconvex deep‐net training.  
  • Characterize the stability boundary (EoS) analytically in terms of Hessian spectra and noise amplitudes, deriving conditions under which discrete‐time training with large $\eta$ remains stable.  
  • Design and implement an adaptive optimization algorithm (EoS‐SGD) that automatically tracks the stability boundary, modulating step sizes and noise schedules to stay at EoS without divergence.  
  • Empirically validate on large‐scale vision and language models (e.g.\ ResNet‐50, Vision Transformer, Transformer encoder–decoders) to demonstrate 2–3× speedups in time-to-accuracy and significant energy savings.  

Significance  
A predictive theory of EoS will transform deep‐learning practice by replacing costly trial-and-error with principled hyperparameter schedules.  In the large‐model era, this can slash GPU‐hour budgets, decrease carbon footprints, and democratically empower smaller labs to train foundation models.  

2. Methodology  

We propose a hybrid theoretical–empirical research design comprising four components: (1) continuous‐time modeling, (2) stability analysis, (3) adaptive algorithm design, and (4) large‐scale empirical validation.  

2.1 Continuous‐Time Modeling of Discrete Updates  
We model stochastic gradient descent (SGD) with step size $\eta$ and gradient noise as the following SDE in parameter space $\theta \in \mathbb{R}^d$:  
$$
d\theta_t \;=\; -\,\nabla L(\theta_t)\,dt \;+\;\sqrt{\tfrac{\eta\,\sigma^2(\theta_t)}{2}}\;dW_t,
$$  
where $L(\theta)$ is the empirical loss, $\sigma^2(\theta)$ quantifies local gradient‐noise variance, and $W_t$ is a standard $d$‐dimensional Wiener process.  Under mild regularity, this approximation is accurate up to $O(\eta^2)$ over time scales $O(1)$.  We will also consider the Fokker–Planck equation for the density $p(\theta,t)$ to study stationary distributions near minima.  

2.2 Stability Boundary (EoS) Analysis  
Locally near a minimum $\theta^*$, write $L(\theta)\approx L(\theta^*) + \tfrac12(\theta-\theta^*)^T H(\theta^*)(\theta-\theta^*)$.  In this quadratic regime, the dynamics decouple along principal directions of $H$.  For a single direction with curvature $\lambda$, the discrete‐time update  
$$
\theta_{k+1} = \theta_k - \eta\,\lambda\,(\theta_k-\theta^*) + \sqrt{\eta}\,\xi_k
$$  
exhibits stability if and only if $\lambda$ lies below the root‐magnitude threshold of the update matrix.  Classical analysis gives $\eta\,\lambda < 2$, but with noise and nonlinearity one observes EoS behavior at $\eta\,\lambda \gtrsim 2$.  We will:  
  • Derive refined stability conditions by applying stochastic averaging and Floquet theory to the SDE, yielding a corrected boundary  
  $$
  \eta\,\lambda_{\max} \;\approx\; 2 + c\,\sqrt{\eta\,\sigma^2(\theta^*)},
  $$  
  where $c$ depends on higher‐order terms in the loss expansion.  
  • Generalize beyond the quadratic regime using perturbation analysis to account for nonzero third derivatives.  
  • Prove, under realistic smoothness and noise‐moment assumptions, that SGD remains stable and convergent on a time scale $O(\eta^{-1})$ when operating at or slightly above the classical boundary (“soft” EoS).  

2.3 Adaptive EoS‐SGD Algorithm  
Building on the above theory, we design EoS‐SGD, which dynamically tracks local curvature and noise to select $\eta_t$ and noise injection levels.  Key steps per iteration $t$:  
1. Compute stochastic gradient $g_t = \nabla L(\theta_t) + \xi_t$.  
2. Estimate top Hessian eigenvalue $\lambda_{\max}(\theta_t)$ via a single‐vector power iteration or Lanczos with $k=2$ steps, costing $O(d)$ additional matvecs.  
3. Estimate local noise variance $\hat\sigma^2_t$ by exponential‐moving‐average of $\|g_t - \nabla\widehat{L}(\theta_t)\|^2$.  
4. Set step size  
   $$
   \eta_t = \min\Bigl\{\frac{2 + c\sqrt{\eta_{t-1}\hat\sigma^2_t}}{\lambda_{\max}(\theta_t)}\,,\;\eta_{\max}\Bigr\},
   $$  
   where $c$ is calibrated via our theoretical bound and $\eta_{\max}$ is a safeguard.  
5. Update  
   $$
   \theta_{t+1} = \theta_t - \eta_t\,g_t.
   $$  
Pseudocode:  
```
Input: θ₀, η₀, ηₘₐₓ, c, momentum μ
for t = 0 to T-1:
  gₜ ← ∇L(θₜ) + noise()
  λₘₐₓ ← EstimateTopEigen(H(θₜ))
  σ̂²ₜ ← UpdateNoiseEstimate(gₜ)
  ηₜ ← min{ (2 + c√(ηₜ₋₁ σ̂²ₜ)) / λₘₐₓ , ηₘₐₓ }
  θₜ₊₁ ← θₜ - ηₜ gₜ + μ(θₜ - θₜ₋₁)
end for
```
This algorithm requires only a small constant overhead per iteration and automatically adapts to curvature and noise levels.  

2.4 Experimental Design and Evaluation Metrics  
We will validate EoS‐SGD on three representative large‐scale settings:  
• Image classification on ImageNet‐1K using ResNet‐50 and Vision Transformer (ViT‐S).  
• Neural Machine Translation on WMT’14 En→De with Transformer‐Base.  
• Masked language modeling on Wikipedia+BooksCorpus with BERT‐Base.  

Baselines: SGD with momentum, Adam, LAMB, QHAdam.  

Implementation & Infrastructure:  
  – Framework: PyTorch + custom CUDA kernels for Hessian matvecs.  
  – Hardware: 8×A100 GPU nodes with NVLink.  
  – Repetitions: 3 independent runs per configuration, same random seeds for comparability.  

Evaluation Metrics:  
  – Training time to reach predefined loss thresholds.  
  – Epochs needed to achieve target top‐1/accuracy or perplexity.  
  – Final validation performance.  
  – Peak Hessian eigenvalue trajectories.  
  – GPU‐hour and kilowatt‐hour consumption (via on‐node power meters).  
  – Statistical significance (paired t‐tests at 95% confidence).  

Ablations: We will study variants that omit curvature estimation or noise‐adaptive terms to isolate each component’s contribution.  

3. Expected Outcomes & Impact  

3.1 Theoretical Contributions  
  • A rigorous continuous‐time framework connecting discrete SGD at large $\eta$ to an SDE that captures EoS phenomena.  
  • Closed‐form characterization of the stochastic stability boundary:  
    $$
    \lambda_{\max} \;<\; \frac{2 + c\,\sqrt{\eta\,\sigma^2}}{\eta}\,.
    $$  
  • Convergence theorems guaranteeing that EoS‐SGD with dynamic $\eta_t$ converges to a stationary point at rate $O(1/\sqrt{T})$ under nonconvex assumptions.  

3.2 Algorithmic Deliverables  
  • Open‐source implementation of EoS‐SGD with plug-and-play APIs for PyTorch.  
  • Extensive empirical evidence showing 2–3× faster time-to-target performance compared to strong baselines, with 20–30% reduction in energy consumption.  
  • Practical recipes and hyperparameter guidelines for training at EoS, including recommended values of $c$, $\eta_{\max}$, and curvature‐estimation frequency.  

3.3 Broader Impacts  
By grounding large‐scale training in principled theory and delivering a robust optimizer, this work will:  
  – Lower the entry barrier for academic and industry groups lacking massive compute budgets.  
  – Decrease the carbon footprint of training foundation models, aligning with green AI initiatives.  
  – Inspire further research on continuous approximations of discrete learning dynamics, fostering a virtuous cycle between theory and practice.  

4. Timeline & Milestones  

Months 1–4  
  • Develop continuous‐time SDE model; derive preliminary stability conditions.  
  • Prototype Hessian‐eigenvalue estimator and noise‐variance tracker.  

Months 5–8  
  • Finalize theoretical proofs for the corrected stability boundary and convergence rates.  
  • Implement core EoS‐SGD algorithm in PyTorch.  

Months 9–12  
  • Run small‐scale experiments on CIFAR-10 and WMT Toy sets to debug and tune.  
  • Prepare workshop/tutorial materials on EoS phenomena.  

Months 13–18  
  • Scale experiments to ImageNet, WMT’14, and BERT pretraining; collect metrics.  
  • Conduct ablation studies and finalize hyperparameter guidelines.  

Months 19–24  
  • Polish code for public release; write and submit conference/journal papers.  
  • Organize an open‐source workshop to disseminate findings and gather community feedback.  

5. Conclusion  

This proposal bridges deep learning theory and large-scale practice by providing the first unified dynamical analysis of the Edge of Stability phenomenon and delivering an adaptive optimizer that exploits EoS for faster, more efficient training.  The combination of rigorous SDE‐based stability analysis, lightweight curvature estimation, and large‐scale empirical validation promises both theoretical advances and immediate practical impact in the era of foundation models.