1. Title  
DiffPDE: Diffusion-Based Neural Solvers for High-Dimensional Partial Differential Equations  

2. Introduction  
Background  
Solving partial differential equations (PDEs) in high dimensions arises in a variety of applications—ranging from turbulent fluid dynamics and quantum chemistry to option pricing in mathematical finance. Classical grid-based solvers (finite differences, finite elements, spectral methods) suffer from the “curse of dimensionality,” with computational cost growing exponentially in the number of spatial dimensions. Recent deep-learning-based approaches—Physics-Informed Neural Networks (PINNs), neural operators (FNOs, DeepONets), and latent PDE solvers—have shown promise in bypassing some of these limitations by leveraging function approximators that generalize across parameters or geometries.  

Parallelly, score-based diffusion models and stochastic differential equation (SDE) solvers have achieved SOTA results in high-dimensional generative modeling tasks. These models learn to reverse a diffusion (forward ­SDE) by approximating the score (gradient of the log-density) at each time step, enabling the generation of complex, high-dimensional distributions. A natural question arises: can we fuse diffusion modeling and PDE constraints to build neural solvers that scale gracefully with dimension and automatically satisfy physical laws?  

Research Objectives  
We propose DiffPDE, a diffusion-based neural PDE solver that:  
• Embeds the PDE’s differential operators into the forward diffusion process to structure the noise injection according to the underlying dynamics.  
• Learns a reverse diffusion (denoising) network via score matching, augmented by physics-informed residuals that enforce PDE constraints and boundary/initial conditions.  
• Generalizes across PDE parameter distributions without retraining, delivering solutions in dimensions up to and beyond 100 with competitive accuracy and significantly reduced runtime compared to PINNs and neural operators.  

Significance  
By marrying the scalability of diffusion models with the rigor of physics-informed learning, DiffPDE stands to:  
• Alleviate the curse of dimensionality for a broad class of parametric, high-dimensional PDEs.  
• Provide uncertainty quantification via the stochastic generation process, yielding credible intervals for solutions.  
• Serve as a unified framework for forward and inverse PDE problems in science and engineering, with applications in fluid simulation, stochastic control, and quantitative finance.  

3. Methodology  
3.1 Problem Setup and Data Generation  
We consider families of parametric PDEs  
$$\mathcal{L}_{\lambda}[u](x) = f(x),\quad x \in \Omega\subset\mathbb R^d,\; \lambda\in\Lambda,$$  
with boundary or initial‐value conditions  
$$\mathcal{B}_{\lambda}[u](x)=g(x),\quad x\in\partial\Omega.$$  
Here $d$ may range from 10 to 200 or more, and $\lambda$ encodes diffusion coefficients, convection fields, or payoff parameters (e.g., strike price in Black–Scholes). We generate training data by sampling $\lambda\sim p(\lambda)$ and solving the PDE for $u^*(x;\lambda)$ at a moderate grid resolution using sparse spectral collocation or Monte Carlo solvers in moderate dimensions ($d\le50$). These high-quality solutions $u^*(\cdot;\lambda)$ serve as “ground truth” to supervise the diffusion network via score matching and to evaluate residual losses.  

3.2 Forward Diffusion Process with PDE Drift  
We define a forward Itô SDE on the PDE solution manifold: for each parameter $\lambda$,  
$$
\mathrm{d}x_t = \mathcal{L}_{\lambda}[x_t]\,\mathrm{d}t + \sqrt{2\beta(t)}\,\mathrm{d}W_t,\quad x_0 = u^*(\cdot;\lambda),
$$  
where $W_t$ is a $d$-dimensional spatial Wiener process, $\beta(t)>0$ is a pre-specified noise schedule, and $\mathcal{L}_{\lambda}$ acts pointwise on $x_t(x)$ as the PDE operator. This choice of drift embeds the physics into the diffusion, so that the forward trajectory $x_t$ smoothly transitions from the true PDE solution toward a near-Gaussian distribution at $t=T$.  

3.3 Reverse-Time SDE and Score Network  
Under mild regularity, the reverse-time SDE is  
$$
\mathrm{d}x_t = \Bigl[-\mathcal{L}_{\lambda}[x_t] - \beta(t)\,\nabla_{x_t}\log p_t(x_t\mid\lambda)\Bigr]\mathrm{d}t + \sqrt{2\beta(t)}\,\mathrm{d}\bar W_t,
$$  
where $\bar W_t$ is a reverse-time Wiener process. We approximate the score $\nabla_{x_t}\log p_t(\cdot\mid\lambda)$ by a neural network $s_\theta(x_t,t;\lambda)$.  

3.4 Hybrid Training Loss  
We train $s_\theta$ to minimize a combination of  
1. Score-Matching Loss  
   $$\mathcal{L}_{\mathrm{SM}}(\theta) 
   = \mathbb{E}_{t,\lambda,x_0,\epsilon}
   \bigl\|\,s_\theta(x_t,t;\lambda)\;-\;\nabla_{x_t}\log q_t(x_t\mid x_0)\bigr\|^2, 
   $$  
   where $x_t = x_0 + \int_0^t \mathcal{L}_{\lambda}[x_s]\,ds + \sqrt{2}\!\int_0^t\!\!\!\sqrt{\beta(s)}\,dW_s$, and $\nabla_{x_t}\log q_t$ is known in closed form for this linear SDE.  
2. PDE Residual Loss  
   We require that the denoised output $x_0^\theta \approx u^*$ satisfies the PDE:  
   $$R_\lambda(x_0^\theta)(x) \triangleq \mathcal{L}_{\lambda}[x_0^\theta](x) - f(x)\,,\quad x\in\Omega.$$  
   We minimize  
   $$\mathcal{L}_{\mathrm{res}}(\theta)
   = \mathbb{E}_{\lambda,x}\bigl\|R_\lambda\bigl(x_0^\theta\bigr)(x)\bigr\|^2_{L^2(\Omega)}
   + \mathbb{E}_{\lambda,x\in\partial\Omega}\bigl\|\mathcal{B}_{\lambda}[x_0^\theta](x)-g(x)\bigr\|^2.$$

The total loss is  
$$
\mathcal{L}(\theta)
= \mathcal{L}_{\mathrm{SM}}(\theta) 
+ \alpha\,\mathcal{L}_{\mathrm{res}}(\theta),
$$  
with $\alpha>0$ balancing generative quality and physics fidelity.  

3.5 Network Architecture  
We use a U-shaped score network $s_\theta$ with the following modules:  
• Positional Encoding of $t$ and $\lambda$: Fourier‐feature embeddings followed by MLPs.  
• Spatial Feature Extractor: a Fourier Neural Operator (FNO) backbone to capture global interactions efficiently in high $d$.  
• Time-Conditioned Residual Blocks: each block ingests the current time embedding and parameter embedding.  

This architecture scales linearly in $d$ (via factorized Fourier transforms) and employs attention briefly at bottleneck layers to handle long‐range dependencies.  

Algorithmic Steps  
1. Sample mini-batch $\{\lambda_i\}_{i=1}^B$, ground truths $\{u^*_i\}$.  
2. For each $\lambda_i$, sample random $t\sim\mathcal U(0,T)$ and Wiener increment $\epsilon\sim\mathcal N(0,I)$.  
3. Compute $x_t$ by one-step Euler–Maruyama of the forward SDE.  
4. Evaluate score-matching target $\nabla_{x_t}\log q_t(x_t\mid u^*)$.  
5. Compute $\mathcal{L}_{\mathrm{SM}}$ and $\mathcal{L}_{\mathrm{res}}$ (via collocation on domain and boundary).  
6. Backpropagate total loss; update $\theta$.  

3.6 Sampling / Inference  
Given a new $\lambda'$:  
1. Initialize $x_T\sim\mathcal N(0,\sigma^2I)$.  
2. Numerically integrate the reverse SDE via a high-order solver (e.g., Heun’s method), using network evaluations of $s_\theta(x_t,t;\lambda')$.  
3. At $t=0$, return $x_0\approx u^*(\cdot;\lambda')$.  

3.7 Experimental Design and Evaluation Metrics  
Benchmarks  
• Heat Equation in $d=10,50,100$ on $\Omega=[0,1]^d$, with random initial data.  
• Black–Scholes PDE in $d=100$ for European basket options.  
• Nonlinear convection–diffusion PDE in $d=50$.  

Baselines  
• PINNs (Raissi et al.)  
• Fourier Neural Operator (Li et al.)  
• Spectral solver (sparse grid collocation, when feasible)  

Metrics  
• Relative $L^2$ error:  
  $$\mathrm{err}_{L^2} = \frac{\|u^* - u_\theta\|_{L^2(\Omega)}}{\|u^*\|_{L^2(\Omega)}}.$$  
• PDE residual norm $\|R_\lambda(u_\theta)\|_{L^2}$.  
• Computational cost: wall‐clock time and memory usage.  
• Uncertainty quantification: empirical coverage of $95\%$ credible intervals from multiple reverse‐SDE samples.  

Ablations  
• Impact of PDE-drift in forward SDE vs. standard (`drift=0') diffusion.  
• Varying $\alpha$ to trade off generative fidelity and physics residual.  
• Architecture choices: FNO backbone vs. plain convolutional net.  

4. Expected Outcomes & Impact  
Expected Outcomes  
• A unified diffusion-based framework (DiffPDE) that solves families of high-dimensional PDEs with relative $L^2$ errors below $1\%$ in 50–100 dimensions, outperforming PINNs by an order of magnitude in speed while matching or exceeding accuracy.  
• Demonstration that embedding the PDE operator in the forward diffusion yields faster convergence and improved generalization across parameters, compared to vanilla score‐based models.  
• Empirical evidence that the stochastic sampling process naturally yields calibrated uncertainty estimates, addressing a major gap in deterministic PDE solvers.  
• Open‐source implementation, with thorough documentation and scripts to reproduce experiments.  

Broader Impact  
Solving high-dimensional PDEs efficiently is a central bottleneck in numerous scientific and engineering domains. DiffPDE’s scalability will enable:  
• Real-time simulation and control in robotics and fluid mechanics (e.g., aerodynamic shape optimization).  
• Faster, more accurate pricing of complex financial derivatives in dimension >50, with built-in risk measures via uncertainty quantification.  
• Accelerated design cycles in materials science and climate modeling, where parameter sweeps over high-dimensional state spaces are routine.  

By releasing code and pretrained models, we foster adoption by both academic researchers and industry practitioners. The diffusion-PDE hybrid paradigm may also spur new theoretical insights at the intersection of stochastic processes, numerical analysis, and deep learning.