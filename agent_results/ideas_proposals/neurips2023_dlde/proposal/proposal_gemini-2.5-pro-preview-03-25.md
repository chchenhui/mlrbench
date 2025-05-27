**1. Title:** PDE-Constrained Diffusion Models for Solving High-Dimensional Partial Differential Equations

**2. Introduction**

**2.1. Background**
Partial Differential Equations (PDEs) are fundamental mathematical tools used to model a vast array of complex phenomena across diverse scientific and engineering disciplines, including fluid dynamics, heat transfer, quantum mechanics, materials science, and financial mathematics. Solving these equations, particularly in high-dimensional settings, remains a significant computational challenge. Traditional numerical methods, such as Finite Difference Methods (FDM), Finite Element Methods (FEM), and spectral methods, often suffer from the "curse of dimensionality," where the computational cost grows exponentially with the number of spatial or parameter dimensions, rendering them intractable for many important problems (e.g., solving the Black-Scholes equation in finance with many underlying assets, or simulating quantum systems with many particles).

In recent years, the deep learning community has made remarkable strides in developing neural network-based approaches for solving PDEs. Methods like Physics-Informed Neural Networks (PINNs) [Raissi et al., 2019] embed the residual of the PDE and boundary conditions directly into the loss function, allowing neural networks to approximate solutions in a mesh-free manner. Neural Operators, such as the Fourier Neural Operator (FNO) [Li et al., 2020], learn mappings between function spaces, enabling efficient solution predictions for varying initial conditions or parameters. While these approaches have shown considerable promise, they often face challenges related to training convergence, scalability to very high dimensions (often requiring substantial computational resources), and efficient handling of complex geometries or boundary conditions.

Simultaneously, diffusion models [Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020] have emerged as state-of-the-art generative models, demonstrating exceptional performance in generating high-fidelity images, audio, and other complex, high-dimensional data. These models are rooted in the mathematics of stochastic differential equations (SDEs) or Markov chains, defining a forward process that gradually adds noise to data and learning a reverse process to map noise back to data samples. Their success in handling high-dimensional distributions naturally suggests their potential applicability to solving high-dimensional PDEs, where the solution itself can be viewed as a complex, structured data distribution. The connection between diffusion models and differential equations is deep, forming a central theme in the "Symbiosis of Deep Learning and Differential Equations" research area.

**2.2. Problem Statement and Research Gap**
Despite progress in neural PDE solvers, efficiently and accurately solving PDEs in regimes with dimensionality $d \gg 10$ (e.g., $d > 100$) remains a formidable task. PINNs can struggle with training dynamics and optimization landscapes, especially for stiff or highly nonlinear problems. Neural operators, while powerful for learning solution maps, may require significant amounts of training data (paired input-output function data) and their performance can degrade when extrapolating far beyond the training distribution of parameters or initial conditions. Furthermore, explicitly incorporating the underlying physics or mathematical structure of the PDE directly into the generative process of high-capacity models like diffusion models is an area ripe for exploration. Existing works combining diffusion and PDEs (e.g., LatentPINNs [Taufik et al., 2023], Physics-Informed Diffusion Models [Johnson & Lee, 2024], Diffusion-based Operators [Haitsiukevich et al., 2024]) have started exploring this intersection, often focusing on latent space representations, specific PDE classes, or using diffusion primarily for parameter conditioning or generating diverse solutions. However, a framework that directly leverages the power of diffusion models operating in the full, high-dimensional solution space, guided explicitly by PDE constraints throughout the denoising process, particularly targeting scalability to hundreds of dimensions, warrants further investigation.

**2.3. Proposed Approach**
This research proposes a novel framework, termed **PDE-Constrained Diffusion Solver (PCDS)**, designed to solve high-dimensional PDEs by integrating PDE physics directly into the generative process of a diffusion model. We treat the PDE solution field $u(x)$ (or $u(x, t)$ discretized over time or treated implicitly) as data points in a high-dimensional space. The core idea is to formulate the PDE solving task as a conditional denoising process.
A standard forward diffusion process gradually corrupts an initial guess or representation of the solution field with noise, typically evolving towards a simple prior distribution (e.g., Gaussian noise). The key innovation lies in the training of the reverse process. We train a time-dependent neural network (score network) $s_\theta$ to approximate the score function (gradient of the log probability density) of the noisy data distribution at different noise levels. Crucially, this training is guided by a **hybrid loss function** that combines:
1.  **Score Matching Objective:** Enforces that the learned reverse process accurately reconstructs the data distribution by matching the score of the true conditional data distribution.
2.  **PDE Residual Constraint:** Explicitly penalizes deviations from the governing PDE and its associated boundary/initial conditions. This term is evaluated on the denoised estimate of the solution predicted by the score network at various noise levels.

This approach aims to leverage the proven scalability of diffusion models for high-dimensional distributions while ensuring the generated solutions adhere to the underlying physics laws encoded by the PDE. By operating potentially directly on the high-dimensional solution field representation (e.g., function values on a grid or collocation points), we aim to bypass the limitations of grid refinement inherent in traditional methods and potentially improve scalability compared to existing neural solvers.

**2.4. Research Objectives**
The primary objectives of this research are:
1.  **Develop the PCDS Framework:** Formulate the mathematical and algorithmic details of the PDE-Constrained Diffusion Solver, including the design of the forward process, the architecture of the score network, and the precise form of the hybrid loss function incorporating score matching and PDE residuals.
2.  **Implement and Train the Model:** Implement the PCDS framework using modern deep learning libraries (e.g., PyTorch, JAX) and develop efficient training strategies suitable for high-dimensional state spaces.
3.  **Validate on Benchmark High-Dimensional PDEs:** Evaluate the accuracy, computational efficiency, and scalability of PCDS on a set of challenging high-dimensional PDEs, including linear (e.g., Heat Equation) and nonlinear (e.g., Black-Scholes, potentially simplified Navier-Stokes related problems) examples, with dimensions ranging up to $d=100$ and beyond.
4.  **Comparative Analysis:** Rigorously compare the performance of PCDS against relevant baselines, including state-of-the-art PINNs, Neural Operators (e.g., FNO), traditional numerical methods (in lower dimensions where feasible), and potentially other diffusion-based PDE solvers (e.g., Johnson & Lee, 2024).
5.  **Investigate Parameterized PDEs:** Explore the capability of PCDS to solve parameterized PDEs, aiming for generalization across unseen parameter values without retraining.

**2.5. Significance**
This research holds significant potential for advancing the field of scientific Ccmputing and machine learning. Successfully developing PCDS would provide a powerful new tool for tackling previously intractable high-dimensional PDE problems critical to scientific discovery and engineering design. It would represent a significant contribution to the growing synergy between deep learning and differential equations, showcasing how sophisticated generative models like diffusion models can be effectively constrained by physical laws. The potential impacts include accelerating simulations in computational finance (e.g., high-dimensional option pricing), statistical physics, quantum chemistry, and fluid dynamics. Furthermore, insights gained from integrating PDE constraints into the diffusion process could feedback into the development of more structured and controllable generative models beyond the PDE domain.

**3. Methodology**

**3.1. Problem Formulation**
We consider a general form of a PDE defined over a spatial domain $\Omega \subset \mathbb{R}^d$ and potentially a time interval $[0, T]$:
$$
\mathcal{N}[u](x, t; \mathcal{P}) = f(x, t) \quad \text{for } x \in \Omega, t \in (0, T]
$$
subject to boundary conditions (BCs):
$$
\mathcal{B}[u](x, t; \mathcal{P}) = g(x, t) \quad \text{for } x \in \partial\Omega, t \in [0, T]
$$
and initial conditions (ICs) if time-dependent:
$$
\mathcal{I}[u](x; \mathcal{P}) = u_0(x) \quad \text{for } x \in \Omega, t = 0
$$
Here, $u(x, t)$ is the unknown solution field, $\mathcal{N}$ is a differential operator (potentially nonlinear), $\mathcal{B}$ defines the boundary conditions, $\mathcal{I}$ defines the initial condition, $f$, $g$, $u_0$ are given functions, and $\mathcal{P}$ represents a set of parameters defining the specific PDE instance (e.g., coefficients like viscosity or diffusivity, domain geometry parameters). Our goal is to find the solution $u$ for potentially very large dimensions $d$.

For implementation, we typically discretize the spatio-temporal domain or represent the solution $u$ using a suitable basis or function representation (e.g., values on a high-dimensional grid, coefficients of a basis expansion, or implicitly via a coordinate-based neural network). Let $\mathbf{u} \in \mathbb{R}^N$ denote the high-dimensional vector representing the discretized solution field, where $N$ scales with the grid size and potentially the number of time steps, or represents the weights of a neural representation.

**3.2. Diffusion Model Preliminaries**
We utilize a continuous-time diffusion model framework based on Stochastic Differential Equations (SDEs). A forward SDE describes the process of gradually adding noise to the data distribution $p_0(\mathbf{u})$ (representing the distribution of PDE solutions) over a time interval $[0, T]$:
$$
\mathrm{d}\mathbf{u}_t = \mathbf{f}(\mathbf{u}_t, t)\mathrm{d}t + g(t)\mathrm{d}\mathbf{w}_t
$$
where $\mathbf{w}_t$ is a standard Wiener process, and $\mathbf{f}(\mathbf{u}_t, t)$ and $g(t)$ are drift and diffusion coefficients chosen such that the distribution $p_T(\mathbf{u}_T)$ at time $T$ approaches a simple prior, typically $\mathcal{N}(0, \mathbf{I})$. A common choice is the Variance Preserving (VP) SDE: $\mathbf{f}(\mathbf{u}_t, t) = -\frac{1}{2} \beta(t) \mathbf{u}_t$ and $g(t) = \sqrt{\beta(t)}$, where $\beta(t)$ is a positive noise schedule.

The corresponding reverse-time SDE, which transforms samples from the prior $p_T$ back into samples from the data distribution $p_0$, is given by [Anderson, 1982; Song et al., 2020]:
$$
\mathrm{d}\mathbf{u}_t = [\mathbf{f}(\mathbf{u}_t, t) - g(t)^2 \nabla_{\mathbf{u}_t} \log p_t(\mathbf{u}_t)]\mathrm{d}t + g(t)\mathrm{d}\bar{\mathbf{w}}_t
$$
where $\bar{\mathbf{w}}_t$ is a standard Wiener process running backward in time, and $\nabla_{\mathbf{u}_t} \log p_t(\mathbf{u}_t)$ is the score function of the noisy data distribution $p_t$ at time $t$.

The core task in training a diffusion model is to learn a time-dependent score approximator network $s_\theta(\mathbf{u}_t, t, \mathcal{C})$ (where $\mathcal{C}$ denotes potential conditioning information, like PDE parameters $\mathcal{P}$) to estimate the true score $\nabla_{\mathbf{u}_t} \log p_t(\mathbf{u}_t | \mathcal{C})$. Training typically minimizes a score-matching objective, such as Denoising Score Matching [Vincent, 2011]:
$$
L_{SM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, T)} \mathbb{E}_{\mathbf{u}_0 \sim p_{data}(\mathbf{u}|\mathcal{C})} \mathbb{E}_{\mathbf{u}_t \sim p_{t|0}(\mathbf{u}_t|\mathbf{u}_0)} \left[ \lambda(t) \| s_\theta(\mathbf{u}_t, t, \mathcal{C}) - \nabla_{\mathbf{u}_t} \log p_{t|0}(\mathbf{u}_t|\mathbf{u}_0) \|^2 \right]
$$
where $p_{t|0}$ is the known transition kernel of the forward SDE, $p_{data}(\mathbf{u}|\mathcal{C})$ is the distribution of true solutions for conditions $\mathcal{C}$, and $\lambda(t)$ is a positive weighting function (e.g., $\lambda(t) = g(t)^2$).

**3.3. Proposed PCDS Framework**
Our PCDS framework adapts the diffusion model process for solving PDEs:

*   **State Representation:** The state vector $\mathbf{u}_t$ represents the (potentially noisy) solution field at diffusion time $t$. This can be the solution values on a high-dimensional grid $u(x_i)$ or coefficients in a suitable basis.
*   **Score Network:** We employ a neural network $s_\theta(\mathbf{u}_t, t, \mathcal{P})$ designed to handle high-dimensional inputs, such as a U-Net architecture [Ronneberger et al., 2015] adapted for grid data or a graph neural network if the discretization is unstructured. The network is conditioned on the diffusion time $t$ and the PDE parameters $\mathcal{P}$.
*   **PDE-Constrained Training:** Inspired by PINNs, we incorporate the PDE physics directly into the training objective. A key aspect is relating the noisy state $\mathbf{u}_t$ and the learned score $s_\theta$ back to an estimate of the clean solution $\mathbf{u}_0$. Using Tweedie's formula or derived estimators from SDE/ODE formulations, we can obtain an estimate $\hat{\mathbf{u}}_0(\mathbf{u}_t, s_\theta, t)$ of the noise-free solution given the noisy state $\mathbf{u}_t$. For instance, in the DDPM parameterization, $\hat{\mathbf{u}}_0 \approx (\mathbf{u}_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(\mathbf{u}_t, t)) / \sqrt{\bar{\alpha}_t}$, where $s_\theta$ relates to the noise prediction $\epsilon_\theta$.
    We define the PDE residual functional $R(\mathbf{u}; \mathcal{P})$, which evaluates the L2 norm (or other suitable norm) of the PDE equation and boundary/initial conditions when applied to a candidate solution $\mathbf{u}$:
    $$
    R(\mathbf{u}; \mathcal{P}) = \| \mathcal{N}[\mathbf{u}] - f \|_{\Omega}^2 + \| \mathcal{B}[\mathbf{u}] - g \|_{\partial\Omega}^2 (+ \| \mathcal{I}[\mathbf{u}] - u_0 \|_{\Omega, t=0}^2 \text{ if applicable})
    $$
    The norms are computed by appropriate integration or summation over the discretized domain/boundary.
*   **Hybrid Loss Function:** The network $s_\theta$ is trained by minimizing a combined loss:
    $$
    L(\theta) = L_{SM}(\theta) + \gamma L_{PDE}(\theta)
    $$
    where $L_{SM}$ is the standard score-matching loss (potentially conditioned on $\mathcal{P}$ if training on parameterized PDEs), and $L_{PDE}$ enforces physical consistency:
    $$
    L_{PDE}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, T)} \mathbb{E}_{\mathbf{u}_0 \sim p_{prior}(\mathbf{u}|\mathcal{C})} \mathbb{E}_{\mathbf{u}_t \sim p_{t|0}(\mathbf{u}_t|\mathbf{u}_0)} \left[ w(t) R(\hat{\mathbf{u}}_0(\mathbf{u}_t, s_\theta, t); \mathcal{P}) \right]
    $$
    Here, $\gamma > 0$ is a hyperparameter balancing the score-matching and physics constraints, and $w(t)$ is an optional weighting function emphasizing certain diffusion times (e.g., small $t$ where the solution structure is clearer). The outer expectation in $L_{PDE}$ might sample from a prior distribution $p_{prior}(\mathbf{u}|\mathcal{C})$ (e.g., samples from the reverse process, or even random fields) if true solutions $p_{data}$ are expensive or unavailable, similar to how PINNs operate without paired data. If some ground truth solutions are available, they can be used in $L_{SM}$.
*   **Inference/Solving:** To obtain a solution $\hat{\mathbf{u}}_0$ for specific parameters $\mathcal{P}$, we start with a sample $\mathbf{u}_T \sim \mathcal{N}(0, \mathbf{I})$ and solve the reverse SDE or its corresponding probability flow ODE numerically using the learned score $s_\theta(\cdot, t, \mathcal{P})$:
    $$
    \mathrm{d}\mathbf{u}_t = [\mathbf{f}(\mathbf{u}_t, t) - g(t)^2 s_\theta(\mathbf{u}_t, t, \mathcal{P})]\mathrm{d}t \quad (\text{ODE version})
    $$
    from $t=T$ down to $t=0$. The resulting $\mathbf{u}_0$ is the estimated PDE solution.

**3.4. Data Collection**
While PCDS aims to reduce reliance on ground truth solution data by incorporating the PDE residual loss, some data might be beneficial for stabilizing $L_{SM}$ training or for validation.
*   For benchmark problems where analytical solutions exist (e.g., high-D heat equation with specific BCs), these will be used for computing $L_{SM}$ and for evaluation.
*   For problems lacking analytical solutions (e.g., Black-Scholes, Navier-Stokes), we will generate reference solutions using highly accurate traditional solvers (spectral methods, high-resolution FEM/FDM) but only in lower-dimensional settings ($d \approx 2-4$) for initial validation and comparison.
*   For high-dimensional validation where traditional solvers fail, we will rely primarily on evaluating the PDE residual $R(\hat{\mathbf{u}}_0; \mathcal{P})$ of the generated solutions. Consistency checks (e.g., conservation laws, statistical properties) will also be used where applicable.
*   For parameterized PDEs, we will sample parameters $\mathcal{P}$ from specified distributions (e.g., uniform or Gaussian ranges for coefficients) to train a conditional PCDS model $s_\theta(\mathbf{u}_t, t, \mathcal{P})$.

**3.5. Experimental Design**
*   **PDE Test Cases:**
    1.  *High-Dimensional Heat Equation:* $\partial_t u = \Delta u + f(x, t)$ in $\Omega = [0, 1]^d$. Vary $d$ from low values (e.g., $d=2$) up to $d=100, 200+$. Test with known analytical solutions and complex source terms $f$.
    2.  *High-Dimensional Black-Scholes Equation:* $\partial_t V + \frac{1}{2} \sum_{i,j=1}^d \sigma_i \sigma_j \rho_{ij} S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} + r \sum_{i=1}^d S_i \frac{\partial V}{\partial S_i} - rV = 0$. Here, $d$ is the number of underlying assets. Focus on $d=10, 50, 100+$. Payoff functions define boundary/terminal conditions.
    3.  *High-Dimensional Poisson Equation:* $-\Delta u = f(x)$ in $\Omega = [0, 1]^d$. A fundamental elliptic PDE. Test scalability with $d$.
    4.  *(Exploratory)* Statistical Steady States of High-Dimensional Systems: Potentially explore applications related to high-dimensional dynamical systems where statistical properties satisfy related PDEs (e.g., Fokker-Planck equation, or finding steady states of Navier-Stokes in high-dimensional parameter spaces or via proper orthogonal decomposition).
*   **Baselines for Comparison:**
    *   PINNs (original formulation and variants).
    *   Neural Operators (FNO, DeepONet).
    *   Traditional methods (FDM, FEM, Spectral) where feasible (low $d$).
    *   Related work: If possible, implement or compare conceptually/empirically to Johnson & Lee (2024) focusing on differences in loss formulation, applicability range, and high-D scaling.
*   **Evaluation Metrics:**
    *   *Accuracy:* Relative $L_2$ error compared to analytical or high-fidelity numerical solutions (where available).
    *   *Physics Fidelity:* PDE residual norm $R(\hat{\mathbf{u}}_0; \mathcal{P})$ over the domain and boundary.
    *   *Computational Cost:* Training time (GPU hours), inference time per solution.
    *   *Scalability:* How accuracy and computational cost vary with dimension $d$. Plot error vs. $d$ and time vs. $d$.
    *   *Generalization (for parameterized PDEs):* Error on PDE instances with parameters $\mathcal{P}$ not seen during training.
    *   *Robustness:* Sensitivity to hyperparameters (e.g., $\gamma$, noise schedule $\beta(t)$, network architecture).

**4. Expected Outcomes & Impact**

**4.1. Expected Outcomes**
1.  **A Novel PCDS Framework:** A fully developed and implemented PDE-Constrained Diffusion Solver framework, including associated code base and documentation.
2.  **High-Dimensional PDE Solutions:** Demonstration of the PCDS framework's ability to generate accurate solutions for benchmark linear and nonlinear PDEs in dimensions significantly higher ($d \geq 100$) than typically accessible by traditional methods or some existing neural solvers.
3.  **State-of-the-Art Performance:** Quantitative results showing that PCDS achieves competitive or superior performance in terms of accuracy, computational efficiency, and scalability compared to relevant baselines (PINNs, FNO, traditional methods) specifically in the high-dimensional regime.
4.  **Parameterized PDE Handling:** Validation of the framework's ability to handle parameterized PDEs, potentially learning solution maps over parameter spaces and generalizing to unseen parameters.
5.  **Algorithmic Insights:** Deeper understanding of the interplay between score-matching objectives and PDE constraints in the context of diffusion models, including the effect of the weighting parameter $\gamma$ and the choice of noise schedule. This includes insights into how the PDE structure influences the learned denoising paths.
6.  **Publications and Software:** High-quality publications in leading machine learning or scientific computing venues (conferences like NeurIPS, ICML, journals like JCP, SIAM journals). Release of open-source code to facilitate reproducibility and further research.

**4.2. Impact**
*   **Scientific and Engineering Advancement:** The PCDS framework has the potential to unlock new possibilities in scientific simulation and engineering design by enabling the solution of previously intractable high-dimensional PDEs. This could accelerate research in fields like:
    *   *Financial Modeling:* Pricing complex derivatives dependent on numerous assets.
    *   *Quantum Mechanics:* Solving the Schrödinger equation for many-body systems.
    *   *Statistical Physics:* Analyzing high-dimensional Fokker-Planck or Boltzmann equations.
    *   *Materials Science:* Modeling phenomena in high-dimensional microstructural or parameter spaces.
    *   *Fluid Dynamics:* Parameter studies or uncertainty quantification involving high-dimensional parameter spaces.
*   **Methodological Contribution:** This research contributes significantly to the rapidly evolving intersection of deep learning and differential equations. It pioneers a specific way of integrating physics constraints into powerful generative diffusion models for solving challenging computational science problems. This could inspire new research directions in physics-informed generative modeling and structured deep learning architectures.
*   **Broadening Applicability of Diffusion Models:** Demonstrates a novel and impactful application domain for diffusion models beyond traditional generative tasks like image synthesis, highlighting their potential as powerful tools for scientific computing.
*   **Potential for Extensions:** The core framework could be extended to tackle related challenging problems, such as solving stochastic PDEs (by incorporating stochasticity naturally within the diffusion framework), solving inverse PDE problems (by conditioning the diffusion process on observational data), and providing inherent uncertainty quantification (leveraging the probabilistic nature of diffusion models, as explored by Purple & Yellow, 2025).

In conclusion, this research proposes a promising new direction for solving high-dimensional PDEs by uniquely combining the generative power of diffusion models with the physical rigor of PDE constraints. By addressing key challenges in scalability and accuracy, the PCDS framework is poised to make significant contributions to both machine learning methodology and computational science and engineering.

**5. References**

1.  Anderson, B. D. O. (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications*, 12(3), 313–326.
2.  Haitsiukevich, K., Poyraz, O., Marttinen, P., & Ilin, A. (2024). Diffusion models as probabilistic neural operators for recovering unobserved states of dynamical systems. *arXiv preprint arXiv:2405.07097*.
3.  Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.
4.  Johnson, A., & Lee, B. (2024). Physics-Informed Diffusion Models for High-Dimensional PDEs. *arXiv preprint arXiv:2403.09876*.
5.  Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. *International Conference on Learning Representations (ICLR)*.
6.  Purple, L., & Yellow, K. (2025). Uncertainty Quantification in Neural PDE Solvers via Diffusion Models. *arXiv preprint arXiv:2502.05678*. [Note: Fictional citation from prompt assumed for context]
7.  Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
8.  Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.
9.  Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. *International Conference on Machine Learning (ICML)*.
10. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. *International Conference on Learning Representations (ICLR)*.
11. Taufik, M. H., & Alkhalifah, T. (2023). LatentPINNs: Generative physics-informed neural networks via a latent representation learning. *arXiv preprint arXiv:2305.07671*.
12. Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation*, 23(7), 1661–1674.
*(Additional relevant citations from the provided list could be added as needed, e.g., survey papers or specific neural operator/PINN works for context).*