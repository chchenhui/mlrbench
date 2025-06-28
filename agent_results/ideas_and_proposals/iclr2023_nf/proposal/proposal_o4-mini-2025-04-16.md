1. Title  
Adaptive-Activation Meta-Learned Neural Fields for Efficient and Accurate PDE Simulation  

2. Introduction  
Background  
Solving partial differential equations (PDEs) underpins computational physics, engineering design, climate modeling, and many other scientific domains. Traditional mesh-based solvers—finite element, finite difference, finite volume methods—rely on discretization, can be computationally expensive for high-dimensional or dynamically changing domains, and struggle to generalize across parameterized or irregular geometries. Physics-informed neural networks (PINNs) [Raissi et al., 2019] and coordinate-based neural fields [Mildenhall et al., 2021] have emerged as mesh-free continuous alternatives that embed governing equations as constraints during training. However, PINNs often face optimization difficulties (sensitive to activation functions, poorly capture multi-scale phenomena) and require retraining for each new boundary/initial condition or geometry.  

Recent advances in meta-learning for PINNs [Iwata et al., 2023; Najian Asl et al., 2025], neural optimizers for physics simulation [Wandel et al., 2024], and adaptive activation functions [Wang et al., 2023] point toward a unified framework that can rapidly adapt to new PDE instances while resolving fine-scale features efficiently. In parallel, geometric feature extractors such as Physics-informed PointNet (PIPN) [Kashefi & Mukerji, 2022, 2023] have shown generalization across multiple irregular domains without retraining.  

Research Objectives  
Building on these insights, we propose a **Meta-Learned Adaptive-Activation Neural Field** (MAANF) framework to solve families of PDEs with:  
- **Spatially adaptive activations** via feature-wise modulation (FiLM) that resolve multi-scale features in a coordinate-based MLP.  
- **Meta-learning** (e.g., Model-Agnostic Meta-Learning, MAML) to produce an initialization enabling rapid adaptation (few gradient steps) to new boundary conditions, source terms, and geometries.  
- **Physics-informed loss** enforcing governing equations and boundary/initial conditions in a mesh-free manner.  

Significance  
By combining adaptive activations with meta-learned initialization, MAANF aims to:  
- Achieve accuracy comparable to or better than state-of-the-art PINNs and mesh-based solvers across a range of PDEs (Navier–Stokes, wave, heat, Burgers).  
- Reduce per-instance training time by an order of magnitude relative to conventional PINNs.  
- Generalize to unseen parameter regimes and irregular domains without full retraining.  
- Provide a scalable, continuous representation of solution fields, enabling downstream tasks such as control and design optimization.  

3. Methodology  
3.1. Problem Setting and Notation  
We consider parametric PDE families of the form  
$$\mathcal{N}_{\lambda}[u](\mathbf{x},t)=0,\quad (\mathbf{x},t)\in \Omega\times[0,T],$$  
with boundary/initial conditions  
$$u(\mathbf{x},0)=u_0(\mathbf{x}),\quad u|_{\partial\Omega}=g_b(\mathbf{x},t).$$  
Here $\lambda$ denotes a task-specific parameter (e.g., Reynolds number, wave speed, geometry encoding). Our goal is to learn a coordinate-based neural field $u_\theta(\mathbf{x},t;\lambda)$ that approximates the solution for any task $\lambda$ after a small number $K$ of fine-tuning steps.  

3.2. Network Architecture  

1. Coordinate Encoder  
   We apply a Fourier feature map $\gamma(\cdot)$ [Tancik et al., 2020] to input coordinates:  
   $$\gamma(\mathbf{x},t)=\left[\sin(2\pi\mathbf{B}[\mathbf{x};t]),\ \cos(2\pi\mathbf{B}[\mathbf{x};t])\right],$$  
   where $\mathbf{B}\sim\mathcal{N}(0,\sigma^2)$ is a random projection matrix.  

2. Base MLP with FiLM Modulation  
   We define an $L$-layer MLP with hidden dimension $H$. Denote $h^{(0)}=\gamma(\mathbf{x},t)$ and for $l=1,\dots,L$:  
   $$z^{(l)} = W^{(l)}h^{(l-1)} + b^{(l)},$$  
   $$h^{(l)} = \sigma\bigl(\alpha^{(l)}(\mathbf{x},t)\odot z^{(l)}\bigr),$$  
   where $\sigma$ is a nonlinearity (e.g., sine or tanh), and $\alpha^{(l)}(\mathbf{x},t)\in\mathbb{R}^H$ is a spatially adaptive scaling vector output by a small **modulation network** $M_\phi^{(l)}$:  
   $$\alpha^{(l)}(\mathbf{x},t) = M_\phi^{(l)}\bigl(\gamma(\mathbf{x},t)\bigr).$$  
   The final output layer maps $h^{(L)}\mapsto u_\theta(\mathbf{x},t)$ via a linear layer.  

3.3. Physics-Informed Loss  
We enforce the PDE residual and boundary/initial conditions over collocation points. For a batch of $N_r$ interior points $(\mathbf{x}_i,t_i)$ and $N_b$ boundary/initial points $(\mathbf{x}_j,t_j)$:  
$$\mathcal{L}_{\rm PDE} = \frac{1}{N_r}\sum_{i=1}^{N_r}\bigl\|\mathcal{N}_{\lambda}[u_\theta](\mathbf{x}_i,t_i)\bigr\|^2,$$  
$$\mathcal{L}_{\rm BC}=\frac{1}{N_b}\sum_{j=1}^{N_b}\bigl|u_\theta(\mathbf{x}_j,t_j)-u_{\rm true}(\mathbf{x}_j,t_j)\bigr|^2.$$  
The total loss is  
$$\mathcal{L}(\theta;\lambda)=\mathcal{L}_{\rm PDE}+\beta\,\mathcal{L}_{\rm BC},$$  
with hyperparameter $\beta>0$. All derivatives in $\mathcal{N}$ are computed via automatic differentiation.  

3.4. Meta-Learning Optimization  
We adopt a MAML-style procedure [Finn et al., 2017]:  
Algorithm MetaTrain  
1. Initialize shared parameters $\theta=\{\{W^{(l)},b^{(l)}\},\phi\}$ randomly.  
2. Repeat until convergence:  
   a. Sample a batch of tasks $\{\lambda_k\}_{k=1}^B$.  
   b. For each $\lambda_k$:  
      i. Sample support collocation sets $\mathcal{S}_k$ and query sets $\mathcal{Q}_k$.  
      ii. Compute adapted parameters via $K$ inner‐loop gradient steps:  
         $$\theta'_k = \theta - \alpha\nabla_\theta \mathcal{L}(\theta;\lambda_k,\mathcal{S}_k).$$  
      iii. Compute query loss $\mathcal{L}(\theta'_k;\lambda_k,\mathcal{Q}_k)$.  
   c. Update meta-parameters with outer gradient:  
      $$\theta \leftarrow \theta - \eta\nabla_\theta \sum_{k=1}^B \mathcal{L}(\theta'_k;\lambda_k,\mathcal{Q}_k).$$  
Here $\alpha$ and $\eta$ are inner and outer learning rates. We use a first-order approximation (FOMAML) to reduce memory.  

3.5. Experimental Design  
PDE Families & Domains  
- **Fluid dynamics**: 2D incompressible Navier–Stokes on rectangular and obstacle-laden domains; Reynolds number $\mathrm{Re}\in[100,1000]$.  
- **Wave propagation**: 2D/3D wave equation in heterogenous media with varying wave speed fields.  
- **Diffusion**: Heat/diffusion and Burgers’ equations in 1D–3D.  
- **Irregular geometries**: Templated via signed distance functions or random domain deformations.  

Data Generation  
For each task $\lambda$:  
- Randomly sample boundary/initial conditions from a parameterized family (e.g., Gaussian blobs, random field realizations).  
- Generate reference solutions $u_{\rm true}$ via high‐resolution FEM or spectral methods on a fine grid.  
- Collocation points: $N_r=2\!:\!10\times10^3$ interior, $N_b=1\!:\!2\times10^3$ boundary/initial.  

Baselines  
- Standard PINN [Raissi et al., 2019] (no meta‐learning, fixed activations).  
- Meta-PINN [Iwata et al., 2023] (meta-learning without adaptive activations).  
- Implicit FOL [Najian Asl et al., 2025].  
- Metamizer [Wandel et al., 2024].  
- PIPN [Kashefi & Mukerji, 2022].  

Ablation Studies  
- Without adaptive activations (constant $\alpha^{(l)}=1$).  
- Without meta-learning (train from scratch per task).  
- Varying number of inner‐loop steps $K$.  
- Compare different modulation architectures (FiLM vs. gating).  

3.6. Evaluation Metrics  
Accuracy  
- Relative $L^2$ error:  
  $$\epsilon_{L^2} = \frac{\|u_\theta - u_{\rm true}\|_{2}}{\|u_{\rm true}\|_{2}}.$$  
- Maximum absolute error $\|u_\theta - u_{\rm true}\|_\infty$.  
- Physical quantities of interest (drag/lift coefficients in fluid flow).  

Efficiency  
- Wall-clock time for adaptation to a new task (inner-loop).  
- Number of gradient steps to reach a target error.  
- Memory footprint.  

Generalization  
- Performance on unseen $\lambda$ outside training distribution.  
- Robustness to domain perturbations.  

4. Expected Outcomes & Impact  
We anticipate that MAANF will achieve the following:  
1. **Rapid Adaptation**  
   Through meta-learning, MAANF will require only a few (e.g., $K\le5$) inner-loop steps to adapt to new boundary conditions, outperforming standard PINNs (which often need hundreds of epochs).  
2. **Multi-Scale Resolution**  
   The spatially adaptive activation strategy will enable accurate capture of sharp gradients and small‐scale features (boundary layers, shock fronts) without manual tuning of activation functions.  
3. **Robust Generalization**  
   MAANF will generalize across parameter regimes (e.g., wide Reynolds numbers) and to irregular domains, reducing the need for retraining on each new geometry.  
4. **Computational Efficiency**  
   We expect up to an order-of-magnitude reduction in per-task training time relative to baseline PINNs and comparable accuracy to FEM at a fraction of the cost in moderate dimensions ($d\le3$).  
5. **Broader Impact**  
   - Provide a mesh‐free, continuous solver paradigm accessible to practitioners in physics, engineering, robotics, and climate science.  
   - Facilitate real-time simulation and control tasks in robotics or digital twins.  
   - Serve as a building block for data‐driven discovery of PDEs and hybrid physics/data models.  

5. References  
• Maziar Raissi, Paris Perdikaris, George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *JCP* (2019).  
• Tomoharu Iwata, Yusuke Tanaka, Naonori Ueda. Meta-learning of Physics-informed Neural Networks for Efficiently Solving Newly Given PDEs. *arXiv:2310.13270* (2023).  
• Reza Najian Asl et al. A Physics-Informed Meta-Learning Framework for the Continuous Solution of Parametric PDEs on Arbitrary Geometries. *arXiv:2504.02459* (2025).  
• Nils Wandel et al. Metamizer: a versatile neural optimizer for fast and accurate physics simulations. *arXiv:2410.19746* (2024).  
• Honghui Wang et al. Learning Specialized Activation Functions for Physics-informed Neural Networks. *arXiv:2308.04073* (2023).  
• Ali Kashefi, Tapan Mukerji. Physics-informed PointNet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries. *CMAME* (2022, 2023).  
• Charles Qi et al. PointNet: Deep learning on point sets for 3D classification and segmentation. *CVPR* (2017).  
• Tancik et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS* (2020).  
• Finn, Chelsea, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML* (2017).