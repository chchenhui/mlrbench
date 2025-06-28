# **Title**  
**Optimal Transport-Driven Neural ODEs for Robust Control Policies**  

# **Introduction**  
**Background**  
Modern control systems face significant challenges in real-world environments, including distribution shifts, model uncertainties, and nonlinear dynamics. Traditional control methods often rely on deterministic or linear approximations, which fail to account for variations in state distributions caused by external perturbations or evolving environments. Neural Ordinary Differential Equations (Neural ODEs) have emerged as a powerful tool for modeling continuous dynamical systems by learning the underlying vector field $ \dot{\mathbf{z}} = f(\mathbf{z}, t; \theta) $, where $ \mathbf{z} \in \mathbb{R}^n $ represents the system’s state, $ t $ is time, and $ \theta $ parameterizes the neural network. However, Neural ODEs alone struggle to enforce robustness against state distribution mismatches or adversarial perturbations, limiting their applicability in safety-critical systems such as robotics and autonomous vehicles.

Recent advances in Optimal Transport (OT) provide a geometric framework to quantify discrepancies between probability distributions via metrics like the Wasserstein distance. This is particularly relevant for control tasks where desired outcomes are defined as target distributions (e.g., reaching a goal state distribution). Existing methods, such as OT-Flow [8], integrate OT with Neural ODEs for generative modeling, but these approaches focus on static distributions and lack explicit control cost optimization. Similarly, distributionally robust control techniques [4,5] use OT-based ambiguity sets to handle uncertainties but often assume known dynamics or restricted policy classes.

**Research Objectives**  
This proposal aims to address these limitations by developing a unified framework that:  
1. Combines Neural ODEs with OT-driven loss functions to learn continuous-time control policies that steer state distributions toward target distributions.  
2. Incorporates Stochastic Optimal Control (SOC) principles via adversarial training to ensure robustness against model uncertainties and unmodeled perturbations.  
3. Establishes theoretical guarantees on convergence, stability, and robustness of the proposed framework.  

**Significance**  
The proposed work has three key impacts:  
1. **Theoretical Unification**: Bridging OT, Neural ODEs, and SOC to enable principled, data-driven control policies with geometric interpretability.  
2. **Practical Advantages**: Enhancing sample efficiency, stability, and robustness in complex tasks such as robotic manipulation under variable friction or supply-chain optimization with stochastic demands.  
3. **Scalability**: Developing efficient approximations for high-dimensional OT computations, enabling real-time deployment in large-scale systems.  

---

# **Methodology**  

## **Problem Formulation**  
Consider a nonlinear dynamical system governed by:  
$$
\dot{\mathbf{z}} = f(\mathbf{z}, t; \theta) + u(t),
$$
where $ u(t) \in \mathbb{R}^m $ is the control input. The goal is to learn a policy $ \pi(\mathbf{z}, t) = u(t) $ that minimizes the Wasserstein-2 distance between the predicted state distribution $ p_t(\mathbf{z}) $ and a target distribution $ p_{\text{target}}(\mathbf{z}) $, while minimizing the control effort. The optimal control problem is:  
$$
\min_{\theta, u} \left[ W_2(p_T, p_{\text{target}})^2 + \lambda \int_0^T \|u(t)\|^2 dt \right],
$$
where $ \lambda > 0 $ balances tracking accuracy and control cost.  

## **Algorithm Design**  

### **1. Neural ODE Dynamics Parametrization**  
The state evolution is modeled as a Neural ODE:  
$$
\dot{\mathbf{z}} = f(\mathbf{z}, t; \theta),
$$
where $ f $ is a neural network with parameters $ \theta $. To enforce robustness, the training process adversarially augments the dynamics with perturbations $ \eta(t) \sim \mathcal{N}(0, \sigma^2) $, leading to:  
$$
\dot{\mathbf{z}} = f(\mathbf{z}, t; \theta) + u(t) + \eta(t).
$$
This aligns with SOC principles, treating perturbations as worst-case disturbances [9].  

### **2. Optimal Transport Loss Function**  
The Wasserstein-2 distance between $ p_t(\mathbf{z}) $ and $ p_{\text{target}}(\mathbf{z}) $ is computed using the Sinkhorn algorithm [7], which approximates:  
$$
W_2(p_t, p_{\text{target}})^2 = \min_{\gamma \in \Gamma(p_t, p_{\text{target}})} \mathbb{E}_{\gamma}[\|\mathbf{z} - \mathbf{z}'\|^2],
$$
where $ \Gamma(p_t, p_{\text{target}}) $ is the set of joint distributions with marginals $ p_t $ and $ p_{\text{target}} $. The loss function becomes:  
$$
\mathcal{L}(\theta, u) = \alpha W_2(p_T, p_{\text{target}})^2 + \beta \int_0^T \|u(t)\|^2 dt + \gamma \text{Lyap}(\mathbf{z}, \theta),
$$
where $ \text{Lyap} $ enforces stability via Control Lyapunov Functions (CLFs) [3]. Coefficients $ \alpha, \beta, \gamma $ trade off tracking accuracy, control effort, and stability.  

### **3. Optimization**  
The problem is solved via gradient descent using the Pontryagin’s Maximum Principle (PMP) [1] and adjoint sensitivity methods [10]. The augmented Lagrangian is:  
$$
\mathcal{L}_{\text{aug}} = \mathcal{L}(\theta, u) + \int_0^T \lambda(t)^\top (\dot{\mathbf{z}} - f(\mathbf{z}, t; \theta) - u(t)) dt,
$$
where $ \lambda(t) $ is the adjoint state. The gradients are computed as:  
$$
\frac{d\lambda}{dt} = -\frac{\partial \mathcal{L}_{\text{aug}}}{\partial \mathbf{z}}, \quad \frac{\partial \mathcal{L}_{\text{aug}}}{\partial \theta} = \int_0^T \lambda(t)^\top \frac{\partial f}{\partial \theta} dt.
$$  

### **4. Adversarial Training**  
To enhance robustness, the model is trained on trajectories with stochastic perturbations $ \eta(t) \sim \mathcal{N}(0, \sigma_t^2) $, where $ \sigma_t^2 $ is optimized to maximize $ \mathcal{L}(\theta, u) $. This adversarial step ensures the policy remains effective under unseen disturbances [4].  

### **5. Scalable Implementation**  
For high-dimensional systems, we approximate $ W_2 $ using entropic OT [7] with Sinkhorn iterations, reducing computation from $ \mathcal{O}(n^3) $ to $ \mathcal{O}(n^2 \log n) $. Batched Trajectory Sampling: States are sampled in batches, and $ p_t(\mathbf{z}) $ is approximated via kernel density estimation.  

## **Experimental Design**  

### **1. Data Collection**  
- **Synthetic Data**: Generate state trajectories using known SDEs with varying drift and diffusion coefficients.  
- **Real-World Data**: Use PyBullet or MuJoCo to simulate robotic manipulation tasks with variable friction and mass parameters.  

### **2. Baselines**  
- **Model-Predictive Control (MPC)**: Standard baseline using quadratic cost.  
- **Variational Inference (VI)**: Learn dynamics via variational inference.  
- **OT-Flow [8]**: Use OT-regularized Neural ODEs without control cost.  

### **3. Evaluation Metrics**  
- **Tracking Error**: $ \|p_T - p_{\text{target}}\|_2 $ (computed via KL divergence and W2).  
- **Control Cost**: $ \int_0^T \|u(t)\|^2 dt $.  
- **Sample Efficiency**: Training steps to converge.  
- **Robustness**: Performance degradation under test-time perturbations.  

### **4. Computational Tools**  
- Implement in PyTorch using `torchdiffeq` for Neural ODE solvers.  
- Use JAX for parallelizing OT computations [6].  

---

# **Expected Outcomes & Impact**  

## **Expected Outcomes**  
1. **Theoretical Contributions**:  
   - A unified framework for robust control via OT-Neural ODE integration.  
   - Convergence guarantees using PMP and Lyapunov stability theory.  
   - Characterization of how adversarial perturbations affect OT-based policy learning.  

2. **Empirical Improvements**:  
   - **20% reduction** in tracking error compared to OT-Flow and MPC baselines on robotic tasks.  
   - **30% improvement** in sample efficiency for reaching target distributions across stochastic environments.  
   - Strong robustness to unmodeled perturbations (e.g., sudden friction changes in robotics).  

3. **Algorithmic Innovations**:  
   - Scalable OT approximations for high-dimensional systems.  
   - Adversarial training protocol for SOC in continuous-time systems.  

## **Impact**  
1. **Control Theory**: Advances theoretical understanding of OT’s role in shaping robust controllers, bridging geometric and stochastic control.  
2. **Machine Learning**: Introduces a new paradigm for policy optimization with formal safety guarantees, extending Neural ODEs’ applicability.  
3. **Applications**: Enables deployment in autonomous vehicles (uncertain road conditions), logistics (stochastic demand forecasting), and energy systems (variable renewable sources).  

This work aligns with the workshop’s emphasis on interdisciplinary progress in learning, control, and dynamical systems, offering a scalable, theoretically grounded methodology for real-world challenges.