Okay, here is a research proposal generated based on the provided task description, research idea, and literature review.

---

**1. Title:** **Optimal Transport-Driven Neural ODEs for Learning Robust Control Policies under Distributional Shifts**

**2. Introduction**

**2.1 Background and Motivation**
The intersection of machine learning, control theory, and dynamical systems is a rapidly evolving frontier, offering powerful new paradigms for understanding and manipulating complex systems (Workshop Task Description). Recent advancements, particularly in deep learning architectures like Neural Ordinary Differential Equations (Neural ODEs) [Chen et al., 2018], provide flexible, continuous-time models for system dynamics, learned directly from data. Neural ODEs naturally represent the evolution of states in dynamical systems, making them promising candidates for data-driven control [Sandoval et al., 2022]. However, a critical challenge in deploying control systems in real-world scenarios, such as robotics, autonomous driving, or supply chain management, is their inherent susceptibility to uncertainties and distributional shifts. These shifts can arise from variations in initial conditions, changing environmental parameters (e.g., friction, load), sensor noise, or model inaccuracies, often leading to performance degradation or instability in controllers trained under nominal conditions.

Traditional robust control methods often rely on predefined uncertainty models (e.g., bounded sets), which may not accurately capture complex, data-driven distributional uncertainties. Reinforcement Learning (RL), while powerful for learning control policies, can struggle with sample efficiency and robustness guarantees, especially when faced with non-stationary environments or distribution shifts not encountered during training.

Optimal Transport (OT) theory provides a principled mathematical framework for comparing probability distributions by measuring the minimal "cost" required to transport mass from one distribution to another [Villani, 2003]. The Wasserstein distance, a key metric derived from OT, offers geometric insights into the space of probability distributions and is increasingly used in machine learning for its ability to handle non-overlapping distributions and capture underlying structure [Peyré & Cuturi, 2019]. Integrating OT principles into the control loop offers a compelling avenue for explicitly managing distributional aspects of system states. By defining control objectives in terms of steering state *distributions* rather than just single trajectories, and quantifying deviations using OT metrics, we can potentially learn policies inherently robust to variations.

This research is motivated by the need for adaptable, theoretically grounded control policies that can maintain performance despite uncertainties and distributional shifts. We hypothesize that bridging the continuous-time dynamics modeling capabilities of Neural ODEs with the distributional comparison power of Optimal Transport can lead to a novel class of robust controllers. The specific idea is to formulate the control problem as steering an initial state distribution to a desired target distribution over time, governed by Neural ODE dynamics, while minimizing an OT-based objective alongside traditional control costs. Furthermore, leveraging principles from Stochastic Optimal Control (SOC) and Distributionally Robust Optimization (DRO) [Blanchet et al., 2023] can enhance robustness against model uncertainties and perturbations. This aligns with the workshop's focus on unifying learning, control, and dynamical systems, particularly exploring OT [Scagliotti & Farinelli, 2023; Pooladian et al., 2024], Neural ODEs [Sandoval et al., 2022], and SOC [Mikami, 2024].

**2.2 Problem Statement**
The core research problem is to develop a control synthesis framework based on Neural ODEs that explicitly optimizes for robustness against distributional shifts in system states and parameters. Specifically, we aim to answer: *How can Optimal Transport metrics be integrated into the training objective of Neural ODE-based controllers to learn policies that effectively steer ensembles of trajectories (representing distributions) towards desired target distributions, while ensuring robustness against unforeseen perturbations and model uncertainties, guided by principles of Stochastic Optimal Control?*

Existing approaches often treat robustness implicitly or rely on assumptions that may not hold for complex distributional uncertainties encountered in practice. Neural ODEs provide the dynamic model, but standard training objectives (e.g., minimizing trajectory error) may not inherently promote distributional robustness. OT offers the tool to measure distances between distributions, but its integration into dynamic control optimization, especially for robustness, requires careful formulation and efficient computation. Addressing the computational complexity [Lit. Review Challenge 1], ensuring stability [Lit. Review Challenge 2], achieving scalability [Lit. Review Challenge 3], providing theoretical underpinnings [Lit. Review Challenge 4], and effectively handling stochasticity [Lit. Review Challenge 5] are key facets of this problem.

**2.3 Research Objectives**
The primary objectives of this research are:

1.  **Develop the OT-NODEC Framework:** Propose and formalize a novel control framework, "Optimal Transport-Driven Neural ODE Control" (OT-NODEC), that integrates Neural ODEs for dynamics modeling, OT metrics for distributional objectives, and a neural network-based control policy.
2.  **Integrate OT for Distributional Control:** Formulate a learning objective that combines standard control costs (e.g., state tracking, effort minimization) with an OT-based cost (e.g., Wasserstein distance) measuring the discrepancy between the propagated state distribution under the learned policy and a desired target distribution at specific time horizons.
3.  **Incorporate Robustness via SOC Principles:** Enhance the framework's robustness by integrating techniques inspired by SOC and DRO. Specifically, employ adversarial training where perturbations, possibly defined within an OT-based uncertainty set around nominal conditions [Aolaritei et al., 2023; Blanchet et al., 2023], are applied to initial states or system parameters during learning to optimize worst-case performance.
4.  **Empirical Validation:** Implement the OT-NODEC framework and validate its performance, robustness, and sample efficiency on challenging control benchmarks, including robotic manipulation tasks with varying physical parameters (e.g., friction, mass) and supply chain optimization problems with stochastic demand distributions. Compare against relevant baselines.
5.  **Theoretical Analysis:** Investigate the theoretical properties of the proposed framework, focusing on conditions for stability, convergence of the learning process, and formal guarantees on robustness concerning the defined distributional uncertainty sets.

**2.4 Significance and Contributions**
This research promises several significant contributions:

1.  **Novel Control Paradigm:** Introduces a new approach to robust control synthesis that explicitly leverages the geometric properties of OT within a continuous-time deep learning framework (Neural ODEs), offering a principled way to handle distributional uncertainty.
2.  **Enhanced Robustness:** Aims to deliver control policies with superior robustness to distribution shifts compared to standard data-driven and model-based control methods, making them more reliable for real-world deployment.
3.  **Bridging Disciplines:** Strengthens the connection between Optimal Transport, deep learning (Neural ODEs), control theory, and dynamical systems, aligning with the workshop's goals and potentially opening new interdisciplinary research avenues [Di Persio & Garbelli, 2023].
4.  **Addressing Key Challenges:** Directly tackles several key challenges identified in the literature, including robustness under uncertainty, stability, and the integration of stochastic elements in learned controllers.
5.  **Theoretical Insights:** Provides theoretical analysis connecting OT geometry, Neural ODE dynamics, and control robustness properties, contributing to a deeper understanding of data-driven control.
6.  **Practical Applicability:** Demonstrates potential applicability in domains like robotics, autonomous systems, and operations research where dealing with uncertainty and variability is paramount.

**3. Methodology**

**3.1 Mathematical Formulation**

*   **System Dynamics:** We consider dynamical systems whose state evolution $\mathbf{z}(t) \in \mathbb{R}^d$ is governed by a Neural ODE, parameterized by $\theta$:
    $$
    \frac{d\mathbf{z}(t)}{dt} = f_{\theta}(\mathbf{z}(t), \mathbf{u}(t), t)
    $$
    where $\mathbf{u}(t) \in \mathbb{R}^m$ is the control input at time $t$. The function $f_{\theta}$ is represented by a neural network.
*   **Control Policy:** The control input is generated by a feedback policy $\pi_{\phi}(\mathbf{z}(t), t)$, also parameterized by a neural network with parameters $\phi$:
    $$
    \mathbf{u}(t) = \pi_{\phi}(\mathbf{z}(t), t)
    $$
*   **State Distribution Evolution:** Given an initial distribution of states $p_0(\mathbf{z})$, the system dynamics and control policy induce a time-evolving distribution $p_t(\mathbf{z})$. The Neural ODE allows us to propagate samples or estimate the density evolution [Grathwohl et al., 2018]. Let $T$ be the control horizon. We are interested in the distribution $p_T(\mathbf{z})$ at the final time.
*   **Optimal Transport Cost:** We use the $p$-Wasserstein distance $W_p(\mu, \nu)$ to quantify the difference between the predicted final state distribution $p_T(\mathbf{z})$ and a desired target distribution $p_{target}(\mathbf{z})$. For $p \ge 1$,
    $$
    W_p^p(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|\mathbf{x} - \mathbf{y}\|^p d\gamma(\mathbf{x}, \mathbf{y})
    $$
    where $\Gamma(\mu, \nu)$ is the set of all joint distributions (transport plans) with marginals $\mu$ and $\nu$. We will primarily focus on $p=2$ (Wasserstein-2 distance), relevant for connections to SOC and linear control [Scagliotti & Farinelli, 2023].
*   **Control Objective:** The goal is to find parameters $(\theta, \phi)$ that minimize a combined objective function over an initial distribution $p_0(\mathbf{z})$:
    $$
    J(\theta, \phi) = \mathbb{E}_{\mathbf{z}_0 \sim p_0} \left[ \int_0^T c(\mathbf{z}(t), \mathbf{u}(t)) dt + g(\mathbf{z}(T)) \right] + \lambda W_p^p(p_T(\mathbf{z}), p_{target}(\mathbf{z}))
    $$
    where $c(\cdot, \cdot)$ is the instantaneous control cost (e.g., quadratic cost on state deviation and control effort $c(\mathbf{z}, \mathbf{u}) = \|\mathbf{z} - \mathbf{z}_{ref}\|^2 + \|\mathbf{u}\|^2$) and $g(\cdot)$ is a terminal cost on individual trajectories. $\lambda > 0$ is a hyperparameter balancing the trajectory costs and the final distributional matching cost. $p_T(\mathbf{z})$ is the distribution of $\mathbf{z}(T)$ resulting from initial states $\mathbf{z}_0 \sim p_0$ propagated through the controlled Neural ODE $d\mathbf{z}/dt = f_{\theta}(\mathbf{z}, \pi_{\phi}(\mathbf{z}, t), t)$.

**3.2 Proposed Framework: OT-NODEC**

The OT-NODEC framework optimizes the Neural ODE dynamics model $f_{\theta}$ (if learned) and the control policy $\pi_{\phi}$ simultaneously using the objective $J(\theta, \phi)$.

*   **Robustness via Adversarial Perturbations:** To achieve robustness against uncertainties (e.g., in $p_0$ or parameters within $f_{\theta}$), we adopt a minimax approach inspired by DRO and robust control. Let $\mathcal{P}_0$ be an uncertainty set for the initial distribution, potentially defined as a Wasserstein ball around a nominal distribution $\hat{p}_0$: $\mathcal{P}_0 = \{ p_0 : W_q(p_0, \hat{p}_0) \le \epsilon \}$. Alternatively, we can consider perturbations $\delta$ to system parameters within $f_{\theta}$ or initial states $\mathbf{z}_0$, belonging to an uncertainty set $\mathcal{U}$. The robust objective becomes:
    $$
    J_{robust}(\theta, \phi) = \sup_{p_0 \in \mathcal{P}_0 \text{ or } \delta \in \mathcal{U}} J(\theta, \phi; p_0 \text{ or } \delta)
    $$
    We aim to solve:
    $$
    \min_{\theta, \phi} J_{robust}(\theta, \phi)
    $$
    In practice, the inner maximization can be approximated by applying adversarial perturbations during training. For instance, finding the perturbation $\delta \in \mathcal{U}$ that maximizes the loss for the current $(\theta, \phi)$ and then minimizing the loss with respect to this worst-case perturbation.

**3.3 Algorithmic Steps**
The training procedure for OT-NODEC involves the following iterative steps:

1.  **Initialization:** Initialize Neural ODE parameters $\theta$ and policy parameters $\phi$. Define the nominal initial distribution $\hat{p}_0$, target distribution $p_{target}$, uncertainty set characterization ($\mathcal{P}_0$ or $\mathcal{U}$), and hyperparameters ($\lambda$, learning rates, $p$ for $W_p$).
2.  **Sampling:** Draw a batch of initial states $\{\mathbf{z}_0^{(i)}\}_{i=1}^N$ from the current estimate of the worst-case initial distribution $p_0^* \in \mathcal{P}_0$ (or from $\hat{p}_0$ if perturbing parameters). Sample target states $\{\mathbf{z}_{target}^{(j)}\}_{j=1}^M$ from $p_{target}$.
3.  **Adversarial Step (Optional but Recommended for Robustness):**
    a. If using adversarial perturbations $\delta \in \mathcal{U}$ (e.g., perturbations to $\mathbf{z}_0$ or parameters in $f_{\theta}$), find the perturbation $\delta^*$ that maximizes the loss $J(\theta, \phi; \delta)$ for the current batch, typically via a few steps of gradient ascent on $\delta$:
       $$
       \delta^* \approx \mathop{\arg \max}_{\delta \in \mathcal{U}} J(\theta, \phi; \delta | \{\mathbf{z}_0^{(i)}\})
       $$
    b. Apply the worst-case perturbation $\delta^*$ for the forward pass.
4.  **Forward Pass (Neural ODE Integration):** For each initial state $\mathbf{z}_0^{(i)}$ (potentially perturbed), integrate the controlled Neural ODE system from $t=0$ to $T$ using a differentiable ODE solver (e.g., `odeint` with adjoint method):
    $$
    \mathbf{z}^{(i)}(T) = \mathbf{z}_0^{(i)} + \int_0^T f_{\theta}(\mathbf{z}^{(i)}(t), \pi_{\phi}(\mathbf{z}^{(i)}(t), t), t) dt
    $$
    Keep track of the trajectory $\{\mathbf{z}^{(i)}(t)\}_{t \in [0,T]}$ and control inputs $\{\mathbf{u}^{(i)}(t) = \pi_{\phi}(\mathbf{z}^{(i)}(t), t)\}_{t \in [0,T]}$. The set $\{\mathbf{z}^{(i)}(T)\}_{i=1}^N$ forms an empirical estimate $\hat{p}_T$ of the final distribution $p_T$.
5.  **Loss Calculation:** Compute the empirical estimate of the objective $J(\theta, \phi)$ or $J_{robust}(\theta, \phi)$:
    a. **Control Cost:** Average the integral and terminal costs over the batch:
       $$
       L_{control} = \frac{1}{N} \sum_{i=1}^N \left[ \int_0^T c(\mathbf{z}^{(i)}(t), \mathbf{u}^{(i)}(t)) dt + g(\mathbf{z}^{(i)}(T)) \right]
       $$
       (The integral is computed numerically alongside the ODE solution).
    b. **OT Cost:** Compute the Wasserstein distance between the empirical distribution $\hat{p}_T$ (represented by samples $\{\mathbf{z}^{(i)}(T)\}$) and the target distribution $p_{target}$ (represented by samples $\{\mathbf{z}_{target}^{(j)}\}$). Use efficient algorithms like Sinkhorn iteration for an entropy-regularized $W_p$ [Cuturi, 2013] or other scalable OT approximations [Gushchin et al., 2023; Onken et al., 2021]:
       $$
       L_{OT} = W_p^p(\hat{p}_T, p_{target})
       $$
    c. **Total Loss:** $L_{total} = L_{control} + \lambda L_{OT}$.
6.  **Backward Pass:** Compute gradients $\nabla_{\theta} L_{total}$ and $\nabla_{\phi} L_{total}$ using backpropagation through the ODE solver (adjoint sensitivity method) and the OT computation.
7.  **Parameter Update:** Update $\theta$ and $\phi$ using an optimizer (e.g., Adam, RMSprop):
    $$
    \theta \leftarrow \theta - \eta_{\theta} \nabla_{\theta} L_{total}
    $$
    $$
    \phi \leftarrow \phi - \eta_{\phi} \nabla_{\phi} L_{total}
    $$
8.  **Iteration:** Repeat steps 2-7 until convergence criteria are met (e.g., loss stabilization, performance on validation set).

**3.4 Experimental Design**

*   **Datasets / Simulation Environments:**
    *   **Robotic Manipulation:** Simulate a planar robot arm (e.g., 2-link or 3-link manipulator) performing reaching or pushing tasks. Introduce distributional shifts by varying:
        *   Initial joint positions/velocities (sampled from different distributions $p_0$).
        *   Physical parameters like link masses, lengths, or joint friction coefficients (sampled from distributions around nominal values).
        *   Payload mass variations.
        Target: A desired distribution of end-effector positions/configurations $p_{target}(\mathbf{z})$ at time $T$.
    *   **Supply Chain Optimization:** Model a multi-echelon inventory system. States $\mathbf{z}(t)$ represent inventory levels. Control inputs $\mathbf{u}(t)$ are replenishment orders. Introduce distributional shifts via:
        *   Varying initial inventory levels $p_0(\mathbf{z})$.
        *   Stochastic customer demand following different distributions (e.g., Gaussian, Poisson, Gamma with varying parameters).
        Target: A desired distribution of inventory levels $p_{target}(\mathbf{z})$ (e.g., centered around safety stock levels with low variance) while minimizing holding and backlog costs.
    *   **Benchmark Control Tasks:** Adapt standard benchmarks like Pendulum or CartPole stabilization/swing-up, but define the task as stabilizing a *distribution* of initial states to a target distribution around the equilibrium, under parametric uncertainty.
*   **Baselines:**
    *   **Neural ODE + Standard Control:** Train dynamics $f_{\theta}$ and policy $\pi_{\phi}$ using only trajectory-based costs ($L_{control}$, i.e., $\lambda=0$).
    *   **Nominal MPC:** Model Predictive Control using the learned Neural ODE $f_{\theta}$ (or a known nominal model) without considering distributional uncertainty.
    *   **Robust MPC:** An MPC variant designed for robustness (e.g., tube-based MPC, scenario-based MPC).
    *   **Standard RL:** Model-free RL algorithms (e.g., SAC, PPO) trained on the nominal environment.
    *   **DRO-based Controller:** A controller optimized using Distributionally Robust Optimization techniques but potentially without Neural ODE dynamics or OT cost structure [e.g., based on Blanchet et al., 2023 ideas].
    *   **Existing OT/NODEs:** If applicable, compare with methods like Opt-ODENet [Miao et al., 2025] focused on safety/stability constraints or other related works [Scagliotti & Farinelli, 2023] if they can be adapted to control.
*   **Evaluation Metrics:**
    *   **Performance:**
        *   Task Success Rate: Percentage of trials successfully reaching the target region/state.
        *   Mean Squared Error (MSE) / Tracking Error: Average deviation from a reference trajectory or target state for nominal conditions.
        *   Wasserstein Distance $W_p(p_T^{actual}, p_{target})$: How well the actual final distribution under the learned policy matches the target distribution, evaluated on test data.
        *   Control Effort: Magnitude of control signals used.
    *   **Robustness:**
        *   Performance Degradation under Shift: Measure the drop in success rate, increase in MSE, or increase in $W_p$ when evaluating the trained policy under various distributional shifts (in $p_0$, parameters) not seen during nominal training. Compare this degradation across methods.
        *   Worst-Case Performance: Evaluate performance under the most challenging perturbations within the defined uncertainty set $\mathcal{U}$.
    *   **Sample Efficiency:** Number of training episodes/samples/gradient steps required to achieve a target level of performance and robustness.
    *   **Computational Cost:** Training time and inference time per control step.
*   **Implementation Details:** Use standard deep learning libraries (PyTorch or JAX) with Neural ODE solvers (`torchdiffeq`, `diffrax`). Utilize libraries for OT computation (POT: Python Optimal Transport, GeomLoss). Experiments will be run on GPU clusters. Code will be made publicly available.

**4. Expected Outcomes & Impact**

**4.1 Expected Research Outcomes**

1.  **A Novel OT-NODEC Framework:** The primary outcome will be the fully developed and implemented OT-NODEC framework, providing a new methodology for designing robust controllers using Neural ODEs and OT. This includes the mathematical formulation, algorithmic implementation, and associated software.
2.  **Empirical Demonstration of Robustness:** We expect experimental results to demonstrate that OT-NODEC significantly outperforms baseline methods in terms of robustness to distributional shifts in initial conditions and system parameters across the selected robotic and supply chain tasks. This will be quantified by lower performance degradation under uncertainty.
3.  **Improved Sample Efficiency (Potentially):** By explicitly shaping the state distribution using OT, the framework might guide the learning process more effectively, potentially leading to improved sample efficiency compared to standard RL or trajectory-following methods, especially in exploring robust strategies.
4.  **Theoretical Contributions:** We aim to provide theoretical analysis regarding the stability of the controlled system under the OT-NODEC policy, convergence properties of the training algorithm (under suitable assumptions), and formal robustness guarantees linked to the Wasserstein distance and the size of the uncertainty set $\epsilon$.
5.  **Publications and Dissemination:** Results will be disseminated through publications in top-tier machine learning and control conferences/journals (e.g., NeurIPS, ICML, CoRL, L4DC, CDC, Automatica) and a presentation at the target workshop. Open-source code release will facilitate reproducibility and further research.

**4.2 Potential Impact**

*   **Advancement in Robust Control:** This research could significantly advance the field of data-driven robust control by providing a principled and effective way to handle complex distributional uncertainties, going beyond traditional set-based uncertainty models.
*   **Bridging Theory and Practice:** By unifying concepts from OT, Neural ODEs, and SOC/DRO, this work will contribute to bridging the gap between theoretical developments in these areas and practical control applications.
*   **Enabling Reliable Autonomous Systems:** Improved robustness is critical for deploying autonomous systems (robots, self-driving cars) in unstructured and unpredictable real-world environments. OT-NODEC could lead to more reliable and adaptable controllers.
*   **Optimizing Complex Systems:** The framework could find applications in optimizing large-scale systems like supply chains, energy grids, or financial systems, where dealing with stochasticity and distributional shifts is crucial for efficiency and resilience.
*   **New Research Directions:** Success in this project could inspire further research into using OT for other aspects of learning in dynamical systems, such as system identification under uncertainty, safe learning, or multi-agent coordination with distributional objectives. It directly addresses the workshop's aim of fostering interdisciplinary research by demonstrating a powerful synergy between learning, control, and OT.

**5. References**

*(Includes papers from the provided literature review and key foundational works)*

1.  Aolaritei, L., Lanzetti, N., Chen, H., & Dörfler, F. (2023). Distributional Uncertainty Propagation via Optimal Transport. *arXiv preprint arXiv:2205.00343*. (Version from 2023 used based on lit review summary context)
2.  Blanchet, J., Kuhn, D., Li, J., & Taskesen, B. (2023). Unifying Distributionally Robust Optimization via Optimal Transport Theory. *arXiv preprint arXiv:2308.05414*.
3.  Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural Ordinary Differential Equations. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.
4.  Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. *Advances in Neural Information Processing Systems (NeurIPS)*, 26.
5.  Di Persio, L., & Garbelli, M. (2023). From Optimal Control to Mean Field Optimal Transport via Stochastic Neural Networks. *Mathematical and Computational Applications*, 28(4), 81.
6.  Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *International Conference on Learning Representations (ICLR)*, 2019.
7.  Gushchin, N., Kolesov, A., Korotin, A., Vetrov, D., & Burnaev, E. (2023). Entropic Neural Optimal Transport via Diffusion Processes. *arXiv preprint arXiv:2211.01156*. (Published at ICRL 2023)
8.  Miao, K., Zhao, L., Wang, H., Gatsis, K., & Papachristodoulou, A. (2025). Opt-ODENet: A Neural ODE Framework with Differentiable QP Layers for Safe and Stable Control Design. *arXiv preprint arXiv:2504.17139*. (Assuming future publication date based on arXiv ID convention)
9.  Mikami, T. (2024). Stochastic Optimal Transport with at Most Quadratic Growth Cost. *arXiv preprint arXiv:2401.14259*. (Version from 2024)
10. Onken, D., Fung, S. W., Li, X., & Ruthotto, L. (2021). OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport. *International Conference on Machine Learning (ICML)*, pp. 8315-8325. PMLR.
11. Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With Applications to Data Science. *Foundations and Trends® in Machine Learning*, 11(5-6), 355-607.
12. Pooladian, A. A., Domingo-Enrich, C., Chen, R. T. Q., & Amos, B. (2024). Neural Optimal Transport with Lagrangian Costs. *arXiv preprint arXiv:2406.00288*.
13. Sandoval, I. O., Petsagkourakis, P., & del Rio-Chanona, E. A. (2022). Neural ODEs as Feedback Policies for Nonlinear Optimal Control. *arXiv preprint arXiv:2210.11245*.
14. Scagliotti, A., & Farinelli, S. (2023). Normalizing Flows as Approximations of Optimal Transport Maps via Linear-Control Neural ODEs. *arXiv preprint arXiv:2311.01404*.
15. Villani, C. (2003). *Topics in Optimal Transportation*. American Mathematical Society.

---