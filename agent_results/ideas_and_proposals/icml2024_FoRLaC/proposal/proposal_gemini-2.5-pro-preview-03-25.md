# **1. Title: Lyapunov-Informed Reinforcement Learning for Provably Stable and Robust Control Policies**

# **2. Introduction**

**2.1 Background**
Reinforcement Learning (RL) has demonstrated remarkable success in solving complex sequential decision-making problems across various domains, including game playing, robotics, and recommendation systems [1, 2]. By learning policies through interaction with an environment to maximize a cumulative reward signal, RL offers a powerful framework for deriving adaptive control strategies directly from data, often bypassing the need for explicit system modeling. However, the majority of contemporary RL algorithms, particularly those employing deep neural networks (Deep RL), lack formal guarantees regarding the stability and safety of the learned policies. This deficiency significantly hinders their deployment in high-stakes, safety-critical applications such as autonomous driving, industrial process control, aerospace systems, and medical robotics, where unpredictable or unstable behavior can have catastrophic consequences [3, 4].

Conversely, control theory has a long-standing tradition of developing methods that provide rigorous guarantees on system behavior, with stability being a cornerstone concept [5]. Lyapunov stability theory, in particular, offers a powerful mathematical framework for analyzing the stability of dynamic systems without explicitly solving the system's differential or difference equations. A scalar function, known as a Lyapunov function, is used to certify that the system's state remains bounded or converges to a desired equilibrium point. Control Lyapunov Functions (CLFs) extend this concept by providing a means to synthesize controllers that guarantee stability [6].

Despite the shared goal of controlling dynamic systems, RL and control theory have historically evolved with limited interaction. RL excels in optimizing performance metrics in complex, possibly unknown environments, while control theory provides the tools for ensuring robustness and stability, often relying on accurate system models. There is a growing recognition that bridging these two fields holds immense potential for developing next-generation control systems that are both high-performing and reliable [7]. Integrating the stability guarantees of control theory, specifically Lyapunov theory, into the adaptive learning framework of RL can pave the way for controllers that learn complex behaviors while adhering to strict safety and stability constraints.

**2.2 Literature Context and Motivation**
Recent years have witnessed increasing interest in integrating Lyapunov concepts into RL frameworks to imbue learned policies with stability guarantees. Several promising approaches have emerged:
*   **Neural Lyapunov Functions:** Researchers have explored using neural networks to learn or approximate Lyapunov functions for nonlinear systems, sometimes combined with RL for data generation or policy refinement ([McCuthcheon et al., 2025; Zhang, 2024]). These works demonstrate the feasibility of learning stability certificates.
*   **CLF-Based RL:** Methods like SAC-CLF ([Chen et al., 2025]) incorporate CLF constraints directly into RL optimization (e.g., Soft Actor-Critic), ensuring the learned policy respects stability conditions during action selection or updates. Similar approaches use Lyapunov-based constraints in policy optimization ([Sun et al., 2023; Han et al., 2023; Jiang et al., 2023]).
*   **Safe Exploration and Training:** Some works focus on ensuring stability not only for the final policy but also during the learning process itself ([Zhang, 2024]).
*   **Distributed Control:** Lyapunov-based RL has also been extended to distributed settings for large-scale systems ([Yao et al., 2024]).
*   **Robustness:** A few studies explicitly target robustness alongside stability using Lyapunov techniques ([Chen et al., 2025; Chen et al., 2024]).

While these studies represent significant progress, several key challenges remain, as highlighted in the literature review:
1.  **Lyapunov Function Design/Learning:** Finding or learning effective Lyapunov functions, especially for complex, high-dimensional nonlinear systems, remains difficult. The learned function must accurately capture stability properties across the relevant state space.
2.  **Balancing Performance and Stability:** Enforcing strict stability constraints can sometimes overly restrict the policy space, potentially hindering the discovery of high-reward policies. Finding the right balance is crucial.
3.  **Computational Complexity:** Jointly learning policies, value functions, and Lyapunov functions, often involving constrained optimization, increases computational demands.
4.  **Robustness to Uncertainty:** Ensuring stability guarantees hold in the presence of model inaccuracies, unmodeled dynamics, or external disturbances is critical for real-world deployment but often requires more sophisticated Lyapunov analysis (e.g., Input-to-State Stability - ISS).

This research proposal directly addresses these challenges by proposing a novel **Lyapunov-Informed Reinforcement Learning (LIRL)** framework. Our approach focuses on:
*   **Joint Optimization:** Co-adapting the policy, value function, and a neural Lyapunov function within a constrained policy optimization framework, specifically utilizing a Lagrangian dual method for principled constraint handling.
*   **Explicit Robustness:** Integrating robustness considerations directly into the Lyapunov conditions and learning process, potentially drawing from robust control theory concepts like ISS Lyapunov functions.
*   **Sample Efficiency:** Investigating techniques inspired by model-based RL or self-supervised learning (similar to [McCuthcheon et al., 2025]) to improve data efficiency in learning the Lyapunov function across the state space.
*   **Theoretical Analysis:** Aiming to provide formal (probabilistic) guarantees on the stability and robustness of the learned policies under specific assumptions.

**2.3 Research Objectives**
The primary objectives of this research are:
1.  **Develop the LIRL Framework:** Formulate and implement a novel RL framework that integrates Lyapunov stability and robustness constraints directly into the policy optimization process using neural networks for the policy, value function, and Lyapunov function. This involves designing the network architectures, the combined loss function incorporating Lagrangian multipliers for constraints, and the update rules.
2.  **Derive Theoretical Guarantees:** Analyze the LIRL framework to establish theoretical conditions under which the learned policy achieves closed-loop stability (e.g., asymptotic stability within a region of attraction) and robustness to bounded disturbances, potentially relating to Input-to-State Stability (ISS).
3.  **Empirical Validation:** Rigorously evaluate the LIRL framework on a suite of challenging continuous control benchmarks, including standard nonlinear systems (e.g., Pendulum, Acrobot) and simulated robotic tasks (e.g., Reacher, Hopper stabilization).
4.  **Comparative Analysis:** Compare the performance, stability, robustness, and sample efficiency of LIRL against state-of-the-art unconstrained RL algorithms (e.g., SAC, DDPG) and existing safe/stable RL methods identified in the literature review.

**2.4 Significance**
This research holds significant potential for advancing both RL and control theory. By systematically integrating Lyapunov stability and robustness principles into RL, we aim to:
*   **Enable RL in Safety-Critical Domains:** Provide a pathway for deploying powerful RL techniques in applications where safety and reliability are paramount, fostering trust in learned controllers.
*   **Enhance Control System Design:** Offer a new methodology for designing high-performance, adaptive controllers for complex nonlinear systems with formal guarantees, potentially outperforming traditional control methods in certain scenarios.
*   **Bridge Theory and Practice:** Contribute to the theoretical foundations of safe and reliable learning systems, aligning with the goals of the "Foundations of RL and Control" workshop by strengthening the connection between the two fields.
*   **Develop Robust AI:** Address a critical limitation of current AI systems by incorporating formal methods for ensuring desirable behavior (stability, robustness) under uncertainty.

The successful completion of this research would provide a validated algorithmic framework and theoretical insights, paving the way for more reliable autonomous systems and industrial automation solutions.

# **3. Methodology**

**3.1 Problem Formulation**
We consider discrete-time nonlinear dynamical systems of the form:
$$x_{t+1} = f(x_t, u_t) + w_t$$
where $x_t \in \mathcal{X} \subseteq \mathbb{R}^n$ is the system state at time $t$, $u_t \in \mathcal{U} \subseteq \mathbb{R}^m$ is the control action, $f: \mathcal{X} \times \mathcal{U} \rightarrow \mathcal{X}$ is the (potentially unknown) system dynamics function, and $w_t \in \mathcal{W}$ represents bounded disturbances or model uncertainties, with $\|w_t\| \leq W_{max}$.

The goal is to learn a stochastic policy $\pi_\theta(u_t | x_t)$, parameterized by $\theta$, that maximizes the expected discounted cumulative reward:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(x_t, u_t) \right]$$
where $r(x_t, u_t)$ is the reward function, $\gamma \in [0, 1)$ is the discount factor, and $\tau = (x_0, u_0, x_1, u_1, \dots)$ is a trajectory generated by the policy $\pi_\theta$ and the system dynamics.

Crucially, we impose a stability constraint motivated by Lyapunov theory. We aim for the closed-loop system under policy $\pi_\theta$ to be stable around a desired equilibrium point $x_{eq}$ (assumed to be the origin without loss of generality). This is formalized using a candidate Lyapunov function $V_L(x; \psi)$, approximated by a neural network with parameters $\psi$.

**3.2 Lyapunov Stability and Robustness Conditions**
A continuous function $V_L: \mathcal{X} \rightarrow \mathbb{R}_{\ge 0}$ is a Lyapunov function candidate if it is positive definite, i.e., $V_L(0) = 0$ and $V_L(x) > 0$ for $x \neq 0$. For stability in the discrete-time setting, we require the expected Lyapunov function value to decrease along trajectories generated by the policy:
$$\mathbb{E}_{x_{t+1} \sim f(x_t, u_t) + w_t, u_t \sim \pi_\theta(\cdot|x_t)} [V_L(x_{t+1}; \psi)] - V_L(x_t; \psi) \leq -\alpha(V_L(x_t; \psi)) + \sigma(\|w_t\|)$$
where $\alpha(\cdot)$ is a class $\mathcal{K}$ function (continuous, strictly increasing, $\alpha(0)=0$) ensuring decrease in the nominal case ($w_t=0$), and $\sigma(\cdot)$ is related to the effect of disturbances, aiming for an Input-to-State Stability (ISS)-like property [8]. A simpler, more practical condition often used is:
$$\mathbb{E}_{x_{t+1}, u_t} [V_L(x_{t+1}; \psi)] - V_L(x_t; \psi) \leq -\delta + \epsilon_{robust}$$
for $x_t$ outside some small neighborhood of the origin, where $\delta > 0$ is a desired decay rate, and $\epsilon_{robust}$ accounts for the worst-case effect of disturbances $w_t$.

The Lyapunov network $V_L(x; \psi)$ itself must be trained to satisfy the positive definiteness condition. This can be encouraged by including a loss term like $\text{ReLU}(-V_L(x; \psi) + \epsilon_V \|x\|^2)$ for some small $\epsilon_V > 0$ over sampled states $x$, ensuring $V_L(x; \psi) \ge \epsilon_V \|x\|^2$.

**3.3 Proposed Framework: Lyapunov-Informed Reinforcement Learning (LIRL)**
We propose an actor-critic framework where the actor (policy $\pi_\theta$) is updated to maximize reward while satisfying the Lyapunov stability constraint, and the critic provides value estimates. A separate neural network learns the Lyapunov function $V_L(x; \psi)$.

*   **Architecture:**
    *   Policy Network $\pi_\theta(u_t | x_t)$: Outputs parameters of a distribution over actions (e.g., mean and variance for a Gaussian policy).
    *   Value Network(s) $Q_{\phi_1}(x_t, u_t), Q_{\phi_2}(x_t, u_t)$: Estimate the expected return (Q-value), potentially using twin networks for stability (as in SAC). We might also use a state-value function $V_\phi(x_t)$.
    *   Lyapunov Network $V_L(x; \psi)$: Outputs a scalar estimate of the Lyapunov function value. Requires positive output (e.g., using squared output layer or ReLU activation). Must satisfy $V_L(0; \psi) = 0$.

*   **Constrained Optimization Problem:**
    The policy optimization problem is formulated as:
    $$\max_\theta J(\theta)$$
    Subject to:
    $$C(x, \theta, \psi) = \mathbb{E}_{u \sim \pi_\theta(\cdot|x), x' \sim f(x,u)+w} [V_L(x'; \psi)] - V_L(x; \psi) + \alpha(V_L(x;\psi)) \leq \bar{\epsilon}_{robust}$$
    for all $x \in \mathcal{X}_{interest}$, where $\mathcal{X}_{interest}$ is the relevant region of the state space, and $\bar{\epsilon}_{robust}$ accounts for the expected worst-case disturbance effects based on $W_{max}$.

*   **Lagrangian Formulation:** We employ a Lagrangian dual approach to handle the constraint. The Lagrangian is:
    $$L(\theta, \lambda) = J(\theta) - \lambda \cdot \mathbb{E}_{x \sim \mathcal{D}} \left[ \max(0, C(x, \theta, \psi) - \bar{\epsilon}_{robust}) \right]$$
    where $\lambda \ge 0$ is the Lagrange multiplier, and the expectation is taken over a state distribution $\mathcal{D}$ (e.g., from the replay buffer). The policy parameters $\theta$ are updated to maximize $L$, while $\lambda$ is updated (typically via gradient ascent) to enforce the constraint:
    $$\theta_{k+1} = \theta_k + \eta_\theta \nabla_\theta L(\theta_k, \lambda_k)$$
    $$\lambda_{k+1} = \max(0, \lambda_k + \eta_\lambda \nabla_\lambda L(\theta_k, \lambda_k)) = \max(0, \lambda_k + \eta_\lambda \mathbb{E}_{x \sim \mathcal{D}} [\max(0, C(x, \theta_k, \psi_k) - \bar{\epsilon}_{robust})])$$

*   **Algorithm Outline (based on an off-policy actor-critic like SAC):**

    1.  Initialize policy network $\pi_\theta$, value networks $Q_{\phi_1}, Q_{\phi_2}$, target value networks $Q'_{\phi'_1}, Q'_{\phi'_2}$, Lyapunov network $V_L(x; \psi)$, target Lyapunov network $V'_L(x; \psi')$, Lagrange multiplier $\lambda \ge 0$, and replay buffer $\mathcal{B}$.
    2.  **Loop for each episode/step:**
        a.  Observe state $x_t$.
        b.  Sample action $u_t \sim \pi_\theta(\cdot|x_t)$.
        c.  Execute $u_t$, observe reward $r_t$ and next state $x_{t+1}$.
        d.  Store transition $(x_t, u_t, r_t, x_{t+1})$ in $\mathcal{B}$.
        e.  **If time to update:**
            i. Sample a minibatch $M = \{(x_j, u_j, r_j, x'_{j})\}_{j=1}^N$ from $\mathcal{B}$.
            ii. **Update Value Networks ($Q_{\phi_i})$:** Compute target Q-values using target networks and Bellman equation (incorporating entropy bonus if using SAC):
            $$y_j = r_j + \gamma \mathbb{E}_{u' \sim \pi_\theta(\cdot|x'_j)} [\min_{i=1,2} Q'_{\phi'_i}(x'_j, u') - \alpha_{ent} \log \pi_\theta(u'|x'_j)]$$
            Update $\phi_i$ by minimizing MSE loss: $\mathcal{L}_Q = \frac{1}{N} \sum_j (Q_{\phi_i}(x_j, u_j) - y_j)^2$.
            iii. **Update Policy Network ($\pi_\theta$):** Update $\theta$ by maximizing the Lagrangian objective:
            $$ \nabla_\theta L \approx \mathbb{E}_{x \sim M, u \sim \pi_\theta(\cdot|x)} [ \nabla_\theta \log \pi_\theta(u|x) \cdot (Q_{\text{target}}(x,u) - \alpha_{ent} \log \pi_\theta(u|x)) - \lambda \cdot \nabla_\theta C(x, \theta, \psi) ] $$
            (Using reparameterization trick for gradient estimation through $Q$ and $C$). The term $Q_{\text{target}}$ is typically $\min_i Q_{\phi_i}(x, u)$. The constraint term $C(x, \theta, \psi)$ involves sampling hypothetical next states under the current policy and evaluating $V_L$. Specifically:
            $$C(x, \theta, \psi) = \mathbb{E}_{u \sim \pi_\theta(\cdot|x), x' \sim \hat{f}(x,u)} [V_L(x'; \psi)] - V_L(x; \psi) + \alpha(V_L(x;\psi))$$
            where $\hat{f}$ might be the observed transition or a learned model.
            iv. **Update Lyapunov Network ($V_L(x; \psi)$):** Train $\psi$ to satisfy Lyapunov conditions. Minimize a loss function like:
            $$ \mathcal{L}_V = \frac{1}{N} \sum_j \left( \text{ReLU}(-V_L(x_j; \psi) + \epsilon_V \|x_j\|^2) + \text{ReLU}( \mathbb{E}_{u_j \sim \pi_\theta(\cdot|x_j)}[V_L(x'_j; \psi')] - V_L(x_j; \psi) + \delta ) \right) $$
            The expectation uses the target Lyapunov network $\psi'$ for stability. The target decrease value $\delta$ might be adaptive or fixed. The first term enforces positive definiteness.
            v. **Update Lagrange Multiplier ($\lambda$):**
            $$\lambda \leftarrow \max(0, \lambda + \eta_\lambda \cdot \frac{1}{N} \sum_j \max(0, C(x_j, \theta, \psi) - \bar{\epsilon}_{robust}) )$$
            vi. **Update Target Networks:** Softly update target network parameters: $\phi' \leftarrow \tau \phi + (1-\tau) \phi'$, $\psi' \leftarrow \tau \psi + (1-\tau) \psi'$.

*   **Handling Unknown Dynamics and Disturbances ($f, w$):** The expectation in the constraint $C(x, \theta, \psi)$ requires knowledge of $x'$. In a model-free setting, we use the observed next states $x'_j$ from the replay buffer. To handle disturbances $w_t$, $\bar{\epsilon}_{robust}$ should be estimated based on assumptions about $W_{max}$ and the Lipschitz constant of $V_L$. Alternatively, robust optimization techniques or learning a dynamics model could be incorporated.

**3.4 Data Collection**
Data will be collected through interaction with simulated environments. We will primarily use an off-policy approach, leveraging a replay buffer $\mathcal{B}$ to store past transitions. This improves sample efficiency compared to on-policy methods. The exploration strategy needs careful consideration; initially, exploration might be broad, but as the Lyapunov function becomes more reliable, exploration could be guided to stay within safer regions, potentially by penalizing actions predicted to violate the Lyapunov decrease condition significantly.

**3.5 Experimental Design**
*   **Environments:** We will use standard continuous control benchmarks from OpenAI Gym and potentially PyBullet/MuJoCo, chosen for their varying dynamics and stability challenges:
    *   *Classic Control:* Inverted Pendulum (stabilization), CartPole (swing-up and stabilization, potentially modified for continuous actions), Acrobot (swing-up).
    *   *Robotics Simulators:* Reacher (reaching a target), Hopper (stabilization/balancing), potentially a simple simulated autonomous vehicle model focusing on lane keeping or obstacle avoidance stability.
    *   *System with known dynamics:* Linear Quadratic Regulator (LQR) or a simple known nonlinear system to verify theoretical properties where ground truth stability is known.

*   **Baselines:**
    *   *Unconstrained RL:* State-of-the-art algorithms like Soft Actor-Critic (SAC) [9] and Deep Deterministic Policy Gradient (DDPG) [10].
    *   *Safe/Stable RL:* Representative methods from the literature review, such as SAC-CLF ([Chen et al., 2025]) or Lyapunov-based constrained policy optimization ([Han et al., 2023]), reimplemented if necessary for fair comparison.

*   **Evaluation Metrics:**
    *   *Performance:* Average cumulative reward per episode, convergence speed (episodes/timesteps to reach target performance).
    *   *Stability:*
        *   Region of Attraction (ROA): Estimate the set of initial states from which the system converges to the equilibrium.
        *   State Trajectory Analysis: Maximum state deviation from equilibrium, settling time, overshoot (for stabilization tasks).
        *   Lyapunov Value Decay: Plot $V_L(x_t; \psi)$ over time for sample trajectories; verify consistent decrease.
        *   Constraint Satisfaction Rate: Percentage of time steps where the Lyapunov decrease condition $C(x, \theta, \psi) \leq \bar{\epsilon}_{robust}$ holds during evaluation.
    *   *Robustness:*
        *   Perturbation Analysis: Introduce various perturbations during evaluation (e.g., constant or random bounded forces, sensor noise on $x_t$, changes in system parameters like mass or friction) and measure performance degradation and stability maintenance compared to baselines.
        *   ISS-like metrics: Measure the maximum state deviation as a function of the magnitude of sustained disturbances ($W_{max}$).
    *   *Computational Cost:* Training time, number of parameters, inference time per action.

*   **Experimental Protocol:** For each environment and algorithm, multiple runs with different random seeds will be conducted. Results will be reported with means and confidence intervals (e.g., standard deviation or standard error). Ablation studies will be performed to assess the contribution of different components of LIRL (e.g., effect of the Lagrange multiplier vs. simple penalty, impact of the robustness term $\bar{\epsilon}_{robust}$). Sensitivity analysis to hyperparameters (e.g., $\eta_\theta, \eta_\lambda, \delta, \tau$) will also be conducted.

# **4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
We expect this research to yield the following outcomes:
1.  **A Novel LIRL Framework:** A complete algorithmic framework (LIRL) for training RL agents with integrated Lyapunov-based stability and robustness constraints, implemented and available as open-source code.
2.  **Theoretical Analysis:** Formal (probabilistic) guarantees on the stability (e.g., ultimate boundedness or asymptotic stability within a learned ROA) and robustness (e.g., Input-to-State Stability margins) of the closed-loop system under the learned LIRL policy, under clearly stated assumptions (e.g., Lipschitz continuity of dynamics and Lyapunov function, sufficient exploration).
3.  **Empirical Validation and Benchmarking:** Comprehensive experimental results demonstrating the effectiveness of LIRL on challenging control benchmarks. We expect LIRL to achieve:
    *   Comparable or slightly reduced task performance (cumulative reward) compared to unconstrained RL baselines, due to the imposed constraints.
    *   Significantly improved stability properties (e.g., smaller state deviations, guaranteed convergence from a larger set of initial states) compared to unconstrained RL.
    *   Quantifiably better robustness to external disturbances and model uncertainties compared to both unconstrained RL and potentially simpler stability-constrained methods that do not explicitly account for robustness.
    *   Clear evidence of the Lyapunov function decrease along system trajectories during evaluation.
4.  **Comparative Insights:** A clear understanding of the trade-offs between performance, stability, robustness, and computational complexity provided by the LIRL framework compared to existing methods. This includes insights into the effectiveness of the Lagrangian approach for balancing reward maximization and constraint satisfaction.
5.  **Publications:** Dissemination of findings through publications in top-tier machine learning (e.g., NeurIPS, ICML, ICLR) and control theory (e.g., CDC, ACC, Automatica) venues, including potential contributions to workshops like the "Foundations of RL and Control" workshop described in the task.

**4.2 Potential Impact**
The successful completion of this research has the potential for significant impact:
*   **Advancing Safe RL:** This work will contribute directly to the field of Safe Reinforcement Learning by providing a principled and theoretically grounded method for incorporating stability, a fundamental aspect of safety in dynamic systems.
*   **Bridging RL and Control Theory:** By integrating core concepts from Lyapunov theory into a modern deep RL framework, this research strengthens the synergistic connection between these two fields, fostering cross-disciplinary understanding and innovation as targeted by the workshop.
*   **Enabling Real-World Deployment:** By providing formal stability and robustness guarantees, LIRL can increase the trustworthiness of RL-based controllers, making them more viable for deployment in safety-critical applications like autonomous vehicles, robotics operating near humans, power grid management, and industrial automation. This could unlock significant economic and societal benefits.
*   **New Research Directions:** This research may open up new avenues for investigation, such as extending the framework to partially observable systems (POMDPs), incorporating safety constraints beyond stability (e.g., state or action constraints), developing adaptive methods for estimating disturbance bounds online, and exploring connections to verification methods for neural networks.
*   **Foundation for Certified AI:** The ability to learn controllers with provable properties like stability is a step towards building certifiable AI systems, where performance and safety can be formally verified, addressing a major bottleneck in the adoption of AI in high-assurance systems.

In summary, this research aims to deliver a robust and reliable approach to learning control policies by deeply integrating Lyapunov stability theory within reinforcement learning. By addressing key limitations of current RL methods, we anticipate that the LIRL framework will be a valuable tool for researchers and practitioners seeking to develop high-performing, safe, and trustworthy autonomous systems.

---
**References** (Placeholder - based on standard RL/Control literature and provided review)
[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.
[2] Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.
[3] Amodei, D., et al. (2016). Concrete problems in AI safety. *arXiv preprint arXiv:1606.06565*.
[4] Garcia, J., & Fernández, F. (2015). A comprehensive survey on safe reinforcement learning. *Journal of Machine Learning Research*, 16(1), 1437-1480.
[5] Khalil, H. K. (2002). *Nonlinear systems*. Prentice Hall.
[6] Artstein, Z. (1983). Stabilization with relaxed controls. *Nonlinear Analysis: Theory, Methods & Applications*, 7(11), 1163-1173.
[7] Recht, B. (2019). A Tour of Reinforcement Learning: The View from Continuous Control. *Annual Review of Control, Robotics, and Autonomous Systems*, 2, 253-279.
[8] Sontag, E. D. (1998). *Mathematical control theory: deterministic finite dimensional systems*. Springer Science & Business Media.
[9] Haarnoja, T., et al. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *International conference on machine learning (ICML)*.
[10] Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.
*+ References [1-10] from the provided literature review.*