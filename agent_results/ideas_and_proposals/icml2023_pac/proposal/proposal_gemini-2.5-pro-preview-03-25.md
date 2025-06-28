```latex
\documentclass[11pt]{article}
\usepackage{amsmath, amssymb, amsthm}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}

\begin{document}

\section*{1. Title}

**PAC-Bayesian Guided Policy Optimization with Uncertainty-Aware Exploration for Sample-Efficient Reinforcement Learning**

\section*{2. Introduction}

**2.1 Background**

Reinforcement Learning (RL) has achieved remarkable success in complex sequential decision-making tasks, ranging from game playing \cite{silver2016mastering} to robotic control \cite{levine2016end}. However, a significant bottleneck hindering the widespread adoption of RL, particularly deep RL methods employing neural networks, is its often-prohibitive *sample inefficiency*. Many state-of-the-art algorithms require millions or even billions of interactions with the environment to learn effective policies \cite{kaiser2019model}. This inefficiency largely stems from the fundamental challenge of balancing exploration (gathering information about the environment) and exploitation (using current knowledge to maximize rewards). Traditional exploration strategies, such as ε-greedy or adding Gaussian noise to actions, are often heuristic, lack strong theoretical guarantees, and can explore inefficiently, revisiting well-understood states or failing to probe critical but rarely encountered situations, especially in environments with sparse or delayed rewards.

Interactive learning settings, including RL, inherently involve a sequential decision-making process where data is collected adaptively based on the learning agent's actions. This non-i.i.d. nature of data collection further complicates theoretical analysis and algorithmic design. While Bayesian methods offer a natural framework for representing and reasoning about uncertainty, crucial for efficient exploration, their integration into deep RL has often focused on empirical performance gains without rigorous theoretical backing regarding generalization or sample complexity \cite{gal2016uncertainty}.

Parallelly, PAC-Bayesian theory \cite{mcallester1999pac} has emerged as a powerful tool within statistical learning theory for providing non-asymptotic generalization bounds for probabilistic predictors. Unlike traditional uniform convergence bounds, PAC-Bayesian bounds characterize the generalization error of a *distribution* over hypotheses (the posterior $Q$) relative to a prior distribution $P$. The bounds typically depend on the Kullback-Leibler (KL) divergence between $Q$ and $P$, and the empirical performance of hypotheses sampled from $Q$. Recently, PAC-Bayesian analysis has shown promise in explaining the generalization capabilities of deep neural networks \cite{neyshabur2017pac} and has begun to be explored in interactive learning settings \cite{tasdighi2024deep, tasdighi2023pac, majumdar2018pac}. These bounds offer a principled way to quantify the trade-off between model complexity (captured by the KL term) and empirical performance, which directly relates to the exploration-exploitation dilemma in RL: excessive exploration corresponds to high uncertainty (potentially large KL or poor empirical fit under the posterior mean), while insufficient exploration leads to confidently learning a suboptimal policy (low KL but poor generalization to unexplored state-action regions).

Despite these initial efforts, a significant gap remains in developing practical deep RL algorithms that *directly optimize* a PAC-Bayesian objective and leverage the inherent uncertainty quantification for *systematic, theoretically grounded exploration*. Existing PAC-Bayesian RL works often focus on bounding specific components like the Bellman error \cite{tasdighi2024deep} or using the bound as a regularizer for the critic \cite{tasdighi2023pac}, rather than driving the core policy optimization and exploration strategy. Furthermore, adapting PAC-Bayesian bounds, traditionally developed for i.i.d. data in supervised learning, to the sequential, non-stationary, and policy-dependent data distribution of RL poses considerable theoretical challenges \cite{chugg2023unified}.

**2.2 Research Objectives**

This research aims to bridge this gap by developing a novel PAC-Bayesian framework for policy optimization in deep RL, designed to enhance sample efficiency through principled, uncertainty-aware exploration. The specific objectives are:

1.  **Develop a PAC-Bayesian Policy Optimization (PBPO) framework:** Formulate a theoretically grounded PAC-Bayesian bound applicable to the expected cumulative reward in an RL setting. Derive a tractable objective function based on this bound for learning a distribution over policies represented by deep neural networks.
2.  **Design an Uncertainty-Aware Exploration Strategy:** Utilize the policy posterior distribution, learned by optimizing the PAC-Bayes objective, to directly guide exploration. Design a mechanism that actively seeks out state-action regions where the policy posterior exhibits high variance or uncertainty.
3.  **Integrate PAC-Bayes Objective with Actor-Critic Methods:** Develop a practical algorithm, tentatively named PAC-Bayes Policy Optimization (PBPO), that integrates the PAC-Bayesian objective into a standard actor-critic architecture (e.g., similar to Soft Actor-Critic (SAC) \cite{haarnoja2018soft} or PPO \cite{schulman2017proximal}), ensuring compatibility with continuous and high-dimensional state-action spaces.
4.  **Provide Theoretical Justification:** Analyze the properties of the proposed PAC-Bayesian bound in the RL context, potentially leveraging tools for non-i.i.d. or time-uniform bounds \cite{chugg2023unified}, to connect the minimization of the bound to improved sample complexity and safe exploration.
5.  **Empirically Validate PBPO:** Evaluate the proposed algorithm on challenging benchmark RL environments (e.g., continuous control suites like MuJoCo, and potentially exploration-focused tasks like Atari games with sparse rewards). Compare its sample efficiency, final performance, and training stability against state-of-the-art baselines.

**2.3 Significance**

This research holds significant potential for advancing both the theory and practice of reinforcement learning:

*   **Improved Sample Efficiency:** By directly optimizing a PAC-Bayes bound and using the resulting uncertainty for exploration, the proposed method aims to achieve significant improvements in sample efficiency compared to current deep RL algorithms, making RL more viable for real-world applications where data collection is expensive or time-consuming (e.g., robotics, healthcare, recommendation systems).
*   **Principled Exploration:** It offers a move away from heuristic exploration strategies towards a theoretically grounded approach based on uncertainty quantification derived from the PAC-Bayes framework, potentially leading to more robust and efficient discovery of optimal policies, especially in complex environments.
*   **Theoretical Advancement:** This work contributes to the burgeoning field of PAC-Bayesian analysis for interactive learning, specifically addressing the challenges of applying these bounds to RL policy optimization and exploration. It aims to provide tighter connections between PAC-Bayesian theory and RL performance guarantees.
*   **Robustness and Safety:** Quantifying policy uncertainty can pave the way for developing safer RL algorithms. High uncertainty estimates in critical states could trigger conservative actions or requests for human intervention, enhancing reliability in real-world deployments.
*   **Alignment with Workshop Goals:** This research directly addresses the core themes of the "PAC-Bayes Meets Interactive Learning" workshop, contributing to the development of practically useful interactive learning algorithms using PAC-Bayesian theory, analyzing exploration-exploitation trade-offs via PAC-Bayes, and potentially offering insights into bounds under distribution shifts inherent in RL.

\section*{3. Methodology}

**3.1 Theoretical Foundation: PAC-Bayes Bounds**

The core idea of PAC-Bayesian theory \cite{mcallester1999pac, catoni2007pac} is to bound the generalization error of a randomized predictor $Q$ (posterior distribution) with respect to a fixed prior distribution $P$ over a hypothesis space $\mathcal{H}$. For a loss function $\ell(h, z)$, where $h \in \mathcal{H}$ is a hypothesis and $z$ is a data point drawn from an unknown distribution $\mathcal{D}$, the goal is to bound the true risk $R(Q) = \mathbb{E}_{h \sim Q} [\mathbb{E}_{z \sim \mathcal{D}}[\ell(h, z)]]$ based on the empirical risk $\hat{R}_N(Q) = \mathbb{E}_{h \sim Q} [\frac{1}{N} \sum_{i=1}^N \ell(h, z_i)]$ observed on $N$ samples $z_1, ..., z_N$. A common form of the PAC-Bayes bound (e.g., based on Donsker-Varadhan's variational formula for KL divergence) states that with probability at least $1-\delta$ over the draw of the $N$ samples:
$$
R(Q) \le \hat{R}_N(Q) + \sqrt{\frac{KL(Q || P) + \ln(N/\delta)}{2N}}
$$
or, using McAllester's bound:
$$
R(Q) \le \inf_{\lambda > 0} \mathbb{E}_{h \sim Q} \left[ \hat{R}_{N, \lambda}(h) + \frac{KL(Q||P) + \ln(1/\delta)}{\lambda N} \right]
$$
where $\hat{R}_{N, \lambda}(h)$ relates to the empirical risk under certain conditions (e.g., bounded loss). More generally, for any convex function $\phi$, with probability $1-\delta$:
$$
\mathbb{E}_{h \sim Q} [R(h)] \le \mathbb{E}_{h \sim Q} [\hat{R}_N(h)] + \sqrt{\frac{KL(Q || P) + \ln(1/\delta)}{2N}} \quad (\text{if loss } \in [0,1])
$$
Our work will adapt these principles to the RL setting.

**3.2 PAC-Bayesian Formulation for RL**

We consider a Markov Decision Process (MDP) defined by $(\mathcal{S}, \mathcal{A}, T, R, \gamma, \rho_0)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $T(s'|s, a)$ is the transition probability function, $R(s, a, s')$ is the reward function, $\gamma \in [0, 1)$ is the discount factor, and $\rho_0$ is the initial state distribution. A policy $\pi(a|s)$ maps states to distributions over actions. The objective is to find a policy $\pi$ that maximizes the expected discounted cumulative reward $J(\pi) = \mathbb{E}_{\tau \sim p(\cdot|\pi)} [\sum_{t=0}^\infty \gamma^t r_t]$, where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$ is a trajectory generated by executing policy $\pi$ ($s_0 \sim \rho_0, a_t \sim \pi(\cdot|s_t), s_{t+1} \sim T(\cdot|s_t, a_t), r_t = R(s_t, a_t, s_{t+1})$).

We propose to learn a distribution $Q$ over policies $\pi$, parameterized by $\theta \in \Theta$. Let $P$ be a prior distribution over $\Theta$. Our goal is to optimize $Q$ to maximize $J(Q) = \mathbb{E}_{\theta \sim Q} [J(\pi_\theta)]$. We will derive a PAC-Bayesian bound on the *negative* of the expected return, treating $-J(\pi_\theta)$ as the "risk" or "loss". A key challenge is that the data (trajectories $\tau$) used to estimate the performance depends on the policy distribution $Q$ itself.

Let $\hat{J}_N(Q)$ be an empirical estimate of the expected return based on $N$ interactions (e.g., total steps or episodes) collected using policies sampled from $Q$ or a related exploration policy. We hypothesize a bound of the form, holding with probability at least $1-\delta$:
$$
-J(Q) \le -\hat{J}_N(Q) + \mathcal{C}(KL(Q || P), N, \delta)
$$
where $\mathcal{C}$ is a complexity term, likely analogous to the $\sqrt{\frac{KL + \ln(1/\delta)}{N}}$ term in supervised learning, but potentially adapted for the sequential, non-i.i.d. nature of RL data using techniques like those developed for time-uniform PAC-Bayes bounds \cite{chugg2023unified} or bounds on dependent data. Minimizing the right-hand side serves as a surrogate objective for maximizing $J(Q)$.

The empirical estimate $\hat{J}_N(Q)$ will likely be based on estimates from a critic network. For instance, in an actor-critic setting, we might use the expected Q-value from the critic $Q^\phi(s, a)$ averaged over states visited and actions sampled according to $Q$:
$$
\hat{J}_N(Q) \approx \hat{\mathbb{E}}_{s \sim d^Q, a \sim \pi_Q(\cdot|s)} [Q^\phi(s, a)]
$$
where $d^Q$ is the state visitation measure induced by $Q$, estimated from collected trajectories, and $\pi_Q(a|s) = \mathbb{E}_{\theta \sim Q}[\pi_\theta(a|s)]$ or involves sampling $\theta \sim Q$.

The optimization objective for the parameters of the posterior distribution $Q$ (e.g., mean $\mu$ and variance $\Sigma$ if $Q = \mathcal{N}(\mu, \Sigma)$) becomes:
$$
\min_{\text{params of } Q} \left\{ -\hat{\mathbb{E}}_{s \sim d^Q, a \sim \pi_Q(\cdot|s)} [Q^\phi(s, a)] + \beta \cdot KL(Q(\theta) || P(\theta)) \right\}
$$
where $\beta$ incorporates the scaling factors ($N, \delta$) from the PAC-Bayes bound $\mathcal{C}$. This objective explicitly trades off maximizing expected performance (first term) with minimizing model complexity relative to the prior (second term).


**3.3 Algorithmic Design: PBPO**

We propose the PAC-Bayes Policy Optimization (PBPO) algorithm, built upon an actor-critic framework.

1.  **Policy Representation:** The policy $\pi_\theta(a|s)$ is represented by a deep neural network with parameters $\theta$. We learn a posterior distribution $Q(\theta)$ over these parameters, typically parameterized as a multivariate Gaussian with diagonal covariance: $Q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$, where $\mu$ and $\sigma$ are outputs of neural networks or learned directly. A simple prior $P(\theta)$ is chosen, e.g., a standard Gaussian $P(\theta) = \mathcal{N}(0, I)$.

2.  **Critic Network(s):** One or two Q-networks $Q^\phi(s, a)$ (parameterized by $\phi$) are learned, similar to SAC or TD3, to estimate the expected return. They are trained using Bellman updates with target networks and potentially techniques like clipped double Q-learning to mitigate overestimation. The target value computation may incorporate entropy terms if building on SAC.
    $$
    y = r + \gamma \mathbb{E}_{a' \sim \pi_{\bar{\theta}}(\cdot|s')} [Q_{\text{target}}^{\phi'}(s', a') - \alpha \log \pi_{\bar{\theta}}(a'|s')] \quad (\text{SAC-style target})
    $$
    where $\bar{\theta}$ might be the mean $\mu$ of $Q$, or sampled from $Q$. The critic loss minimizes the TD error:
    $$
    L_{\text{critic}}(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} [(Q^\phi(s, a) - y)^2]
    $$
    where $\mathcal{B}$ is the replay buffer.

3.  **Actor Optimization (PBPO Objective):** The parameters of the policy posterior $Q(\theta)$ (i.e., $\mu$ and $\sigma$) are updated to minimize the PAC-Bayesian objective derived in Section 3.2. Using the reparameterization trick ($\theta = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$), we can write the objective as:
    $$
    L_{\text{actor}}(\mu, \sigma) = \mathbb{E}_{s \sim \mathcal{B}, \epsilon \sim \mathcal{N}(0,I)} [ -Q^{\phi}(s, \pi_{\mu + \sigma \odot \epsilon}(s)) ] + \beta \cdot KL(\mathcal{N}(\mu, \text{diag}(\sigma^2)) || \mathcal{N}(0, I))
    $$
    (If using SAC-style entropy maximization, the Q-value term might be replaced by $Q^{\phi}(s, a_\theta) - \alpha \log \pi_\theta(a_\theta|s)$ where $a_\theta \sim \pi_\theta(\cdot|s)$, averaged over $\theta \sim Q$). The coefficient $\beta$ balances exploitation (minimizing negative Q-value) and complexity/implicit exploration drive (minimizing KL divergence). $\beta$ might be annealed or adaptively tuned based on the theoretical bound derivation.

4.  **Uncertainty-Aware Exploration:** Exploration is driven by the uncertainty captured in the posterior $Q(\theta)$. Instead of adding noise heuristically (like in DDPG or TD3) or relying solely on entropy (like in SAC), we propose:
    *   **Thompson Sampling:** During interaction with the environment, sample a policy parameter vector $\theta_k \sim Q(\theta)$ for each episode (or potentially more frequently) and act according to $\pi_{\theta_k}(a|s)$. This naturally explores actions/states where different policies sampled from the posterior disagree.
    *   **Variance-Based Exploration (Alternative):** Estimate the variance of the action or Q-value under the posterior $Q$. For instance, estimate $Var_{\theta \sim Q}[\pi_\theta(a|s)]$ or $Var_{\theta \sim Q}[Q^\phi(s, \pi_\theta(a|s))]$. Use this variance as an exploration bonus or to directly modify the action selection probability, encouraging exploration in high-variance regions. This might require approximations or ensemble methods applied to the policy network based on $Q$. We will primarily focus on Thompson Sampling due to its direct connection to sampling from the posterior $Q$ optimized via the PAC-Bayes objective.

5.  **Handling Non-stationarity:** The inherent non-stationarity in RL (data distribution changes as policy improves) can be partially addressed by:
    *   Using replay buffers, which break temporal correlations but introduce off-policy learning challenges.
    *   Potentially employing time-uniform PAC-Bayes bounds in the theoretical analysis \cite{chugg2023unified}.
    *   Adapting the prior $P$ or the balance coefficient $\beta$ over time, though this requires careful theoretical justification.

**Pseudocode for PBPO Training Loop:**

```
Initialize policy posterior parameters (μ, σ), critic parameters (φ1, φ2), target networks (φ1', φ2').
Initialize prior P(θ) (e.g., N(0, I)). Set PAC-Bayes coefficient β.
Initialize replay buffer B.

for episode k = 1 to K do:
  Sample policy parameters θ_k ~ Q(θ | μ, σ) = N(μ, diag(σ^2)).
  Observe initial state s_0.
  for t = 0 to T-1 do:
    # Uncertainty-Aware Exploration (Thompson Sampling)
    Select action a_t ~ π_{θ_k}(·|s_t).
    Execute a_t, observe reward r_t and next state s_{t+1}.
    Store transition (s_t, a_t, r_t, s_{t+1}) in B.

    # Update networks periodically (e.g., every step or N steps)
    Sample a minibatch of transitions {(s_j, a_j, r_j, s_{j+1})} from B.

    # Critic Update
    Compute target value y_j (e.g., using target networks and policy samples).
    Update critic parameters φ1, φ2 by minimizing L_critic(φ1, φ2) on the batch.

    # Actor Update (PBPO Objective)
    Update policy posterior parameters μ, σ by minimizing L_actor(μ, σ) using reparameterization trick and critic Q^{\phi1}.
    (Optionally include entropy term if based on SAC framework).

    # Update target networks
    Update φ1', φ2' towards φ1, φ2 (polyak averaging).
  end for
end for
```

**3.4 Experimental Design**

*   **Environments:** We will evaluate PBPO on standard benchmarks:
    *   **Continuous Control:** OpenAI Gym environments using the MuJoCo physics engine (e.g., Hopper-v3, Walker2d-v3, Ant-v3, Humanoid-v3). These benchmarks are widely used and test locomotion skills.
    *   **Challenging Exploration (if feasible):** Selected Atari games known for sparse rewards or requiring deep exploration (e.g., Montezuma's Revenge, Pitfall!). This would require adapting PBPO for discrete action spaces.
    *   **Classic Control (for initial validation):** CartPole, Pendulum for faster prototyping and debugging.
*   **Baselines:** We will compare PBPO against state-of-the-art RL algorithms known for good performance and sample efficiency:
    *   Soft Actor-Critic (SAC) \cite{haarnoja2018soft}: A strong baseline for continuous control, using entropy maximization for exploration.
    *   Proximal Policy Optimization (PPO) \cite{schulman2017proximal}: A popular and robust policy gradient method.
    *   Deep Deterministic Policy Gradient (DDPG) \cite{lillicrap2015continuous} / Twin Delayed DDPG (TD3) \cite{fujimoto2018addressing}: Baselines for continuous control using deterministic policies with added noise for exploration.
    *   If implementations are available: PAC-Bayes Actor-Critic (PBAC) \cite{tasdighi2024deep} and PAC-Bayes SAC \cite{tasdighi2023pac} to directly compare against related PAC-Bayesian RL approaches.
*   **Evaluation Metrics:**
    *   **Sample Efficiency:** Average cumulative reward as a function of environment steps or episodes. We will measure the number of steps required to reach specific performance thresholds.
    *   **Asymptotic Performance:** Final average reward achieved after a fixed large number of training steps (e.g., 1 million or 3 million steps for MuJoCo).
    *   **Training Stability:** Variance of performance across multiple training runs with different random seeds. We will report means and standard deviations (or confidence intervals) for all metrics.
*   **Ablation Studies:** To understand the contribution of each component:
    *   Compare PBPO with Thompson sampling exploration vs. PBPO using standard ε-greedy or Gaussian noise exploration (keeping the PAC-Bayes objective).
    *   Compare PBPO vs. a standard actor-critic algorithm (e.g., SAC) but using Thompson sampling based on a simple Bayesian interpretation (e.g., dropout uncertainty) instead of the PAC-Bayes driven posterior.
    *   Analyze the effect of the PAC-Bayes coefficient $\beta$ on the exploration-exploitation trade-off and performance.
*   **Implementation Details:** Implementations will be based on standard libraries like PyTorch. We will use standard network architectures and hyperparameters for baselines where available and tune PBPO-specific parameters (like $\beta$) appropriately. All experiments will be run across multiple random seeds (e.g., 5-10) for statistical robustness.

\section*{4. Expected Outcomes & Impact}

*   **Expected Outcomes:**
    1.  **A Novel RL Algorithm (PBPO):** A practical deep RL algorithm that optimizes a policy distribution by minimizing a PAC-Bayesian bound and utilizes the learned posterior for uncertainty-aware exploration (Thompson Sampling).
    2.  **Theoretical Framework:** A derivation and analysis of a PAC-Bayesian bound tailored for the RL setting, providing theoretical insight into how minimizing this bound can lead to sample-efficient and robust policy learning. This may involve leveraging or extending techniques for non-i.i.d. data.
    3.  **Empirical Validation:** Comprehensive experimental results on standard RL benchmarks demonstrating that PBPO achieves superior or competitive sample efficiency and performance compared to state-of-the-art baselines like SAC and PPO.
    4.  **Ablation Study Insights:** Clear evidence from ablation studies isolating the benefits of the PAC-Bayesian objective and the uncertainty-aware exploration strategy derived from it.
    5.  **Open-Source Implementation:** A reusable implementation of the PBPO algorithm will be made available to facilitate further research.

*   **Impact:**
    1.  **Advancing Sample-Efficient RL:** This work aims to address a critical limitation of current RL methods, potentially leading to algorithms usable in real-world scenarios where interaction is costly (e.g., robotics, autonomous systems, clinical trials).
    2.  **Bridging Theory and Practice:** By developing a practical algorithm directly motivated by PAC-Bayesian theory, this research strengthens the link between theoretical learning guarantees and empirical RL performance, fostering further interaction between these communities.
    3.  **Principled Exploration Strategies:** The project will contribute a principled, uncertainty-driven exploration mechanism grounded in PAC-Bayes theory, moving beyond common heuristics and potentially offering better performance, particularly in challenging exploration tasks.
    4.  **Foundation for Safer RL:** The explicit quantification of policy uncertainty within the PAC-Bayesian framework could serve as a foundation for developing RL systems with enhanced safety and reliability guarantees.
    5.  **Contribution to PAC-Bayes in Interactive Learning:** This research directly addresses the goals of the workshop by demonstrating a novel application of PAC-Bayesian theory to develop effective algorithms for interactive learning (specifically RL) and providing new theoretical and empirical insights into PAC-Bayesian analysis of exploration-exploitation trade-offs.

\section*{5. References}

\begin{thebibliography}{99}

\bibitem{catoni2007pac}
Olivier Catoni.
\newblock PAC-Bayesian supervised classification: The thermodynamics of statistical learning.
\newblock {\em IMS Lecture Notes Monograph Series}, 51:xiv+1-241, 2007.

\bibitem{chugg2023unified}
Ben Chugg, Hongjian Wang, and Aaditya Ramdas.
\newblock A Unified Recipe for Deriving (Time-Uniform) PAC-Bayes Bounds.
\newblock {\em arXiv preprint arXiv:2302.03421}, 2023.

\bibitem{fujimoto2018addressing}
Scott Fujimoto, Herke van Hoof, and David Meger.
\newblock Addressing Function Approximation Error in Actor-Critic Methods.
\newblock In {\em International conference on machine learning (ICML)}, 2018.

\bibitem{gal2016uncertainty}
Yarin Gal and Zoubin Ghahramani.
\newblock Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.
\newblock In {\em International conference on machine learning (ICML)}, 2016.

\bibitem{haarnoja2018soft}
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.
\newblock Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
\newblock In {\em International conference on machine learning (ICML)}, 2018.

\bibitem{kaiser2019model}
Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al.
\newblock Model-Based Reinforcement Learning for Atari.
\newblock In {\em International Conference on Learning Representations (ICLR)}, 2020. (arXiv:1903.00374)

\bibitem{levine2016end}
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel.
\newblock End-to-end training of deep visuomotor policies.
\newblock {\em Journal of Machine Learning Research (JMLR)}, 17(39):1--40, 2016.

\bibitem{lillicrap2015continuous}
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra.
\newblock Continuous control with deep reinforcement learning.
\newblock In {\em International Conference on Learning Representations (ICLR)}, 2016. (arXiv:1509.02971)

\bibitem{majumdar2018pac}
Anirudha Majumdar, Alec Farid, and Anoopkumar Sonar.
\newblock PAC-Bayes Control: Learning Policies that Provably Generalize to Novel Environments.
\newblock {\em arXiv preprint arXiv:1806.04225}, 2018.

\bibitem{mcallester1999pac}
David A. McAllester.
\newblock PAC-Bayesian model averaging.
\newblock In {\em Proceedings of the twelfth annual conference on Computational learning theory (COLT)}, pages 164--170, 1999.

\bibitem{neyshabur2017pac}
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro.
\newblock Exploring generalization in deep learning.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2017.

\bibitem{schulman2017proximal}
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
\newblock Proximal Policy Optimization Algorithms.
\newblock {\em arXiv preprint arXiv:1707.06347}, 2017.

\bibitem{silver2016mastering}
David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al.
\newblock Mastering the game of Go with deep neural networks and tree search.
\newblock {\em Nature}, 529(7587):484--489, 2016.

\bibitem{tasdighi2023pac}
Bahareh Tasdighi, Abdullah Akgül, Manuel Haussmann, Kenny Kazimirzak Brink, and Melih Kandemir.
\newblock PAC-Bayesian Soft Actor-Critic Learning.
\newblock {\em arXiv preprint arXiv:2301.12776}, 2023.

\bibitem{tasdighi2024deep}
Bahareh Tasdighi, Manuel Haussmann, Nicklas Werge, Yi-Shan Wu, and Melih Kandemir.
\newblock Deep Exploration with PAC-Bayes.
\newblock {\em arXiv preprint arXiv:2402.03055}, 2024.

\end{thebibliography}

\end{document}
```