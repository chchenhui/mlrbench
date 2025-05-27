## 1. Title: Neuro-inspired Data-Efficient Reinforcement Learning via Predictive Coding and Active Inference

## 2. Introduction

**2.1 Background**
Artificial Intelligence (AI), particularly driven by advances in Artificial Neural Networks (ANNs), has achieved remarkable success across diverse domains, including sophisticated language generation (Brown et al., 2020) and complex visual tasks (Ramesh et al., 2022). However, a significant bottleneck hindering broader applicability, especially in real-world scenarios, remains the often-prohibitive data requirements of prevailing algorithms, particularly in Reinforcement Learning (RL). Standard RL agents typically necessitate millions, sometimes billions, of interactions with their environment to converge upon effective policies (Mnih et al., 2015; Schulman et al., 2017). This contrasts sharply with biological intelligence, where organisms demonstrate rapid learning and adaptation from surprisingly sparse data.

The burgeoning field of NeuroAI seeks to bridge this gap by drawing inspiration from the principles governing computation and learning in the brain (Hassabis et al., 2017). Two prominent theories from computational neuroscience offer compelling mechanisms for efficient learning and decision-making: Predictive Coding (PC) and Active Inference (AIF) (Friston, 2010; Rao & Ballard, 1999). PC posits that the brain constantly generates predictions about sensory inputs and updates its internal models based on prediction errors. AIF extends this, proposing that actions are selected not merely to achieve goals but fundamentally to minimize *surprise* or, more formally, *expected free energy*, thereby actively seeking information that resolves uncertainty about the world and the agent's goals. This framework suggests perception and action are deeply intertwined processes aimed at maintaining homeostasis and reducing prediction errors over time.

Recent work has begun exploring the integration of these principles into AI systems. For instance, Rao et al. (2022a) introduced active predictive coding for learning hierarchical world models, combining self-supervised learning and RL. Ororbia et al. (2025) proposed meta-representational predictive coding (MPC) as a biologically plausible self-supervised framework driven by AIF principles. Gklezakos & Rao (2022) further explored APCNs for learning compositional structures. These studies highlight the potential of neuro-inspired mechanisms, but a comprehensive RL framework explicitly leveraging AIF for action selection to achieve drastic improvements in sample efficiency remains an active area of research. Integrating AIF directly into the action-selection process of an RL agent, using a world model learned via PC, promises a principled approach to intrinsically motivated exploration and efficient goal achievement.

**2.2 Research Problem**
The primary research problem addressed by this proposal is the **high sample complexity of contemporary RL algorithms**. This limitation restricts their use in domains where data collection is expensive, time-consuming, or dangerous (e.g., robotics, healthcare, autonomous driving). Existing RL approaches often rely on random exploration strategies (e.g., epsilon-greedy) or complex intrinsic motivation heuristics, which can be inefficient, particularly in environments with sparse rewards or complex state spaces requiring structured exploration. The challenge lies in developing an RL paradigm that learns efficiently by making *informed* decisions about exploration, guided by the imperative to reduce uncertainty and refine its understanding of the environment, much like biological agents appear to do.

**2.3 Research Objectives**
This research aims to develop and evaluate a novel RL framework, termed Predictive Coding Active Inference RL (PCAI-RL), designed to significantly enhance data efficiency. The specific objectives are:

1.  **Develop the PCAI-RL Framework:** Formalize and implement an RL agent architecture that integrates a hierarchical Predictive Coding network for world model learning with an Active Inference module for action selection based on Expected Free Energy minimization.
2.  **Implement the PCAI-RL Agent:** Build a functional software implementation of the PCAI-RL agent, capable of interacting with standard RL environment benchmarks. This includes developing efficient algorithms for PC-based model updates and EFE computation.
3.  **Evaluate Sample Efficiency:** Quantitatively assess the sample efficiency of the PCAI-RL agent compared to state-of-the-art model-free (e.g., PPO, DQN) and model-based (e.g., DreamerV3, Dyna-Q) RL algorithms on a suite of benchmark tasks, particularly those known for exploration challenges or sparse rewards.
4.  **Analyze Exploration Behavior:** Qualitatively and quantitatively analyze the exploration strategies emerging from the AIF-driven action selection, comparing them to the exploration patterns of baseline algorithms.
5.  **Investigate Scalability and Robustness:** Assess the computational requirements and robustness of the PCAI-RL framework across different environment complexities and task types.

**2.4 Significance**
This research holds significant potential impacts:

*   **Advancing Data-Efficient RL:** By successfully demonstrating improved sample efficiency, this work could provide a pathway towards more practical RL applications in data-constrained domains.
*   **Bridging AI and Neuroscience:** It serves as a computational testbed for neuroscientific theories (PC/AIF), potentially offering insights back into the functional roles of these mechanisms in biological learning and decision-making. This aligns directly with the goals of the NeuroAI field.
*   **Novel AI Architectures:** The PCAI-RL framework may inspire new classes of AI architectures that learn and reason more effectively by integrating perception, action, and model-building in a principled, uncertainty-aware manner.
*   **Intrinsically Motivated Exploration:** It offers a principled, first-principles approach (minimizing EFE) to intrinsic motivation and exploration, potentially overcoming limitations of ad-hoc heuristics.
*   **Contribution to NeuroAI Community:** This research directly addresses key NeuroAI themes, including Neuro-inspired Computations, Self-supervised Systems (via PC world modeling), and Neuro-inspired reasoning and decision-making (via AIF).

## 3. Methodology

**3.1 Theoretical Framework: Predictive Coding and Active Inference**

**Predictive Coding (PC):** PC models assume a hierarchical generative model where higher layers predict the activity of lower layers, ultimately predicting sensory input. Learning occurs by minimizing the prediction error ($PE$) propagated up the hierarchy. In our context, the agent learns a probabilistic generative model $p(o, s)$ of observations ($o$) and latent states ($s$). Inference involves optimizing variational estimates $q(s)$ of the true posterior $p(s|o)$ to minimize the variational free energy $F$:

$$ F(o, q(s)) = \underbrace{D_{KL}[q(s) || p(s)]}_{\text{Complexity}} - \underbrace{\mathbb{E}_{q(s)}[\log p(o|s)]}_{\text{Accuracy}} $$

Minimizing $F$ ensures the approximate posterior $q(s)$ is close to the true posterior $p(s|o)$ while maximizing the evidence (marginal likelihood) of the observations under the model. Updates typically involve gradient descent on $F$ with respect to the parameters of the generative model and the sufficient statistics of $q(s)$.

**Active Inference (AIF):** AIF extends PC by incorporating action. Agents select actions ($\pi$ or $a$) that minimize the *expected* free energy $G(\pi)$ over a future time horizon:

$$ G(\pi) = \sum_{\tau} \mathbb{E}_{q(o_\tau, s_\tau | \pi)} [ \log q(s_\tau | \pi) - \log p(o_\tau, s_\tau | \pi) ] $$

Where $q(o_\tau, s_\tau | \pi)$ is the predicted distribution over future outcomes and states given a policy $\pi$ (sequence of actions). The expected free energy $G$ can be decomposed into terms reflecting instrumental value (achieving preferred outcomes) and epistemic value (reducing uncertainty):

$$ G(\pi) \approx \underbrace{-\mathbb{E}_{q}[\log p(o_\tau | s_{goal})]}_{\text{Pragmatic/Instrumental Value}} + \underbrace{\mathbb{E}_{q}[\text{Information Gain about model params}]}_{\text{Epistemic Value (Exploration)}} $$

Or more commonly formulated as minimizing expected divergence from goal states and maximizing expected information gain about the world states/parameters. Minimizing $G$ leads to actions that fulfill goals (approach preferred states $s_{goal}$) while simultaneously exploring parts of the environment that promise the greatest reduction in uncertainty about the world dynamics or state representations.

**3.2 Proposed Agent Architecture: PCAI-RL**

The PCAI-RL agent will consist of three core interacting modules:

1.  **Hierarchical Predictive Coding World Model:**
    *   **Structure:** A deep, potentially recurrent neural network implementing hierarchical predictive coding principles, inspired by architectures like those in Rao et al. (2022a) or Gklezakos & Rao (2022). Higher levels represent more abstract temporal or spatial states, predicting the activity of lower levels. The lowest level predicts sensory observations (e.g., pixels, physical state variables).
    *   **Function:** Learns a generative model $p(o_{t+1}, s_{t+1} | s_t, a_t)$ of the environment dynamics.
    *   **Learning:** Parameters are updated via gradient descent on the variational free energy $F$, computed based on prediction errors between predicted and actual subsequent observations/states. This is essentially self-supervised learning of the world model.

2.  **Inference Module:**
    *   **Structure:** Integrated within the PC network or as a separate optimization process.
    *   **Function:** At each timestep $t$, infers the current latent state $s_t$ by optimizing the variational posterior $q(s_t)$ to minimize the current free energy $F_t$ given the observation $o_t$. This involves propagating prediction errors and updating beliefs (sufficient statistics of $q(s_t)$).

3.  **Active Inference Action Selection Module:**
    *   **Structure:** A planning or policy module operating on the latent state space $s$.
    *   **Function:** Plans sequences of potential actions $a_{t:t+H}$ over a horizon $H$. For each sequence (policy $\pi$), it uses the learned world model to predict future states and observations $q(o_{t+1:t+H}, s_{t+1:t+H} | s_t, \pi)$. It then calculates the expected free energy $G(\pi)$ associated with each policy. The action $a_t$ corresponding to the policy $\pi^*$ minimizing $G(\pi)$ is selected.
    *   **EFE Calculation:** The calculation of $G(\pi)$ will involve simulating rollouts using the learned world model and evaluating the expected divergence from goal states (if defined) and the expected information gain (e.g., reduction in uncertainty about model parameters or latent states). This explicitly balances exploration and exploitation. Goal states may be implicitly defined via a prior preference distribution $p(o_{goal})$ or explicitly via reward signals integrated into the pragmatic value term.

**3.3 Algorithmic Steps (Conceptual)**

For each timestep $t$:

1.  **Observe:** Receive observation $o_t$ from the environment.
2.  **Infer State:** Update the belief state $q(s_t)$ by minimizing the free energy $F(o_t, q(s_t))$ using the PC network (inference dynamics).
3.  **Plan Actions:**
    a. Generate a set of candidate action sequences (policies) $\Pi = \{\pi_1, \pi_2, ..., \pi_K\}$ starting from the current inferred state $s_t$. (e.g., using sampling methods like Cross-Entropy Method or gradient-based optimization).
    b. For each policy $\pi \in \Pi$:
        i. Simulate future trajectories $(s_{t+1:\tau}, o_{t+1:\tau})$ using the learned world model $p(o_{\tau'}, s_{\tau'} | s_{\tau'-1}, a_{\tau'-1})$ under policy $\pi$.
        ii. Calculate the expected free energy $G(\pi)$ based on these predicted trajectories, incorporating pragmatic value (preference satisfaction/reward prediction) and epistemic value (information gain/uncertainty reduction).
4.  **Select Action:** Choose the policy $\pi^* = \text{argmin}_{\pi \in \Pi} G(\pi)$. Select the first action $a_t$ from $\pi^*$.
5.  **Execute Action:** Apply action $a_t$ to the environment.
6.  **Learn World Model:** Observe the next state/observation $o_{t+1}$. Calculate prediction errors based on $o_{t+1}$ and the prediction made from $s_t, a_t$. Update the parameters of the PC world model using gradient descent on the free energy (or a surrogate loss like negative log-likelihood combined with KL divergence terms).
7.  **Update Goal/Preferences (Optional):** If applicable, update the representation of goal states or preferences based on received rewards or task specifications.

**3.4 Data Collection: Environments and Benchmarks**

We will evaluate PCAI-RL on a range of environments designed to test sample efficiency and exploration capabilities:

*   **Classic Control (with sparse rewards):** Modify tasks like MountainCar, CartPole (swing-up variant), or Acrobot to provide rewards only upon task completion, making exploration critical.
*   **MiniGrid Environments:** Procedurally generated grid-world environments (Chevalier-Boisvert et al., 2018) offering challenges in navigation, exploration (e.g., `MiniGrid-Empty-Random-Goal`), and sparse rewards (e.g., `MiniGrid-KeyCorridor`). These allow systematic study of exploration in complex state spaces.
*   **DeepMind Control Suite / MuJoCo (with sparse rewards):** Standard continuous control benchmarks (Tassa et al., 2018). We will utilize variants with sparse rewards (e.g., reaching a specific target location) to stress exploration and data efficiency.
*   **Atari Games (Potentially):** Select games known to be hard exploration problems (e.g., Montezuma's Revenge, Pitfall!) (Bellemare et al., 2016) as stretch goals, depending on the computational feasibility of PCAI-RL.

**3.5 Experimental Design**

*   **Baselines:** We will compare PCAI-RL against:
    *   **Model-Free RL:** Proximal Policy Optimization (PPO) (Schulman et al., 2017) and Deep Q-Networks (DQN) (Mnih et al., 2015) or its variants (e.g., Rainbow). These represent standard, widely used algorithms requiring high sample counts.
    *   **Model-Based RL:** DreamerV3 (Hafner et al., 2023) or PlaNet (Hafner et al., 2019), which learn world models but typically optimize actions purely for reward maximization within the learned model. Dyna-Q (Sutton, 1991) as a simpler model-based baseline.
    *   **Exploration-Focused Baselines:** Algorithms incorporating specific exploration bonuses (e.g., RND - Burda et al., 2018) if applicable to the chosen environments.
*   **Evaluation Metrics:**
    *   **Primary Metric:** Sample Efficiency curves (Average Return vs. Number of Environment Steps/Interactions). We will assess the number of steps required to reach specific performance thresholds.
    *   **Asymptotic Performance:** Final performance level achieved after a large number of steps.
    *   **Exploration Metrics:** State visitation frequency/entropy, time to find the first reward (in sparse tasks), qualitative analysis of trajectories.
    *   **Computational Cost:** Training time (wall-clock and/or GPU hours), possibly approximate FLOPs per environment step (though harder to measure accurately across different architectures). This addresses concerns raised by work like SPEQ (Romeo et al., 2025) regarding computational overhead.
*   **Experimental Setup:**
    *   For each algorithm and environment, run multiple independent trials (e.g., 5-10) with different random seeds to ensure statistical significance.
    *   Report mean performance with confidence intervals (e.g., 95% CI) or standard error.
    *   Use standardized implementations of baseline algorithms where possible (e.g., from Stable Baselines3, Tianshou, or original author implementations).
    *   Carefully tune hyperparameters for all algorithms (including PCAI-RL) using a consistent methodology (e.g., limited grid search or random search on a validation set/task variant) to ensure fair comparisons. Key PCAI-RL hyperparameters include PC network architecture, learning rates, EFE planning horizon $H$, components weighting in EFE, and action generation method.

## 4. Expected Outcomes & Impact

**4.1 Expected Outcomes**

1.  **Demonstration of Superior Sample Efficiency:** We expect PCAI-RL to significantly outperform standard model-free and model-based RL baselines in terms of the number of environment interactions required to achieve competent or optimal performance, especially in sparse-reward and complex exploration tasks. The AIF principle, by actively seeking informative states, should guide exploration more effectively than random noise or simple reward-driven planning.
2.  **Comparable or Superior Asymptotic Performance:** While the primary goal is sample efficiency, we anticipate that PCAI-RL will achieve final performance levels comparable to, or potentially exceeding, baselines, as the learned world model and principled exploration strategy should eventually lead to robust policies.
3.  **Qualitative Insights into Exploration:** We expect analysis of agent trajectories and state visitations to reveal more systematic and directed exploration patterns in PCAI-RL compared to the often inefficient or random exploration of baselines. This would provide concrete evidence for the effectiveness of EFE minimization as an exploration drive.
4.  **A Functional, Neuro-inspired RL Framework:** The project will deliver a working implementation of the PCAI-RL agent, providing a valuable tool and foundation for future research in NeuroAI and sample-efficient RL. Initial results may highlight the computational trade-offs involved.
5.  **Potential for Improved World Models:** The PC mechanism might lead to the learning of more structured or interpretable generative world models compared to standard autoencoder-based world models used in some model-based RL.

**4.2 Potential Challenges and Mitigation**

*   **Computational Complexity:** Calculating EFE involves simulating future trajectories and evaluating uncertainty/information gain, which can be computationally intensive, especially with long planning horizons or complex world models. Mitigation: Explore approximations for EFE (e.g., using variational bounds), employ efficient planning techniques (e.g., sampling-based methods), optimize the implementation, and potentially utilize parallel computation. Investigating methods like SPEQ (Romeo et al., 2025) for managing update ratios might be relevant if computational cost becomes prohibitive.
*   **Tuning Complexity:** PCAI-RL introduces new hyperparameters related to the PC model, the inference process, and the EFE calculation (e.g., weighting of pragmatic vs. epistemic value). Mitigation: Systematic hyperparameter sweeps, sensitivity analysis, potentially developing adaptive mechanisms for balancing exploration/exploitation within the EFE framework.
*   **Stability of PC Learning:** Training deep predictive coding networks can sometimes face stability issues. Mitigation: Employ architectural best practices (e.g., normalization layers, appropriate activation functions), careful initialization, adaptive learning rates, and potentially regularization techniques.
*   **Accuracy of Learned World Model:** The effectiveness of AIF planning relies heavily on the accuracy of the PC world model. An inaccurate model could lead to suboptimal or misleading EFE calculations. Mitigation: Ensure sufficient capacity and training stability for the PC network; potentially incorporate model uncertainty explicitly into the EFE calculation.

**4.3 Impact**

Successful completion of this research will make significant contributions to both AI and computational neuroscience. It will provide a novel, biologically-inspired framework for tackling the critical challenge of data efficiency in RL, potentially unlocking applications in robotics, autonomous systems, and other domains where data is scarce. By demonstrating the computational viability and benefits of PC and AIF principles within an RL context, it will strengthen the ties between the fields, offering a concrete example of how neuroscientific theories can lead to tangible improvements in AI capabilities, directly contributing to the goals of the NeuroAI community. Furthermore, the insights gained into how AIF drives exploration and learning could inform our understanding of similar processes in the brain, furthering the cycle of discovery between natural and artificial intelligence.

---
**References mentioned in the proposal text (Implicit or explicit based on provided context):**

*   Bellemare, M. G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying count-based exploration and intrinsic motivation. *Advances in neural information processing systems*, 29.
*   Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in neural information processing systems*, 33, 1877-1901.
*   Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. *arXiv preprint arXiv:1810.12894*.
*   Chevalier-Boisvert, M., Willems, L., & Pal, S. (2018). Minimalistic gridworld environment for openai gym. *https://github.com/maximecb/gym-minigrid*.
*   Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature reviews neuroscience*, 11(2), 127-138.
*   Gklezakos, D. C., & Rao, R. P. N. (2022). Active Predictive Coding Networks: A Neural Solution to the Problem of Learning Reference Frames and Part-Whole Hierarchies. *arXiv preprint arXiv:2201.08813*.
*   Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019). Dream to control: Learning behaviors by latent imagination. *International Conference on Learning Representations*.
*   Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*.
*   Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. (2017). Neuroscience-inspired artificial intelligence. *Neuron*, 95(2), 245-258.
*   Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
*   Ororbia, A., Friston, K., & Rao, R. P. N. (2025). Meta-Representational Predictive Coding: Biomimetic Self-Supervised Learning. *arXiv preprint arXiv:2503.21796* (Note: Fictional future date used as provided).
*   Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*.
*   Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature neuroscience*, 2(1), 79-87.
*   Rao, R. P. N., Gklezakos, D. C., & Sathish, V. (2022a). Active Predictive Coding: A Unified Neural Framework for Learning Hierarchical World Models for Perception and Planning. *arXiv preprint arXiv:2210.13461*.
*   Romeo, C., Macaluso, G., Sestini, A., & Bagdanov, A. D. (2025). SPEQ: Stabilization Phases for Efficient Q-Learning in High Update-To-Data Ratio Reinforcement Learning. *arXiv preprint arXiv:2501.08669* (Note: Fictional future date used as provided).
*   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
*   Sutton, R. S. (1991). Dyna, an integrated architecture for learning, planning, and reacting. *ACM Sigart Bulletin*, 2(4), 160-163.
*   Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., Casas, D. d. L., ... & Riedmiller, M. (2018). Deepmind control suite. *arXiv preprint arXiv:1801.00690*.