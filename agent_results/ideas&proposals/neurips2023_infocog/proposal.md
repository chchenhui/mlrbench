**1. Title:** **Optimizing Human-AI Communication in Cooperative Tasks via the Information Bottleneck Principle**

**2. Introduction**

*   **Background:** The advent of increasingly capable artificial intelligence (AI) presents significant opportunities for human-AI collaboration across diverse domains, from complex manufacturing and scientific discovery to everyday assistance and creative endeavours. Effective collaboration, however, fundamentally relies on efficient and intuitive communication between human and AI partners. Current AI systems often struggle with communication: some overwhelm human partners with excessive, unfiltered information, hindering cognitive processing and decision-making (Endsley, 1995), while others provide overly sparse or ambiguous signals, leading to potentially critical misunderstandings and degraded task performance (Klein et al., 2004). This communication gap represents a major bottleneck to realizing the full potential of human-AI teaming. Information theory provides a powerful mathematical framework for reasoning about communication under constraints. Specifically, the Information Bottleneck (IB) principle (Tishby et al., 2000) offers a principled approach to optimize the trade-off between retaining relevant information (informativeness) and compressing the representation (complexity or bandwidth). By framing communication as a process of extracting task-relevant information from an agent's internal state while minimizing the complexity of the communicated signal, the IB principle holds significant promise for developing AI agents that communicate more effectively with human partners. Recent work has started exploring IB for multi-agent communication (Wang et al., 2020) and grounding communication in task utility (Tucker et al., 2022), but a focused application to optimize communication *specifically* for human cognitive constraints within goal-oriented cooperative tasks remains an important research area. Furthermore, the InfoCog workshop's emphasis on bridging information theory, machine learning, and cognitive science, particularly concerning the computation and validation of information-theoretic formalisms in cognitive systems and the development of human-aligned agents, makes it an ideal venue for this research.

*   **Problem Statement:** The central problem addressed by this proposal is how to train AI agents participating in cooperative tasks to communicate with human partners in a way that is simultaneously maximally informative for the joint task objective and minimally complex from the perspective of human cognitive processing limitations. The challenge lies in finding a principled way to balance the need for conveying sufficient task-relevant information (e.g., intentions, perceptions, warnings) against the need for conciseness and clarity to avoid overwhelming the human partner and facilitate rapid understanding. Existing approaches often rely on hand-crafted communication protocols or heuristics, lacking a formal optimisation framework that directly considers this trade-off.

*   **Research Objectives:** This research aims to develop and evaluate a novel framework for learning efficient human-AI communication strategies grounded in the Information Bottleneck principle integrated within a reinforcement learning (RL) context. The specific objectives are:
    1.  **Develop an IB-RL Framework for Human-AI Communication:** Formulate a learning objective that combines standard RL rewards for task completion with an Information Bottleneck regularizer on the agent's communication policy. This objective will explicitly trade off maximizing the mutual information between the communication signal and task-relevant variables against minimizing the mutual information between the signal and the agent's complete internal state.
    2.  **Implement the Framework using Deep Variational Methods:** Utilize deep neural networks and variational inference techniques (Variational Information Bottleneck - VIB; Alemi et al., 2017) to approximate the IB objective and learn complex, state-dependent communication policies within simulated cooperative environments.
    3.  **Train AI Agents with Efficient Communication Strategies:** Train agents using the proposed IB-RL framework in representative cooperative tasks (e.g., collaborative navigation, joint assembly, puzzle-solving) requiring communication. We will investigate how varying the IB trade-off parameter ($\beta$) influences the learned communication protocols and their properties (e.g., lexicon size, message length, semantic content).
    4.  **Evaluate the Effectiveness and Efficiency of Learned Communication:** Empirically evaluate the performance of agents trained with the IB-RL framework compared to relevant baselines. Evaluation will encompass:
        *   Joint human-AI task performance (e.g., success rate, completion time).
        *   Communication efficiency (e.g., information rate, message complexity).
        *   Human-centric metrics (e.g., perceived cognitive load, communication clarity, trust, ability to predict agent behaviour).

*   **Significance:** This research is significant for several reasons. Firstly, it addresses a critical bottleneck in human-AI collaboration, potentially leading to more effective and seamless teaming in complex applications like autonomous driving, assistive robotics, and collaborative analysis. Secondly, it advances the use of information-theoretic principles (specifically IB) in the design of intelligent agents, contributing to the theoretical foundations of machine learning and AI alignment, directly aligning with the InfoCog workshop themes. Thirdly, by explicitly considering human cognitive limits through the IB compression objective, it offers a path towards AI systems that are not just capable but also more naturally interactable and understandable for humans, fostering better trust and adoption. Fourthly, it responds directly to the call for methods that validate information-theoretic formalisms in cognition and apply them to training human-aligned agents. Finally, the project will contribute novel methodologies for estimating and optimizing information-theoretic quantities (via VIB) in the context of interactive RL agents, relevant to researchers focused on computation and estimation within the information theory community.

**3. Methodology**

*   **Conceptual Framework:** We model the human-AI cooperative task as a Partially Observable Markov Decision Process (POMDP) or a Dec-POMDP setting where communication is essential. The AI agent possesses an internal state, $X$, which encapsulates its observations, beliefs, intentions, and possibly its planned actions. To cooperate effectively, the agent needs to communicate relevant aspects of this state to the human partner. We formulate this communication generation process using the Information Bottleneck principle. The goal is to learn a stochastic communication policy, represented by an encoder $p(Z|X)$, that maps the internal state $X$ to a communication signal $Z$. This signal $Z$ could be discrete tokens (e.g., words, symbols) or potentially structured representations. The IB objective seeks to find an encoder that maximizes the mutual information $I(Z; Y)$ between the signal $Z$ and a *task-relevant variable* $Y$, while simultaneously minimizing the mutual information $I(X; Z)$ between the signal $Z$ and the agent's full internal state $X$. The variable $Y$ represents what the *human needs to know* to successfully cooperate (e.g., agent's goal, intended next move, perceived hazard location, confidence level). The term $I(X; Z)$ quantifies the complexity or bandwidth of the communication channel; minimizing it forces the agent to compress its state representation into a concise signal, implicitly considering cognitive efficiency.

*   **Mathematical Formulation:**
    The Information Bottleneck optimization problem is defined by the Lagrangian:
    $$ \mathcal{L}_{IB} = I(Z; Y) - \beta I(X; Z) $$
    where $Z$ is the compressed representation (communication signal) obtained from the source variable $X$ (agent's internal state) via a probabilistic encoder $p(z|x)$. $Y$ is the relevance variable (task-relevant information), which is assumed to follow a joint distribution $p(x, y)$. $\beta$ is a Lagrange multiplier that controls the trade-off between retaining relevant information about $Y$ and compressing $X$. A larger $\beta$ encourages more compression (simpler communication), while a smaller $\beta$ prioritizes informativeness about $Y$.

    Since directly optimizing mutual information terms is often intractable, especially with high-dimensional $X$ and complex dependencies learned by deep neural networks, we will employ the Variational Information Bottleneck (VIB) framework (Alemi et al., 2017). VIB provides a tractable lower bound on $I(Z; Y)$ and an upper bound on $I(X; Z)$. The objective becomes maximizing:
    $$ \mathcal{L}_{VIB} \approx \mathbb{E}_{p(x,y)} \left[ \int p(z|x) \log q(y|z) dz \right] - \beta \mathbb{E}_{p(x)} \left[ D_{KL}(p(z|x) || r(z)) \right] $$
    Here:
    *   $p(z|x)$ is the stochastic encoder (communication policy) parameterized by a neural network $\theta$. It maps agent state $x$ to a distribution over signals $z$.
    *   $q(y|z)$ is a variational approximation to $p(y|z)$, parameterized by a neural network $\phi$. It predicts the relevance variable $Y$ given the communication signal $Z$. This network effectively learns to decode the relevant information from the learned communication protocol.
    *   $r(z)$ is a prior distribution over the communication signals (e.g., a standard Gaussian for continuous $Z$, or a uniform categorical distribution for discrete $Z$).
    *   $D_{KL}(\cdot || \cdot)$ denotes the Kullback-Leibler divergence. The KL term $D_{KL}(p(z|x) || r(z))$ approximates $I(X; Z)$ and acts as the compression cost.

    **Integration with Reinforcement Learning:** We embed the VIB objective within a deep RL framework (e.g., Actor-Critic). The agent's policy $\pi(a, z | s)$ will output both an action $a$ for the environment and a communication signal $z$, conditioned on its current state $s$ (which corresponds to $X$ in the IB formulation).
    1.  **Actor Network:** Parameterizes the policy $\pi_\theta(a, z | s)$. The communication part of this network implements the VIB encoder $p_\theta(z|s)$.
    2.  **Critic Network:** Estimates the value function $V_\psi(s)$ or state-action value function $Q_\psi(s, a)$ to guide policy updates.
    3.  **Auxiliary Decoder Network:** Parameterizes $q_\phi(y|z)$ to predict the relevance variable $Y$ from the generated communication $Z$.

    The overall loss function for training the actor network will combine the standard RL policy gradient loss (e.g., A2C or PPO loss) with the VIB objective:
    $$ \mathcal{L}_{Total} = \mathcal{L}_{RL} - \lambda \mathcal{L}_{VIB} $$
    where $\mathcal{L}_{RL}$ promotes actions leading to high task rewards, $\mathcal{L}_{VIB}$ encourages informative yet concise communication (as defined above), and $\lambda$ is a hyperparameter balancing task performance and communication optimisation. The relevance variable $Y$ needs careful definition based on the task; it could be the agent's intended action, target location, assessment of risk, or even a prediction of the human's needs based on the shared state. $Y$ might be directly available in simulation (e.g., agent's ground truth goal) or require auxiliary prediction based on the agent's state.

*   **Algorithmic Steps:**
    1.  Initialize policy network $\pi_\theta$, value network $V_\psi$, VIB decoder $q_\phi$, and prior $r(z)$.
    2.  For each episode:
        a. Agent observes state $s_t$.
        b. Agent samples action $a_t$ and communication signal $z_t$ from $\pi_\theta(a, z | s_t)$.
        c. Agent executes $a_t$ in the environment, receives reward $r_t$ and next state $s_{t+1}$. The signal $z_t$ is hypothetically communicated (and potentially used by a simulated human model or logged for later analysis).
        d. Store transition $(s_t, a_t, z_t, r_t, s_{t+1})$ in a buffer.
        e. Define/extract the relevance variable $y_t$ corresponding to $s_t$ (e.g., agent's target at time $t$).
    3.  Periodically update network parameters:
        a. Compute RL advantage estimates (e.g., GAE).
        b. Compute RL loss $\mathcal{L}_{RL}$ (e.g., PPO clipped objective).
        c. Compute VIB loss $\mathcal{L}_{VIB}$ using $s_t$, $z_t$, $y_t$, requiring forward passes through $p_\theta(z|s)$ and $q_\phi(y|z)$. The expectation is approximated using samples from the buffer.
        d. Compute the total loss $\mathcal{L}_{Total} = \mathcal{L}_{RL} - \lambda \mathcal{L}_{VIB}$.
        e. Perform gradient descent on $\theta$, $\psi$, and $\phi$ using $\mathcal{L}_{Total}$ (or separate updates for RL and VIB components).
    4.  Repeat steps 2-3 until convergence.

*   **Data Collection / Environment:** We will primarily use simulated cooperative environments that necessitate communication. Potential candidates include:
    *   **Overcooked-AI environment (Carroll et al., 2019):** A benchmark for human-AI cooperation requiring coordination in a shared kitchen. Communication can signal intentions, needed ingredients, or locations.
    *   **Collaborative Block Construction:** A simpler grid-world task where agents and humans need to coordinate building structures, requiring communication about plans and placements.
    *   **Constrained Human-AI Cooperation (CHAIC) variants (Du et al., 2024):** Environments where AI assists a (simulated) human with constraints, requiring inference and communication about needs and actions.
    Data collected will include state trajectories, actions, generated communication signals, task rewards, and the defined relevance variable $Y$. For human-centric evaluations, we will conduct user studies with participants interacting with agents trained with different communication strategies (IB-based vs. baselines). Data will include task performance logs, communication logs, and subjective questionnaires (e.g., NASA-TLX for cognitive load, custom questionnaires for clarity, trust, usability).

*   **Experimental Design:**
    *   **Baselines:**
        1.  *No Communication:* Agent trained with RL only, without explicit communication channel.
        2.  *Full Communication:* Agent communicates its entire internal state representation (or a large subset) without IB compression.
        3.  *Rule-Based Communication:* Agent uses pre-defined rules to communicate specific state information.
        4.  *Alternative Compression:* Agent trained using a standard autoencoder on its state, communicating the latent code (focusing on reconstruction, not task relevance).
        5.  *Related Methods:* If feasible, reimplementations or comparisons with concepts from PLLB (Srivastava et al., 2024) focusing on rule generation, or IMAC (Wang et al., 2020) adapted for human-AI.
    *   **Ablation Studies:**
        1.  Vary the IB trade-off parameter $\beta$ to study its impact on communication strategy (complexity vs. informativeness) and task performance.
        2.  Vary the definition of the relevance variable $Y$ (e.g., goal vs. next action vs. perceived obstacle).
        3.  Compare different VIB encoder/decoder architectures (e.g., MLP, RNN).
        4.  Evaluate the impact of the VIB regularizer weight $\lambda$.
    *   **Validation:**
        1.  *Simulation Studies:* Compare joint task performance (success rate, completion time) and communication metrics (message rate, $I(X; Z)$ estimate, $I(Z; Y)$ estimate) across different agents and $\beta$ values in the simulated environments. Analyze the emergent communication protocols (e.g., semantics of signals).
        2.  *Human Subject Studies:* Recruit participants to perform cooperative tasks with AI agents trained using different communication strategies (our IB method vs. key baselines). Measure:
            *   Objective performance: Joint task success, time.
            *   Human performance: Individual efficiency, error rates.
            *   Subjective measures: Perceived cognitive load (NASA-TLX), perceived communication clarity, usefulness, trust in the agent, overall satisfaction (using Likert scales and potentially open-ended questions).

*   **Evaluation Metrics:**
    *   **Task Performance:** Success Rate, Task Completion Time, Score (task-dependent).
    *   **Communication Efficiency:**
        *   Estimated $I(X; Z)$ (via VIB KL term): Measures communication complexity/rate.
        *   Estimated $I(Z; Y)$ (via VIB decoder term): Measures task-relevant informativeness.
        *   Message Length / Rate: Average number of bits or tokens transmitted per unit time or decision step.
        *   Lexicon Size (for discrete signals): Number of unique signals used.
    *   **Human Factors (from user studies):**
        *   NASA-TLX Score: Standardized measure of perceived workload.
        *   Subjective Ratings: Clarity, Usefulness, Predictability, Trust (e.g., 1-7 Likert scales).
        *   Qualitative Feedback: User comments on interaction experience.
        *   Human Task Efficiency: Time taken by the human for sub-tasks, error rates.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  A novel and principled IB-RL framework specifically designed for optimizing human-AI communication in cooperative tasks.
    2.  AI agents trained using this framework that exhibit communication strategies demonstrably balancing task informativeness and cognitive efficiency for the human partner.
    3.  Empirical results from simulations and human subject studies quantifying the benefits of the IB-based approach compared to baseline methods in terms of joint performance, communication efficiency, and human experience (e.g., reduced cognitive load, increased clarity).
    4.  Insights into the nature of efficient human-AI communication, including how the optimal trade-off ($\beta$) varies with task complexity and human needs, and the characteristics of emergent communication protocols.
    5.  Validated methodology for applying and evaluating VIB within interactive RL agents for communication optimisation.
    6.  Contributions to the InfoCog community through publications and potential open-source code release, fostering interdisciplinary work.

*   **Impact:**
    *   **Scientific Impact:** This research will contribute to the fundamental understanding of communication in mixed human-AI systems by providing a formal optimisation framework based on information theory. It bridges machine learning (RL, deep generative models, VIB), cognitive science (cognitive load, human communication principles), and information theory, directly addressing the goals of the InfoCog workshop. It will also provide insights into approximating and optimizing information-theoretic quantities in complex learning systems.
    *   **Practical Impact:** The development of AI agents that communicate more effectively and efficiently with humans can significantly enhance human-AI collaboration across various applications. This could lead to:
        *   *Improved Performance & Safety:* More successful and faster completion of joint tasks, reduced misunderstandings in safety-critical situations (e.g., autonomous vehicle informing driver, robot assisting surgeon).
        *   *Enhanced User Experience:* Reduced cognitive burden on human partners, leading to less frustration and fatigue, and increased trust and adoption of AI systems.
        *   *More Natural Interaction:* AI communication that feels more intuitive and less overwhelming, potentially adapting to individual user needs by adjusting the $\beta$ parameter.
    *   **Alignment with InfoCog:** The proposed work directly addresses multiple InfoCog topics: novel information-theoretic approaches to cognitive functions (communication as coordination), validation methods (human studies), computation/estimation of IT quantities (VIB in RL), and application to training human-aligned agents. By focusing on the human aspect of the communication loop, it strongly promotes the development of AI systems designed for effective interaction and cooperation with people.