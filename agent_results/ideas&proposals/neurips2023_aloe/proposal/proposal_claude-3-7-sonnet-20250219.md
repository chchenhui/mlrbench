# Self-Evolving Curriculum Generation with LLM-Guided Feedback for Open-Ended Reinforcement Learning

## Introduction

Reinforcement learning (RL) has demonstrated impressive capabilities in training agents to master specific tasks across various domains, from game playing to robotic control. However, traditional RL approaches face a fundamental limitation: once an agent masters a predefined task, the learning process typically stagnates. This limitation contradicts the nature of intelligence in biological systems, which continuously adapt to novel challenges and develop increasingly sophisticated capabilities throughout their lifetimes.

Open-ended learning (OEL) systems aim to address this limitation by creating learning environments that generate an endless stream of challenges, pushing agents to continuously improve their capabilities. The concept of open-endedness draws inspiration from biological evolution, where organisms continually adapt to changing environments and competitive pressures. In artificial learning systems, open-endedness holds the promise of producing agents with increasingly general capabilities, including the ability to handle novel scenarios not explicitly considered during system design.

Current approaches to curriculum learning in RL typically rely on hand-crafted task sequences or simplistic difficulty progression mechanisms. These methods require significant human expertise, are labor-intensive, and often fail to identify the most effective learning trajectories for agents. Moreover, they typically lack the ability to generate truly novel challenges that could drive the emergence of unexpected capabilities.

The integration of large language models (LLMs) with reinforcement learning presents a promising direction for automating and enhancing curriculum design. Recent work such as CurricuLLM (Ryu et al., 2024) and ExploRLLM (Ma et al., 2024) has demonstrated the potential of LLMs to generate task specifications and guide exploration in RL. However, these approaches do not fully leverage the potential of LLMs to create truly open-ended learning systems where the curriculum evolves based on the agent's capabilities and limitations.

This research proposes a novel framework called **Self-Evolving Curriculum with LLM-Guided Feedback (SELF)** that leverages LLMs as meta-controllers to generate adaptive curricula for open-ended reinforcement learning. SELF creates a closed feedback loop where an LLM analyzes the agent's performance and failure modes to generate new task specifications that target specific skill gaps. These tasks are then instantiated in a simulator, and the agent's performance is evaluated to inform the next iteration of task generation. By incorporating quality-diversity mechanisms, the framework ensures that the generated tasks remain diverse and challenging, preventing curriculum collapse and promoting the emergence of novel capabilities.

The primary research objectives of this study are:
1. To develop a framework that leverages LLMs to automatically generate adaptive curricula for RL agents based on their current capabilities and limitations
2. To design mechanisms that ensure the diversity and increasing complexity of generated tasks while preventing curriculum collapse
3. To evaluate the effectiveness of the proposed approach in facilitating continuous learning and improving generalization to unseen tasks
4. To investigate the transferability of skills learned through the self-evolving curriculum to real-world scenarios

The significance of this research lies in its potential to advance open-ended learning systems that can sustain continuous improvement without human intervention. By automating curriculum design through LLM-guided feedback, the proposed approach could significantly reduce the human effort required to train capable agents while potentially discovering novel learning trajectories that human designers might not consider. Furthermore, by continuously presenting agents with diverse and increasingly challenging tasks, the approach may improve out-of-distribution generalization and robustness, addressing a key limitation of current RL systems.

## Methodology

The Self-Evolving Curriculum with LLM-Guided Feedback (SELF) framework consists of four main components: (1) a reinforcement learning agent, (2) an LLM-based meta-controller for task generation, (3) a task instantiation module, and (4) a quality-diversity filter. These components interact in a closed feedback loop to create an adaptive curriculum that evolves based on the agent's capabilities and limitations.

### 3.1 Reinforcement Learning Agent

The RL agent in our framework can be any policy-based learning algorithm capable of learning from experience and adapting to new tasks. For our experiments, we will implement both a Proximal Policy Optimization (PPO) agent and a Soft Actor-Critic (SAC) agent to evaluate the effectiveness of our approach across different RL algorithms.

The agent interacts with the environment following the standard Markov Decision Process (MDP) formulation. At each time step $t$, the agent observes the state $s_t$, selects an action $a_t$ according to its policy $\pi(a_t|s_t)$, receives a reward $r_t$, and transitions to a new state $s_{t+1}$. The agent's objective is to maximize the expected cumulative reward $\mathbb{E}[\sum_{t=0}^{T} \gamma^t r_t]$, where $\gamma \in [0, 1]$ is the discount factor and $T$ is the episode length.

For each task in the curriculum, the agent will maintain a separate policy or utilize a task-conditioned policy depending on the complexity of the tasks. We will explore both approaches and analyze their impact on learning efficiency and transfer.

### 3.2 LLM-based Meta-Controller

The meta-controller is responsible for analyzing the agent's performance and generating new task specifications. We will use a state-of-the-art LLM (e.g., GPT-4 or Claude) as the meta-controller, fine-tuned on a dataset of RL task descriptions and performance analyses to improve its ability to generate meaningful and appropriate tasks.

The meta-controller takes as input:
1. A description of the agent's current capabilities (tasks it can solve)
2. Detailed analysis of failure modes (tasks it cannot solve and specific failure patterns)
3. A history of previously generated tasks and their impact on the agent's learning
4. The environment's action and observation space specifications

The output of the meta-controller is a set of new task specifications in a structured format that can be parsed and instantiated in the simulation environment. Each task specification includes:
1. A natural language description of the task
2. Required environment configurations (objects, initial conditions, etc.)
3. Success criteria and reward function definition
4. Expected skill improvements for the agent

The meta-controller is prompted to generate tasks that specifically target the agent's current limitations while ensuring a gradual progression in difficulty. The prompting strategy follows this template:

```
Given the agent's current capabilities:
[List of mastered tasks and skills]

And its failure modes:
[Detailed analysis of unsuccessful tasks and specific failure patterns]

Generate N new tasks that:
1. Target the identified skill gaps
2. Provide a gradual increase in difficulty
3. Require combining previously learned skills in novel ways
4. Are diverse in terms of required skills and solution strategies

For each task, provide:
1. A clear description
2. Environment configuration parameters
3. Success criteria and reward function
4. The primary skills this task is designed to improve
```

To ensure that the meta-controller generates increasingly challenging tasks, we implement a difficulty scaling mechanism based on the agent's performance. Let $p_i$ be the agent's performance on task $i$, measured as the success rate over a set of evaluation episodes. The difficulty level $d$ for the next set of tasks is adjusted according to:

$$d_{t+1} = d_t \cdot (1 + \alpha \cdot (p_{avg} - p_{target}))$$

where $p_{avg}$ is the average performance across all tasks in the current batch, $p_{target}$ is the target performance level (typically around 0.7), and $\alpha$ is a scaling factor that controls the rate of difficulty adjustment.

### 3.3 Task Instantiation Module

The task instantiation module translates the natural language task specifications generated by the meta-controller into executable environments. This module consists of:

1. **Parser**: Extracts structured information from the LLM's output
2. **Environment Generator**: Creates or configures simulation environments based on the parsed specifications
3. **Reward Function Constructor**: Implements the specified reward function

For environment generation, we will use modular simulation frameworks such as MuJoCo for robotic control tasks and Procgen for procedurally generated game environments. The environment generator will expose a set of parameterizable elements that can be configured based on the task specifications, including:

- Object types, positions, and properties
- Agent initial conditions
- Environmental dynamics (e.g., friction, gravity)
- Task-specific constraints and rules

The reward function constructor translates the specified success criteria into a mathematical reward function. For complex specifications, the LLM itself will be used to generate code for the reward function. The general form of the reward function will be:

$$r(s_t, a_t, s_{t+1}) = r_{task}(s_t, a_t, s_{t+1}) + r_{shaping}(s_t, s_{t+1})$$

where $r_{task}$ is the task-specific reward component and $r_{shaping}$ is an optional shaping component to provide more informative feedback to the agent.

### 3.4 Quality-Diversity Filter

To prevent curriculum collapse and ensure that the agent is exposed to a diverse and balanced set of tasks, we implement a quality-diversity filter inspired by quality-diversity algorithms in evolutionary computation. This filter evaluates generated tasks based on two main criteria:

1. **Quality**: The expected learning impact of the task, measured by its alignment with the agent's current skill gaps
2. **Diversity**: The dissimilarity of the task compared to previously selected tasks

For each generated task $i$, we compute a quality score $q_i$ and a diversity score $div_i$. The quality score is derived from the estimated learning potential:

$$q_i = \beta \cdot (1 - p_{est,i}) \cdot c_i$$

where $p_{est,i}$ is the estimated success probability of the agent on task $i$ (provided by a prediction model trained on previous task performances), $c_i$ is the complexity of the task (estimated based on the number and types of skills required), and $\beta$ is a scaling parameter.

The diversity score measures the task's dissimilarity to the set of previously selected tasks $S$:

$$div_i = \min_{j \in S} d(i, j)$$

where $d(i, j)$ is a distance function between tasks $i$ and $j$. This distance function combines semantic distance (based on LLM embeddings of task descriptions) and behavioral distance (based on differences in agent behavior when attempting the tasks).

Tasks are selected to maximize a weighted combination of quality and diversity:

$$score_i = \lambda \cdot q_i + (1 - \lambda) \cdot div_i$$

where $\lambda \in [0, 1]$ is a parameter that controls the trade-off between quality and diversity.

### 3.5 Complete Algorithm

The complete SELF algorithm operates as follows:

1. **Initialization**:
   - Initialize the RL agent with a random policy
   - Define a set of basic initial tasks to establish baseline capabilities
   - Set initial difficulty level $d_0$

2. **Main Loop**:
   For each iteration $t = 0, 1, 2, \ldots$:
   
   a. **Agent Learning Phase**:
      - Train the agent on the current set of tasks until a performance threshold is reached or a maximum number of episodes is completed
      - Record performance metrics and trajectories for each task
   
   b. **Performance Analysis**:
      - Identify tasks the agent can solve (success rate above threshold)
      - Identify tasks the agent cannot solve and analyze failure patterns
      - Compute performance statistics across all tasks
   
   c. **Task Generation**:
      - Provide the meta-controller with performance analysis
      - Generate a set of candidate tasks with difficulty level $d_t$
      - Update difficulty level $d_{t+1}$ based on current performance
   
   d. **Task Filtering and Selection**:
      - Compute quality and diversity scores for each candidate task
      - Select a subset of tasks that maximize the combined score
      - Add selected tasks to the curriculum
   
   e. **Curriculum Update**:
      - Remove mastered tasks with success rate above a high threshold
      - Retain challenging tasks with intermediate success rates
      - Add the newly selected tasks

3. **Evaluation**:
   - Periodically evaluate the agent on a set of held-out test tasks
   - Measure transfer performance on real-world tasks (if applicable)
   - Calculate the ODD score (Out-of-Distribution Difficulty) to assess generalization

The formal algorithm is presented below:

```
Algorithm: Self-Evolving Curriculum with LLM-Guided Feedback (SELF)

Input: Initial task set T_0, RL agent A, meta-controller M, quality-diversity filter F, 
       performance thresholds θ_success and θ_mastery, maximum iterations max_iter

Initialize:
    - Agent policy π_0
    - Task set T = T_0
    - Difficulty level d = d_0
    - Task history H = {}

For t = 0 to max_iter:
    // Training phase
    For each task τ in T:
        Train agent A on task τ for n_episodes
        Record performance p_τ and trajectories Z_τ
    
    // Performance analysis
    Successful_tasks = {τ ∈ T | p_τ > θ_success}
    Failed_tasks = {τ ∈ T | p_τ ≤ θ_success}
    Mastered_tasks = {τ ∈ T | p_τ > θ_mastery}
    Analyze failure patterns in Failed_tasks and Z_τ
    
    // Task generation
    Update H with current task performance data
    Candidate_tasks = M.generate_tasks(Successful_tasks, Failed_tasks, H, d)
    d = update_difficulty(d, p_avg, p_target)
    
    // Task filtering
    Selected_tasks = F.filter(Candidate_tasks, T, H)
    
    // Curriculum update
    T = (T - Mastered_tasks) ∪ Selected_tasks
    
    // Evaluation
    If t % eval_freq == 0:
        Evaluate agent on held-out test tasks
        Calculate ODD score
        If applicable, evaluate transfer to real-world tasks

Return: Final agent policy π_final
```

### 3.6 Experimental Design

To evaluate the effectiveness of the SELF framework, we will conduct experiments in three different domains:

1. **Robotic manipulation tasks** using MuJoCo
2. **Procedurally generated 2D games** using Procgen
3. **Navigation and exploration** in 3D environments using Unity

For each domain, we will compare the SELF framework against the following baselines:

1. **Fixed curriculum**: A hand-designed sequence of tasks with increasing difficulty
2. **Random task selection**: Randomly sampling tasks from a large pool
3. **PLR (Prioritized Level Replay)**: A state-of-the-art automated curriculum learning method
4. **UED (Unsupervised Environment Design)**: An adversarial approach to generating challenging environments
5. **CurricuLLM**: A recent approach that uses LLMs for curriculum design but without the quality-diversity filter and closed-loop feedback

We will evaluate the approaches using the following metrics:

1. **Learning efficiency**: The number of environment interactions required to reach a target performance level
2. **Generalization performance**: Success rate on unseen tasks within the same domain
3. **ODD score (Out-of-Distribution Difficulty)**: A measure of performance on increasingly out-of-distribution tasks
4. **Skill emergence**: The number and complexity of emergent skills not explicitly encoded in the initial tasks
5. **Sim2real transfer**: For robotic tasks, the performance gap between simulation and real-world execution

Each experiment will be repeated with 5 different random seeds to ensure statistical significance. For the LLM-based components, we will conduct ablation studies to analyze the impact of different prompt designs and LLM architectures on task generation quality.

## Expected Outcomes & Impact

This research is expected to yield several significant outcomes with broad implications for the field of reinforcement learning and artificial intelligence:

### 4.1 Technical Contributions

1. **Novel Framework for Open-Ended Learning**: The SELF framework represents a new approach to creating truly open-ended learning systems that can generate an endless stream of increasingly challenging tasks tailored to an agent's capabilities. This addresses a fundamental limitation of current RL systems that typically stagnate once predefined tasks are mastered.

2. **Effective Integration of LLMs with RL**: By leveraging LLMs as meta-controllers for curriculum design, this research will establish new methods for harnessing the knowledge and reasoning capabilities of foundation models to guide reinforcement learning. The proposed approach provides a practical mechanism for translating natural language task specifications into executable environments and reward functions.

3. **Quality-Diversity Mechanisms for Curriculum Design**: The incorporation of quality-diversity principles into curriculum generation will advance our understanding of how to prevent curriculum collapse and ensure continuous learning progress. The proposed filtering mechanisms could be adapted to other curriculum learning contexts beyond the specific framework presented here.

4. **Metrics for Open-Ended Learning**: The development and validation of metrics like the ODD score will contribute to our ability to quantify the success of open-ended learning systems, addressing a significant challenge in the field.

### 4.2 Practical Applications

1. **Robotics and Embodied AI**: The ability to automatically generate diverse and increasingly complex tasks could significantly accelerate the development of robotic systems capable of operating in unstructured environments. By exposing robots to a broader range of scenarios during training, the proposed approach could improve generalization to real-world conditions.

2. **Sim2Real Transfer**: The continuous adaptation of the curriculum based on agent performance could help bridge the reality gap in robotics by gradually introducing variations that reflect real-world complexities. This could reduce the need for manual domain randomization techniques currently used in sim2real transfer.

3. **Intelligent Tutoring Systems**: The principles developed in this research could inform the design of adaptive educational technologies that automatically generate personalized learning content based on a student's current understanding and specific knowledge gaps.

4. **Game AI and Procedural Content Generation**: The framework could enhance procedural content generation in games by creating challenges that adapt to a player's skill level and play style, potentially leading to more engaging and personalized gaming experiences.

### 4.3 Broader Impact

1. **Advancing AGI Research**: By enabling continuous learning and adaptation to novel challenges, this research contributes to the development of more generally capable AI systems. The ability to self-generate curricula that drive the emergence of increasingly sophisticated capabilities aligns with the long-term goals of artificial general intelligence research.

2. **Reducing Human Intervention**: Automating curriculum design reduces the need for human experts to handcraft learning scenarios, potentially making advanced AI more accessible and reducing development costs. This could democratize access to sophisticated AI systems across various domains.

3. **Understanding Emergent Complexity**: The proposed framework provides a controlled environment to study the emergence of complex behaviors from simple learning rules. This could offer insights into how intelligence emerges in biological systems and inform theories of cognitive development.

4. **Ethical Considerations**: As with any advanced AI research, there are ethical considerations regarding the potential for unintended emergent behaviors. By incorporating explicit quality-diversity mechanisms and performance monitoring, the proposed framework includes safeguards to guide the learning process in beneficial directions.

In summary, the SELF framework represents a significant step toward creating truly open-ended learning systems capable of continuous adaptation and improvement. By automatically generating curricula that evolve with the agent's capabilities, this approach has the potential to overcome fundamental limitations of current reinforcement learning methods and contribute to the development of more generally capable AI systems.