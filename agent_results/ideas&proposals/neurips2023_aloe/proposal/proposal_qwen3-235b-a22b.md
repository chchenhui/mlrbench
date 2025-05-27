# Self-Generating Adaptive Curricula for Open-Ended Reinforcement Learning

## Introduction

### Background  
Traditional reinforcement learning (RL) frameworks operate in static environments with predefined tasks, leading to rapid stagnation once the agent achieves mastery. In contrast, biological agents (e.g., humans) develop robust, general intelligence through lifelong interactions with dynamic, open-ended environments. These environments generate novel challenges that co-evolve with the agent's capabilities, fostering continual learning and out-of-distribution generalization. The challenge of emulating this behavior in artificial systems lies at the core of open-ended learning (OEL). Recent advances in large language models (LLMs) as task designers and the proliferation of scalable simulators have reignited interest in automating curriculum creation. However, existing approaches often rely on manual task design, fixed objectives, or simplified assumptions about agent capabilities, limiting their ability to sustain long-term learning.

### Research Objectives  
This proposal tackles the following problem:  
**How can we design a self-sustaining RL framework where a meta-controller dynamically generates adaptive curricula based on the agent's skill gaps, ensuring continual growth in complex environments?**  

Three technical objectives define this work:  
1. **Curriculum Synthesis via LLMs**: Develop a meta-controller using LLMs to generate task specifications conditioned on agent performance, failure modes, and environmental complexity.  
2. **Quality-Diversity Integration**: Filter generated tasks using quality-diversity (QD) algorithms to prioritize diverse, high-impact challenges that prevent capability collapse.  
3. **ODD-Score Quantification**: Introduce a formal metric (Out-of-Distribution Difficulty, ODD) to assess task complexity and agent adaptability, linking simulated curricula to real-world sim2real transfer.  

### Significance  
This research directly addresses critical challenges in OEL: stagnation after mastery, the labor inefficiency of manual curricula, and poor sim2real translation. By creating a closed-loop system where agents automatically generate and solve evolving challenges, the framework enables:  
- **Continual Learning**: Sustained skill acquisition without task saturation.  
- **Generalization**: Transfer to novel tasks and real-world settings via ODD-aligned curricula.  
- **Scalability**: Low-cost, automated curriculum design for complex RL domains.  

The proposed method bridges the gap between foundational LLM capabilities (e.g., CurricuLLM [Ryu et al., 2024]) and open-ended exploration (e.g., UED [Jiang, 2023]), while addressing limitations in generalization and computational efficiency identified in recent benchmarks.

---

## Methodology

### System Architecture  
The framework consists of **four interconnected components** (Figure 1):  
1. **Agent**: A deep RL policy (e.g., a PPO-based actor-critic network) trained on environmental tasks.  
2. **Meta-Controller (LLM)**: A large language model (e.g., GPT-4 or Mistral-Large) that generates task specifications based on agent data.  
3. **Simulator**: A 3D environment (e.g., MuJoCo, PyBullet, or Unity) where tasks are instantiated.  
4. **ODD-Metric Module**: A quantifier of task difficulty and agent robustness.  

---

### Research Design  

#### **Data Collection and Preprocessing**  
The agent accumulates trajectories $\tau = \{(s_t, a_t, r_t)\}_{t=1}^T$ across training episodes. Trajectories are analyzed to:  
- **Identify Failures**: Use clustering (e.g., k-means) on states $s_t$ where rewards $r_t$ fall below a threshold:  
  $$
  \mathcal{F} = \{s_t \in \mathcal{S} \mid r_t < \eta\},
  $$  
  where $\eta$ is a failure threshold.  
- **Extract Behavior Descriptors**: Compute low-dimensional descriptors $\mathbf{b} \in \mathbb{R}^d$ (e.g., action histograms, motion patterns) for diversity tracking.  

#### **Task Generation via LLM Meta-Controller**  
The meta-controller iteratively generates tasks $\mathcal{T} = \{t_i\}$ through four stages:  

1. **Prompt Engineering**:  
   The LLM receives a structured prompt summarizing:  
   - Current policy performance (success rate, reward statistics).  
   - Failure clusters $\mathcal{F}$.  
   - Archive of previously solved tasks.  
   Example prompt (contextualized for robotic manipulation):  
   > *“Based on trajectories where the robot fails to grasp objects (mean success rate 32%), generate 5 new tasks that either: (1) Modify object shapes in failure states, (2) Introduce occlusions, or (3) Combine grasping with navigation. Prioritize tasks requiring compound reasoning and test edge cases in motor control.”*

2. **Task Specification**:  
   The LLM outputs natural-language tasks $t_i$ (e.g., *“Grasp a cylindrical object while resisting lateral wind forces”*). These are parsed by a domain-specific language (DSL) compiler into executable environments.  

3. **Quality-Diversity Filtering**:  
   A QD algorithm (e.g., CVT-MAP-Elites [Vassiliades et al., 2018]) filters tasks based on:  
   - **Diversity**: Measured by behavioral distance in descriptor space:  
     $$
     D(t) = \frac{1}{|\mathcal{T}_{\text{archive}}|} \sum_{t' \in \mathcal{T}_{\text{archive}}} \|\mathbf{b}_t - \mathbf{b}_{t'}\|_2.
     $$  
   - **Impact**: Predicted difficulty $\hat{d}$ (via a logistic regression model of past agent performance).  
   Tasks in the top $\kappa$-th percentile of $D(t) \times (1 - \text{performance})$ are selected to ensure novelty and difficulty.  

4. **ODD-Score Assignment**:  
   Define ODD-score for task $t$ as:  
   $$
   \text{ODD}(t) = \frac{1}{1 + \exp(\gamma(P_{train}(t) - P_{ood}(t)))},
   $$  
   where $P_{train}(t)$ is the agent’s in-distribution performance on $t$, $P_{ood}(t)$ evaluates zero-shot transfer to perturbed variants of $t$ (e.g., altered friction coefficients, randomized object sizes), and $\gamma$ controls sharpness. High $\text{ODD} \gg 0.5$ indicates tasks demanding generalization.  

#### **Training Pipeline**  
The process repeats in epochs:  
1. **Episode Rollouts**: Agent trains on $\mathcal{T}_{\text{current}}$.  
2. **Failure Analysis**: Update failure clusters $\mathcal{F}$.  
3. **LLM Interaction**: Generate $\mathcal{T}_{\text{new}}$ with diversity-impact filtering.  
4. **Archive Update**: Remove obsoleted tasks (e.g., those with $P_{train} > 90\%$ for 5 epochs).  

---

### Experimental Design  

#### **Environments**  
1. **MuJoCo**: Complex control tasks (e.g., HumanoidLocomotion-v3).  
2. **PyBullet**: Robotics manipulation (e.g., KUKA-PickAndPlace).  
3. **Minecraft-Like Simulation**: Open-ended exploration with procedurally generated worlds.  

#### **Baselines**  
- **Fixed Curriculum**: Manually designed task hierarchy (e.g., [Ryu et al., 2024]).  
- **UED-Based OEL**: Environment generator trained to adversarially challenge agent [Jiang, 2023].  
- **Random Task Sampling**: Uniformly random task selection without QD filtering.  

#### **Evaluation Metrics**  
1. **Learning Sustainability**:  
   - Task acquisition rate $R(t) = \frac{\text{Tasks solved}}{\text{Total tasks}}$ over 10,000 episodes.  
   - Plateau detection: If $dR/dt < \epsilon$ for 200 episodes, mark stagnation.  
2. **Generalization**:  
   - Zero-shot performance on held-out tasks $\mathcal{T}_{\text{test}}$ unseen during training.  
   - Out-of-distribution accuracy (OODA) computed via $P_{ood}$ (see ODD equation).  
3. **Sim2Real Transfer**:  
   - Quantify performance drop $\Delta P = P_{sim} - P_{real}$ on a NAO humanoid robot [Aldebaran Robotics].  
4. **Efficiency**:  
   - Compute cost (e.g., GPU/LLM API hours) per 1k episode steps.  
   - Ablation studies isolating QD filtering, LLM prompting strategy.  

---

## Expected Outcomes & Impact  

### Anticipated Results  
1. **Sustained Learning Dynamics**:  
   Our framework will delay stagnation by 40–60% compared to fixed curricula in MuJoCo tasks (e.g., from 2,000 to 8,000 episodes).  
2. **Generalization and ODD Correlation**:  
   Tasks with $\text{ODD} > 0.7$ will show 20% higher zero-shot success than baseline tasks.  
3. **Sim2Real Effectiveness**:  
   Humanoid locomotion policies trained in OEL will achieve 80% of sim performance in real-world experiments, doubling the transfer rate of UED baselines.  
4. **Diversity Quantification**:  
   QD filtering will produce task distributions with 30% higher behavioral diversity (measured by pairwise descriptor distance) than random sampling.  

### Theoretical and Practical Impact  
- **OEL Theory**: Establish ODD-score as a formal measure of open-ended complexity, enabling comparative analysis across frameworks.  
- **Automated Curriculum Design**: Reduce human intervention by 70% over methods like CurricuLLM while expanding task scope beyond manual specification.  
- **Domain Applications**: Unlock scalable OEL in robotics (e.g., agile locomotion) and creative industries (e.g., game design).  

By merging LLM-driven task generation with RL scalability, this work takes a step toward AI systems that learn *as humans do*—through relentless, self-driven adaptation to an ever-changing world.