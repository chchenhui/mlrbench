# Equivariant World Models for Sample-Efficient Robot Learning with Geometric Priors

## 1. Introduction

Robotic systems operate in physical environments governed by fundamental geometric principles and symmetries. From factory automation to home assistance, robots must recognize and adapt to spatial transformations—objects may be rotated differently, tasks may require operation from various angles, and navigation may traverse similar but spatially transformed terrains. Despite these inherent symmetries, most learning algorithms for robotic control fail to explicitly encode these geometric priors, resulting in data-hungry models that struggle to generalize across symmetric variations of tasks.

Traditional approaches to robot learning, particularly in reinforcement learning (RL), treat each configuration of the environment as a unique state, even when these configurations are merely symmetric transformations of previously encountered scenarios. This leads to redundant learning and poor sample efficiency, as the robot must experience many variations of essentially the same physical situation. The challenge is particularly acute in world models—neural network representations that predict environment dynamics and rewards—where spatial relationships are critical for accurate forecasting.

Recent advances in geometric deep learning have demonstrated the power of symmetry-aware neural architectures in computer vision, molecular modeling, and other domains. By designing neural networks with equivariance to specific transformation groups (e.g., rotations, translations), these models can generalize across symmetric transformations with minimal additional data. However, the application of such principles to world modeling for robotic control remains relatively unexplored, presenting a significant opportunity to improve sample efficiency and generalization in robot learning.

This research proposes a framework for equivariant world models that explicitly respect the symmetries present in robotic tasks. By incorporating group-equivariant neural network architectures, we aim to develop models that can predict next-state transitions and rewards while preserving equivariance to transformations such as rotations, translations, and their compositions. Our approach leverages recent advances in geometric deep learning and applies them to the specific challenges of world modeling for robotic control.

The significance of this research lies in its potential to dramatically reduce the sample complexity of robot learning while improving generalization to unseen but geometrically related scenarios. By encoding geometric priors directly into the model architecture, we enable robots to learn more efficiently from limited data and transfer knowledge across symmetric variations of tasks. This has implications not just for research systems but for practical deployments in dynamic, unstructured environments where adaptability is crucial.

Furthermore, this work contributes to the nascent intersection of geometric deep learning and embodied artificial intelligence, aligning with broader efforts to identify computational principles shared between biological and artificial neural systems. Just as neural circuits in the brain mirror the geometric structure of the systems they represent, our equivariant world models will encode the geometric structure of robotic tasks, potentially illuminating common principles for efficient information processing.

## 2. Methodology

### 2.1 Problem Formulation

We frame our approach within the Markov Decision Process (MDP) formulation, where a robot interacts with an environment characterized by states $s \in \mathcal{S}$, actions $a \in \mathcal{A}$, a transition function $P(s'|s,a)$, and a reward function $R(s,a)$. The goal is to learn a policy $\pi(a|s)$ that maximizes expected cumulative rewards.

A world model in this context consists of two components:
1. A dynamics model $f_\theta: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ that predicts the next state given the current state and action
2. A reward model $r_\phi: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ that predicts the immediate reward

The key innovation in our approach is designing these models to be equivariant with respect to specific transformation groups $G$ (e.g., SE(2) or SE(3)) that characterize the symmetries of the task. 

### 2.2 Equivariant World Model Architecture

#### 2.2.1 Transformation Groups

We focus on two primary transformation groups relevant to robotic tasks:

1. **SE(2)**: The special Euclidean group in 2D, representing planar rotations and translations, relevant for tabletop manipulation and 2D navigation.

2. **SE(3)**: The special Euclidean group in 3D, representing 3D rotations and translations, relevant for free-space manipulation and 3D navigation.

For a transformation $g \in G$ (where $G$ is either SE(2) or SE(3)), a function $f$ is G-equivariant if:

$$f(g \cdot x) = g \cdot f(x)$$

This property ensures that transforming the input and then applying the function yields the same result as applying the function and then transforming the output.

#### 2.2.2 State Representation

The state space $\mathcal{S}$ will consist of:

1. **Visual observations**: RGB-D images $I_t \in \mathbb{R}^{H \times W \times 4}$ or point clouds $P_t \in \mathbb{R}^{N \times 3}$
2. **Proprioceptive information**: Joint positions, velocities, and torques $q_t, \dot{q}_t, \tau_t$
3. **Task-specific information**: Goal specifications, object properties, etc.

We encode this heterogeneous state through a multi-stream architecture:

1. **Visual encoder** $E_v$: For image inputs, we use an equivariant convolutional neural network based on group convolutions. For point clouds, we utilize a SE(3)-equivariant point cloud encoder.

2. **Proprioceptive encoder** $E_p$: A standard MLP for encoding the robot's internal state.

3. **Task encoder** $E_t$: A context-dependent encoder for task specifications.

The combined state embedding is given by:

$$z_t = [E_v(I_t \text{ or } P_t), E_p(q_t, \dot{q}_t, \tau_t), E_t(\text{task})]$$

#### 2.2.3 Equivariant Dynamics Model

Our dynamics model consists of:

1. **Equivariant backbone**: A neural network $f_\theta$ built using group-equivariant layers that respect the symmetry group $G$.

2. **Action integration**: Actions are incorporated through a steerable action conditioning module.

The dynamics model predicts the next state embedding:

$$\hat{z}_{t+1} = f_\theta(z_t, a_t)$$

For SE(2)-equivariant models, we implement group convolutional layers using:

$$[f * \psi](x) = \sum_{y \in \mathbb{Z}^2} \sum_{g \in G} f(g^{-1}y) \psi(g^{-1}(x-y))$$

For SE(3)-equivariant models, we utilize tensor field networks or Euclidean neural networks that operate on 3D data while preserving equivariance.

#### 2.2.4 Equivariant Reward Model

The reward model $r_\phi$ predicts the immediate reward from the state embedding and action:

$$\hat{r}_t = r_\phi(z_t, a_t)$$

For the reward model to be fully equivariant, we ensure that:

$$r_\phi(g \cdot z_t, g \cdot a_t) = r_\phi(z_t, a_t)$$

This invariance property is appropriate for rewards that should not change under symmetric transformations of the task.

### 2.3 Training Methodology

#### 2.3.1 Data Collection

We collect training data using:

1. **Random exploration**: Initially generating diverse experiences through random actions.
2. **On-policy collection**: Gathering data by executing the current policy.
3. **Symmetry-aware data augmentation**: Applying transformations from group $G$ to existing trajectories to generate additional valid training examples.

The dataset $\mathcal{D}$ consists of tuples $(s_t, a_t, s_{t+1}, r_t)$ from these sources.

#### 2.3.2 World Model Training

The world model is trained to minimize:

$$\mathcal{L}_{WM}(\theta, \phi) = \mathcal{L}_{dyn}(\theta) + \lambda \mathcal{L}_{reward}(\phi)$$

where:

$$\mathcal{L}_{dyn}(\theta) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ \|f_\theta(z_t, a_t) - z_{t+1}\|^2 \right]$$

$$\mathcal{L}_{reward}(\phi) = \mathbb{E}_{(s_t, a_t, r_t) \sim \mathcal{D}} \left[ (r_\phi(z_t, a_t) - r_t)^2 \right]$$

Additionally, we enforce equivariance through an explicit equivariance loss:

$$\mathcal{L}_{equiv}(\theta) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}, g \sim G} \left[ \|f_\theta(g \cdot z_t, g \cdot a_t) - g \cdot f_\theta(z_t, a_t)\|^2 \right]$$

The final loss becomes:

$$\mathcal{L}_{total}(\theta, \phi) = \mathcal{L}_{WM}(\theta, \phi) + \gamma \mathcal{L}_{equiv}(\theta)$$

#### 2.3.3 Policy Learning with Equivariant World Models

We train a policy $\pi_\psi(a|s)$ using the world model through:

1. **Model-based RL**: Using the world model to simulate trajectories and optimize the policy.
2. **Model-predictive control**: Using the world model for short-horizon planning.

The policy training objective is:

$$\mathcal{L}_{policy}(\psi) = -\mathbb{E}_{\tau \sim \pi_\psi, f_\theta, r_\phi} \left[ \sum_{t=0}^T \gamma^t r_\phi(z_t, a_t) \right]$$

To ensure policy equivariance, we either:
1. Build an equivariant policy network, or
2. Train with symmetry-augmented data and add an equivariance regularization term.

### 2.4 Experimental Design

#### 2.4.1 Simulation Environments

We evaluate our approach in the following environments:

1. **Manipulation tasks**:
   - Peg insertion with varying orientations
   - Object rearrangement with rotational symmetry
   - Articulated object manipulation

2. **Navigation tasks**:
   - 2D navigation with environmental symmetries
   - 3D terrain traversal with variable starting positions

#### 2.4.2 Real-Robot Implementation

We deploy and evaluate our models on:

1. **Manipulation platform**: A 7-DOF robot arm with a parallel gripper
2. **Mobile robot**: A differential drive robot with LIDAR and camera

#### 2.4.3 Baselines and Comparisons

We compare our equivariant world model against:

1. **Standard world models**: Models without equivariance constraints (e.g., standard MLPs, CNNs)
2. **Data-augmented baselines**: Non-equivariant models trained with symmetry-augmented data
3. **Model-free methods**: Direct RL approaches like SAC, PPO
4. **Existing geometric approaches**: Prior work on geometric RL without world models

#### 2.4.4 Evaluation Metrics

We evaluate the performance using:

1. **Sample efficiency**: Number of environment interactions required to reach a performance threshold
2. **Generalization**: Performance on unseen but symmetrically related tasks
3. **Prediction accuracy**: World model accuracy across symmetry transforms
4. **Task success rate**: Percentage of successful task completions
5. **Transfer performance**: Success rate on real robots after simulation training

### 2.5 Implementation Details

#### 2.5.1 Network Architectures

1. **Visual encoder**:
   - For images: 6-layer equivariant CNN with group convolutions
   - For point clouds: EGNN (Equivariant Graph Neural Network)

2. **Proprioceptive encoder**:
   - 3-layer MLP with [256, 128, 64] units

3. **Dynamics model**:
   - Steerable E(n)-equivariant network with 5 message-passing layers

4. **Reward model**:
   - Invariant network using pooling over equivariant features

#### 2.5.2 Training Parameters

- **Optimizer**: Adam with learning rate 3e-4
- **Batch size**: 256
- **Training steps**: 500,000
- **Data collection**: 100,000 initial random steps, then on-policy collection
- **Augmentation**: 10 random symmetry transformations per original sample

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

The proposed research is expected to yield several significant outcomes:

1. **Enhanced Sample Efficiency**: A primary expected outcome is a substantial reduction in the amount of training data required to achieve high-performance robotic control. By incorporating geometric priors through equivariant architectures, we anticipate at least a 2-5x reduction in data requirements compared to non-equivariant baselines.

2. **Improved Generalization**: The equivariant world models should demonstrate superior generalization to novel but geometrically related scenarios. We expect successful zero-shot transfer to unseen object orientations, positions, and scales without additional training.

3. **Better Transferability**: By encoding geometric invariances that apply equally in simulation and reality, we anticipate improved sim-to-real transfer, reducing the reality gap that typically hampers deployment of simulation-trained models.

4. **Accurate Long-Horizon Prediction**: The equivariant constraints should improve the stability and accuracy of world model predictions over longer time horizons, enabling more effective planning and policy optimization.

5. **New Architectural Patterns**: This research will yield novel neural network architectures specifically designed for robotic world modeling that respect geometric symmetries while handling the heterogeneous and multimodal data typical in robotics.

6. **Benchmark Results**: We will provide comprehensive benchmarks comparing equivariant and non-equivariant approaches across a range of robotic tasks, establishing a foundation for future research in this direction.

### 3.2 Broader Impact

The broader impact of this research extends across multiple domains:

1. **Robotics Applications**: More sample-efficient and generalizable learning will enable robots to be deployed in less structured environments like homes, retail spaces, and disaster zones, where adaptability to spatial variations is crucial.

2. **Manufacturing Automation**: Improved sample efficiency and generalization will make it economically viable to automate small-batch manufacturing processes that currently rely on human flexibility.

3. **Interdisciplinary Connections**: This work bridges computational neuroscience, geometric deep learning, and robotics, potentially illuminating common principles for efficient neural representation across biological and artificial systems.

4. **Reduced Computational Resources**: By requiring less data and training time, this approach reduces the computational burden and environmental impact of robot learning.

5. **Accelerated Research**: Better sample efficiency means faster iteration cycles for robotics researchers, potentially accelerating the pace of discovery in the field.

6. **Educational Impact**: The explicit incorporation of geometric principles into learning systems provides an intuitive framework for understanding and teaching robot learning, potentially broadening participation in robotics research.

7. **Safety and Reliability**: More robust generalization across spatial variations contributes to safer robotic systems that can handle unexpected environmental changes.

This research contributes directly to the workshop's focus on symmetry and geometry in neural representations, demonstrating how computational principles observed in biological neural systems can be implemented in artificial systems to improve performance. By encoding geometric structure into world models, we mirror the way neural circuits in the brain preserve the geometric structure of the systems they represent, potentially illuminating universal principles for efficient neural computation.