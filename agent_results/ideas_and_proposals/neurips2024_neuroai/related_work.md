1. **Title**: Active Predictive Coding: A Unified Neural Framework for Learning Hierarchical World Models for Perception and Planning (arXiv:2210.13461)
   - **Authors**: Rajesh P. N. Rao, Dimitrios C. Gklezakos, Vishwas Sathish
   - **Summary**: This paper introduces a framework called active predictive coding, which combines hypernetworks, self-supervised learning, and reinforcement learning to develop hierarchical world models. The approach addresses challenges in learning compositional representations and solving large-scale planning problems by integrating task-invariant state transition networks with task-dependent policy networks across multiple abstraction levels.
   - **Year**: 2022

2. **Title**: Meta-Representational Predictive Coding: Biomimetic Self-Supervised Learning (arXiv:2503.21796)
   - **Authors**: Alexander Ororbia, Karl Friston, Rajesh P. N. Rao
   - **Summary**: This work presents meta-representational predictive coding (MPC), a neurobiologically plausible framework for self-supervised learning based on the free energy principle. MPC learns to predict representations of sensory input across parallel streams, utilizing active inference to drive the learning of representations through sequences of decisions that sample informative portions of the sensorium.
   - **Year**: 2025

3. **Title**: Active Predictive Coding Networks: A Neural Solution to the Problem of Learning Reference Frames and Part-Whole Hierarchies (arXiv:2201.08813)
   - **Authors**: Dimitrios C. Gklezakos, Rajesh P. N. Rao
   - **Summary**: The authors propose Active Predictive Coding Networks (APCNs), which use hypernetworks and reinforcement learning to dynamically generate recurrent neural networks capable of predicting parts and their locations within intrinsic reference frames. APCNs address the challenge of learning part-whole hierarchies and compositional representations, demonstrating their effectiveness on datasets like MNIST, Fashion-MNIST, and Omniglot.
   - **Year**: 2022

4. **Title**: SPEQ: Stabilization Phases for Efficient Q-Learning in High Update-To-Data Ratio Reinforcement Learning (arXiv:2501.08669)
   - **Authors**: Carlo Romeo, Girolamo Macaluso, Alessandro Sestini, Andrew D. Bagdanov
   - **Summary**: This paper introduces SPEQ, a method that improves computational efficiency in reinforcement learning by alternating between online, low Update-To-Data (UTD) ratio training phases and offline stabilization phases. By fine-tuning Q-functions without collecting new environment interactions during stabilization phases, SPEQ achieves comparable results to state-of-the-art algorithms while reducing gradient updates and training time.
   - **Year**: 2025

**Key Challenges:**

1. **Sample Efficiency**: Many reinforcement learning algorithms require extensive interactions with the environment to learn effective policies, leading to high sample complexity and inefficiency.

2. **Learning Compositional Representations**: Developing models that can learn and generalize compositional representations, such as part-whole hierarchies, remains a significant challenge in creating flexible and interpretable systems.

3. **Balancing Exploration and Exploitation**: Effectively managing the trade-off between exploring new, informative states and exploiting known rewarding actions is crucial for efficient learning but remains difficult to achieve.

4. **Computational Efficiency**: High Update-To-Data (UTD) ratios can improve sample efficiency but often lead to increased computational costs, necessitating methods that balance both aspects effectively.

5. **Biologically Plausible Learning Mechanisms**: Integrating neurobiologically inspired principles, such as predictive coding and active inference, into artificial systems poses challenges in terms of model complexity and alignment with biological processes. 