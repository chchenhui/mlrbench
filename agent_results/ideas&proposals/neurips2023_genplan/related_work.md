1. **Title**: NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains (arXiv:2503.00870)
   - **Authors**: Wonje Choi, Jinwoo Park, Sanghyun Ahn, Daehee Lee, Honguk Woo
   - **Summary**: This paper introduces NeSyC, a neuro-symbolic framework that emulates the hypothetico-deductive model to generalize actionable knowledge for embodied agents. By integrating Large Language Models (LLMs) and symbolic tools, NeSyC continually formulates and validates knowledge from limited experiences. The framework employs a contrastive generality improvement scheme and a memory-based monitoring scheme to enhance the agent's ability to solve complex tasks across diverse open-domain environments. Experiments demonstrate its effectiveness in various benchmarks, including ALFWorld, VirtualHome, Minecraft, RLBench, and real-world robotic scenarios.
   - **Year**: 2025

2. **Title**: Hierarchical Neuro-Symbolic Decision Transformer (arXiv:2503.07148)
   - **Authors**: Ali Baheri, Cecilia O. Alm
   - **Summary**: This work presents a hierarchical neuro-symbolic control framework that combines classical symbolic planning with transformer-based policies to address complex, long-horizon decision-making tasks. At the high level, a symbolic planner constructs interpretable sequences of operators based on logical propositions, ensuring adherence to global constraints and goals. At the low level, each symbolic operator is translated into a sub-goal token that conditions a decision transformer to generate fine-grained action sequences in uncertain, high-dimensional environments. Theoretical analysis and empirical evaluations in grid-world tasks demonstrate the framework's superiority over purely end-to-end neural approaches in terms of success rates and policy efficiency.
   - **Year**: 2025

3. **Title**: VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning (arXiv:2410.23156)
   - **Authors**: Yichao Liang, Nishanth Kumar, Hao Tang, Adrian Weller, Joshua B. Tenenbaum, Tom Silver, João F. Henriques, Kevin Ellis
   - **Summary**: The authors introduce VisualPredicator, a framework that learns abstract world models using neuro-symbolic predicates to enhance robot planning. By combining symbolic and neural knowledge representations, the system forms task-specific abstractions that expose essential elements while abstracting away sensorimotor complexities. An online algorithm is proposed for inventing such predicates and learning abstract world models. Comparative studies across five simulated robotic domains show that VisualPredicator offers better sample complexity, stronger out-of-distribution generalization, and improved interpretability compared to existing methods.
   - **Year**: 2024

4. **Title**: NeSIG: A Neuro-Symbolic Method for Learning to Generate Planning Problems (arXiv:2301.10280)
   - **Authors**: Carlos Núñez-Molina, Pablo Mesejo, Juan Fernández-Olivares
   - **Summary**: NeSIG is a domain-independent method for automatically generating planning problems that are valid, diverse, and difficult to solve. The authors formulate problem generation as a Markov Decision Process and train generative policies using Deep Reinforcement Learning. Experiments on classical domains demonstrate that NeSIG can generate problems of significantly greater difficulty than domain-specific generators while reducing human effort. Additionally, it generalizes to larger problems than those seen during training.
   - **Year**: 2023

**Key Challenges:**

1. **Sample Efficiency in Meta-Learning**: Developing meta-learning algorithms that are sample-efficient remains a significant challenge, as training on diverse environments requires substantial data to generalize effectively.

2. **Alignment of Symbolic and Neural Components**: Ensuring that symbolic planners and neural sub-policies are well-aligned is complex, as discrepancies can lead to suboptimal performance or failure in task execution.

3. **Generalization to Unseen Domains**: Achieving robust cross-domain generalization is difficult due to the variability in task structures and environmental dynamics, which can hinder the transferability of learned policies.

4. **Formal Verification of Neural Policies**: Integrating formal verification methods with neural policies to ensure constraint satisfaction and safety during deployment poses technical challenges, especially in dynamic and uncertain environments.

5. **Computational Complexity of Bi-Level Optimization**: Implementing bi-level optimization to align symbolic abstractions with sub-policy capabilities can be computationally intensive, potentially limiting scalability and real-time applicability. 