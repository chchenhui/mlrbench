1. **Title**: SHARPIE: A Modular Framework for Reinforcement Learning and Human-AI Interaction Experiments (arXiv:2501.19245)
   - **Authors**: Hüseyin Aydın, Kevin Godin-Dubois, Libio Goncalvez Braz, Floris den Hengst, Kim Baraka, Mustafa Mert Çelikok, Andreas Sauter, Shihan Wang, Frans A. Oliehoek
   - **Summary**: SHARPIE is a modular framework designed to facilitate experiments involving reinforcement learning (RL) agents and human participants. It offers a versatile wrapper for RL environments and algorithm libraries, a web interface for participants, logging utilities, and supports deployment on cloud platforms. The framework aims to standardize human-RL interaction studies, enabling research into interactive reward specification, learning from human feedback, and human-AI teaming.
   - **Year**: 2025

2. **Title**: Online Preference-based Reinforcement Learning with Self-augmented Feedback from Large Language Model (arXiv:2412.16878)
   - **Authors**: Songjun Tu, Jingbo Sun, Qichao Zhang, Xiangyuan Lan, Dongbin Zhao
   - **Summary**: This paper introduces RL-SaLLM-F, a technique that leverages large language models (LLMs) to generate self-augmented trajectories and provide preference labels for reward learning in online preference-based reinforcement learning (PbRL). By addressing the challenge of obtaining real-time human feedback, the method enhances the quality and efficiency of feedback without relying on privileged information, offering a lightweight solution for online PbRL tasks.
   - **Year**: 2024

3. **Title**: Strategyproof Reinforcement Learning from Human Feedback (arXiv:2503.09561)
   - **Authors**: Thomas Kleine Buening, Jiarui Gan, Debmalya Mandal, Marta Kwiatkowska
   - **Summary**: The authors investigate the strategic behavior of individuals providing feedback in reinforcement learning from human feedback (RLHF) scenarios. They demonstrate that existing RLHF methods are susceptible to manipulation, leading to misaligned policies. To address this, they propose a pessimistic median algorithm that is approximately strategyproof and converges to the optimal policy under certain conditions, highlighting the trade-off between incentive alignment and policy alignment.
   - **Year**: 2025

4. **Title**: RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback (arXiv:2309.00267)
   - **Authors**: Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash
   - **Summary**: This study compares Reinforcement Learning from AI Feedback (RLAIF) with traditional Reinforcement Learning from Human Feedback (RLHF). The authors demonstrate that RLAIF, which utilizes AI-generated preferences, achieves performance comparable to RLHF across tasks like summarization and dialogue generation. They also introduce direct-RLAIF (d-RLAIF), a technique that bypasses reward model training by obtaining rewards directly from an off-the-shelf LLM during RL, achieving superior performance to canonical RLAIF.
   - **Year**: 2023

5. **Title**: KTO: Model Alignment as Prospect Theoretic Optimization
   - **Authors**: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
   - **Summary**: The authors propose KTO, a direct alignment method that optimizes models end-to-end on human-labeled outputs, reducing misalignment risks associated with proxy objectives. KTO constructs a relaxed generalization to preference distributions by requiring only binary feedback signals, reflecting human loss aversion and risk aversion. This approach aims to achieve tighter alignment with human values and improved interpretability compared to traditional RLHF methods.
   - **Year**: 2024

6. **Title**: Understanding Likelihood Over-optimisation in Direct Alignment Algorithms
   - **Authors**: Zhengyan Shi, Sander Land, Acyr Locatelli, Matthieu Geist, Max Bartolo
   - **Summary**: This paper examines the phenomenon of likelihood over-optimization in direct alignment algorithms, where models trained end-to-end on human-labeled outputs may overfit to the training data, leading to misalignment. The authors analyze the causes of this issue and propose mitigation strategies to balance model performance and alignment with human preferences.
   - **Year**: 2024

7. **Title**: Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms
   - **Authors**: Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna
   - **Summary**: The authors investigate the scaling behavior of reward model over-optimization in direct alignment algorithms. They identify that as models scale, the risk of over-optimizing the reward model increases, potentially leading to misaligned policies. The paper provides insights into the trade-offs between model size, reward model complexity, and alignment fidelity, offering guidelines for designing scalable and aligned AI systems.
   - **Year**: 2024

8. **Title**: The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization
   - **Authors**: Shengyi Huang, Michael Noukhovitch, Arian Hosseini, Kashif Rasul, Weixun Wang
   - **Summary**: This case study provides a comprehensive analysis of implementing RLHF using Proximal Policy Optimization (PPO) for the task of TL;DR summarization. The authors detail the challenges encountered, such as reward hacking and instability, and present solutions to improve training stability and alignment with human preferences. The paper serves as a practical guide for researchers implementing RLHF in similar contexts.
   - **Year**: 2024

9. **Title**: The N Implementation Details of RLHF with PPO
   - **Authors**: Yiyang Feng
   - **Summary**: This paper offers an in-depth exploration of the implementation details of RLHF using PPO. It discusses the nuances of reward modeling, policy optimization, and the integration of human feedback. The author provides insights into best practices and common pitfalls, contributing to the broader understanding of RLHF methodologies.
   - **Year**: 2025

10. **Title**: Proximal Policy Optimization - Spinning Up documentation
    - **Authors**: OpenAI
    - **Summary**: This documentation provides a comprehensive overview of Proximal Policy Optimization (PPO), a popular reinforcement learning algorithm. It covers theoretical foundations, implementation details, and practical considerations, serving as a valuable resource for researchers and practitioners working with PPO in various applications, including human-AI interaction scenarios.
    - **Year**: 2025

**Key Challenges:**

1. **Dynamic Human Preferences**: Human preferences are not static; they evolve over time and across contexts. Developing AI systems that can adapt to these changes in real-time remains a significant challenge.

2. **Bidirectional Adaptation**: Achieving effective bidirectional adaptation requires AI systems to not only learn from human feedback but also to provide interpretable explanations that empower users to influence AI behavior actively. Balancing this interaction is complex.

3. **Non-Stationarity in Human-AI Interaction**: The dynamic nature of human-AI interactions introduces non-stationarity, making it difficult for AI systems to maintain alignment over time without continuous learning mechanisms.

4. **Scalability of Human Feedback**: Collecting high-quality human feedback at scale is resource-intensive. Methods like RLAIF attempt to address this by leveraging AI-generated feedback, but ensuring the reliability and alignment of such feedback poses challenges.

5. **Interpretability and Transparency**: Providing human-centric explanations of AI decisions is crucial for user trust and control. Designing systems that generate interpretable feedback while maintaining performance is an ongoing challenge. 