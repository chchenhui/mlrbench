1. **Title**: Improving Multimodal Interactive Agents with Reinforcement Learning from Human Feedback (arXiv:2211.11602)
   - **Authors**: Josh Abramson, Arun Ahuja, Federico Carnevale, Petko Georgiev, Alex Goldin, Alden Hung, Jessica Landon, Jirka Lhotka, Timothy Lillicrap, Alistair Muldal, George Powell, Adam Santoro, Guy Scully, Sanjana Srivastava, Tamara von Glehn, Greg Wayne, Nathaniel Wong, Chen Yan, Rui Zhu
   - **Summary**: This paper presents a method to enhance multimodal interactive agents by integrating reinforcement learning from human feedback (RLHF). The authors collected human-agent interaction data in a simulated 3D environment and developed the "Inter-temporal Bradley-Terry" (IBT) model to capture human judgments. Agents trained with IBT-derived rewards demonstrated improved performance across various metrics, including human evaluations during live interactions.
   - **Year**: 2022

2. **Title**: PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training (arXiv:2106.05091)
   - **Authors**: Kimin Lee, Laura Smith, Pieter Abbeel
   - **Summary**: PEBBLE introduces an off-policy interactive reinforcement learning algorithm that efficiently utilizes human feedback. By actively querying human preferences and relabeling past experiences, the method enhances sample efficiency. Additionally, unsupervised pre-training is employed to further improve learning efficiency. The approach is validated on complex locomotion and robotic manipulation tasks.
   - **Year**: 2021

3. **Title**: Accelerating Reinforcement Learning Agent with EEG-based Implicit Human Feedback (arXiv:2006.16498)
   - **Authors**: Duo Xu, Mohit Agarwal, Ekansh Gupta, Faramarz Fekri, Raghupathy Sivakumar
   - **Summary**: This study explores the use of EEG-based error-related potentials (ErrP) as implicit human feedback to accelerate reinforcement learning agents. The authors propose a framework that integrates ErrP signals into the learning process, demonstrating improved learning efficiency and robustness in 2D navigational games through real user experiments.
   - **Year**: 2020

4. **Title**: Creating Multimodal Interactive Agents with Imitation and Self-Supervised Learning (arXiv:2112.03763)
   - **Authors**: DeepMind Interactive Agents Team, Josh Abramson, Arun Ahuja, Arthur Brussee, Federico Carnevale, Mary Cassin, Felix Fischer, Petko Georgiev, Alex Goldin, Mansi Gupta, Tim Harley, Felix Hill, Peter C Humphreys, Alden Hung, Jessica Landon, Timothy Lillicrap, Hamza Merzic, Alistair Muldal, Adam Santoro, Guy Scully, Tamara von Glehn, Greg Wayne, Nathaniel Wong, Chen Yan, Rui Zhu
   - **Summary**: The authors present a method for developing multimodal interactive agents by combining imitation learning of human-human interactions in a simulated environment with self-supervised learning. The resulting agents, termed MIA, successfully interact with humans 75% of the time, highlighting the effectiveness of this approach in creating agents capable of natural human interaction.
   - **Year**: 2021

5. **Title**: Reinforcement Learning from Human Feedback
   - **Authors**: Various
   - **Summary**: This article provides an overview of reinforcement learning from human feedback (RLHF), discussing its background, motivation, methods for collecting human feedback, applications, and limitations. It highlights the challenges in defining reward functions and the benefits of incorporating human preferences to guide agent behavior.
   - **Year**: 2025

**Key Challenges:**

1. **Interpretation of Implicit Feedback**: Accurately interpreting multimodal implicit human feedback, such as gestures, facial expressions, and tone, remains challenging due to the complexity and variability of human behaviors.

2. **Data Efficiency**: Collecting and utilizing human feedback efficiently is crucial, as obtaining large-scale, high-quality data can be resource-intensive and time-consuming.

3. **Adaptation to Non-Stationary Preferences**: Human preferences and environments are dynamic; developing agents that can adapt to these changes over time is a significant challenge.

4. **Integration of Multimodal Signals**: Effectively combining various feedback modalities into a cohesive learning framework requires sophisticated models capable of handling diverse data types.

5. **Scalability and Generalization**: Ensuring that agents trained with human feedback can generalize across different tasks and scale to real-world applications without overfitting to specific scenarios is a persistent challenge. 