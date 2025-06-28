# Self-Generating Adaptive Curricula for Open-Ended Reinforcement Learning

## Introduction

The rapid advancements in deep reinforcement learning (RL) and large language models (LLMs) have enabled agents to excel in increasingly challenging tasks. However, once the task is mastered, the learning process typically terminates. In contrast, human intelligence continually evolves through open-ended learning, where agents and environments co-evolve, presenting an endless stream of novel challenges. This open-ended learning (OEL) is crucial for developing agents with general capabilities and the ability to adapt to unexpected scenarios.

The proposed research, "Self-Generating Adaptive Curricula for Open-Ended Reinforcement Learning," aims to develop a framework that leverages a large language model (LLM) as a meta-controller to generate adaptive curricula for RL agents. This approach seeks to sustain an endless learning loop, promoting generalization and adaptability in real-world settings. The motivation behind this research is to automate task design using the agent’s own capabilities, thereby avoiding the laborious and potentially incomplete nature of handcrafted curricula.

### Research Objectives

1. **Automated Curriculum Generation**: Develop a method to automatically generate task specifications using an LLM, conditioned on the agent's current policy performance and failure modes.
2. **Quality-Diversity Filtering**: Implement a quality-diversity filter to retain high-impact, diverse tasks, preventing curriculum collapse.
3. **Online ODD-Score Metrics**: Track the emergence of new capabilities using online Out-of-Distribution Difficulty (ODD) metrics.
4. **Sim2Real Transfer**: Improve sim2real transfer by approximating real-world complexity through the evolving curriculum.

### Significance

The proposed research addresses the critical need for open-ended learning in RL, offering a scalable and automated approach to curriculum design. By leveraging LLMs, the framework can generate diverse and challenging tasks, fostering continuous learning and generalization. This research holds significant potential for improving the adaptability and robustness of RL agents, particularly in real-world applications where the environment is dynamic and unpredictable.

## Methodology

### Research Design

The methodology consists of three main components: the LLM-based curriculum generator, the quality-diversity filter, and the evaluation metrics.

#### LLM-Based Curriculum Generator

The LLM serves as a meta-controller that generates new task specifications based on the agent's current performance and failure modes. The process involves the following steps:

1. **Policy Performance Logging**: The agent logs its trajectories and identifies behaviors it cannot yet solve.
2. **Skill Gap Identification**: The LLM processes the logged trajectories to identify skill gaps, i.e., behaviors that the agent cannot perform.
3. **Task Specification Generation**: The LLM generates task specifications conditioned on the identified skill gaps. These specifications can range from simple variations to novel compound objectives.
4. **Task Instantiation**: The generated tasks are instantiated in a simulator or via scripted environments.

#### Quality-Diversity Filter

To prevent curriculum collapse, a quality-diversity filter is employed to retain high-impact, diverse tasks. The filter evaluates tasks based on their expected impact on the agent's learning and diversity in task characteristics. The evaluation criteria include:

- **Expected Impact**: Measures the potential improvement in the agent's performance.
- **Diversity**: Ensures that the selected tasks cover a wide range of challenges and skills.

The quality-diversity filter uses a multi-objective optimization approach to balance these criteria, ensuring that the curriculum remains diverse and effective.

#### Evaluation Metrics

The performance of the adaptive curriculum is evaluated using the following metrics:

- **ODD-Score**: Tracks the emergence of new capabilities by measuring the difficulty of tasks that the agent encounters. A higher ODD-score indicates that the agent is continually challenged with novel and complex tasks.
- **Generalization Performance**: Evaluates the agent's ability to generalize its skills to unseen tasks.
- **Sim2Real Transfer**: Measures the agent's performance in real-world scenarios after training on the generated curriculum.

### Experimental Design

The experimental design involves the following steps:

1. **Environment Setup**: Establish a simulated environment or scripted environments for task instantiation.
2. **Agent Initialization**: Initialize the RL agent with a base policy.
3. **Curriculum Generation**: Apply the LLM-based curriculum generator to generate tasks based on the agent's current performance and failure modes.
4. **Quality-Diversity Filtering**: Apply the quality-diversity filter to select high-impact, diverse tasks.
5. **Agent Training**: Train the RL agent on the selected tasks.
6. **Performance Evaluation**: Evaluate the agent's performance using the ODD-score, generalization performance, and sim2Real transfer metrics.

### Mathematical Formulation

The LLM-based curriculum generation can be mathematically formulated as follows:

1. **Policy Performance Logging**:
   \[
   P_{t} = f(A_{t}, E_{t}, S_{t})
   \]
   where \( P_{t} \) is the policy performance at time \( t \), \( A_{t} \) is the agent's actions, \( E_{t} \) is the environment state, and \( S_{t} \) is the reward signal.

2. **Skill Gap Identification**:
   \[
   G_{t} = \text{Identify}(P_{t}, S_{t})
   \]
   where \( G_{t} \) is the skill gap at time \( t \), identified by analyzing the policy performance and reward signal.

3. **Task Specification Generation**:
   \[
   T_{t} = \text{Generate}(G_{t})
   \]
   where \( T_{t} \) is the task specification at time \( t \), generated by the LLM based on the identified skill gap.

4. **Quality-Diversity Filtering**:
   \[
   QD_{t} = \text{Filter}(T_{t})
   \]
   where \( QD_{t} \) is the quality-diversity filtered task at time \( t \), retaining high-impact and diverse tasks.

## Expected Outcomes & Impact

### Expected Outcomes

1. **Adaptive Curricula**: A framework that automatically generates adaptive curricula for RL agents using an LLM.
2. **Quality-Diversity Filter**: A method to retain high-impact, diverse tasks, preventing curriculum collapse.
3. **Online ODD-Score Metrics**: A metric to track the emergence of new capabilities in RL agents.
4. **Improved Sim2Real Transfer**: Enhanced performance in real-world scenarios after training on the generated curriculum.

### Impact

The proposed research has the potential to significantly advance the field of open-ended learning in RL. By automating curriculum design and promoting continuous learning, the framework can foster the development of agents with general capabilities and adaptability. The research outcomes can be applied to various domains, including robotics, autonomous systems, and real-world applications, where the environment is dynamic and unpredictable. Furthermore, the proposed methodology can serve as a foundation for future research in open-ended learning, curriculum design, and sim2real transfer.

## Conclusion

The proposed research, "Self-Generating Adaptive Curricula for Open-Ended Reinforcement Learning," aims to develop a framework that leverages a large language model as a meta-controller to generate adaptive curricula for RL agents. This approach seeks to sustain an endless learning loop, promoting generalization and adaptability in real-world settings. By addressing the critical challenges of automating curriculum design, ensuring generalization to unseen tasks, balancing exploration and exploitation, improving sim2real transfer, and enhancing computational efficiency, the research holds significant potential for advancing the field of open-ended learning in RL.