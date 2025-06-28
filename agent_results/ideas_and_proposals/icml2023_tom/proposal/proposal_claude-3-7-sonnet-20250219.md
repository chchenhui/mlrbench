# Meta-ToM: A Meta-Learning Framework for Theory of Mind Adaptation in Conversational AI

## 1. Introduction

Theory of Mind (ToM) – the cognitive ability to attribute mental states to oneself and others – forms the foundation of successful human social interaction. Traditional conversational AI systems operate with limited or fixed models of user mental states, often failing to capture the nuanced, evolving nature of human beliefs, knowledge, intentions, and desires during dialogue. This limitation leads to responses that feel generic, misaligned, or frustrating to users, particularly in extended or complex interactions.

Recent advances in large language models (LLMs) have significantly improved the fluency and contextual awareness of conversational agents (Jafari et al., 2025; Cross et al., 2024). However, even state-of-the-art systems struggle with authentic Theory of Mind reasoning – the ability to develop and maintain accurate mental models of users that evolve through conversation. Existing approaches to ToM in conversational agents typically employ either rule-based symbolic approaches (Sclar et al., 2023) or end-to-end neural architectures (Qiu et al., 2023), but both face significant limitations. Symbolic approaches lack flexibility and scalability, while neural approaches require extensive training data and struggle with rapid adaptation to new users.

The central challenge lies in developing conversational agents that can quickly form and refine accurate mental models of individual users based on minimal interaction. This capability is essential for truly personalized, efficient, and trustworthy human-AI communication. While recent meta-learning approaches have shown promise for personalization in dialogue systems (Johnson & Lee, 2024), they have not fully leveraged the potential of meta-learning specifically for ToM adaptation.

This research proposes Meta-ToM, a novel meta-learning framework that endows conversational agents with a dedicated Theory of Mind module capable of few-shot adaptation to new users. Our approach synthesizes insights from cognitive science, meta-learning algorithms, and dialogue systems to create a conversational agent that can rapidly build and refine user-specific mental models. The central innovation lies in the application of Model-Agnostic Meta-Learning (MAML) to ToM inference, enabling the system to learn how to efficiently update its representations of user mental states using only a handful of interactions.

The significance of this research extends across multiple domains. For human-computer interaction, it offers a path toward more natural, efficient, and satisfying AI assistants that truly understand user perspectives. In cognitive science and AI alignment, it provides a computational framework for modeling and implementing ToM capabilities that more closely resemble human social cognition. Furthermore, it addresses pressing challenges in personalization, adaptation, and responsible AI development by creating systems that can better represent and respect individual user differences.

## 2. Methodology

Our methodology integrates meta-learning techniques with Theory of Mind modeling in a novel framework for conversational AI. The Meta-ToM approach consists of four main components: (1) a synthetic dialogue corpus with mental state annotations, (2) a neural ToM architecture, (3) a meta-learning procedure for few-shot adaptation, and (4) an integrated dialogue generation system. We detail each component below.

### 2.1 Synthetic Dialogue Corpus with Mental State Annotations

To overcome the data annotation challenge identified in prior work, we will construct a large-scale synthetic dialogue corpus with comprehensive mental state annotations. This corpus will be generated using a three-stage process:

1. **Base Dialogue Generation**: We will utilize an existing large language model (e.g., GPT-4) to generate diverse multi-turn dialogues across various domains (e.g., customer service, information-seeking, collaborative problem-solving).

2. **Mental State Annotation**: For each dialogue turn, we will annotate the following latent mental states:
   - Beliefs ($B$): Propositions the user believes to be true about the world
   - Knowledge ($K$): Information the user possesses or lacks
   - Goals ($G$): The user's immediate and overall objectives
   - Intentions ($I$): The user's communicative intentions for each utterance

3. **Persona Clustering**: We will cluster the generated dialogues into distinct "personas" based on patterns in mental states, creating synthetic "users" with consistent characteristics for meta-training.

The annotation process will be semi-automated, using structured prompting of LLMs followed by human verification on a subset to ensure quality. The resulting corpus will contain 10,000 dialogues with 5-15 turns each, clustered into 200 distinct personas.

### 2.2 Neural ToM Architecture

We propose a dual-encoder architecture for the ToM module that separately models the agent's beliefs about the world and its beliefs about the user's mental states. The module consists of:

1. **Mental State Encoder**: A transformer-based encoder that processes dialogue history $H_t$ until turn $t$ and outputs a representation of the user's mental state $M_t$:

$$M_t = \text{Encoder}_M(H_t; \theta_M)$$

The mental state $M_t$ is decomposed into specific components:

$$M_t = [B_t, K_t, G_t, I_t]$$

2. **Belief Update Function**: A neural network that updates the agent's belief about the user's mental state based on new utterances:

$$M_{t+1} = M_t + \text{Update}(M_t, u_{t+1}; \theta_U)$$

where $u_{t+1}$ is the user's utterance at turn $t+1$.

3. **Mental State Attention**: A mechanism that selectively attends to relevant aspects of the mental state representation when generating responses:

$$\alpha_t = \text{Attention}(q_t, M_t; \theta_A)$$

where $q_t$ is a query vector derived from the current dialogue context.

The complete ToM module is parameterized by $\theta_\text{ToM} = [\theta_M, \theta_U, \theta_A]$, which will be subject to meta-learning.

### 2.3 Meta-Learning Procedure for Few-Shot Adaptation

We employ Model-Agnostic Meta-Learning (MAML) to train the ToM module for rapid adaptation. The meta-learning objective is to find initial parameters $\theta_\text{ToM}^*$ that can be quickly fine-tuned to specific users with minimal data:

1. **Meta-Training**: For each synthetic persona $p$ in our corpus, we sample a support set $\mathcal{S}_p$ and a query set $\mathcal{Q}_p$ of dialogues:

   a. Compute adapted parameters for persona $p$:
   
   $$\theta_p = \theta_\text{ToM} - \alpha \nabla_{\theta_\text{ToM}} \mathcal{L}_\text{ToM}(\mathcal{S}_p; \theta_\text{ToM})$$
   
   where $\alpha$ is the adaptation learning rate and $\mathcal{L}_\text{ToM}$ is a supervised loss comparing predicted mental states with ground truth annotations.
   
   b. Evaluate the adapted model on the query set:
   
   $$\mathcal{L}_\text{meta}(\theta_\text{ToM}) = \sum_{p} \mathcal{L}_\text{ToM}(\mathcal{Q}_p; \theta_p)$$
   
   c. Update the meta-parameters:
   
   $$\theta_\text{ToM} \leftarrow \theta_\text{ToM} - \beta \nabla_{\theta_\text{ToM}} \mathcal{L}_\text{meta}(\theta_\text{ToM})$$
   
   where $\beta$ is the meta learning rate.

2. **Few-Shot Adaptation**: During deployment, the ToM module adapts to a new user after observing $k$ turns of dialogue (where $k$ is typically 3-5 turns):

   $$\theta_\text{user} = \theta_\text{ToM}^* - \alpha \nabla_{\theta_\text{ToM}^*} \mathcal{L}_\text{adapt}(H_k; \theta_\text{ToM}^*)$$

   where $H_k$ is the history of the first $k$ turns and $\mathcal{L}_\text{adapt}$ is an adaptation loss based on response quality signals.

### 2.4 Integrated Dialogue Generation System

The Meta-ToM framework integrates with a dialogue generation system through a joint optimization process:

1. **Response Generation**: Given dialogue history $H_t$ and inferred mental state $M_t$, the system generates a response $r_t$:

   $$r_t = \text{Generator}(H_t, M_t; \theta_G)$$

2. **Joint Loss Function**: During training, we optimize a multi-component loss function:

   $$\mathcal{L} = \lambda_1 \mathcal{L}_\text{ToM} + \lambda_2 \mathcal{L}_\text{LM} + \lambda_3 \mathcal{L}_\text{alignment}$$

   where:
   - $\mathcal{L}_\text{ToM}$ measures the accuracy of mental state prediction
   - $\mathcal{L}_\text{LM}$ is the standard language modeling loss
   - $\mathcal{L}_\text{alignment}$ ensures responses are aligned with the predicted mental states

3. **Continual Adaptation**: During extended conversations, the system periodically updates its mental state model based on user interactions:

   $$\theta_\text{user}^{(t+n)} = \theta_\text{user}^{(t)} - \alpha \nabla_{\theta_\text{user}^{(t)}} \mathcal{L}_\text{adapt}(H_{t:t+n}; \theta_\text{user}^{(t)})$$

   where $n$ is an update interval (typically 5-10 turns).

### 2.5 Experimental Design and Evaluation

We will evaluate Meta-ToM through a comprehensive set of experiments:

1. **Benchmark Evaluation**: We will test on established ToM benchmarks, including:
   - ToMi (Sclar et al., 2023)
   - False-belief tasks adapted for dialogue
   - MIND-Dial (Qiu et al., 2023)

2. **Simulated User Evaluation**: We will create a simulation environment with agents exhibiting diverse mental states and interaction patterns to test adaptation speed and accuracy.

3. **Human Evaluation Study**: We will conduct a user study with 100 participants interacting with three systems:
   - Meta-ToM (our approach)
   - A standard dialogue system without ToM capabilities
   - A dialogue system with fixed ToM modeling

   Each participant will engage in 3 conversations (one with each system) of approximately 15 turns each, focusing on information-seeking and collaborative tasks.

4. **Evaluation Metrics**:
   - **ToM Accuracy**: How accurately the system predicts user mental states (for simulated users where ground truth is available)
   - **Adaptation Speed**: Number of turns required to achieve stable ToM performance
   - **Task Success Rate**: Completion rate for collaborative and information-seeking tasks
   - **User Satisfaction**: Self-reported measures using standardized questionnaires
   - **Perceived Empathy**: Ratings of how well the system understood the user's perspective
   - **Conversation Efficiency**: Number of turns required to achieve user goals

5. **Ablation Studies**: We will conduct ablation experiments to assess the contribution of each component:
   - Meta-learning vs. standard learning
   - Different mental state components (beliefs, knowledge, goals, intentions)
   - Adaptation frequency and strategies

## 3. Expected Outcomes & Impact

The Meta-ToM framework is expected to deliver several significant outcomes that advance the state of the art in ToM modeling for conversational AI:

### 3.1 Technical Advancements

1. **Rapid Adaptation Capability**: We expect Meta-ToM to demonstrate the ability to form accurate mental models of users after just 3-5 turns of conversation, significantly outperforming systems without meta-learning capabilities. This will be measured by ToM accuracy metrics on held-out test data.

2. **Generalization Across Domains**: The meta-learning approach should enable the system to generalize ToM capabilities across different conversational domains and tasks, demonstrating robust performance even in scenarios not seen during training.

3. **Efficient Representation Learning**: The framework will develop more compact and efficient representations of user mental states, requiring fewer parameters than end-to-end approaches while maintaining or improving performance.

4. **Integration Blueprint**: We will provide a detailed blueprint for integrating ToM modules with existing dialogue systems, offering a practical path for enhancing current conversational AI with ToM capabilities.

### 3.2 Research Contributions

1. **Computational ToM Model**: Meta-ToM will contribute a novel computational model of Theory of Mind that bridges cognitive science theories with practical AI implementations, potentially informing future cognitive science research.

2. **Evaluation Framework**: Our comprehensive evaluation methodology will establish new standards for assessing ToM capabilities in conversational agents, addressing current gaps in evaluation metrics and benchmarks.

3. **Synthetic Data Generation**: The techniques developed for creating annotated ToM training data will benefit researchers facing similar data scarcity challenges in other areas of AI and cognitive modeling.

### 3.3 Practical Impact

1. **Enhanced Personalization**: Conversational agents equipped with Meta-ToM will deliver more personalized experiences by rapidly adapting to individual user characteristics, preferences, and communication styles.

2. **Improved User Satisfaction**: We anticipate significantly higher user satisfaction ratings for Meta-ToM compared to baseline systems, particularly in extended conversations and complex tasks.

3. **Reduced Conversational Friction**: By better modeling user knowledge and intentions, Meta-ToM should reduce misunderstandings and repetition in conversations, leading to more efficient interactions.

4. **Applications in Critical Domains**: The framework will enable more effective AI assistants in domains where understanding user perspectives is crucial, such as healthcare, education, and customer support.

### 3.4 Broader Impact

1. **Advancing Human-AI Collaboration**: Meta-ToM will contribute to the development of AI systems that can better understand and anticipate human needs, fostering more natural and productive human-AI collaboration.

2. **Ethical AI Development**: The explicit modeling of user mental states in Meta-ToM provides opportunities for greater transparency and control in how AI systems represent and respond to users.

3. **Foundations for Social AI**: This research establishes foundations for socially aware AI systems that can navigate complex interpersonal dynamics, with potential applications in group interactions and collaborative environments.

4. **Cross-disciplinary Insights**: The integration of cognitive science, machine learning, and dialogue systems in Meta-ToM will generate insights that benefit multiple research communities and accelerate progress in artificial social intelligence.

By developing a flexible, adaptive Theory of Mind capability for conversational agents, Meta-ToM addresses a fundamental challenge in human-AI interaction while advancing our understanding of computational social cognition. The framework's emphasis on rapid, few-shot adaptation distinguishes it from previous approaches and positions it to significantly impact both research and applications in conversational AI.