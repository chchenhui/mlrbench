\section{Title}  
**Socially-Aligned Intrinsic Reward Learning from Multimodal Implicit Human Feedback**  

---

\section{Introduction}  
**Background**  
Interactive machine learning systems often rely on explicit human feedback, such as scalar rewards or predefined demonstrations, to guide agent behavior. However, humans naturally communicate intent through socially rich, multimodal cues—including speech, gestures, facial expressions, and gaze—that remain underutilized in current frameworks. This gap limits agents’ ability to adapt to dynamic environments and personalized user needs, particularly in socially complex domains like healthcare, education, and assistive robotics. Existing approaches, such as reinforcement learning from human feedback (RLHF), struggle with **data inefficiency**, **non-stationary feedback semantics**, and **multimodal fusion challenges**, as highlighted by recent studies (Abramson et al., 2022; Lee et al., 2021).  

**Research Objectives**  
This research aims to design a novel framework for **learning socially aligned intrinsic rewards** by interpreting implicit multimodal feedback during human-agent interaction. Specifically, we propose to:  
1. Develop a **contrastive learning model** to map heterogeneous feedback signals (e.g., speech prosody, gaze patterns) into a unified latent space representing human intent.  
2. Integrate **inverse reinforcement learning (IRL)** and **meta-reinforcement learning (meta-RL)** to infer non-stationary reward functions and adapt policies to evolving user preferences.  
3. Validate the framework’s ability to generalize across diverse users and tasks while maintaining computational efficiency.  

**Significance**  
By addressing the interpretation of implicit feedback and adaptation to dynamic contexts, this work will advance interactive AI systems in three ways:  
- **Personalization**: Agents that leverage naturalistic human cues can better align with individual user needs, such as adapting tutoring strategies to a student’s confusion detected via facial expressions.  
- **Scalability**: Reducing reliance on hand-crafted rewards enables deployment in real-world settings where explicit feedback is impractical (e.g., assistive robotics for users with motor impairments).  
- **Social Coordination**: Intrinsic reward learning fosters human-AI collaboration by grounding agent behavior in socially intelligible signals.  

---

\section{Methodology}  
The framework comprises three stages: (1) multimodal feedback encoding, (2) intrinsic reward learning, and (3) meta-adaptive policy optimization.  

\subsection{Data Collection and Multimodal Encoding}  
**Dataset Construction**  
- **Environment**: Simulate human-agent interactions in a virtual education domain (e.g., robot tutor and student solving math problems).  
- **Modalities**: Collect synchronized data streams—speech transcripts, eye gaze (Tobii eyetracker), facial expressions (OpenFace), and gestures (Kinect).  
- **Feedback Annotation**: Pair interactions with post-hoc human ratings of agent effectiveness to train latent reward predictors.  

**Contrastive Learning for Multimodal Alignment**  
A transformer encoder processes each modality independently, then maps them into a shared latent space using contrastive learning:  
$$  
\mathcal{L}_{\text{cont}} = -\log \frac{\exp(s(\mathbf{h}_i, \mathbf{h}_j)/\tau)}{\sum_{k=1}^K \exp(s(\mathbf{h}_i, \mathbf{h}_k)/\tau)},  
$$  
where $\mathbf{h}_i, \mathbf{h}_j$ are latent vectors of aligned feedback pairs (e.g., speech + gaze), $s(\cdot)$ is a similarity function, and $\tau$ is temperature.  

\subsection{Inverse Reward Learning from Latent Feedback}  
Model human intent as a reward function $r_\theta(\mathbf{z}_t)$, where $\mathbf{z}_t$ is the fused latent state at timestep $t$. Using maximum entropy IRL:  
$$  
\theta^* = \arg\max_\theta \mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T r_\theta(\mathbf{z}_t)\right] - H(\pi_\theta),  
$$  
where $H(\pi_\theta)$ is the policy entropy. The reward predictor is pre-trained on human annotations and fine-tuned via interaction.  

\subsection{Meta-Adaptive Policy Optimization}  
To handle non-stationary human preferences, employ a Meta-Neural Process (MNP):  
1. **Context Encoder**: Embeds interaction history $\mathcal{H} = \{(\mathbf{z}_i, a_i)\}_{i=1}^N$ into a task-specific context $\mathbf{c}$.  
2. **Adaptation**: Update policy parameters $\phi$ via gradient descent on a small support set $\mathcal{S}$:  
$$  
\phi' = \phi - \alpha \nabla_\phi \mathcal{L}_{\text{RL}}(\mathcal{S}; \mathbf{c}),  
$$  
3. **Meta-Training**: Optimize initialization $\phi$ to minimize loss on query sets $\mathcal{Q}$ across tasks.  

\subsection{Experimental Design}  
**Baselines**  
- PEBBLE (Lee et al., 2021): State-of-the-art interactive RL with preference relabeling.  
- MIA (Abramson et al., 2021): Multimodal agent trained via imitation learning.  
- EEG-RL (Xu et al., 2020): Implicit feedback via physiological signals.  

**Evaluation Metrics**  
1. **Task Performance**: Success rate in simulated scenarios (e.g., student quiz scores).  
2. **Alignment Score**: Correlation between predicted rewards and human satisfaction ratings.  
3. **Adaptation Speed**: Time taken to adjust to preference shifts (e.g., after simulating a user’s change in teaching style).  
4. **User Study**: Subjective ratings of agent "social awareness" on a 7-point Likert scale.  

**Implementation Details**  
- **Simulation**: Unity-based environment with customizable user Feedback APIs.  
- **Training**: Pre-train multimodal encoders on 500+ annotated interactions, then meta-train policies with Proximal Policy Optimization (PPO).  

---

\section{Expected Outcomes \& Impact}  
**Anticipated Outcomes**  
1. A **Unified Multimodal Encoding Model** capable of interpreting implicit feedback with $\geq85\%$ alignment to human intent (validated via user studies).  
2. **Meta-Adaptive Policies** that adjust to preference shifts 2$\times$ faster than PEBBLE in non-stationary environments.  
3. **Generalization Benchmarks**: Demonstration of the framework’s applicability to healthcare (e.g., assistive robots interpreting patient discomfort) and education (e.g., tutors detecting student engagement).  

**Broader Impact**  
- **Ethical AI**: Reducing dependence on explicit feedback lowers barriers for users with limited capacity to provide structured input (e.g., individuals with cognitive disabilities).  
- **HCI Advancement**: Insights into multimodal feedback integration can inform next-generation adaptive interfaces.  
- **Theoretical Contribution**: Formal analysis of the minimal assumptions required for interaction-grounded learning (e.g., identifiability conditions for latent rewards).  

This work bridges the gap between machine learning and human-computer interaction, paving the way for AI systems that learn “in the wild” through natural, socially meaningful exchanges.