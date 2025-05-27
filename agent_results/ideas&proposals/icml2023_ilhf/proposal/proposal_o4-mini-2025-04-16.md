1. Title  
Socially-Aligned Intrinsic Reward Learning via Multimodal Human Feedback

2. Introduction  
Background  
Interactive learning algorithms that adapt to human users are crucial for next-generation AI systems in domains such as assistive robotics, education, and healthcare. Traditional reinforcement learning (RL) agents rely on hand-crafted scalar rewards or explicit demonstrations to infer user intent. However, humans naturally communicate intent through a rich spectrum of implicit multimodal signals—facial expressions, eye gaze, tone of voice, gestures, and natural language hints. Current interactive ML techniques underutilize these signals, thereby limiting adaptability to diverse users and rapidly changing environments.  

Research Objectives  
This project aims to design and validate a unified framework in which an agent:  
• Learns an intrinsic reward function by interpreting multimodal implicit human feedback without any pre-specified semantics.  
• Adapts online to non-stationary user preferences and environmental dynamics via meta-reinforcement learning.  
• Demonstrates improved sample-efficiency, alignment with user intent, and generalization across tasks.  

Significance  
By moving beyond hand-crafted rewards, our framework will:  
• Enable socially aware agents that can interpret subtle cues (e.g., a student’s confused look signifies “slow down”).  
• Reduce annotation cost and human burden compared to explicit labeling.  
• Foster scalable AI assistants that seamlessly integrate into real-world social settings.  

3. Methodology  
Our methodology consists of four major components: (A) Multimodal Data Collection, (B) Latent Reward Encoder, (C) Intrinsic Reward Inference via Inverse Reinforcement Learning, and (D) Meta-RL Adaptation. Finally, we detail an experimental design to validate the framework.  

A. Multimodal Data Collection  
We construct an interactive testbed where a human participant interacts with an agent in a sequential decision‐making task (e.g., a robot tutor guiding a user through a block‐stacking exercise). During each time step $t$, we record:  
  – Visual stream $I_t$ (RGB frames capturing facial/gesture cues)  
  – Audio stream $A_t$ (speech transcripts + prosody features)  
  – Eye‐gaze vector $g_t\in\mathbb{R}^2$ (screen coordinates)  
  – Agent state $s_t\in\mathcal{S}$ and chosen action $a_t\in\mathcal{A}$  
  – Optional explicit scalar reward $r_t$ (for baseline comparison)  

We align all modalities at 10 Hz and annotate a small subset by human experts to bootstrap initial training.  

B. Latent Reward Encoder  
We train a transformer‐based encoder $E_\phi$ that maps concatenated multimodal features to a shared latent space:  
$$z_t = E_\phi\bigl(I_t, A_t, g_t, s_t, a_t\bigr)\;\in\mathbb{R}^d\,. $$  
Encoder architecture:  
 1. Modality‐specific feature extractors (CNN for $I_t$, BiLSTM for $A_t$, MLP for $g_t$)  
 2. Concatenate and feed into a multi‐layer transformer with self‐attention heads.  
 3. Project the output to a $d$‐dimensional embedding $z_t$.  

Contrastive Pre‐training  
We pre‐train $\phi$ using a contrastive InfoNCE loss on positive pairs (time‐aligned true feedback) and negative pairs (shuffled feedback):  
$$\mathcal{L}_{NCE}(\phi) = -\sum_{t=1}^N \log\frac{\exp\bigl(z_t^\top z_t^+/ \tau\bigr)}{\sum_{j=1}^N \exp\bigl(z_t^\top z_j^- / \tau\bigr)},$$  
where $z_t^+$ is the embedding of the matching feedback at time $t$, $z_j^-$ are negatives, and $\tau$ is a temperature hyperparameter.  

C. Intrinsic Reward Inference via Inverse Reinforcement Learning  
We posit that the human’s implicit feedback corresponds to an unknown reward function $r_\theta(s,a; z)$. We learn $\theta$ by matching the agent’s policy $\pi_\theta$ to the observed behavior distribution under human feedback. Using a Maximum Entropy IRL objective [Ziebart et al.]:  
$$\max_\theta \;\mathbb{E}_{\tau\sim\mathcal{D}}\bigl[\,\sum_{t=0}^{T}\,r_\theta(s_t,a_t;z_t)\bigr]\;-\;\log Z(\theta),$$  
where $\tau=(s_0,a_0,\dots,s_T)$ is a trajectory from the human‐guided data $\mathcal{D}$ and  
$$Z(\theta)=\sum_{\tau'}\exp\bigl(\sum_t r_\theta(s_t',a_t';z_t')\bigr)$$  
is the partition function. In practice, we approximate gradients via importance sampling or guided cost learning; see Algorithm 1.  

Algorithm 1: Intrinsic Reward Inference  
Input: Dataset $\mathcal{D}$ of human interactions, encoder $E_\phi$  
Initialize $\theta_0$ randomly  
for iteration $k=1\ldots K$ do  
  1. Sample minibatch of trajectories $\{\tau^i\}$ from $\mathcal{D}$  
  2. For each trajectory, compute embeddings $\{z_t^i\}$ via $E_\phi$  
  3. Compute reward sequences $r_t^i = r_\theta(s_t^i,a_t^i;z_t^i)$  
  4. Estimate gradient:  
     $$\nabla_\theta \;=\;\mathbb{E}_{\tau\sim\mathcal{D}}\bigl[\nabla_\theta\sum_t r_\theta(s_t,a_t;z_t)\bigr]\;-\;\mathbb{E}_{\tau'\sim\pi_\theta}\bigl[\nabla_\theta\sum_t r_\theta(s_t',a_t';z_t')\bigr]$$  
  5. Update $\theta_{k+1} = \theta_k + \eta\,\nabla_\theta$  
end for  

D. Meta‐Reinforcement Learning for Non‐Stationary Adaptation  
Human preferences and contexts change over time. We employ a Model‐Agnostic Meta‐Learning (MAML) style procedure [Finn et al.] to learn initialization parameters $\psi$ of both the encoder $\phi$ and IRL module $\theta$ that quickly adapt on a user‐specific or session‐specific dataset $\mathcal{D}_i$.  

Inner Loop (Adaptation on task $i$):  
$$\theta_i' = \theta - \alpha\nabla_\theta \mathcal{L}_{\text{IRL}}(\theta, \mathcal{D}_i),\quad 
  \phi_i' = \phi - \alpha\nabla_\phi \mathcal{L}_{\text{NCE}}(\phi, \mathcal{D}_i).$$  

Outer Loop (Meta‐optimization across tasks):  
$$\min_{\psi=\{\theta,\phi\}}\sum_{i=1}^M \mathcal{L}_{\text{RL}}\bigl(\theta_i',\phi_i';\mathcal{E}_i\bigr),$$  
where $\mathcal{L}_{\text{RL}}$ is the expected negative return under the adapted policy in environment $\mathcal{E}_i$ for user $i$.  

E. Experimental Design  
Environments:  
• Simulated grid‐world tutoring tasks with synthetic users exhibiting pre‐defined gaze/gesture patterns.  
• Real‐world human‐robot tutoring in a block‐stacking task with 20 participants.  

Baselines:  
1. Standard RL with hand‐crafted reward.  
2. RLHF using explicit user ratings at each step.  
3. Prior multimodal RLHF without meta‐adaptation.  

Evaluation Metrics:  
• Cumulative true reward (when available) and predicted intrinsic reward correlation: Pearson’s $r$.  
• Sample efficiency: interactions needed to reach a performance threshold.  
• Adaptation speed: number of gradient steps until user satisfaction plateaus.  
• User alignment score: a questionnaire rating how well the agent “understood” the user on a 5-point Likert scale.  
• Generalization: performance on held‐out tasks or with new users.  

Statistical Analysis  
Apply repeated‐measures ANOVA to compare methods across metrics, with post-hoc Tukey tests for pairwise differences (significance level $p<0.05$).  

4. Expected Outcomes & Impact  
Expected Outcomes  
• A validated algorithmic framework that reliably maps multimodal implicit feedback to an intrinsic reward function without any hand‐crafted semantics.  
• Meta-learned encoders and IRL modules that adapt within a few dozen interactions to new users and evolving preferences.  
• Empirical evidence of faster learning and higher alignment scores compared to standard RLHF and explicit‐feedback baselines.  
• An open‐source multimodal human‐agent interaction dataset and reference implementation.  

Broader Impact  
Our approach will advance socially aware robotics and personalized tutoring by:  
• Reducing reliance on costly, explicit feedback annotation.  
• Enabling assistive systems in healthcare (e.g., rehabilitation robots interpreting patient discomfort) and education (e.g., tutors that sense confusion via gaze).  
• Informing HCI design by quantifying how different implicit signals contribute to human–machine alignment.  
• Providing methodological foundations for inclusive interfaces that adapt to diverse users, including those with disabilities.  

5. References  
[1] Abramson et al. “Improving Multimodal Interactive Agents with Reinforcement Learning from Human Feedback.” arXiv:2211.11602, 2022.  
[2] Lee, Smith, Abbeel. “PEBBLE: Feedback‐Efficient Interactive RL via Relabeling & Unsupervised Pre‐training.” arXiv:2106.05091, 2021.  
[3] Xu et al. “Accelerating RL Agent with EEG‐based Implicit Human Feedback.” arXiv:2006.16498, 2020.  
[4] DeepMind Team et al. “Creating Multimodal Interactive Agents with Imitation and Self‐Supervised Learning.” arXiv:2112.03763, 2021.  
[5] Finn, Abbeel, Levine. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML, 2017.  

This proposal outlines a comprehensive plan to harness implicit, multimodal human feedback for learning intrinsic rewards, offering a pathway toward socially aligned, rapidly adapting AI agents.