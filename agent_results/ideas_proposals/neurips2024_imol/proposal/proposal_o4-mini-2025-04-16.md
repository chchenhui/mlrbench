Title  
Adaptive Contextual Goal Generation for Lifelong Hierarchical Intrinsic Motivation  

1. Introduction  
1.1 Background  
Humans and animals effortlessly acquire broad repertoires of skills over their lifetimes without explicit external rewards, guided instead by curiosity and intrinsic drives. Computational models of intrinsically motivated learning (IM) (Oudeyer et al., 2007; Barto, 2013; Schmidhuber, 2021) have demonstrated how agents can self-generate learning signals—such as prediction errors or information gains—to explore and master complex environments. Hierarchical reinforcement learning (HRL) further decomposes tasks into sub-tasks or “options” (Sutton et al., 1999), enabling temporal abstraction and skill reuse. Recent works—h-DQN (Kulkarni et al., 2016), HIDIO (Zhang et al., 2021), and self-play based sub-goal embedding (Sukhbaatar et al., 2018)—combine hierarchy and IM to improve exploration in sparse-reward domains.  

Despite these advances, existing IM-based HRL agents face four critical shortcomings:  
• Rigid goal spaces. Pre-defined or statically learned goal sets fail to adapt when environmental dynamics change.  
• Exploration–exploitation imbalance. Agents either over-explore low-value regions or prematurely exploit learned behaviors, limiting open-ended skill growth.  
• Skill retention and transfer. Learned options can degrade (“catastrophic forgetting”) or prove hard to compose for novel tasks.  
• Scalability. Hierarchies often become unwieldy as task complexity grows, impeding training stability.  

1.2 Research Objectives  
We propose to address these challenges by developing a Hierarchical Intrinsic Motivation (HIM) framework with **Adaptive Contextual Goal Generation (ACGG)**. The key objectives are:  
(1) Dynamically adapt intrinsic goals to evolving environmental contexts.  
(2) Balance exploration and exploitation over long horizons using meta-level learning progress signals.  
(3) Retain and reuse learned skills via a compositional skill library and few-shot transfer.  
(4) Scale gracefully by employing attention-based context encoding and modular policy architectures.  

1.3 Significance  
By endowing agents with the ability to self-assess when to explore new behaviors versus refining existing skills—based on environmental cues—ACGG will push toward truly open-ended lifelong learning. Such autonomy is crucial for real-world deployment (e.g., household robots, scientific discovery assistants) where hand-crafted reward schedules are infeasible. Moreover, our integration of meta-learning, attention-based context modeling, and skill compositionality stands to advance both the theoretical understanding of intrinsically motivated HRL and its practical capabilities.  

2. Methodology  
We detail the ACGG framework along four components: hierarchical architecture, intrinsic reward design, contextual goal generator, and skill library with few-shot transfer.  

2.1 Hierarchical Architecture  
We adopt a two-level hierarchy:  
• **Meta-level policy** $\pi_\psi(g\,|\,c)$ selects high-level goals $g$ based on a context encoding $c$. Goals can be abstract descriptors (e.g., “map new region,” “refine grasp skill”).  
• **Skill (lower-level) policies** $\pi_\theta(a\,|\,s,g)$ execute actions $a$ in state $s$ to attain the current goal $g$.  

Every $H$ environment steps, the meta-policy issues a new goal $g_t$. The skill policy is conditioned on $g_t$ until completion or timeout.  

2.2 Intrinsic Reward and Learning Progress  
Each skill policy receives an intrinsic reward motivated by prediction error on a learned forward model $f_\phi$:  
$$
r^i_t \;=\;\|\,s_{t+1} - f_\phi(s_t,a_t)\|_2^2\,.  
$$  
We track **learning progress** for goal $g$ as the absolute change in prediction error over a sliding window:  
$$
LP_t(g)\;=\;\bigl|\,E_{t}(g)\;-\;E_{t-\Delta}(g)\bigr|,\quad
E_t(g)=\frac1\Delta\sum_{k=t-\Delta}^{t-1}r^i_k\,.  
$$  
The meta-level reward for choosing $g$ is the cumulative learning progress over the next $H$ steps:  
$$
R^M_t \;=\;\sum_{t'=t}^{t+H-1}LP_{t'}(g)\,.  
$$  

2.3 Contextual Goal Generation  
We design the meta-policy $\pi_\psi$ to condition on a context vector $c_t$ that summarizes recent environmental statistics:  
$$
c_t \;=\;\mathrm{Enc}_\omega\bigl(\{s_{t-K},\dots,s_t\},\{r^e_{t-K},\dots,r^e_{t-1}\}\bigr)\,,
$$  
where $r^e$ includes any sparse extrinsic rewards or collision signals. $\mathrm{Enc}_\omega$ is a small Transformer or LSTM with attention to highlight novel or unpredictable sensory features. Goals are sampled from a Gaussian policy:  
$$
g_t\;\sim\;\mathcal{N}\bigl(\mu_\psi(c_t),\,\Sigma_\psi(c_t)\bigr)\,.  
$$  
The meta-policy is trained via policy gradient (e.g., PPO) to maximize discounted cumulative $R^M_t$:  
$$
\nabla_\psi J(\psi)=\mathbb{E}\Bigl[\sum_t\nabla_\psi\log\pi_\psi(g_t|c_t)\bigl(G_t^M-b(c_t)\bigr)\Bigr]\,.  
$$  

2.4 Skill Library and Few-Shot Transfer  
We maintain a growing library $\mathcal{L}=\{(g_i,\theta_i)\}$ of past goals and their optimized skill parameters. To accelerate adaptation to a new goal $g$, we compute attention weights over $\mathcal{L}$:  
$$
w_i = \mathrm{softmax}\bigl(\cos\bigl(h_\chi(g),\,k_i\bigr)\bigr)\,,\quad
\theta_{\mathrm{init}} = \sum_i w_i\theta_i\,,
$$  
where $h_\chi$ and $k_i$ are learned embeddings of goals. The skill policy for a novel goal is initialized at $\theta_{\mathrm{init}}$ and fine-tuned with intrinsic rewards, enabling few-shot transfer.  

2.5 Training Algorithm  
Algorithm 1 outlines the joint training loop.  

Algorithm 1: ACGG Joint Training  
1. Initialize parameters $\psi,\theta,\phi,\omega,\chi$; empty library $\mathcal{L}$.  
2. for iteration = 1 to $N$ do  
3.   Collect an episode:  
4.     Observe state $s_t$; compute context $c_t$; sample $g_t\sim\pi_\psi(g|c_t)$.  
5.     for $h=0$ to $H-1$ do  
6.       Sample action $a_{t+h}\sim\pi_\theta(a\,|\,s_{t+h},g_t)$.  
7.       Observe $s_{t+h+1}$; compute $r^i_{t+h}$ and update forward model $f_\phi$.  
8.     end for  
9.   Compute meta-reward $R^M_t=\sum_{k=0}^{H-1}LP_{t+k}(g_t)$.  
10.  Store $(c_t,g_t,R^M_t)$ in meta replay.  
11.  Update $\psi$ via PPO on meta replay.  
12.  Update $\theta$ via intrinsic-reward actor-critic.  
13.  If performance on $g_t$ > threshold, add $(g_t,\theta)$ to $\mathcal{L}$.  
14. end for  

2.6 Experimental Design  
Environments: We will evaluate on two suites:  
• **Procedural 3D Navigation** (random maze layouts in Habitat or MiniGrid).  
• **Multi-Object Manipulation** (MuJoCo or Robosuite tasks with movable blocks).  

Baselines:  
– Static-Goal HRL (h-DQN style)  
– HIDIO without contextual adaptation  
– Random goal sampling  
– Non-hierarchical curiosity-driven RL (ICM, RND)  

Metrics:  
1. **Task Coverage**: proportion of distinct sub-regions or object configurations mastered.  
2. **Adaptation Speed**: episodes to reach a performance threshold on unseen tasks.  
3. **Skill Reusability**: success rate when composing library skills to solve novel tasks.  
4. **Sample Efficiency**: environment steps to achieve specified coverage.  
5. **Computation Overhead**: runtime per training iteration.  

We will conduct ablation studies by toggling: (i) context encoder, (ii) library reuse, (iii) learning progress vs. pure prediction error. Statistical significance will be assessed via paired t-tests across random seeds.  

3. Expected Outcomes & Impact  
3.1 Expected Outcomes  
• **Adaptive Goal Selection**: Agents will automatically bias goal sampling toward either exploration (high uncertainty) or exploitation (skill refinement) depending on environment richness.  
• **Improved Coverage and Speed**: Compared to static baselines, ACGG is expected to achieve ≥ 20 % higher task coverage and 30 % faster adaptation on held-out mazes and manipulation tasks.  
• **High Skill Reusability**: Few-shot transfer via the skill library should yield at least 50 % reduction in fine-tuning steps for novel goals.  
• **Scalable Hierarchy**: Attention-based context encoding will allow the meta-policy’s complexity to grow sub-linearly with environmental variability.  

3.2 Broader Impacts  
• **Autonomy in Real World Systems**: ACGG paves the way for robots that self-drive their learning curriculum, reducing the need for hand-tuned reward engineering.  
• **Cross-Disciplinary Insights**: The proposed context encoding and progress-driven meta-learning resonate with cognitive theories of curiosity and developmental stages, fostering dialogue between AI, psychology, and neuroscience.  
• **Open-Ended AI Safety**: Understanding how to constrain intrinsic drives without stifling autonomy may inform safer AGI development.  
• **Community Resources**: We will release code, pretrained skill libraries, and procedural environment generators to accelerate future IMOL research.  

In summary, our ACGG framework addresses key challenges in intrinsically motivated open-ended learning by dynamically generating context-aware goals, balancing exploration and exploitation, and leveraging lifelong skill accumulation. We anticipate that these innovations will significantly advance both the theory and practice of autonomous lifelong learning systems.