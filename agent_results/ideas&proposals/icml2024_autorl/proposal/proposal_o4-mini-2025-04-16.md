Title  
HyperPrompt: Dynamic Hyperparameter Adaptation in Reinforcement Learning via LLM-Based Meta-Learning  

Introduction  
Background  
Reinforcement learning (RL) has achieved remarkable successes in domains such as game playing, robotics, and automated control. Yet practical deployment of RL remains challenging: algorithm performance is highly sensitive to choices of hyperparameters (e.g., learning rates, exploration coefficients, discount factors), and optimal settings often shift over the course of training. Static tuning methods—grid search, random search, Bayesian optimization—optimize for fixed configurations, ignoring that the “hyperparameter landscape” dynamically evolves as the agent learns [1]. This brittleness restricts RL’s applicability to novel tasks and inflates computation costs, as repeated offline tuning is expensive.

Parallel work in AutoML has advanced hyperparameter optimization (HPO) frameworks—OptFormer, ARLBench—that automate offline tuning for RL [2,3]. Meanwhile, meta-reinforcement learning (Meta-RL) devises algorithms that adapt at test time by leveraging experience from prior tasks. More recently, large language models (LLMs) have exhibited compelling in-context meta-learning capabilities: when prompted with demonstrations, they generalize to unseen tasks without parameter updates [4]. These in-context abilities suggest a new hybrid: treat LLMs as meta-learners that, when fed RL trajectories and performance signals, predict context-aware hyperparameter updates online.

Research Objectives  
1. Develop HyperPrompt, a framework that uses a pre-trained LLM as a meta-policy to output dynamic hyperparameter schedules conditioned on streaming RL feedback.  
2. Formalize hyperparameter adaptation as a meta-Markov decision process (meta-MDP) and derive training objectives for LLM-based hyperparameter controllers.  
3. Evaluate HyperPrompt on procedurally generated, diverse RL benchmarks (NetHack, Procgen, MuJoCo), comparing sample efficiency, final performance, and compute overhead against static tuning, OptFormer [2], and ARLBench [3].  

Significance  
HyperPrompt aims to (a) reduce manual and compute burdens of RL hyperparameter tuning; (b) improve robustness and generalization to unseen environments; and (c) create a unified AutoRL pipeline combining LLM meta-learning with online adaptation. Success will democratize RL deployment and accelerate future AutoRL research by demonstrating the first real‐time, LLM-driven hyperparameter controller.  

Methodology  
Overall Framework  
We cast hyperparameter adaptation as a meta-MDP $\mathcal{M}_{meta} = (\mathcal{S}_{meta}, \mathcal{A}_{meta}, P_{meta}, R_{meta}, \gamma_{meta})$, where at each RL training step $t$:  
– Meta-state $s_t^{meta}$ comprises a window of recent transitions and hyperparameter history:  
$$
s_t^{meta} = \{(o_{t-\tau}, a_{t-\tau}, r_{t-\tau}, \theta_{t-\tau})\}_{\tau=1}^{L},
$$  
with $o_t$ the RL observation, $a_t$ the agent’s action, $r_t$ the reward, and $\theta_t$ the hyperparameter vector at step $t$.  
– Meta-action $a_t^{meta} = \Delta \theta_t$ is the update to hyperparameters.  
– The transition kernel $P_{meta}$ is implicit: the next meta-state aggregates new transitions as the base RL agent evolves under updated $\theta_{t+1}$.  
– Meta-reward $R_{meta}(s_t^{meta}, a_t^{meta})$ reflects improvements in RL performance; for example, the change in running average return over a fixed horizon:  
$$
R_{meta} = \bar R_{t+\Delta} - \bar R_{t},
$$  
where $\bar R_{t} = \frac{1}{K}\sum_{k=0}^{K-1} r_{t-k}$.  

HyperPrompt uses an LLM parameterized by $\phi$ as an approximate meta-policy $\pi_{\phi}(a_t^{meta}\mid s_t^{meta})$. We will implement two phases: meta-training (offline) and online deployment (adaptation).

1. Meta-Training  
Data Collection  
– Sample a diverse suite of source tasks $\{\mathcal{T}_i\}$ drawn from procedurally generated environments (Procgen, NetHack, varied MuJoCo morphologies).  
– For each task, run baseline tuning (e.g., Bayesian optimization or OptFormer) to collect transition‐level tuples $(s_t^{meta}, \theta_t, \Delta \theta_t, R_{meta})$ over multiple seeds.  
– Construct a dataset $\mathcal D = \{(\mathbf{P}_j, \Delta \theta_j)\}$, where prompt $\mathbf{P}_j$ encodes the concatenated string of:  
   • Recent trajectory tokens (observations, actions, rewards).  
   • Hyperparameter values at each step.  
   • Performance summary statistics.  

Model Fine-Tuning  
We fine-tune a base LLM (e.g., GPT-2 Medium) on $\mathcal D$ in a supervised manner: given prompt $\mathbf{P}_j$, predict $\Delta \theta_j$. The loss is cross-entropy over discretized hyperparameter updates or mean squared error if using continuous outputs:  
$$
\mathcal L_{SL}(\phi) = \frac{1}{|\mathcal D|}\sum_{j}\|\pi_{\phi}(\mathbf{P}_j)-\Delta \theta_j\|^2_{2}.
$$  

Meta-RL Fine-Tuning  
To further align the LLM’s suggestions with long-term performance, we apply a policy gradient objective on held-out validation tasks. Roll out the RL agent that uses $\pi_{\phi}$ to update hyperparameters, collect meta-rewards $R_{meta}$, and optimize:  
$$
J(\phi) = \mathbb{E}_{\tau\sim\mathcal M_{meta}} \Bigl[\sum_{t}\gamma_{meta}^t R_{meta}(s_t^{meta}, a_t^{meta})\Bigr].
$$  
The gradient is estimated via REINFORCE with baseline $b(s_t^{meta})$:  
$$
\nabla_{\phi}J = \mathbb{E}\bigl[\sum_{t}\nabla_{\phi}\log\pi_{\phi}(a_t^{meta}\mid s_t^{meta})\bigl(R_{meta} - b(s_t^{meta})\bigr)\bigr].
$$  

2. Online Deployment  
At training time on a novel target task, we initialize hyperparameters $\theta_0$ to default values. Every $U$ RL steps, we:  
a. Extract recent window $s_t^{meta}$ and serialize into prompt $\mathbf{P}$.  
b. Query the fine-tuned LLM to obtain $\Delta\theta_t = \pi_{\phi}(\mathbf{P})$.  
c. Update $\theta_{t+1} = \theta_t + \Delta \theta_t$.  
d. Continue RL training with updated $\theta_{t+1}$.  

Algorithmic Summary  
Algorithm 1: Meta-Training  
1. Collect $\mathcal D$ by offline tuning.  
2. Supervised fine-tune LLM on $\mathcal D$.  
3. Apply meta-RL fine-tuning on validation tasks.  
4. Save final parameters $\phi^*$.  

Algorithm 2: Online Adaptation  
1. Initialize $\theta_0$.  
2. For $t=0\ldots T$:  
   a. Run base RL agent for $U$ steps under $\theta_t$.  
   b. Build prompt $\mathbf{P}_t$ from recent trajectories.  
   c. Query $\Delta \theta_t = \mathrm{LLM}_{\phi^*}(\mathbf{P}_t)$.  
   d. $\theta_{t+1}\leftarrow \theta_t+\Delta\theta_t$.  

Experimental Design  
Benchmarks & Baselines  
– Environments: Procgen (10 games), NetHack Challenge, MuJoCo Ant-Morphology tasks.  
– Baselines:  
   • Static default hyperparameters.  
   • Offline Bayesian Optimization (BO) per task.  
   • OptFormer [2].  
   • ARLBench standardized HPO [3].  

Metrics  
1. Sample Efficiency: number of environment interactions required to reach a target return $R^*$.  
2. Learning Stability: variance of returns across seeds.  
3. Final Performance: mean episode return at training horizon $T$.  
4. Computational Overhead: wall-clock time spent on hyperparameter adaptation (inference time of LLM).  
We report area under the learning curve (AUC):  
$$
\mathrm{AUC} = \int_{0}^{T}\bar R_t\,dt,
$$  
and normalized performance gains over baselines.

Ablations  
– Vary LLM size (GPT-2 Small, Medium, Large).  
– Compare supervised-only vs meta-RL fine-tuning.  
– Window length $L$ and update interval $U$.  
– Zero-shot vs few-shot prompts (injecting expert tuning demonstrations).  

Implementation Details  
– LLM trained with AdamW, learning rate $5\mathrm{e}{-5}$, batch size 8 prompts, for 50 epochs.  
– Meta-RL fine-tuning uses PPO with clip 0.2, $\gamma_{meta}=0.99$, learning rate $3\mathrm{e}{-6}$.  
– RL agents use PPO for discrete tasks and SAC for continuous benchmarks.  
– Experiments run on 4 NVIDIA A100 GPUs; compute budget capped at 50k GPU hours.  

Expected Outcomes & Impact  
We anticipate HyperPrompt to:  
1. Achieve significant reductions (30–50%) in sample complexity compared to static and offline-tuned baselines, demonstrating real-time hyperparameter agility.  
2. Exhibit robust generalization: extended to unseen procedurally generated tasks without additional fine-tuning.  
3. Outperform OptFormer and ARLBench by leveraging in-context meta-learning to adapt continually, not just offline.  
4. Scale gracefully: moderate LLM sizes (150M–350M parameters) should suffice, keeping inference costs manageable.

Broader Impact  
HyperPrompt will lower barriers for RL practitioners lacking large compute resources or expert HPO knowledge. By embedding the meta-tuning logic in an LLM, users can apply RL “out-of-the-box” to new domains (robotics, finance, healthcare simulators) with minimal manual intervention. The framework also demonstrates a novel synergy between LLM in-context learning and AutoML, opening avenues for LLM-driven AutoRL components (policy discovery, curriculum design).  

Potential Risks & Mitigations  
– Overfitting to source tasks: mitigate via diverse environment sampling and regularization during meta-RL.  
– Inference latency: constrain prompt length and employ model distillation for a lightweight controller.  
– Safety: ensure hyperparameter suggestions remain within safe bounds by clipping $\Delta\theta$.  

References  
[1] Mohan et al., “AutoRL Hyperparameter Landscapes,” arXiv:2304.02396, 2023.  
[2] Eimer et al., “Hyperparameters in RL and How To Tune Them,” arXiv:2306.01324, 2023.  
[3] Becktepe et al., “ARLBench: Benchmarking for HPO in RL,” arXiv:2409.18827, 2024.  
[4] Wan et al., “ReMA: Learning to Meta-think for LLMs,” arXiv:2503.09501, 2025.