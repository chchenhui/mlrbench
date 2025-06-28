Title: Self-Generating Adaptive Curricula for Open-Ended Reinforcement Learning

1. Introduction  
Background  
Open-ended learning (OEL) seeks to mirror the never-ending stream of challenges faced by humans and biological organisms by continuously generating novel tasks that drive the acquisition of increasingly general capabilities. Recent advances in deep reinforcement learning (RL) have produced agents that excel at fixed tasks, but once mastery is reached, learning typically plateaus. In parallel, large language models (LLMs) have demonstrated powerful generative and reasoning abilities that can be harnessed to automate aspects of training design. Integrating LLMs as meta-controllers for curriculum generation can unlock truly open-ended RL systems, where the agent’s own behavior informs the generation of fresh challenges, closing the loop between agent and environment.  

Research Objectives  
1. To develop a closed-loop framework in which an LLM consumes the RL agent’s performance data and failure modes, then produces new task specifications automatically.  
2. To formalize a quality-diversity filtering mechanism that selects high-impact, diverse tasks and prevents curriculum collapse.  
3. To design and implement open-ended difficulty metrics (e.g., ODD-score) that quantify novelty and challenge for continual tracking.  
4. To validate the approach across simulated benchmarks and real-world sim-to-real transfer, comparing against existing unmanned curricula methods (CurricuLLM (Ryu et al., 2024), UED (Jiang, 2023)).

Significance  
A successful demonstration of self-generating curricula would constitute a major step toward lifelong, robustly general agents. By automating the discovery of edge-of-capability tasks, we can reduce human engineering effort, improve out-of-distribution generalization, and accelerate sim-to-real deployment in robotics, autonomous systems, and interactive LLM agents.

2. Methodology  
2.1 Framework Overview  
We propose the Self-Generating Adaptive Curriculum (SGAC) framework, comprising three modules: (i) RL agent training, (ii) failure mode extraction, and (iii) LLM-based task generation with quality-diversity filtering. This loop repeats for $T$ iterations, yielding an ever-expanding curriculum.

2.2 Notation  
– $\pi_{\theta}$: agent policy parameterized by $\theta$.  
– $\mathcal{E}$: environment generator mapping task specification $z$ to a Markov Decision Process $\mathcal{M}(z)$.  
– $\mathcal{D}_{t}$: dataset of trajectories collected up to iteration $t$.  
– $Z_{t}$: set of task specifications seen so far.  

2.3 RL Agent & Data Collection  
At iteration $t$, the agent interacts with a batch of environments $\{\mathcal{M}(z)\mid z\in B_{t}\}$, where $B_{t}\subset Z_{t}$ is the current active curriculum. We employ a standard off-policy or on-policy RL algorithm (e.g., SAC or PPO) to update
$$
\theta_{t+1} = \arg\max_{\theta}\;\mathbb{E}_{\tau\sim\pi_{\theta},\,z\in B_{t}}\Bigl[\sum_{i=0}^{H}\gamma^i r(s_i,a_i)\Bigr].
$$
Trajectories $\tau=(s_0,a_0,\dots,s_H,a_H)$ and per-episode returns $R(\tau)$ are appended to $\mathcal{D}_{t+1}$.

2.4 Failure Mode Extraction  
From $\mathcal{D}_{t+1}$, we identify “skill gaps” by clustering low-return trajectories or states where the agent’s value function $V_{\theta_t}(s)$ is below a threshold. We summarize each failure mode as a feature vector $f\in\mathbb{R}^d$, capturing state clusters, action missteps, and subgoal failures.  

2.5 LLM-Driven Task Generation  
An LLM (e.g., GPT-4) is prompted with a structured description of failure modes $\{f_k\}$, along with agent performance summaries. A template prompt encourages the LLM to propose $N$ new task specifications $\{z_i\}_{i=1}^N$ expressed as parameterized environment descriptors (e.g., obstacle counts, reward shaping rules). These are immediately parsed by the environment generator $\mathcal{E}$ into simulable MDPs.

2.6 Quality-Diversity Filtering  
To avoid generating redundant or trivial tasks, we compute for each candidate $z_i$:
  • Difficulty  
    $$\mathrm{Diff}(z_i)=1 - \frac{\mathbb{E}_{\tau\sim\pi_{\theta_t}}\bigl[R(\tau\mid z_i)\bigr]}{R_{\max}(z_i)},$$  
    where $R_{\max}(z_i)$ is an estimate of the maximal achievable return.  
  • Novelty  
    $$\mathrm{Nov}(z_i)=\min_{z_j \in Z_{t}}\lVert \phi(z_i)-\phi(z_j)\rVert,$$  
    where $\phi(\cdot)$ is a task-embedding network (e.g., a small autoencoder trained on task parameters).  
We define an ODD-score  
$$
\mathrm{ODD}(z_i)=\lambda\,\mathrm{Diff}(z_i)\;+\;(1-\lambda)\,\mathrm{Nov}(z_i),
$$  
and select the top $K$ tasks by ODD to form $B_{t+1}$. We then update $Z_{t+1}\leftarrow Z_{t}\cup B_{t+1}$.

2.7 Algorithm Pseudocode  
```  
Initialize θ0, Z0←{z₀…zM} (basic seed tasks), D0←∅  
for t=0…T−1 do  
  Collect trajectories from environments in Bt using πθt  
  Update πθt→θt+1 via RL algorithm  
  Extract failure modes {fk} from Dt+1  
  Prompt LLM with {fk}, performance stats → generate {z_i}N i=1  
  For each z_i compute Nov(z_i), Diff(z_i), ODD(z_i)  
  Select top-K by ODD→Bt+1, Zt+1←Zt∪Bt+1  
end for  
```  

2.8 Experimental Design  
Benchmarks  
– ProcGen (Brockman et al., 2019) for procedurally-generated navigation tasks.  
– Obstacle Tower (Propose et al., 2022) for hierarchical challenge.  
– MuJoCo-based robotics manipulation (via Robosuite).  
Baselines  
1. Static uniform sampling of tasks  
2. CurricuLLM (Ryu et al., 2024)  
3. Unsupervised Environment Design (Jiang, 2023)  
4. Ablations of SGAC: (a) no QD filter, (b) random LLM prompts.  
Metrics  
– Average return on held-out unseen tasks  
– ODD-score progression over iterations  
– Coverage: fraction of discovered task-embeddings  
– Sim2Real Transfer Success Rate (for manipulation)  
– Sample efficiency: environment steps to reach performance thresholds  
Evaluation Protocol  
– Each method is run with 5 random seeds.  
– Statistical tests (t-test, p<0.05) on primary metrics.  
– Qualitative analysis of emergent task complexity.

3. Expected Outcomes & Impact  
3.1 Anticipated Results  
– Demonstration that SGAC sustains non-trivial learning far beyond fixed curricula, as measured by ODD-score growth and coverage metrics.  
– Superior generalization to held-out tasks compared to static and prior automated curricula approaches.  
– Improved sim-to-real transfer in robotic benchmarks, with fewer fine-tuning steps required.  
– Ablation studies validating the crucial role of quality-diversity filtering and LLM-driven task creativity.  

3.2 Broader Impact  
Scientific  
This work will advance the theory and practice of open-ended learning by providing a concrete instantiation of an LLM-in-the-loop curriculum generator. The proposed ODD metrics and QD filtering mechanism will serve as foundational tools for future OEL research.  
Societal  
Agents trained via SGAC hold promise for lifelong adaptation in dynamic real-world domains: household robots that continually self-improve, game agents that evolve with player strategies, and adaptive network controllers in telecommunication. Automated curricula can reduce human engineering effort and democratize the creation of generalist agents.  
Ethical Considerations  
We will audit generated tasks for safety and appropriateness, incorporate human-in-the-loop oversight for high-risk domains (e.g., autonomous vehicles), and ensure that LLM prompts do not produce malicious or biased challenges.  
Long-Term Vision  
By validating self-generating curricula, this project lays the groundwork for truly open-ended machine intelligence. Future extensions may incorporate multi-agent co-evolution, richer world simulators, and fully emergent environments where both tasks and evaluation criteria evolve without human specification.