1. Title  
Dynamic Curriculum Benchmark for Emergent Planning and Theory‐of‐Mind in Large Language Models

2. Introduction  
Background  
Large language models (LLMs) such as GPT‐4, LLaMA‐2, and PaLM have exhibited remarkable “emergent” capabilities in tasks far beyond next‐token prediction—ranging from multi‐step reasoning to rudimentary theory‐of‐mind (ToM) inference. However, current evaluation suites remain largely static: they present a fixed set of problems at predetermined difficulty levels. Static benchmarks cannot adapt to an LLM’s evolving skill profile, nor can they precisely locate the point at which planning or ToM abilities “turn on.” Without adaptive evaluation, comparing models of different architectures or data regimes is akin to racing on uneven terrain.

Research Objectives  
We propose to design, implement, and validate a Dynamic Curriculum Benchmark (DCB) for LLMs that:  
• Algorithmically generates sequences of planning, navigation, and ToM tasks whose difficulty automatically scales to an LLM’s performance.  
• Quantifies emergence thresholds for each cognitive skill by tracing performance trajectories over difficulty levels.  
• Compares end‐to‐end fine‐tuned LLMs with modular, augmented LLMs (e.g., LLM+planner or LLM+external memory).  
• Incorporates human‐in‐the‐loop auditing to ensure scoring fidelity and catch pathological behaviors.

Significance  
A dynamic, data‐driven benchmark will provide:  
• Fine‐grained cognitive profiles that pinpoint when and how models acquire higher‐order reasoning.  
• A level playing field for comparing architectures, training regimes, and multimodal extensions.  
• Actionable guidance for model designers seeking to push LLMs beyond surface‐level language modelling into robust planning and social cognition.  

3. Related Work  
Hypothetical Minds [Cross et al., 2024] introduces a modular LLM agent architecture with explicit ToM hypotheses in a multi‐agent RL environment. Their hierarchical planner improves coordination but relies on a static set of tasks.  
CogGPT and CogBench [Lv et al., 2024] propose an iterative cognitive mechanism and a bespoke benchmark for lifelong dynamics. CogBench remains fixed in its task progression and does not adapt to model‐specific strengths or weaknesses.  
Theory of Mind for Multi‐Agent Collaboration [Li et al., 2023] compares LLM agents to RL baselines in text‐game ToM tasks, revealing that explicit belief representations can boost accuracy but long‐horizon contexts degrade performance. Their evaluation, however, is limited to a handful of difficulty tiers.  
Emergent Response Planning in LLMs [Dong et al., 2025] probes LLM hidden states to show latent planning signals. While illuminating, their static probing does not translate into an adaptive evaluation protocol.  

Key challenges remain: (1) building an adaptive curriculum that tracks an LLM’s progression in planning and ToM, (2) identifying sharp emergence points in these abilities, (3) managing long contexts without coherence collapse, and (4) validating automated scoring with human auditors.

4. Methodology  
4.1 Overview of the Dynamic Curriculum Benchmark (DCB)  
DCB is a closed‐loop evaluation system in which an LLM’s success on recent tasks informs the generation of subsequent, appropriately harder or easier tasks. We organize tasks into three domains: Planning, Navigation, and Theory of Mind. Each domain has a continuous difficulty parameter $d\in[0,1]$.  

4.2 Task Generation and Difficulty Parameterization  
• Planning Tasks: Puzzles requiring an LLM to propose a sequence of $k$ steps to achieve a goal. Difficulty $d$ controls $k$ (the plan length), branching factor of action space, and presence of distractor goals.  
• Navigation Prompts: First‐person scenarios in a grid‐world described textually. Difficulty $d$ scales grid size, number of obstacles, and multi‐agent interactions.  
• ToM Scenarios: Short stories with one or more agents harboring private beliefs. Difficulty $d$ adjusts chain length of nested beliefs (“Alice thinks that Bob believes that …”) and ambiguity in agent goals.

Each task $T$ is represented by a parameter vector $(d, \tau)$ where $d$ is difficulty and $\tau$ selects one of the three domains. A template library and procedural generators ensure diversity.

4.3 Curriculum Sampler via Multi‐Armed Bandit  
We model task selection as a non‐stationary multi‐armed bandit over discretized difficulty bins $\{d_i\}_{i=1}^N$. Let $Q_i(t)$ be the estimated success rate for bin $i$ at time $t$, and $N_i(t)$ the count of trials. We choose bin $i_t$ by maximizing an Upper Confidence Bound:  
$$
i_t = \arg\max_{i}\Bigl[\,Q_i(t) + c\sqrt{\frac{\ln\bigl(\sum_j N_j(t)\bigr)}{N_i(t)}}\Bigr],
$$  
where $c>0$ balances exploration vs. exploitation. Once $i_t$ is chosen, sample a random task $T$ from $\tau$‐domain at difficulty $d_i$.  

After the LLM’s response, we compute a binary reward $r_t\in\{0,1\}$—1 for fully correct planning steps, correct map navigation instructions, or accurate ToM inference. We update:  
$$
N_{i_t}(t+1)\leftarrow N_{i_t}(t)+1,\quad
Q_{i_t}(t+1)\leftarrow Q_{i_t}(t) + \frac{r_t - Q_{i_t}(t)}{N_{i_t}(t+1)}.
$$

4.4 Emergence Point Estimation  
For each domain $\tau$, define the emergence difficulty  
$$
\hat d_\tau = \min\Bigl\{d_i\;\Big|\;\frac{1}{|\mathcal{T}_{d_i}|}\sum_{t\in\mathcal{T}_{d_i}}r_t\ge\delta\Bigr\},
$$  
where $\delta$ is a success threshold (e.g.\ $\delta=0.75$) and $\mathcal{T}_{d_i}$ the set of trials at $d_i$. We record $(\hat d_\tau)$ as a signature of when the LLM “masters” domain $\tau$.

4.5 Human‐in‐the‐Loop Audit  
To catch edge‐case failures or hallucinations, we randomly sample $5\%$ of tasks for human review. Auditors verify correctness and assign a refinement score $h_t\in\{-1,0,1\}$ (−1 for false positive auto‐score, 0 for uncertain, +1 for correct). We adjust $Q_i$ as:  
$$
Q_{i_t}\leftarrow Q_{i_t} + \eta\,h_t,\quad \eta\ll1,
$$  
penalizing bins where auto‐scoring is unreliable.

4.6 Experimental Design  
We will evaluate three classes of models:  
1. Vanilla LLMs (GPT‐3.5, GPT‐4) prompted zero‐ and few‐shot.  
2. Fine‐tuned LLMs trained end‐to‐end on static planning/ToM corpora.  
3. Modular LLMs augmented with external planners or memory (e.g.\ Hypothetical Minds style).  

For each model, we run the DCB for a fixed budget of $T=2{,}000$ tasks. As a baseline, we also run a static benchmark matching the cumulative difficulty distribution seen by DCB.  

4.7 Evaluation Metrics  
• Success Rate Trajectory $s_\tau(d_i) = Q_i$ for each bin.  
• Emergence Difficulty $\hat d_\tau$.  
• Sample Efficiency: number of trials to reach $\delta$‐success.  
• Human‐Score Discrepancy: average $|h_t|$ over sampled tasks.  
• Cross‐Model Comparison: statistical tests (ANOVA) on $\hat d_\tau$ and sample efficiency.  

We will also fit a logistic performance curve  
$$
s_\tau(d)\approx \frac{1}{1+\exp(-\alpha_\tau(d-d_{0,\tau}))}
$$  
and compare parameters $(d_{0,\tau},\alpha_\tau)$ across models.

5. Expected Outcomes & Impact  
Expected Outcomes  
• A publicly released DCB suite with code, templates, and evaluation scripts.  
• Fine‐grained cognitive profiles $(\hat d_\tau,\alpha_\tau)$ for each model in Planning, Navigation, and ToM.  
• Demonstration of how dynamic curricula outperform static benchmarks in pinpointing emergent abilities and in sample efficiency.  
• A human‐verified error catalogue illustrating common hallucination patterns in each domain.

Impact  
This work will:  
• Establish a new standard for adaptive, difficulty‐aware evaluation of LLM cognitive skills.  
• Illuminate fundamental limits of current LLMs in long‐horizon planning and nested belief reasoning.  
• Inform the design of future model architectures (e.g.\ hybrid LLM+planner) by spotlighting precise emergence thresholds.  
• Provide the community with tools to benchmark not only language fluency but core cognitive competencies—bridging AI, cognitive science, and neuroscience.  

By quantifying when and how LLMs acquire higher‐order reasoning, DCB will catalyze research into architectures that more closely mirror human cognitive development and theory‐of‐mind acquisition.