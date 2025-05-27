# Title  
**Dynamic Curriculum Benchmark for Emergent Planning and Theory-of-Mind in Large Language Models**

# Introduction  
## Background  
Large Language Models (LLMs) have demonstrated remarkable proficiency in tasks requiring natural language understanding, dialogue generation, and even rudimentary reasoning. However, their capacity to acquire and exhibit higher-order cognitive abilities—such as *planning* and *Theory of Mind (ToM)*—remains poorly understood. Planning requires systematic reasoning across sequential steps to achieve a goal, while ToM involves inferring and reasoning about others' mental states (e.g., beliefs, intentions). These skills are hallmarks of human cognition and critical for applications in robotics, collaborative agents, and AI alignment. Despite evidence of emergent reasoning abilities in LLMs (Zhichen Dong et al., 2025), current evaluation frameworks rely on static, handcrafted benchmarks that cannot measure how cognitive skills evolve with task complexity or model scaling. This limitation hampers comparative analyses of LLM architectures and obstructs the development of models tailored for robust cognitive tasks.  

## Research Objectives  
This proposal aims to address three core challenges identified in recent literature:  
1. **Adaptive Benchmarking**: Static benchmarks, such as the Theory-of-Mind Challenge Set (CogGPT: Lv et al., 2024), fail to track incremental progression or identify phase transitions in cognitive capabilities. A dynamic framework is needed to scale tasks based on performance.  
2. **Emergent Behavior Identification**: Studies like *Hypothetical Minds* (Cross et al., 2024) highlight the unpredictability of ToM capabilities in LLMs, suggesting that emergence depends on parameter/data scaling. DCB seeks to formalize thresholds for such abilities.  
3. **Human-in-the-Loop Validation**: Automating scoring risks missing edge cases and qualitative nuances in tasks like mental state attribution. The work of Li et al. (2023) on ToM in multi-agent games underscores the necessity of human auditors for interpretability.  

To achieve this, we propose the **Dynamic Curriculum Benchmark (DCB)**, a framework that generates task sequences algorithmically in planning, ToM, and navigation domains. DCB uses reinforcement learning (RL) to adjust difficulty based on model performance and integrates structured human audits to validate results.  

## Significance  
This research will:  
1. **Enable Granular Cognitive Profiling**: Unlike benchmarks such as CogBench (Lv et al., 2024), DCB will map performance trajectories across task complexity levels, revealing which LLMs or architectures reliably develop planning or ToM.  
2. **Clarify Architectural Trade-offs**: By comparing fine-tuned end-to-end models (e.g., those optimized for specific tasks via gradient-based updates) to modular frameworks (e.g., those augmented with external planners or belief state trackers), we can address the debate on the robustness of different approaches raised in *Hypothetical Minds*.  
3. **Inform Training Paradigms**: The literature suggests that iterative cognitive mechanisms (Lv et al., 2024) and belief state modules (Li et al., 2023) improve performance. DCB will identify how these components aid skill acquisition under dynamic curricula.  
4. **Mitigate Hallucination Risks**: Task state hallucination—a key failure mode in prior studies (Li et al., 2023)—will be explicitly evaluated through both automatic metrics (e.g., consistency checks) and human oversight.  

By bridging gaps in cognitive benchmarks and model evaluation, DCB could serve as a foundational tool for AI researchers, cognitive scientists, and developers seeking to align LLMs with human-like reasoning and social cognition.

# Methodology  
## Task Domain Definitions  
DCB evaluates LLMs through three cognitive domains:  
1. **Planning Tasks**:  
   - *Structure*: Grid worlds or symbolic puzzles (e.g., Tower of Hanoi, Sokoban) requiring step-by-step decomposition of goals.  
   - *Complexity Scaling*: Start with single-agent, 2-step problems and progress to multi-step puzzles with conditional branches or distractors (e.g., "Move object A to B only if C is unblocked").  
2. **Theory-of-Mind Tasks**:  
   - *Structure*: Narrative-based scenarios where agents infer others’ hidden beliefs, desires, or goals. Example: "Agent X knows the key is in location A, but Agent Y, unaware of X's knowledge, hides it in location B. What does X expect Y to do next?"  
   - *Stages*: Begin with first-person belief attribution (Lv et al., 2024) and advance to multi-agent, recursive ToM (e.g., predicting Agent Y's belief about Agent X's belief).  
3. **Navigation Tasks**:  
   - *Structure*: Maze traversal or spatial reasoning in dynamic environments with changing obstacles or goal locations.  
   - *Complexity Scaling*: Progress from fixed mazes to partially observable environments requiring memory-based planning (e.g., re-planning after path disruptions).  

## Dynamic Curriculum Design  
The curriculum will be algorithmically generated and adapt to the LLM's performance history using a reinforcement learning (RL) task sampler.  

### Algorithm Overview  
1. **Task Representation**:  
   Each task is encoded as a tuple $(s_i, a_i, r_i)$, where $s_i$ is the task state, $a_i$ is the model's response, and $r_i$ is the reward (0/1 success flag).  
2. **Curriculum Sampler**:  
   - Initialize a base policy $\pi_0(s)$ to generate "easy" tasks (e.g., 2-step plans, single-agent ToM prompts).  
   - Use an RL agent with policy gradient updates to learn a transition function $T(s, d)$, where $d$ denotes task difficulty.  
   - Policy update via REINFORCE:  
     $$\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^{H} \nabla_\theta \log \pi_\theta(s_t) \cdot \left( \sum_{t'=t}^{H} \gamma^{t'-t} r_t \right) \right]$$  
     Here, $\theta$ parameterizes the RL agent, $\tau$ is the trajectory of task attempts, $H$ is the task horizon, and $\gamma = 0.9$ discounts future rewards.  
3. **Reward Function**:  
   - To balance exploration and exploitation, the reward is defined as:  
     $$R(\tau, \theta) = \alpha \cdot \mathbb{E}_{s \sim \tau}[\text{acc}(s)] + \beta \cdot \mathbb{E}_{s \sim \tau}[1/\text{solve\_time}(s)] - \gamma \cdot \mathbb{E}_{s \sim \tau}[H(s)]$$  
     - $\text{acc}(s)$: Task success (0/1).  
     - $\text{solve\_time}(s)$: Time (or steps) to solve the task.  
     - $H(s)$: Hallucination score (e.g., inconsistency with ground truth).  
   - Hyperparameters $\alpha, \beta, \gamma$ are tuned via Bayesian optimization to prioritize accurate, efficient performance.  
4. **Difficulty Scaling**:  
   - If the model solves $ \geq 90\% $ of tasks in a bin, the RL agent unlocks higher-complexity tasks (e.g., adding agents in ToM or steps in planning).  
   - If accuracy drops $ \leq 50\% $, the curriculum reverts to intermediate难度.  

## Benchmarking Pipeline  
### Implementation Steps  
1. **Initial Data Collection**:  
   - Generate 1,000 low-difficulty tasks per domain using templates and rule-based engines.  
   - Annotate each task with gold-standard solutions and metadata (e.g., step count, agent count).  
2. **Training the RL Sampler**:  
   - Pre-train the RL agent on simulated LLMs with known capabilities to seed curriculum generation.  
   - Train $\pi_\theta$ using proximal policy optimization (PPO) to maximize $R(\tau, \theta)$.  
3. **Benchmarking Protocols**:  
   - **Model Selection**: Evaluate leading LLMs (GPT-4, Falcon, Llama3) and modular architectures (e.g., ToM-augmented models from Hypothetical Minds, CogGPT's iterative memory modules).  
   - **Execution Loop**:  
     ```python
     for episode in range(NUM_EPISODES):
         sample a task s from pi_theta
         run LLM on s using task-specific prompt engineering
         record acc(s), solve_time(s), hallucination(s)
         update T(s, d) and recompute emergence probability curves
     ```  
4. **Automatic Evaluation Metrics**:  
   - **Task Success Rate**: Average accuracy over the curriculum.  
   - **Emergence Threshold**: Minimum difficulty $d$ where $\text{acc}(d) > 80\%$.  
   - **Cognitive Trajectory**: Model accuracy over time $t$, formalized as $\mathcal{T}_\theta(t) = \sigma(w \cdot t + b)$, where $\sigma$ is the logistic function.  
5. **Human-in-the-Loop Validation**:  
   - Randomly audit 10% of tasks where models make unexpected responses (e.g., incorrect ToM inferences despite high accuracy).  
   - Deploy Turkers or in-house experts to rate model outputs for:  
     - Factual consistency (0–1).  
     - Mental state attribution correctness (ToM-specific metric).  
     - Creativity (e.g., novel plans beyond gold-standard solutions).  
   - Aggregate human scores using the **Dawid-Skene estimator** for noisy crowdsourcing labels:  
     $$P(y|h) = \arg\max_{c \in \{\text{Correct}, \text{Incorrect}\}} \sum_{i=1}^{N} \log P(c|h_i, \pi_i)$$  
     where $h_i$ is labeler $i$'s history of accuracy and $\pi_i$ their predicted confusion matrix.  

## Model Architecture Comparison  
### Fine-Tuned vs. Modular LLMs  
We will compare two architectures:  
1. **End-to-End Fine-Tuned Models**: GPT-4 or Llama3 fine-tuned on DCB tasks without external modules.  
2. **Modular Architectures**: CogGPT-style hierarchical planners (Lv et al., 2024) and Hypothetical Minds' ToM modules (Cross et al., 2024). For example:  
   - **Iterative Reasoning**: Use CogGPT’s "reflection" layer to revisit previous steps.  
   - **Belief State Tracking**: Add a separate neural network to encode ToM inferences, interacting with the LLM via structured intermediate representations.  
3. **Quantitative Measures**:  
   - ToM: Accuracy in predicting hidden beliefs ($A_{\text{ToM}}$).  
   - Planning: Ratio of valid steps to ground-truth solution ($R_{\text{plan}}$).  
   - Hallucination: Jaccard similarity between model outputs and reference solutions ($J_{\text{halluc}}$).  

## Ablation Studies  
- **Effect of RL Curriculum**: Compare DCB with a static curriculum baseline (fixed task progression).  
- **Role of Modular Components**: For modular models, disable belief trackers or memory buffers and measure performance degradation.  
- **Human Audit Contribution**: Correlate human scores with automatic metrics to assess when hallucinations escape detection.  

# Expected Outcomes & Impact  
## Cognitive Profiling of LLMs  
DCB will produce **task-specific cognitive profiles** comparing models across:  
- **Emergence Trajectories**: For ToM, we hypothesize that models like Hypothetical Minds (Cross et al., 2024) will show earlier emergence (lower $d$) than end-to-end LLMs, validating the role of modular design.  
- **Solving Efficiency**: Fine-tuned LLMs may perform well on low-complexity tasks but plateau under recursion or long-horizon planning. Modular architectures, with explicit belief/state representations, should maintain performance.  

## Architectural Insights  
1. **Planning vs. Modular Enhancement**:  
   - DCB will test whether augmenting LLMs with separate planners (e.g., using CogGPT’s iterative memory) mitigates the long-horizon context collapse observed in *Emergent Response Planning* (Zhichen Dong et al., 2025).  
2. **ToM Emergence Conditions**:  
   - Literature suggests ToM arises from training on narratives (Li et al., 2023). DCB will test scaling laws by measuring how $d_{\text{emerg}}$ decreases with parameter size. For example:  
     $$d_{\text{emerg}} = \frac{\sum_{i=1}^{k} \mathcal{T}_\theta(t_i)}{|\mathcal{T}_\theta(t_i) > 0.8|}$$  
     where $t_i$ are episodes and $k$ is the maximum curriculum length.  

## Benchmark Design Improvements  
- **Dynamic Evaluation Paradigm**: Static benchmarks treat all models as static entities, ignoring their potential for learning. DCB’s RL-driven difficulty scaling offers a fairer, more informative assessment.  
- **Human-AI Consistency Metrics**: By correlating human audits with automatic scores, we will identify tasks where LLMs excel qualitatively but fail quantitative checks (e.g., plausible yet incorrect ToM inferences).  

## Broader Implications  
1. **Model Development**: The emergence thresholds revealed by DCB could guide architectural choices. For instance, if modular systems consistently outperform fine-tuned ones at $d > 5$, model designers might prioritize external belief trackers.  
2. **Cognitive Science**: Comparing LLM trajectories with human studies could reveal similarities in learning progression. For example, if LLMs require similar task sequences to develop ToM as humans do, this would support parallels in cognitive mechanisms.  
3. **AI Alignment**: Early detection of planning or ToM failures could inform safety protocols for LLMs used in high-stakes domains.  

# Conclusion  
The Dynamic Curriculum Benchmark represents a paradigm shift in evaluating LLMs as evolving cognitive systems rather than fixed entities. By integrating RL-driven task generation, modular architecture benchmarks, and rigorous human validation, DCB will systematically unravel how and when LLMs acquire critical skills like long-horizon planning and multi-agent ToM reasoning. The expected outcomes include standardized cognitive profiles for models, evidence-based design principles for robust architectures, and a framework that bridges AI research with cognitive psychology. Future work will explore multimodal extensions (e.g., vision-language reasoning) and causal analysis of emergence mechanisms to address challenges like task state hallucination. This proposal aligns with the workshop’s goal of situating LLMs in the intelligence landscape by providing an actionable toolkit for measuring the interplay between scale, architecture, and cognition.  

# References  
1. Cross et al. (2024). *Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models.*  
2. Lv et al. (2024). *CogGPT: Unleashing the Power of Cognitive Dynamics on Large Language Models.*  
3. Li et al. (2023). *Theory of Mind for Multi-Agent Collaboration via Large Language Models.*  
4. Dong et al. (2025). *Emergent Response Planning in LLM.*