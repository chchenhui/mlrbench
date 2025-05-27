1. Title  
Biologically-Inspired Semantic Memory Architecture with Adaptive Forgetting Mechanisms for Long-Horizon LLM Agents  

2. Introduction  
Background  
Large language model (LLM) agents have demonstrated remarkable capabilities in single-turn and short-session tasks, but they struggle to maintain coherent, contextually relevant behavior over extended multi-session interactions or complex multi-step tasks. Their fixed context window and lack of explicit memory management lead to two major issues: (1) catastrophic forgetting of early information and (2) context overload from accumulating irrelevant details. In contrast, human cognition employs a dual-pathway memory system, consisting of a fast-changing episodic store and a slower semantic store, coupled with active forgetting to prune unneeded information and consolidate important patterns.  

Research Objectives  
This project aims to design, implement, and evaluate a biologically inspired semantic memory architecture for LLM agents that:  
• maintains a hierarchical semantic network of concepts and relations,  
• adapts via a controllable forgetting mechanism based on recency, relevance, and importance,  
• compresses episodic experiences into generalized semantic representations, and  
• learns optimal forgetting parameters through reinforcement learning.  

Significance  
By endowing LLM agents with human-like memory consolidation and forgetting, we expect to achieve:  
• improved long-horizon coherence and consistency,  
• reduced computational and memory overhead by limiting context bloat,  
• dynamic adaptation to evolving tasks and user preferences, and  
• a step toward cognitively aligned AI systems that better mirror human memory dynamics.  

3. Methodology  

3.1 Overview of the Dual-Pathway Memory Architecture  
Our system comprises two interacting modules:  
1. Episodic Memory Store (EMS): a temporary log of raw events or textual observations during agent–environment interaction.  
2. Semantic Memory Network (SMN): a graph-based representation of abstracted concepts and their interrelations.  

Data Flow and Operations  
• On each new observation $o_t$, the LLM agent updates its working context and appends $o_t$ to the EMS.  
• Periodically (every $T_c$ steps), the EMS is consolidated into the SMN via clustering and embedding compression.  
• After consolidation, the SMN undergoes an adaptive forgetting step that prunes nodes based on a computed forgetting score.  
• During generation, the agent’s prompt is enriched by retrieved subgraphs from the SMN relevant to the current context.  

3.2 Semantic Memory Network Design  
Representation  
• Nodes $m_i$ in the SMN represent semantic concepts with embedding vectors $\mathbf{v}_i \in \mathbb{R}^d$.  
• Edges $e_{ij}$ encode relations (co-occurrence, causality, hierarchical links), stored in an adjacency matrix $A$.  

Embedding Update via Graph Convolution  
We use a two-layer Graph Convolutional Network (GCN) to refine node embeddings after each consolidation:  
$$\tilde A = A + I,\quad \tilde D_{ii} = \sum_j \tilde A_{ij},$$  
$$H^{(l+1)} = \sigma\bigl(\tilde D^{-\tfrac12}\tilde A\,\tilde D^{-\tfrac12} H^{(l)} W^{(l)}\bigr)$$  
with $H^{(0)} = [\mathbf{v}_1;\dots;\mathbf{v}_N]$ and $W^{(l)}$ trainable weights.  

Consolidation and Compression  
Every $T_c$ steps, we compress clusters of episodic embeddings $\{\mathbf{e}_k\}$ into a new semantic node $\mathbf{v}_{new}$:  
• Perform K-means clustering on $\mathbf{e}_k$ to identify $C$ clusters.  
• For each cluster $c$, compute the centroid:  
$$\mathbf{v}_c = \frac1{|c|}\sum_{\mathbf{e}_k\in c} \mathbf{e}_k\,.$$  
• Add a node with embedding $\mathbf{v}_c$ and update $A$ via similarity thresholding: connect nodes if $\cos(\mathbf{v}_c,\mathbf{v}_j)>\theta$.  

3.3 Adaptive Forgetting Mechanism  
We define three metrics for each semantic node $m_i$:  
1. Recency:  
$$\text{rec}_i = \exp\bigl(-\lambda\,(t - t_i)\bigr),$$  
where $t_i$ is the last activation time.  
2. Relevance:  
$$\text{rel}_i = \cos\bigl(\mathbf{v}_i,\mathbf{c}_t\bigr),$$  
with $\mathbf{c}_t$ the current context embedding.  
3. Importance: derived from graph centrality (e.g., PageRank):  
$$\text{imp}_i = \text{PR}(m_i)\,\big/\sum_j \text{PR}(m_j)\,.$$  

Forgetting Score and Pruning  
We combine metrics into a single score:  
$$s_i(\alpha,\beta,\gamma)\;=\;\alpha\,\text{rec}_i\;+\;\beta\,\text{rel}_i\;+\;\gamma\,\text{imp}_i\,.$$  
We define the retention probability via a logistic function:  
$$P_\text{retain}(m_i)\;=\;\frac{1}{1 + \exp\bigl(-s_i(\alpha,\beta,\gamma)\bigr)}\,. $$  
At each forgetting step, nodes with $P_\text{retain}(m_i)<\tau$ (threshold) are pruned.  

Algorithm 1: SMN Update with Consolidation and Forgetting  
1. Input: new observation $o_t$, EMS, SMN, parameters $(\alpha,\beta,\gamma,\tau,T_c)$.  
2. Append embedding of $o_t$ to EMS.  
3. If $t\mod T_c=0$:  
   a. Consolidate EMS into SMN (K-means + centroid insertion).  
   b. Update SMN embeddings via GCN.  
   c. For each node $m_i$: compute recency, relevance, importance; compute $P_\text{retain}(m_i)$.  
   d. Remove nodes with $P_\text{retain}(m_i)<\tau$.  
   e. Clear EMS or retain a sliding window.  
4. Return updated SMN.  

3.4 Learning Forgetting Parameters via Reinforcement Learning  
State and Action  
• State $s_t$: summary of SMN statistics (e.g., size, average relevance) and task context.  
• Action $a_t$: continuous adjustment of $(\alpha,\beta,\gamma,\tau)$.  

Reward Design  
We define a composite reward for an episode of length $T$:  
$$R = R_\text{task} - \lambda_\text{m} \,\bigl\lvert\lvert\text{SMN}\bigr\rvert\lvert_1\,, $$  
where $R_\text{task}$ measures task success (e.g., multi-session dialogue coherence score or game completion), and $\lvert\lvert\text{SMN}\rvert\lvert_1$ is the total node count penalized by $\lambda_\text{m}$.  

RL Algorithm  
We employ Proximal Policy Optimization (PPO) to learn a policy $\pi_\theta(a_t\mid s_t)$ that adjusts forgetting parameters online. Policies are parameterized by a small feed-forward network. We update $\theta$ every $K$ episodes using the clipped surrogate objective from PPO.  

3.5 Experimental Design and Validation  
Datasets and Environments  
• Multi-Session Dialogue: a collection of customer support dialogues spanning 5–10 turns per session across multiple days.  
• Text-Based Interactive Tasks (ALFWorld): agents execute household tasks requiring long sequences of actions and memory of past instructions.  
• Research Assistance Benchmark: multi-session writing and summarization tasks where earlier documents inform later writing.  

Baselines for Comparison  
• MemoryBank (arXiv:2305.10250)  
• M+ (arXiv:2502.00592)  
• RecallM (arXiv:2307.02738)  
• Vanilla LLM with sliding window context  

Evaluation Metrics  
1. Coherence Score: entity-grid based coherence ($\uparrow$ higher is better).  
2. Task Success Rate: percentage of completed tasks or correct answers.  
3. Memory Efficiency: average SMN size over time.  
4. Information Retention vs. Forgetting:  
   • Retention Accuracy: recall of required earlier facts.  
   • Forgetting Precision: rate of removal of truly irrelevant nodes (manual annotation).  
5. Computational Overhead: average inference time per turn.  

Statistical Analysis  
For each configuration, we run 5 random seeds. We report mean and standard error for all metrics and perform paired t-tests to assess significance of improvements over baselines ($p<0.05$).  

Implementation Details  
• Base LLM: GPT-style decoder with 6B parameters.  
• Embedding dimension $d=512$.  
• Consolidation interval $T_c=50$ steps.  
• Forgetting threshold $\tau$ initialized to $0.5$.  
• PPO hyperparameters: clip factor $0.2$, learning rate $3\cdot10^{-5}$, batch size 64.  
• Training on 8 NVIDIA A100 GPUs.  

4. Expected Outcomes & Impact  
Expected Outcomes  
• Demonstration that our dual-pathway memory with adaptive forgetting yields a 15–25% improvement in coherence and task success over state-of-the-art memory-augmented LLMs.  
• Reduction of average memory graph size by 30–50%, enabling lower latency and resource usage.  
• Learned forgetting parameters that exhibit human-like patterns: quick forgetting of low-importance events and long retention of core concepts.  
• A modular open-source library implementing the semantic memory architecture and benchmarks.  

Impact  
This research will advance the field of LLM agents by providing a principled, cognitively inspired framework for long-term memory management. Applications include multi-session virtual assistants, educational tutors that recall student progress, autonomous agents in simulation and robotics that require long-term planning, and research support tools that track and integrate multi-document knowledge over extended writing sessions. Beyond performance gains, our work bridges cognitive science insights with AI system design, laying groundwork for more human-aligned, efficient, and trustworthy language agents.