Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **Cognitively-Inspired Semantic Memory Architecture with Adaptive Forgetting for Enhanced LLM Agent Performance**

**2. Introduction**

**2.1 Background**
Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation, paving the way for sophisticated LLM agents capable of performing complex tasks autonomously in diverse environments (Workshop Description). These agents leverage natural language not just for communication but also for reasoning and planning, interacting with tools and environments to achieve specified goals. However, a significant bottleneck hindering their long-term operational effectiveness is memory management. Current LLM agents often rely on expanding context windows or simple vector database retrieval (Retrieval-Augmented Generation - RAG), leading to challenges. As interactions extend or tasks become more complex, they either suffer from context window limitations, lose track of crucial past information (a form of catastrophic forgetting, similar to issues discussed in [3, 8]), or become inundated with irrelevant data, degrading performance and coherence.

Human cognition, in contrast, exhibits robust and efficient memory mechanisms. We possess the ability to retain vast amounts of information over long periods, selectively retrieve relevant knowledge, and crucially, forget information that is no longer pertinent or has been consolidated into more abstract representations. This process of forgetting is not merely a passive decay but an active, adaptive mechanism essential for cognitive efficiency, learning, and generalization. Inspired by these biological processes, particularly the dynamics of semantic memory formation and selective forgetting, this research proposes a novel memory architecture for LLM agents.

**2.2 Problem Statement**
Standard LLM agent architectures lack sophisticated, long-term memory systems with integrated forgetting mechanisms. This deficiency leads to several key problems:
1.  **Limited Long-Term Coherence:** Agents struggle to maintain consistent personas, recall distant but relevant facts, or track evolving goals across extended interactions or multi-step tasks.
2.  **Context Window Overload:** Reliance on injecting all potentially relevant history into the LLM's limited context window is inefficient and often infeasible for very long tasks, leading to information truncation or excessive computational cost.
3.  **Inefficient Information Retrieval:** Simple vector similarity searches may retrieve superficially similar but contextually inappropriate memories, failing to capture deeper semantic relationships or temporal dynamics [2].
4.  **Inability to Adaptively Prune Information:** Unlike humans, current systems lack mechanisms to intelligently discard outdated or irrelevant information, leading to memory clutter and potential retrieval of conflicting or obsolete data [cf. 6, where updates are addressed but adaptive pruning less so].

Addressing these limitations requires moving beyond simple memory stores towards architectures that actively manage information lifecycle, mirroring the balance between retention and forgetting observed in human cognition. While research exists on LLM memory augmentation [2, 4, 6, 10] and machine unlearning/forgetting [1, 5, 7, 9], there is a need for an integrated system specifically designed for autonomous agents that combines structured semantic memory with adaptive, cognitively-inspired forgetting mechanisms optimized for ongoing task performance.

**2.3 Proposed Solution: Semantic Memory with Adaptive Forgetting (SMAF)**
We propose the development of a **Semantic Memory Architecture with Adaptive Forgetting (SMAF)** for LLM agents. This architecture comprises two core, interacting components:
1.  **A Dynamic Semantic Network:** This component represents knowledge as a structured graph where nodes are concepts (entities, facts, events, goals) and edges represent semantic relationships (e.g., "is-a", "causes", "related-to", temporal sequence). Information extracted from the agent's interactions and reasoning processes is continuously embedded and integrated into this network, allowing for the representation of complex, interconnected knowledge beyond simple episodic recall. This aligns with explorations into associative memories [4].
2.  **A Cognitively-Inspired Adaptive Forgetting Mechanism:** This mechanism dynamically adjusts the accessibility or existence of information within the semantic network based on calculated metrics reflecting recency, relevance, and estimated importance. It aims to mimic human memory consolidation and pruning, compressing detailed episodic traces into generalized semantic knowledge and selectively weakening or removing less critical information over time. This contrasts with explicit unlearning tasks [1, 5, 9] by focusing on adaptive memory management for agent efficiency rather than data removal for security or privacy.

The forgetting process will be tunable and potentially optimized using Reinforcement Learning (RL), where the agent learns optimal forgetting strategies based on its performance in long-running tasks.

**2.4 Research Objectives**
The primary objectives of this research are:
1.  **Design and Implement the SMAF Architecture:** Develop the data structures and algorithms for the dynamic semantic network and the adaptive forgetting mechanism.
2.  **Integrate SMAF with an LLM Agent:** Create the interface mechanism allowing an LLM agent to interact with the SMAF system for memory storage, retrieval, and continuous updating during task execution.
3.  **Develop Adaptive Forgetting Algorithms:** Formulate and implement algorithms based on recency, relevance, and importance heuristics, inspired by cognitive models of forgetting (e.g., decay, interference, consolidation).
4.  **Optimize Forgetting Parameters using RL:** Implement an RL framework to dynamically tune the parameters of the forgetting mechanism (e.g., decay rates, relevance thresholds, importance weights) based on agent task performance signals.
5.  **Evaluate SMAF Performance:** Rigorously evaluate the SMAF-enhanced agent against baseline models on complex, long-duration tasks, measuring improvements in coherence, task success, information recall accuracy, and computational efficiency (context window usage, processing time).

**2.5 Significance**
This research holds significant potential for advancing the field of LLM agents. By equipping agents with a more human-like memory system, we anticipate:
*   **Enhanced Agent Capabilities:** Enabling agents to undertake more complex, longer-duration tasks requiring sustained coherence and historical context (e.g., research assistants, personalized tutors, multi-session planning assistants).
*   **Improved Efficiency:** Reducing the reliance on massive context windows, potentially lowering computational costs and latency.
*   **Cognitive Plausibility:** Creating agents whose information processing and retention patterns are more aligned with human cognition, potentially leading to more natural and predictable interactions.
*   **Insights into Memory Mechanisms:** Providing a computational testbed for exploring theories of human memory consolidation and forgetting. This aligns with the workshop themes of exploring memory mechanisms, reasoning/planning, and conceptual frameworks for language agents.

**3. Methodology**

**3.1 Conceptual Framework: SMAF**
The SMAF system operates alongside a core LLM agent. Interactions and internal reasoning steps of the agent are processed to extract key information (entities, events, facts, relationships, goals). This information is then structured and stored in the SMAF, which consists of the Semantic Network and the Forgetting Mechanism manager. During task execution, the agent queries SMAF based on the current context to retrieve relevant memories, which are then injected into the LLM's prompt to inform its next action or response.

**3.2 Semantic Network Component**
*   **Representation:** The semantic network will be implemented as a directed graph $G = (V, E)$, where $V$ is a set of nodes representing concepts, and $E$ is a set of edges representing relationships.
    *   **Nodes ($v \in V$):** Each node will store:
        *   A concept identifier (e.g., entity name, event description).
        *   A semantic embedding vector (derived from the LLM or a dedicated embedding model).
        *   Metadata: Timestamp of creation/last access ($t_{created}$, $t_{last\_access}$), type (e.g., entity, fact, goal, episodic event), source (e.g., user input, internal reasoning).
        *   An activation/salience score $A(v)$, influenced by recency, relevance, and importance.
    *   **Edges ($e \in E$):** Each edge $(u, v)$ will represent a relationship (e.g., `type_of`, `located_at`, `causes`, `precedes`) and may store a weight or confidence score, and potentially temporal information.
*   **Information Integration:**
    1.  **Input Processing:** Incoming text (user prompts, tool outputs, agent's internal monologue) is parsed by the LLM (or a dedicated module) to identify key concepts and their relationships.
    2.  **Node Creation/Update:** Identified concepts are mapped to nodes in the graph. New concepts result in new nodes; existing concepts have their metadata (e.g., $t_{last\_access}$) updated. Embeddings are generated for new concepts.
    3.  **Edge Creation/Update:** Relationships between concepts are added as edges, potentially strengthening existing edges if the relationship is re-asserted.
    4.  **Hierarchical Structuring (Optional):** Explore mechanisms for creating hierarchical relationships (e.g., using graph algorithms or LLM-driven classification) to organize concepts abstractly.

**3.3 Adaptive Forgetting Mechanism**
This mechanism periodically (or continuously) evaluates nodes and edges in the semantic network to adjust their salience or potentially prune them.
*   **Metrics Calculation:** For each node $v$ at time $t$:
    *   **Recency Score ($S_{recency}$):** Based on the time elapsed since the last access. A simple exponential decay model:
        $$S_{recency}(v, t) = e^{-\lambda_r (t - t_{last\_access}(v))}$$
        where $\lambda_r$ is the recency decay rate parameter.
    *   **Relevance Score ($S_{relevance}$):** Based on similarity to the current task context $C$ and access frequency.
        $$S_{relevance}(v, C) = w_{sim} \cdot \text{sim}(\text{emb}(v), \text{emb}(C)) + w_{freq} \cdot \log(1 + \text{access\_count}(v))$$
        where $\text{sim}$ is a similarity function (e.g., cosine similarity), $\text{emb}(\cdot)$ denotes the embedding vector, and $w_{sim}, w_{freq}$ are weighting parameters. The context embedding $\text{emb}(C)$ could be the embedding of the recent interaction history or current goal.
    *   **Importance Score ($S_{importance}$):** Estimated intrinsic importance of the concept. This could be based on graph centrality (e.g., PageRank score within the semantic network), explicit user designation, or connection to high-level goals.
        $$S_{importance}(v) = w_{cent} \cdot \text{Centrality}(v) + w_{goal} \cdot \text{GoalRelevance}(v) + w_{user} \cdot \text{UserSpecifiedImportance}(v)$$
        where $w_{cent}, w_{goal}, w_{user}$ are weights.
*   **Retention Score Calculation:** A combined score determining the likelihood of retaining the node:
    $$S_{retention}(v, t, C) = f(S_{recency}(v, t), S_{relevance}(v, C), S_{importance}(v))$$
    where $f$ could be a weighted sum: $f = \alpha S_{recency} + \beta S_{relevance} + \gamma S_{importance}$. The weights $\alpha, \beta, \gamma$ (along with $\lambda_r, w_{sim}, w_{freq}, w_{cent}, w_{goal}, w_{user}$) are key parameters for tuning.
*   **Pruning/Consolidation:**
    *   **Salience Adjustment:** The retention score $S_{retention}$ can directly update the node's activation/salience $A(v)$. Low-salience nodes are less likely to be retrieved.
    *   **Threshold-Based Pruning:** Nodes whose $S_{retention}$ falls below a dynamic threshold $\theta$ for a sustained period might be pruned (removed) from the active memory graph.
    *   **Memory Consolidation:** Instead of outright deletion, implement a mechanism where detailed episodic nodes (e.g., specific interaction turns) with low retention scores are compressed or generalized into existing or new semantic nodes (e.g., "User frequently asks about topic X" instead of storing every question). This mimics the transition from episodic to semantic memory.

**3.4 Integration with LLM Agent**
*   **Memory Querying:** When the agent needs to generate a response or plan the next step, it formulates a query based on the current context/goal. This query is used to find relevant nodes in the semantic network. Retrieval can be based on:
    *   Embedding similarity between the query and node embeddings.
    *   Graph traversal starting from recently accessed or contextually relevant nodes.
    *   Filtering based on node salience $A(v)$.
*   **Context Augmentation:** The retrieved information (e.g., text summaries of key nodes/subgraphs) is formatted and prepended to the LLM's input prompt, providing relevant long-term context. The amount of information retrieved can be capped based on salience scores and context window limits.
*   **Memory Updating:** The agent's output and internal reasoning steps feed back into the Semantic Network Component for continuous integration.

**3.5 Optimization via Reinforcement Learning (RL)**
To adapt the forgetting mechanism to specific tasks and environments, we will use RL to learn optimal parameters for the retention score calculation (e.g., $\alpha, \beta, \gamma, \lambda_r, \theta$).
*   **State ($s$):** Representation of the current memory state (e.g., graph size, average salience, distribution of node types) and task context (e.g., current goal, recent interaction summary, task progress).
*   **Action ($a$):** Adjustments to the forgetting parameters (e.g., increase/decrease $\lambda_r$, modify weights $\alpha, \beta, \gamma$, change pruning threshold $\theta$). Actions could be continuous adjustments or selection from discrete profiles.
*   **Reward ($r$):** A composite reward function based on:
    *   Task Success: Primary signal (e.g., +1 for successful completion, 0 otherwise, or partial rewards for milestones).
    *   Coherence/Quality Metrics: Intermediate rewards based on LLM-based evaluation of response coherence/relevance, or human feedback scores.
    *   Efficiency Penalty: Negative reward proportional to memory size or retrieval latency, encouraging pruning.
    *   Information Recall Penalty: Negative reward if the agent fails a probe question about information that should have been retained according to task requirements but was forgotten.
*   **Algorithm:** Proximal Policy Optimization (PPO) or similar policy gradient methods suitable for continuous or discrete parameter tuning. The RL agent learns a policy $\pi(a|s)$ that maximizes expected cumulative reward.

**3.6 Data Collection and Simulation Environment**
We will utilize or create simulated environments and datasets that necessitate long-term memory:
1.  **Multi-Session QA:** A dataset where questions in later sessions refer back to information provided or discussed in earlier sessions.
2.  **Long-Document Analysis/Summarization:** Tasks requiring the agent to read and synthesize information from lengthy documents over multiple interactions.
3.  **Simulated Personal Assistant:** An environment where the agent assists a user with complex, multi-step tasks evolving over time (e.g., planning a trip, managing a project), requiring memory of user preferences, past decisions, and task state. Example source: dialogues from datasets like MultiWOZ, adapted for longer interactions, or newly generated synthetic data.

**3.7 Experimental Design**
1.  **Baselines:**
    *   Base LLM (e.g., GPT-4, Llama-3) with no external memory (limited context window).
    *   LLM + Standard RAG (Vector DB storing raw interaction chunks or summaries, retrieved via similarity search).
    *   LLM + Existing memory architectures (e.g., simplified versions of MemoryBank [6] or RecallM [2], if feasible to reimplement).
2.  **Proposed Model:** LLM + SMAF (with manually tuned parameters and with RL-optimized parameters).
3.  **Ablation Studies:**
    *   SMAF with no forgetting mechanism (only semantic network accumulation).
    *   SMAF with non-adaptive forgetting (fixed parameters, simple decay).
    *   SMAF variants with different components of the retention score disabled (e.g., no relevance, no importance).
4.  **Evaluation Tasks:** Performance will be measured across the tasks defined in 3.6.

**3.8 Evaluation Metrics**
*   **Quantitative:**
    *   **Task Success Rate:** Percentage of tasks completed successfully according to predefined criteria.
    *   **Information Recall Accuracy:** F1-score on probe questions targeting specific information from different points in the interaction history.
    *   **Coherence Score:** Automated metrics (e.g., using another LLM as a judge) or human ratings for consistency and logical flow across long interactions.
    *   **Context Window Efficiency:** Average number of tokens retrieved from memory and fed into the LLM context per turn.
    *   **Computational Cost:** Memory storage size, retrieval latency, update time.
    *   **Forgetting Appropriateness:** Measure precision/recall of forgetting (did it forget irrelevant info? did it retain relevant info?). Requires annotated test sets. Compare against machine unlearning metrics [1, 5, 9].
*   **Qualitative:**
    *   Human evaluation of agent responses regarding relevance, consistency, and naturalness over long dialogues.
    *   Analysis of the semantic network structure and pruning patterns formed during tasks.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
1.  **A Novel SMAF Architecture:** A fully specified and implemented semantic memory system incorporating adaptive, cognitively-inspired forgetting mechanisms for LLM agents.
2.  **Demonstrated Performance Improvements:** Quantitative results showing that SMAF-enhanced agents significantly outperform baseline models on long-term coherence, task success rates, and accurate information recall in complex, extended tasks. We expect to show benefits over simple RAG and memory systems without sophisticated forgetting [cf. 2, 6, 10].
3.  **Efficient Memory Management:** Evidence that SMAF effectively manages memory size and reduces the required LLM context length compared to naive context expansion, while mitigating catastrophic forgetting [3, 8].
4.  **Adaptively Tuned Forgetting:** Demonstration that RL can successfully optimize forgetting parameters, leading to better task performance compared to manually tuned or fixed parameters.
5.  **Analysis of Forgetting Dynamics:** Insights into the types of information prioritized for retention versus forgetting by the system under different task pressures, potentially mirroring aspects of human memory consolidation.

**4.2 Impact**
*   **Advancement of LLM Agents:** This research will contribute a more robust and scalable memory solution, pushing the boundaries of what LLM agents can achieve in complex, long-running scenarios relevant to the workshop's focus on autonomous agents and reasoning/planning.
*   **Bridge between AI and Cognitive Science:** By operationalizing concepts from cognitive memory models (semantic networks, consolidation, adaptive forgetting), this work provides a computational framework relevant to both AI development and cognitive science research, addressing the theme of drawing from related fields.
*   **Practical Applications:** Improved memory will enhance the usability of LLM agents in real-world applications like personalized education, long-term therapeutic chatbots, continuous research assistants, and sophisticated control systems requiring persistent state and knowledge.
*   **New Research Directions:** This work may open up further research into more nuanced cognitive mechanisms for memory (e.g., interference-based forgetting, emotional tagging of memories) and multi-modal memory integration within LLM agents.

By developing and validating the SMAF architecture, this research aims to significantly enhance the capabilities and efficiency of LLM agents, making them more effective partners for complex, long-term human endeavors.

**5. Bibliography**

[1] Wang, H., Jing, Y., Sun, H., Wang, Y., Wang, J., Liao, J., & Tao, D. (2025). Erasing Without Remembering: Safeguarding Knowledge Forgetting in Large Language Models. *arXiv preprint arXiv:2502.19982*.

[2] Kynoch, B., Latapie, H., & van der Sluis, D. (2023). RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models. *arXiv preprint arXiv:2307.02738*.

[3] Liao, C., Xie, R., Sun, X., Sun, H., & Kang, Z. (2024). Exploring Forgetting in Large Language Model Pre-Training. *arXiv preprint arXiv:2410.17018*.

[4] Zanzotto, F. M., Ruzzetti, E. S., Xompero, G. A., Ranaldi, L., Venditti, D., Ranaldi, F., Giannone, C., Favalli, A., & Romagnoli, R. (2025). MeMo: Towards Language Models with Associative Memory Mechanisms. *arXiv preprint arXiv:2502.12851*.

[5] Xu, H., Zhao, N., Yang, L., Zhao, S., Deng, S., Wang, M., Hooi, B., Oo, N., Chen, H., & Zhang, N. (2025). ReLearn: Unlearning via Learning for Large Language Models. *arXiv preprint arXiv:2502.11190*.

[6] Zhong, W., Guo, L., Gao, Q., Ye, H., & Wang, Y. (2023). MemoryBank: Enhancing Large Language Models with Long-Term Memory. *arXiv preprint arXiv:2305.10250*.

[7] Pan, Z., Zhang, S., Zheng, Y., Li, C., Cheng, Y., & Zhao, J. (2024). Multi-Objective Large Language Model Unlearning. *arXiv preprint arXiv:2412.20412*.

[8] Li, H., Ding, L., Fang, M., & Tao, D. (2024). Revisiting Catastrophic Forgetting in Large Language Model Tuning. *arXiv preprint arXiv:2406.04836*.

[9] Wang, L., Zeng, X., Guo, J., Wong, K.-F., & Gottlob, G. (2024). Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language Models. *arXiv preprint arXiv:2402.05813*.

[10] Wang, Y., Krotov, D., Hu, Y., Gao, Y., Zhou, W., McAuley, J., Gutfreund, D., Feris, R., & He, Z. (2025). M+: Extending MemoryLLM with Scalable Long-Term Memory. *arXiv preprint arXiv:2502.00592*.

---