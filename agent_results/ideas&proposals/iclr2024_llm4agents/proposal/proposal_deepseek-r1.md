**Research Proposal: A Dual-Pathway Semantic Memory Architecture with Forgetting Mechanisms for Enhanced Long-Term Task Performance in LLM Agents**

---

### 1. **Title**  
*A Dual-Pathway Semantic Memory Architecture with Forgetting Mechanisms for Enhanced Long-Term Task Performance in LLM Agents*

---

### 2. **Introduction**  

**Background**  
Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning, planning, and tool augmentation. However, managing memory over extended interactions remains a critical weakness. Current LLM agents often face *catastrophic forgetting*—losing previously acquired knowledge when processing new information—or become burdened by excessive contextual data, leading to reduced coherence and efficiency. Human cognition addresses this challenge through sophisticated memory consolidation and forgetting mechanisms that prioritize semantically meaningful information while pruning less relevant details. Recent works like MemoryBank (Zhong et al., 2023) and RecallM (Kynoch et al., 2023) have explored dynamic memory systems, but none fully emulate the hierarchical organization and selective retention observed in human memory.

**Research Objectives**  
This study proposes *Semantic Memory with Adaptive Forgetting* (SMAF), a dual-pathway architecture inspired by human cognition to address LLM agents’ memory limitations. Key objectives include:  
1. Designing a hierarchical semantic network to organize concepts and relationships dynamically.  
2. Developing a forgetting mechanism guided by **recency**, **relevance**, and **importance** metrics.  
3. Implementing reinforcement learning (RL) to optimize forgetting parameters based on task performance.  
4. Validating the system’s ability to improve long-term coherence, reduce computational overhead, and align memory retention patterns with human-like behaviors.  

**Significance**  
By bridging insights from cognitive science and machine learning, SMAF has the potential to advance LLM agents’ ability to handle complex, multi-session tasks (e.g., research assistance, strategic planning) while reducing reliance on oversized context windows. This aligns with the workshop’s focus on *memory mechanisms* and *reasoning/planning*, offering a novel framework for scalable, cognitively inspired language agents.

---

### 3. **Methodology**  

#### **3.1 Semantic Memory Architecture**  
The SMAF architecture has two components:  

**A. Hierarchical Semantic Network**  
- **Structure**: A graph-based network where nodes represent semantic concepts (e.g., entities, actions) and edges encode relational weights. Nodes are organized hierarchically, with higher-level abstractions (e.g., “scientific research”) linked to specific instances (e.g., “transformer models”).  
- **Updating Mechanism**:  
  1. **Embedding**: New input is embedded using a pre-trained encoder (e.g., Sentence-BERT).  
  2. **Similarity Matching**: Compute cosine similarity between the new embedding $v_{\text{new}}$ and existing nodes $v_i$:  
     $$\text{sim}(v_{\text{new}}, v_i) = \frac{v_{\text{new}} \cdot v_i}{\|v_{\text{new}}\| \|v_i\|}.$$  
     Nodes with similarity > threshold $\tau$ are merged.  
  3. **Hierarchical Integration**: If no match is found, a new node is created and linked to parent concepts via co-occurrence statistics.  

**B. Forgetting Mechanism**  
Three metrics govern pruning:  
1. **Recency** ($R_t$): Exponential decay of node activations over time:  
   $$R_t(n) = \lambda \cdot R_{t-1}(n) + (1-\lambda) \cdot f_{\text{access}}(n),$$  
   where $\lambda$ is a decay rate and $f_{\text{access}}$ measures recent accesses.  
2. **Relevance** ($R_l$): Contextual alignment with the current task, computed via attention weights between the node and task prompt.  
3. **Importance** ($I$): Node centrality in the semantic network, measured using PageRank.  

The combined forgetting score is:  
$$F(n) = \alpha R_t(n) + \beta R_l(n) + \gamma I(n),$$  
where $\alpha, \beta, \gamma$ are learnable weights. Nodes with $F(n) < \theta$ are compressed into higher-level abstractions or pruned.  

#### **3.2 Reinforcement Learning for Forgetting Optimization**  
We train a policy network $P_\phi$ to adjust $\alpha, \beta, \gamma$ and $\theta$ dynamically. The reward function balances:  
- **Task Success** ($S$): Accuracy in achieving task goals.  
- **Memory Efficiency** ($E$): Reduction in retained nodes vs. baseline.  
- **Coherence Penalty** ($C$): Measured via entropy of generated text.  

Reward: $R = w_1 S + w_2 E - w_3 C$, where $w_i$ are task-specific weights.  
The policy network is trained using proximal policy optimization (PPO) across diverse environments.  

#### **3.3 Experimental Design**  

**Datasets**  
- **Dialogue Tasks**: Multi-session conversations from Multi-Session Chat (MSC) dataset.  
- **Planning Tasks**: Long-horizon ALFRED (Embodied QA) and project management simulations.  
- **Synthetic Benchmarks**: Custom sequences testing memory retention over 1,000+ steps.  

**Baselines**  
- MemoryBank (Zhong et al., 2023)  
- RecallM (Kynoch et al., 2023)  
- MeMo (Zanzotto et al., 2025)  

**Evaluation Metrics**  
1. **Coherence Score**: GPT-4-based metric evaluating logical consistency in multi-turn outputs.  
2. **Task Success Rate**: Accuracy in completing defined objectives (e.g., answering questions correctly).  
3. **Normalized Perplexity**: Measure of language quality relative to context length.  
4. **Memory Footprint**: Average nodes retained vs. ground-truth relevant concepts.  
5. **Human Evaluation**: Expert assessments of output relevance and coherence.  

**Ablation Studies**  
Testing individual components (e.g., removing RL optimization or hierarchical clustering) to isolate contributions.

---

### 4. **Expected Outcomes & Impact**  

**Expected Outcomes**  
1. SMAF will outperform baseline models by **15–20%** in task success rate and coherence on extended tasks.  
2. Reduction in memory footprint by **30–50%** compared to vector database approaches.  
3. Quantifiable alignment with human-like forgetting patterns, measured via similarity to Ebbinghaus-like retention curves.  
4. Open-source framework for adaptive memory management in LLM agents.  

**Impact**  
This work addresses the workshop’s focus on *memory mechanisms* and *reasoning/planning* by providing:  
- **Practical Tools**: SMAF can enhance AI assistants in healthcare, education, and autonomous systems where long-term coherence is critical.  
- **Theoretical Insights**: A cognitively grounded framework for understanding memory in LLMs, bridging AI and neuroscience.  
- **Resource Efficiency**: Reduced computational costs for handling long contexts, enabling deployment on resource-constrained devices.  

---

By integrating hierarchical memory organization with intelligent forgetting, this research proposes a paradigm shift in how LLM agents process and retain information, moving closer to the adaptability and efficiency of human cognition.