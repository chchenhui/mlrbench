# **Biologically-Inspired Semantic Memory Architecture for LLM Agents with Dynamic Forgetting Mechanisms**

## **1. Introduction**

### **1.1 Background**  
Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, their effectiveness in long-term, multi-step tasks remains limited by inefficient memory management. Traditional LLMs either retain all information indiscriminately, leading to context saturation, or rely on external databases, which introduce latency and scalability issues. In contrast, human cognition employs dynamic forgetting mechanisms to prune irrelevant information while preserving task-critical knowledge. This biological inspiration motivates the development of a semantic memory architecture that mimics human memory consolidation and forgetting processes.

Recent research highlights critical challenges in memory management for LLMs, including **catastrophic forgetting** (Liao et al., 2024), the trade-off between **retention and forgetting** (Zanzotto et al., 2025), and the lack of **temporal understanding** (Zhong et al., 2023). For example, Wang et al. (2025) demonstrated that standard unlearning techniques often degrade model performance, while MemoryBank (Zhong et al., 2023) introduced a forgetting curve-inspired update mechanism but struggled with multi-modality integration. These gaps underscore the need for a holistic architecture that unifies semantic organization, context-aware forgetting, and reinforcement learning (RL)-optimized parameters.

---

### **1.2 Research Objectives**  
This proposal aims to:  
1. Design a **dual-pathway semantic memory architecture** combining a hierarchical semantic network for concept organization and a dynamic forgetting mechanism guided by recency, relevance, and importance.  
2. Implement RL to optimize forgetting parameters, balancing task performance and memory efficiency.  
3. Validate the architecture’s ability to reduce context window requirements, improve coherence over long-term interactions, and align with human-like forgetting patterns.  
4. Develop robust evaluation metrics for memory unlearning, extending benchmarks like UGBench (Wang et al., 2025) to measure both knowledge retention and task fidelity.

---

### **1.3 Significance**  
This research will advance LLM agents in three key ways:  
- **Efficiency**: By pruning non-essential information, the model will reduce computational overhead and adapt to limited context windows.  
- **Long-Term Coherence**: Enhanced memory management will improve performance in extended tasks (e.g., multi-session research or legal case analysis).  
- **Ethical Compliance**: The forgetting mechanism will enable targeted unlearning of sensitive data (e.g., GDPR compliance), addressing concerns in applications like healthcare and finance.  

Furthermore, this work bridges cognitive science and machine learning by formalizing human memory principles (e.g., synaptic pruning and consolidation) into algorithmic components.

---

## **2. Methodology**

### **2.1 System Overview**  
The proposed architecture consists of two core components:  
1. **Semantic Network**: A hierarchical graph of concepts, where nodes represent entities/concepts and edges encode relationships (e.g., similarity, causality).  
2. **Dynamic Forgetting Mechanism**: A RL-optimized module that evaluates nodes for pruning based on:  
   - **Recency**: Time since the concept was last accessed.  
   - **Relevance**: Similarity to current task context (measured via cosine similarity between embeddings).  
   - **Importance**: Task-specific value learned through RL rewards.  

---

### **2.2 Semantic Network Design**  
#### **2.2.1 Graph Representation**  
The semantic network is modeled as a weighted directed graph $ G = (V, E) $, where:  
- $ V $: Nodes represent embedded concepts (e.g., $ \mathbf{v}_i \in \mathbb{R}^d $ for a pre-trained LLM’s embedding space).  
- $ E $: Edges $ e_{ij} $ encode relationships, computed via attention weights or knowledge graph triples (e.g., from ConceptNet).  

New information is integrated by:  
1. **Embedding Mapping**: Input sentences are converted into embeddings $ \mathbf{x}_t $ using a frozen LLM encoder (e.g., BERT).  
2. **Clustering**: Similar nodes are clustered using hierarchical agglomerative clustering (HAC) to form abstract semantic layers.  
3. **Edge Update**: Relationships are recalculated using the dot product between cluster centroids:  
   $$ 
   e_{ij} = \sigma(\mathbf{c}_i^\top \mathbf{c}_j + b) 
   $$  
   where $ \sigma $ is the sigmoid function, $ \mathbf{c}_i $ is the centroid of cluster $ i $, and $ b $ is a bias term.  

#### **2.2.2 Temporal Compression**  
Episodic memories (detailed events) are compressed into semantic concepts over time using an exponential decay model:  
$$ 
\mathbf{c}_i^{(t)} = \lambda \mathbf{c}_i^{(t-1)} + (1-\lambda)\mathbf{x}_t 
$$  
where $ \lambda \in [0,1] $ controls the rate of compression (higher values prioritize stability over novelty).

---

### **2.3 Dynamic Forgetting Mechanism**  
#### **2.3.1 Forget Score Computation**  
Each node $ i $ in $ G $ is assigned a **forget score** $ F_i $:  
$$ 
F_i = \alpha R(t_i) + \beta S(r_i, q) + \gamma I_i 
$$  
where:  
- $ R(t_i) = e^{-\delta (t - t_i)} $: Recency decay ($ \delta $: decay rate, $ t_i $: last access time).  
- $ S(r_i, q) = \frac{\mathbf{r}_i^\top \mathbf{q}}{\|\mathbf{r}_i\|\|\mathbf{q}\|} $: Relevance score (cosine similarity between node embedding $ \mathbf{r}_i $ and query vector $ \mathbf{q} $).  
- $ I_i $: Importance weight learned via RL (see Section 2.4).  
- $ \alpha + \beta + \gamma = 1 $: Normalized weights.  

Nodes with $ F_i < \tau $ (threshold) are pruned.

#### **2.3.2 RL-Optimized Thresholding**  
The parameters $ \alpha, \beta, \gamma, \tau $ are optimized using proximal policy optimization (PPO) with:  
- **State Space**: Current semantic graph $ G $, task context embedding $ \mathbf{q} $, and interaction history.  
- **Action Space**: Adjustments to $ \alpha, \beta, \gamma, \tau $.  
- **Reward Function**:  
  $$ 
  \text{Reward} = \frac{\text{Task Accuracy}}{\lambda_{\text{context}} \cdot |V| + \lambda_{\text{forget}}} 
  $$  
  where $ \lambda_{\text{context}} $ penalizes large memory sizes and $ \lambda_{\text{forget}} $ encourages removal of irrelevant nodes.  

---

### **2.4 Experimental Design**  
#### **2.4.1 Datasets**  
- **LongQA**: A dataset for multi-turn question-answering with evolving context (modified from MemoryBank).  
- **Multi-Session Planning**: Simulated task logs for project management, requiring knowledge persistence across sessions.  
- **Legal Case Analysis**: Contracts and regulations with periodic updates (e.g., GDPR amendments).  

#### **2.4.2 Baselines**  
- **Standard LLM (no memory)**: GPT-4 with 8k context window.  
- **RecallM** (Kynoch et al., 2023): Temporal memory update architecture.  
- **MemoryBank** (Zhong et al., 2023): Forgetting curve-based memory.  
- **M+** (Wang et al., 2025): Retriever-augmented long-term memory.  

#### **2.4.3 Evaluation Metrics**  
1. **Task Accuracy**: F1 score on downstream tasks.  
2. **Memory Efficiency**: Reduction in context size ($ |V| $) over time.  
3. **Coherence Score**: Consistency of responses across sessions (BERTScore similarity).  
4. **Forgetting Efficacy**: UGBench metrics (Wang et al., 2025) for unlearning accuracy.  
5. **Latency**: Inference time per token.  

#### **2.4.4 Ablation Studies**  
- **Component Analysis**: Compare performance with/without RL, semantic clustering, or dynamic thresholds.  
- **Scalability**: Test on contexts up to 100k tokens.  

---

## **3. Expected Outcomes & Impact**

### **3.1 Technical Contributions**  
- **Biological Plausibility**: A memory architecture that mirrors synaptic pruning, enabling human-like forgetting while preserving task-critical knowledge.  
- **RL-Driven Optimization**: Demonstration of adaptive forgetting parameter learning to balance accuracy and memory constraints.  
- **Benchmark Advancement**: Enhanced evaluation protocols for unlearning and long-term coherence, extending UGBench and LongQA.  

### **3.2 Performance Metrics**  
We expect the proposed architecture to:  
- Achieve **15-20% higher coherence scores** over MemoryBank and M+ in multi-session tasks.  
- Reduce **context usage by 50%** without degrading task accuracy (e.g., Legal Case Analysis).  
- Outperform ReLearn (Xu et al., 2025) in forgetting efficacy by 10-15% on UGBench.  

### **3.3 Scientific and Societal Impact**  
- **Scientific**: Establish a theoretical framework connecting neural memory consolidation in humans and LLMs.  
- **Applications**: Enable LLMs to function in dynamic domains (e.g., medicine, law) where outdated information must be purged.  
- **Ethics**: Provide GDPR/CCPA-compliant unlearning mechanisms to remove sensitive data without retraining.  

### **3.4 Risks and Mitigation**  
- **Over-Forgetting**: Critical nodes may be erroneously pruned. Mitigation: Redundant storage of high-importance concepts.  
- **Bias Amplification**: Forgetting could disproportionately remove underrepresented concepts. Mitigation: Fairness-aware RL rewards.  

---

## **4. Conclusion**  
This proposal outlines a biologically-inspired memory architecture that addresses critical challenges in LLM agents, including catastrophic forgetting, context saturation, and ethical unlearning. By integrating a hierarchical semantic graph with RL-optimized forgetting, the work bridges cognitive science and machine learning, advancing the deployment of LLMs in long-term, real-world applications. Future work will explore multi-modal extensions (e.g., integrating vision with semantic memory) and neurobiological analogs like dopamine-mediated reinforcement signals.  

---

**Word Count**: ~2000