# VeriMem: A Veracity-Aware Memory Architecture for Mitigating Hallucinations in LLM Agents

## 1. Introduction

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex, multi-turn interactions across various domains. These agents increasingly rely on persistent memory systems to maintain context across interactions, enabling more coherent and personalized experiences. However, a critical challenge has emerged: LLM agents frequently hallucinate or propagate biases when recalling information from their memory stores. This phenomenon severely undermines trust in agentic applications, particularly in high-stakes domains such as healthcare, finance, and legal services, where factual accuracy is paramount.

Current memory architectures for LLM agents, while effective at storing and retrieving information, lack robust mechanisms to verify the accuracy of stored memories or to distinguish between factual and potentially hallucinated or biased information. As highlighted by Sumers et al. (2023) in their work on cognitive architectures for language agents, existing systems often prioritize recall efficiency over veracity, leading to the propagation and potential amplification of inaccuracies over time. Similarly, A-MEM (Xu et al., 2025) addresses the organization of memories through interconnected knowledge networks but does not incorporate systematic verification mechanisms.

The problem of hallucinations in LLMs has been approached from various angles. Zou et al. (2024) proposed MemVR for multimodal LLMs, which reinjects visual prompts when uncertainty is detected. Ding et al. (2024) introduced Rowen, which employs adaptive retrieval augmentation to mitigate hallucinations. However, these approaches primarily focus on immediate hallucination detection rather than embedding veracity awareness directly within the memory architecture of LLM agents.

This research proposes VeriMem, a novel veracity-driven memory architecture designed to enhance the trustworthiness of LLM agents by incorporating reliability metrics directly into the memory storage and retrieval processes. VeriMem aims to address the following objectives:

1. Develop a robust framework for assigning and updating veracity scores to memories stored within LLM agent systems.
2. Design efficient mechanisms for fact-checking and validating stored information against trusted external sources.
3. Create a dynamic retrieval system that prioritizes high-veracity memories and triggers appropriate validation processes for potentially unreliable information.
4. Implement uncertainty estimation to flag low-confidence recalls and initiate supplementary evidence-gathering procedures.
5. Evaluate the effectiveness of VeriMem in reducing hallucinations and mitigating biases across a diverse range of agent tasks while maintaining adaptability and performance.

The significance of this research lies in its potential to substantially improve the trustworthiness of LLM agents in real-world applications. By addressing the critical issue of memory reliability, VeriMem could enable the deployment of agentic systems in domains where factual accuracy is essential, such as healthcare decision support, financial advising, and educational applications. Furthermore, by integrating veracity-awareness directly into the memory architecture, VeriMem represents a step toward creating inherently safer and more reliable AI systems, aligning with broader goals of responsible AI development.

## 2. Methodology

The VeriMem architecture integrates veracity assessment directly into the memory lifecycle of LLM agents, spanning from initial memory formation to retrieval and utilization. This section details the components, processes, and evaluation methods that comprise the VeriMem methodology.

### 2.1 System Architecture

VeriMem is designed as a modular architecture that can augment existing LLM agent frameworks. The core components include:

1. **Memory Storage Module**: Extends traditional memory storage with veracity metadata.
2. **Veracity Assessment Engine**: Assigns and updates veracity scores for stored memories.
3. **Dynamic Retrieval Controller**: Manages memory access based on veracity thresholds.
4. **Uncertainty Estimator**: Identifies potential hallucinations during memory recall.
5. **External Validation Interface**: Connects with trusted knowledge sources for fact-checking.

Figure 1 illustrates the interaction between these components:

```
[User Query] → [LLM Agent] → [Memory Retrieval Request]
                    ↓
[Dynamic Retrieval Controller] → [Memory Storage]
                    ↓
[Retrieved Memories + Veracity Scores] → [Uncertainty Estimator]
                    ↓
         [If uncertain] → [External Validation Interface]
                    ↓
[Final Memory Selection] → [LLM Agent] → [Response]
```

### 2.2 Memory Representation

Each memory item in VeriMem is represented as a tuple:

$$M = (c, t, v, s, r)$$

Where:
- $c$ represents the content (text, embedding, or structured data)
- $t$ is the timestamp of creation/last update
- $v$ is the veracity score (range [0,1])
- $s$ denotes the source(s) of information
- $r$ records the revision history

### 2.3 Veracity Assessment

The veracity assessment occurs at two key points: during initial memory formation and periodically through scheduled reviews.

#### 2.3.1 Initial Veracity Assignment

When a new memory is created, its veracity score is calculated as:

$$v_{\text{initial}} = \alpha \cdot v_{\text{source}} + \beta \cdot v_{\text{content}} + \gamma \cdot v_{\text{consistency}}$$

Where:
- $v_{\text{source}}$ evaluates the reliability of information sources (0-1)
- $v_{\text{content}}$ estimates factual correctness through initial verification (0-1)
- $v_{\text{consistency}}$ measures alignment with existing high-veracity memories (0-1)
- $\alpha, \beta, \gamma$ are weighting factors such that $\alpha + \beta + \gamma = 1$

For source reliability, we implement a hierarchical assessment:

$$v_{\text{source}} = 
\begin{cases}
1.0, & \text{for verified external knowledge bases} \\
0.8, & \text{for user-provided information with high confidence} \\
0.6, & \text{for agent-inferred information with supporting evidence} \\
0.4, & \text{for uncorroborated user claims} \\
0.2, & \text{for agent speculations without evidence}
\end{cases}$$

Content verification employs lightweight fact-checking against trusted external knowledge sources:

$$v_{\text{content}} = \lambda \cdot \text{sim}(c, c_{\text{verified}}) + (1-\lambda) \cdot \text{conf}_{\text{LLM}}(c)$$

Where:
- $\text{sim}(c, c_{\text{verified}})$ is the semantic similarity between the memory content and verified information from trusted sources
- $\text{conf}_{\text{LLM}}(c)$ is the LLM's self-reported confidence in the generated content
- $\lambda$ is a balancing parameter (typically 0.7)

Consistency assessment evaluates alignment with existing high-veracity memories:

$$v_{\text{consistency}} = \frac{1}{|H|} \sum_{h \in H} \text{sim}(c, h) \cdot v_h$$

Where:
- $H$ is the set of relevant high-veracity memories
- $\text{sim}(c, h)$ is the semantic similarity between the new content and memory $h$
- $v_h$ is the veracity score of memory $h$

#### 2.3.2 Veracity Updates

Veracity scores are updated periodically through a scheduled review process or triggered by contradictory information. The update function is:

$$v_{\text{updated}} = (1-\delta) \cdot v_{\text{current}} + \delta \cdot v_{\text{verification}}$$

Where:
- $v_{\text{current}}$ is the existing veracity score
- $v_{\text{verification}}$ is the new verification result
- $\delta$ is a learning rate parameter (typically 0.3)

The verification result is obtained through more thorough fact-checking against multiple external sources and evaluation of temporal relevance:

$$v_{\text{verification}} = \frac{\sum_{i=1}^{n} w_i \cdot \text{match}_i(c)}{\sum_{i=1}^{n} w_i} \cdot \text{decay}(t)$$

Where:
- $\text{match}_i(c)$ is the match score against the $i$-th external source
- $w_i$ is the weight assigned to the $i$-th source
- $\text{decay}(t)$ is a temporal decay function based on the age of the memory

### 2.4 Retrieval Mechanism

VeriMem employs a dynamic retrieval process that incorporates veracity awareness into memory selection:

$$R(q, M, \theta) = \text{TopK}(\{m_i \in M | v_i \geq \theta(q)\}, q, k)$$

Where:
- $q$ is the query or context
- $M$ is the set of available memories
- $\theta(q)$ is the dynamic veracity threshold based on query criticality
- $\text{TopK}$ selects the $k$ most relevant memories above the threshold

The dynamic threshold function adapts based on the criticality of the query:

$$\theta(q) = \theta_{\text{base}} + \phi \cdot \text{criticality}(q)$$

Where:
- $\theta_{\text{base}}$ is the base veracity threshold (typically 0.5)
- $\phi$ is a scaling factor
- $\text{criticality}(q)$ is a measure of how critical accurate information is for the given query

When retrieved memories fall below the threshold but are still relevant, VeriMem initiates on-the-fly external validation:

$$
m_{\text{validated}} = 
\begin{cases}
\text{ExternalLookup}(c_i), & \text{if } \text{CanValidate}(c_i) = \text{True} \\
\{m_i, \text{flag}_{\text{uncertain}}\}, & \text{otherwise}
\end{cases}
$$

### 2.5 Uncertainty Estimation

The uncertainty estimator identifies potential hallucinations during memory utilization:

$$U(m, q, \text{context}) = f_{\text{uncertain}}(v_m, \text{consistency}(m, \text{context}), \text{self-eval}(m, q))$$

Where:
- $v_m$ is the veracity score of the memory
- $\text{consistency}(m, \text{context})$ evaluates logical consistency with current context
- $\text{self-eval}(m, q)$ is the LLM's self-evaluation of confidence

When uncertainty is detected above a threshold $\tau$, the agent initiates additional evidence gathering:

$$
\text{Action} = 
\begin{cases}
\text{SeekEvidence}(m), & \text{if } U(m, q, \text{context}) > \tau \\
\text{UseMemory}(m), & \text{otherwise}
\end{cases}
$$

### 2.6 Integration with ReAct Framework

VeriMem integrates into a ReAct-style reasoning loop, augmenting the thought-action-observation cycle with veracity awareness:

1. **Thought**: Includes veracity assessment of relevant memories
2. **Action**: Incorporates evidence-gathering when uncertainty is detected
3. **Observation**: Updates memory veracity based on external validation results

### 2.7 Experimental Design

We evaluate VeriMem through a comprehensive set of experiments designed to assess its effectiveness in reducing hallucinations and biases while maintaining task performance.

#### 2.7.1 Datasets

We employ four datasets representing different agent interaction scenarios:

1. **Long-form Dialogue**: Multi-session conversations (500 dialogues, average 15 turns each)
2. **Code Development**: Programming tasks requiring debugging and iterative development (200 projects)
3. **Information Synthesis**: Research tasks requiring fact integration from multiple sources (300 topics)
4. **Factual Q&A**: Question-answering sessions with varying factual complexity (1000 Q&A pairs)

#### 2.7.2 Baseline Systems

VeriMem is compared against the following baselines:

1. Standard LLM agent without persistent memory
2. LLM agent with basic memory (no veracity awareness)
3. A-MEM system (Xu et al., 2025)
4. Rowen (Ding et al., 2024) with retrieval augmentation

#### 2.7.3 Evaluation Metrics

Performance is assessed using multiple metrics:

1. **Hallucination Rate (HR)**: 
   $$\text{HR} = \frac{\text{Number of factually incorrect statements}}{\text{Total number of factual statements}}$$

2. **Bias Amplification Score (BAS)**:
   $$\text{BAS} = \frac{1}{|B|} \sum_{b \in B} \frac{\text{Bias level in output}}{\text{Bias level in input}}$$
   where $B$ is a set of known bias dimensions

3. **Information Retention (IR)**:
   $$\text{IR} = \frac{\text{Correctly recalled information}}{\text{Total information to be recalled}}$$

4. **Task Completion Rate (TCR)**:
   $$\text{TCR} = \frac{\text{Successfully completed tasks}}{\text{Total tasks}}$$

5. **Human Trust Score (HTS)**: Expert evaluators rate system trustworthiness on a 1-5 scale

#### 2.7.4 Ablation Studies

We conduct ablation studies to assess the contribution of each VeriMem component:

1. VeriMem without dynamic thresholding
2. VeriMem without uncertainty estimation
3. VeriMem without periodic veracity updates
4. VeriMem with different external validation sources

#### 2.7.5 Implementation Details

The experimental implementation uses:
- Base LLM: GPT-4 or equivalent frontier model
- External knowledge sources: Wikipedia, Wikidata, arXiv, news APIs
- Memory storage: Vector database with metadata (Pinecone/Chroma)
- Embedding model: E5-large for semantic similarity calculations
- Fact-checking API: combination of knowledge base lookups and specialized fact-verification models

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

The implementation of VeriMem is expected to yield several significant outcomes:

1. **Reduced Hallucination Rates**: We anticipate a 40-60% reduction in hallucination rates compared to baseline memory systems, particularly in fact-intensive domains. This improvement will derive from the systematic verification of stored information and the prioritization of high-veracity memories during retrieval.

2. **Mitigated Bias Propagation**: By incorporating veracity assessment that considers multiple sources and contradictory evidence, VeriMem should reduce bias amplification by approximately 35% across diverse topics. This will be especially notable in domains where subjective judgments are common.

3. **Enhanced Task Performance**: Despite the additional verification overhead, we expect VeriMem to improve overall task completion rates by 15-20%, particularly for complex tasks requiring accurate recall of multiple facts. This improvement stems from the reduction in reasoning errors based on faulty premises.

4. **Increased Human Trust**: User studies are expected to show a significant increase in trust scores (at least 30% improvement) as users experience fewer instances of obvious hallucinations or biased responses. This improved trustworthiness will be particularly important for high-stakes applications.

5. **Adaptable Verification Depth**: The dynamic thresholding mechanism will demonstrate effective adaptation to query criticality, applying more rigorous verification for sensitive domains while maintaining efficiency for casual interactions.

6. **Transparent Uncertainty Communication**: In cases where veracity cannot be conclusively established, the uncertainty estimation component will provide clear indications of confidence levels, enabling users to make informed decisions about the reliability of the information.

7. **Efficient Integration**: The modular design of VeriMem will allow successful integration with existing agent architectures without requiring fundamental redesigns or extensive retraining of base models.

### 3.2 Research Impact

The successful development of VeriMem will have several important impacts on the field of trustworthy AI agents:

1. **Advancing Memory Architecture Design**: VeriMem introduces a new paradigm for LLM agent memory systems that places veracity at the center of the design, potentially influencing future memory architectures for AI systems.

2. **Enabling High-Stakes Applications**: By significantly reducing hallucinations and biases, VeriMem will make LLM agents more suitable for deployment in critical domains such as healthcare, legal services, and education, where factual accuracy is essential.

3. **Establishing New Evaluation Standards**: The evaluation framework developed for this research, including metrics for hallucination rates and bias amplification, could become standard benchmarks for assessing the trustworthiness of agentic systems.

4. **Bridging Knowledge Management and LLM Research**: VeriMem connects traditional knowledge management approaches (with their emphasis on provenance and verification) with modern LLM-based agents, establishing valuable cross-disciplinary connections.

5. **Informing Regulatory Frameworks**: The principles and mechanisms underlying VeriMem could inform regulatory standards for trustworthy AI systems, particularly regarding memory management and factual accuracy guarantees.

### 3.3 Limitations and Future Work

While VeriMem represents a significant advancement in trustworthy agent memory, several limitations must be acknowledged:

1. **Dependency on External Knowledge Sources**: VeriMem's effectiveness is partially constrained by the coverage and quality of available external knowledge sources. Future work should explore methods for operating effectively in domains with limited reference data.

2. **Computational Overhead**: The verification processes introduce additional computational requirements that may impact system latency. Further optimization could reduce this overhead through more efficient fact-checking methods.

3. **Cultural and Contextual Biases**: While VeriMem addresses many forms of bias, it may still be susceptible to biases embedded in the external knowledge sources themselves. Future research should explore methods for detecting and mitigating these deeper biases.

4. **Expanding to Multimodal Memories**: The current design focuses primarily on textual information. Future extensions should incorporate veracity assessment for multimodal memories including images, audio, and video.

In conclusion, VeriMem represents a promising approach to enhancing the trustworthiness of LLM agents by embedding veracity awareness directly into memory architecture. By systematically addressing hallucinations and biases at the memory level, VeriMem has the potential to significantly advance the development of safe and reliable AI agents for real-world applications.