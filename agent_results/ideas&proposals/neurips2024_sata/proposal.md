Okay, here is a research proposal for VeriMem, structured according to your requirements and incorporating the provided information.

## Research Proposal

**1. Title:** VeriMem: A Veracity-Driven Memory Architecture for Enhancing Trustworthiness in LLM Agents

**2. Introduction**

*   **Background:** Large Language Model (LLM) agents, capable of complex reasoning, tool use, and interaction over extended periods, represent a significant leap in artificial intelligence (Russell & Norvig, 2020). A crucial component enabling their long-term coherence and adaptability is persistent memory, allowing them to store and recall past interactions, observations, and derived knowledge (Xu et al., 2025; Sumers et al., 2023). However, the very nature of LLMs, prone to generating plausible-sounding but factually incorrect information (hallucinations) and reflecting or amplifying societal biases present in their training data (Bender et al., 2021), poses a critical safety and trustworthiness challenge when integrated into memory systems. Unchecked, these agents risk populating their memory stores with inaccuracies and biases, which are then retrieved and acted upon in subsequent interactions, potentially leading to cascading errors, harmful outcomes, and erosion of user trust. This is particularly concerning in high-stakes domains such as healthcare decision support, financial advising, or educational tutoring, where reliability and factual accuracy are paramount.

*   **Problem Statement & Literature Gap:** Current research on LLM agent memory primarily focuses on efficient storage, retrieval mechanisms, and structural organization, often using vector databases for semantic similarity searches (Xu et al., 2025; Sumers et al., 2023). While techniques exist to mitigate hallucinations at inference time, such as retrieval-augmented generation (RAG) or input validation (Zou et al., 2024; Ding et al., 2024), they often do not explicitly address the *veracity* of information *already stored* in the agent's persistent memory. Information ingested from prior interactions or generated internally by the agent might be stored without rigorous verification. Although recent conceptual work highlights the need for veracity awareness in memory (Doe et al., 2024; Brown et al., 2023; Lee et al., 2024; Thompson et al., 2025; Chen et al., 2024; Harris et al., 2023), a comprehensive, practical architecture that dynamically assesses, updates, and utilizes veracity information within a standard agent reasoning loop is still lacking. Existing approaches often lack detailed mechanisms for continuous verification, dynamic thresholding based on context, or explicit uncertainty handling linked to agent actions. Key challenges remain in developing reliable and efficient veracity scoring, balancing trustworthiness with adaptability, implementing lightweight fact-checking, mitigating bias propagation through memory, and integrating such systems into existing agent frameworks (as highlighted in the literature review).

*   **Proposed Solution: VeriMem:** This research proposes **VeriMem**, a novel memory architecture designed to enhance the safety and trustworthiness of LLM agents by explicitly managing the veracity of stored information. VeriMem augments standard agent memory modules by associating each memory entry with a quantifiable **veracity score**. This score is assigned upon memory creation and periodically updated through lightweight, automated fact-checking against designated trusted external corpora (e.g., curated knowledge bases, reliable news archives, scientific repositories). During memory retrieval, VeriMem employs a **dynamic veracity threshold**: memories falling below this threshold are flagged and trigger specific actions, such as re-validation against external sources, replacement with freshly retrieved and verified information, or complete disregard if verification fails. Furthermore, VeriMem incorporates an **uncertainty estimation** mechanism, flagging memory recalls where veracity is low or conflicting evidence exists. This uncertainty signal can prompt the agent to explicitly state its uncertainty, seek clarification, request human oversight, or initiate subroutines to gather further evidence before acting upon potentially unreliable information. VeriMem is designed to be integrated seamlessly within established agent reasoning frameworks, such as the ReAct paradigm (Yao et al., 2022).

*   **Research Objectives:** This research aims to:
    1.  Design and develop the VeriMem architecture, including algorithms for veracity score assignment, updating, dynamic thresholding, and uncertainty estimation.
    2.  Implement VeriMem as a modular component compatible with standard LLM agent frameworks (initially focusing on a ReAct-style agent).
    3.  Develop and curate appropriate trusted external corpora and efficient fact-checking mechanisms suitable for continuous memory verification.
    4.  Empirically evaluate VeriMem's effectiveness in reducing hallucinations and mitigating bias propagation compared to baseline memory systems on tasks requiring long-term memory recall.
    5.  Analyze the trade-offs between veracity assurance, task performance, and computational overhead introduced by VeriMem.

*   **Significance:** This research directly addresses critical challenges in building safe and trustworthy AI agents, a key focus of the workshop. By mitigating hallucinations and reducing bias originating from unreliable memory, VeriMem aims to significantly enhance agent reliability, particularly for long-term interactions and applications in sensitive domains. It contributes a novel architectural component for agent design, provides methodologies for evaluating memory trustworthiness, and tackles identified challenges in veracity assessment and integration. Successful development of VeriMem would represent a substantial step towards deploying LLM agents that are not only capable but also demonstrably more dependable and aligned with human values.

**3. Methodology**

This section details the research design for developing and evaluating VeriMem.

*   **VeriMem Architecture:**
    *   **Memory Representation:** Each memory unit $m$ in VeriMem will be represented as a tuple:
        $$m = (\text{content}, \mathbf{e}, v, u, t_{\text{created}}, t_{\text{updated}}, s, h)$$
        where:
        *   $\text{content}$ is the textual or structured information stored (e.g., past utterance, observation, derived fact).
        *   $\mathbf{e}$ is the vector embedding of the content for retrieval (e.g., using Sentence-BERT).
        *   $v \in [0, 1]$ is the veracity score, representing the assessed likelihood of the content being factually accurate.
        *   $u \in [0, 1]$ is the uncertainty score, quantifying the confidence in the veracity assessment $v$.
        *   $t_{\text{created}}$ and $t_{\text{updated}}$ are timestamps for creation and last veracity update.
        *   $s$ is metadata about the source of the information (e.g., 'user_input', 'agent_inference', 'external_tool_X').
        *   $h$ is the history of veracity checks (optional, for traceability).

    *   **Veracity Score Assignment (Write Time):** When a new memory $m$ is created:
        *   An initial veracity score $v_0$ is assigned based on heuristics:
            *   High $v_0$ (e.g., 0.9) for memories derived directly from trusted external tools or databases.
            *   Medium $v_0$ (e.g., 0.7) for user inputs (assuming cooperative user, but potentially containing errors).
            *   Lower $v_0$ (e.g., 0.5) for agent's own inferences or summaries, requiring subsequent verification.
        *   Optionally, a quick initial check against a high-confidence KB can be performed for factual claims.

    *   **Veracity Score Update (Continuous Verification):**
        *   **Triggering Updates:** Updates can be triggered periodically (e.g., background process checking older or lower-veracity memories) or event-driven (e.g., when a low-veracity memory is retrieved, or when new potentially contradictory information is encountered).
        *   **Fact-Checking Process:**
            1.  **Claim Extraction:** Identify check-worthy factual claims within the memory `content`. This might involve specific NLP techniques or reliance on the LLM's own capabilities guided by prompts.
            2.  **Evidence Retrieval:** Formulate queries based on the claim and retrieve relevant snippets from pre-defined **Trusted External Corpora (TECs)**. TECs will include sources like a recent Wikipedia dump, Wikidata, curated news archives (e.g., Reuters, AP), and potentially domain-specific databases (e.g., medical KBs like MedLine for a healthcare agent). This requires careful selection and possibly API access management.
            3.  **Verification:** Use a Natural Language Inference (NLI) model (e.g., fine-tuned DeBERTa or ELECTRA) to assess the relationship between the memory claim and the retrieved evidence. The model outputs probabilities for {Entailment, Contradiction, Neutral}.
                $$P(\text{relationship} | \text{claim}, \text{evidence})$$
            4.  **Score Update:** Update the veracity score $v$ based on the NLI outcome. A simple approach: increase $v$ for entailment, decrease $v$ significantly for contradiction, and slightly decrease or keep stable for neutral/insufficient evidence. A Bayesian update rule could also be employed:
                $$v_{new} = P(\text{True} | \text{evidence}) \approx \frac{P(\text{Entailment}) \cdot v_{old}}{P(\text{Entailment}) \cdot v_{old} + P(\text{Contradiction}) \cdot (1 - v_{old}) + P(\text{Neutral}) \cdot v_{old}}$$
                (This is a simplified example; the exact update rule needs refinement).
        *   **Efficiency:** To ensure "lightweight" checking, we will explore strategies like: prioritizing checks for frequently accessed or low-veracity memories, using smaller/distilled NLI models, caching verification results for identical claims, and optimizing evidence retrieval (e.g., pre-indexing TECs).

    *   **Veracity-Aware Retrieval:**
        1.  **Initial Retrieval:** Given a query $q$, retrieve candidate memories $\{m_1, m_2, ..., m_k\}$ using standard semantic similarity search on embeddings $\mathbf{e}$.
        2.  **Veracity Filtering & Augmentation:** For each candidate $m_i$:
            *   Check its veracity score $v_i$ against a **dynamic veracity threshold** $T_v$. $T_v$ can be context-dependent (e.g., higher for critical tasks) or adaptive (e.g., based on the agent's overall performance or recent hallucination rate).
            *   **If $v_i \ge T_v$**: The memory is considered reliable and can be used directly.
            *   **If $v_i < T_v$**: The memory is flagged as potentially unreliable. The agent can:
                *   **Trigger Re-validation:** Perform an immediate fact-check (as described above). If successful, update $v_i$ and use the memory if $v_i$ now exceeds $T_v$.
                *   **Perform External Lookup:** Disregard $m_i$ and instead query the TECs or trusted web search APIs directly for the required information related to query $q$. The newly retrieved information can be used and potentially stored as a new memory with high initial veracity.
                *   **Utilize with Uncertainty:** Use the memory but explicitly signal uncertainty (see below).

    *   **Uncertainty Estimation and Handling:**
        *   The uncertainty score $u$ can be calculated based on:
            *   The veracity score itself (e.g., $u = 1 - |2v - 1|$, higher uncertainty near $v=0.5$).
            *   Variance or entropy from multiple fact-checks if available.
            *   Detection of conflicting information during retrieval or fact-checking.
        *   **Handling High Uncertainty:** If a retrieved memory has high uncertainty ($u > T_u$, where $T_u$ is an uncertainty threshold), or if re-validation/external lookup yields conflicting results:
            *   The agent's reasoning process (e.g., ReAct prompt) is injected with an uncertainty signal (e.g., appending "[UNCERTAINTY DETECTED]" to the observation).
            *   The agent can be prompted to: state its uncertainty to the user, ask clarifying questions, request human input/verification, or initiate a sub-task to gather more definitive evidence.

    *   **Integration with ReAct:** VeriMem will be integrated into a ReAct (Yao et al., 2022) agent framework. The standard `RetrieveMemory(query)` action will be replaced by `VeriMemRetrieve(query)`. The Observation step will now include not just the memory content but also its veracity status (e.g., "Retrieved memory M1 (content='...', veracity=0.9)" or "Retrieved memory M2 (content='...', veracity=0.3). Flagged as unreliable. Attempting external lookup... Found external info E1 (source=Wikipedia, content='...')"). The agent's LLM prompt will be engineered to understand and utilize this veracity information in its Thought process to decide the next Action.

*   **Data Collection and Preparation:**
    *   **Task Datasets:** We will use existing datasets adapted for long-term agent interaction, potentially including:
        *   Dialogue-based QA (e.g., portions of QReCC or TREC CAsT, modified to span multiple sessions).
        *   Multi-session planning or task completion benchmarks (simulated or using existing frameworks like ALFWorld, potentially extended).
        *   Code debugging tasks over multiple interactions where past context (code state, previous errors) is stored in memory.
        *   Synthetic datasets designed to explicitly test hallucination injection and propagation through memory.
    *   **Trusted External Corpora (TECs):** We will compile TECs from:
        *   A recent static dump of English Wikipedia.
        *   Wikidata knowledge graph.
        *   Selected subsets of news archives (e.g., Reuters Corpus, Associated Press archives) focusing on factual reporting.
        *   Relevant scientific repositories (e.g., PubMed abstracts) if using a biomedical task.
        *   Access to a trusted search engine API (e.g., Google Search API, Bing Search API) with domain filtering will be explored for dynamic, up-to-date information.

*   **Experimental Design:**
    *   **Baselines:**
        1.  **Vanilla ReAct Agent:** A standard ReAct agent using a simple vector store for memory with cosine similarity retrieval, without any veracity checking.
        2.  **ReAct + Standard RAG:** ReAct agent that uses external retrieval (like the TECs) *only* when its internal knowledge is insufficient (e.g., based on confidence scores from the LLM itself) but does not verify its persistent memory content.
        3.  **(Optional/Stretch Goal):** ReAct + A-MEM (Xu et al., 2025), if implementation is feasible, to compare against advanced memory structuring without explicit veracity focus.
    *   **Tasks:** The selected tasks (dialogue QA, planning, debugging) will be designed such that success relies heavily on recalling accurate information from previous interactions/observations stored in memory. We will specifically design scenarios where the agent might encounter or generate misinformation to store.
    *   **Evaluation Metrics:**
        1.  **Hallucination Rate:** Assessed using:
            *   **Factuality Scores
            **: Metrics like FactScore (Min et al., 2023) or similar, comparing agent outputs relying on memory against ground truth or TEC evidence.
             *   **Human Evaluation:** Manual annotation by trained evaluators assessing the factual accuracy of memory-dependent agent responses on a Likert scale or binary correct/incorrect basis.
        2.  **Bias Mitigation:**
            *   **Stereotype/Bias Benchmarks:** Using benchmarks like WinoBias (Zhao et al., 2018) or SEAT (May et al., 2019), adapted to measure bias propagation. We will test if the agent's memory recall disproportionately retrieves or reinforces biased information previously encountered or generated, comparing VeriMem against baselines after exposure to biased inputs. Measure stereotype association scores in generated text.
            *   **Qualitative Analysis:** Reviewing memory contents and agent outputs for biased statements or reasoning patterns.
        3.  **Task Performance:**
            *   **Accuracy/Success Rate:** Task-specific metrics (e.g., QA accuracy, planning task completion rate, code debugging success rate).
            *   **Efficiency:** Number of turns/steps to complete the task.
        4.  **Computational Overhead:**
            *   **Latency:** Measure the average time taken per agent step, particularly focusing on memory write and retrieval operations involving verification.
            *   **Resource Usage:** Monitor CPU/GPU usage and memory footprint.
        5.  **Veracity Score Utility:** Analyze the correlation between assigned veracity scores ($v$) and actual ground truth correctness. Evaluate the effectiveness of the dynamic threshold $T_v$ and uncertainty handling $T_u$.

    *   **Ablation Studies:** We will perform ablation studies by disabling specific components of VeriMem (e.g., periodic updates, dynamic thresholding, uncertainty handling) to understand their individual contributions.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A Functional VeriMem Prototype:** A demonstrable implementation of the VeriMem architecture integrated within a ReAct-style LLM agent.
    2.  **Empirical Validation:** Quantitative results demonstrating that VeriMem significantly reduces hallucination rates and mitigates bias propagation in long-term agent memory compared to baseline approaches on selected benchmark tasks.
    3.  **Performance Analysis:** Clear data on the trade-offs involved, quantifying the impact of VeriMem on task success rates, interaction efficiency, and computational overhead.
    4.  **Insights into Veracity Management:** Findings on the effectiveness of different veracity scoring strategies, fact-checking mechanisms (including efficiency), dynamic thresholding approaches, and uncertainty handling protocols.
    5.  **Contribution to Agent Safety:** A concrete architectural contribution towards building more reliable and controllable LLM agents.
    6.  **Open-Source Code/Library (Potential):** Release of the VeriMem module to facilitate further research.

*   **Impact:**
    *   **Enhanced Trustworthiness:** By directly tackling the reliability of agent memory, VeriMem aims to increase user trust and confidence in LLM agents, making them more suitable for real-world deployment.
    *   **Improved Safety:** Reducing the likelihood of agents acting on false or biased information stored in their memory directly contributes to safer AI systems, especially in sensitive application domains.
    *   **Advancement in Agent Architectures:** VeriMem provides a novel architectural component that addresses a critical limitation in current agent designs, potentially influencing future research on cognitive architectures for AI (Sumers et al., 2023).
    *   **Alignment with Workshop Goals:** This research directly addresses the workshop's core themes of "safe reasoning and memory," "controlling agents" (by intervening based on veracity), and "agent evaluation and accountability" (by making memory veracity explicit and checkable).
    *   **Foundation for Future Work:** This work can serve as a foundation for exploring more advanced veracity mechanisms, multi-modal memory verification (extending beyond text), privacy-preserving fact-checking, and understanding the interplay between memory veracity and multi-agent interactions.

In conclusion, the VeriMem proposal outlines a timely and crucial research direction to improve the reliability of LLM agents. By introducing a principled architecture for managing memory veracity, this work promises to make significant contributions to the development of safe, trustworthy, and dependable agentic AI systems.

**References** (Using placeholders as specific citations beyond the lit review are illustrative)

*   Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? *FAccT*.
*   Brown, A., White, R., & Green, M. (2023). Trustworthy Memory Management in AI Agents. *arXiv:2311.04567*.
*   Chen, O., Brown, D., & Wilson, S. (2024). Fact-Checking Mechanisms in LLM Memory Systems to Prevent Hallucinations. *arXiv:2404.08923*.
*   Ding, H., Pang, L., Wei, Z., Shen, H., & Cheng, X. (2024). Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models. *arXiv:2402.10612*.
*   Doe, J., Smith, J., & Johnson, E. (2024). Veracity-Aware Memory Systems for Large Language Models. *arXiv:2405.09876*.
*   Harris, W., Clark, E., & Lewis, B. (2023). Dynamic Veracity Thresholds in LLM Memory Retrieval for Bias Reduction. *arXiv:2312.05678*.
*   Lee, S., Kim, D., & Martinez, L. (2024). Bias Mitigation in LLM Memory Systems via Veracity Scoring. *arXiv:2403.11234*.
*   May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). On Measuring Social Biases in Sentence Encoders. *NAACL*.
*   Min, S., et al. (2023). FactScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. *ICML*.
*   Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson.
*   Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023). Cognitive Architectures for Language Agents. *arXiv:2309.02427*.
*   Thompson, M., Adams, R., & Liu, K. (2025). Enhancing LLM Agent Reliability through Veracity-Aware Memory Architectures. *arXiv:2501.06789*.
*   Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). A-MEM: Agentic Memory for LLM Agents. *arXiv:2502.12110*.
*   Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv:2210.03629*.
*   Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2018). Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. *NAACL*.
*   Zou, X., Wang, Y., Yan, Y., Huang, S., Zheng, K., Chen, J., Tang, C., & Hu, X. (2024). Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models. *arXiv:2410.03577*.