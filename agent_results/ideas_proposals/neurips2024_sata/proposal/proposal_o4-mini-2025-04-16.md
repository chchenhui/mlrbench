1. Title  
VeriMem: A Veracity-Driven Memory Architecture for Safe and Trustworthy LLM Agents  

2. Introduction  
Background  
Recent advances in large language model (LLM) agents have demonstrated remarkable capabilities in tasks ranging from open-domain dialogue to complex decision-making. Core to these successes is the integration of persistent memory modules that allow agents to reference, update, and build upon prior interactions. Systems such as A-MEM (Xu et al., 2025) dynamically organize knowledge networks to enhance adaptability, while MemVR (Zou et al., 2024) reinjects visual prompts to mitigate hallucinations in multimodal settings. However, as agents become deployed in high-stakes domains—healthcare, finance, legal—their tendency to hallucinate or propagate latent biases undermines trust and safety.  

Prior work on veracity-aware memories (Doe et al., 2024; Brown et al., 2023; Lee et al., 2024) has laid the groundwork by assigning confidence scores and periodically fact-checking stored entries against trusted corpora. Dynamic thresholding approaches (Harris et al., 2023) further refine retrieval safety by revalidating low-score items. Despite these advances, existing systems often suffer one or more of the following limitations:  
• Heavy computational overhead in continuous fact-checking that degrades runtime performance.  
• Rigid verification pipelines that inhibit swift adaptation to novel information.  
• Incomplete handling of uncertainty during retrieval, leading to unchecked propagation of low-quality memories.  

Research Objectives  
We propose VeriMem, a novel veracity-driven memory architecture designed to:  
1. Assign and update veracity scores to memory entries in a lightweight, continuous manner.  
2. Dynamically filter and prioritize high-veracity memories during retrieval, while handling low-veracity recalls through on-the-fly revalidation or external lookup.  
3. Estimate retrieval uncertainty and seamlessly integrate human-in-the-loop oversight when confidence is low.  
4. Validate the approach on multi-step reasoning tasks—dialogue history management, code debugging sequences, and knowledge-based QA—measuring hallucination rates, bias amplification, and task performance against leading baselines (A-MEM, Rowen, MemVR).  

Significance  
By endowing LLM agents with veracity awareness and uncertainty estimation, VeriMem directly addresses key safety challenges in agentic AI: preventing the spread of hallucinations, mitigating bias amplification, and enabling controlled, accountable long-term interactions. Its lightweight fact-checking design ensures practicality in real-time deployments, while its modular integration within the ReAct framework promotes broad applicability across agent architectures. VeriMem thus paves the way for more trustworthy intelligence in domains where errors carry real-world consequences.  

3. Methodology  
3.1 Overview of VeriMem Architecture  
VeriMem augments a standard LLM agent pipeline with three core modules: (1) Veracity Tagger (write time), (2) Continuous Veracity Updater (background), and (3) Veracity-Aware Retriever (read time), supplemented by an Uncertainty Estimator and Human-in-the-Loop Controller. Figure 1 (omitted) depicts the dataflow: user input → memory read & reasoning (with ReAct style) → action → memory write with initial veracity tagging → periodic veracity update.  

3.2 Veracity Tagger (Initial Score Assignment)  
At write time, each new memory entry $m_i$ is annotated with an initial veracity score $v_i^{(0)}\in[0,1]$. We compute $v_i^{(0)}$ as a weighted ensemble of source-based and content-based credibility measures:  
$$
v_i^{(0)} = \frac{\sum_{j=1}^J w_j\cdot s_j(m_i)}{\sum_{j=1}^J w_j}.
$$  
Here $s_j(m_i)$ represents the $j$-th credibility signal (e.g., source reputation, internal consistency score, external verification confidence), and $w_j$ its relative weight. For example:  
• Source reputation: normalized trust score of the information origin (user vs. web API).  
• Semantic consistency: cosine similarity between $m_i$ and corroborating knowledge-base entries.  
• Model-internal reliability: confidence of auxiliary entailment models that $m_i$ is factual.  

Weights $w_j$ are calibrated on a held-out validation set via grid search to optimize initial retrieval precision.  

3.3 Continuous Veracity Updater  
To adapt to evolving facts, VeriMem performs lightweight, periodic fact-checks on stored memories. At each update epoch $t\to t+1$, for each $m_i$ we obtain a fresh external veracity estimate $\hat v_i^{(t+1)}$ by querying trusted corpora (e.g., Wikipedia API, domain-specific knowledge bases, news feeds) through a fast entailment classifier. We then apply exponential smoothing:  
$$
v_i^{(t+1)} = \alpha\,v_i^{(t)} + (1 - \alpha)\,\hat v_i^{(t+1)},
$$  
where $\alpha\in[0,1]$ balances memory inertia against new evidence. Higher $\alpha$ preserves historical confidence; lower $\alpha$ accelerates adaptation to corrected or outdated information.  

To limit overhead, updates are batched and scheduled based on memory “age” and “access frequency.” Specifically, each $m_i$ is assigned an update priority  
$$
p_i = \lambda_1\,\text{Age}(m_i) + \lambda_2\,\text{Freq}(m_i),
$$  
and only the top $K$ prioritized entries undergo checking per cycle. Hyperparameters $(\alpha,\lambda_1,\lambda_2,K)$ are tuned to achieve a target throughput.  

3.4 Veracity-Aware Retrieval  
When the agent requires historical context, VeriMem retrieves a candidate set of memories and filters them using a dynamic veracity threshold $\tau$. We compute $\tau$ at retrieval time as:  
$$
\tau = \mu_v - \gamma\,\sigma_v,
$$  
where $\mu_v$ and $\sigma_v$ are the mean and standard deviation of $\{v_i\}$ over all memories, and $\gamma$ a sensitivity parameter. This adaptive threshold ensures that only relatively reliable memories are considered.  

Retrieval proceeds in two stages:  
1. Candidate Selection: use semantic embeddings to select the top $N$ entries most relevant to the current query.  
2. Veracity Filtering: from those $N$, retain only entries with $v_i \ge \tau$.  

If fewer than $M$ memories pass the threshold, VeriMem either (a) re-validates low-score entries on-the-fly, raising their scores if corroborated, or (b) performs an external lookup to fetch real-time factual content. The choice is governed by a retrieval cost budget and latency constraints.  

3.5 Uncertainty Estimation & Human Oversight  
Even high-veracity memories may lead to low-confidence inferences. We compute an uncertainty measure $H$ for each reasoning step based on token-level entropy:  
$$
H = -\sum_{w\in V} P(w\mid \text{context})\,\log P(w\mid \text{context}),
$$  
where $V$ is the model’s vocabulary. If $H > H_{\max}$, the system flags the decision for human review or triggers an evidence-gathering subroutine (e.g., re-ask the user, fetch additional sources). This mechanism integrates seamlessly into the agent’s ReAct reasoning loop, allowing controlled escalation for critical or ambiguous queries.  

3.6 Integration with ReAct Reasoning Loop  
We adopt the ReAct paradigm (Yao et al., 2022) to interleave reasoning steps, actions, and memory operations. Algorithm 1 below summarizes the overall loop:  

```  
Algorithm 1: VeriMem-Enhanced ReAct Loop  
Input: user query q, memory store M  
Output: agent response r  

1.  context ← RetrieveVeracious(q, M)  
2.  while not Done do  
3.     action ← LLMReason(context)  
4.     if action = “TOOL_CALL” then  
5.         tool_result ← ExecuteTool(action)  
6.         context ← context ∪ tool_result  
7.     end if  
8.     if LLMConfidence(context) < Hmax then  
9.         escalate_for_review()  
10.        break  
11.    end if  
12. end while  
13. r ← LLMRespond(context)  
14. M ← M ∪ {(context, initialVeracity(context))}  
15. return r  
```  

This design ensures memory reads are veracity-aware and memory writes carry veracity tags. Human oversight is invoked dynamically based on uncertainty.  

3.7 Experimental Design  
Datasets and Tasks  
We evaluate VeriMem on three multi-step tasks requiring persistent memory:  
• Dialogue History Tracking: Wizard-of-Wikipedia styled conversations where maintaining factual consistency across turns is critical.  
• Code Debugging Sequences: Problem-solution dialogues in CodeContests, testing the agent’s ability to recall prior debugging steps.  
• Knowledge-based QA: Sequential queries on dynamic news articles and Wikipedia updates, simulating evolving real-world contexts.  

Baselines  
• Standard LLM agent with naive persistent memory (no veracity).  
• A-MEM (Xu et al., 2025): structured agentic memory without explicit veracity scoring.  
• Rowen (Ding et al., 2024): adaptive retrieval for hallucination mitigation.  
• Veracity-Aware Memory (Doe et al., 2024): static thresholding approach.  

Evaluation Metrics  
1. Hallucination Rate: percentage of model statements not grounded in any valid source (automated via ground-truth alignment + human spot checks).  
2. Task Accuracy: final answer correctness for QA and debugging; coherence and relevance in dialogue (BLEU, ROUGE, human rating).  
3. Bias Amplification Index: change in sentiment or demographic bias when recalling memory, measured via established fairness metrics (Lee et al., 2024).  
4. Retrieval Latency & Throughput: average retrieval time per memory read, and maximum queries per second.  
5. Human Escalation Frequency: rate of uncertainty flags requiring oversight.  

Ablations  
We conduct ablation studies on:  
• Smoothing factor $\alpha$ (0.5, 0.7, 0.9)  
• Update batch size $K$ (10, 50, 100)  
• Threshold sensitivity $\gamma$ (0.5, 1.0, 1.5)  
• Uncertainty cutoff $H_{\max}$ (empirically determined).  

Statistical Analysis  
Results will be averaged over 5 random seeds; significance tested with paired t-tests (p<0.05).  

Implementation Details  
• Backbone: GPT-4 via OpenAI API (temperature=0.0 for deterministic reasoning).  
• Memory Store: FAISS vector index for semantic retrieval + key-value store for veracity scores.  
• External Fact-Checking: Wikipedia REST API and a distilled DeBERTa-based entailment classifier.  
• Infrastructure: GPUs for embedding and entailment tasks; retrieval and LLM hosted on cloud instances to measure realistic latency.  

4. Expected Outcomes & Impact  
We anticipate that VeriMem will:  
• Reduce hallucination rates by at least 30% relative to standard LLM agents and by 15–20% relative to existing veracity-aware baselines, as measured on dialogue and QA tasks.  
• Improve task accuracy by 8–12% over A-MEM and Rowen, due to the pruning of unreliable memories and dynamic revalidation.  
• Lower bias amplification by 20–25% through targeted exclusion of low-veracity, biased entries during retrieval.  
• Maintain retrieval latency within 50 ms on average for civilian deployments, demonstrating practicality for real-time systems.  

Broader Impact  
VeriMem advances the safety and trustworthiness of agentic AI by systematically embedding veracity and uncertainty awareness into long-term memory. In high-stakes settings—clinical decision support, financial advising, legal assistance—such a memory architecture can prevent the costly propagation of errors and biases. By providing an open-source reference implementation and detailed benchmarking, this work will foster community adoption and spark further research into veracity-driven control methods, human-in-the-loop accountability, and multi-agent safety dynamics.  

In summary, VeriMem offers a principled, scalable solution for mitigating hallucinations and bias in LLM agents, bridging a critical gap between raw language prowess and the stringent reliability demands of real-world applications.