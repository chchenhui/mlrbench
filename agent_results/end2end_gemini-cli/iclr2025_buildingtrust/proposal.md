### **1. Title: Dynamic Policy Enforcers: Adaptive Guardrails for Trustworthy LLM Applications**

### **2. Introduction**

#### **2.1. Background and Motivation**

The rapid integration of Large Language Models (LLMs) into diverse, high-stakes applications—from healthcare and finance to customer service and legal analysis—has brought the challenge of ensuring their trustworthiness to the forefront of AI research. As these models become integral components of systems used by millions, the potential for them to generate harmful, biased, non-compliant, or factually incorrect content poses significant risks to users, developers, and society. Consequently, building robust safety mechanisms, or "guardrails," has become a critical priority.

Current approaches to LLM safety predominantly rely on static guardrails. These mechanisms range from simple keyword filters and regular expressions to more sophisticated, yet rigid, programmable logic as seen in toolkits like NeMo Guardrails (Rebedea et al., 2023). While foundational, these static systems exhibit a critical flaw: they are brittle and slow to adapt. The landscape of safety is not static; it is a dynamic environment shaped by evolving legal regulations (e.g., GDPR, AI Act), emerging adversarial attack vectors (e.g., complex jailbreaks), and context-specific user policies. When a new threat or compliance requirement emerges, static guardrails often necessitate a slow, resource-intensive cycle of manual re-engineering, data collection, and model retraining or fine-tuning. This inherent latency creates a dangerous window of vulnerability, undermining trust and exposing applications to legal and reputational damage.

Recent research has begun to explore more advanced safety solutions. Frameworks like LlamaFirewall (Chennabasappa et al., 2025) offer layered defenses, while methods like SafeInfer (Banerjee et al., 2024) and InferenceGuard (Ji et al., 2025) perform safety alignment at decoding time. Others, such as GuardAgent (Xiang et al., 2024) and RSafe (Zheng et al., 2025), use a secondary LLM as a safeguard. However, a significant gap remains. Many of these approaches either focus on adapting to new *threats* through continuous learning (Luo et al., 2025; Wang et al., 2025) or require developers to write new code or complex prompts to update safety logic. There is a pressing need for a system that can adapt to new declarative *policies* in real-time, guided by simple, human-readable instructions.

This proposal addresses this gap by introducing the concept of a **Dynamic Policy Enforcer**: a specialized, smaller LLM that acts as a real-time validation layer for a primary, large-capability LLM. The core innovation lies in training this enforcer to interpret and apply safety policies provided as natural language text at inference time. Instead of retraining a model to learn a new rule, a system administrator can simply update the enforcer's policy prompt. This paradigm shifts LLM safety from a static, code-driven process to a dynamic, semantically-driven one, enabling instantaneous, zero-shot adaptation to new guidelines and fostering perpetually compliant and trustworthy LLM applications.

#### **2.2. Research Objectives**

This research aims to design, build, and evaluate a framework for dynamic, policy-driven LLM guardrails. The primary objectives are:

1.  **To develop the Dynamic Policy Enforcer (DPE) framework:** We will design and implement a two-model architecture where a smaller, efficient enforcer LLM monitors and validates the output of a primary LLM based on a dynamically supplied natural language policy.
2.  **To train a robust and efficient policy interpretation model:** We will create a high-quality dataset and employ state-of-the-art fine-tuning techniques to train the enforcer model to accurately interpret and apply complex, nuanced policies to LLM-generated content.
3.  **To create a novel benchmark for evaluating dynamic guardrail adaptation:** A key contribution will be the development of a benchmark, "DynoSafeBench," specifically designed to measure a guardrail's ability to adapt to policy changes at inference time without retraining.
4.  **To conduct a comprehensive evaluation of the DPE framework:** We will rigorously evaluate our proposed system against relevant baselines, analyzing the trade-offs between safety enforcement accuracy, adaptability, utility preservation, and computational latency.

#### **2.3. Significance**

This research is significant for both the academic and industrial communities dedicated to building trustworthy AI. By enabling real-time policy adaptation, our work provides a practical path towards creating LLM systems that can remain compliant with rapidly changing regulatory and ethical standards. This agility reduces the significant operational costs and risks associated with model updates. Academically, this project introduces a new paradigm for LLM safety control and provides a standardized benchmark, DynoSafeBench, to spur further research into adaptive guardrails. Ultimately, by making safety mechanisms more responsive and transparent, this research will contribute to building greater human trust in LLM-driven applications, a central goal of this workshop.

### **3. Methodology**

Our research is structured into four distinct phases: (1) Framework Architecture Design, (2) Dataset Curation and Enforcer Model Training, (3) DynoSafeBench Benchmark Development, and (4) Experimental Validation and Analysis.

#### **3.1. Phase 1: Framework Architecture Design**

We propose a cascaded, two-LLM architecture. Let $L_P$ denote the primary, large-capability LLM (e.g., Llama-3-70B, GPT-4) responsible for fulfilling the user's request. Let $L_E$ denote the Dynamic Policy Enforcer, a smaller, highly efficient LLM (e.g., Llama-3-8B, Mistral-7B).

The workflow is as follows:
1.  A user submits a prompt, $u$, to the application.
2.  The primary LLM, $L_P$, generates an initial response, $o = L_P(u)$.
3.  The response $o$ is passed to the Dynamic Policy Enforcer, $L_E$, along with a dynamically supplied natural language policy, $\mathcal{P}$. The policy $\mathcal{P}$ is a text document outlining the rules the output must adhere to (e.g., "Do not provide financial advice," "Avoid discussing political candidates by name," "Ensure all medical information is sourced from the WHO website").
4.  The enforcer model $L_E$ processes these inputs and produces a structured decision, $d$. The decision object will contain a verdict and a justification:
    *   **Verdict:** A categorical label, e.g., `ALLOW`, `BLOCK`, or `REWRITE_SUGGESTION`.
    *   **Justification:** A natural language explanation citing the specific policy rule that was violated or upheld.

The final system output is then determined by this decision. An `ALLOW` verdict releases the response $o$ to the user. A `BLOCK` verdict replaces it with a canned refusal. A `REWRITE_SUGGESTION` could potentially be fed back to $L_P$ for a corrected response in more advanced implementations. For this research, we will focus on the `ALLOW`/`BLOCK` classification task.

#### **3.2. Phase 2: Dataset Curation and Enforcer Model Training**

The success of the DPE hinges on the quality of its training data. We will create a comprehensive, synthetic dataset designed to teach the enforcer model to reason about policy compliance.

**Data Synthesis:** We will use a state-of-the-art teacher model (e.g., GPT-4o or Claude 3 Opus) for data generation. The process involves a multi-step prompting strategy:
1.  **Policy Generation:** We will first generate a diverse set of hundreds of policies ($\mathcal{P}$). These policies will vary in complexity, domain (e.g., legal, medical, corporate), and specificity.
2.  **Example Generation:** For each policy $\mathcal{P}$, we will prompt the teacher model to generate pairs of (prompt $u$, response $o$) that are either compliant or non-compliant with $\mathcal{P}$. We will specifically prompt for edge cases and subtle violations.
3.  **Labeling and Justification:** For each generated example $(o, \mathcal{P})$, we will prompt the teacher model to produce a final label: a verdict (`ALLOW` or `BLOCK`) and a detailed natural language justification that quotes the relevant part of the policy and explains how the response $o$ complies or fails to comply.

This will result in a dataset of tuples: $(\text{response}, \text{policy}, \text{verdict}, \text{justification})$.

**Model Selection and Fine-tuning:**
We will select a small-to-medium sized open-source LLM for $L_E$, such as Mistral-7B or Llama-3-8B, to balance performance and inference latency. We will use Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA (Low-Rank Adaptation), to train the enforcer. The model will be fine-tuned on a supervised learning objective.

The input to the model $L_E$ will be a structured prompt template:
```
### Policy:
{policy_text}

### LLM Response:
{response_text}

### Verdict and Justification:
```
The model is trained to generate the structured output (e.g., a JSON object) that follows the template. The training objective is to minimize the standard cross-entropy loss for the next-token prediction task. Formally, given a dataset $\mathcal{D} = \{(o_i, \mathcal{P}_i, d_i)\}_{i=1}^N$ where $d_i$ is the target decision (verdict and justification), the fine-tuning objective for the enforcer model with parameters $\theta_E$ is to minimize:
$$
\mathcal{L}(\theta_E) = - \sum_{i=1}^{N} \sum_{j=1}^{|d_i|} \log P(d_{i,j} | d_{i, <j}, o_i, \mathcal{P}_i; \theta_E)
$$
where $d_{i,j}$ is the $j$-th token of the target decision string for the $i$-th example.

#### **3.3. Phase 3: DynoSafeBench Benchmark Development**

A key contribution of this work is a benchmark to rigorously measure *dynamic adaptability*. Static safety benchmarks are insufficient because they do not test a system's ability to change its behavior based on new, unseen rules at inference time. DynoSafeBench will be structured to specifically test this capability.

**Benchmark Components:**
1.  **Base Policies ($\mathcal{P}_{base}$):** A set of 10-15 fundamental safety policies covering common areas like hate speech, self-harm, and misinformation.
2.  **Policy Updates ($\Delta\mathcal{P}$):** For each base policy, we will create a set of "update pairs" $(\mathcal{P}_{v1}, \mathcal{P}_{v2})$. $\mathcal{P}_{v2}$ will be a modification of $\mathcal{P}_{v1}$, representing a realistic policy change. Examples include:
    *   **Broadening:** "Do not discuss politics" $\rightarrow$ "Do not discuss politics, religion, or social justice issues."
    *   **Narrowing/Exception:** "No medical advice" $\rightarrow$ "No medical advice, but it is acceptable to quote directly from the CDC website."
    *   **Complete Reversal:** "Promotion of cryptocurrency is allowed" $\rightarrow$ "All discussion of cryptocurrency is forbidden."
3.  **Test Set ($T$):** A curated set of prompts and corresponding reference LLM outputs designed to trigger violations under specific policy versions. For each update pair $(\mathcal{P}_{v1}, \mathcal{P}_{v2})$, the test set will contain examples that are:
    *   Unsafe under both $\mathcal{P}_{v1}$ and $\mathcal{P}_{v2}$.
    *   Safe under both $\mathcal{P}_{v1}$ and $\mathcal{P}_{v2}$.
    *   **Crucially, unsafe under $\mathcal{P}_{v1}$ but safe under $\mathcal{P}_{v2}$ (and vice versa).** These "differential" examples are key to measuring adaptation.

#### **3.4. Phase 4: Experimental Validation and Analysis**

We will conduct a thorough experimental evaluation of the DPE framework using DynoSafeBench.

**Baselines for Comparison:**
1.  **Zero-Shot Primary LLM ($L_P$ + System Prompt):** The primary LLM itself (e.g., Llama-3-70B) given the policy text in its system prompt. This tests if a large model can self-police effectively with dynamic instructions.
2.  **Static Hard-coded Guardrails:** A system using keyword lists and regex rules derived from the policies. This represents the traditional, brittle approach.
3.  **Programmable Guardrails:** An implementation using a toolkit like NeMo Guardrails where policies are translated into its specific Colang syntax. This requires manual effort for each policy update.
4.  **LLM-as-Judge (Static):** A general-purpose LLM (like GPT-4) prompted to judge compliance without specific fine-tuning on our policy reasoning task. This baseline isolates the effect of our specialized training.

**Evaluation Metrics:**
*   **Safety Accuracy:**
    *   **True Positive Rate (TPR):** Correctly identifying and blocking unsafe content. TPR = $\frac{TP}{TP+FN}$.
    *   **False Positive Rate (FPR):** Incorrectly blocking safe content. FPR = $\frac{FP}{FP+TN}$. A lower FPR indicates better utility preservation.
    *   **F1-Score:** The harmonic mean of precision and recall for the `BLOCK` class.
*   **Dynamic Adaptability Score (DAS):** This novel metric will measure how effectively the system adapts to a policy change from $\mathcal{P}_{v1}$ to $\mathcal{P}_{v2}$. For the subset of "differential" test cases, DAS is the accuracy of the system when evaluated with the new policy $\mathcal{P}_{v2}$ *immediately after being evaluated with* $\mathcal{P}_{v1}$, with no retraining.
    $$
    \text{DAS} = \text{Accuracy}(\text{Model} | \text{Differential Test Set}, \mathcal{P}_{v2})
    $$
*   **Performance:**
    *   **Latency:** The average additional time (in milliseconds) introduced by the enforcer model per request.
*   **Utility:**
    *   We will measure the quality of the `ALLOW`'ed responses. We can use an LLM-as-a-judge approach (e.g., using GPT-4 to rate helpfulness on a scale of 1-10) on a standard instruction-following benchmark (e.g., a subset of MT-Bench) with and without our DPE to quantify any degradation in response quality.

The evaluation will systematically test our DPE and all baselines on DynoSafeBench, measuring their performance across these metrics. We will pay special attention to the a) safety accuracy under base policies and b) Dynamic Adaptability Score on policy updates, which we hypothesize will be the key advantage of our proposed system.

### **4. Expected Outcomes & Impact**

This research is poised to deliver several key outcomes with significant academic and practical impact.

#### **4.1. Expected Outcomes**

1.  **A Novel Dynamic Policy Enforcer (DPE) Framework:** The primary outcome will be a fully realized and tested framework for adaptive LLM guardrails. This includes the architectural design, implementation details, and best practices for integrating an enforcer LLM into a larger system.
2.  **An Open-Source Fine-Tuned Enforcer Model:** We will release our fine-tuned enforcer model to the research community. This will provide a tangible, ready-to-use tool for developers and a strong baseline for future work in this domain.
3.  **The DynoSafeBench Public Benchmark:** We will publicly release the DynoSafeBench dataset, including all policies, policy updates, and test cases. This will be the first benchmark specifically designed to measure the dynamic adaptability of LLM safety systems, providing a much-needed tool for standardized evaluation and driving progress in the field.
4.  **A Comprehensive Analysis of Safety-Utility Trade-offs:** Our experimental results will yield a detailed analysis of the performance of the DPE framework. This will include quantitative data on its accuracy, adaptability, and latency, as well as its impact on the utility of the primary LLM. This analysis will offer valuable insights for practitioners weighing the costs and benefits of deploying such a system.

#### **4.2. Impact**

The impact of this research aligns directly with the core themes of the Workshop on Building Trust in Language Models and Applications.

*   **For Academia:** This work will introduce and formalize a new and promising direction in LLM safety research, shifting the focus from static defense to dynamic, policy-driven adaptation. The DynoSafeBench benchmark will establish a common ground for comparing different adaptive safety mechanisms, fostering more rigorous and reproducible science. It challenges the community to move beyond evaluating safety on fixed datasets and towards evaluating resilience in dynamic environments.

*   **For Industry and Practitioners:** The DPE framework offers a direct and practical solution to a major operational bottleneck. Organizations deploying LLMs can use this approach to drastically reduce the time and cost required to align their applications with new regulations, internal content standards, or platform-specific rules. This agility is crucial for maintaining compliance and managing risk in real-world deployments. By enabling faster responses to emerging safety threats, our work will help make AI systems safer and more reliable for end-users.

*   **For Trust and Ethics:** Ultimately, the greatest impact lies in enhancing the trustworthiness of LLMs. A system that can visibly and rapidly adapt to new safety rules is inherently more transparent and accountable. It provides a clear mechanism for human oversight, allowing administrators to steer model behavior using intuitive, natural language policies. This fosters greater confidence among users, regulators, and the public, paving the way for the responsible and ethical adoption of powerful language technologies.