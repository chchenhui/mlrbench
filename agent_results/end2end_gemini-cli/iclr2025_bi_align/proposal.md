### **1. Title: Proactive Alignment Auditing via Interactive Counterfactual Probing**

### **2. Introduction**

**2.1 Background**

The rapid proliferation of large-scale, general-purpose AI systems, particularly Large Language Models (LLMs), has made the challenge of AI alignment more urgent than ever. Alignment, the process of ensuring AI systems act in accordance with human values and intentions, is critical for safe and beneficial deployment. The predominant paradigm for aligning LLMs has been Reinforcement Learning from Human Feedback (RLHF), a method that trains models based on human preferences for pre-generated responses. While effective to a degree, this approach is fundamentally *reactive*. It refines a model based on past feedback but fails to proactively uncover latent biases, misaligned reasoning patterns, or potential failure modes that may only manifest in novel, unforeseen contexts post-deployment. This reactive stance constitutes a significant vulnerability in our efforts to build robustly aligned AI.

This limitation highlights the inadequacy of viewing alignment as a static, unidirectional process of simply shaping AI to human specifications. As argued in recent scholarship (Shen, 2024), a more holistic and effective paradigm is **bidirectional human-AI alignment**. This framework conceives of alignment as a dynamic, mutual process involving two symbiotic directions:
1.  **Aligning AI with Humans (AI-centered):** The traditional focus on integrating human values, specifications, and feedback into the AI's training and behavior.
2.  **Aligning Humans with AI (Human-centered):** The often-overlooked necessity of empowering humans to build accurate mental models of the AI, understand its capabilities and limitations, and critically evaluate its outputs, thereby preserving human agency and enabling effective collaboration.

Achieving this bidirectional alignment requires new methods that move beyond static feedback collection. Interactive frameworks, as explored by Terry et al. (2023), are essential. These frameworks must allow users to not just react to AI outputs, but to actively probe, explore, and understand the AI's internal decision-making landscape. Counterfactual reasoning—exploring "what if" scenarios—has emerged as a powerful tool for this purpose. Research has shown its potential for generating difficult training examples (Ji et al., 2023), creating human-centric explanations (Domnich et al., 2024), and directly fine-tuning models (Butcher, 2024; Liu et al., 2023).

**2.2 Research Objectives**

This project introduces **Proactive Alignment Auditing via Interactive Counterfactual Probing (PAAC-Pro)**, a novel framework designed to operationalize the principles of bidirectional alignment. Instead of passively rating outputs, PAAC-Pro empowers users to actively audit an AI's alignment by systematically generating and testing counterfactual scenarios. This interactive process allows for the dynamic exploration of a model's sensitivities to changes in semantically meaningful attributes (e.g., gender, race, political affiliation), thereby revealing hidden biases and misaligned reasoning.

The core objectives of this research are:

1.  **To design and implement the PAAC-Pro framework**, an interactive system that facilitates the generation, execution, and visualization of counterfactual probes for LLMs.
2.  **To formalize and implement a targeted fine-tuning pipeline** that leverages the counterfactual data generated during an audit to efficiently rectify identified alignment failures.
3.  **To empirically evaluate the efficacy of PAAC-Pro in facilitating bidirectional alignment** through a mixed-methods user study. We will assess:
    *   **(Human-to-AI Alignment):** Whether the framework enhances users' understanding of the model's behavior, biases, and decision boundaries compared to a standard interface.
    *   **(AI-to-Human Alignment):** Whether fine-tuning on the collected counterfactual data leads to more robustly aligned models than those trained with conventional methods.

**2.3 Significance**

This research makes several significant contributions. Scientifically, it proposes a concrete, testable methodology that transitions AI alignment from a reactive to a proactive endeavor. It provides a practical implementation of the bidirectional alignment theory, bridging the gap between HCI and ML by creating a human-in-the-loop system that serves both human understanding and model improvement. Societally, PAAC-Pro represents a powerful tool for AI auditors, developers, and even end-users to enhance AI safety and fairness. By enabling the discovery and mitigation of hidden biases before they cause widespread harm, this work contributes to the development of more trustworthy, equitable, and transparent AI systems.

### **3. Methodology**

Our methodology is structured around three core components: (1) the design of the PAAC-Pro interactive framework, (2) the development of a counterfactual-driven model alignment procedure, and (3) a rigorous experimental plan to validate our approach.

**3.1 The PAAC-Pro Framework**

The PAAC-Pro framework is an interactive system that operates in a two-stage loop, directly corresponding to the two directions of bidirectional alignment.

**Stage 1: Interactive Auditing & Human Understanding (Human-to-AI Alignment)**

This stage focuses on empowering the user to explore the model's behavior. The workflow is as follows (see Figure 1):

1.  **Initial Prompt:** A user inputs an initial prompt, $P$, into the interface (e.g., "Write a short biography of a successful CEO."). The base model, $M$, generates an initial response, $R = M(P)$.
2.  **Counterfactual Perturbation:** The user decides to probe the model's sensitivity to a specific attribute. Instead of requiring the user to manually rewrite the prompt, the PAAC-Pro interface assists them. It uses a smaller, specialized LLM to perform named entity recognition and semantic role labeling on $P$ to identify key concepts (e.g., "CEO," "successful," implied gender/nationality). These concepts are presented to the user as interactive "knobs." The user can select a knob (e.g., "Gender") and a new value (e.g., "Female"). This user-defined semantic perturbation is denoted by $\delta$.
3.  **Counterfactual Generation:** The system automatically generates a new, grammatically correct counterfactual prompt, $P'$, by applying the perturbation $\delta$ to $P$. For our example, $P'$ might become "Write a short biography of a successful female CEO." The model then generates the counterfactual response, $R' = M(P')$.
4.  **Semantic Difference Visualization:** Simply presenting $R$ and $R'$ side-by-side is insufficient for complex text. PAAC-Pro will provide an intuitive visualization of the semantic differences. We will employ a two-pronged approach:
    *   **Sentence-level Diffing:** We will use a sentence embedding model (e.g., Sentence-BERT) to compute vector representations for each sentence in $R$ and $R'$. We then compute the cosine distance between aligned sentence pairs. Sentences with a distance above a certain threshold are highlighted, indicating a significant semantic shift.
    *   **Global Summary of Differences:** We will use a powerful cross-encoder or another LLM prompted as an evaluator to generate a natural language summary of the key thematic differences between $R$ and $R'$. For instance, it might output: "The original response emphasized 'risk-taking' and 'market domination,' while the counterfactual response emphasized 'team-building' and 'sustainable growth'."
5.  **User Annotation:** Based on this visualization, the user gains a deeper understanding of the model's implicit associations. The user can then annotate the interaction, for example, by flagging it as "Reveals Unacceptable Gender Bias." This entire interaction tuple, $(P, P', R, R', A)$, where $A$ is the annotation, is logged for the next stage.

**Stage 2: Model Rectification & Alignment (AI-to-Human Alignment)**

This stage uses the data collected from the audit to improve the model's alignment.

1.  **Data Curation:** The logged interaction tuples form a high-quality dataset, $\mathcal{D}$, of "difficult" examples that pinpoint specific alignment failures.
2.  **Preference Pair Generation:** For each tuple $(P, P', R, R', A)$ where a bias was flagged ($A = \text{'bias'}$), we use this information to fine-tune the model. Drawing inspiration from Counterfactual DPO (Butcher, 2024), we create preference data. The "rejected" response, $R_{\text{rejected}}$, is the biased output (either $R$ or $R'$). The "preferred" response, $R_{\text{preferred}}$, needs to be a bias-free version. We generate $R_{\text{preferred}}$ by feeding the biased response back to a powerful LLM with an instruction to rewrite it while preserving factual content but removing the identified bias (e.g., "Rewrite this biography to be gender-neutral in its portrayal of leadership styles").
3.  **Model Fine-tuning:** We use the curated preference pairs $(P_i, R_{i, \text{preferred}}, R_{i, \text{rejected}})$ to fine-tune the base model $M$ using Direct Preference Optimization (DPO). The DPO loss function aims to increase the likelihood of preferred responses and decrease the likelihood of rejected ones, relative to a reference model $\pi_{\text{ref}}$ (which is typically the initial supervised fine-tuned model before DPO). The loss is given by:
    $$
    \mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = - \mathbb{E}_{(P, R_w, R_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(R_w | P)}{\pi_{\text{ref}}(R_w | P)} - \beta \log \frac{\pi_{\theta}(R_l | P)}{\pi_{\text{ref}}(R_l | P)} \right) \right]
    $$
    where $R_w$ is the preferred response, $R_l$ is the rejected response, $\pi_{\theta}$ is the policy of the model being trained, and $\beta$ is a temperature parameter. This targeted fine-tuning is more data-efficient than standard RLHF as it focuses directly on correcting observed failures.

**3.2 Experimental Design**

We will conduct a comprehensive evaluation involving both human participants and automated benchmarks.

**Models:** Our base model ($M_{\text{base}}$) will be a publicly available, state-of-the-art LLM (e.g., Llama-3 8B-Instruct).

**3.2.1 User Study: Evaluating the Human-to-AI Alignment Direction**

This study will assess whether PAAC-Pro improves users' ability to understand and audit the AI.

*   **Participants:** We will recruit 40-50 participants from diverse backgrounds, screened for basic literacy and experience with web interfaces.
*   **Design:** A between-subjects design with two conditions:
    *   **Control Group (N=20-25):** Participants interact with the base LLM via a standard chat interface.
    *   **Experimental Group (N=20-25):** Participants interact with the same base LLM using our PAAC-Pro framework.
*   **Tasks:** All participants will be given a set of predefined auditing goals, such as "Investigate the model's potential gender bias in professional contexts" and "Examine how the model portrays individuals from different socioeconomic backgrounds."
*   **Metrics & Data Collection:**
    *   **Quantitative Metrics:** Number of distinct biases/failure modes identified per participant; time to first discovery.
    *   **Qualitative Metrics:** We will use think-aloud protocols during the sessions to capture user thought processes.
    *   **Post-Task Surveys:** We will administer questionnaires using 7-point Likert scales to measure:
        *   **Perceived Understanding:** "I have a clear understanding of the AI's biases on this topic."
        *   **Sense of Agency:** "I felt in control of the process of auditing the AI."
        *   **Trust & Transparency:** "I feel the interface made the AI's behavior more transparent."
        *   The System Usability Scale (SUS) to evaluate the interface's usability.

**Hypothesis H1:** Participants in the experimental group (using PAAC-Pro) will identify significantly more alignment issues and report higher levels of perceived understanding, agency, and transparency compared to the control group.

**3.2.2 Model Evaluation: Evaluating the AI-to-Human Alignment Direction**

This study will assess the effectiveness of our fine-tuning method. We will compare three models:

1.  **$M_{\text{base}}$:** The original, pre-trained model.
2.  **$M_{\text{RLHF}}$:** The base model fine-tuned using a generic, publicly available preference dataset (e.g., Anthropic's HHH dataset) of comparable size to the data we collect.
3.  **$M_{\text{PAAC}}$:** The base model fine-tuned using the DPO method described above on the dataset $\mathcal{D}$ collected from the PAAC-Pro user study.

*   **Evaluation Metrics:**
    *   **Automated Benchmarks:** We will evaluate all three models on standard alignment benchmarks like BBQ (for bias), TruthfulQA (for truthfulness), and ToxiGen (for toxicity) to assess generalization.
    *   **Targeted Human Evaluation:** We will create a new set of challenging "red-teaming" prompts specifically designed to trigger the biases investigated during the user study. We will then conduct a blind, pairwise comparison study where expert human annotators (separate from the user study participants) are shown the responses from the three models and asked to choose the best one based on criteria of fairness, accuracy, and harmlessness. We will calculate the win-rate of $M_{\text{PAAC}}$ against $M_{\text{base}}$ and $M_{\text{RLHF}}$.

**Hypothesis H2:** $M_{\text{PAAC}}$ will show a significantly higher win-rate against $M_{\text{base}}$ on the targeted evaluation set compared to $M_{\text{RLHF}}$, demonstrating the data efficiency and effectiveness of our targeted fine-tuning approach.

### **4. Expected Outcomes & Impact**

This research is expected to produce a range of tangible outcomes and have a significant impact on the field of AI alignment.

**Expected Outcomes:**

1.  **A Functional Open-Source Prototype:** We will develop and release an open-source, web-based implementation of the PAAC-Pro framework, allowing other researchers and practitioners to use, extend, and validate our methods.
2.  **A Novel Counterfactual Alignment Dataset:** The data collected from our user study, consisting of interactive counterfactual probes, model responses, and human annotations, will be curated and released. This dataset, denoted $\mathcal{D}$, will be a valuable resource for research into interactive alignment, bias mitigation, and human-AI interaction.
3.  **Empirical Validation of Bidirectional Alignment:** Our experimental results will provide strong empirical evidence for the benefits of the PAAC-Pro framework. We expect to demonstrate quantitatively that (a) the interactive system improves human auditors' ability to find alignment failures, and (b) the data collected is highly effective for targeted model fine-tuning.
4.  **Design Guidelines for Interactive AI Auditing Tools:** Based on our findings from the user study, we will synthesize a set of actionable design principles for creating effective, human-centric tools that support proactive and collaborative AI alignment.

**Impact:**

*   **Scientific Impact:** This project will pioneer a shift in the AI alignment paradigm, moving from reactive feedback to proactive, exploratory auditing. By providing a concrete, reproducible methodology for bidirectional alignment, it will bridge the gap between theoretical frameworks (Shen, 2024; Terry et al., 2023) and practical application. It will advance the state of the art in human-in-the-loop machine learning, explainable AI, and interactive systems.
*   **Societal and Ethical Impact:** The widespread deployment of AI systems necessitates robust auditing and oversight mechanisms. PAAC-Pro offers a practical pathway for developers, regulators, and third-party auditors to rigorously vet AI models for hidden biases and potential harms before they impact society. By making AI behavior more transparent and controllable, our work aims to foster greater public trust in AI and promote the development of systems that are more fair, equitable, and aligned with diverse human values. This work contributes directly to responsible AI development and could inform future policies and standards for AI certification and accountability.
*   **Interdisciplinary Contribution:** This research is inherently interdisciplinary, weaving together techniques and perspectives from Machine Learning (DPO, fine-tuning), Human-Computer Interaction (interactive design, user studies), and the social sciences (bias, fairness, cognition). It will foster collaboration and provide a common ground for these fields to collectively address the critical challenge of human-AI alignment.