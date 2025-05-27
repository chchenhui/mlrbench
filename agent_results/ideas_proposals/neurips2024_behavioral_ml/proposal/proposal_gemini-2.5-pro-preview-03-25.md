## Research Proposal

**1. Title:** Cognitive Architecture-Guided Training and Inference for Verifiably Human-Like Reasoning in Large Language Models

**2. Introduction**

*   **Background:** Large Language Models (LLMs) have demonstrated remarkable capabilities in processing and generating human language, achieving state-of-the-art performance on a wide array of tasks (Brown et al., 2020; OpenAI, 2023). However, their success often masks underlying limitations. LLMs frequently exhibit "black box" characteristics, producing outputs without transparent, step-by-step reasoning processes that align with human cognition (Bommasani et al., 2021). This opacity hinders trust, makes debugging difficult, and complicates efforts to align AI behavior with human values and expectations, particularly in high-stakes domains like healthcare, finance, and education. While techniques like Chain-of-Thought (CoT) prompting (Wei et al., 2022) elicit intermediate reasoning steps, these steps are often post-hoc rationalizations rather than reflections of a structured, human-like cognitive process, and their faithfulness to the model's internal computations is not guaranteed (Turpin et al., 2023).

    Concurrently, the behavioral sciences, particularly cognitive science, have developed detailed computational models of human cognition known as cognitive architectures (e.g., ACT-R (Anderson, 2007), Soar (Laird, 2012), CLARION (Sun, 2016)). These architectures provide formal, psychologically validated frameworks describing the mechanisms underlying human memory, attention, learning, problem-solving, and decision-making. They decompose complex cognitive tasks into sequences of elementary information processing steps, offering a principled way to model human reasoning.

    Integrating insights from cognitive architectures into LLMs presents a promising avenue for bridging the gap between AI performance and human-like reasoning. Recent work has begun exploring this intersection. For instance, Binz & Schulz (2023) demonstrated that LLMs fine-tuned on psychological data can approximate human behavior, while Sumers et al. (2023) proposed the CoALA framework for structuring language agents with cognitive components. Wu et al. (2024) introduced LLM-ACTR, integrating ACT-R concepts into LLMs for manufacturing decision-making. Other studies focus on aligning LLM outputs with cognitive models (Doe & Smith, 2023; Lee & Kim, 2024) or using cognitive preference alignment (Cai et al., 2025). These pioneering efforts highlight the potential but also underscore key challenges, including robust alignment, scalability, evaluation, generalization, and balancing performance with interpretability (Lit Review Points 1-5).

    Our research aims to build upon this foundation by proposing a novel framework that directly leverages the procedural, step-by-step knowledge embodied in cognitive architectures to guide *both* the training and inference phases of LLMs. We hypothesize that by explicitly constraining LLMs to follow reasoning pathways predicted by established cognitive models, we can produce models that not only perform well but also reason in ways that are demonstrably more aligned with human cognitive processes, thereby enhancing their transparency, trustworthiness, and interpretability.

*   **Research Objectives:** This research aims to develop and evaluate a framework for Cognitive Architecture-Guided Language Model Training and Inference (CAG-LMTI). The primary objectives are:
    1.  **Develop a methodology for generating "cognitive traces"**: Formalize the process of extracting step-by-step reasoning sequences (cognitive traces) from computational cognitive architectures (initially focusing on ACT-R) for specific reasoning tasks (e.g., syllogistic reasoning, simple physics problem-solving, planning).
    2.  **Implement a hybrid training objective for LLMs**: Design and implement a novel loss function that combines standard language modeling objectives (e.g., next-token prediction) with a cognitive alignment loss, explicitly penalizing deviations between the LLM's generated reasoning steps and the reference cognitive traces.
    3.  **Develop a cognitive architecture-constrained decoding mechanism**: Create an inference-time algorithm that guides the LLM's generation process, prioritizing token sequences that adhere to the sequential logic predicted by the corresponding cognitive architecture model for the given task.
    4.  **Empirically evaluate the framework**: Rigorously assess the effectiveness of CAG-LMTI on benchmark reasoning tasks by measuring:
        *   Task performance (accuracy, solution quality).
        *   Cognitive alignment (similarity between LLM reasoning steps and cognitive traces).
        *   Behavioral congruence (similarity between LLM output patterns/errors and human behavioral data).
        *   Interpretability and perceived naturalness (through user studies).
    5.  **Analyze trade-offs and generalization**: Investigate the relationship between cognitive alignment, task performance, and computational cost. Assess the framework's ability to generalize to variations of trained tasks and potentially unseen tasks within the same domain.

*   **Significance:** This research holds significant potential for advancing the field of Behavioral Machine Learning and AI alignment.
    1.  **Enhanced Transparency and Interpretability:** By grounding LLM reasoning in established cognitive models (Johnson & Williams, 2024; Chen & Brown, 2023; Green & White, 2024), the proposed framework aims to make their decision-making processes more transparent and understandable to humans, addressing a critical limitation of current systems.
    2.  **Improved Human-AI Alignment:** Aligning LLM reasoning pathways with human cognitive processes can lead to AI systems whose behavior is more predictable, relatable, and aligned with human expectations and mental models, fostering trust and facilitating collaboration.
    3.  **Principled Integration of Behavioral Science:** This work provides a concrete methodology for incorporating formal computational models from cognitive science into the core training and inference loops of powerful AI systems, moving beyond qualitative insights towards verifiable computational integration (Martinez & Wilson, 2023).
    4.  **Contributions to Cognitive Science:** The framework can serve as a tool for testing and refining cognitive theories at scale, by comparing the performance and behavior of LLMs guided by different cognitive models against human data.
    5.  **Practical Applications:** Success in this research could pave the way for more trustworthy and collaborative AI applications in areas requiring explainable reasoning, such as personalized education (tutoring systems that understand student reasoning patterns), healthcare (diagnostic assistants that can explain their rationale), and human-AI teaming (AI partners whose actions are interpretable). This directly aligns with the goals of the Workshop on Behavioral Machine Learning, particularly concerning Alignment, Evaluation, Computational Cognitive Science, and Interpretability.

**3. Methodology**

This research will employ a constructive and empirical methodology, involving the development of a novel training and inference framework and its rigorous evaluation on selected reasoning tasks.

*   **Conceptual Framework:**
    The core idea is to use a computational cognitive architecture (initially ACT-R due to its maturity and detailed process models) as a "reference model" for human-like reasoning on specific tasks. We will develop ACT-R models for chosen tasks, simulate their execution to generate step-by-step cognitive traces, and then use these traces to guide an LLM during training and inference.

*   **Data Collection and Cognitive Trace Generation:**
    1.  **Task Selection:** We will focus on tasks where human reasoning processes have been well-studied and can be reasonably modeled within cognitive architectures. Initial candidates include:
        *   *Syllogistic Reasoning:* Using datasets like SNLI (Bowman et al., 2015) adapted for explicit reasoning steps, or dedicated datasets like LogiQA (Liu et al., 2020).
        *   *Simple Physics Problem Solving:* Problems solvable with qualitative reasoning or basic formulas (e.g., datasets like PhysQA (Lin et al., 2023) subsetted for simplicity).
        *   *Rule-Based Planning/Scheduling:* Simple tasks requiring sequential decision-making based on explicit rules (e.g., simplified versions of Blocks World or logistics planning).
    2.  **Cognitive Model Development:** For each selected task domain, we will develop corresponding ACT-R models. These models will incorporate standard ACT-R modules (e.g., declarative memory, procedural memory, goal stack, buffers) and simulate the cognitive steps involved in solving the task (e.g., encoding the problem, retrieving relevant facts/rules, applying rules, updating goals).
    3.  **Trace Generation:** We will execute the ACT-R models on specific instances of the selected tasks. The output will be a structured *cognitive trace*, represented as a sequence of symbolic states or operations (e.g., `[GOAL: solve_syllogism] -> [RETRIEVE: premise_1] -> [ENCODE: relation(A, B)] -> [RETRIEVE: premise_2] -> [ENCODE: relation(B, C)] -> [APPLY_RULE: transitive_inference] -> [DECLARE: conclusion(A, C)]`). These traces will serve as the ground truth for human-like reasoning steps. We will need to map these symbolic traces to natural language representations suitable for LLM processing (e.g., "First, I read premise 1...", "Then, I applied the transitivity rule...").

*   **Cognitive Architecture-Guided Training:**
    1.  **Base LLM:** We will use a pre-trained transformer-based LLM (e.g., Llama-3, Mistral, or potentially smaller models for faster iteration like those discussed by Cai et al., 2025).
    2.  **Hybrid Training Objective:** We propose a hybrid loss function $L_{total}$ for fine-tuning the LLM on task-specific data paired with cognitive traces. The loss combines a standard language modeling loss ($L_{LM}$) with a cognitive alignment loss ($L_{CA}$).
        $$ L_{total} = L_{LM} + \lambda L_{CA} $$
        where $\lambda$ is a hyperparameter balancing the two objectives.
        *   $L_{LM}$: Standard cross-entropy loss for predicting the next token in the target solution and/or the natural language reasoning steps.
            $$ L_{LM} = - \sum_{i} \log P(y_i | y_{<i}, x) $$
            where $x$ is the input problem, and $y_i$ are the tokens of the target output sequence (reasoning trace + final answer).
        *   $L_{CA}$: This loss measures the discrepancy between the LLM's generated intermediate reasoning steps (prompted or embedded) and the target cognitive trace. Let $T = (t_1, t_2, ..., t_N)$ be the target cognitive trace (sequence of $N$ symbolic steps) and $R = (r_1, r_2, ..., r_M)$ be the reasoning steps generated or represented by the LLM. We need a mapping function $f$ to align $R$ to $T$. $L_{CA}$ could be formulated using:
            *   *Sequence Matching Loss:* Penalize mismatches (e.g., using sequence alignment algorithms like Smith-Waterman or edit distance) between the generated reasoning structure and the target trace.
            *   *Step-wise Consistency Loss:* If the LLM generates step-by-step reasoning, compare the semantic or structural similarity of each generated step $r_j$ with the corresponding target step $t_k$. For instance, using KL divergence if steps are represented probabilistically, or a cross-entropy loss if specific reasoning operators are predicted at each step.
            $$ L_{CA} = \sum_{k=1}^{N} D(f(R)_k || t_k) $$
            where $D$ is a distance or divergence measure, and $f(R)_k$ is the generated reasoning representation aligned with the $k$-th cognitive step $t_k$. The exact formulation will depend on how reasoning steps are elicited/represented in the LLM.

*   **Cognitive Architecture-Constrained Decoding:**
    During inference, standard decoding methods like beam search generate sequences based solely on the LLM's learned probability distribution. We propose modifying this process to incorporate guidance from the cognitive architecture.
    1.  **State Tracking:** Maintain an estimate of the current state within the cognitive model's predicted reasoning pathway based on the sequence generated so far.
    2.  **Step Prediction:** At each generation step, use the corresponding cognitive model (or a pre-compiled policy derived from it) to predict the *next likely* cognitive operation(s) or state transition(s).
    3.  **Probability Modulation:** Modify the LLM's output probability distribution over the next token(s). Increase the probability of tokens/phrases that correspond to the predicted cognitive operation(s) and decrease the probability of those that deviate significantly. This can be formulated as:
        $$ P_{constrained}(y_i | y_{<i}, x, s_k) \propto P_{LLM}(y_i | y_{<i}, x) \times \exp(\beta \cdot \text{Score}(y_i, \text{Predict}(s_k))) $$
        where $s_k$ is the current estimated cognitive state, $\text{Predict}(s_k)$ gives the set of likely next cognitive steps, $\text{Score}(y_i, \cdot)$ measures the consistency of generating token $y_i$ with the predicted steps, and $\beta$ controls the strength of the constraint.
    4.  **Beam Search Integration:** Integrate this modulation into the scoring function used in beam search, guiding the search towards beams that align with the cognitive trace.

*   **Experimental Design:**
    1.  **Datasets:** Use the selected task datasets (Syllogisms, Physics, Planning) with generated cognitive traces. Ensure sufficient training, validation, and test splits.
    2.  **Baselines:**
        *   *Base LLM (zero-shot):* The pre-trained LLM without fine-tuning.
        *   *Base LLM (standard fine-tuning):* The LLM fine-tuned only on task input-output pairs ($L_{total} = L_{LM}$).
        *   *Base LLM (CoT fine-tuning):* Fine-tuned on input -> reasoning steps -> output data, without explicit cognitive alignment ($L_{total} = L_{LM}$ on augmented data).
        *   *(Optional)* Other relevant cognitively-inspired models if implementations are available (e.g., based on CoALA or simplified versions of LLM-ACTR concepts).
    3.  **Experimental Conditions:**
        *   *CAG-LMTI (Training Only):* LLM trained with the hybrid objective ($L_{total} = L_{LM} + \lambda L_{CA}$) but using standard decoding.
        *   *CAG-LMTI (Inference Only):* LLM trained with standard fine-tuning ($L_{total} = L_{LM}$) but using cognitive architecture-constrained decoding.
        *   *CAG-LMTI (Full):* LLM trained with the hybrid objective and using constrained decoding.
    4.  **Ablation Studies:** Systematically vary $\lambda$ and $\beta$. Evaluate the impact of different components (hybrid loss vs. constrained decoding). Test the framework using different base LLM sizes. Compare guidance from different cognitive architectures (if time permits, e.g., simple rule-based vs. ACT-R).
    5.  **Evaluation Metrics:**
        *   *Task Performance:* Accuracy (Syllogisms, Physics), Plan validity/optimality (Planning), BLEU/ROUGE scores (if output is complex text).
        *   *Cognitive Alignment:*
            *   *Trace Similarity:* Measure the normalized edit distance (e.g., Levenshtein) or semantic similarity (e.g., using embeddings like Sentence-BERT) between the LLM's generated reasoning steps (explicitly prompted or inferred) and the ground truth cognitive traces.
            *   *Step Prediction Accuracy:* If applicable, measure the accuracy of the LLM predicting the specific cognitive operator at each step.
        *   *Behavioral Congruence:* Collect human performance data (accuracy, common errors, reaction times if feasible) on the same task instances. Compare the distribution of LLM outputs and errors with human patterns (e.g., correlation of item difficulty, similarity of error types).
        *   *Interpretability & Naturalness:* Conduct user studies. Present participants with problems solved by different models (Baseline vs. CAG-LMTI). Ask them to rate the explanations/reasoning steps on clarity, logical coherence, trustworthiness, and human-likeness (e.g., using Likert scales). Collect qualitative feedback.
        *   *Computational Cost:* Measure training time, inference latency, and memory requirements.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A Novel Framework (CAG-LMTI):** A functional software implementation of the proposed cognitive architecture-guided training and inference framework.
    2.  **Trained Models:** LLMs fine-tuned using the CAG-LMTI framework for selected reasoning tasks, demonstrating improved cognitive alignment compared to baselines.
    3.  **Empirical Results:** Quantitative results demonstrating the effectiveness of the framework in terms of task performance, cognitive alignment, behavioral congruence, and human-perceived interpretability. This includes data addressing the trade-offs between these aspects (Challenge 5).
    4.  **Analysis of Generalization:** Insights into the framework's ability to generalize across variations within a task domain and potentially to new, related domains (addressing Challenge 4).
    5.  **Benchmark Contributions:** Potentially, new benchmark datasets combining reasoning tasks with corresponding cognitive traces derived from formal models, facilitating further research in this area (addressing Challenge 3 regarding evaluation).
    6.  **Mitigation of Challenges:** The methodology directly addresses key challenges: explicit alignment mechanisms tackle Challenge 1; ablation studies and metric analysis address Challenge 5; robust evaluation metrics address Challenge 3; focusing initially on smaller models and specific tasks helps manage Challenge 2 (Scalability), whilst assessing generalization addresses Challenge 4.

*   **Impact:**
    This research aims to make significant contributions to the intersection of AI, machine learning, and cognitive science.
    1.  **Scientific Impact:** It will provide a more robust and verifiable method for integrating computational cognitive models into LLMs, moving beyond surface-level mimicry towards deeper process alignment. This contributes directly to the goals of the Workshop on Behavioral Machine Learning, especially in the areas of Alignment, Evaluation using human models, and Computational Cognitive Science. Success would offer strong evidence for the utility of cognitive architectures in shaping AI reasoning. It could also provide a novel computational testbed for cognitive theories themselves.
    2.  **Technological Impact:** The framework promises to yield LLMs that are more interpretable, trustworthy, and predictable. This is crucial for deploying AI systems in collaborative or safety-critical applications where understanding the AI's reasoning is paramount (e.g., education, healthcare, debugging complex systems). By producing human-like reasoning, these models could facilitate more natural and effective human-AI interaction.
    3.  **Broader Impact:** By fostering AI systems that reason in ways more aligned with human cognition, this work could help bridge the gap between AI capabilities and societal acceptance, promoting responsible AI development. It provides a concrete pathway towards building AI systems that are not just powerful pattern recognizers but also more transparent reasoning partners.

In conclusion, the proposed research offers a principled approach to imbuing LLMs with more human-like, interpretable reasoning by leveraging the rich theoretical grounding of computational cognitive architectures. Through the development and rigorous evaluation of the CAG-LMTI framework, we expect to significantly advance the state-of-the-art in building more transparent, trustworthy, and behaviorally aligned AI systems.