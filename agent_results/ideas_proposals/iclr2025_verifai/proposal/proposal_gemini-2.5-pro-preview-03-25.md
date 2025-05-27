Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** LLM-TAC: Automated Tactic Generation for Interactive Theorem Provers via Fine-tuning and Reinforcement Learning

**2. Introduction**

**2.1 Background**
Formal verification offers rigorous guarantees of correctness critical for high-assurance software, complex hardware designs, and formalized mathematics. Interactive Theorem Provers (ITPs) like Coq [The Coq Development Team, 2023] and Lean [de Moura et al., 2015] are powerful tools in this domain, enabling users to construct machine-checkable proofs. However, the practical application of ITPs is often hindered by the significant manual effort required to guide the proof process. Users must meticulously select and apply sequences of "tactics"—small programs that manipulate the proof state—to discharge proof obligations. This process demands deep expertise in both the domain being formalized and the intricacies of the specific ITP and its tactic language, creating a steep learning curve and acting as a major bottleneck in large-scale verification projects [Harrison et al., 2014].

Recent advancements in large language models (LLMs) have demonstrated remarkable capabilities in code generation, natural language understanding, and complex reasoning tasks [Brown et al., 2020; OpenAI, 2023]. This success has spurred interest in applying LLMs to automate aspects of formal methods, including theorem proving. Several promising directions have emerged, focusing on tasks like premise selection, proof step suggestion, and even automated proof search [Yang et al., 2023; Welleck & Saha, 2023; Thakur et al., 2023; Song et al., 2024]. These works highlight the potential for LLMs to act as "copilots" [Song et al., 2024], assisting human provers and potentially reducing the manual burden.

However, existing approaches often focus on suggesting single proof steps or rely heavily on in-context learning with large, general-purpose models (like GPT-4 [Thakur et al., 2023]), which can be computationally expensive and may not fully leverage the structured nature of proof development. Furthermore, generating effective, multi-step tactic *sequences* that strategically advance the proof remains a significant challenge. Key difficulties identified in the literature include accurately encoding the rich proof context, ensuring the syntactic and semantic validity of generated tactics, integrating tightly with ITP environments for verification, obtaining sufficient high-quality training data, and achieving generalization across diverse mathematical domains [Yang et al., 2023; Welleck & Saha, 2023].

This research proposal directly addresses the need for automated tactic generation within the VerifAI workshop's theme of "Generative AI for formal methods." By automating tactic discovery, we aim to enhance verification practices and make formal methods more accessible. Our approach integrates probabilistic generation (LLMs) with deterministic verification (ITP execution), aligning with the workshop's goal of bridging AI and formal analysis.

**2.2 Problem Statement**
The core problem addressed by this research is the **manual bottleneck in interactive theorem proving caused by the laborious process of tactic selection and application.** This bottleneck limits the scalability of formal verification efforts and restricts its adoption to a smaller community of experts. Automating the generation of effective tactic sequences could dramatically accelerate proof development for both novices and experienced users.

**2.3 Proposed Solution: LLM-TAC Framework**
We propose **LLM-TAC**, a novel framework designed to automate the generation and refinement of tactic sequences for ITPs like Coq and Lean. LLM-TAC employs a two-stage process combined with a reinforcement learning loop:

1.  **Context-Aware Tactic Sequence Generation:** Given a proof obligation (a goal state and local hypotheses), LLM-TAC uses a fine-tuned LLM, augmented with retrieval mechanisms over relevant library definitions and previously proven lemmas, to propose candidate tactic sequences. The model is specifically trained to generate syntactically valid and potentially effective sequences of tactics.
2.  **ITP-Verified Execution and Feedback:** The proposed tactic sequences are systematically executed within the target ITP (Coq or Lean). The ITP serves as a perfect verifier:
    *   **Success:** If a sequence successfully transforms the proof state or closes a subgoal, the state-tactic pair (or sequence) is logged as positive feedback. The new proof state becomes the input for the next generation step.
    *   **Failure:** If a sequence fails (e.g., syntax error, tactic application error, doesn't make progress), the failure information (e.g., error message, invalid state) is captured as negative feedback.
3.  **Reinforcement Learning from Proof Feedback:** We utilize reinforcement learning (RL), specifically techniques like Proximal Policy Optimization (PPO) [Schulman et al., 2017], to continuously refine the LLM's tactic generation policy. Rewards are based on proof progress (e.g., closing subgoals, reducing goal complexity, successfully applying tactics), guiding the LLM towards generating more effective and efficient tactic sequences over time.

**2.4 Research Objectives**
The primary objectives of this research are:

1.  **Develop a Robust Context Encoder:** Design and implement a method to effectively encode the ITP proof state (goal, hypotheses, available definitions/lemmas) into a format suitable for LLM input, incorporating retrieval augmentation for relevant context from project libraries.
2.  **Fine-tune LLMs for Tactic Generation:** Fine-tune suitable pre-trained LLMs (e.g., Code Llama [Rozière et al., 2023]) on existing corpora of Coq and Lean proofs to generate syntactically plausible tactic sequences.
3.  **Implement ITP Integration and Verification Loop:** Create robust interfaces between the LLM generation module and the Coq/Lean ITPs to execute generated tactics and parse feedback (success, failure, new proof state).
4.  **Develop and Apply Reinforcement Learning:** Formulate the tactic generation task as an RL problem and apply algorithms like PPO to optimize the LLM policy using feedback signals derived from ITP execution (proof progress, tactic validity).
5.  **Evaluate LLM-TAC Performance:** Empirically evaluate the effectiveness of LLM-TAC on standard ITP benchmarks (e.g., subsets of Coq's standard library, mathcomp; Lean's mathlib [The mathlib Community, 2020]), measuring its ability to automate proof steps and reduce manual effort compared to baselines. Specifically, we aim to achieve a significant reduction (targeting 50%) in the number of tactic applications requiring manual intervention on benchmark proof goals.
6.  **Release Artifacts:** Package the trained models, fine-tuning scripts, ITP integration code, and evaluation benchmarks for public release to facilitate reproducibility and further research.

**2.5 Significance**
This research lies at the critical intersection of generative AI and formal methods. By successfully developing LLM-TAC, we expect to:

*   **Accelerate Formal Verification:** Substantially reduce the time and effort required for developing formal proofs, potentially enabling the verification of larger and more complex systems.
*   **Lower the Barrier to Entry:** Make ITPs more accessible to non-experts by automating challenging aspects of proof construction.
*   **Advance AI for Code/Formal Languages:** Contribute novel techniques for applying LLMs to highly structured, formal languages, addressing challenges in context understanding, generation accuracy, and integration with verification tools. This aligns with the VerifAI special theme on LLMs for Code Generation, extending it to the "code" of formal proofs (tactics).
*   **Provide a Hybrid AI-FM Framework:** Offer a concrete example of how probabilistic AI methods can be synergistically combined with deterministic formal tools, leveraging the scalability of the former and the soundness guarantees of the latter.
*   **Produce Valuable Resources:** Deliver open-source tools and models that benefit the formal methods and AI research communities.

**3. Methodology**

**3.1 Overall Architecture**
The LLM-TAC framework consists of three main components integrated into a feedback loop:

1.  **Context Encoder:** Takes the current proof state (goal, hypotheses) from the ITP and retrieves relevant definitions/lemmas. Encodes this information into a prompt for the LLM.
2.  **Tactic Generator (LLM):** A fine-tuned LLM that takes the encoded context and generates candidate tactic sequences.
3.  **Verifier & Feedback Module:** Interfaces with the ITP (Coq/Lean) to execute the generated tactics, determines success/failure, and extracts feedback (new proof state, error messages, proof progress metrics) to update the RL policy and potentially generate new training data.

**3.2 Data Collection and Preparation**
*   **Initial Supervised Fine-Tuning (SFT) Data:** We will leverage existing large-scale proof corpora, such as:
    *   Coq: Standard Library, CompCert [Leroy, 2009], MathComp [The Mathematical Components Team, 2021].
    *   Lean: Mathlib [The mathlib Community, 2020].
    We will parse these proof scripts to extract pairs of (Proof Context, Applied Tactic Sequence). The Proof Context will include the goal state and local hypotheses just before the tactic sequence was applied. We will need robust parsers for Coq's Vernacular and Ltac/Ltac2, and Lean's tactic language. Tools like `serapi` for Coq and potentially LeanDojo's infrastructure [Yang et al., 2023] for Lean can be adapted.
*   **Dynamic Data Generation (for RL):** During the RL phase, successful tactic sequences generated by the LLM and verified by the ITP will be added to a replay buffer or used directly for policy updates. Failed attempts and the resulting error messages can also be used, potentially for learning to avoid common pitfalls or as negative examples.

**3.3 Contextual Encoding**
The input to the LLM must represent the current proof obligation accurately.

*   **Proof State Representation:** The goal and local hypotheses will be serialized into a textual format. For Coq, this could be the standard output format; for Lean, the widget state format. Type information will be preserved.
*   **Retrieval Augmentation:** To provide necessary background context (definitions, lemmas, theorems) without overwhelming the LLM's context window, we will employ retrieval mechanisms.
    *   **Candidate Corpus:** Relevant definitions and theorem statements from the project's dependencies (e.g., `mathcomp` libraries for a `mathcomp` proof, `mathlib` for a `mathlib` proof).
    *   **Retrieval Method:** We will experiment with embedding-based retrieval techniques such as Dense Passage Retrieval (DPR) [Karpukhin et al., 2020]. The proof goal and hypotheses will form the query, embedded using a model like Sentence-BERT [Reimers & Gurevych, 2019]. The library items (definitions, theorem statements) will be pre-embedded and indexed (e.g., using FAISS [Johnson et al., 2019]). The top-k retrieved items will be included in the LLM prompt.
*   **Prompt Formatting:** The final prompt will be structured clearly, demarcating the goal, local hypotheses, and retrieved library items. E.g., `[GOAL] <goal_text> [HYPOTHESES] <hyp1_text> ... <hypN_text> [CONTEXT] <retrieved_item1> ... <retrieved_itemK> [TACTIC]`.

**3.4 Tactic Generation Model**
*   **Base Model:** We will start with pre-trained LLMs adept at code generation, such as Code Llama [Rozière et al., 2023] or potentially domain-specific models if available. We will likely experiment with models of varying sizes (e.g., 7B, 13B parameters).
*   **Supervised Fine-Tuning (SFT):** The base model will be fine-tuned on the data collected in Section 3.2. The objective is to maximize the likelihood of generating the ground-truth tactic sequence given the proof context:
    $$ \max_{\theta} \sum_{(C, T) \in \mathcal{D}_{\text{SFT}}} \log P_{\theta}(T | C) $$
    where $C$ is the encoded context, $T$ is the target tactic sequence, $\mathcal{D}_{\text{SFT}}$ is the supervised dataset, and $\theta$ represents the LLM parameters.
*   **Generation Strategy:** During inference, we will use techniques like beam search or nucleus sampling [Holtzman et al., 2019] to generate multiple candidate tactic sequences.

**3.5 Verification and Feedback Loop**
*   **ITP Interaction:** We will develop Python wrappers that communicate with Coq (using `serapi` or `coq-lsp`) and Lean (potentially using `lean-client-python` or interfaces similar to LeanDojo/LLMSTEP). These wrappers will:
    1.  Send the current proof state to the Context Encoder.
    2.  Receive generated tactic sequences from the Tactic Generator.
    3.  Execute each sequence within the ITP.
    4.  Parse the ITP's response to determine:
        *   Success: Tactic applied, new proof state achieved (potentially closing the goal/subgoal).
        *   Failure: Syntax error, type error, tactic failure message.
*   **Feedback Signal:** The outcome of the verification step provides the feedback signal R used for RL.

**3.6 Reinforcement Learning for Refinement**
*   **Problem Formulation:**
    *   **State (s):** The encoded proof context $C$.
    *   **Action (a):** A generated tactic sequence $T$.
    *   **Policy ($\pi_{\theta}$):** The LLM $P_{\theta}(T | C)$, parameterized by $\theta$.
    *   **Reward (R):** A numerical value reflecting the quality of the action. We will design a reward function that encourages proof progress. For example:
        *   +1.0 if the tactic sequence successfully applies.
        *   +5.0 if a subgoal is closed.
        *   +10.0 if the main goal is closed.
        *   -0.1 if the tactic sequence is syntactically valid but fails to apply.
        *   -0.5 if the tactic sequence is syntactically invalid (potentially harsher penalty).
        *   Intermediate rewards based on goal complexity reduction (e.g., number of hypotheses, size of the term).
*   **RL Algorithm:** We propose using Proximal Policy Optimization (PPO) [Schulman et al., 2017] due to its stability and performance in language model fine-tuning [Ouyang et al., 2022]. PPO optimizes a clipped surrogate objective function, balancing exploration and exploitation while ensuring policy updates do not deviate too drastically from the previous policy. The objective function is approximately:
    $$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right] $$
    where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio, $\hat{A}_t$ is the estimated advantage function, and $\epsilon$ is a hyperparameter defining the clipping range.
*   **Training Procedure:** The RL training involves iteratively:
    1.  Sampling proof states (either from a dataset or during interactive proving).
    2.  Generating tactic sequences using the current policy $\pi_{\theta}$.
    3.  Executing sequences in the ITP and collecting rewards $R$.
    4.  Estimating advantage values $\hat{A}_t$.
    5.  Updating the policy parameters $\theta$ using PPO.

**3.7 Experimental Design**
*   **Target ITPs:** Coq and Lean 4.
*   **Datasets/Benchmarks:**
    *   We will curate representative subsets of proofs from well-established libraries:
        *   Coq: `Stdlib`, `MathComp` (e.g., `algebra`, `ssreflect`).
        *   Lean 4: `Mathlib` (e.g., core algebra, analysis, topology sections).
    *   The benchmarks will consist of specific theorems or lemmas, starting from the initial goal state.
*   **Baselines:**
    1.  **Manual Proving:** (As ground truth/upper bound on quality, lower bound on automation). Data on human proof effort may be estimated from corpus statistics or small user studies.
    2.  **SFT-only LLM-TAC:** Our model without the RL fine-tuning step.
    3.  **Existing Automated Tools/Tactics:** Standard auto-tactics (`auto`, `eauto` in Coq; `aesop`, `simp` in Lean), possibly combined with simple search.
    4.  **Relevant Prior Work (if comparable models/setups are available):** E.g., results reported for ReProver [Yang et al., 2023] or COPRA [Thakur et al., 2023], adapted to our benchmarks if possible.
*   **Evaluation Metrics:**
    1.  **Proof Automation Rate (% Goals Closed):** Percentage of benchmark goals fully closed automatically by LLM-TAC within a fixed resource limit (e.g., time, number of tactic attempts).
    2.  **Step Reduction Rate:** For goals closed, compare the number of tactic *sequences* generated by LLM-TAC versus the number of *individual* tactics in the corresponding manual proof section. Aim for the target 50% reduction in manual tactic invocations.
    3.  **Tactic Sequence Quality:** Analyze the length, efficiency (e.g., runtime), and human-readability of generated proofs compared to manual ones.
    4.  **Success Rate per Attempt:** Percentage of generated tactic sequences that successfully apply (pass ITP check).
    5.  **Computational Cost:** Inference time, training time.
*   **Ablation Studies:**
    1.  **Impact of Retrieval:** Compare performance with and without retrieval augmentation.
    2.  **Impact of RL:** Compare SFT-only model vs SFT+RL model using the metrics above.
    3.  **Impact of Base Model:** Evaluate performance using different underlying LLMs.
    4.  **Impact of Reward Function:** Test variations in the reward shaping.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
*   **A Functional LLM-TAC System:** An implemented prototype capable of interacting with Coq and Lean, generating tactic sequences, verifying them, and learning from feedback.
*   **Fine-tuned LLM Models:** Trained models (SFT and SFT+RL versions) specialized for tactic generation in Coq and Lean, demonstrating improved performance over generic models.
*   **Benchmark Results:** Quantitative evaluation results on standard benchmarks demonstrating the effectiveness of LLM-TAC, specifically aiming to show:
    *   A significant automation rate on benchmark proof goals.
    *   Achievement of (or progress towards) the goal of a 50% reduction in manual tactic steps required compared to baseline manual proofs on the selected benchmarks.
    *   Improved performance compared to SFT-only baselines and potentially existing automated tactics/tools.
*   **Open Source Release:** Publicly released code for the LLM-TAC framework (encoding, generation, verification, RL loop), fine-tuning scripts, curated benchmark datasets, and the trained model weights. This aligns with the goals of reproducibility and community building.
*   **Research Publications:** Dissemination of findings through publications at relevant conferences and workshops (like VerifAI).

**4.2 Impact**
*   **Democratization of Formal Methods:** By reducing the expertise needed for tactic-level proof development, LLM-TAC can make ITPs more accessible to software engineers, mathematicians, and students, broadening the user base of formal verification.
*   **Increased Productivity:** Automating tedious proof steps allows experts to focus on higher-level proof strategy and creative problem-solving, significantly accelerating the development of large verified systems and mathematical libraries.
*   **Synergy between AI and Formal Methods:** This work serves as a strong demonstration of how probabilistic AI can be effectively integrated with rigorous formal methods. The ITP acts as a perfect verifier for the LLM's suggestions, creating a trustworthy AI-assisted workflow. This directly contributes to the central theme of the VerifAI workshop.
*   **Advancement in AI for Code/Formal Reasoning:** The techniques developed for context encoding, structured generation, and RL fine-tuning using formal verifier feedback can inform future research on applying AI to other structured domains like program synthesis, code optimization, and automated reasoning in general.
*   **Addressing Key Challenges:** This research directly tackles identified challenges in the field, such as improving tactic generation accuracy via RL and tight ITP integration, leveraging proof context effectively through retrieval, and generating valuable training data dynamically.

**4.3 Potential Challenges and Mitigation**
*   **Scalability:** Applying RL to large LLMs can be computationally intensive. We will explore parameter-efficient fine-tuning techniques (e.g., LoRA [Hu et al., 2021]) and efficient RL algorithms.
*   **Reward Sparsity:** Meaningful rewards (like closing a goal) might be infrequent. We will design denser rewards based on intermediate progress and explore curiosity-driven exploration methods if needed.
*   **Semantic vs. Syntactic Correctness:** The LLM might generate syntactically correct but semantically nonsensical tactics. The ITP verification loop directly addresses this, and the RL process should penalize such generations.
*   **Generalization:** Models trained on specific libraries might not generalize well to entirely new domains. We will aim for diverse training data and evaluate cross-domain performance.

In conclusion, LLM-TAC proposes a novel and promising approach to significantly alleviate the tactic generation bottleneck in interactive theorem proving. By combining state-of-the-art LLMs with the formal guarantees of ITPs through a reinforcement learning framework, this research has the potential to substantially enhance the productivity and accessibility of formal verification.

**5. References**

*   Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in neural information processing systems*, *33*, 1877-1901.
*   de Moura, L., & Kong, S. (2021). The Lean Theorem Prover (System Description). In *Automated Deduction–CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings, Part II* (pp. 651-661). Springer International Publishing. (Note: While Lean 4 is target, original Lean system paper often cited).
*   Harrison, J., Urban, J., & Wiedijk, F. (2014). History of interactive theorem proving. In *Computational Logic* (pp. 135-214). Handbook of the History of Logic, Vol 9. Elsevier.
*   Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). The curious case of neural text degeneration. *arXiv preprint arXiv:1904.09751*.
*   Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
*   Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, *7*(3), 535-547.
*   Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *arXiv preprint arXiv:2004.04906*.
*   Leroy, X. (2009). Formal verification of a realistic compiler. *Communications of the ACM*, *52*(7), 107-115.
*   OpenAI. (2023). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.
*   Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, *35*, 27730-27744.
*   Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *arXiv preprint arXiv:1908.10084*.
*   Rozière, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X. E., ... & Synnaeve, G. (2023). Code Llama: Open Foundation Models for Code. *arXiv preprint arXiv:2308.12950*.
*   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
*   Song, P., Yang, K., & Anandkumar, A. (2024). Towards Large Language Models as Copilots for Theorem Proving in Lean. *arXiv preprint arXiv:2404.12534*.
*   Thakur, A., Tsoukalas, G., Wen, Y., Xin, J., & Chaudhuri, S. (2023). An In-Context Learning Agent for Formal Theorem-Proving. *arXiv preprint arXiv:2310.04353*.
*   The Coq Development Team. (2023). The Coq Proof Assistant Reference Manual. Version 8.18.
*   The mathlib Community. (2020). The Lean mathematical library. In *Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs* (pp. 367-381).
*   The Mathematical Components Team. (2021). The Mathematical Components library. https://math-comp.github.io/math-comp/
*   Welleck, S., & Saha, R. (2023). LLMSTEP: LLM Proofstep Suggestions in Lean. *arXiv preprint arXiv:2310.18457*.
*   Yang, K., Swope, A. M., Gu, A., Chalamala, R., Song, P., Yu, S., ... & Anandkumar, A. (2023). LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. *arXiv preprint arXiv:2306.15626*.

---