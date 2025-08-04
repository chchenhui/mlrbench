# **Neuro-Symbolic Repair: Correcting LLM-Generated Code via SMT-Informed Self-Correction**

### **1. Introduction**

#### **Background**
The proliferation of Large Language Models (LLMs) has catalyzed a paradigm shift in software engineering, with models like OpenAI's GPT-4, Google's Gemini, and Meta's Llama demonstrating remarkable capabilities in code generation. These models can translate natural language descriptions into functional code across numerous programming languages, promising to accelerate development, lower the barrier to entry for novice programmers, and automate tedious coding tasks. However, the probabilistic foundation of these models is both their greatest strength and their most significant weakness. While capable of generating syntactically valid and often plausible code, LLMs lack an inherent understanding of formal correctness. Consequently, their outputs can contain subtle logical flaws, fail on critical edge cases, or introduce security vulnerabilities that are difficult to detect through standard testing procedures (Chen et al., 2023).

Concurrently, the field of formal methods has long provided a rigorous foundation for software and hardware verification. Techniques such as theorem proving, model checking, and symbolic execution offer mathematical guarantees of correctness against a given formal specification. Among these, Satisfiability Modulo Theories (SMT) solvers have emerged as powerful and versatile tools for automated reasoning. By checking the satisfiability of logical formulas over various background theories (e.g., integers, arrays, bit-vectors), SMT solvers can effectively find concrete counterexamples that demonstrate how a piece of code violates a specified property. However, formal methods face their own challenges, including the difficulty of writing complete specifications, the computational expense of verification, and their limited ability to perform creative, large-scale code synthesis or repair (Wang et al., 2024).

This research proposal situated at the confluence of these two domains, directly addressing the central themes of the VerifAI workshop. Recent work has explored self-correction mechanisms where LLMs refine their own code based on feedback from execution results, such as failing unit tests (Zhang et al., 2023; Cho et al., 2025). While effective, this approach is limited by the quality and coverage of the test suite. A unit test can confirm that code fails for a specific input but often does not reveal the underlying logical reason for the failure or expose systemic vulnerabilities across a class of inputs. Our work seeks to bridge this gap by creating a more potent and precise feedback loop. Instead of relying solely on test-case failures, we propose using an SMT solver to find deep, logical bugs and then translate the resulting formal counterexample into targeted, natural language advice to guide the LLM's own powerful self-correction capabilities. This neuro-symbolic approach combines the bug-finding precision of formal solvers with the contextual reasoning and fluent code-writing abilities of LLMs, creating a synergistic system that is more robust than either paradigm alone.

#### **Research Objectives**
This project aims to design, implement, and evaluate a novel framework, **SMT-Repair**, which leverages SMT-informed feedback to enhance the correctness of LLM-generated code. The primary objectives are:

1.  **To develop a novel neuro-symbolic framework for iterative code repair.** This involves creating a closed-loop system that seamlessly integrates an LLM for code generation, an SMT-based verification engine for bug detection, and a novel translation module that bridges the symbolic and neural components.

2.  **To design and implement a "Counterexample-to-Prompt" (C2P) translation module.** This core component will be responsible for converting the abstract, formal counterexamples produced by an SMT solver (e.g., variable assignments that lead to a property violation) into concise, informative, and human-readable natural language prompts.

3.  **To empirically validate the effectiveness and efficiency of the SMT-Repair framework.** We will conduct a rigorous experimental evaluation to compare the correctness of code produced by our framework against baselines, including zero-shot LLM generation and self-correction methods guided by traditional unit-test feedback.

4.  **To analyze the dynamics of neuro-symbolic interaction for code correction.** Through quantitative and qualitative analysis, we will investigate how LLMs respond to formally-grounded feedback and characterize the trade-offs between the deeper bug-finding capability of SMT solvers and their associated computational overhead.

#### **Significance**
This research promises significant contributions to both the scientific community and practical software development. Scientifically, it presents a novel architecture for fusing generative AI with formal methods, a key challenge highlighted by the VerifAI workshop. By using the SMT solver not for direct repair but as an intelligent information source for the LLM, our work explores a new point in the design space of neuro-symbolic systems. Practically, this research has the potential to produce more reliable and trustworthy AI-powered coding assistants. By proactively identifying and correcting deep logical flaws before they reach a human developer or a production environment, this approach can enhance software quality, improve developer productivity, and build greater trust in AI-generated artifacts. This work directly addresses the workshop's special theme on enhancing LLM-driven code generation by integrating techniques from the formal methods community.

### **2. Methodology**

Our proposed research methodology is structured around the development and evaluation of the **SMT-Repair** framework. The framework operates as an iterative refinement loop, depicted in Figure 1, composed of four primary stages: Generation, Verification, Translation, and Self-Correction.

  
*(A conceptual diagram would be placed here showing the loop: 1. LLM generates code from prompt -> 2. Code + Spec go to SMT Verifier -> 3. If Bug, SMT Solver returns Counterexample -> 4. C2P Module translates CEX to NL feedback -> 5. Feedback is added to prompt and looped back to 1. If no Bug, exit.)*

#### **Data Collection and Specification Generation**
The primary dataset for our experiments will be **HumanEval**, a widely-accepted benchmark for evaluating code generation models. It consists of 164 programming problems, each with a function signature, a detailed docstring, and a set of canonical unit tests used for evaluation.

A critical prerequisite for our methodology is the availability of formal specifications. As manual creation is not scalable, we will employ a hybrid strategy to generate these specifications ($\Phi$):

1.  **Manual Specification for a Core Subset:** For a hand-picked subset of problems involving clear mathematical or logical properties (e.g., sorting, searching, number theory), we will manually author formal specifications in the form of pre-conditions and post-conditions. For a Python function `def sort_list(arr: list) -> list:`, a post-condition could be expressed as `\forall i \in [0, \text{len(return\_value)}-2]: \text{return\_value}[i] \le \text{return\_value}[i+1]`.

2.  **Automated Property Inference:** To scale our approach, we will leverage the problem docstrings and provided unit tests to automatically infer properties. We will use techniques inspired by property-based testing. For instance, a docstring stating a function "reverses a string" implies the property `reverse(reverse(s)) == s`. Similarly, tests like `assert f([1,2]) == [2,1]` can be abstracted to infer that the output is a permutation of the input. We will use lightweight static analysis and pattern matching on the docstrings to generate these assertions.

#### **Core Algorithmic Steps**

The SMT-Repair process is iterative. Let $P_0$ be the initial problem prompt (docstring and signature), and $N_{max}$ be the maximum number of allowed repair attempts.

**Step 1: Initial Code Generation (Iteration $i=0$)**
The process begins with the LLM generating an initial code candidate, $C_0$, based on the prompt $P_0$.
$$
C_0 = \text{LLM}(P_0)
$$
We will use a state-of-the-art instruction-tuned LLM, such as GPT-4 or Llama-3-70B-Instruct, with a low temperature setting (e.g., $T=0.2$) to promote deterministic and high-quality outputs.

**Step 2: Formal Verification via SMT Solver**
The generated code $C_i$ is combined with its formal specifications $\Phi$ to create a verification problem. The goal is to check if the code satisfies the specifications, which is equivalent to checking the unsatisfiability of the logical formula representing the code's execution path combined with the negation of the specifications.
$$
\text{Verify}(C_i, \Phi) \rightarrow \text{Result} \in \{\text{UNSAT}, (\text{SAT}, M_i)\}
$$
*   To achieve this, we will use a Python-to-SMT translation layer. We can leverage existing symbolic execution tools like `crosshair-tool` or implement a focused translator for a tractable subset of Python (integers, booleans, lists). This layer will convert the Python function $C_i$ and the specification $\Phi$ into an SMT-LIB formula.
*   The formula is then passed to an SMT solver, such as **Z3**.
*   If the result is **UNSAT** (unsatisfiable), it means no counterexample was found, and the code $C_i$ is considered correct with respect to $\Phi$. The loop terminates successfully.
*   If the result is **SAT** (satisfiable), the solver has found a bug. It returns a model, $M_i$, which is a concrete counterexample—a set of input variable assignments that causes the code to violate a specification. For example, for a function `add(x,y)`, a counterexample for an overflow bug might be $M_i = \{x \mapsto 2^{31}-1, y \mapsto 1\}$.

**Step 3: Counterexample-to-Prompt (C2P) Translation**
This is the innovative core of our framework. If verification fails, the counterexample model $M_i$ is passed to the C2P module, $T$. This module translates the symbolic model into a natural language feedback string, $F_i$.
$$
F_i = T(M_i, C_i, \Phi)
$$
The translation strategy will be template-based and context-aware:
*   **Basic Template:** "Your code fails for the following inputs: `x=...`, `y=...`."
*   **Error-Specific Template:** If the symbolic execution trace can identify the error type (e.g., `ArithmeticError`, `IndexError`), the template becomes more informative: "Your code produces an integer overflow when `x` is the maximum integer value and `y` is 1."
*   **Post-condition Violation Template:** If a specific post-condition $\phi \in \Phi$ was violated, the feedback will reference it: "Your sorting function fails on the input `arr=[3,1,2]`. The output should be sorted, but your code returned `[3,1,2]`. The condition `result[0] <= result[1]` was violated."

**Step 4: Iterative Self-Correction**
The natural language feedback $F_i$ is appended to the prompt history to create a new, more informed prompt, $P_{i+1}$, for the next iteration.
$$
P_{i+1} = P_i \oplus \text{CorrectionPrompt}(F_i)
$$
where $\oplus$ represents structured concatenation and `CorrectionPrompt` is a predefined conversational wrapper, such as:
`"\nHere is a report on a bug found in your previous code:\n---\n" + F_i + "\n---\nPlease provide a corrected version of the function."`
The LLM then generates a new code candidate, $C_{i+1} = \text{LLM}(P_{i+1})$. The process increments $i$ and returns to Step 2. The loop terminates if verification passes or if the maximum number of iterations, $N_{max}$ (e.g., $N_{max}=5$), is reached.

#### **Experimental Design and Validation**

To rigorously evaluate SMT-Repair, we will conduct a controlled experiment comparing its performance against two strong baselines.

**Baselines:**
1.  **Zero-Shot LLM:** The initial code generation $C_0$ without any repair attempts. This establishes the baseline capability of the LLM.
2.  **Unit-Test-Based Self-Correction (UT-Repair):** A self-correction loop inspired by prior work (Chen et al., 2023). In this baseline, the generated code $C_i$ is executed against the canonical unit tests. If a test fails, the feedback prompt $F_i$ will be of the form: "Your code failed the test case `assert f(args) == expected_output`. Your function returned `actual_output` instead." This mirrors current self-debugging practices.

**Experimental Setup:**
*   **Models:** We will use one or more publicly accessible, high-performance LLMs (e.g., via the APIs for GPT-4-Turbo and Llama-3-70B-Instruct).
*   **Environment:** All experiments will be conducted in a controlled environment to ensure reproducibility. We will fix model parameters like temperature and top-p.
*   **Dataset:** We will perform our evaluation on the full HumanEval dataset, using our hybrid specification generation method. We will report on the subset of problems for which we can successfully generate specifications.

**Evaluation Metrics:**
1.  **Functional Correctness (Primary):**
    *   **pass@k:** This is the standard metric for code generation. We will measure `pass@1` by running our entire process (generation and up to $N_{max}$ repairs) once for each problem and checking if the final code candidate passes all hidden unit tests. We will compare the `pass@1` scores of SMT-Repair, UT-Repair, and the Zero-Shot baseline. This metric will be our primary indicator of success.
2.  **Repair Efficiency:**
    *   **Average Repair Iterations:** For problems that are successfully solved, we will measure the mean number of iterations required by SMT-Repair and UT-Repair. We hypothesize that the targeted feedback from SMT-Repair will lead to a lower number of iterations.
    *   **Convergence Rate:** We will plot the percentage of problems solved as a function of the iteration number ($i=0, 1, ..., N_{max}$). This will visualize how quickly each method converges to a correct solution.
    *   **Wall-Clock Time Analysis:** We will profile the time spent in each component (LLM inference, SMT verification, translation). This will allow for a detailed analysis of the overhead introduced by the SMT solver and help determine the practical viability of the approach.
3.  **Qualitative Analysis:**
    *   We will perform a manual error analysis on a sample of problems where SMT-Repair succeeds and the baselines fail (and vice-versa). We will examine the generated feedback prompts ($F_i$) and the corresponding code modifications made by the LLM to gain insights into the neuro-symbolic reasoning process.

### **3. Expected Outcomes & Impact**

This research is poised to deliver a set of tangible outcomes that will advance the state of the art at the intersection of generative AI and formal verification.

#### **Expected Outcomes**
1.  **A Fully-Functional, Open-Source SMT-Repair Prototype:** We will release the complete implementation of our framework, including the Python-to-SMT translation layer and the C2P module. This will provide a valuable tool for the research community to build upon and replicate our results.

2.  **State-of-the-Art Performance on Code Correction:** We expect SMT-Repair to significantly outperform both zero-shot generation and unit-test-based self-correction in terms of the `pass@1` metric on the HumanEval benchmark. The formal rigor of SMT-based bug finding is anticipated to uncover and enable the correction of complex logical errors and edge cases that are missed by standard unit tests, leading to a higher final correctness rate.

3.  **Quantitative Evidence of Improved Repair Efficiency:** We hypothesize that the precision of SMT-generated feedback will reduce the number of repair attempts needed to find a correct solution. Our results will include a quantitative comparison of the average number of iterations and convergence rates, illustrating that while each SMT-Repair iteration may be slower, the overall path to a correct solution may be shorter and more reliable.

4.  **Novel Insights into Neuro-Symbolic Prompting:** Our qualitative analysis will shed light on how complex, formal reasoning can be effectively "communicated" to an LLM through natural language. This will contribute to the broader field of prompt engineering, offering strategies for guiding LLMs in tasks that require high levels of logical precision and structured reasoning.

#### **Potential Challenges and Mitigation Strategies**

*   **Challenge 1: Scalability of Python-to-SMT Translation:** Translating the full semantics of a dynamic language like Python into SMT-LIB is notoriously difficult, especially with complex data structures and libraries.
    *   **Mitigation:** Our approach will be pragmatic. We will initially focus on a well-defined, verifiable subset of Python that covers a large portion of the HumanEval problems (arithmetic, lists, basic control flow). We will leverage existing libraries like `py-smt` and `crosshair-tool` where possible and clearly define the scope of our verification.

*   **Challenge 2: Computational Overhead of SMT Solvers:** The verification step could become a bottleneck, making the iterative process impractically slow.
    *   **Mitigation:** We will implement strict timeouts for the SMT solver in each iteration. The goal is not to exhaustively prove correctness but to find a single counterexample quickly. If the solver times out, we can fall back to the UT-Repair mechanism for that iteration, creating a hybrid system that balances rigor and speed.

*   **Challenge 3: LLM's Response to Feedback:** The LLM might misunderstand, ignore, or over-correct in response to the C2P feedback, potentially entering a loop of incorrect repairs.
    *   **Mitigation:** We will experiment with various prompt templates for the C2P module, fine-tuning the verbosity and structure of the feedback. The inclusion of a maximum iteration limit ($N_{max}$) prevents infinite loops. A qualitative analysis of failed repair chains will inform future improvements to the prompting strategy.

#### **Broader Impact**
The impact of this research extends beyond the immediate academic contribution. For the **VerifAI community**, this project provides a concrete and novel instantiation of the synthesis between formal methods and generative AI, directly answering the workshop's call for research on integrating formal tools to enhance LLMs. By demonstrating a practical feedback loop, we contribute a powerful pattern for building more reliable AI systems.

For the **software engineering industry**, this work represents a step towards building truly trustworthy AI coding assistants. By embedding formal verification directly into the code generation lifecycle, the SMT-Repair framework offers a path to creating tools that not only write code faster but also write code that is safer, more robust, and less prone to costly bugs. This could fundamentally improve the reliability of automated software development and CI/CD pipelines.

Finally, for the broader fields of **AI Safety and Trustworthiness**, our research exemplifies a "neuro-symbolic guardrail" approach. By using a deterministic, symbolic reasoner to check and guide the outputs of a probabilistic, generative model, we demonstrate a scalable principle for making AI systems more aligned with human-specified constraints and intentions. This fusion of powerful generation with rigorous verification is a critical step towards building artificial intelligence that we can confidently deploy in high-stakes environments.