### **1. Title: Execution-Trace Alignment: Fine-tuning Code LLMs with Step-wise Causal Program Feedback**

### **2. Introduction**

**2.1 Background and Motivation**
Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating source code, catalyzing a paradigm shift in software development. Models like CodeLlama, StarCoder, and GPT-4 are increasingly integrated into developer workflows, automating tasks ranging from code completion to entire function synthesis. However, despite their fluency, these Code LLMs frequently generate code that is syntactically correct but semantically or logically flawed, containing subtle bugs that lead to runtime errors or incorrect outputs. The predominant method for improving model reliability is through alignment fine-tuning, where models are adjusted based on human preferences or external feedback.

Current alignment techniques, such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), often rely on a coarse, binary feedback signal derived from unit test execution: the code either passes (positive reward) or fails (negative reward). This sparse feedback is fundamentally limited. It informs the model *that* its generation was incorrect, but provides no insight into *why* it failed—was it an off-by-one error, a type mismatch, or a flawed algorithm? This lack of granular, causal information severely hampers the model's ability to learn the intricate, multi-step process of debugging and self-correction, which is central to human programming expertise. To build more robust and intelligent coding agents, we must move beyond binary success signals and embrace richer, more explanatory feedback mechanisms.

Recent research has begun to explore the potential of incorporating dynamic program information. For instance, TRACED (Ding et al., 2023) pioneered the use of execution traces in a pre-training context to improve code representation. Concurrently, studies by Haque et al. (2025) have shown that including execution traces in prompts can aid program repair, though its effectiveness is highly dependent on prompting strategies. Other works like FAIT (Fan et al., 2025) focus on identifying error-sensitive code segments, and Sakharova et al. (2025) have integrated symbolic execution to enrich feedback. These efforts highlight a clear trend: the path to more capable Code LLMs lies in leveraging deeper, execution-aware program understanding. However, a systematic methodology for using step-wise, causal execution traces as a direct feedback signal in the alignment fine-tuning loop remains an open and critical research area.

**2.2 Research Objectives**
This research proposes **Execution-Trace Alignment (ETA)**, a novel fine-tuning paradigm that leverages detailed, step-by-step execution traces to teach Code LLMs not just to code, but to reason and debug. Instead of a simple pass/fail signal, we provide the model with structured feedback detailing the program's execution flow, intermediate variable states, and the precise location and nature of errors. This causal trace acts as a "thought process" that the model can learn from, mirroring how human developers analyze bugs.

The primary objectives of this research are:
1.  **To develop an automated framework for generating structured, causal execution traces from faulty code.** This framework will instrument code, execute it against test cases, and capture a detailed log of its dynamic behavior, including variable states and error propagation.
2.  **To design and implement the Execution-Trace Alignment (ETA) methodology.** We will investigate two distinct fine-tuning strategies:
    *   **ETA-RM:** A reinforcement learning approach using a reward model trained to score the quality of an execution trace, rewarding not just correctness but also partial progress.
    *   **ETA-DPO:** A preference-based method that uses trace-informed heuristics to create a preference dataset, directly teaching the model to prefer code with more "correct" execution paths.
3.  **To conduct a comprehensive empirical evaluation of ETA.** We will fine-tune state-of-the-art base Code LLMs using ETA and benchmark their performance against standard baselines (SFT, binary-reward RLHF) on established code generation (HumanEval, MBPP, APPS) and program repair (QuixBugs) datasets.
4.  **To perform an in-depth analysis of the model's emergent capabilities.** We will qualitatively and quantitatively assess whether ETA enhances multi-step reasoning, self-correction, and the ability to fix complex, non-trivial bugs, providing insights into how granular feedback shapes model behavior.

**2.3 Significance**
This research is poised to make significant contributions to the field of deep learning for code. By introducing Execution-Trace Alignment, we aim to establish a new, more effective alignment paradigm that moves beyond sparse rewards and equips Code LLMs with genuine debugging capabilities. The successful development of this methodology will lead to more reliable, robust, and autonomous AI programming assistants. Such agents could drastically improve developer productivity by not only generating initial code drafts but also by identifying, explaining, and fixing their own errors. This directly addresses the challenges of "Post-training and Alignment for Code" and paves the way for advanced "Agentic Methods for Programming Tasks," as highlighted by the DL4C workshop. Furthermore, by committing to release our code, datasets, and models, we contribute to "Open Science and Responsible AI," providing the community with valuable tools and reproducible artifacts to build upon.

### **3. Methodology**

Our proposed methodology is divided into three core stages: (1) Data Generation: Creating a dataset of code, test cases, and their corresponding execution traces; (2.1) ETA-RM: Defining and training a trace-aware reward model for PPO-based fine-tuning; and (2.2) ETA-DPO: Curating trace-informed preference pairs for DPO-based fine-tuning; and (3) Experimental Design: A rigorous plan for evaluating our methods against strong baselines.

**3.1 Data Collection and Trace Generation**
The foundation of our approach is a rich dataset containing not just code problems and solutions, but also the dynamic behavior of those solutions.

**3.1.1 Base Datasets and Code Generation**
We will leverage established code generation benchmarks such as **HumanEval**, **MBPP (Mostly Basic Python Programs)**, and the more challenging **APPS** dataset. For each problem in these benchmarks, which consists of a natural language prompt, a function signature, and several unit tests, we will use a pre-trained base Code LLM (e.g., CodeLlama-7B-Instruct, StarCoder2-7B) to generate a diverse set of candidate solutions ($k > 100$ per problem). This process will naturally yield a mix of correct solutions that pass all tests and a majority of incorrect solutions that fail for various reasons.

**3.1.2 Automated Execution Trace Generation**
This is the cornerstone of our methodology. For each generated program, we will run it against the provided unit tests and capture a detailed execution trace using a custom instrumentation framework built upon Python's `sys.settrace` functionality. This allows us to monitor program execution on a line-by-line basis without source code modification.

The generated trace will be a structured sequence of events. Each event in the sequence will be represented as a JSON object with the following fields:
*   `line_number`: The line number being executed.
*   `event_type`: The type of event (e.g., 'line', 'call', 'return', 'exception').
*   `variable_states`: A snapshot of the local variables and their values at that point in execution. To manage verbosity, we will capture variable states only at 'line' events and limit the depth of complex objects.
*   `error_info`: If an `exception` event occurs, this field will contain the `error_type` (e.g., `ZeroDivisionError`), `error_message`, and the traceback.

**Trace Serialization:** For model consumption, this structured trace will be serialized into a compact, human-readable text format. For example:
```
Problem: "Divide two numbers a and b."
Generated Code:
1: def divide(a, b):
2:   if b == 0:
3:     return float('inf')
4:   result = a / b
5:   return result

Test Case: divide(10, 0)
Execution Trace:
[L1] call: a=10, b=0
[L2] line: a=10, b=0 | Condition (b == 0) is True
[L3] line: a=10, b=0 | Returning float('inf')
[L3] return: 'inf'
Test Result: PASS
---
Generated Code:
1: def divide(a, b):
2:   result = a / b
3:   return result

Test Case: divide(10, 0)
Execution Trace:
[L1] call: a=10, b=0
[L2] line: a=10, b=0 | EXCEPTION: ZeroDivisionError("division by zero")
Test Result: FAIL
```
This library of (prompt, generated_code, execution_trace, test_result) tuples will form the master dataset for both of our proposed alignment methods.

**3.2 Execution-Trace Alignment (ETA) Methods**

**3.2.1 Method 1: Reinforcement Learning with a Trace-based Reward Model (ETA-RM)**
This method uses Proximal Policy Optimization (PPO) fine-tuning, guided by a sophisticated reward model that understands execution traces.

*   **Reward Model (RM) Design and Training:**
    Our reward model, $R_\phi$, will be a transformer encoder (e.g., initialized from CodeBERT) followed by a regression head. It takes a concatenated input of `(problem_prompt, generated_code, serialized_trace)` and outputs a scalar reward.
    To train $R_\phi$, we require a preference dataset of trace pairs $(t_w, t_l)$, where $t_w$ is the "winning" (preferred) trace and $t_l$ is the "losing" (dispreferred) trace for the same problem. Preferences will be determined by the following hierarchy of heuristics:
    1.  **Correctness:** A trace from a correct, passing program is always preferred over a trace from a failing program.
    2.  **Error Proximity:** For two failing traces, the one that executes more steps before crashing is preferred. This rewards progress.
    3.  **Error Severity:** Traces ending in runtime errors (e.g., `IndexError`) are preferred over those with syntax errors, as the former indicate a more complete program structure.
    4.  **Partial Correctness:** If intermediate assertions can be checked, a trace that passes more assertions is preferred.

    The reward model is trained to minimize the binary cross-entropy loss on these preferences:
    $$
    \mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(c_w, t_w), (c_l, t_l) \sim D} \left[ \log \left( \sigma(R_\phi(c_w, t_w) - R_\phi(c_l, t_l)) \right) \right]
    $$
    where $(c_w, t_w)$ is the code-trace pair for the winner and $(c_l, t_l)$ for the loser.

*   **PPO Fine-tuning:**
    The base Code LLM (the policy, $\pi_\theta$) is fine-tuned using PPO. At each step, for a given prompt $x$ from our dataset, the policy $\pi_\theta$ generates a code completion $y$. This code is executed to get its trace $t$, and the reward is calculated as $r = R_\phi(x, y, t)$. The PPO objective maximizes this reward while regularizing against large policy shifts:
    $$
    \text{maximize}_{\theta} \quad \mathbb{E}_{x \sim D, y \sim \pi_{\theta}(\cdot|x)} \left[ r - \beta \cdot \text{KL}(\pi_{\theta}(\cdot|x) || \pi_{\text{ref}}(\cdot|x)) \right]
    $$
    where $\pi_{\text{ref}}$ is the initial supervised fine-tuned (SFT) model and $\beta$ is the KL-divergence coefficient.

**3.2.2 Method 2: Direct Preference Optimization with Execution Traces (ETA-DPO)**
This method bypasses the explicit reward modeling step and directly optimizes the LLM on preference pairs.

*   **Preference Data Curation:**
    We will create a preference dataset $D = \{(x, y_w, y_l)\}$, where for a prompt $x$, $y_w$ is the preferred code generation and $y_l$ is the dispreferred generation. The choice of winner/loser is based on the same trace-informed heuristics as in ETA-RM: a correct solution is preferred over a failing one, and for two failing solutions, the one with the "better" trace (e.g., runs longer, has a less severe error) is chosen as the winner. This directly encodes the debugging intuition into the preference data.

*   **DPO Fine-tuning:**
    We fine-tune the SFT base model $\pi_\text{ref}$ using the DPO objective function. DPO implicitly optimizes a reward function by directly working with log-probabilities of preferred and dispreferred responses. The loss function is:
    $$
    \mathcal{L}_{DPO}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
    $$
    This objective encourages the model $\pi_\theta$ to increase the relative log-probability of the winning response $y_w$ over the losing response $y_l$. The hyperparameter $\beta$ controls how much the model deviates from the reference policy. ETA-DPO is computationally more efficient than PPO as it does not require training a separate reward model or sampling from the policy during training.

**3.3 Experimental Design**
We will conduct a rigorous set of experiments to validate our approach.

*   **Base Models:** We will use publicly available, state-of-the-art Code LLMs such as **CodeLlama-7B-Instruct** and **StarCoder2-7B** as our base models ($\pi_\text{ref}$).

*   **Baselines for Comparison:**
    1.  **SFT Base Model:** The original, instruction-tuned model without any further alignment.
    2.  **Binary-Reward PPO:** A standard RLHF baseline fine-tuned with PPO where the reward is simply $1$ if the code passes all unit tests and $0$ otherwise. This is our primary ablation to demonstrate the value of granular trace feedback.
    3.  **Trace-in-Prompt:** A strong baseline inspired by Haque et al. (2025), where a model is prompted with a buggy code snippet and its execution trace, and asked to fix it in a few-shot setting. This tests the utility of traces as input versus as a training signal.

*   **Evaluation Datasets and Tasks:**
    1.  **Code Generation:** We will evaluate on held-out problems from **HumanEval**, **MBPP**, and **APPS**.
    2.  **Program Repair:** We will use a program repair benchmark like **QuixBugs** to explicitly measure self-correction capabilities. For each bug, we will provide the model with the faulty code and ask for a fix.

*   **Evaluation Metrics:**
    1.  **Functional Correctness:** The primary metric will be **`pass@k`**, calculated as $1 - \frac{C(n-k, n)}{C(n,k)}$, where we generate $n$ samples per problem and check if at least one passes the tests. We will focus on `pass@1` for its practical relevance.
    2.  **Repair Accuracy:** For the program repair task, we will measure the percentage of bugs the model successfully fixes in a single attempt.
    3.  **Qualitative Analysis:** We will manually inspect a sample of generations from our ETA models versus the baselines. We will categorize the types of errors the models fix (e.g., off-by-one, null pointer, algorithmic logic flaw) to understand if ETA leads to a deeper, more sophisticated form of reasoning. We will also analyze cases where ETA fails to understand what benefits or limitations our trace representation has.

### **4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
We anticipate that this research will yield several key outcomes:
1.  **A Novel and Superior Alignment Methodology:** We expect both ETA-RM and ETA-DPO to significantly outperform the binary-reward PPO baseline and the SFT model on all evaluation benchmarks. We hypothesize that ETA-DPO may prove more stable and effective due to its direct optimization nature. We project a substantial improvement in `pass@1` on challenging datasets like APPS, where multi-step reasoning is crucial.
2.  **More Robust and Capable Code Models:** The resulting fine-tuned models (e.g., CodeLlama-7B-ETA) will not only be better code generators but will also exhibit demonstrable self-correction abilities. When faced with a bug, they will be more likely to produce a valid fix compared to models trained without causal feedback.
3.  **An Open-Source Framework and Dataset:** In line with the principles of Open Science, we will release our entire framework as an open-source library. This includes the code for trace generation, the curated preference datasets derived from traces, and the fine-tuning scripts for both PPO and DPO. This will provide a valuable resource for the research community to build upon.
4.  **Actionable Insights into Model Reasoning:** Our qualitative analysis will provide critical insights into how Code LLMs process and learn from different forms of feedback. By comparing the behavior of models trained on binary vs. trace-based signals, we will shed light on the mechanisms that enable emergent debugging and reasoning skills.

**4.2 Impact**
The impact of this research will extend across scientific and practical domains:
*   **Scientific Impact:** This work will challenge the prevailing reliance on sparse, binary feedback in LLM alignment for code. By demonstrating the efficacy of causal, step-wise execution traces, we aim to establish a new frontier in alignment research, inspiring further exploration into structured and explanatory feedback mechanisms for complex, logic-driven tasks beyond just code. It will contribute directly to the fields of Machine Learning for Code, Reinforcement Learning, and Automated Software Engineering.
*   **Practical Impact:** The primary practical impact lies in the creation of significantly more reliable AI-powered developer tools. An AI assistant that can reason about its own errors, explain the cause of a bug using a trace, and propose a valid fix represents a major leap in Human-Computer Interaction for code. This will enhance developer productivity, reduce debugging time, and make AI-assisted software development more trustworthy and effective. By improving the core debugging competency of Code LLMs, this research lays the groundwork for the next generation of truly agentic systems capable of autonomously tackling complex software engineering tasks.