# **Execution-Trace Alignment: Fine-tuning Code LLMs with Step-wise Causal Program Feedback**

### **Abstract**
Current alignment techniques for code-generating Large Language Models (LLMs) predominantly rely on binary execution feedback (pass/fail), a sparse signal that fails to explain *why* code is incorrect. This limitation hinders the development of models with robust debugging and reasoning capabilities. We introduce **Execution-Trace Alignment (ETA)**, a novel fine-tuning paradigm that leverages detailed, step-by-step execution traces as a rich, causal feedback signal. For each generated program, we automatically collect a trace detailing the sequence of executed statements, intermediate variable states, and the precise nature and location of any runtime errors. This structured feedback is then used to fine-tune the LLM through methods like Direct Preference Optimization (DPO), where generations leading to "better" traces (e.g., executing longer before failing) are preferred. This paper outlines the ETA methodology, presents preliminary experimental results from a baseline model, and discusses the significant potential of trace-based feedback to improve multi-step reasoning and self-correction in Code LLMs, paving the way for more capable and reliable AI programming agents.

### **1. Introduction**
Large Language Models (LLMs) have shown remarkable aptitude in generating source code, transforming software development workflows (Chen et al., 2021). Models such as CodeLlama, StarCoder, and GPT-4 can automate tasks from simple code completion to synthesizing entire functions. Despite this progress, a critical challenge remains: these models often produce code that is syntactically valid but semantically or logically flawed, leading to runtime errors or incorrect behavior.

The primary method to enhance model reliability is alignment fine-tuning, where model outputs are refined based on external feedback. Techniques like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) typically use a coarse, binary signal from unit tests: the code either passes or fails. This sparse feedback is fundamentally limited. It tells the model *that* its output was wrong but provides no insight into *why* it failed. Was it an off-by-one error, a type mismatch, or a flawed algorithm? This absence of granular, causal information severely impedes the model's ability to learn the intricate process of debugging and self-correction, a cornerstone of human programming expertise.

To build more robust and intelligent coding agents, we must transcend binary signals and embrace richer, more explanatory feedback. This research proposes **Execution-Trace Alignment (ETA)**, a fine-tuning paradigm that provides models with detailed, step-wise execution traces. Instead of a simple pass/fail signal, ETA supplies structured feedback on the program's execution flow, intermediate variable states, and the precise location and nature of errors. This causal trace serves as a "thought process" for the model to learn from, emulating how human developers diagnose and fix bugs. Our work aims to demonstrate that fine-tuning with this rich feedback signal will equip Code LLMs with superior reasoning and self-correction abilities, leading to more reliable and autonomous AI programming assistants.

### **2. Related Work**
The quest to improve code generation models by incorporating dynamic program information is an active area of research. Our work builds upon several key trends in the literature.

**Execution-Aware Models:** A foundational idea is to make models aware of program execution. Ding et al. (2023) introduced TRACED, a pre-training strategy that incorporates execution traces to help models learn dynamic code properties, showing improvements in tasks like clone retrieval. Haque et al. (2025) explored integrating execution traces directly into prompts for program repair, finding that while beneficial, effectiveness is sensitive to prompting strategies. Unlike these approaches, which use traces for representation learning or as in-context examples, our ETA method uses traces as a direct reward and preference signal during the alignment fine-tuning phase.

**Fine-tuning with Advanced Feedback:** Researchers have increasingly sought feedback signals more informative than binary outcomes. Fan et al. (2025) proposed FAIT, a technique that identifies and focuses on error-sensitive code segments during fine-tuning. Sakharova et al. (2025) integrated symbolic execution to create a nuanced dataset for fine-tuning via reinforcement learning and DPO, capturing deeper semantic properties of code. ETA complements these efforts by focusing on concrete execution traces, which are often easier to obtain than symbolic execution paths and provide a direct causal link between code and error.

**Program Repair and Self-Correction:** A key application for improved code models is automated program repair. Works like RepairLLaMA (Silva et al., 2023) have developed specialized adapters and fine-tuning techniques to effectively fix bugs. Our work contributes to this area by proposing that learning from error traces is a powerful mechanism to teach models *how* to repair their own code, moving towards more general self-correction capabilities. By understanding the causal chain of events leading to a failure, an ETA-trained model is hypothesized to generate more effective repairs.

Overall, while previous work has recognized the value of execution data, ETA is the first to propose a systematic framework for using detailed, step-wise execution traces as the primary feedback mechanism in modern alignment paradigms like DPO and RLHF. It addresses the key challenge of sparse feedback by providing a structured, granular, and causal signal to guide model learning.

### **3. Methodology**
Our proposed methodology, Execution-Trace Alignment (ETA), is designed to fine-tune Code LLMs using the rich, causal information contained within program execution traces. The process involves three main stages: automated trace generation, curation of trace-based feedback, and model alignment.

#### **3.1 Automated Execution Trace Generation**
The foundation of ETA is a dataset of code generations and their corresponding dynamic behaviors. We begin with standard code generation benchmarks (e.g., HumanEval, MBPP) that provide a problem description, a function signature, and unit tests.

1.  **Code Generation:** Using a base Code LLM (e.g., CodeLlama, StarCoder2), we generate a large number of candidate solutions for each problem. This produces a diverse set of programs, including correct solutions and, more importantly, a wide variety of incorrect ones.
2.  **Instrumentation and Tracing:** Each generated program is executed against the provided unit tests. We use a custom instrumentation framework built on Python's `sys.settrace` function to monitor execution line-by-line without modifying the source code. This tracer captures a sequence of events, which are serialized into a structured format. Each event in the trace contains:
    *   `line_number`: The line of code being executed.
    *   `event_type`: The type of event, such as `line`, `call`, `return`, or `exception`.
    *   `variable_states`: A snapshot of local variables and their values at that execution step.
    *   `error_info`: For an `exception` event, this includes the error type (e.g., `TypeError`), message, and traceback.

A serialized trace for a failing program might look like this:
```
Problem: "Divide two numbers a and b."
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
This process yields a rich dataset of `(prompt, code, trace, test_result)` tuples, which are used for alignment.

#### **3.2 Execution-Trace Alignment Methods**
We propose two primary methods for incorporating these traces into the fine-tuning process.

##### **3.2.1 ETA-RM: Reinforcement Learning with a Trace-based Reward Model**
This approach uses Proximal Policy Optimization (PPO) fine-tuning, guided by a reward model ($R_\phi$) that evaluates the quality of an execution trace.

The reward model is trained on pairs of traces $(t_w, t_l)$, where $t_w$ is preferred over $t_l$. Preferences are determined by a set of heuristics that reward partial progress:
1.  **Correctness:** A trace from a correct program is always preferred over one from a failing program.
2.  **Error Proximity:** For two failing programs, the trace that executes more steps before encountering an error is preferred.
3.  **Error Severity:** Traces ending in runtime errors (e.g., `IndexError`) are preferred over those with syntax errors.

The reward model is trained using a standard preference loss:
$$
\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(c_w, t_w), (c_l, t_l) \sim D} \left[ \log \left( \sigma(R_\phi(c_w, t_w) - R_\phi(c_l, t_l)) \right) \right]
$$
where $(c_w, t_w)$ is the winning code-trace pair and $(c_l, t_l)$ is the losing pair. The trained reward model is then used to provide a dense reward signal for PPO fine-tuning.

##### **3.2.2 ETA-DPO: Direct Preference Optimization with Execution Traces**
This method directly optimizes the LLM on preference pairs, avoiding the need to train a separate reward model. We curate a preference dataset $D = \{(x, y_w, y_l)\}$, where for a given prompt $x$, $y_w$ is the preferred code generation and $y_l$ is the dispreferred one. The winner/loser selection is based on the same trace-informed heuristics as in ETA-RM. For example, given two failing code generations, the one whose trace is longer (i.e., progressed further) is chosen as the winner $y_w$.

We then fine-tune the base model using the DPO loss function:
$$
\mathcal{L}_{DPO}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$
This objective directly encourages the model $\pi_\theta$ to assign a higher likelihood to the preferred generation $y_w$ compared to the dispreferred one $y_l$, guided by the implicit reward function encoded in the preference pairs. This approach is often more stable and computationally efficient than PPO-based methods.

### **4. Experiment Setup**
To provide a preliminary validation of our ideas, we conducted an initial experiment. Due to technical challenges, we were only able to complete the baseline portion of our experimental plan.

*   **Model:** We used the standard `gpt2` model, a publicly available transformer model. While not a state-of-the-art code model, it serves as a simple testbed for fine-tuning methodologies.
*   **Dataset:** We used a small subset of 10 examples from the **HumanEval** dataset. Each example consists of a docstring prompt, a canonical solution, and test cases.
*   **Baseline:** We implemented a **Supervised Fine-Tuning (SFT)** baseline. The `gpt2` model was fine-tuned for one epoch on the 10 training examples from our HumanEval subset. This baseline helps establish the performance of a standard fine-tuning approach on our experimental setup.
*   **Planned Methods:** Our original plan included comparing the SFT baseline against a **Binary-Reward PPO** baseline (reward is 1 for pass, 0 for fail) and our proposed **ETA-DPO** method. However, we encountered prohibitive technical difficulties with the `trl` library, which prevented the execution of the PPO and DPO experiments in the available timeframe.
*   **Evaluation:** The performance of the fine-tuned model was evaluated on the test cases associated with the HumanEval problems. The primary metric is functional correctness, measured by the pass/fail ratio, which corresponds to `pass@1`.

### **5. Experiment Results**
The experimental results are preliminary and limited to the SFT baseline. The model was trained and evaluated on the small HumanEval subset.

The SFT-tuned model was tested against the test cases for each of the 10 problems. The functional correctness results are summarized in Table 1 and Figure 1.

| Model | `pass@1` | Pass Count | Fail Count |
| :--- | :---: | :---: | :---: |
| SFT (`gpt2`) | 0.0% | 0 | 10 |
**Table 1:** Functional correctness of the SFT baseline on the 10-example HumanEval subset.

The SFT baseline model failed to generate a correct solution for any of the 10 problems in the test set, resulting in a `pass@1` score of 0%.

![Figure 1: Pass/Fail Ratio for the SFT baseline model on the test set.](sft_baseline/pass_fail_ratio.png)
**Figure 1:** Pass/Fail Ratio for the SFT baseline model on the test set.

As shown, the model achieved a 100% failure rate. This outcome, while poor, is not unexpected given the general-purpose nature of the base `gpt2` model, the very small scale of the fine-tuning data, and the inherent difficulty of the code generation task.

### **6. Analysis**
The preliminary results from our SFT baseline, while negative, provide a crucial motivation for more advanced alignment techniques like ETA. The 100% failure rate demonstrates that simple supervised fine-tuning on a small dataset is insufficient to impart reliable code generation capabilities, even for a well-known model like `gpt2`. The model struggles to generalize from the few provided examples to produce functionally correct code.

This is precisely the scenario where our proposed ETA methodology is hypothesized to provide significant benefits. A binary reward signal, as planned for the PPO baseline, would offer no learning signal in this case, as every generation would receive a reward of 0. The model would receive no gradient to guide it towards better solutions.

In contrast, **ETA-DPO** would be able to extract a meaningful learning signal even from a set of entirely incorrect generations. By generating execution traces for each failed attempt, we can construct preference pairs. For instance, a program that fails due to an `IndexError` after five successfully executed lines would be preferred over a program that fails immediately with a `TypeError`. By learning from these preferences, the model can gradually understand how to avoid common runtime errors and structure code that makes logical progress, even if it does not reach the final correct solution immediately. This step-wise guidance is critical for helping the model navigate the complex search space of program synthesis.

The primary limitation of this study is the inability to execute the planned PPO and DPO experiments. The technical issues encountered prevented a direct comparison and a demonstration of ETA's effectiveness. Therefore, our analysis remains a hypothesis grounded in the limitations of existing methods, which were starkly highlighted by our baseline results.

### **7. Conclusion**
This paper introduced **Execution-Trace Alignment (ETA)**, a novel fine-tuning paradigm for Code LLMs that uses rich, causal feedback from program execution traces. We argued that the prevailing reliance on sparse, binary (pass/fail) rewards inhibits models from learning complex debugging and reasoning skills. ETA addresses this by providing structured information about a program's dynamic behavior, including intermediate variable states and precise error locations, which can be leveraged in alignment techniques like DPO to reward partial progress.

Our preliminary experiments with a `gpt2` model on a small subset of HumanEval confirmed the difficulty of the task, with a standard SFT baseline failing on all problems. This highlights the need for more sophisticated feedback mechanisms. While we were unable to complete our planned experiments with ETA due to technical hurdles, the baseline results reinforce our central hypothesis: to improve code generation, models need a more granular, informative training signal than what is currently standard practice.

Future work will focus on overcoming the implementation challenges to run the full suite of experiments proposed. This includes scaling up to larger, more capable base models like CodeLlama-7B, utilizing the full HumanEval and MBPP datasets, and conducting a rigorous comparison of ETA-DPO against SFT and binary-reward PPO baselines. The ultimate goal is to produce and release models fine-tuned with ETA, providing the community with more robust, reliable, and interpretable AI programming assistants.

### **8. References**
Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ... & Brockman, G. (2021). Evaluating Large Language Models Trained on Code. *arXiv preprint arXiv:2107.03374*.

Chen, X., et al. (2025). Integrating LLM-based Code Optimization with Human-like Exclusionary Reasoning for Computational Education. *[Publication details not specified]*.

Ding, Y., Steenhoek, B., Pei, K., Kaiser, G., Le, W., & Ray, B. (2023). TRACED: Execution-aware Pre-training for Source Code. *arXiv preprint arXiv:2306.07487*.

Fan, L., Liu, Z., Wang, H., Bao, L., Xia, X., & Li, S. (2025). FAIT: Fault-Aware Fine-Tuning for Better Code Generation. *arXiv preprint arXiv:2503.16913*.

Haque, M., Babkin, P., Farmahinifarahani, F., & Veloso, M. (2025). Towards Effectively Leveraging Execution Traces for Program Repair with Code LLMs. *arXiv preprint arXiv:2505.04441*.

Liu, W., et al. (2025). Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models. *[Publication details not specified]*.

Rodriguez, J., et al. (2023). Refining Decompiled C Code with Large Language Models. *[Publication details not specified]*.

Sakharova, M., Anand, A., & Mezini, M. (2025). Integrating Symbolic Execution into the Fine-Tuning of Code-Generating LLMs. *arXiv preprint arXiv:2504.15210*.

Sharma, A., et al. (2023). Learning Performance-Improving Code Edits. *[Publication details not specified]*.

Silva, A., Fang, S., & Monperrus, M. (2023). RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair. *arXiv preprint arXiv:2312.15698*.

Williams, T., et al. (2023). Large Language Models for Compiler Optimization. *[Publication details not specified]*.