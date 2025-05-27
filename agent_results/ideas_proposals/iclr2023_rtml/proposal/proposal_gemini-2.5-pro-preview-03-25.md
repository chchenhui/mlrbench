## 1. Title

**PEFT-Unlearn: A Scalable Framework for Targeted Machine Unlearning in Large Language Models using Parameter-Efficient Fine-Tuning**

## 2. Introduction

**2.1 Background**

The rapid advancement and deployment of Large Language Models (LLMs) have revolutionized numerous fields, demonstrating remarkable capabilities in natural language understanding, generation, and reasoning. These models, often comprising billions or trillions of parameters trained on web-scale datasets, form the backbone of modern AI applications ranging from chatbots and content creation tools to sophisticated decision support systems. However, the very scale that empowers these models also introduces significant risks related to trustworthiness and reliability (Task Description). LLMs have been shown to inherit and potentially amplify societal biases present in their training data, generate toxic or harmful content, and inadvertently memorize and leak sensitive personal information (e.g., names, addresses, medical details) encountered during pre-training.

These issues pose substantial challenges for deploying LLMs in mission-critical domains like healthcare, finance, and education, where fairness, privacy, and robustness are paramount. Regulatory frameworks like the European Union's General Data Protection Regulation (GDPR), with its "right to be forgotten," further mandate the ability to remove specific user data from trained models upon request. The naive solution – retraining the entire model from scratch on a modified dataset excluding the problematic data – is computationally prohibitive for large-scale models, often requiring weeks or months of GPU time and incurring enormous financial and environmental costs.

This necessitates the development of **Machine Unlearning** techniques: methods designed to efficiently remove the influence of specific data points or subsets (the "forget set") from a trained model, ideally approximating the state the model would have reached if trained without that data from the outset, while preserving the model's general knowledge and performance on the remaining data (the "retain set"). While various unlearning approaches exist, many traditional methods struggle to scale effectively to LLMs, lack precision in targeting specific data influences, or significantly degrade overall model utility, leading to catastrophic forgetting. Recent works have begun exploring Parameter-Efficient Fine-Tuning (PEFT) techniques, such as Low-Rank Adaptation (LoRA) or Adapters, as promising avenues for more efficient model modification, including unlearning (e.g., [1, 2, 3]). These methods modify only a small fraction of the model's parameters, drastically reducing computational overhead.

**2.2 Research Objectives**

This research proposes **PEFT-Unlearn**, a novel framework designed to address the critical need for scalable, precise, and efficient machine unlearning specifically tailored for LLMs. The core idea is to strategically integrate PEFT methods with gradient-based influence estimation techniques to isolate and remove the impact of unwanted data (e.g., privacy-sensitive, biased, or toxic content) with minimal computational cost and impact on model utility.

The primary objectives of this research are:

1.  **Develop a Scalable PEFT-based Unlearning Algorithm:** Design and implement a novel algorithm that leverages PEFT modules (e.g., LoRA) to encapsulate and subsequently neutralize the influence of specified forget data $D_f$ within a pre-trained LLM.
2.  **Achieve Targeted and Effective Unlearning:** Ensure the algorithm effectively removes the specific information contribution of $D_f$, verified through rigorous evaluation metrics including membership inference attacks (MIAs), extraction attack resistance, and performance degradation on tasks dependent on $D_f$.
3.  **Preserve General Model Utility:** Minimize the negative impact of the unlearning process on the model's performance on the retain set $D_r$ and general downstream tasks, avoiding catastrophic forgetting.
4.  **Ensure Computational Efficiency:** Demonstrate significant computational savings (targeting <5% overhead of full retraining) in terms of time, memory, and FLOPs compared to retraining and other baseline unlearning methods.
5.  **Investigate Formal Unlearning Guarantees:** Explore the potential for the PEFT-Unlearn framework to provide formal guarantees, potentially approximating $\epsilon$-differential unlearning, ensuring a mathematically rigorous definition of data removal.
6.  **Establish Benchmarks and Tools:** Create a comprehensive benchmark suite for evaluating LLM unlearning techniques and release an open-source implementation of the PEFT-Unlearn framework.

**2.3 Significance**

This research directly tackles crucial challenges highlighted in the call for trustworthy and reliable large-scale machine learning models. By enabling efficient and effective removal of problematic data from LLMs, PEFT-Unlearn offers several significant contributions:

*   **Enhanced Trustworthiness:** Directly mitigates risks associated with privacy violations (data leakage) and fairness (bias amplification) by providing a mechanism to correct models post-deployment.
*   **Regulatory Compliance:** Provides a practical pathway for organizations deploying LLMs to comply with data protection regulations like GDPR, particularly the "right to be forgotten".
*   **Economic Feasibility:** Offers a cost-effective alternative to full model retraining, making responsible AI practices more accessible and sustainable for developers and organizations.
*   **Scientific Advancement:** Advances the field of machine unlearning by proposing a novel integration of PEFT and influence analysis tailored for the unique challenges of LLMs, contributing new algorithms and theoretical insights. Potentially informs future pre-training or fine-tuning strategies designed with unlearning capabilities in mind.
*   **Societal Benefit:** Ultimately contributes to the development of safer, fairer, and more reliable AI systems, fostering greater public trust and enabling the responsible application of LLMs in sensitive domains.

## 3. Methodology

**3.1 Overall Framework: PEFT-Unlearn**

The proposed PEFT-Unlearn framework operates on a pre-trained LLM, denoted by its parameters $\theta$, trained on an initial dataset $D = D_r \cup D_f$, where $D_r$ is the retain set and $D_f$ is the forget set. The goal is to produce an updated model $\theta'$ that approximates the model $\theta^*$ obtained by training only on $D_r$, such that $\theta'$ exhibits minimal influence from $D_f$ while retaining knowledge from $D_r$. The framework consists of the following key steps:

**Step 1: Influence-Guided PEFT Module Training**

Instead of directly modifying the base model $\theta$, we introduce PEFT modules (e.g., LoRA layers) with parameters $\phi$. The core idea is to make these PEFT parameters capture the influence of the forget set $D_f$, potentially guided by influence estimation.

Let $L(D, \theta)$ be the training loss function. We consider two potential strategies for this step:

*   **Strategy A (Corrective Fine-tuning):** Start with the original model $\theta$. Fine-tune the model using a PEFT method (parameters $\phi$) primarily focused on the forget set $D_f$. This could involve fine-tuning only on $D_f$ for a few steps, or fine-tuning on the full dataset $D$ but using gradient information related to $D_f$ to guide the updates to $\phi$. For instance, we could use influence functions or gradient analysis [5] to identify base parameters $\theta_{\text{influenced}}$ most affected by $D_f$. We then attach PEFT modules specifically to these layers/parameters and fine-tune them to capture the $D_f$-specific updates. The LoRA update for a weight matrix $W_0$ is $W = W_0 + BA$, where $A$ and $B$ are low-rank matrices ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, with $r \ll d, k$). The parameters $\phi$ would correspond to the entries of $A$ and $B$. The goal is that $\phi$ primarily represents the contribution of $D_f$.

*   **Strategy B (Dedicated Forget Modules):** Augment the pre-trained model $\theta$ with dedicated PEFT modules $\phi_f$. Train these modules specifically to 'memorize' or specialize on the forget set $D_f$, while keeping the base parameters $\theta$ frozen. This is akin to some prompt-tuning unlearning approaches [3] but using methods like LoRA potentially allows interaction with deeper model representations. The training objective for $\phi_f$ could be minimizing the loss on $D_f$:
    $$\min_{\phi_f} L(D_f, \theta, \phi_f)$$

**Step 2: Unlearning via PEFT Module Manipulation**

Once the PEFT parameters $\phi_f$ (representing the influence of $D_f$) are obtained, the unlearning operation targets these parameters specifically, leaving the base parameters $\theta$ largely untouched.

*   **Option 1 (Module Nullification/Removal):** If $\phi_f$ effectively isolated the $D_f$ influence (especially plausible with Strategy B or highly targeted Strategy A), unlearning can be achieved by simply setting $\phi_f = 0$ or removing the PEFT module entirely. This corresponds to reverting the model's state concerning the $D_f$-specific knowledge encoded in $\phi_f$.

*   **Option 2 (Gradient Ascent on PEFT):** To more actively counteract the learned influence of $D_f$, perform gradient ascent on the forget set loss $L(D_f, \theta, \phi_f)$, updating *only* the PEFT parameters $\phi_f$:
    $$\phi_f \leftarrow \phi_f + \eta \nabla_{\phi_f} L(D_f, \theta, \phi_f)$$
    This step aims to explicitly increase the model's 'error' or 'surprise' on the forget data, effectively forcing it to unlearn the patterns specific to $D_f$. This is related to gradient ascent methods [4] but constrained to the low-dimensional PEFT parameter space, potentially mitigating catastrophic forgetting and improving stability. The learning rate $\eta$ and the number of ascent steps are crucial hyperparameters.

**Step 3: Refinement Tuning (Optional but Recommended)**

The unlearning step (especially gradient ascent) might slightly perturb the model's overall calibration or fluency. To counteract this and reinforce knowledge from the retain set $D_r$, a lightweight refinement step can be performed.

*   Use a small, clean subset $D'_{r} \subset D_r$ or a separate calibration dataset $D_{calib}$.
*   Perform a few steps of fine-tuning, potentially using the same PEFT parameters $\phi_f$ or introducing a new, small PEFT module $\phi_{refine}$, optimizing a standard language modeling or task-specific objective on $D'_{r}$ or $D_{calib}$:
    $$\min_{\phi_f \text{ or } \phi_{refine}} L(D'_{r}, \theta, \phi_f / \phi_{refine})$$
    This helps stabilize the model and recover any minor utility loss incurred during unlearning, similar to approaches like ReLearn [9].

**3.2 Formal Guarantees: Towards Differential Unlearning**

We will investigate the potential of PEFT-Unlearn to satisfy a notion of approximate differential unlearning. $\epsilon$-differential unlearning requires that the distribution of the unlearned model $\mathcal{A}(D)$ (output of PEFT-Unlearn on $D$ for forgetting $D_f$) is close to the distribution of the model retrained from scratch $\mathcal{A}(D_r)$. Formally, for adjacent datasets $D$ and $D'$ differing by one sample $z \in D_f$, the unlearning mechanism $\mathcal{U}$ satisfies $\epsilon$-differential unlearning if for all possible output models $M$:
$$ \text{Pr}[\mathcal{U}(D) = M] \le e^{\epsilon} \text{Pr}[\mathcal{A}(D \setminus \{z\}) = M] $$
Achieving exact differential unlearning is highly challenging. We will focus on providing theoretical analysis for approximate guarantees under specific assumptions (e.g., convexity for certain loss landscapes, properties of PEFT updates). The isolation provided by PEFT modules might simplify the analysis compared to modifying the full parameter space. We may leverage techniques related to differential privacy guarantees for gradient-based updates or noise addition during the unlearning (gradient ascent) or refinement steps, drawing inspiration from differentially private optimization and recent work on privacy in unlearning [10]. We aim to quantify the privacy leakage related to $D_f$ using metrics like the success rate of Membership Inference Attacks (MIAs) and analyze how PEFT-Unlearn reduces this leakage compared to the original model.

**3.3 Experimental Design**

**3.3.1 Datasets and Models**

*   **Models:** We will primarily experiment with publicly available LLMs of varying sizes, such as the Llama family (e.g., Llama-2 7B, 13B) and potentially smaller models like GPT-2 for faster prototyping.
*   **Datasets:** Experiments will utilize subsets of large pre-training corpora (e.g., The Pile, C4) where specific documents or records can be designated as the forget set $D_f$.
    *   *Privacy Unlearning:* Identify documents containing Personally Identifiable Information (PII) using heuristic methods or existing annotated datasets (e.g., Enron email dataset subsets). $D_f$ = PII-containing documents.
    *   *Bias/Toxicity Unlearning:* Identify documents exhibiting strong biases (e.g., gender, race) or containing toxic language using classifiers (e.g., Perspective API). $D_f$ = biased/toxic documents.
    *   *Concept/Domain Unlearning:* Select documents related to a specific niche topic or domain (e.g., a specific author's work, a particular outdated scientific theory). $D_f$ = documents related to the target concept.
    *   *Synthetic Data:* Use controlled settings where a model is explicitly trained on specific key-value pairs or factual snippets designated as $D_f$.

**3.3.2 PEFT Method**

We will primarily focus on **LoRA (Low-Rank Adaptation)** due to its proven effectiveness and efficiency. We will experiment with different configurations (rank $r$, target layers, scaling factor $\alpha$). We may also explore other methods like Adapters for comparison.

**3.3.3 Baselines**

We will compare PEFT-Unlearn against several baselines:

1.  **Original Model:** The model trained on $D = D_r \cup D_f$. (Upper bound for utility, lower bound for unlearning efficacy).
2.  **Retrained Model:** The model trained from scratch only on $D_r$. (Gold standard for unlearning efficacy and utility).
3.  **Fine-tuning on Retain Set (FT-R):** Fine-tuning the original model on $D_r$. (Simple heuristic for unlearning).
4.  **Gradient Ascent (GA):** Standard unlearning via gradient ascent on $D_f$ applied to the full model or selected layers [4].
5.  **Influence Function-based methods:** Approximate unlearning by correcting parameters based on influence functions.
6.  **Other PEFT-based Unlearning:** Implementations or variants of existing PEFT unlearning methods like Fast-NTK [1] or LMEraser [3] if feasible within the scope.

**3.3.4 Evaluation Metrics**

We will evaluate the methods across three dimensions:

1.  **Unlearning Efficacy:**
    *   **Membership Inference Attack (MIA) Success Rate:** Train an MIA classifier to distinguish whether a given sample from $D_f$ was used in training the *unlearned* model. Lower success rate indicates better unlearning. Compare against MIA on the original and retrained models.
    *   **Extraction Attack Success:** For specific types of $D_f$ (e.g., PII, specific facts), design prompts to elicit the forgotten information. Measure the success rate or probability of extracting the targeted content.
    *   **Forget Set Loss:** Measure the model's loss (e.g., perplexity) on $D_f$. Effective unlearning should ideally increase this loss compared to the original model, moving closer to the retrained model's loss.
    *   **Downstream Task Performance (Forget-related):** If $D_f$ represents a specific concept or bias, evaluate performance on downstream tasks sensitive to it (e.g., accuracy on a classification task based on $D_f$, toxicity score of generated text). Unlearning should degrade performance specifically related to $D_f$.

2.  **Model Utility Preservation:**
    *   **Retain Set Performance:** Measure the model's loss (e.g., perplexity) on a held-out portion of $D_r$. Performance should remain close to the original or ideally the retrained model.
    *   **General Downstream Tasks:** Evaluate performance on standard NLP benchmarks (e.g., GLUE, SuperGLUE, summarization, translation tasks) unrelated to $D_f$. Performance should be minimally impacted compared to the original/retrained model.
    *   **Generation Quality:** Assess the fluency, coherence, and relevance of text generated by the unlearned model using metrics like BLEU, ROUGE, and potentially human evaluation.

3.  **Efficiency:**
    *   **Unlearning Time:** Wall-clock time required for the unlearning process.
    *   **Computational Cost:** Total FLOPs or GPU hours consumed.
    *   **Memory Footprint:** Peak GPU memory usage during unlearning.
    *   **Parameter Overhead:** Number and percentage of parameters modified or added (size of $\phi$). Compare these against the cost of full retraining.

**3.3.5 Ablation Studies**

We will conduct ablation studies to understand the contribution of each component of PEFT-Unlearn:
*   Impact of different PEFT methods (LoRA vs. Adapters).
*   Effectiveness of different unlearning operations (Nullification vs. Gradient Ascent).
*   Necessity and impact of the refinement tuning step.
*   Sensitivity to hyperparameters (LoRA rank $r$, gradient ascent learning rate $\eta$, number of steps).
*   Role of influence estimation in guiding PEFT module placement/training (Strategy A vs. B).

## 4. Expected Outcomes & Impact

**4.1 Expected Outcomes**

This research is expected to deliver the following concrete outcomes:

1.  **A Novel PEFT-Unlearn Algorithm:** A well-defined and empirically validated algorithm for scalable and targeted unlearning in LLMs using parameter-efficient fine-tuning.
2.  **Comprehensive Empirical Evaluation:** Extensive experimental results demonstrating the effectiveness of PEFT-Unlearn in removing different types of problematic data (privacy-sensitive, biased, specific concepts) from LLMs, benchmarked against retraining and state-of-the-art unlearning methods.
3.  **Efficiency Analysis:** Quantitative evidence showcasing the significant computational advantages (time, cost, memory) of PEFT-Unlearn over full model retraining, aiming to validate the <5% overhead target relative to retraining.
4.  **Utility Preservation Assessment:** Rigorous evaluation showing that PEFT-Unlearn maintains high performance on general knowledge and downstream tasks, effectively mitigating catastrophic forgetting.
5.  **Analysis of Formal Guarantees:** Theoretical insights and empirical validation (e.g., via MIA robustness) regarding the approximate differential unlearning properties of the proposed framework.
6.  **Open-Source Toolkit and Benchmark:** A publicly released codebase implementing the PEFT-Unlearn framework and associated evaluation scripts, along with standardized datasets and protocols forming an LLM unlearning benchmark to facilitate future research and comparison.

**4.2 Potential Impact**

The successful completion of this research will have significant impacts:

*   **Scientific Impact:** This work will significantly advance the understanding and capabilities of machine unlearning, particularly within the challenging context of large-scale models. It will shed light on the interplay between PEFT techniques and data attribution/removal, potentially influencing the design of future LLMs that are inherently more amenable to unlearning. The development of rigorous benchmarks will foster reproducible research in this critical area.
*   **Technological and Practical Impact:** PEFT-Unlearn provides a much-needed practical tool for AI developers and organizations. By offering an efficient method to remove specific data influences, it enables better lifecycle management of deployed LLMs, allowing for updates to remove harmful biases, toxic generation patterns, or comply with user data deletion requests without incurring the prohibitive cost of retraining. This lowers the barrier to adopting responsible AI practices.
*   **Societal Impact:** By making LLMs more trustworthy and controllable, this research contributes directly to mitigating the negative societal consequences associated with AI. Enabling effective privacy preservation and bias reduction helps protect individuals and marginalized groups. Facilitating compliance with regulations like GDPR strengthens data rights. Overall, this work fosters the development of AI systems that are safer, fairer, and more aligned with human values, ultimately increasing public trust and enabling the responsible adoption of powerful LLM technology in sensitive application domains.

This research directly addresses the core themes of the "Trustworthy and Reliable Large-Scale Machine Learning Models" task by providing a novel method, focusing on verifiable guarantees (privacy via unlearning), developing privacy-preserving approaches, and ultimately contributing to more robust and ethically aligned large-scale AI systems.

## 5. References

[1] Li, G., Hsu, H., Chen, C.-F., & Marculescu, R. (2023). *Fast-NTK: Parameter-Efficient Unlearning for Large-Scale Models*. arXiv:2312.14923.

[2] Chowdhury, S. B. R., Choromanski, K., Sehanobish, A., Dubey, A., & Chaturvedi, S. (2024). *Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning*. arXiv:2406.16257.

[3] Xu, J., Wu, Z., Wang, C., & Jia, X. (2024). *LMEraser: Large Model Unlearning through Adaptive Prompt Tuning*. arXiv:2404.11056.

[4] Pan, Z., Zhang, S., Zheng, Y., Li, C., Cheng, Y., & Zhao, J. (2024). *Multi-Objective Large Language Model Unlearning*. arXiv:2412.20412. (*Note: arXiv ID likely fictional as > current date*)

[5] Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., & Liu, S. (2023). *SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation*. arXiv:2310.12508.

[6] Yao, J., Chien, E., Du, M., Niu, X., Wang, T., Cheng, Z., & Yue, X. (2024). *Machine Unlearning of Pre-trained Large Language Models*. arXiv:2402.15159.

[7] Geng, J., Li, Q., Woisetschlaeger, H., Chen, Z., Wang, Y., Nakov, P., Jacobsen, H.-A., & Karray, F. (2025). *A Comprehensive Survey of Machine Unlearning Techniques for Large Language Models*. arXiv:2503.01854. (*Note: arXiv ID likely fictional*)

[8] Zhang, J.-C., Xiong, Y.-J., Xia, C.-M., Zhu, D.-H., & Qiu, X.-H. (2025). *Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace*. arXiv:2503.01419. (*Note: arXiv ID likely fictional*)

[9] Xu, H., Zhao, N., Yang, L., Zhao, S., Deng, S., Wang, M., Hooi, B., Oo, N., Chen, H., & Zhang, N. (2025). *ReLearn: Unlearning via Learning for Large Language Models*. arXiv:2502.11190. (*Note: arXiv ID likely fictional*)

[10] Lev, O., & Wilson, A. (2024). *Faster Machine Unlearning via Natural Gradient Descent*. arXiv:2407.08169.