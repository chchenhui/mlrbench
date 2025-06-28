## 1. Title

**Theoretical Foundations and Practical Guidelines for Optimal Data Epochs in Large Language Model Pretraining**

## 2. Introduction

**2.1 Background**
The advent of Large Language Models (LLMs) like GPT-4, Llama, and PaLM has revolutionized natural language processing and artificial intelligence, demonstrating remarkable capabilities in generation, reasoning, and interaction (Brown et al., 2020; Touvron et al., 2023). Training these behemoths, however, represents a monumental undertaking, consuming vast amounts of data, computational resources (often hundreds of millions of dollars), and time (Kaplan et al., 2020; Hoffmann et al., 2022). As models scale into the hundreds of billions or trillions of parameters, optimizing the pretraining process for efficiency and effectiveness becomes paramount.

A common, yet theoretically under-explored, aspect of LLM pretraining is the practice of **data recycling**, where the model is trained on the same dataset for multiple **epochs** (passes through the entire dataset). While single-pass training over massive unique datasets is sometimes advocated (e.g., Chinchilla's scaling laws suggest prioritizing data over parameters beyond a point; Hoffmann et al., 2022), practical constraints often necessitate or encourage multiple epochs. These constraints include limited availability of truly unique high-quality data, curriculum learning strategies, or simply extending training duration when initial convergence is slow or compute is available but new data is not.

Despite its prevalence, the optimal number of training epochs ($E$) remains largely determined by heuristics, empirical trial-and-error, or resource availability, rather than principled understanding. As highlighted by the workshop's focus, classical machine learning theory often falls short in explaining phenomena observed in deep learning, particularly at scale. Repeating data introduces complex dynamics: it alters the statistical properties of gradient estimates, potentially accelerates convergence initially but risks overfitting and memorization later, and its impact on the quality of learned representations for downstream tasks is unclear. Existing literature touches upon related aspects: methods for continued pretraining reuse models (Parmar et al., 2024; Qin et al., 2023), data pruning selects valuable subsets (Marion et al., 2023), and data recycling is explored in instruction-tuning (Li et al., 2023). Some empirical studies (Doe et al., 2023; Blue et al., 2024) and preliminary theoretical analyses (Johnson et al., 2023; Grey et al., 2024) have begun investigating data repetition, confirming the risks of overfitting and hinting at complex trade-offs. However, a comprehensive theoretical framework specifically analyzing the multi-epoch pretraining dynamics of LLMs, linking the number of epochs to convergence, generalization, *and* representation quality, is still missing. This gap prevents practitioners from making informed decisions about data recycling strategies, potentially leading to suboptimal resource allocation and model performance (White et al., 2024).

**2.2 Research Objectives**
This research aims to bridge this gap by developing a rigorous theoretical framework and practical guidelines for determining the optimal number of data epochs in LLM pretraining. Our primary objectives are:

1.  **Develop a Theoretical Model:** Construct a mathematical framework to analyze the impact of the number of data epochs ($E$) on the pretraining dynamics of LLMs. This model will focus on how data repetition affects gradient statistics, optimization trajectories, and the effective loss landscape.
2.  **Analyze Convergence Properties:** Investigate the relationship between $E$ and the convergence speed and stability of optimization algorithms (e.g., AdamW) commonly used for LLM pretraining. We aim to understand if and when multiple epochs offer convergence benefits, potentially linking this to phenomena like the Edge of Stability (Cohen et al., 2021).
3.  **Characterize Generalization Effects:** Analyze how $E$ influences the generalization ability of the pretrained model. This involves studying the trade-off between empirical risk minimization on the recycled dataset and population risk minimization, with a focus on identifying potential overfitting or memorization thresholds related to $E$.
4.  **Evaluate Representation Quality:** Explore the impact of $E$ on the quality and structure of the learned representations, assessed through downstream task performance and intrinsic geometric or informational properties.
5.  **Derive Practical Guidelines:** Synthesize theoretical insights and empirical findings to propose principled heuristics or guidelines for selecting an appropriate range for $E$, considering factors like dataset size ($N$), model scale ($P$), data diversity, and computational budget ($C$).
6.  **Empirical Validation:** Validate the theoretical predictions and derived guidelines through carefully designed experiments on representative LLM architectures and datasets.

**2.3 Significance**
This research holds significant potential for both theoretical advancement and practical impact within the machine learning community, directly addressing key themes of the "Mathematics of Modern Machine Learning" workshop:

*   **Reconciling Theory and Practice:** By providing a mathematical treatment of data recycling, a common heuristic in LLM training, we aim to close the gap between optimization theory and deep learning practice.
*   **Understanding Generalization:** The study will shed light on how training procedures, specifically the number of data passes, influence generalization in overparameterized models, contributing to the understanding of implicit bias and overfitting in LLMs.
*   **Informing Foundation Model Training:** Our findings will offer insights into the "Effect of Data" and potentially scaling laws, particularly how data reuse interacts with model/compute scaling.
*   **Resource Optimization:** Providing theoretically grounded guidelines for choosing $E$ can lead to substantial savings in computational resources, energy consumption, and time associated with LLM pretraining, enabling more efficient and sustainable AI development.
*   **Improved Model Development:** A better understanding of the trade-offs involved in data recycling can lead to more robust training recipes and potentially better final model performance by avoiding detrimental effects of excessive or insufficient data repetition.

## 3. Methodology

This research will employ a combination of theoretical analysis grounded in stochastic optimization, statistical learning theory, and potentially information geometry, alongside rigorous empirical validation.

**3.1 Theoretical Framework**

We consider the standard LLM pretraining setup. Let $\theta \in \mathbb{R}^P$ be the model parameters, $D = \{d_1, \dots, d_N\}$ be the pretraining dataset of size $N$, and $L(\theta; d)$ be the loss function for a single data point $d$ (e.g., cross-entropy for next-token prediction). The overall training objective is to minimize the empirical risk:
$$ L_D(\theta) = \frac{1}{N} \sum_{i=1}^N L(\theta; d_i) $$
Training proceeds using a stochastic optimization algorithm, typically AdamW (Loshchilov & Hutter, 2019), via mini-batch gradients. Let $B$ be the batch size. Training runs for $T$ total steps, spanning $E = T / (N/B)$ epochs.

**3.1.1 Modeling the Effect of Epochs on Optimization**

*   **Gradient Statistics:** We will model the stochastic gradient $\hat{g}_t(\theta_t)$ computed on a mini-batch $S_t$ at iteration $t$. Crucially, when $E > 1$, the same data point $d_i$ can appear in mini-batches at different iterations $t_1, t_2, \dots, t_E$. We will analyze how data repetition affects the properties of $\hat{g}_t$:
    *   **Bias:** The gradient remains an unbiased estimator of the empirical gradient $\nabla L_D(\theta_t)$ within an epoch (assuming uniform sampling without replacement within the epoch, or with replacement across the full dataset).
    *   **Variance:** The variance of the stochastic gradient, $\mathbb{V}[\hat{g}_t(\theta_t)]$, is key. We hypothesize that for $E>1$, the "effective" variance might behave differently than in the single-epoch setting. Initially, reusing data might reduce variance related to infrequent data points, but correlated noise across epochs could hinder progress later. We will model the variance decay as a function of $t$ and $E$. Let $g_i(\theta) = \nabla L(\theta; d_i)$. The variance is $\mathbb{V}[\hat{g}_t] = \frac{1}{|S_t|} \mathbb{V}_{d \sim S_t}[g_i(\theta_t)]$. How does the distribution of $g_i(\theta_t)$ change across epochs for the *same* $i$?
    *   **Gradient Correlation:** We will analyze the correlation between gradients involving the same data point across different epochs: $\text{Corr}(g_i(\theta_{t_1}), g_i(\theta_{t_2}))$ where $t_1$ and $t_2$ correspond to processing $d_i$ in different epochs. High correlation might suggest diminishing returns from repeated passes.
*   **Convergence Analysis:** We will adapt existing convergence analyses for SGD and Adam(W) (e.g., Kingma & Ba, 2015; Reddi et al., 2018; Defossez et al., 2022) to incorporate the effect of $E$. We aim to derive bounds on the convergence rate, potentially of the form $O(f(E, T))$, showing how the number of epochs influences the number of steps $T$ required to reach a certain loss value. We will consider assumptions relevant to deep learning, such as potential non-convexity, smoothness, and possibly the Polyak-Łojasiewicz (PL) condition locally. We will investigate if there's an optimal $E$ that maximizes convergence speed for a fixed compute budget ($T$ fixed) or minimizes compute for a target loss. This may involve analyzing how $E$ interacts with learning rate schedules and adaptive optimizers. Does increasing $E$ push the dynamics towards or away from the "Edge of Stability"?

**3.1.2 Modeling the Effect of Epochs on Generalization**

*   **Overfitting and Memorization:** Data recycling inherently increases the risk of overfitting to the training set $D$. We will model the gap between the empirical risk $L_D(\theta_T)$ and the true population risk $L_{pop}(\theta_T) = \mathbb{E}_{d \sim \text{true data dist.}}[L(\theta_T; d)]$. We will explore how generalization measures (e.g., model norm $\|\theta\|$, sharpness $\max_{\|v\|=1} L_D(\theta+v) - L_D(\theta)$, PAC-Bayes bounds) evolve as a function of $E$. We hypothesize that there exists a critical number of epochs $E^*$ beyond which generalization performance starts to degrade significantly due to fitting noise or memorizing specific examples. This analysis will connect to the literature on implicit bias, investigating whether multiple epochs alter the characteristics of the solution found by the optimizer (e.g., moving from flatter to sharper minima).
*   **Information Theoretic Perspective:** Drawing inspiration from information bottleneck theory (Tishby & Zaslavsky, 2015) and information geometry (Amari, 1998; Grey et al., 2024), we may model the training process as finding an efficient representation of the data. How does recycling data affect the mutual information $I(\theta; D)$ between the model parameters and the training data? Does excessive recycling lead to an undesirable increase in this mutual information, indicative of overfitting? We could analyze the trajectory of the Fisher Information matrix or related geometric quantities as a function of $E$.

**3.1.3 Modeling the Effect of Epochs on Representation Quality**

The ultimate goal of pretraining is often to learn representations useful for downstream tasks. We will model how properties of the learned representations $h(\theta; d)$ (activations at certain layers) change with $E$.
*   **Linear Probing:** We hypothesize that initial epochs rapidly improve representation quality for downstream tasks, but later epochs might yield diminishing returns or even degrade performance due to overfitting specific pretraining artefacts. We can model the performance of a simple linear classifier trained on frozen representations $h(\theta_E; d)$ as a function of $E$.
*   **Representation Similarity:** Using metrics like Centered Kernel Alignment (CKA) (Kornblith et al., 2019), we can analyze how representations evolve across layers and across epochs. Does increasing $E$ lead to representational collapse or stabilization? How does the similarity between representations learned at epoch $e$ and $e+1$ change?

**3.2 Data Collection**
We will use standard, publicly available pretraining datasets to ensure reproducibility.
*   **Primary Corpus:** A subset of a large corpus like The Pile (Gao et al., 2020) or C4 (Raffel et al., 2020). We will use subsets of varying sizes (e.g., 1B, 10B, 50B tokens) to study the interaction between $N$ and $E$.
*   **Downstream Task Datasets:** Standard benchmarks like GLUE (Wang et al., 2018), SuperGLUE (Wang et al., 2019), SQuAD (Rajpurkar et al., 2016), and potentially summarization or generation datasets will be used for evaluating fine-tuned model performance.
*   **Probing Datasets:** Datasets designed for linguistic probing (e.g., Conneau et al., 2018) might be used to assess intrinsic representation quality.

**3.3 Experimental Design**

Our experiments will be designed to validate theoretical predictions and provide practical insights.

*   **Model Architectures:** We will use standard Transformer-based architectures (Vaswani et al., 2017), likely variants of GPT-2/3 or Llama, at different scales (e.g., 100M, 500M, 1B+ parameters, subject to computational resources) to observe if effects vary with model size $P$.
*   **Controlled Variables:**
    *   **Number of Epochs (E):** This is the primary variable. We will systematically vary $E$ (e.g., 1, 2, 4, 8, 16, ...) while controlling other factors.
    *   **Dataset Size (N):** Experiments will be repeated across different dataset sizes.
    *   **Compute Budget (T):** We will conduct two types of experiments:
        1.  *Fixed Compute:* Keep total training steps $T$ (or total tokens processed $T \times B$) constant. Increasing $E$ means decreasing the amount of unique data used (by sampling a subset of size $N/E$ from the full dataset and repeating it $E$ times). This isolates the effect of repetition vs. data freshness for a fixed budget.
        2.  *Variable Compute:* Use a fixed dataset $D$ of size $N$ and increase the total steps $T$ proportionally to $E$. This measures the absolute benefit/detriment of additional epochs on the same data.
    *   **Optimizer & Hyperparameters:** We will primarily use AdamW with standard hyperparameters (learning rate schedule with warmup and decay, weight decay) based on best practices for the chosen model scale, keeping them fixed across different values of $E$ where possible, while potentially exploring interactions (e.g., does the optimal LR schedule change with $E$?).
*   **Measurements & Evaluation Metrics:**
    *   **Training Dynamics:** Training loss, validation loss (on a held-out split of the pretraining data), gradient norm, update norm, gradient variance (estimated from mini-batches). Steps/time to reach specific loss milestones.
    *   **Generalization (Pretraining Domain):** Perplexity (PPL) on held-out pretraining data.
    *   **Generalization (Downstream Tasks):** Fine-tune pretrained checkpoints ($\theta_E$ from different $E$) on a suite of downstream tasks. Report standard metrics (Accuracy, F1 score, ROUGE, etc.). Both full fine-tuning and few-shot performance will be assessed.
    *   **Memorization:** Measure the rate at which models reproduce verbatim sequences from the training set (Carlini et al., 2022).
    *   **Representation Quality:** Performance on linear probing tasks. CKA similarity between layer representations across different epochs and compared to a 'golden' model trained on unique data (if feasible).
*   **Baselines:**
    *   $E=1$ training (single pass over the data).
    *   Training on larger unique datasets for equivalent compute (if possible within the fixed compute setting).
    *   Existing heuristics for setting epoch counts (if identifiable from literature or practice).
*   **Statistical Analysis:** Results will be reported with means and standard deviations over multiple runs with different random seeds. Statistical significance tests (e.g., t-tests, ANOVA) will be used to compare performance across different values of $E$.

**3.4 Potential Challenges and Mitigation**
*   **Computational Cost:** Training LLMs is expensive. We will mitigate this by using moderate model sizes and dataset subsets for extensive ablation studies, potentially scaling up key experiments. We will leverage efficient training libraries (e.g., DeepSpeed, Megatron-LM) and available compute clusters.
*   **Theoretical Complexity:** Deriving tight, non-vacuous theoretical bounds for deep learning is notoriously difficult. We may need to rely on simplified assumptions (e.g., local properties like smoothness or PL, specific noise models) or approximations. Our goal is a framework that captures the *trends* and *trade-offs* related to $E$, even if exact constants are elusive.
*   **Confounding Factors:** Many hyperparameters interact. We will fix most hyperparameters based on best practices and systematically vary $E$, acknowledging that co-tuning other parameters (like LR schedule length relative to $E$) might yield different results, which could be explored in follow-up work.

## 4. Expected Outcomes & Impact

**4.1 Expected Outcomes**

1.  **A Comprehensive Theoretical Framework:** We expect to develop the first comprehensive theoretical model that explicitly incorporates the number of data epochs ($E$) into the analysis of LLM pretraining, covering optimization dynamics, generalization, and representation learning.
2.  **Quantitative Relationships:** The framework is expected to yield mathematical relationships (bounds, scaling laws, or qualitative descriptions) characterizing how $E$ influences:
    *   Convergence rate (e.g., saturation or degradation point).
    *   Generalization gap (e.g., threshold for overfitting onset).
    *   Downstream task performance (e.g., optimal $E$ for transferability).
    *   Representation geometry/information content.
3.  **Identification of Optimal Regimes:** We anticipate identifying different regimes based on $N, P, C$:
    *   Regime 1 (Data-rich / Compute-limited): Single epoch ($E=1$) or very few epochs might be optimal.
    *   Regime 2 (Data-limited / Compute-rich): Multiple epochs ($E>1$) might be beneficial up to a certain point $E^*$.
    *   Regime 3 (Balanced): An intermediate $E$ provides the best trade-off.
4.  **Evidence-Based Guidelines:** Based on theoretical insights and empirical validation, we expect to formulate practical, data-driven guidelines or heuristics for choosing $E$. These guidelines could potentially take the form of a recommended range for $E$ given $N, P$, and desired properties (e.g., optimize for speed vs. peak performance).
5.  **Validated Empirical Findings:** We expect robust empirical results across various model/data scales confirming or refining the theoretical predictions, including clear demonstrations of the benefits and drawbacks (e.g., overfitting curves) of data recycling.
6.  **Insights into Training Dynamics:** The research should provide deeper insights into *why* data recycling affects LLMs the way it does, potentially linking it to gradient noise reduction, feature learning saturation, or implicit regularization effects.

**4.2 Impact**

*   **Scientific Impact:** This work will contribute significantly to the theoretical understanding of large-scale deep learning, particularly addressing the interplay between optimization algorithms, data utilization strategies, generalization, and representation learning in the context of foundation models. It directly aligns with the workshop's goals of advancing the mathematical foundations of modern ML and bridging the theory-practice gap. The findings could stimulate further research into optimal data scheduling and resource allocation in deep learning.
*   **Practical Impact:** The research offers substantial practical benefits for the AI community. By providing principled guidelines for selecting the number of pretraining epochs, we can help practitioners:
    *   **Optimize Resource Usage:** Reduce computational costs, energy consumption, and training time by avoiding unnecessary or detrimental data recycling. This is crucial for making LLM development more sustainable and accessible.
    *   **Improve Model Quality:** Avoid performance degradation caused by excessive overfitting due to too many epochs, potentially leading to better generalizing and more reliable models.
    *   **Streamline the Pretraining Workflow:** Replace ad-hoc decisions about epoch counts with more informed strategies, reducing the amount of expensive trial-and-error.
*   **Broader Impact:** By contributing to more efficient AI development methods, this research supports the broader goals of democratizing AI technology and promoting responsible innovation. Understanding the fundamental trade-offs in data usage can also inform discussions on data governance and efficiency in the era of large models.

In conclusion, this research promises to deliver valuable theoretical insights and practical tools for optimizing a critical aspect of LLM pretraining, directly contributing to the mathematical understanding and efficient practice of modern machine learning.

## 5. References

(Based on the provided literature review)

1.  Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*, 10(2), 251-276.
2.  Blue, L., & Red, M. (2024). Overfitting Risks in Repeated Data Exposure During LLM Pretraining. *arXiv preprint arXiv:2405.45678*.
3.  Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.
4.  Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., ... & Raffel, C. (2022). Quantifying Memorization Across Neural Language Models. *arXiv preprint arXiv:2202.07646*.
5.  Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. *International Conference on Learning Representations*.
6.  Conneau, A., Kruszewski, G., Lample, G., Barrault, L., & Baroni, M. (2018). What you can cram into a single vector: Probing sentence embeddings for linguistic properties. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*.
7.  Defossez, A., Bottou, L., Bach, F., & Usunier, N. (2022). Convergence analysis of Adam-like algorithms without vanishing learning rates or bounded gradients. *arXiv preprint arXiv:2203.09089*.
8.  Doe, J., & Smith, J. (2023). Understanding the Impact of Data Repetition on Large Language Models. *arXiv preprint arXiv:2306.12345*. (*Note: Fictional citation from review*)
9.  Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., ... & Leahy, R. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. *arXiv preprint arXiv:2101.00027*.
10. Green, M., & Brown, S. (2024). Data Recycling Strategies for Efficient LLM Training. *arXiv preprint arXiv:2403.34567*. (*Note: Fictional citation from review*)
11. Grey, S., & White, T. (2024). Information Geometry Approaches to Data Recycling in LLMs. *arXiv preprint arXiv:2406.56789*. (*Note: Fictional citation from review*)
12. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., ... & Sifre, L. (2022). Training Compute-Optimal Large Language Models. *arXiv preprint arXiv:2203.15556*.
13. Johnson, A., & Lee, B. (2023). Theoretical Insights into Data Recycling in Neural Network Training. *arXiv preprint arXiv:2311.67890*. (*Note: Fictional citation from review*)
14. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling Laws for Neural Language Models. *arXiv preprint arXiv:2001.08361*.
15. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*.
16. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *International Conference on Machine Learning*.
17. Li, M., Chen, L., Chen, J., He, S., Huang, H., Gu, J., & Zhou, T. (2023). Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning. *arXiv preprint arXiv:2310.11716*.
18. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations*.
19. Marion, M., Üstün, A., Pozzobon, L., Wang, A., Fadaee, M., & Hooker, S. (2023). When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale. *arXiv preprint arXiv:2309.04564*.
20. Parmar, J., Satheesh, S., Patwary, M., Shoeybi, M., & Catanzaro, B. (2024). Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models. *arXiv preprint arXiv:2407.07263*.
21. Qin, Y., Qian, C., Han, X., Lin, Y., Wang, H., Xie, R., ... & Zhou, J. (2023). Recyclable Tuning for Continual Pre-training. *arXiv preprint arXiv:2305.08702*.
22. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research*, 21(140), 1-67.
23. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*.
24. Reddi, S. J., Kale, S., & Kumar, S. (2018). On the Convergence of Adam and Beyond. *International Conference on Learning Representations*.
25. Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle. *IEEE Information Theory Workshop (ITW)*.
26. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv preprint arXiv:2307.09288*.
27. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*, 30.
28. Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. *Proceedings of the 2018 EMNLP Workshop BlackboxNLP*.
29. Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., ... & Bowman, S. R. (2019). SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. *Advances in Neural Information Processing Systems*, 32.
30. White, E., & Black, D. (2024). Balancing Data Efficiency and Model Performance in LLM Pretraining. *arXiv preprint arXiv:2401.23456*. (*Note: Fictional citation from review*)