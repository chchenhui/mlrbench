1. **Title**: SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models (arXiv:2403.02509)
   - **Authors**: Xiang Gao, Jiaxin Zhang, Lalla Mouatadid, Kamalika Das
   - **Summary**: This paper introduces SPUQ, a method that generates perturbations for LLM inputs and samples outputs for each perturbation to quantify both aleatoric and epistemic uncertainties. The approach improves model uncertainty calibration, reducing Expected Calibration Error (ECE) by 50% on average.
   - **Year**: 2024

2. **Title**: Inv-Entropy: A Fully Probabilistic Framework for Uncertainty Quantification in Language Models (arXiv:2506.09684)
   - **Authors**: Haoyi Song, Ruihan Ji, Naichen Shi, Fan Lai, Raed Al Kontar
   - **Summary**: The authors propose a probabilistic framework that models input-output pairs as two Markov chains with transition probabilities defined by semantic similarity. They introduce Inv-Entropy, an uncertainty measure evaluating the diversity of the input space conditioned on a given output through systematic perturbations.
   - **Year**: 2025

3. **Title**: LUQ: Long-text Uncertainty Quantification for LLMs (arXiv:2403.20279)
   - **Authors**: Caiqi Zhang, Fangyu Liu, Marco Basaldella, Nigel Collier
   - **Summary**: This study presents LUQ, a sampling-based UQ approach specifically designed for long-text generation. LUQ outperforms existing methods in correlating with the model's factuality scores and introduces LUQ-Ensemble, which selects responses with the lowest uncertainty from multiple models to improve factual accuracy.
   - **Year**: 2024

4. **Title**: Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs (arXiv:2505.20045)
   - **Authors**: Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Mrinmaya Sachan, Preslav Nakov, Artem Shelmanov
   - **Summary**: The authors propose RAUQ, an unsupervised approach leveraging intrinsic attention patterns in transformers to detect hallucinations efficiently. By analyzing attention weights, RAUQ identifies "uncertainty-aware" heads and computes sequence-level uncertainty scores in a single forward pass, outperforming state-of-the-art UQ methods with minimal computational overhead.
   - **Year**: 2025

5. **Title**: Improving Uncertainty Quantification in Large Language Models via Semantic Embeddings (arXiv:2410.22685)
   - **Authors**: Yashvir S. Grewal, Edwin V. Bonilla, Thang D. Bui
   - **Summary**: This paper introduces a method that leverages semantic embeddings to achieve smoother and more robust estimation of semantic uncertainty in LLMs. By capturing semantic similarities without depending on sequence likelihoods, the approach reduces biases introduced by irrelevant words and offers an amortized version for efficient uncertainty estimation.
   - **Year**: 2024

6. **Title**: Benchmarking Large Language Model Uncertainty for Prompt Optimization (arXiv:2409.10044)
   - **Authors**: Pei-Fu Guo, Yun-Da Tsai, Shou-De Lin
   - **Summary**: The authors introduce a benchmark dataset to evaluate uncertainty metrics, focusing on Answer, Correctness, Aleatoric, and Epistemic Uncertainty. They analyze models like GPT-3.5-Turbo and Meta-Llama-3.1-8B-Instruct, highlighting the need for improved metrics that are optimization-objective-aware to better guide prompt optimization.
   - **Year**: 2024

7. **Title**: MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty (arXiv:2408.06816)
   - **Authors**: Yongjin Yang, Haneul Yoo, Hwaran Lee
   - **Summary**: This paper presents MAQA, a dataset designed to evaluate uncertainty quantification methods under the presence of data uncertainty. The study assesses five UQ methods across various tasks, finding that entropy and consistency-based methods estimate model uncertainty well even under data uncertainty, while others struggle depending on the tasks.
   - **Year**: 2024

8. **Title**: Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach (arXiv:2404.15993)
   - **Authors**: Linyu Liu, Yu Pan, Xiaocheng Li, Guanting Chen
   - **Summary**: The authors propose a supervised approach that utilizes labeled datasets to estimate the uncertainty of LLM responses. By leveraging hidden activations, the method enhances uncertainty estimation across various tasks and demonstrates robust transferability in out-of-distribution settings.
   - **Year**: 2024

9. **Title**: Fact-Checking the Output of Large Language Models via Token-Level Uncertainty Quantification (arXiv:2403.04696)
   - **Authors**: Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem Shelmanov, Sergey Petrakov, Haonan Li, Hamdy Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, Alexander Panchenko, Timothy Baldwin, Preslav Nakov, Maxim Panov
   - **Summary**: This study introduces a fact-checking and hallucination detection pipeline based on token-level uncertainty quantification. The proposed Claim Conditioned Probability (CCP) method measures the uncertainty of specific claims expressed by the model, demonstrating strong improvements in detecting unreliable predictions across multiple LLMs and languages.
   - **Year**: 2024

10. **Title**: Uncertainty Quantification for Clinical Outcome Predictions with (Large) Language Models (arXiv:2411.03497)
    - **Authors**: Zizhang Chen, Peizhao Li, Xiaomeng Dong, Pengyu Hong
    - **Summary**: The authors explore uncertainty quantification in LLMs for clinical prediction tasks using electronic health records. They propose ensemble methods and multi-task prediction prompts to reduce uncertainty, validating their framework with longitudinal clinical data from over 6,000 patients across ten prediction tasks.
    - **Year**: 2024

**Key Challenges:**

1. **Computational Efficiency**: Many existing uncertainty quantification methods, such as full Monte Carlo dropout or ensemble approaches, are computationally intensive, making them impractical for real-time applications.

2. **Scalability**: Adapting uncertainty quantification techniques to handle the scale and complexity of large language models without compromising performance remains a significant challenge.

3. **Calibration Accuracy**: Ensuring that the confidence scores produced by uncertainty quantification methods accurately reflect the true likelihood of correctness is crucial, yet difficult to achieve consistently.

4. **Detection of Hallucinations**: Effectively identifying and mitigating instances where models generate plausible but incorrect information (hallucinations) is a persistent issue in deploying LLMs in critical domains.

5. **Generalization Across Tasks**: Developing uncertainty quantification methods that generalize well across diverse tasks and datasets without extensive retraining or fine-tuning is an ongoing challenge. 