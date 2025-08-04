1. **Title**: Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models (arXiv:2503.05757)
   - **Authors**: Prasenjit Dey, Srujana Merugu, Sivaramakrishnan Kaveri
   - **Summary**: This paper introduces Uncertainty-Aware Fusion (UAF), an ensemble framework that combines multiple LLMs based on their accuracy and self-assessment capabilities to reduce hallucinations in factoid question answering. UAF outperforms existing hallucination mitigation methods by 8% in factual accuracy.
   - **Year**: 2025

2. **Title**: Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs (arXiv:2406.15927)
   - **Authors**: Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal
   - **Summary**: The authors propose Semantic Entropy Probes (SEPs), a method for uncertainty quantification in LLMs that approximates semantic entropy from hidden states of a single generation. SEPs effectively detect hallucinations without the computational overhead of generating multiple samples.
   - **Year**: 2024

3. **Title**: Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation (arXiv:2505.21072)
   - **Authors**: Ekaterina Fadeeva, Aleksandr Rubashevskii, Roman Vashurin, Shehzaad Dhuliawala, Artem Shelmanov, Timothy Baldwin, Preslav Nakov, Mrinmaya Sachan, Maxim Panov
   - **Summary**: This work introduces FRANQ, a method for hallucination detection in Retrieval-Augmented Generation (RAG) outputs. FRANQ applies uncertainty quantification techniques to estimate factuality based on faithfulness to retrieved context, improving detection of factual errors in RAG-generated responses.
   - **Year**: 2025

4. **Title**: Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers (arXiv:2504.19254)
   - **Authors**: Dylan Bouchard, Mohit Singh Chauhan
   - **Summary**: The authors present a framework for zero-resource hallucination detection by adapting various uncertainty quantification techniques into standardized confidence scores. They introduce a tunable ensemble approach that combines individual scores, outperforming existing hallucination detection methods.
   - **Year**: 2025

5. **Title**: keepitsimple at SemEval-2025 Task 3: LLM-Uncertainty based Approach for Multilingual Hallucination Span Detection (arXiv:2505.17485)
   - **Authors**: Not specified
   - **Summary**: This paper describes an LLM-uncertainty-based method for hallucination span detection across multiple languages. Utilizing entropy-based uncertainty measures from sample responses, the approach accurately detects hallucinated spans without additional training, performing competitively in various languages.
   - **Year**: 2025

6. **Title**: A Survey on Uncertainty Quantification of Large Language Models: Taxonomy, Open Research Challenges, and Future Directions (arXiv:2412.05563)
   - **Authors**: Not specified
   - **Summary**: This survey examines the integration of uncertainty quantification techniques in LLM-enabled applications, focusing on hallucination detection and content analysis. It discusses various methods and their effectiveness in detecting hallucinations and improving the reliability of LLM outputs.
   - **Year**: 2024

7. **Title**: Cost-Effective Hallucination Detection for LLMs (arXiv:2407.21424)
   - **Authors**: Not specified
   - **Summary**: The authors propose a cost-effective method for hallucination detection in LLMs, leveraging uncertainty quantification techniques to identify hallucinations without significant computational overhead. The approach balances performance and efficiency, making it suitable for practical applications.
   - **Year**: 2024

8. **Title**: Uncertainty Quantification for Large Language Models - ACL Anthology (ACL 2025)
   - **Authors**: Artem Shelmanov, Maxim Panov, Roman Vashurin, Artem Vazhentsev, Ekaterina Fadeeva, Timothy Baldwin
   - **Summary**: This tutorial provides a systematic introduction to uncertainty quantification for LLMs in text generation tasks. It covers theoretical foundations, state-of-the-art methods, and practical examples using the LM-Polygraph framework, equipping participants to implement UQ in their applications.
   - **Year**: 2025

9. **Title**: Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach (arXiv:2404.15993)
   - **Authors**: Not specified
   - **Summary**: The paper presents a supervised approach to uncertainty estimation and quantification in LLMs, aiming to improve the reliability of model outputs. The method involves training models to predict their uncertainty, enhancing the detection of hallucinations and other errors.
   - **Year**: 2024

10. **Title**: Improving Uncertainty Quantification in Large Language Models via Semantic Embeddings (arXiv:2410.22685)
    - **Authors**: Not specified
    - **Summary**: This work explores the use of semantic embeddings to enhance uncertainty quantification in LLMs. By incorporating semantic information, the proposed method aims to provide more accurate uncertainty estimates, aiding in the detection of hallucinations and improving model trustworthiness.
    - **Year**: 2024

**Key Challenges:**

1. **Differentiating Between Types of Uncertainty**: Accurately disentangling epistemic (model) and aleatoric (data) uncertainties remains challenging, impacting the ability to selectively mitigate harmful hallucinations while preserving creative outputs.

2. **Computational Efficiency**: Many uncertainty quantification methods require significant computational resources, making them less practical for real-time applications or deployment in resource-constrained environments.

3. **Generalization Across Tasks and Languages**: Ensuring that uncertainty quantification techniques are effective across diverse tasks and languages is difficult, as methods may not generalize well beyond their training conditions.

4. **Calibration of Uncertainty Estimates**: Developing well-calibrated uncertainty estimates that accurately reflect the true likelihood of hallucinations is essential but remains an open challenge.

5. **Balancing Creativity and Accuracy**: Implementing mechanisms that allow LLMs to maintain creative generation capabilities while minimizing factual inaccuracies without being overly conservative is a delicate balance to achieve. 