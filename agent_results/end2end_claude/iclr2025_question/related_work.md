1. **Title**: Hallucination Detection in Large Language Models with Metamorphic Relations (arXiv:2502.15844)
   - **Authors**: Borui Yang, Md Afif Al Mamun, Jie M. Zhang, Gias Uddin
   - **Summary**: This paper introduces MetaQA, a self-contained hallucination detection approach that leverages metamorphic relations and prompt mutation. Unlike existing methods, MetaQA operates without external resources and is compatible with both open-source and closed-source LLMs. It outperforms the state-of-the-art zero-resource hallucination detection method, SelfCheckGPT, across multiple datasets and LLMs.
   - **Year**: 2025

2. **Title**: Uncertainty Quantification of Large Language Models through Multi-Dimensional Responses (arXiv:2502.16820)
   - **Authors**: Tiejin Chen, Xiaoou Liu, Longchao Da, Jia Chen, Vagelis Papalexakis, Hua Wei
   - **Summary**: The authors propose a multi-dimensional uncertainty quantification framework that integrates semantic and knowledge-aware similarity analysis. By generating multiple responses and leveraging auxiliary LLMs to extract implicit knowledge, the framework constructs similarity matrices and applies tensor decomposition to derive a comprehensive uncertainty representation, capturing both semantic variations and factual consistency.
   - **Year**: 2025

3. **Title**: PFME: A Modular Approach for Fine-grained Hallucination Detection and Editing of Large Language Models (arXiv:2407.00488)
   - **Authors**: Kunquan Deng, Zeyu Huang, Chen Li, Chenghua Lin, Min Gao, Wenge Rong
   - **Summary**: This paper presents the Progressive Fine-grained Model Editor (PFME), a framework designed to detect and correct fine-grained hallucinations in LLMs. PFME consists of two collaborative modules: the Real-time Fact Retrieval Module and the Fine-grained Hallucination Detection and Editing Module. Experimental results demonstrate that PFME outperforms existing methods in fine-grained hallucination detection tasks.
   - **Year**: 2024

4. **Title**: HuDEx: Integrating Hallucination Detection and Explainability for Enhancing the Reliability of LLM responses (arXiv:2502.08109)
   - **Authors**: Sujeong Lee, Hayoung Lee, Seongsoo Heo, Wonik Choi
   - **Summary**: The authors propose HuDEx, an explanation-enhanced hallucination-detection model aimed at enhancing the reliability of LLM-generated responses by both detecting hallucinations and providing detailed explanations. HuDEx surpasses larger LLMs, such as Llama3 70B and GPT-4, in hallucination detection accuracy while maintaining reliable explanations.
   - **Year**: 2025

5. **Title**: DecoPrompt: Decoding Prompts Reduces Hallucinations when Large Language Models Meet False Premises (arXiv:2411.07457)
   - **Authors**: Nan Xu, Xuezhe Ma
   - **Summary**: This paper introduces DecoPrompt, a new prompting algorithm designed to mitigate hallucinations in LLMs when encountering false premises. DecoPrompt leverages LLMs to "decode" false-premise prompts without eliciting hallucinated outputs, effectively reducing hallucinations across different LLMs.
   - **Year**: 2024

6. **Title**: Uncertainty Quantification for In-Context Learning of Large Language Models (arXiv:2402.10189)
   - **Authors**: Chen Ling, Xujiang Zhao, Xuchao Zhang, Wei Cheng, Yanchi Liu, Yiyou Sun, Mika Oishi, Takao Osaki, Katsushi Matsuda, Jie Ji, Guangji Bai, Liang Zhao, Haifeng Chen
   - **Summary**: The authors delve into the predictive uncertainty of LLMs associated with in-context learning, highlighting that such uncertainties may stem from both the provided demonstrations (aleatoric uncertainty) and ambiguities tied to the model's configurations (epistemic uncertainty). They propose a novel formulation and estimation method to quantify both types of uncertainties.
   - **Year**: 2024

7. **Title**: CodeMirage: Hallucinations in Code Generated by Large Language Models (arXiv:2408.08333)
   - **Authors**: Vibhor Agarwal, Yulong Pei, Salwa Alamir, Xiaomo Liu
   - **Summary**: This paper introduces CodeMirage, a benchmark dataset for studying hallucinations in code generated by LLMs. The authors propose a methodology for code hallucination detection and experiment with various LLMs, discussing mitigation strategies for code hallucinations.
   - **Year**: 2024

8. **Title**: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models (arXiv:2303.08896)
   - **Authors**: Potsawee Manakul, Adian Liusie, Mark J. F. Gales
   - **Summary**: The authors propose SelfCheckGPT, a sampling-based approach that can be used to fact-check the responses of black-box models in a zero-resource fashion, i.e., without an external database. SelfCheckGPT leverages the idea that if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts.
   - **Year**: 2023

9. **Title**: Uncertainty Quantification in Large Language Models Through Convex Hull Analysis (arXiv:2406.19712)
   - **Authors**: Ferhat Ozgur Catak, Murat Kuzlu
   - **Summary**: This study proposes a novel geometric approach to uncertainty quantification using convex hull analysis. The method leverages the spatial properties of response embeddings to measure the dispersion and variability of model outputs, indicating that the uncertainty of the model for LLMs depends on the prompt complexity, the model, and the temperature setting.
   - **Year**: 2024

10. **Title**: Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation (arXiv:2305.15852)
    - **Authors**: Niels Mündler, et al.
    - **Summary**: The authors investigate self-contradictory hallucinations in LLMs, proposing a novel prompting-based framework designed to effectively detect and mitigate self-contradictions. The detector achieves high accuracy, e.g., around 80% F1 score when prompting ChatGPT, and the mitigation algorithm iteratively refines the generated text to remove contradictory information while preserving text fluency and informativeness.
    - **Year**: 2023

**Key Challenges:**

1. **Integration of Uncertainty Quantification into Reasoning Processes**: Current methods often treat uncertainty as a post-hoc calculation rather than an integral part of the reasoning process, making it challenging to flag potential hallucinations during generation.

2. **Computational Efficiency**: Incorporating uncertainty quantification and hallucination detection into LLMs can be computationally intensive, especially when requiring multiple forward passes or complex algorithms, hindering real-time applications.

3. **Explainability and Transparency**: Providing clear explanations for uncertainty and hallucinations in LLM outputs is challenging, yet essential for user trust, particularly in high-stakes domains like healthcare and law.

4. **Generalization Across Domains**: Developing methods that effectively quantify uncertainty and detect hallucinations across various domains and tasks remains a significant challenge due to the diverse nature of language and knowledge.

5. **Dependence on External Resources**: Many existing approaches rely on external databases or resources for fact-checking, which can suffer from issues such as low availability, incomplete coverage, privacy concerns, high latency, low reliability, and poor scalability. 