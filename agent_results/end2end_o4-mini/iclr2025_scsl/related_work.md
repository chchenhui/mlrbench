1. **Title**: RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (arXiv:2411.04097)
   - **Authors**: Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, Curtis Langlotz
   - **Summary**: This paper introduces RaVL, a method that identifies and mitigates spurious correlations in fine-tuned vision-language models by focusing on local image features. RaVL employs region-level clustering to detect spurious correlations and utilizes a region-aware loss function to reduce reliance on these correlations during fine-tuning. The approach demonstrates significant improvements in zero-shot classification accuracy across various model architectures and data domains.
   - **Year**: 2024

2. **Title**: Seeing What's Not There: Spurious Correlation in Multimodal LLMs (arXiv:2503.08884)
   - **Authors**: Parsa Hosseini, Sumit Nawathe, Mazda Moayeri, Sriram Balasubramanian, Soheil Feizi
   - **Summary**: This study investigates spurious biases in Multimodal Large Language Models (MLLMs) and introduces SpurLens, a pipeline that automatically identifies spurious visual cues without human supervision. The findings reveal that spurious correlations lead to over-reliance on irrelevant cues for object recognition and amplify object hallucination. The paper also explores mitigation strategies, such as prompt ensembling and reasoning-based prompting, to enhance MLLM reliability.
   - **Year**: 2025

3. **Title**: General Debiasing for Multimodal Sentiment Analysis (arXiv:2307.10511)
   - **Authors**: Teng Sun, Juntong Ni, Wenjie Wang, Liqiang Jing, Yinwei Wei, Liqiang Nie
   - **Summary**: The authors propose a debiasing framework for Multimodal Sentiment Analysis (MSA) that reduces reliance on spurious correlations by assigning smaller weights to samples with larger biases. The framework disentangles robust and biased features in each modality and employs inverse probability weighting to enhance out-of-distribution generalization. Empirical results demonstrate improved generalization on multiple unimodal and multimodal testing sets.
   - **Year**: 2023

4. **Title**: Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning (arXiv:2304.03916)
   - **Authors**: Yu Yang, Besmira Nushi, Hamid Palangi, Baharan Mirzasoleiman
   - **Summary**: This paper addresses spurious correlations in multi-modal models during fine-tuning by leveraging different modalities to detect and separate spurious attributes from the affected class. The proposed method employs a multi-modal contrastive loss function that expresses spurious relationships through language, leading to improved accuracy and activation maps focused on relevant class features.
   - **Year**: 2023

5. **Title**: Multimodal Representation Learning
   - **Authors**: Various
   - **Summary**: This article discusses graph-based methods for multimodal representation learning, which utilize graph structures to model relationships between entities across different modalities. Approaches like cross-modal graph neural networks (CMGNNs) and probabilistic graphical models (PGMs) are highlighted for their ability to learn joint representations that preserve cross-modal similarities, enabling effective integration of heterogeneous data.
   - **Year**: 2025

**Key Challenges**:

1. **Detection of Spurious Correlations**: Identifying spurious correlations in multimodal datasets is complex due to the intricate interactions between different modalities and the subtle nature of these correlations.

2. **Generalization Across Domains**: Ensuring that models generalize well across diverse domains and datasets remains a significant challenge, as spurious correlations can vary widely in different contexts.

3. **Scalability of Mitigation Strategies**: Developing scalable methods to mitigate spurious correlations without extensive computational resources is crucial for practical applications.

4. **Evaluation Benchmarks**: The lack of comprehensive and standardized benchmarks for assessing robustness to spurious correlations hinders the evaluation and comparison of different approaches.

5. **Integration of Multimodal Data**: Effectively integrating information from multiple modalities while minimizing the influence of spurious correlations requires advanced techniques and remains an ongoing research area. 