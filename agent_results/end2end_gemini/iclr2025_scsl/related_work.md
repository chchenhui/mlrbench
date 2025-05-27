1. **Title**: RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (arXiv:2411.04097)
   - **Authors**: Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, Curtis Langlotz
   - **Summary**: This paper introduces RaVL, a method that identifies and mitigates spurious correlations in fine-tuned vision-language models by focusing on local image features. RaVL employs region-level clustering to detect specific image features contributing to classification errors and utilizes a region-aware loss function to reduce reliance on these spurious correlations. The approach demonstrates significant improvements in zero-shot classification accuracy across various model architectures and data domains.
   - **Year**: 2024

2. **Title**: Spuriousness-Aware Meta-Learning for Learning Robust Classifiers (arXiv:2406.10742)
   - **Authors**: Guangtao Zheng, Wenqian Ye, Aidong Zhang
   - **Summary**: The authors propose SPUME, a meta-learning framework designed to train image classifiers that are robust to spurious correlations without requiring prior knowledge of these correlations. SPUME utilizes a pre-trained vision-language model to extract textual attributes from images, curates data with varying class-attribute correlations, and employs a novel metric to measure the spuriousness of these correlations. The framework demonstrates improved generalization on benchmark datasets by reducing reliance on spurious features.
   - **Year**: 2024

3. **Title**: Seeing What's Not There: Spurious Correlation in Multimodal LLMs (arXiv:2503.08884)
   - **Authors**: Parsa Hosseini, Sumit Nawathe, Mazda Moayeri, Sriram Balasubramanian, Soheil Feizi
   - **Summary**: This study investigates spurious biases in Multimodal Large Language Models (MLLMs) and introduces SpurLens, a pipeline that leverages GPT-4 and open-set object detectors to automatically identify spurious visual cues without human supervision. The findings reveal that spurious correlations lead to two major failure modes in MLLMs: over-reliance on spurious cues for object recognition and object hallucination. The paper also explores mitigation strategies such as prompt ensembling and reasoning-based prompting to enhance MLLM reliability.
   - **Year**: 2025

4. **Title**: Out of Spuriousity: Improving Robustness to Spurious Correlations Without Group Annotations (arXiv:2407.14974)
   - **Authors**: Phuong Quynh Le, Jörg Schlötterer, Christin Seifert
   - **Summary**: The authors present an approach to extract a subnetwork from a fully trained network that does not rely on spurious correlations, without requiring group annotations. By assuming that data points with the same spurious attribute cluster together in the representation space, they employ a supervised contrastive loss to force the model to unlearn spurious connections. This method improves worst-group performance and supports the hypothesis that a subnetwork exists within a dense network that utilizes only invariant features for classification tasks.
   - **Year**: 2024

5. **Title**: Explore Spurious Correlations at the Concept Level in Language Models for Text Classification (arXiv:2311.08648)
   - **Authors**: Yuhang Zhou, Paiheng Xu, Xiaoyu Liu, Bang An, Wei Ai, Furong Huang
   - **Summary**: This paper examines spurious correlations in language models at the concept level, highlighting how models may rely on non-causal associations due to imbalanced label distributions. The authors employ ChatGPT to assign concept labels to texts and assess concept bias in models during fine-tuning or in-context learning. They introduce a data rebalancing technique that incorporates ChatGPT-generated counterfactual data to balance label distribution and mitigate spurious correlations, demonstrating improved robustness over traditional token removal approaches.
   - **Year**: 2023

6. **Title**: Towards Robust Text Classification: Mitigating Spurious Correlations with Causal Learning (arXiv:2411.01045)
   - **Authors**: Yuqing Zhou, Ziwei Zhu
   - **Summary**: The authors propose the Causally Calibrated Robust Classifier (CCR), which integrates a causal feature selection method based on counterfactual reasoning and an unbiased inverse propensity weighting loss function to reduce reliance on spurious correlations in text classification tasks. By focusing on selecting causal features, CCR enhances model robustness and demonstrates state-of-the-art performance on various datasets without requiring group labels.
   - **Year**: 2024

7. **Title**: UnLearning from Experience to Avoid Spurious Correlations (arXiv:2409.02792)
   - **Authors**: Jeff Mitchell, Jesús Martínez del Rincón, Niall McLaughlin
   - **Summary**: This paper introduces UnLearning from Experience (ULE), a method that trains two classification models in parallel—a student and a teacher model—to address spurious correlations. The student model learns spurious correlations, while the teacher model is trained to solve the same classification problem while avoiding the student's mistakes. The teacher model uses the gradient of the student's output with respect to its input to unlearn spurious correlations, demonstrating effectiveness on multiple datasets.
   - **Year**: 2024

8. **Title**: Causality for Large Language Models (arXiv:2410.15319)
   - **Authors**: Anpeng Wu, Kun Kuang, Minqin Zhu, Yingrong Wang, Yujia Zheng, Kairong Han, Baohong Li, Guangyi Chen, Fei Wu, Kun Zhang
   - **Summary**: This survey explores how causality can enhance Large Language Models (LLMs) at various stages of their lifecycle, from token embedding learning to evaluation. The authors discuss the limitations of current LLMs that rely on probabilistic modeling, which often captures spurious correlations, and propose integrating causal reasoning to build more reliable and ethically aligned AI systems. They outline six promising future directions to advance LLM development and enhance their causal reasoning capabilities.
   - **Year**: 2024

9. **Title**: Data Augmentations for Improved (Large) Language Model Generalization (arXiv:2310.12803)
   - **Authors**: Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, David Blei
   - **Summary**: The authors propose using counterfactual data augmentation, guided by knowledge of the causal structure of the data, to simulate interventions on spurious features and learn more robust text classifiers. They demonstrate that this strategy is effective in prediction problems where the label is spuriously correlated with an attribute, showing improved out-of-distribution accuracy compared to baseline invariant learning algorithms.
   - **Year**: 2023

10. **Title**: Spurious Correlations in Machine Learning: A Survey (arXiv:2402.12715)
    - **Authors**: Wenqian Ye, Guangtao Zheng, Xu Cao, Yunsheng Ma, Aidong Zhang
    - **Summary**: This survey provides a comprehensive review of spurious correlations in machine learning systems, discussing their impact on model generalization and robustness. The authors present a taxonomy of current state-of-the-art methods for addressing spurious correlations, summarize existing datasets, benchmarks, and metrics, and discuss recent advancements and future challenges in the field.
    - **Year**: 2024

**Key Challenges:**

1. **Identification of Unknown Spurious Correlations**: Many existing methods require prior knowledge of spurious attributes, making it challenging to detect and mitigate unknown spurious correlations that may arise in diverse datasets.

2. **Scalability of Debiasing Techniques**: Manually annotating spurious correlations is labor-intensive and not scalable, especially when dealing with large and complex datasets.

3. **Generalization Across Domains**: Models trained to mitigate spurious correlations in one domain may not generalize well to other domains, limiting their applicability in real-world scenarios.

4. **Balancing Model Performance and Robustness**: Efforts to reduce reliance on spurious correlations can sometimes lead to a decrease in overall model performance, creating a trade-off between robustness and accuracy.

5. **Integration of Causal Reasoning**: Incorporating causal reasoning into machine learning models to distinguish between causal and spurious correlations remains a complex and underexplored area, requiring further research and development. 