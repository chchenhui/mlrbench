1. **Title**: Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective (arXiv:2411.01045)
   - **Authors**: Yuqing Zhou, Ziwei Zhu
   - **Summary**: This paper introduces the Causally Calibrated Robust Classifier (CCR) to address spurious correlations in text classification. CCR employs counterfactual reasoning for causal feature selection and an inverse propensity weighting loss function to reduce reliance on spurious features, enhancing model robustness without requiring group labels.
   - **Year**: 2024

2. **Title**: Seeing What's Not There: Spurious Correlation in Multimodal LLMs (arXiv:2503.08884)
   - **Authors**: Parsa Hosseini, Sumit Nawathe, Mazda Moayeri, Sriram Balasubramanian, Soheil Feizi
   - **Summary**: The authors investigate spurious biases in Multimodal Large Language Models (MLLMs) using SpurLens, a pipeline that identifies spurious visual cues without human supervision. They reveal that spurious correlations lead to over-reliance on such cues and object hallucination, calling for improved evaluation and mitigation strategies in MLLMs.
   - **Year**: 2025

3. **Title**: Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning (arXiv:2304.03916)
   - **Authors**: Yu Yang, Besmira Nushi, Hamid Palangi, Baharan Mirzasoleiman
   - **Summary**: This work presents a method to address spurious correlations in multi-modal models like CLIP during fine-tuning. By leveraging different modalities to detect and separate spurious attributes using a multi-modal contrastive loss function, the approach improves model accuracy and directs activation maps toward actual classes rather than spurious attributes.
   - **Year**: 2023

4. **Title**: mPLUG: Effective and Efficient Vision-Language Learning by Cross-modal Skip-connections (arXiv:2205.12005)
   - **Authors**: Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, Ji Zhang, Songfang Huang, Fei Huang, Jingren Zhou, Luo Si
   - **Summary**: mPLUG introduces a vision-language foundation model with cross-modal skip-connections to enhance computational efficiency and information symmetry. Pre-trained on large-scale image-text pairs, it achieves state-of-the-art results in various vision-language tasks, demonstrating strong zero-shot transferability to multiple video-language tasks.
   - **Year**: 2022

5. **Title**: Multimodal Representation Learning
   - **Authors**: Wikipedia contributors
   - **Summary**: This article discusses various methods for multimodal representation learning, including graph-based approaches and diffusion maps. It highlights techniques like cross-modal graph neural networks and multi-view diffusion maps that model relationships between entities across different modalities to achieve coherent low-dimensional representations.
   - **Year**: 2025

**Key Challenges**:

1. **Identifying Spurious Correlations Across Modalities**: Detecting and distinguishing spurious correlations that span multiple modalities without explicit annotations remains a significant challenge.

2. **Developing Causal Feature Selection Methods**: Creating methods that can effectively identify and select causal features in multi-modal data to mitigate reliance on spurious correlations is complex.

3. **Ensuring Model Robustness to Out-of-Distribution Data**: Enhancing the generalization capabilities of multi-modal models to perform reliably on out-of-distribution data where spurious correlations may not hold is crucial.

4. **Balancing Computational Efficiency with Model Complexity**: Designing models that effectively address spurious correlations while maintaining computational efficiency and scalability is a persistent challenge.

5. **Developing Comprehensive Evaluation Benchmarks**: Establishing rigorous and comprehensive benchmarks to assess the robustness of multi-modal models against spurious correlations is essential for progress in this field. 