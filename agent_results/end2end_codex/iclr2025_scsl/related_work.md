1. **Title**: RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (arXiv:2411.04097)
   - **Authors**: Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, Curtis Langlotz
   - **Summary**: This paper introduces RaVL, a method that identifies and mitigates spurious correlations in fine-tuned vision-language models by focusing on local image features. RaVL employs region-level clustering to detect spurious correlations and utilizes a region-aware loss function to reduce reliance on these correlations during fine-tuning. The approach demonstrates significant improvements in zero-shot classification accuracy across various domains.
   - **Year**: 2024

2. **Title**: MM-SpuBench: Towards Better Understanding of Spurious Biases in Multimodal LLMs (arXiv:2406.17126)
   - **Authors**: Wenqian Ye, Guangtao Zheng, Yunsheng Ma, Xu Cao, Bolin Lai, James M. Rehg, Aidong Zhang
   - **Summary**: The authors present MM-SpuBench, a comprehensive visual question-answering benchmark designed to evaluate multimodal large language models' reliance on spurious correlations. The benchmark covers nine categories of spurious correlations across five open-source image datasets, providing insights into the persistence of spurious biases in these models and highlighting the need for new mitigation methodologies.
   - **Year**: 2024

3. **Title**: Seeing What's Not There: Spurious Correlation in Multimodal LLMs (arXiv:2503.08884)
   - **Authors**: Parsa Hosseini, Sumit Nawathe, Mazda Moayeri, Sriram Balasubramanian, Soheil Feizi
   - **Summary**: This study investigates spurious biases in multimodal large language models and introduces SpurLens, a pipeline that leverages GPT-4 and open-set object detectors to automatically identify spurious visual cues without human supervision. The findings reveal that spurious correlations lead to over-reliance on non-essential cues and object hallucinations, emphasizing the need for rigorous evaluation methods and mitigation strategies to enhance model reliability.
   - **Year**: 2025

4. **Title**: Benchmarking Robustness of Multimodal Image-Text Models under Distribution Shift (arXiv:2212.08044)
   - **Authors**: Jielin Qiu, Yi Zhu, Xingjian Shi, Florian Wenzel, Zhiqiang Tang, Ding Zhao, Bo Li, Mu Li
   - **Summary**: The authors investigate the robustness of 12 popular open-source image-text models under common perturbations across five tasks, including image-text retrieval and visual reasoning. They introduce new multimodal robustness benchmarks by applying various image and text perturbation techniques, revealing that these models are particularly sensitive to image perturbations and highlighting the need for improved robustness in real-world applications.
   - **Year**: 2022

5. **Title**: Mitigating Spurious Correlations with Causal Logit Perturbation (arXiv:2505.15246)
   - **Authors**: Xiaoling Zhou, Wei Ye, Rui Xie, Shikun Zhang
   - **Summary**: This paper introduces the Causal Logit Perturbation (CLP) framework, which trains classifiers with generated causal logit perturbations for individual samples to mitigate spurious associations between non-causal attributes and classes. The framework employs a perturbation network optimized through an online meta-learning algorithm and leverages human causal knowledge, demonstrating state-of-the-art performance across various biased learning scenarios.
   - **Year**: 2025

6. **Title**: A Survey on Benchmarks of Multimodal Large Language Models (arXiv:2408.08632)
   - **Authors**: Jian Li
   - **Summary**: This survey provides a comprehensive review of 200 benchmarks and evaluations for multimodal large language models, focusing on aspects such as perception, understanding, cognition, reasoning, and specific domains. It discusses the limitations of current evaluation methods and explores promising future directions, emphasizing the importance of robust evaluation frameworks to support the development of MLLMs.
   - **Year**: 2024

7. **Title**: Benchmarking Zero-Shot Robustness of Multimodal Foundation Models: A Pilot Study (arXiv:2403.10499)
   - **Authors**: Chenguang Wang, Ruoxi Jia, Xin Liu, Dawn Song
   - **Summary**: The authors present a comprehensive evaluation of the robustness of multimodal foundation models, such as CLIP, under various distribution shifts and adversarial attacks. Their findings indicate significant robustness drops compared to supervised models, especially under synthetic distribution shifts and adversarial attacks, highlighting the need for improved robustness in zero-shot multimodal models.
   - **Year**: 2024

8. **Title**: Evaluating MLLMs with Multimodal Multi-image Reasoning Benchmark (arXiv:2506.04280)
   - **Authors**: Ziming Cheng, Binrui Xu, Lisheng Gong, Zuhe Song, Tianshuo Zhou, Shiqi Zhong, Siyu Ren, Mingxiang Chen, Xiangchao Meng, Yuxin Zhang, Yanlin Li, Lei Ren, Wei Chen, Zhiyuan Huang, Mingjie Zhan, Xiaojie Wang, Fangxiang Feng
   - **Summary**: This paper introduces the Multimodal Multi-image Reasoning Benchmark (MMRB), designed to evaluate structured visual reasoning across multiple images. MMRB comprises 92 sub-tasks covering spatial, temporal, and semantic reasoning, with multi-solution, chain-of-thought-style annotations generated by GPT-4o and refined by human experts, providing a comprehensive evaluation framework for MLLMs.
   - **Year**: 2025

9. **Title**: The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models (arXiv:2503.03122)
   - **Authors**: Zichao Li
   - **Summary**: The authors address the issue of unimodal spurious correlations in multimodal reward models, which hinder generalization to out-of-distribution data. They introduce a shortcut-aware learning algorithm that dynamically reweights training samples, shifting the distribution toward better multimodal understanding and reducing dependence on unimodal spurious correlations, resulting in significant improvements in generalization and downstream task performance.
   - **Year**: 2025

10. **Title**: The Multiple Dimensions of Spuriousness in Machine Learning (arXiv:2411.04696)
    - **Authors**: Samuel J. Bell, Skyler Wang
    - **Summary**: This paper explores the various interpretations of spuriousness in machine learning, moving beyond the traditional causal/non-causal dichotomy. The authors conceptualize multiple dimensions of spuriousness, including relevance, generalizability, human-likeness, and harmfulness, highlighting how these interpretations influence the development and evaluation of machine learning models.
    - **Year**: 2024

**Key Challenges:**

1. **Detection of Unknown Spurious Correlations**: Identifying spurious correlations without prior knowledge remains a significant challenge, as existing methods often rely on manual annotations or predefined biases, limiting their scalability and effectiveness.

2. **Robustness Across Modalities**: Ensuring that multimodal models maintain robustness when processing diverse data types (e.g., images, text, audio) is complex, as spurious correlations can manifest differently across modalities, complicating detection and mitigation efforts.

3. **Scalable Benchmarking**: Developing comprehensive and scalable benchmarks that accurately assess model robustness against spurious correlations is challenging, particularly when considering the vast and varied nature of real-world data distributions.

4. **Mitigation Without Performance Trade-offs**: Implementing strategies to mitigate spurious correlations often leads to performance trade-offs, where improvements in robustness may come at the cost of accuracy or efficiency, posing a challenge in balancing these aspects.

5. **Generalization to Unseen Data**: Ensuring that models generalize well to unseen data without relying on spurious correlations is difficult, as models may inadvertently learn shortcuts that do not hold in different contexts or distributions, affecting their reliability and applicability. 