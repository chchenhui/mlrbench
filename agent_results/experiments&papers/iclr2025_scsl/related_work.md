The following literature review focuses on recent advancements in mitigating spurious correlations and enhancing causal feature extraction in machine learning, particularly aligning with the principles of Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS).

**1. Related Papers:**

1. **Title**: Spurious Correlations in Machine Learning: A Survey (arXiv:2402.12715)
   - **Authors**: Wenqian Ye, Guangtao Zheng, Xu Cao, Yunsheng Ma, Aidong Zhang
   - **Summary**: This survey comprehensively reviews the issue of spurious correlations in machine learning, categorizing current methods for addressing them and summarizing existing datasets, benchmarks, and metrics to aid future research.
   - **Year**: 2024

2. **Title**: Elastic Representation: Mitigating Spurious Correlations for Group Robustness (arXiv:2502.09850)
   - **Authors**: Tao Wen, Zihan Wang, Quan Zhang, Qi Lei
   - **Summary**: The authors propose Elastic Representation (ElRep), which applies Nuclear- and Frobenius-norm penalties on the representation from the last layer of a neural network to mitigate spurious correlations and improve group robustness.
   - **Year**: 2025

3. **Title**: Spuriousness-Aware Meta-Learning for Learning Robust Classifiers (arXiv:2406.10742)
   - **Authors**: Guangtao Zheng, Wenqian Ye, Aidong Zhang
   - **Summary**: This paper introduces SPUME, a meta-learning framework that iteratively detects and mitigates spurious correlations by utilizing a pre-trained vision-language model to extract text-format attributes from images, enabling the classifier to learn invariant features without prior knowledge of spurious correlations.
   - **Year**: 2024

4. **Title**: UnLearning from Experience to Avoid Spurious Correlations (arXiv:2409.02792)
   - **Authors**: Jeff Mitchell, Jesús Martínez del Rincón, Niall McLaughlin
   - **Summary**: The authors propose UnLearning from Experience (ULE), a method that trains two models in parallel—a student and a teacher—where the teacher model learns to avoid the spurious correlations exploited by the student model, enhancing robustness against spurious features.
   - **Year**: 2024

5. **Title**: RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (arXiv:2411.04097)
   - **Authors**: Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, Curtis Langlotz
   - **Summary**: RaVL introduces a region-aware loss function that focuses on relevant image regions, mitigating spurious correlations in fine-tuned vision-language models by identifying and addressing spurious relationships at the local image feature level.
   - **Year**: 2024

6. **Title**: Unifying Causal Representation Learning with the Invariance Principle (arXiv:2409.02772)
   - **Authors**: Dingling Yao, Dario Rancati, Riccardo Cadei, Marco Fumero, Francesco Locatello
   - **Summary**: This work unifies various causal representation learning approaches by aligning representations to known data symmetries, suggesting that preserving data symmetries is crucial for discovering causal variables and improving model generalization.
   - **Year**: 2024

7. **Title**: Towards Causal Representation Learning and Deconfounding from Indefinite Data (arXiv:2305.02640)
   - **Authors**: Hang Chen, Xinyu Yang, Qing Yang
   - **Summary**: The authors propose a causal strength variational model to learn causal representations from indefinite data, addressing challenges like low sample utilization and distribution assumptions, and disentangling causal graphs into observed and latent variable relations.
   - **Year**: 2023

8. **Title**: Right for the Wrong Reason: Can Interpretable ML Techniques Detect Spurious Correlations? (arXiv:2307.12344)
   - **Authors**: Susu Sun, Lisa M. Koch, Christian F. Baumgartner
   - **Summary**: This study evaluates the effectiveness of interpretable machine learning techniques in detecting spurious correlations, finding that methods like SHAP and Attri-Net can reliably identify faulty model behavior due to spurious features.
   - **Year**: 2023

9. **Title**: On Feature Learning in the Presence of Spurious Correlations (arXiv:2210.11369)
   - **Authors**: Pavel Izmailov, Polina Kirichenko, Nate Gruver, Andrew Gordon Wilson
   - **Summary**: The authors analyze how deep classifiers rely on spurious features and demonstrate that retraining the last layer on a balanced validation dataset can isolate robust features, improving worst-group accuracy on various benchmarks.
   - **Year**: 2022

10. **Title**: Not Only the Last-Layer Features for Spurious Correlations: All Layer Deep Feature Reweighting (arXiv:2409.14637)
    - **Authors**: Humza Wajid Hameed, Geraldin Nanfack, Eugene Belilovsky
    - **Summary**: This paper extends feature reweighting to all layers of a neural network, using a feature selection strategy to mitigate spurious correlations and improve worst-group accuracy across standard benchmarks.
    - **Year**: 2024

**2. Key Challenges:**

1. **Identifying Spurious Features Without Supervision**: Many methods require prior knowledge or annotations of spurious features, which are often unavailable or impractical to obtain.

2. **Balancing Model Complexity and Robustness**: Techniques that mitigate spurious correlations may introduce additional complexity, potentially affecting model efficiency and interpretability.

3. **Generalization Across Diverse Domains**: Ensuring that methods effective in one domain or dataset generalize well to others remains a significant challenge.

4. **Trade-offs Between In-Distribution Performance and Robustness**: Enhancing robustness to spurious correlations can sometimes lead to decreased performance on in-distribution data.

5. **Scalability of Intervention-Based Methods**: Implementing synthetic interventions or reweighting strategies at scale, especially in high-dimensional data, poses computational and practical challenges. 