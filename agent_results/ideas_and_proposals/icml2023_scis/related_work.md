1. **Title**: Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation (arXiv:2307.07907)
   - **Authors**: Wenhao Ding, Laixi Shi, Yuejie Chi, Ding Zhao
   - **Summary**: This paper introduces Robust State-Confounded Markov Decision Processes (RSC-MDPs) to address spurious correlations in reinforcement learning. The authors propose an empirical algorithm that outperforms baselines in self-driving and manipulation tasks by mitigating reliance on spurious features.
   - **Year**: 2023

2. **Title**: On Counterfactual Data Augmentation Under Confounding (arXiv:2305.18183)
   - **Authors**: Abbavaram Gowtham Reddy, Saketh Bachu, Saloni Dash, Charchit Sharma, Amit Sharma, Vineeth N Balasubramanian
   - **Summary**: The authors analyze how confounding biases impact classifiers and present a causal viewpoint on counterfactual data augmentation. They propose a simple algorithm for generating counterfactual images to mitigate confounding effects, demonstrating effectiveness on MNIST variants and CelebA datasets.
   - **Year**: 2023

3. **Title**: Spuriousness-Aware Meta-Learning for Learning Robust Classifiers (arXiv:2406.10742)
   - **Authors**: Guangtao Zheng, Wenqian Ye, Aidong Zhang
   - **Summary**: This work introduces SPUME, a meta-learning framework designed to train classifiers robust to spurious correlations. By detecting and mitigating spurious correlations through meta-learning tasks, the approach achieves state-of-the-art results on multiple benchmark datasets.
   - **Year**: 2024

4. **Title**: Out of spuriousity: Improving robustness to spurious correlations without group annotations (arXiv:2407.14974)
   - **Authors**: Phuong Quynh Le, Jörg Schlötterer, Christin Seifert
   - **Summary**: The authors propose a method to extract a subnetwork from a fully trained network that does not rely on spurious correlations. Utilizing supervised contrastive loss, the approach improves worst-group performance without requiring group annotations.
   - **Year**: 2024

5. **Title**: Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation (arXiv:2405.03649)
   - **Authors**: Guangtao Zheng, Wenqian Ye, Aidong Zhang
   - **Summary**: This paper presents a framework that automatically constructs fine-grained training labels to improve classifier robustness against spurious correlations. By identifying different prediction behaviors in a novel spuriousness embedding space, the method outperforms prior approaches on five real-world datasets.
   - **Year**: 2024

6. **Title**: Trained Models Tell Us How to Make Them Robust to Spurious Correlation without Group Annotation (arXiv:2410.05345)
   - **Authors**: Mahdi Ghaznavi, Hesam Asadollahzadeh, Fahimeh Hosseini Noohdani, Soroush Vafaie Tabar, Hosein Hasani, Taha Akbari Alvanagh, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah
   - **Summary**: The authors introduce EVaLS, a method that uses losses from an ERM-trained model to construct a balanced dataset, enhancing robustness to spurious correlations without requiring group annotations. The approach achieves near-optimal worst-group accuracy.
   - **Year**: 2024

7. **Title**: Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference (arXiv:2404.13815)
   - **Authors**: Yujin Han, Difan Zou
   - **Summary**: This work proposes GIC, a method that accurately infers group labels to improve worst-group performance. By training a spurious attribute classifier based on spurious correlation properties, GIC enhances robustness without prior group information.
   - **Year**: 2024

8. **Title**: Revisiting Spurious Correlation in Domain Generalization (arXiv:2406.11517)
   - **Authors**: Bin Qin, Jiangmeng Li, Yi Li, Xuesong Wu, Yupeng Wang, Wenwen Qiang, Jianwen Cao
   - **Summary**: The authors build a structural causal model for representation learning to analyze spurious correlations. They introduce a propensity score weighted estimator to control confounding bias, demonstrating effectiveness on synthetic and real OOD datasets.
   - **Year**: 2024

9. **Title**: The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations (arXiv:2407.13957)
   - **Authors**: Tyler LaBonte, John C. Hill, Xinchen Zhang, Vidya Muthukumar, Abhishek Kumar
   - **Summary**: This paper identifies nuanced behaviors of finetuned models on worst-group accuracy. The authors propose a mixture method combining class-balancing techniques to improve robustness against spurious correlations across vision and language tasks.
   - **Year**: 2024

10. **Title**: The Multiple Dimensions of Spuriousness in Machine Learning (arXiv:2411.04696)
    - **Authors**: Samuel J. Bell, Skyler Wang
    - **Summary**: The authors explore various interpretations of spuriousness in machine learning, highlighting dimensions such as relevance, generalizability, human-likeness, and harmfulness. This work contributes to understanding and addressing spurious correlations in ML models.
    - **Year**: 2024

**Key Challenges:**

1. **Identification of Spurious Features**: Accurately detecting spurious features without prior knowledge or annotations remains a significant challenge, as models may inadvertently learn and rely on these correlations.

2. **Data Augmentation Complexity**: Generating counterfactual examples that modify only spurious features while preserving true labels is complex, especially when spurious features are intricate or unknown.

3. **Model Invariance Enforcement**: Ensuring that models become invariant to spurious features through training strategies like consistency loss is challenging, particularly without explicit group labels.

4. **Generalization to Unseen Data**: Achieving robust out-of-distribution generalization requires models to effectively disregard spurious correlations, which is difficult without comprehensive understanding of potential spurious features.

5. **Evaluation Metrics and Benchmarks**: Developing standardized metrics and benchmarks to assess robustness against spurious correlations is essential but remains an open challenge in the field. 