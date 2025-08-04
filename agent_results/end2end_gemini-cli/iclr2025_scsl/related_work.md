1. **Title**: Elastic Representation: Mitigating Spurious Correlations for Group Robustness (arXiv:2502.09850)
   - **Authors**: Tao Wen, Zihan Wang, Quan Zhang, Qi Lei
   - **Summary**: This paper introduces Elastic Representation (ElRep), a method that applies Nuclear- and Frobenius-norm penalties to the final layer representations of neural networks. ElRep aims to reduce reliance on spurious correlations by promoting feature diversity, thereby enhancing group robustness without significantly impacting in-distribution performance.
   - **Year**: 2025

2. **Title**: RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models (arXiv:2411.04097)
   - **Authors**: Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, Curtis Langlotz
   - **Summary**: RaVL presents a fine-grained approach to identify and mitigate spurious correlations in vision-language models. By employing region-level clustering and a novel region-aware loss function, the method focuses on relevant image regions, improving zero-shot classification accuracy and reducing reliance on spurious features.
   - **Year**: 2024

3. **Title**: Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning (arXiv:2304.03916)
   - **Authors**: Yu Yang, Besmira Nushi, Hamid Palangi, Baharan Mirzasoleiman
   - **Summary**: This work addresses spurious correlations in multi-modal models like CLIP during fine-tuning. It introduces a multi-modal contrastive loss function that leverages different modalities to detect and separate spurious attributes, enhancing model accuracy and directing attention to core features.
   - **Year**: 2023

4. **Title**: ShortcutProbe: Probing Prediction Shortcuts for Learning Robust Models (arXiv:2505.13910)
   - **Authors**: Guangtao Zheng, Wenqian Ye, Aidong Zhang
   - **Summary**: ShortcutProbe is a post hoc framework designed to identify and mitigate prediction shortcuts in models without requiring group labels. By analyzing a model's latent space, it detects non-robust prediction behaviors and retrains the model to be invariant to these shortcuts, thereby improving robustness.
   - **Year**: 2025

5. **Title**: UnLearning from Experience to Avoid Spurious Correlations (arXiv:2409.02792)
   - **Authors**: Jeff Mitchell, Jesús Martínez del Rincón, Niall McLaughlin
   - **Summary**: This paper proposes the UnLearning from Experience (ULE) approach, which utilizes parallel training of student and teacher models. The student model learns spurious correlations, while the teacher model is trained to avoid these mistakes, enhancing robustness against spurious correlations.
   - **Year**: 2024

6. **Title**: Assessing Robustness to Spurious Correlations in Post-Training Language Models (arXiv:2505.05704)
   - **Authors**: Julia Shuieh, Prasann Singhal, Apaar Shanker, John Heyer, George Pu, Samuel Denton
   - **Summary**: This study evaluates the robustness of post-training algorithms like Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and KTO against spurious correlations across various tasks. Findings indicate that no single strategy universally outperforms others, emphasizing the need for task-specific approaches.
   - **Year**: 2025

7. **Title**: Prompting is a Double-Edged Sword: Improving Worst-Group Robustness of Foundation Models
   - **Authors**: Amrith Setlur, Saurabh Garg, Virginia Smith, Sergey Levine
   - **Summary**: This paper examines the impact of prompting on the robustness of foundation models to spurious correlations. It highlights that while prompting can improve performance under certain distribution shifts, it may not effectively address spurious correlations, suggesting the need for alternative strategies.
   - **Year**: 2024

8. **Title**: Are Vision Transformers Robust to Spurious Correlations?
   - **Authors**: [Not specified]
   - **Summary**: This research investigates the susceptibility of Vision Transformers (ViTs) to spurious correlations. It finds that ViTs, especially when pre-trained on large datasets, can still be vulnerable to spurious correlations, underscoring the importance of training configurations and data diversity.
   - **Year**: 2023

9. **Title**: Trained Models Tell Us How to Make Them Robust to Spurious Correlation without Group Annotation
   - **Authors**: Mahdi Ghaznavi, Hesam Asadollahzadeh, Fahimeh Hosseini Noohdani, Soroush Vafaie Tabar, Hosein Hasani, Taha Akbari Alvanagh, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah
   - **Summary**: This work introduces a method to enhance model robustness to spurious correlations without requiring group annotations. By utilizing environment-based validation and loss-based sampling, the approach identifies and mitigates spurious correlations effectively.
   - **Year**: 2024

10. **Title**: Enhancing Model Robustness and Fairness with Causality: A Regularization Approach (arXiv:2110.00911)
    - **Authors**: Zhao Wang, Kai Shu, Aron Culotta
    - **Summary**: This paper proposes a regularization method that integrates causal knowledge during model training. By emphasizing causal features and de-emphasizing spurious ones, the approach aims to build models that are both robust and fair.
    - **Year**: 2024

**Key Challenges:**

1. **Computational Constraints**: Fine-tuning large foundation models for robustness is computationally intensive, making it impractical for many applications.

2. **Data Annotation Limitations**: Existing methods often require extensive group-annotated datasets, which are costly and may not capture all spurious correlations.

3. **Catastrophic Forgetting**: Ensuring that models retain previously learned information while mitigating spurious correlations remains a significant challenge.

4. **Scalability of Robustness Methods**: Developing parameter-efficient techniques that can be applied to various models and tasks without excessive computational overhead is essential.

5. **Identification of Spurious Features**: Automatically detecting and distinguishing spurious features from core, invariant features without prior knowledge is a complex task. 