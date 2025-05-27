1. **Title**: Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models (arXiv:2503.05757)
   - **Authors**: Prasenjit Dey, Srujana Merugu, Sivaramakrishnan Kaveri
   - **Summary**: This paper introduces Uncertainty-Aware Fusion (UAF), an ensemble framework that combines multiple LLMs based on their accuracy and self-assessment capabilities to reduce hallucinations in factoid question answering. UAF outperforms existing hallucination mitigation methods by 8% in factual accuracy.
   - **Year**: 2025

2. **Title**: Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs (arXiv:2406.15927)
   - **Authors**: Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal
   - **Summary**: The authors propose Semantic Entropy Probes (SEPs), a method for uncertainty quantification in LLMs that approximates semantic entropy from hidden states of a single generation. SEPs are simple to train and effectively detect hallucinations without the need for multiple model generations.
   - **Year**: 2024

3. **Title**: Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers (arXiv:2504.19254)
   - **Authors**: Dylan Bouchard, Mohit Singh Chauhan
   - **Summary**: This work presents a versatile framework for zero-resource hallucination detection by adapting various uncertainty quantification techniques into standardized confidence scores. The authors introduce a tunable ensemble approach that combines individual confidence scores, demonstrating improved performance over existing methods.
   - **Year**: 2025

4. **Title**: Fact-Checking the Output of Large Language Models via Token-Level Uncertainty Quantification (arXiv:2403.04696)
   - **Authors**: Ekaterina Fadeeva, Aleksandr Rubashevskii, Artem Shelmanov, Sergey Petrakov, Haonan Li, Hamdy Mubarak, Evgenii Tsymbalov, Gleb Kuzmin, Alexander Panchenko, Timothy Baldwin, Preslav Nakov, Maxim Panov
   - **Summary**: The authors propose a fact-checking and hallucination detection pipeline based on token-level uncertainty quantification. Their method, Claim Conditioned Probability (CCP), measures the uncertainty of specific claims expressed by the model, showing strong improvements in detecting hallucinations across multiple LLMs and languages.
   - **Year**: 2024

5. **Title**: Hallucination Detection in Large Language Models with Metamorphic Relations (arXiv:2502.15844)
   - **Authors**: Borui Yang, Md Afif Al Mamun, Jie M. Zhang, Gias Uddin
   - **Summary**: This paper introduces MetaQA, a hallucination detection approach that leverages metamorphic relations and prompt mutation without relying on external resources. MetaQA operates effectively on both open-source and closed-source LLMs, outperforming existing methods in precision, recall, and F1 score.
   - **Year**: 2025

6. **Title**: Verify when Uncertain: Beyond Self-Consistency in Black Box Hallucination Detection (arXiv:2502.15845)
   - **Authors**: Yihao Xue, Kristjan Greenewald, Youssef Mroueh, Baharan Mirzasoleiman
   - **Summary**: The authors explore cross-model consistency checking between a target model and an additional verifier LLM to improve hallucination detection. They propose a two-stage detection algorithm that dynamically switches between self-consistency and cross-consistency based on an uncertainty interval, enhancing detection performance while reducing computational cost.
   - **Year**: 2025

7. **Title**: Iterative Deepening Sampling for Large Language Models (arXiv:2502.05449)
   - **Authors**: Weizhe Chen, Sven Koenig, Bistra Dilkina
   - **Summary**: This work proposes an iterative deepening sampling algorithm framework designed to enhance self-correction and generate higher-quality samples in LLMs. The method achieves higher success rates on challenging reasoning tasks and provides detailed ablation studies across diverse settings.
   - **Year**: 2025

8. **Title**: To Know or Not To Know? Analyzing Self-Consistency of Large Language Models under Ambiguity (arXiv:2407.17125)
   - **Authors**: Anastasiia Sedova, Robert Litschko, Diego Frassinelli, Benjamin Roth, Barbara Plank
   - **Summary**: The authors analyze the proficiency and consistency of state-of-the-art LLMs in applying factual knowledge when prompted with ambiguous entities. Their experiments reveal that LLMs struggle with choosing the correct entity reading, highlighting the need to address entity ambiguity for more trustworthy models.
   - **Year**: 2024

9. **Title**: Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation (ACL Anthology 2023.emnlp-main.949)
   - **Authors**: Xiaohua Wang, Yuliang Yan, Longtao Huang, Xiaoqing Zheng, Xuanjing Huang
   - **Summary**: This paper introduces a framework that leverages statistical decision theory and Bayesian sequential analysis to optimize the trade-off between costs and benefits during the hallucination detection process. The approach reduces response times and surpasses existing methods in both efficiency and precision.
   - **Year**: 2023

10. **Title**: Uncertainty Quantification in Large Language Models through Convex Hull Analysis (Discover Artificial Intelligence, Volume 4, Article 90)
    - **Authors**: Not specified
    - **Summary**: This study proposes a geometric approach to uncertainty quantification using convex hull analysis. By leveraging the spatial properties of response embeddings, the method measures the dispersion and variability of model outputs, providing a clear and interpretable metric for developing more reliable LLMs.
    - **Year**: 2024

**Key Challenges:**

1. **Computational Efficiency**: Many existing uncertainty quantification and hallucination detection methods, such as ensemble approaches and Bayesian approximations, are computationally intensive, making them impractical for large-scale deployment.

2. **Detection Accuracy**: Achieving high precision and recall in detecting hallucinations remains challenging, as models may still produce plausible-sounding but incorrect outputs that are difficult to identify.

3. **Scalability**: Methods that require multiple model generations or extensive external resources may not scale effectively, limiting their applicability in real-world scenarios.

4. **Interpretability**: Providing interpretable uncertainty measures and hallucination indicators is crucial for user trust, yet many current methods lack transparency in their uncertainty assessments.

5. **Balancing Creativity and Accuracy**: Mitigating hallucinations without stifling the generative creativity of LLMs is a delicate balance that current methods struggle to maintain. 