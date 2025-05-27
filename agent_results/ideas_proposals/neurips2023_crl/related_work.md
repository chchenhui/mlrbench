1. **Title**: Causally Disentangled Generative Variational AutoEncoder (arXiv:2302.11737)
   - **Authors**: Seunghwan An, Kyungwoo Song, Jong-June Jeon
   - **Summary**: This paper introduces a supervised learning technique for Variational AutoEncoders (VAEs) that enables the learning of causally disentangled representations and the generation of causally disentangled outcomes. The approach, termed Causally Disentangled Generation (CDG), emphasizes that supervised regularization of the encoder alone is insufficient for achieving causal disentanglement. The authors explore necessary and sufficient conditions for CDG and propose a universal metric for evaluating causal disentanglement, supported by empirical results on image and tabular datasets.
   - **Year**: 2023

2. **Title**: Causal Flow-based Variational Auto-Encoder for Disentangled Causal Representation Learning (arXiv:2304.09010)
   - **Authors**: Di Fan, Yannian Kou, Chuanhou Gao
   - **Summary**: This work presents the Disentangled Causal Variational Auto-Encoder (DCVAE), a supervised VAE framework that integrates causal flows into the representation learning process. Unlike traditional methods that assume independence among factors, DCVAE accounts for interdependencies and causal relationships among factors, leading to more meaningful and interpretable disentangled representations. The model demonstrates superior performance in causal disentanglement and intervention experiments on both synthetic and real-world datasets.
   - **Year**: 2023

3. **Title**: Causal Disentangled Variational Auto-Encoder for Preference Understanding in Recommendation (arXiv:2304.07922)
   - **Authors**: Siyu Wang, Xiaocong Chen, Quan Z. Sheng, Yihong Zhang, Lina Yao
   - **Summary**: This paper introduces the Causal Disentangled Variational Auto-Encoder (CaD-VAE), designed to learn causal disentangled representations from user interaction data in recommender systems. Unlike existing methods that enforce independence among factors, CaD-VAE considers causal relationships between semantically related factors, utilizing structural causal models to generate representations that describe these relationships. The approach outperforms existing methods, offering a promising solution for understanding complex user behavior in recommendation systems.
   - **Year**: 2023

4. **Title**: Causal Representation Learning via Counterfactual Intervention (AAAI Conference on Artificial Intelligence)
   - **Authors**: Xiutian Li, Siqi Sun, Rui Feng
   - **Summary**: This study proposes a causally disentangling framework aimed at learning unbiased causal effects by introducing inductive and dataset biases into traditional causal graphs. The authors employ counterfactual interventions with reweighted loss functions to eliminate the negative effects of these biases, thereby enhancing causal representation learning. The method demonstrates state-of-the-art results on both real-world and synthetic datasets.
   - **Year**: 2024

5. **Title**: Concept-free Causal Disentanglement with Variational Graph Auto-Encoder (arXiv:2311.10638)
   - **Authors**: Jingyun Feng, Lin Zhang, Lili Yang
   - **Summary**: This paper presents an unsupervised solution for causal disentanglement in graph data, termed Concept-free Causal Variational Graph Auto-Encoder (CCVGAE). The approach incorporates a novel causal disentanglement layer into the Variational Graph Auto-Encoder, enabling the learning of concept structures directly from data without relying on predefined concepts. The method achieves significant improvements over baselines in terms of AUC, demonstrating its effectiveness in learning disentangled representations for graphs.
   - **Year**: 2023

6. **Title**: Causal Contrastive Learning for Counterfactual Regression Over Time (arXiv:2406.00535)
   - **Authors**: Mouad El Bouchattaoui, Myriam Tami, Benoit Lepetit, Paul-Henry Cournède
   - **Summary**: This work introduces a novel approach to counterfactual regression over time, emphasizing long-term predictions. The method leverages Recurrent Neural Networks (RNNs) complemented by Contrastive Predictive Coding (CPC) and Information Maximization (InfoMax) to capture long-term dependencies in the presence of time-varying confounders. The approach achieves state-of-the-art counterfactual estimation results on both synthetic and real-world data, marking the pioneering incorporation of CPC in causal inference.
   - **Year**: 2024

7. **Title**: Interventional Causal Representation Learning (arXiv:2209.11924)
   - **Authors**: Kartik Ahuja, Divyat Mahajan, Yixin Wang, Yoshua Bengio
   - **Summary**: This paper explores the role of interventional data in causal representation learning, highlighting that such data carries geometric signatures of latent factors' support. The authors prove that latent causal factors can be identified up to permutation and scaling given data from perfect interventions. The results underscore the unique power of interventional data in enabling provable identification of latent factors without assumptions about their distributions or dependency structures.
   - **Year**: 2022

8. **Title**: Disentangled Representation via Variational AutoEncoder for Continuous Treatment Effect Estimation (arXiv:2406.02310)
   - **Authors**: Ruijing Cui, Jianbin Sun, Bingyu He, Kewei Yang, Bingfeng Ge
   - **Summary**: This study proposes the Dose-Response curve estimator via Variational AutoEncoder (DRVAE), designed to disentangle covariates into instrumental factors, confounding factors, adjustment factors, and external noise factors. The approach facilitates the estimation of treatment effects under continuous treatment settings by balancing the disentangled confounding factors. Extensive results on synthetic and semi-synthetic datasets demonstrate that DRVAE outperforms current state-of-the-art methods.
   - **Year**: 2024

9. **Title**: Unpicking Data at the Seams: VAEs, Disentanglement and Independent Components (arXiv:2410.22559)
   - **Authors**: Carl Allen
   - **Summary**: This paper delves into the understanding of disentanglement in Variational Autoencoders (VAEs), showing how the choice of diagonal posterior covariance matrices promotes mutual orthogonality between columns of the decoder's Jacobian. The author demonstrates how this linear independence translates to statistical independence, completing the understanding of how the VAE's objective identifies independent components of the data.
   - **Year**: 2024

10. **Title**: ContraCLM: Contrastive Learning For Causal Language Model (arXiv:2210.01185)
    - **Authors**: Nihal Jain, Dejiao Zhang, Wasi Uddin Ahmad, Zijian Wang, Feng Nan, Xiaopeng Li, Ming Tan, Ramesh Nallapati, Baishakhi Ray, Parminder Bhatia, Xiaofei Ma, Bing Xiang
    - **Summary**: This work presents ContraCLM, a contrastive learning framework applied at both token-level and sequence-level to enhance the discrimination ability of causal language models. The approach improves the expressiveness of representations, making causal language models better suited for tasks beyond language generation, achieving significant improvements on Semantic Textual Similarity and Code-to-Code Search tasks.
    - **Year**: 2022

**Key Challenges:**

1. **Identifiability of Latent Causal Factors**: Ensuring that latent causal factors can be accurately identified and disentangled from observational data remains a significant challenge, particularly when relying solely on observational data without interventional information.

2. **Incorporating Causal Relationships**: Traditional representation learning methods often assume independence among factors, which does not hold in real-world scenarios where factors exhibit causal relationships. Developing models that effectively incorporate these causal dependencies is complex.

3. **Bias in Causal Graphs**: Existing causal representation learning methods may introduce biases within the causal graph, leading to the learning of biased causal effects in latent space. Addressing and eliminating these biases is crucial for accurate causal representation learning.

4. **Scalability and Efficiency**: Many causal representation learning methods, especially those involving complex models like transformers, face challenges related to computational efficiency and scalability, particularly when applied to large-scale datasets.

5. **Generalization and Robustness**: Ensuring that learned causal representations generalize well across different domains and are robust to domain shifts and adversarial attacks is a persistent challenge in the field. 