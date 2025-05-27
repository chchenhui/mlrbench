**Related Papers:**

1. **Title**: SEER: Self-Explainability Enhancement of Large Language Models' Representations (arXiv:2502.05242)
   - **Authors**: Guanxu Chen, Dongrui Liu, Tao Luo, Jing Shao
   - **Summary**: This paper introduces SEER, a method that enhances the self-explainability of large language models by aggregating similar concepts and disentangling different ones within the representation space. SEER provides faithful explanations alongside model outputs and demonstrates improvements in tasks related to trustworthiness, such as safety risk classification and detoxification.
   - **Year**: 2025

2. **Title**: Compositional Concept-Based Neuron-Level Interpretability for Deep Reinforcement Learning (arXiv:2502.00684)
   - **Authors**: Zeyu Jiang, Hai Huang, Xingquan Zuo
   - **Summary**: The authors propose a concept-based interpretability method that offers fine-grained explanations at the neuron level for deep reinforcement learning models. By analyzing neuron activations in relation to compositional concepts, the method provides insights into the decision-making processes of policy and value networks.
   - **Year**: 2025

3. **Title**: A Close Look at Decomposition-based XAI-Methods for Transformer Language Models (arXiv:2502.15886)
   - **Authors**: Leila Arras, Bruno Puri, Patrick Kahardipraja, Sebastian Lapuschkin, Wojciech Samek
   - **Summary**: This study evaluates decomposition-based explainability methods, such as ALTI-Logit and LRP, applied to transformer language models. The authors conduct quantitative and qualitative analyses to assess the effectiveness of these methods in providing insights into model decisions.
   - **Year**: 2025

4. **Title**: TextGenSHAP: Scalable Post-hoc Explanations in Text Generation with Long Documents (arXiv:2312.01279)
   - **Authors**: James Enouen, Hootan Nakhost, Sayna Ebrahimi, Sercan O Arik, Yan Liu, Tomas Pfister
   - **Summary**: TextGenSHAP introduces an efficient post-hoc explanation method tailored for large language models handling long documents. By incorporating language model-specific techniques, it significantly reduces computation time for generating token-level and document-level explanations, enhancing understanding of model outputs.
   - **Year**: 2023

5. **Title**: Explainability for Large Language Models: A Survey (arXiv:2309.01029)
   - **Authors**: Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Mengnan Du
   - **Summary**: This survey provides a comprehensive overview of explainability techniques for transformer-based language models. It categorizes methods based on training paradigms and discusses goals, dominant approaches, evaluation metrics, and challenges in explaining large language models.
   - **Year**: 2023

6. **Title**: Enhancing the Interpretability of SHAP Values Using Large Language Models (arXiv:2409.00079)
   - **Authors**: Xianlong Zeng
   - **Summary**: The paper explores the use of large language models to translate SHAP value outputs into plain language explanations, making model interpretability more accessible to non-technical users. The approach aims to maintain the accuracy of SHAP values while improving clarity and usability.
   - **Year**: 2024

7. **Title**: Large Language Models as Evaluators for Recommendation Explanations (arXiv:2406.03248)
   - **Authors**: Xiaoyu Zhang, Yishan Li, Jiayin Wang, Bowen Sun, Weizhi Ma, Peijie Sun, Min Zhang
   - **Summary**: This study investigates the potential of large language models to serve as evaluators for recommendation explanations. By leveraging LLMs' capabilities in instruction following and common-sense reasoning, the authors aim to provide accurate, reproducible, and cost-effective evaluations of recommendation explanations.
   - **Year**: 2024

8. **Title**: ConceptGlassbox: Guided Concept-Based Explanation for Deep Neural Networks
   - **Authors**: Rania El Shawi
   - **Summary**: ConceptGlassbox introduces a framework for providing local explanations by learning high-level, transparent concept definitions. The approach incorporates human knowledge and feedback to facilitate the acquisition of concepts with minimal labeling effort, enhancing the interpretability of deep neural networks.
   - **Year**: 2024

9. **Title**: Black-box language model explanation by context length probing (arXiv:2212.14815)
   - **Authors**: Ondřej Cífka, Antoine Liutkus
   - **Summary**: The authors present context length probing, a novel technique for explaining causal language models by analyzing predictions as a function of available context length. This model-agnostic method assigns importance scores to different contexts without requiring access to model internals beyond token-level probabilities.
   - **Year**: 2022

10. **Title**: Human-Centered Concept Explanations for Neural Networks (arXiv:2202.12451)
    - **Authors**: Chih-Kuan Yeh, Been Kim, Pradeep Ravikumar
    - **Summary**: This paper reviews the emerging area of concept-based explanations, focusing on methods that align with human understanding. It discusses approaches to automatically extract concepts, measure their usefulness, and addresses challenges in providing human-centered explanations for neural networks.
    - **Year**: 2022

**Key Challenges:**

1. **Mapping Internal States to Human-Understandable Concepts**: Translating the complex internal representations of large language models into concepts that are intuitive and meaningful to humans remains a significant challenge.

2. **Scalability of Explanation Methods**: Developing explanation techniques that can efficiently handle the vast and intricate architectures of large language models, especially when processing long documents or extensive datasets, is difficult.

3. **Faithfulness and Accuracy of Explanations**: Ensuring that the generated explanations accurately reflect the model's decision-making process without introducing misleading or incorrect information is a critical concern.

4. **Balancing Transparency with Model Performance**: Enhancing the interpretability of models without compromising their performance or introducing significant computational overhead poses a delicate balance.

5. **User-Centric Explanation Design**: Creating explanations that are not only technically accurate but also accessible and useful to end-users with varying levels of technical expertise is a persistent challenge. 