Retrieval-Augmented Generation (RAG) has become a pivotal paradigm in enhancing the capabilities of foundation models by integrating external knowledge sources. The proposed MRIA framework aims to provide a scalable and efficient method for attributing outputs to specific data sources within RAG-based systems. This literature review explores recent advancements related to MRIA's objectives, focusing on works published between 2023 and 2025.

**1. Related Papers**

1. **Title**: "Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation" (arXiv:2505.16415)
   - **Authors**: Ruizhe Li, Chen Chen, Yuchen Hu, Yanjun Gao, Xi Wang, Emine Yilmaz
   - **Summary**: This study introduces ARC-JSD, a method leveraging Jensen-Shannon Divergence to efficiently attribute generated content to specific context segments in RAG systems without additional fine-tuning.
   - **Year**: 2025

2. **Title**: "Trustworthiness in Retrieval-Augmented Generation Systems: A Survey" (arXiv:2409.10102)
   - **Authors**: Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Zheng Liu, Chaozhuo Li, Zhicheng Dou, Tsung-Yi Ho, Philip S. Yu
   - **Summary**: This survey assesses the trustworthiness of RAG systems across dimensions like factuality, robustness, and transparency, highlighting the importance of reliable data attribution.
   - **Year**: 2024

3. **Title**: "Chain-of-Retrieval Augmented Generation" (arXiv:2501.14342)
   - **Authors**: Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, Furu Wei
   - **Summary**: The paper presents CoRAG, a method that dynamically reformulates queries during generation, enhancing the adaptability and accuracy of RAG systems.
   - **Year**: 2025

4. **Title**: "Distributed Retrieval-Augmented Generation" (arXiv:2505.00443)
   - **Authors**: Chenhao Xu, Longxiang Gao, Yuan Miao, Xi Zheng
   - **Summary**: This work introduces DRAG, a framework that eliminates the need for centralized knowledge bases in RAG systems, enhancing data privacy and scalability.
   - **Year**: 2025

5. **Title**: "Dynamic and Parametric Retrieval-Augmented Generation" (arXiv:2506.06704)
   - **Authors**: Weihang Su, Qingyao Ai, Jingtao Zhan, Qian Dong, Yiqun Liu
   - **Summary**: This tutorial explores dynamic and parametric approaches in RAG, focusing on adaptive retrieval and parameter-level knowledge integration to improve efficiency.
   - **Year**: 2025

6. **Title**: "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach" (arXiv:2407.16833)
   - **Authors**: Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky
   - **Summary**: This study compares RAG with long-context LLMs, proposing a hybrid approach that balances performance and computational cost.
   - **Year**: 2024

7. **Title**: "A Retrieval-Augmented Generation Framework for Academic Literature Navigation in Data Science" (arXiv:2412.15404)
   - **Authors**: Ahmet Yasin Aytar, Kemal Kilic, Kamer Kaya
   - **Summary**: The paper presents an AI-based RAG system designed to assist data scientists in efficiently navigating academic literature, highlighting the importance of accurate information retrieval.
   - **Year**: 2024

8. **Title**: "SIM-Shapley: A Stable and Computationally Efficient Approach to Shapley Value Approximation" (arXiv:2505.08198)
   - **Authors**: Wangxuan Fan, Siqi Li, Doudou Zhou, Yohei Okada, Chuan Hong, Molei Liu, Nan Liu
   - **Summary**: This work introduces SIM-Shapley, a method inspired by stochastic optimization to approximate Shapley values efficiently, reducing computation time significantly.
   - **Year**: 2025

9. **Title**: "Provably Accurate Shapley Value Estimation via Leverage Score Sampling" (arXiv:2410.01917)
   - **Authors**: Christopher Musco, R. Teal Witter
   - **Summary**: The paper presents Leverage SHAP, a modification of Kernel SHAP that provides accurate Shapley value estimates with reduced computational requirements.
   - **Year**: 2024

10. **Title**: "A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values" (arXiv:2506.05216)
    - **Authors**: Tyler Chen, Akshay Seshadri, Mattia J. Villani, Pradeep Niroula, Shouvanik Chakrabarti, Archan Ray, Pranav Deshpande, Romina Yalovetzky, Marco Pistoia, Niraj Kumar
    - **Summary**: This work provides a unified framework encompassing various Shapley value estimators, offering theoretical guarantees and practical improvements in computation.
    - **Year**: 2025

**2. Key Challenges**

1. **Scalability of Attribution Methods**: Existing attribution techniques often struggle with the computational demands of large-scale RAG systems, necessitating more efficient algorithms.

2. **Integration Complexity**: Seamlessly incorporating attribution frameworks like MRIA into existing RAG pipelines without disrupting performance remains a significant challenge.

3. **Accuracy vs. Efficiency Trade-off**: Balancing the precision of attribution methods with their computational efficiency is crucial for practical deployment.

4. **Evaluation Metrics**: Developing standardized benchmarks and metrics to assess the effectiveness of attribution methods in RAG contexts is essential for progress.

5. **Transparency and Interpretability**: Ensuring that attribution methods provide clear and interpretable insights into the contributions of retrieved documents to generated outputs is vital for user trust and compliance. 