1. **Title**: ReLearn: Unlearning via Learning for Large Language Models (arXiv:2502.11190)
   - **Authors**: Haoming Xu, Ningyuan Zhao, Liming Yang, Sendong Zhao, Shumin Deng, Mengru Wang, Bryan Hooi, Nay Oo, Huajun Chen, Ningyu Zhang
   - **Summary**: This paper introduces ReLearn, a data augmentation and fine-tuning pipeline designed to effectively unlearn specific information in large language models (LLMs). The authors propose new evaluation metrics—Knowledge Forgetting Rate (KFR), Knowledge Retention Rate (KRR), and Linguistic Score (LS)—to assess the balance between forgetting targeted information and maintaining overall model performance. Experiments demonstrate that ReLearn successfully achieves targeted forgetting while preserving high-quality output.
   - **Year**: 2025

2. **Title**: Unlearn What You Want to Forget: Efficient Unlearning for LLMs (arXiv:2310.20150)
   - **Authors**: Jiaao Chen, Diyi Yang
   - **Summary**: This work presents an efficient unlearning framework for LLMs that introduces lightweight unlearning layers learned with a selective teacher-student objective into transformers. The framework allows for the removal of specific data without retraining the entire model. A fusion mechanism is also proposed to effectively combine different unlearning layers, enabling the handling of sequential forgetting operations. Experiments on classification and generation tasks demonstrate the effectiveness of the proposed methods compared to state-of-the-art baselines.
   - **Year**: 2023

3. **Title**: CodeUnlearn: Amortized Zero-Shot Machine Unlearning in Language Models Using Discrete Concept (arXiv:2410.10866)
   - **Authors**: YuXuan Wu, Bonaventure F. P. Dossou, Dianbo Liu
   - **Summary**: The authors propose an amortized unlearning approach using codebook features and Sparse Autoencoders (SAEs) to efficiently unlearn targeted information in LLMs. By leveraging a bottleneck to decompose the activation space and regulate information flow, the method effectively removes specific topics while preserving the model's performance on unrelated data. This approach marks a significant step towards real-world applications of machine unlearning in LLMs.
   - **Year**: 2024

4. **Title**: UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models (arXiv:2402.10052)
   - **Authors**: Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramon Huerta, Ivan Vulić
   - **Summary**: UNDIAL introduces a novel unlearning method that leverages self-distillation to adjust logits and selectively reduce the influence of targeted tokens. This technique ensures smooth convergence and avoids catastrophic forgetting, even in challenging unlearning tasks with large datasets and sequential unlearning requests. Extensive experiments show that UNDIAL achieves robustness in unlearning and scalability while maintaining stable training dynamics.
   - **Year**: 2024

5. **Title**: Large Language Model Unlearning via Embedding-Corrupted Prompts (arXiv:2406.07933)
   - **Authors**: Chris Yuhao Liu, Yaxuan Wang, Jeffrey Flanigan, Yang Liu
   - **Summary**: This paper presents Embedding-Corrupted (ECO) Prompts, a lightweight unlearning framework for LLMs that addresses knowledge entanglement and unlearning efficiency. Instead of relying on the LLM itself to unlearn, the method enforces an unlearned state during inference by employing a prompt classifier to identify and safeguard prompts to forget. The approach effectively achieves unlearning with minimal side effects and scales to LLMs ranging from 0.5B to 236B parameters without additional cost.
   - **Year**: 2024

6. **Title**: Multi-Objective Large Language Model Unlearning (arXiv:2412.20412)
   - **Authors**: Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao
   - **Summary**: The authors explore the Gradient Ascent (GA) approach in LLM unlearning and identify challenges such as gradient explosion and catastrophic forgetting. To address these issues, they propose the Multi-Objective Large Language Model Unlearning (MOLLM) algorithm, which formulates LLM unlearning as a multi-objective optimization problem. The approach modifies the cross-entropy loss to overcome gradient explosion and calculates a common descent update direction to enable the model to forget target data while preserving utility. Empirical results show that MOLLM outperforms state-of-the-art GA-based LLM unlearning methods.
   - **Year**: 2024

7. **Title**: Privacy in Fine-tuning Large Language Models: Attacks, Defenses, and Future Directions (arXiv:2412.16504)
   - **Authors**: Hao Du, Shang Liu, Lele Zheng, Yang Cao, Atsuyoshi Nakamura, Lei Chen
   - **Summary**: This comprehensive survey examines privacy challenges associated with fine-tuning LLMs, highlighting vulnerabilities to various privacy attacks, including membership inference, data extraction, and backdoor attacks. The authors review defense mechanisms designed to mitigate privacy risks in the fine-tuning phase, such as differential privacy, federated learning, and knowledge unlearning, discussing their effectiveness and limitations. The paper identifies key gaps in existing research and proposes directions to advance the development of privacy-preserving methods for fine-tuning LLMs.
   - **Year**: 2024

8. **Title**: PrivacyMind: Large Language Models Can Be Contextual Privacy Protection Learners (arXiv:2310.02469)
   - **Authors**: Yijia Xiao, Yiqiao Jin, Yushi Bai, Yue Wu, Xianjun Yang, Xiao Luo, Wenchao Yu, Xujiang Zhao, Yanchi Liu, Quanquan Gu, Haifeng Chen, Wei Wang, Wei Cheng
   - **Summary**: PrivacyMind introduces a novel paradigm for fine-tuning LLMs that effectively injects domain-specific knowledge while safeguarding inference-time data privacy. The authors offer a theoretical analysis for model design and benchmark various techniques such as corpus curation, penalty-based unlikelihood in training loss, and instruction-based tuning. Extensive experiments across diverse datasets and scenarios demonstrate the effectiveness of the approaches, with instruction tuning using both positive and negative examples standing out as a promising method.
   - **Year**: 2023

9. **Title**: Practical Unlearning for Large Language Models (arXiv:2407.10223)
   - **Authors**: Chongyang Gao, Lixu Wang, Chenkai Weng, Xiao Wang, Qi Zhu
   - **Summary**: This paper introduces the O3 framework, which includes an Out-Of-Distribution (OOD) detector to measure the similarity between input and unlearning data, and an Orthogonal low-rank adapter (LoRA) for continuously unlearning requested data. The OOD detector is trained with a novel contrastive entropy loss and utilizes a local-global layer-aggregated scoring mechanism. The orthogonal LoRA achieves parameter disentanglement among continual unlearning requests. Extensive experiments indicate that O3 consistently achieves the best trade-off between unlearning effectiveness and utility preservation, especially when facing continuous unlearning requests.
   - **Year**: 2024

10. **Title**: A Comprehensive Survey of Machine Unlearning Techniques for Large Language Models (arXiv:2503.01854)
    - **Authors**: Jiahui Geng, Qing Li, Herbert Woisetschlaeger, Zongxiong Chen, Yuxia Wang, Preslav Nakov, Hans-Arno Jacobsen, Fakhri Karray
    - **Summary**: This study investigates machine unlearning techniques within the context of LLMs, offering a comprehensive taxonomy of existing unlearning studies. The authors categorize current unlearning approaches, summarizing their strengths and limitations, and review evaluation metrics and benchmarks. The paper outlines promising directions for future research, highlighting key challenges and opportunities in the field.
    - **Year**: 2025

**Key Challenges:**

1. **Balancing Unlearning and Utility Preservation**: Achieving effective unlearning of specific information while maintaining the overall performance and utility of the LLM remains a significant challenge.

2. **Computational Efficiency**: Developing unlearning methods that are computationally efficient and scalable to large models without requiring full retraining is crucial.

3. **Handling Sequential Unlearning Requests**: Managing multiple and continuous unlearning requests without degrading model performance poses a complex challenge.

4. **Ensuring Robustness Against Privacy Attacks**: Protecting LLMs from various privacy attacks, such as membership inference and data extraction, during and after the unlearning process is essential.

5. **Developing Reliable Evaluation Metrics**: Establishing comprehensive and reliable metrics to assess the effectiveness of unlearning methods and their impact on model performance is necessary for advancing the field. 