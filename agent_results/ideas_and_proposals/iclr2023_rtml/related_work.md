**Related Papers:**

1. **Title**: Fast-NTK: Parameter-Efficient Unlearning for Large-Scale Models (arXiv:2312.14923)
   - **Authors**: Guihong Li, Hsiang Hsu, Chun-Fu Chen, Radu Marculescu
   - **Summary**: This paper introduces "Fast-NTK," an NTK-based unlearning algorithm that incorporates parameter-efficient fine-tuning methods, such as fine-tuning batch normalization layers in CNNs or visual prompts in vision transformers. The approach significantly reduces computational complexity, enabling scalability to larger neural networks and datasets while maintaining performance comparable to retraining on the retain set alone.
   - **Year**: 2023

2. **Title**: Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning (arXiv:2406.16257)
   - **Authors**: Somnath Basu Roy Chowdhury, Krzysztof Choromanski, Arijit Sehanobish, Avinava Dubey, Snigdha Chaturvedi
   - **Summary**: The authors propose "Sequence-aware Sharded Sliced Training (S3T)," an exact unlearning framework that enhances deletion capabilities while minimizing performance impact. By utilizing a lightweight parameter-efficient fine-tuning approach, S3T enables efficient unlearning by deactivating layers affected by data deletion, reducing retraining costs and improving model performance.
   - **Year**: 2024

3. **Title**: LMEraser: Large Model Unlearning through Adaptive Prompt Tuning (arXiv:2404.11056)
   - **Authors**: Jie Xu, Zihan Wu, Cong Wang, Xiaohua Jia
   - **Summary**: LMEraser introduces an efficient machine unlearning approach for large models by employing a prompt tuning architecture. The method partitions the training dataset into public and private subsets, using public data to train the backbone and private data to optimize prompts. This adaptive prompt tuning mechanism reduces unlearning costs and maintains model performance.
   - **Year**: 2024

4. **Title**: Multi-Objective Large Language Model Unlearning (arXiv:2412.20412)
   - **Authors**: Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao
   - **Summary**: This paper explores the Gradient Ascent approach in LLM unlearning, addressing challenges like gradient explosion and catastrophic forgetting. The proposed Multi-Objective Large Language Model Unlearning (MOLLM) algorithm formulates unlearning as a multi-objective optimization problem, enabling the model to forget target data while preserving utility.
   - **Year**: 2024

5. **Title**: SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation (arXiv:2310.12508)
   - **Authors**: Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, Sijia Liu
   - **Summary**: SalUn introduces the concept of 'weight saliency' for machine unlearning, directing attention toward specific model weights to improve effectiveness and efficiency. The method effectively erases the influence of forgetting data, classes, or concepts in both image classification and generation tasks, narrowing the performance gap with exact unlearning.
   - **Year**: 2023

6. **Title**: Machine Unlearning of Pre-trained Large Language Models (arXiv:2402.15159)
   - **Authors**: Jin Yao, Eli Chien, Minxin Du, Xinyao Niu, Tianhao Wang, Zezhou Cheng, Xiang Yue
   - **Summary**: This study investigates machine unlearning in pre-trained LLMs, presenting a comprehensive framework encompassing a critical analysis of seven unlearning methods. The authors establish a robust benchmark for unlearning performance, demonstrating that these methods are significantly more computationally efficient than retraining.
   - **Year**: 2024

7. **Title**: A Comprehensive Survey of Machine Unlearning Techniques for Large Language Models (arXiv:2503.01854)
   - **Authors**: Jiahui Geng, Qing Li, Herbert Woisetschlaeger, Zongxiong Chen, Yuxia Wang, Preslav Nakov, Hans-Arno Jacobsen, Fakhri Karray
   - **Summary**: This survey investigates machine unlearning techniques within the context of LLMs, offering a comprehensive taxonomy of existing unlearning studies. The authors categorize current approaches, summarize their strengths and limitations, and outline promising directions for future research.
   - **Year**: 2025

8. **Title**: Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace (arXiv:2503.01419)
   - **Authors**: Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia, Dong-Hai Zhu, Xi-He Qiu
   - **Summary**: The authors propose a novel parameter-efficient fine-tuning method that combines deconvolution with subspace learning, reducing the number of parameters required for fine-tuning by eight times. Experimental results demonstrate superior training efficiency and performance compared to existing models.
   - **Year**: 2025

9. **Title**: ReLearn: Unlearning via Learning for Large Language Models (arXiv:2502.11190)
   - **Authors**: Haoming Xu, Ningyuan Zhao, Liming Yang, Sendong Zhao, Shumin Deng, Mengru Wang, Bryan Hooi, Nay Oo, Huajun Chen, Ningyu Zhang
   - **Summary**: ReLearn introduces a data augmentation and fine-tuning pipeline for effective unlearning in LLMs. The approach addresses challenges in existing unlearning methods by preserving response fluency and relevance, achieving targeted forgetting while maintaining high-quality output.
   - **Year**: 2025

10. **Title**: Faster Machine Unlearning via Natural Gradient Descent (arXiv:2407.08169)
    - **Authors**: Omri Lev, Ashia Wilson
    - **Summary**: This paper proposes a novel algorithm leveraging Natural Gradient Descent to efficiently and reliably delete data from machine learning models trained using Empirical Risk Minimization. The approach ensures strong privacy guarantees for convex models and develops a practical Min/Max optimization algorithm for non-convex models.
    - **Year**: 2024

**Key Challenges:**

1. **Computational Efficiency**: Developing unlearning methods that are computationally efficient and scalable to large models without requiring full retraining remains a significant challenge.

2. **Preserving Model Utility**: Ensuring that unlearning processes effectively remove specific data influences while maintaining the overall performance and utility of the model is complex.

3. **Addressing Catastrophic Forgetting**: Unlearning specific data points without causing the model to forget other important information is a delicate balance that needs to be maintained.

4. **Formal Privacy Guarantees**: Providing formal privacy guarantees, such as differential unlearning, to ensure compliance with regulations like GDPR is a critical yet challenging aspect of machine unlearning.

5. **Generalization Across Domains**: Developing unlearning methods that are effective across various tasks and domains, including both classification and generation tasks, poses a significant challenge. 