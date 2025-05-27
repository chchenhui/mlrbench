1. **Title**: FedHPL: Efficient Heterogeneous Federated Learning with Prompt Tuning and Logit Distillation (arXiv:2405.17267)
   - **Authors**: Yuting Ma, Lechao Cheng, Yaxiong Wang, Zhun Zhong, Xiaohua Xu, Meng Wang
   - **Summary**: This paper introduces FedHPL, a federated learning framework designed to address challenges in heterogeneous settings by employing prompt tuning and logit distillation. It utilizes visual prompts to fine-tune pre-trained models efficiently and leverages logit distillation for knowledge aggregation, enhancing performance and convergence speed in federated environments.
   - **Year**: 2024

2. **Title**: FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models (arXiv:2310.01467)
   - **Authors**: Jingwei Sun, Ziyue Xu, Hongxu Yin, Dong Yang, Daguang Xu, Yiran Chen, Holger R. Roth
   - **Summary**: FedBPT presents a framework for federated prompt tuning of large language models without accessing model parameters. By focusing on training optimal prompts and utilizing gradient-free optimization methods, it reduces communication and computational overhead while maintaining competitive performance.
   - **Year**: 2023

3. **Title**: Federated Foundation Models: Privacy-Preserving and Collaborative Learning for Large Models (arXiv:2305.11414)
   - **Authors**: Sixing Yu, J. Pablo Muñoz, Ali Jannesari
   - **Summary**: This work discusses the integration of foundation models with federated learning to enable privacy-preserving and collaborative training across multiple users. It explores the benefits, challenges, and future research directions in federated foundation models, emphasizing data privacy and scalability.
   - **Year**: 2023

4. **Title**: FedDTPT: Federated Discrete and Transferable Prompt Tuning for Black-Box Large Language Models (arXiv:2411.00985)
   - **Authors**: Jiaqi Wu, Simin Chen, Yuzhe Yang, Yijiang Li, Shiyue Hou, Rui Jing, Zehua Wang, Wei Chen, Zijian Tian
   - **Summary**: FedDTPT introduces a federated discrete and transferable prompt tuning method for black-box large language models. It employs token-level discrete prompt optimization and semantic similarity-based filtering to enhance accuracy, reduce communication overhead, and ensure robustness in non-iid data settings.
   - **Year**: 2024

5. **Title**: Federated Learning of Large Language Models with Parameter-Efficient Prompt Tuning and Adaptive Optimization (arXiv:2310.15080)
   - **Authors**: Tianshi Che, Ji Liu, Yang Zhou, Jiaxiang Ren, Jiwen Zhou, Victor S. Sheng, Huaiyu Dai, Dejing Dou
   - **Summary**: This paper proposes FedPepTAO, a parameter-efficient prompt tuning approach with adaptive optimization for federated learning of large language models. It addresses client drift problems and enhances performance and efficiency through partial prompt tuning and novel optimization methods.
   - **Year**: 2023

6. **Title**: Efficient Federated Prompt Tuning for Black-box Large Pre-trained Models (arXiv:2310.03123)
   - **Authors**: Zihao Lin, Yan Sun, Yifan Shi, Xueqian Wang, Lifu Huang, Li Shen, Dacheng Tao
   - **Summary**: This work explores federated black-box prompt tuning (Fed-BBPT) to enable efficient tuning of large pre-trained models without accessing model parameters. It addresses challenges related to memory constraints, model privacy, and data privacy in federated learning scenarios.
   - **Year**: 2023

7. **Title**: Federated Prompt Learning for Weather Foundation Models on Devices (arXiv:2305.14244)
   - **Authors**: Shengchao Chen, Guodong Long, Tao Shen, Jing Jiang, Chengqi Zhang
   - **Summary**: FedPoD is introduced as a federated prompt learning framework for weather foundation models on devices. It employs adaptive prompt tuning and dynamic graph modeling to handle data heterogeneity and communication efficiency in on-device weather forecasting.
   - **Year**: 2023

8. **Title**: FedPT: Federated Proxy-Tuning of Large Language Models on Resource-Constrained Edge Devices (arXiv:2410.00362)
   - **Authors**: Zhidong Gao, Yu Zhang, Zhenxiao Zhang, Yanmin Gong, Yuanxiong Guo
   - **Summary**: FedPT introduces a federated proxy-tuning framework for large language models on resource-constrained edge devices. It collaboratively tunes a smaller model and combines its knowledge with a larger pre-trained model, reducing computation, communication, and memory overhead while maintaining performance.
   - **Year**: 2024

9. **Title**: Exploring Parameter-Efficient Fine-Tuning to Enable Foundation Models in Federated Learning (arXiv:2210.01708)
   - **Authors**: Guangyu Sun, Umar Khalid, Matias Mendieta, Pu Wang, Chen Chen
   - **Summary**: This paper investigates parameter-efficient fine-tuning methods to enable foundation models in federated learning. It evaluates the performance of such methods across various settings, demonstrating significant reductions in communication overhead while maintaining competitive performance.
   - **Year**: 2022

10. **Title**: FedP²EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs (arXiv:2502.04387)
    - **Authors**: Royson Lee, Minyoung Kim, Fady Rezk, Rui Li, Stylianos I. Venieris, Timothy Hospedales
    - **Summary**: FedP²EFT proposes a federated learning-to-personalize method for multilingual large language models in cross-device settings. It collaboratively learns optimal personalized parameter-efficient fine-tuning structures for each client, enhancing performance in multilingual federated learning scenarios.
    - **Year**: 2025

**Key Challenges:**

1. **Data Heterogeneity**: Variations in data distributions across clients can lead to client drift and degraded model performance in federated learning settings.

2. **Communication Overhead**: Transmitting large model updates between clients and the server can be resource-intensive, especially with large language models.

3. **Privacy Preservation**: Ensuring data privacy while collaboratively training models without sharing raw data remains a significant challenge.

4. **Model Accessibility**: Limited access to model parameters in black-box settings restricts the ability to fine-tune models effectively.

5. **Resource Constraints**: Deploying and fine-tuning large models on resource-constrained devices poses computational and memory challenges. 