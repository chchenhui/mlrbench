1. **Title**: FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models (arXiv:2310.01467)
   - **Authors**: Jingwei Sun, Ziyue Xu, Hongxu Yin, Dong Yang, Daguang Xu, Yiran Chen, Holger R. Roth
   - **Summary**: This paper introduces Federated Black-box Prompt Tuning (FedBPT), a framework designed to fine-tune pre-trained language models (PLMs) in federated learning settings without accessing model parameters. By focusing on training optimal prompts and utilizing gradient-free optimization methods, FedBPT reduces communication and computational overhead while preserving privacy. Experiments demonstrate significant reductions in communication and memory costs while maintaining competitive performance.
   - **Year**: 2023

2. **Title**: Federated Foundation Models: Privacy-Preserving and Collaborative Learning for Large Models (arXiv:2305.11414)
   - **Authors**: Sixing Yu, J. Pablo Muñoz, Ali Jannesari
   - **Summary**: This work proposes the Federated Foundation Models (FFMs) paradigm, combining foundation models with federated learning to enable privacy-preserving and collaborative learning across multiple users. The paper discusses integrating FL into the lifecycle of foundation models, covering pre-training, fine-tuning, and application, and outlines future research directions, including federated prompt tuning and continual learning.
   - **Year**: 2023

3. **Title**: FedDTPT: Federated Discrete and Transferable Prompt Tuning for Black-Box Large Language Models (arXiv:2411.00985)
   - **Authors**: Jiaqi Wu, Simin Chen, Yuzhe Yang, Yijiang Li, Shiyue Hou, Rui Jing, Zehua Wang, Wei Chen, Zijian Tian
   - **Summary**: This paper introduces FedDTPT, a federated discrete and transferable prompt tuning approach for black-box large language models. The method employs token-level discrete prompt optimization using a feedback loop based on prediction accuracy, enabling gradient-free prompt optimization through the model's API. The server utilizes an attention mechanism based on semantic similarity and clustering strategies to enhance prompt aggregation. Experimental results show improved accuracy, reduced communication overhead, and robustness to non-IID data.
   - **Year**: 2024

4. **Title**: Efficient Federated Prompt Tuning for Black-box Large Pre-trained Models (arXiv:2310.03123)
   - **Authors**: Zihao Lin, Yan Sun, Yifan Shi, Xueqian Wang, Lifu Huang, Li Shen, Dacheng Tao
   - **Summary**: The authors propose Federated Black-Box Prompt Tuning (Fed-BBPT), an approach that enables efficient tuning of large pre-trained models in federated settings without accessing model parameters or private datasets. By leveraging a central server to collaboratively train a prompt generator through regular aggregation and utilizing zero-order optimization via APIs, Fed-BBPT addresses memory constraints and privacy concerns. Evaluation across 40 datasets in computer vision and natural language processing tasks demonstrates the robustness of the proposed model.
   - **Year**: 2023

**Key Challenges:**

1. **Data Heterogeneity**: In federated learning, clients often possess non-IID (independent and identically distributed) data, leading to challenges in model convergence and performance consistency across clients.

2. **Communication Efficiency**: Transmitting large model updates between clients and the server can result in significant communication overhead, necessitating strategies to reduce the amount of data exchanged.

3. **Privacy Preservation**: Ensuring that sensitive client data remains private during the federated learning process is paramount, requiring robust privacy-preserving mechanisms.

4. **Computational Constraints**: Clients may have limited computational resources, making it challenging to fine-tune large foundation models locally without efficient optimization techniques.

5. **Model Accessibility**: In scenarios where clients only have access to model APIs without direct access to model parameters, developing effective tuning methods that operate under these constraints is essential. 