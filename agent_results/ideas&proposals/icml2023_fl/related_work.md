1. **Title**: SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models (arXiv:2308.06522)
   - **Authors**: Sara Babakniya, Ahmed Roushdy Elkordy, Yahya H. Ezzeldin, Qingfeng Liu, Kee-Bong Song, Mostafa El-Khamy, Salman Avestimehr
   - **Summary**: This paper explores the application of Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA, in Federated Learning (FL) settings for language tasks. It introduces SLoRA, a method that addresses performance degradation in heterogeneous data scenarios through a novel data-driven initialization technique. SLoRA achieves performance comparable to full fine-tuning with significantly reduced training time and sparse updates.
   - **Year**: 2023

2. **Title**: FeDeRA: Efficient Fine-tuning of Language Models in Federated Learning Leveraging Weight Decomposition (arXiv:2404.18848)
   - **Authors**: Yuxuan Yan, Qianqian Yang, Shunpu Tang, Zhiguo Shi
   - **Summary**: FeDeRA extends the LoRA method by initializing low-rank matrices using singular value decomposition (SVD) of pre-trained weight matrices. This approach enhances performance in non-i.i.d. data scenarios within FL, achieving results comparable to full fine-tuning while significantly reducing the number of trainable parameters and training time.
   - **Year**: 2024

3. **Title**: FedP$^2$EFT: Federated Learning to Personalize Parameter Efficient Fine-Tuning for Multilingual LLMs (arXiv:2502.04387)
   - **Authors**: Royson Lee, Minyoung Kim, Fady Rezk, Rui Li, Stylianos I. Venieris, Timothy Hospedales
   - **Summary**: This work introduces FedP$^2$EFT, a federated learning-to-personalize method for multilingual large language models (LLMs) in cross-device FL settings. It collaboratively learns optimal personalized PEFT structures for each client via Bayesian sparse rank selection, outperforming existing personalized fine-tuning methods and complementing various FL methods.
   - **Year**: 2025

4. **Title**: FedMCP: Parameter-Efficient Federated Learning with Model-Contrastive Personalization (arXiv:2409.00116)
   - **Authors**: Qianyi Zhao, Chen Qu, Cen Chen, Mingyuan Fan, Yanhao Wang
   - **Summary**: FedMCP introduces a parameter-efficient fine-tuning method with model-contrastive personalization for FL. It employs global and private adapter modules within clients and introduces a model-contrastive regularization term to balance universal and client-specific knowledge, effectively providing personalized models tailored to individual clients.
   - **Year**: 2024

5. **Title**: FedPEAT: Convergence of Federated Learning, Parameter-Efficient Fine Tuning, and Emulator Assisted Tuning for Artificial Intelligence Foundation Models with Mobile Edge Computing (arXiv:2310.17491)
   - **Authors**: Terence Jie Chua, Wenhan Yu, Jun Zhao, Kwok-Yan Lam
   - **Summary**: FedPEAT combines Emulator-Assisted Tuning (EAT) with PEFT to form Parameter-Efficient Emulator-Assisted Tuning (PEAT) and extends it into federated learning as Federated PEAT (FedPEAT). This approach uses adapters, emulators, and PEFT for federated model tuning, enhancing model privacy and memory efficiency, and is adaptable to various neural networks.
   - **Year**: 2023

6. **Title**: Federated Foundation Models: Privacy-Preserving and Collaborative Learning for Large Models (arXiv:2305.11414)
   - **Authors**: Sixing Yu, J. Pablo Muñoz, Ali Jannesari
   - **Summary**: This paper proposes the Federated Foundation Models (FFMs) paradigm, combining the benefits of foundation models and federated learning to enable privacy-preserving and collaborative learning across multiple end-users. It discusses potential benefits, challenges, and future research avenues in FFM, including pre-training, fine-tuning, and federated prompt tuning.
   - **Year**: 2023

7. **Title**: FedPT: Federated Proxy-Tuning of Large Language Models on Resource-Constrained Edge Devices (arXiv:2410.00362)
   - **Authors**: Zhidong Gao, Yu Zhang, Zhenxiao Zhang, Yanmin Gong, Yuanxiong Guo
   - **Summary**: FedPT introduces a framework for federated fine-tuning of black-box large language models, requiring access only to their predictions over the output vocabulary. Devices collaboratively tune a smaller model, and the server combines this knowledge with the larger pre-trained model to construct a proxy-tuned model, reducing computation, communication, and memory overhead while maintaining competitive performance.
   - **Year**: 2024

8. **Title**: Federated Generative Learning with Foundation Models (arXiv:2306.16064)
   - **Authors**: Jie Zhang, Xiaohua Qi, Bo Zhao
   - **Summary**: This work proposes a novel federated learning framework where clients create text embeddings tailored to their local data and send them to the server. The server synthesizes informative training data using foundation generative models with these embeddings, enhancing communication efficiency, robustness to data heterogeneity, and privacy protection.
   - **Year**: 2023

9. **Title**: FedCoLLM: A Parameter-Efficient Federated Co-tuning Framework for Large and Small Language Models (arXiv:2411.11707)
   - **Authors**: Tao Fan
   - **Summary**: FedCoLLM introduces a parameter-efficient federated framework designed for co-tuning large and small language models. It utilizes lightweight adapters with small language models to facilitate knowledge exchange between server and clients, respecting data privacy while minimizing computational and communication overhead.
   - **Year**: 2024

10. **Title**: Exploring Parameter-Efficient Fine-Tuning to Enable Foundation Models in Federated Learning (arXiv:2210.01708)
    - **Authors**: Guangyu Sun, Umar Khalid, Matias Mendieta, Pu Wang, Chen Chen
    - **Summary**: This paper investigates the use of parameter-efficient fine-tuning in federated learning, introducing the FedPEFT framework. By locally tuning and globally sharing a small portion of model weights, it achieves significant reductions in communication overhead while maintaining competitive performance across various federated learning scenarios.
    - **Year**: 2022

**Key Challenges:**

1. **Data Heterogeneity**: Variations in data distributions across clients can lead to performance degradation in federated learning settings, making it challenging to achieve consistent model performance.

2. **Resource Constraints**: Limited computational, memory, and communication resources on client devices hinder the deployment and fine-tuning of large foundation models in federated learning environments.

3. **Privacy Preservation**: Ensuring data privacy while enabling effective model training remains a significant challenge, especially when dealing with sensitive or personal data.

4. **Efficient Communication**: Reducing communication overhead during model updates is crucial to maintain the scalability and efficiency of federated learning systems.

5. **Personalization**: Developing methods that allow models to adapt to individual client data while maintaining generalization across the federation is essential for effective federated learning. 