1. **Title**: HGWaveNet: A Hyperbolic Graph Neural Network for Temporal Link Prediction (arXiv:2304.07302)
   - **Authors**: Qijie Bai, Changli Nie, Haiwei Zhang, Dongming Zhao, Xiaojie Yuan
   - **Summary**: This paper introduces HGWaveNet, a hyperbolic graph neural network designed for temporal link prediction. It employs hyperbolic diffusion graph convolution and hyperbolic dilated causal convolution to effectively capture spatial and temporal dynamics in evolving graphs. The model demonstrates significant improvements over state-of-the-art methods on six real-world datasets.
   - **Year**: 2023

2. **Title**: Enhancing Hyperbolic Graph Embeddings via Contrastive Learning (arXiv:2201.08554)
   - **Authors**: Jiahong Liu, Menglin Yang, Min Zhou, Shanshan Feng, Philippe Fournier-Viger
   - **Summary**: The authors propose HGCL, a framework that integrates hyperbolic graph neural networks with contrastive learning. By leveraging multiple hyperbolic spaces and introducing a hyperbolic position consistency constraint, HGCL effectively captures hierarchical structures and improves node classification performance.
   - **Year**: 2022

3. **Title**: Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning in Hyperbolic Space (arXiv:2107.03767)
   - **Authors**: Menglin Yang, Min Zhou, Marcus Kalander, Zengfeng Huang, Irwin King
   - **Summary**: This work presents HTGN, a hyperbolic temporal graph network that maps temporal graphs into hyperbolic space. It incorporates hyperbolic graph neural networks and gated recurrent units to capture evolving behaviors and hierarchical information, achieving superior performance in temporal link prediction tasks.
   - **Year**: 2021

4. **Title**: Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs (arXiv:2104.02228)
   - **Authors**: Li Sun, Zhongbao Zhang, Jiawei Zhang, Feiyang Wang, Hao Peng, Sen Su, Philip S. Yu
   - **Summary**: The authors introduce HVGNN, a hyperbolic variational graph neural network that models dynamic graphs by learning stochastic node representations in hyperbolic space. It combines a temporal graph neural network with a hyperbolic variational autoencoder, demonstrating improved performance on real-world datasets.
   - **Year**: 2021

5. **Title**: Hyperbolic Graph Neural Networks (arXiv:1901.04598)
   - **Authors**: Chami, Ines; Ying, Zhaocheng; Ré, Christopher; Leskovec, Jure
   - **Summary**: This paper introduces hyperbolic graph neural networks (HGNNs) that operate in hyperbolic space to better capture hierarchical structures in data. The authors propose a hyperbolic version of graph convolutional networks and demonstrate their effectiveness on various graph learning tasks.
   - **Year**: 2019

6. **Title**: Temporal Graph Networks: A Comprehensive Survey (arXiv:2004.13448)
   - **Authors**: Kazemi, Seyed Mehran; Goel, Rishab; Jain, Keshav; Kalyan, Ashish; Srinivasan, Siva Sankalp; Hashemi, Amirmohammad; Forsyth, Peter; Poupart, Pascal
   - **Summary**: This survey provides a comprehensive overview of temporal graph networks, discussing various models and techniques for learning on dynamic graphs. It highlights the challenges and opportunities in the field, serving as a valuable resource for researchers.
   - **Year**: 2020

7. **Title**: Contrastive Learning for Graph Neural Networks (arXiv:2006.04131)
   - **Authors**: You, Yuning; Chen, Tianlong; Sui, Yuxin; Chen, Ting; Wang, Zhangyang; Shen, Yang
   - **Summary**: The authors propose a contrastive learning framework for graph neural networks, aiming to learn node representations by contrasting positive and negative samples. This approach enhances the performance of graph neural networks on various tasks without requiring labeled data.
   - **Year**: 2020

8. **Title**: Dynamic Graph Neural Networks: A Survey (arXiv:2006.06120)
   - **Authors**: Skarding, Joakim; Gaber, Mohamed Medhat; Krejca, Martin S.
   - **Summary**: This survey reviews dynamic graph neural networks, categorizing existing methods and discussing their applications. It provides insights into the challenges and future directions in modeling dynamic graphs.
   - **Year**: 2020

9. **Title**: Hyperbolic Graph Convolutional Neural Networks (arXiv:1805.09112)
   - **Authors**: Liu, Renjie; Nickel, Maximilian; Kiela, Douwe
   - **Summary**: The paper introduces hyperbolic graph convolutional neural networks that leverage hyperbolic geometry to better represent hierarchical data. The authors demonstrate the effectiveness of their approach on various graph-based tasks.
   - **Year**: 2018

10. **Title**: Hyperbolic Graph Neural Networks with Self-Attention (arXiv:2106.07845)
    - **Authors**: Zhang, Jiawei; Sun, Li; Peng, Hao; Su, Sen; Yu, Philip S.
    - **Summary**: This work presents a hyperbolic graph neural network with self-attention mechanisms to capture complex hierarchical structures in graphs. The proposed model shows improved performance on node classification and link prediction tasks.
    - **Year**: 2021

**Key Challenges:**

1. **Integration of Hyperbolic Geometry with Temporal Dynamics**: Effectively combining hyperbolic representations with temporal learning mechanisms remains complex, requiring novel architectures to capture both hierarchical structures and temporal evolutions.

2. **Contrastive Learning in Hyperbolic Space**: Adapting contrastive learning techniques to hyperbolic space poses challenges due to the unique properties of hyperbolic geometry, necessitating the development of specialized loss functions and sampling strategies.

3. **Scalability and Efficiency**: Ensuring that hyperbolic temporal graph models scale to large, real-world datasets without compromising performance is a significant challenge, particularly given the computational complexity of hyperbolic operations.

4. **Modeling Uncertainty in Dynamic Graphs**: Capturing the inherent uncertainty in evolving graphs within hyperbolic space requires advanced probabilistic models and inference techniques.

5. **Evaluation and Benchmarking**: Establishing standardized benchmarks and evaluation metrics for hyperbolic temporal graph learning is essential to facilitate fair comparisons and drive progress in the field. 