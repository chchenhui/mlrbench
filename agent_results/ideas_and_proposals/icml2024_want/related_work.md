1. **Title**: Reducing Activation Recomputation in Large Transformer Models (arXiv:2205.05198)
   - **Authors**: Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro
   - **Summary**: This paper introduces sequence parallelism and selective activation recomputation to minimize redundant computations in large transformer models. By integrating these techniques with tensor parallelism, the authors achieve a 5x reduction in activation memory and a 29% increase in training speed for a 530B parameter GPT-3 style model.
   - **Year**: 2022

2. **Title**: Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey (arXiv:2403.14608)
   - **Authors**: Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang
   - **Summary**: This survey examines various parameter-efficient fine-tuning (PEFT) algorithms designed to adapt large models to specific tasks with minimal additional parameters and computational resources. It discusses the performance, computational overhead, and system implementation costs associated with different PEFT approaches.
   - **Year**: 2024

3. **Title**: Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models (arXiv:2401.00625)
   - **Authors**: Guangji Bai, Zheng Chai, Chen Ling, Shiyu Wang, Jiaying Lu, Nan Zhang, Tingwei Shi, Ziyang Yu, Mengdan Zhu, Yifei Zhang, Xinyuan Song, Carl Yang, Yue Cheng, Liang Zhao
   - **Summary**: This survey addresses the challenges of high resource consumption in large language models (LLMs) and reviews techniques aimed at enhancing their resource efficiency. It categorizes methods based on optimization focus—computational, memory, energy, financial, and network resources—and their applicability across various stages of an LLM's lifecycle.
   - **Year**: 2024

4. **Title**: Dynamic Tensor Rematerialization (arXiv:2006.09616)
   - **Authors**: Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, Zachary Tatlock
   - **Summary**: This paper presents Dynamic Tensor Rematerialization (DTR), an online algorithm for checkpointing that supports dynamic models by introducing a greedy eviction policy. DTR achieves comparable performance to optimal static checkpointing and is implemented in PyTorch by interposing on tensor allocations and operator calls.
   - **Year**: 2020

**Key Challenges:**

1. **Balancing Memory Savings and Computational Overhead**: Activation checkpointing reduces memory usage but introduces re-computation overhead. Developing strategies that effectively balance these aspects without compromising training efficiency remains a significant challenge.

2. **Dynamic Adaptation to Training Phases**: Static checkpointing strategies may not adapt well to different training stages. Implementing dynamic methods that adjust checkpointing decisions based on the evolving importance of activations during training is complex.

3. **Efficient Gradient Impact Estimation**: Accurately estimating the impact of activations on gradient updates with minimal computational overhead is crucial. Developing lightweight proxies or metrics for this purpose is challenging.

4. **Integration with Distributed Training Frameworks**: Incorporating advanced checkpointing strategies into existing distributed training frameworks without introducing significant complexity or performance degradation is a non-trivial task.

5. **Ensuring Convergence and Model Performance**: While optimizing for computational efficiency, it is essential to ensure that these strategies do not adversely affect model convergence or final performance, which requires careful validation and testing. 