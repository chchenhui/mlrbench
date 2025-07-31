1. **Title**: MoxE: Mixture of xLSTM Experts with Entropy-Aware Routing for Efficient Language Modeling (arXiv:2505.01459)
   - **Authors**: Abdoul Majid O. Thiombiano, Brahim Hnich, Ali Ben Mrad, Mohamed Wiem Mkaouer
   - **Summary**: This paper introduces MoxE, a novel architecture that combines Extended Long Short-Term Memory (xLSTM) with the Mixture of Experts (MoE) framework. It features an entropy-based routing mechanism to dynamically assign tokens to specialized experts, enhancing efficiency and scalability in large language models.
   - **Year**: 2025

2. **Title**: ResMoE: Space-efficient Compression of Mixture of Experts LLMs via Residual Restoration (arXiv:2503.06881)
   - **Authors**: Mengting Ai, Tianxin Wei, Yifan Chen, Zhichen Zeng, Ritchie Zhao, Girish Varatkar, Bita Darvish Rouhani, Xianfeng Tang, Hanghang Tong, Jingrui He
   - **Summary**: ResMoE presents a framework for compressing Mixture-of-Experts (MoE) Transformers by extracting a common expert and approximating residuals between this expert and the original ones. This approach reduces parameter count by up to 75% while maintaining performance, facilitating more efficient inference.
   - **Year**: 2025

3. **Title**: MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts (arXiv:2506.07533)
   - **Authors**: Wei Tao, Haocheng Lu, Xiaoyang Qu, Bin Zhang, Kai Lu, Jiguang Wan, Jianzong Wang
   - **Summary**: MoQAE introduces a mixed-precision quantization method using a mixture of quantization-aware experts. It employs a chunk-based routing mechanism and a lightweight fine-tuning process to balance model accuracy and memory usage, effectively reducing KV cache memory consumption during long-context inference.
   - **Year**: 2025

4. **Title**: SMILE: Scaling Mixture-of-Experts with Efficient Bi-level Routing (arXiv:2212.05191)
   - **Authors**: Chaoyang He, Shuai Zheng, Aston Zhang, George Karypis, Trishul Chilimbi, Mahdi Soltanolkotabi, Salman Avestimehr
   - **Summary**: SMILE proposes a bi-level routing mechanism to improve the scalability of Mixture-of-Experts models. By exploiting heterogeneous network bandwidth and splitting routing into two levels, it achieves a 2.5x speedup in pretraining throughput without compromising convergence speed.
   - **Year**: 2022

5. **Title**: Efficient Content-Based Sparse Attention with Routing Transformers
   - **Authors**: Not specified
   - **Summary**: This work introduces the Routing Transformer, which utilizes content-based sparse attention to manage long sequences efficiently. By selecting sparsity patterns based on content similarity, it reduces the quadratic complexity of traditional attention mechanisms, making it suitable for long-context tasks.
   - **Year**: 2022

6. **Title**: Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
   - **Authors**: Not specified
   - **Summary**: This paper introduces Default MoE, a method that enhances training stability and performance in sparse Mixture-of-Experts models by providing dense gradient updates to the router. It substitutes missing expert activations with default outputs, enabling more effective training.
   - **Year**: 2025

7. **Title**: MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
   - **Authors**: Shiyi Cao et al.
   - **Summary**: MoE-Lightning addresses the challenge of deploying Mixture-of-Experts models on memory-constrained GPUs. It introduces techniques for efficient expert selection and memory management, enabling high-throughput inference without significant performance degradation.
   - **Year**: 2024

8. **Title**: Lynx: Enabling Efficient MoE Inference through Dynamic Batch-Aware Expert Selection
   - **Authors**: Vima Gupta et al.
   - **Summary**: Lynx proposes a dynamic, batch-aware expert selection mechanism for Mixture-of-Experts models. By adapting expert selection based on input batches, it improves inference efficiency and load balancing across experts, reducing latency and computational overhead.
   - **Year**: 2024

9. **Title**: MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching
   - **Authors**: Not specified
   - **Summary**: MoE-Gen introduces a module-based batching strategy to facilitate high-throughput inference of Mixture-of-Experts models on a single GPU. This approach optimizes resource utilization and reduces inference latency, making MoE models more practical for deployment.
   - **Year**: 2025

10. **Title**: MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
    - **Authors**: Not specified
    - **Summary**: MoE-Lens explores the hardware limitations of serving large Mixture-of-Experts language models under resource constraints. It presents strategies for optimizing inference throughput and resource utilization, aiming to make MoE models more accessible and efficient.
    - **Year**: 2025

**Key Challenges:**

1. **Memory Efficiency**: Managing the KV cache in transformer models for long-context tasks remains challenging due to its quadratic growth, leading to increased memory consumption and reduced efficiency.

2. **Adaptive Routing**: Developing effective routing mechanisms in Mixture-of-Experts models that can dynamically and efficiently assign tokens to appropriate experts without introducing significant computational overhead is complex.

3. **Training Stability**: Ensuring stable and efficient training of sparse Mixture-of-Experts models is difficult, particularly when dealing with issues like unbalanced expert utilization and gradient updates.

4. **Inference Latency**: Reducing inference latency in large language models, especially those employing Mixture-of-Experts architectures, is critical for real-time applications but remains a significant hurdle.

5. **Scalability**: Scaling Mixture-of-Experts models to handle ultra-long contexts and streaming data without compromising performance or efficiency poses substantial challenges in both model design and hardware utilization. 