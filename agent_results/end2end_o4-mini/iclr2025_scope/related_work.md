1. **Title**: ZACK: Zero-Overhead LLM Inference Acceleration via Dimensionality Compression of the Key-Value Cache (arXiv:2408.04107)
   - **Authors**: Zeyu Zhang, Haiying Shen
   - **Summary**: ZACK introduces a KV dimensionality compression system that achieves zero-overhead compression and decompression, reducing attention computation time. It employs adaptive compression, tailoring KV compression rates across heads and layers based on their contributions to inference, maximizing overall compression while maintaining accuracy.
   - **Year**: 2024

2. **Title**: DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs (arXiv:2412.14838)
   - **Authors**: Xiabin Zhou, Wenbin Wang, Minyan Zeng, Jiaxian Guo, Xuebo Liu, Li Shen, Min Zhang, Liang Ding
   - **Summary**: DynamicKV proposes a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to specific tasks. It establishes global and per-layer maximum KV cache budgets, periodically updating KV cache sizes during inference, achieving significant compression while maintaining performance.
   - **Year**: 2024

3. **Title**: RazorAttention: Efficient KV Cache Compression Through Retrieval Heads (arXiv:2407.15891)
   - **Authors**: Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Shikuan Hong, Yiwu Yao, Gongyi Wang
   - **Summary**: RazorAttention introduces a compression technique that preserves all token information by maintaining a full cache for crucial retrieval heads and discarding remote tokens in non-retrieval heads. It includes a "compensation token" mechanism to recover information from dropped tokens, achieving over 70% reduction in KV cache size without noticeable performance impact.
   - **Year**: 2024

4. **Title**: UNComp: Uncertainty-Aware Long-Context Compressor for Efficient Large Language Model Inference (arXiv:2410.03090)
   - **Authors**: Jing Xiong, Jianghan Shen, Fanghua Ye, Chaofan Tao, Zhongwei Wan, Jianqiao Lu, Xun Wu, Chuanyang Zheng, Zhijiang Guo, Lingpeng Kong, Ngai Wong
   - **Summary**: UNComp presents an uncertainty-aware compression scheme that leverages matrix entropy to estimate model uncertainty across layers and heads at the token sequence level. By adaptively compressing both hidden states and the KV cache, it achieves significant speedup and compression with minimal performance loss.
   - **Year**: 2024

**Key Challenges**:

1. **Balancing Compression and Performance**: Achieving significant KV cache compression without degrading model performance remains a critical challenge.

2. **Task-Specific Adaptation**: Developing compression methods that adapt to the unique demands of different tasks is essential for maintaining efficiency and effectiveness.

3. **Preserving Crucial Information**: Ensuring that compression techniques do not discard important tokens or information necessary for accurate inference is a significant hurdle.

4. **Computational Overhead**: Implementing compression methods that do not introduce additional computational overhead during inference is vital for practical deployment.

5. **Scalability**: Designing compression strategies that scale effectively with increasing model sizes and longer contexts is a persistent challenge in the field. 