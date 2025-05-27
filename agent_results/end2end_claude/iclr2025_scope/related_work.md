1. **Title**: DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs (arXiv:2412.14838)
   - **Authors**: Xiabin Zhou, Wenbin Wang, Minyan Zeng, Jiaxian Guo, Xuebo Liu, Li Shen, Min Zhang, Liang Ding
   - **Summary**: This paper introduces DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to specific tasks. By establishing global and per-layer maximum KV cache budgets and periodically updating KV cache sizes during inference, DynamicKV retains only 1.7% of the KV cache size while achieving approximately 85% of the full KV cache performance on LongBench.
   - **Year**: 2024

2. **Title**: MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference (arXiv:2502.17599)
   - **Authors**: Zhongwei Wan, Hui Shen, Xin Wang, Che Liu, Zheda Mai, Mi Zhang
   - **Summary**: MEDA proposes a dynamic layer-wise KV cache allocation method for efficient multimodal long-context inference. Utilizing cross-modal attention entropy to determine KV cache size at each layer, MEDA achieves up to 72% KV cache memory reduction and 2.82 times faster decoding speed, while maintaining or enhancing performance on various multimodal tasks in long-context settings.
   - **Year**: 2025

3. **Title**: KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse (arXiv:2502.16002)
   - **Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang
   - **Summary**: KVLink introduces an approach for efficient KV cache reuse in large language models by precomputing the KV cache of each document independently and concatenating them during inference. This method reduces redundant computation, improving question answering accuracy by an average of 4% over state-of-the-art methods and reducing time-to-first-token by up to 90% compared to standard LLM inference.
   - **Year**: 2025

4. **Title**: RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression (arXiv:2502.14051)
   - **Authors**: Payman Behnam, Yaosheng Fu, Ritchie Zhao, Po-An Tsai, Zhiding Yu, Alexey Tumanov
   - **Summary**: RocketKV presents a training-free KV cache compression strategy designed to reduce memory bandwidth and capacity demand during the decode phase. It employs a two-stage process: coarse-grain KV cache eviction with SnapKV++ and fine-grain top-k sparse attention. RocketKV provides end-to-end speedup by up to 3× and peak memory reduction by up to 31% in the decode phase on an NVIDIA H100 GPU, with negligible accuracy loss on various long-context tasks.
   - **Year**: 2025

5. **Title**: ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens
   - **Authors**: Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen
   - **Summary**: This paper introduces ∞Bench, a benchmark designed to evaluate language models' performance on contexts exceeding 100,000 tokens. It provides a comprehensive assessment of models' abilities to handle extremely long contexts, highlighting the challenges and limitations in current long-context processing methods.
   - **Year**: 2024

6. **Title**: ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding
   - **Authors**: Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, Omer Levy
   - **Summary**: ZeroSCROLLS presents a zero-shot benchmark aimed at evaluating language models' capabilities in understanding and processing long texts without task-specific fine-tuning. It emphasizes the importance of efficient long-context comprehension and identifies areas where current models fall short.
   - **Year**: 2023

7. **Title**: LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks
   - **Authors**: Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang
   - **Summary**: LongBench v2 offers an updated benchmark focusing on realistic long-context multitasks, aiming to assess models' deeper understanding and reasoning abilities over extended contexts. It provides insights into the effectiveness of various approaches in handling long-context scenarios.
   - **Year**: 2025

8. **Title**: RULER: What's the Real Context Size of Your Long-Context Language Models?
   - **Authors**: Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh
   - **Summary**: RULER investigates the actual context size that long-context language models can effectively utilize. By analyzing models' performance across different context lengths, it sheds light on the practical limitations and potential areas for improvement in long-context processing.
   - **Year**: 2024

9. **Title**: Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?
   - **Authors**: Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua, Devendra Singh Sachan
   - **Summary**: This paper explores whether long-context language models can effectively replace traditional retrieval, retrieval-augmented generation (RAG), and SQL-based methods. It evaluates the models' capabilities in handling tasks that typically require external information retrieval, highlighting their strengths and weaknesses.
   - **Year**: 2024

10. **Title**: A Benchmark for Learning to Translate a New Language from One Grammar Book
    - **Authors**: Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, Luke Melas-Kyriazi
    - **Summary**: This work presents a benchmark designed to assess language models' ability to learn and translate a new language using only a single grammar book. It emphasizes the challenges in adapting models to new languages with limited resources, relevant to efficient long-context understanding.
    - **Year**: 2023

**Key Challenges:**

1. **Memory and Computational Constraints**: Managing the KV cache for long-context processing imposes significant memory and computational demands, limiting the scalability and efficiency of language models.

2. **Task-Specific Adaptation**: Developing adaptive KV cache management strategies that cater to the unique requirements of different tasks remains a complex challenge, as fixed patterns may not capture task-specific characteristics.

3. **Maintaining Performance with Compression**: Achieving substantial KV cache compression without degrading model performance is difficult, as aggressive compression can lead to loss of essential contextual information.

4. **Efficient Multimodal Processing**: Handling long-context scenarios in multimodal settings introduces additional complexity, requiring models to effectively manage and integrate information across different modalities.

5. **Benchmarking and Evaluation**: Establishing comprehensive benchmarks to evaluate models' long-context understanding and processing capabilities is essential but challenging, as it requires realistic and diverse tasks that reflect real-world applications. 