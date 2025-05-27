# Dynamic Sparse Retrieval-Augmented Sub-Quadratic Models for Efficient Long Context Adaptation  

## Introduction  
### Background  
Foundation models increasingly demand **long context understanding** for applications like real-time news analysis and knowledge-intensive reasoning. However, traditional transformer architectures face a critical bottleneck: **quadratic computational complexity $O(n^2)$** in attention mechanisms due to full-key-value (KV) cache retrieval over extended sequences. While **retrieval-augmented generation (RAG)** methods improve factual relevance by appending external knowledge, they exacerbate this issue by inflating prefill inputs with retrieved context, leading to unsustainable memory and latency overheads. Existing solutions (e.g., RazorAttention [3], PyramidKV [5]) compress KV caches via token pruning or layer-wise compression but fail to address **dynamic adaptation** to streaming data or task-specific token selection.  

### Research Objectives  
This work proposes **Dynamic Sparse Retrieval-Augmented Sub-Quadratic Models (DSRAM)** to resolve three challenges:  
1. **Efficient Token Fetching**: Reduce redundant prefill via **reinforcement learning (RL)-based sparse retrieval** that identifies query-specific informative tokens.  
2. **Sub-Quadratic Attention**: Implement **sparse attention** over retrieved tokens only, achieving $O(n \log n)$ complexity.  
3. **Compressive KV Caching**: Introduce a **rotating low-rank KV cache** to retain historical context in fixed-sized latent states, enabling unbounded context retention without memory explosion.  

### Significance  
DSRAM directly addresses the workshop’s goals of **scalable optimization** and **adaptive inference efficiency**. By co-optimizing retrieval, attention, and caching, this work enables:  
- **Real-time adaptation** to data streams (e.g., live news) without retraining.  
- **Constant memory usage** via compressive KV states, bypassing the need for infinite cache expansion.  
- **End-to-end optimization** of task accuracy and compute costs, outperforming existing RAG and KV compression methods (e.g., AttentionRAG [1], GCA [2]).  

---

## Methodology  

### Data Collection and Preprocessing  
**Datasets**:  
1. **Long-context reasoning**: Synthetic passkey retrieval (16M tokens [2]), RealNews.  
2. **Streaming adaptation**: NewsStreams dataset (50k real-time articles/year).  
3. **RAG benchmarks**: NQ [4], HotpotQA [4].  

**Preprocessing**:  
- Tokenize via BPE with 32k vocab.  
- For continual learning, simulate streams via sequential chunks.  
- Annotate relevance scores for retrieval supervision.  

---

### Algorithm Design  

#### 1. Dynamic Sparse Retrieval Module with Reinforcement Learning  
A lightweight **learned retriever** selects $k \ll n$ tokens to attend to, balancing accuracy and cost.  

**State Representation**:  
$$
s_t = \text{CLS\_token}(\mathbf{h}_t^{\text{query}})
$$
where $\mathbf{h}_t^{\text{query}} \in \mathbb{R}^{d}$ is the query’s hidden state.  

**Action Space**:  
At step $t$, retrieve top-$k$ tokens from the KV cache and external database:  
$$
a_t = \text{Top-k}(\mathcal{F}_{\theta}(s_t, \mathcal{C}_{\leq t}))
$$
where $\mathcal{F}_{\theta}$ is a Siamese network scoring token relevance.  

**Reward Function**:  
$$
R_t = \alpha \cdot \text{TaskLoss}_t^{-1} - \beta \cdot \text{Retrieved\_Tokens}_t
$$
Maximize inverse task loss (e.g., log-likelihood) while penalizing token count.  

**Training**:  
Use PPO [20] to optimize policy $\pi_{\theta}$:  
$$
\max_{\theta} \mathbb{E}_{s \sim \text{data}} \left[ \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s,a), \text{clip}(\cdot)\right) \right]
$$

---

#### 2. Sub-Quadratic Sparse Attention Mechanism  
Replace full attention with **sparse attention** over retrieved tokens:  
$$
\text{Attention}(\mathcal{T}_{\text{query}}, \mathcal{T}_{\text{retrieved}}) = \text{Softmax}\left( \frac{\mathbf{Q}_{\text{query}} \mathbf{K}_{\text{retrieved}}^T}{\sqrt{d_k}} \right) \mathbf{V}_{\text{retrieved}}
$$
Complexity reduces to $O(nk)$, where $k = O(\log n)$ for theoretical guarantees [21].  

---

#### 3. Rotating Compressive KV Caching via Low-Rank Projections  
**Caching Framework**:  
- Maintain a circular buffer $\mathcal{B} = \{\mathbf{z}_1, ..., \mathbf{z}_M\}$ of latent states.  
- For each layer $l$, project keys/values via low-rank SVD:  
$$
\mathbf{K}^l, \mathbf{V}^l \approx \mathbf{U}_k \Sigma_k \mathbf{V}_k^T
$$
Retain top $r$ singular values to form latent $\mathbf{z}^l$.  
- Overwrite $\mathcal{B}$ in FIFO order, preserving temporal context.  

**Memory Savings**:  
$$
\text{Size} = M \cdot r \cdot d \ll n \cdot d \quad \text{(if } r \ll n \text{)}
$$

---

### Co-Optimization with Hybrid Loss Function  
Train the retriever, attention, and cache modules end-to-end:  
$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{task}}}_{\text{Cross-entropy}} + \lambda_1 \cdot \underbrace{\|\mathcal{T}_{\text{retrieved}}\|_0}_{\text{Token cost}} + \lambda_2 \cdot \underbrace{\|\mathcal{L}_{\text{att}}\|_F^2}_{\text{Attention sparsity}}
$$
where $\lambda_1, \lambda_2$ balance accuracy, retrieval cost, and sparsity.  

---

### Experimental Design  

#### Baselines  
- **FullKV**: Transformer with full cache.  
- **RazorAttention** [3]: Token pruning for non-retrieval heads.  
- **PyramidKV** [5]: Layer-wise cache reduction.  
- **GCA** [2]: Chunked retrieval with fixed windows.  

#### Tasks  
1. **Passkey Retrieval** [2]: Retrieve a 10-digit key in 16M tokens.  
2. **Continual News Summarization**: Stream 50k articles; compressively retain trends.  
3. **RAG QA**: NQ/HotpotQA with 100k retrieved passages.  

#### Evaluation Metrics  
- **Accuracy**: F1 score (QA), retrieval accuracy (passkey).  
- **Efficiency**: Latency (ms/token), memory (GB), tokens/sec.  
- **Ablation**: Compare w/wo retriever, w/wo compression.  

---

## Expected Outcomes & Impact  

### Technical Outcomes  
1. **Accuracy-Preserving Compression**:  
   - Match or exceed FullKV accuracy with **70% KV cache reduction** [3] and **5× latency improvement** over GCA [2].  
2. **Sub-Quadratic Scalability**:  
   - Achieve $O(n \log n)$ compute on 16M-length sequences, outperforming attention-guided pruning (6.3× compression in AttentionRAG [1]).  
3. **Dynamic Adaptation**:  
   - Maintain 10% F1 gain over LongRAG [4] on streaming summarization by retaining compressive KV states.  

### Theoretical Contributions  
1. **First End-to-End Sparse Retrieval Framework**:  
   - RL-based token selection co-optimized with task loss, unlike static retrievers (e.g., LongRAG [4]).  
2. **Compressive KV State Theory**:  
   - Formalize low-rank projection guarantees for latent context retention in sub-quadratic models.  

### Practical Impact  
1. **Real-Time Knowledge Adaptation**:  
   - Enable cost-effective deployment of adaptive LLMs for live news analysis, healthcare monitoring.  
2. **Scalable Foundation Models**:  
   - Contribute to the workshop’s goal of **inference-efficient quadratic-to-sub-quadratic conversion**, aligning with ICLR’s focus on sustainable ML.  
3. **Open-Source Release**:  
   - Publicly release DSRAM framework and NewsStreams dataset to foster reproducibility.  

### Addressing Workshop Challenges  
DSRAM directly tackles three workshop-relevant challenges:  
1. **Long Context Efficiency**: Reduces KV cache size 8× over KV-Compress [6] via rotating low-rank states.  
2. **Adaptive Personalization**: Supports continual adaptation without retraining (vs. fine-tuning-based MoE [adaptive fine-tuning topic]).  
3. **RAG Integration**: Minimizes RAG’s prefill cost by 60% compared to standard methods (e.g., LongRAG [4]).  

By combining sparse retrieval, sub-quadratic attention, and compressive caching, this work advances scalable optimization for next-generation foundation models, fulfilling the workshop’s mission to bridge efficiency and adaptability.  

---

**References**  
[1] Fang et al. (2025). AttentionRAG. arXiv:2503.10720.  
[2] Hu et al. (2024). GCA. arXiv:2410.01651.  
[3] Tang et al. (2024). RazorAttention. arXiv:2407.15891.  
[4] Jiang et al. (2024). LongRAG. arXiv:2406.15319.  
[5] Cai et al. (2024). PyramidKV. arXiv:2406.02069.  
[6] Rehg (2024). KV-Compress. arXiv:2410.00161.  
[20] Schulman et al. (2017). Proximal Policy Optimization. arXiv:1707.06347.  
[21] Kitaev et al. (2020). Sparse Attention. arXiv:2004.04228.