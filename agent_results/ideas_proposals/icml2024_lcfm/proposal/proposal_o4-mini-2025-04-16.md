Title  
Attention-Guided Dynamic KV Cache Compression for Efficient Long-Context Inference  

1. Introduction  
Background  
Autoregressive long-context foundation models (LCFMs) like GPT-3, LLaMA, and their successors achieve remarkable performance on tasks requiring the synthesis of information over thousands to millions of tokens. During inference, these models maintain a Key-Value (KV) cache of all past hidden states to compute self-attention efficiently. However, the KV cache grows linearly with sequence length, leading to prohibitive memory consumption and inference latency—especially on resource-constrained hardware or edge devices. Existing uniform compression methods (e.g., static quantization or periodic pruning) risk discarding context that is critical for long-range dependencies, degrading perplexity or downstream task performance.  

Research Objectives  
This proposal aims to develop an Attention-Guided Dynamic KV Cache Compression (AG-DKC) framework that:  
• Adapts compression strength at the granularity of tokens or token blocks, guided by historical attention patterns.  
• Retains high-fidelity representations for tokens frequently attended by future queries, while aggressively compressing less salient tokens.  
• Minimizes memory footprint and inference latency with minimal impact on perplexity and task accuracy.  

Significance  
By prioritizing memory and compute resources toward contextually salient information, AG-DKC has the potential to:  
• Enable LCFM inference on devices with limited memory.  
• Extend practical maximum context lengths without horizontal scaling of hardware.  
• Provide a general compression framework applicable across model architectures and tasks.  

2. Methodology  
Our approach comprises four key components: attention history aggregation, importance scoring, adaptive compression, and rigorous evaluation.  

2.1 Data Collection and Preprocessing  
We will evaluate AG-DKC on a suite of benchmark corpora and tasks commonly used to measure long-context performance:  
• LongBench (Chen et al., 2024): question answering, summarization, and coreference resolution over documents of length 16K–100K tokens.  
• PG-19 (Rae et al., 2020): long-range language modeling on classic literature.  
• ArXiv & GitHub code spans: code completion tasks over large codebases.  

Preprocessing steps:  
1. Tokenization using the model’s native byte-pair encoding (BPE) or SentencePiece vocabulary.  
2. Partition each document into overlapping windows of length L (e.g., 8K, 16K, 32K).  
3. During inference, maintain standard autoregressive generation, storing past key and value matrices.  

2.2 Attention History Aggregation  
For each past token index $j \in \{1,\dots,t-1\}$ at layer $\ell$, we record the cumulative attention weight it has received from all subsequent queries up to time $t$. Let  
$$\alpha_{q\to j}^{(\ell)} = \text{attention weight from query position }q\text{ to key }j\text{ in layer }\ell.$$  
We define the normalized historical attention score:  
$$H_j^{(\ell)}(t) = \frac{1}{t-1}\sum_{q=j+1}^{t} \alpha_{q\to j}^{(\ell)}\,. $$  
We then aggregate across layers by a weighted sum or max:  
$$S_j(t) = \max_{\ell}\bigl\{H_j^{(\ell)}(t)\bigr\}\quad\text{or}\quad S_j(t)=\sum_\ell w_\ell\,H_j^{(\ell)}(t)\,,$$  
where $w_\ell$ can reflect the relative importance of layer $\ell$.  

2.3 Adaptive Compression Decision  
Given importance scores $S_j(t)$, we divide tokens into $K$ importance tiers via thresholding:  
• Tier 1 (high importance): $S_j(t)\ge\tau_{K-1}$  
• Tier 2: $\tau_{K-2}\le S_j(t)<\tau_{K-1}$  
…  
• Tier K (low importance): $S_j(t)<\tau_1$  

Thresholds $\{\tau_k\}$ may be set by:  
• Fixed quantiles (e.g., top 20% as Tier 1), or  
• Learned during a short calibration phase using held-out data.  

Each tier $k$ is assigned a compression configuration $(b_k,p_k,e_k)$:  
• $b_k$: number of quantization bits (e.g., 8 → 4 → 2 bits for tiers 1 → 3).  
• $p_k$: pruning probability (fraction of tokens in tier $k$ to drop entirely).  
• $e_k$: eviction schedule (after how many time-steps to evict and re-prefill from uncompressed KV, if at all).  

2.4 Compression Techniques  
We combine three complementary compression operations:  

2.4.1 Quantization  
Uniform affine quantization per key/value vector. For a selected token block $j$ with original vector $v_j\in\mathbb{R}^d$, we compute  
$$\hat v_j = \mathrm{round}\Bigl(\frac{v_j - \min(v_j)}{\Delta_k}\Bigr)\cdot \Delta_k + \min(v_j),\quad \Delta_k=\frac{\max(v_j)-\min(v_j)}{2^{b_k}-1}\,. $$  

2.4.2 Pruning  
With probability $p_k$, we drop entire key/value pairs—corresponding query attention falls back to caching earlier residual representations or approximate via neighborhood interpolation.  

2.4.3 Controlled Eviction & Re-prefill  
For tokens with persistently low $S_j(t)$, we may evict their KV entries after $e_k$ steps, forcing the model to re-prefill that part of context with fresh uncompressed computation if needed.  

2.5 Algorithmic Pseudocode  

```
Input: Sequence length T, model layers L, importance tiers K, thresholds {τ_k}, compression configs {(b_k,p_k,e_k)}
Initialize: empty KV_cache, attention_history H_j^{(ℓ)}=0 ∀ j,ℓ
For t in 1..T:
  1. Compute next token via self-attention over current KV_cache.
  2. For each layer ℓ and each past index j:
       Record α_{t→j}^{(ℓ)}, update H_j^{(ℓ)}(t).
  3. Compute S_j(t) for all j.
  4. Assign each token j to tier k where τ_{k-1} ≤ S_j(t) < τ_k.
  5. For tokens in each tier k:
       Quantize KV_cache[j] to b_k bits.
       With probability p_k, prune KV_cache[j].
       If (t - last_access_j) ≥ e_k, evict KV_cache[j].
  6. Append new key/value to KV_cache uncompressed.
```

2.6 Implementation Details  
• Framework: We will implement AG-DKC as a PyTorch extension compatible with HuggingFace Transformers.  
• Mixed Precision: Compression operations are applied on FP16 keys/values, quantizing down to INT8/INT4.  
• Attention Recording Overhead: We accumulate only scalar attention weights; additional memory overhead is $O(T)$, negligible compared to KV size $O(T\times d)$.  
• Hyperparameters: Number of tiers $K\in\{2,3,4\}$, thresholds by deciles, quantization bits from $\{2,4,8\}$.  

2.7 Experimental Design and Evaluation Metrics  
Baselines  
• Full KV cache (no compression).  
• FastKV (Jo et al., 2025): token-selective propagation.  
• DynamicKV (Zhou et al., 2024): task-aware layer budgets.  
• KV-Distill (Chari et al., 2025): distilled context adapter.  

Evaluation Tasks & Metrics  
• Language Modeling Perplexity: lower is better.  
• Downstream QA Accuracy / Summarization ROUGE: measure task performance drop.  
• Memory Footprint: peak GPU memory usage during inference.  
• Latency & Throughput: time-to-first-token (TTFT) and tokens/sec.  
• Ablation Studies: impact of number of tiers $K$, choice of aggregation (max vs. sum), and compression configurations $(b_k,p_k,e_k)$.  

Statistical Validation  
• Each experiment repeated 5 times with different random seeds; report mean ± standard deviation.  
• Significance testing (paired t-tests) to compare AG-DKC against baselines.  

3. Expected Outcomes & Impact  
We anticipate that AG-DKC will:  
• Reduce KV cache size by 50–75% with less than 1.0% increase in perplexity on long-context language modeling tasks.  
• Improve inference throughput by 1.5×–2.5× and reduce TTFT by up to 30%.  
• Maintain downstream QA and summarization performance within 2% of full-cache baselines.  
• Generalize across multiple LCFM architectures (GPT-2, LLaMA-2) and a diverse set of tasks (language modeling, QA, summarization, code completion).  

Broader Impact  
By significantly lowering memory and compute barriers, AG-DKC can enable deployment of powerful LCFMs:  
• On edge devices (e.g., mobile, embedded systems) for on-device understanding of large documents.  
• In cloud settings where reduced memory translates directly into cost savings.  
• As a building block for future research on selective attention and resource-aware transformer architectures.  

We will open-source our implementation, detailed benchmarks, and configuration recipes to facilitate adoption by both academic and industrial practitioners.