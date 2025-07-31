1 Title  
Dynamic Compressive Memory with Adaptive Routing for Sub-Quadratic Long-Context Foundation Models

2 Introduction  
Background  
Contemporary transformer‐based foundation models excel at a wide range of tasks but incur quadratic complexity in attention when processing long contexts or streaming inputs. As context length $N$ grows, both compute ($O(N^2)$) and memory (KV cache size $O(N)$ for keys and values) scale poorly. This bottleneck limits deployment in scenarios requiring ultra-long contexts—such as continuous news summarization, multi-turn dialogue spanning thousands of tokens, and long-form question answering—where real-time inference and memory efficiency are critical. Recent work on sub-quadratic attention (sparse, kernel-based) and mixture-of-experts (MoE) addresses parts of this challenge but often treats memory compression and routing separately.

Research Objectives  
This proposal aims to design and evaluate an end-to-end trainable framework that enables:  
- Constant-size memory under streaming or long-context settings via differentiable compressive summaries of past KV pairs.  
- Adaptive routing of each query to raw KV, compressed “super-KV” clusters, or an external retrieval store via a lightweight gating network.  
- Joint optimization of compression fidelity, routing efficiency, and retrieval relevance to ensure near-lossless accuracy while achieving sub-quadratic compute and memory costs.  

Significance  
By integrating compressive memory with a dynamic router, our method promises:  
- Sub-quadratic attention complexity $O(NM + M^2)$ with fixed $M \ll N$.  
- 2–5× speedups and 60–80% memory savings on 16k–32k contexts.  
- Seamless adaptation to incoming data streams and retrieval-augmented generation (RAG).  
This work addresses critical gaps in efficient long‐context understanding and personalized adaptation for foundation models, advancing both theoretical foundations and practical deployments.

3 Related Work  
Mixture-of-Experts (MoE) and Routing  
- MoxE (Thiombiano et al., 2025) introduces entropy-aware routing among xLSTM experts to improve efficiency; our gating draws inspiration from its entropy regularizer but operates over memory banks rather than RNN experts.  
- SMILE (He et al., 2022) uses bi-level routing for throughput gains; we extend the idea by routing at inference time between raw KV, compressed KV, and retrieval.  
- Default MoE (2025) stabilizes sparse MoE training with dense backprop; our end-to-end training similarly balances gradient flow across compression and routing modules.

Memory Compression and Summarization  
- Compressive Transformers (Dai et al., 2019) first compress old memories via fixed heuristics. Our approach replaces heuristics with differentiable online clustering, allowing learned summarization.  
- ResMoE (Ai et al., 2025) compresses MoE parameters via residual experts; we compress activations (KV pairs), a complementary direction.

Sparse/Sub-Quadratic Attention  
- Routing Transformer (2022) selects content-based sparse attention patterns. Our method generalizes sparsity by grouping KV pairs into $M$ prototypes, reducing pairwise attention to $O(NM)$.  
- Compressive KV (2021) and kernel-based attention reduce memory but lack adaptive selection; our gating network dynamically chooses which memory source to attend.

Retrieval-Augmented Methods  
- RAG (Lewis et al., 2020) integrates retrieval during generation; we incorporate a retrieval store as one routing destination, trading off between local compressed memory and external knowledge.

4 Methodology  
Overview  
Our framework consists of three components:  
- Differentiable online clustering module that compresses the growing KV cache into $M$ super-KV pairs.  
- Adaptive routing network that directs each query to one of three sources: raw KV, compressed super-KV, or an external retrieval store.  
- Joint loss combining task supervision, compression fidelity, and retrieval relevance.

4.1 Data Collection and Preprocessing  
We target three long-context tasks:  
- Streaming Summarization: arXiv Bulk Summarization (arXiv papers split into 8k+ token streams).  
- Long-Form QA: NarrativeQA and ELI5 extended with 16k‐token contexts.  
- Multi-Turn Dialogue: MultiWOZ augmented with system logs of length 10k+ tokens.

Input streams are segmented at sliding windows of length $L$ (e.g., 4k tokens) with overlap $o$ (e.g., 512 tokens) to simulate streaming. Standard tokenization (Byte-Pair Encoding) ensures consistent vocabulary. We precompute retrieval index embeddings using DPR (Karpukhin et al., 2020) over external knowledge sources (Wikipedia, news corpus).

4.2 Compressive Clustering of KV Pairs  
At time step $t$, let the accumulated KV cache be  
$$\mathcal{K}_t = \{(K_i, V_i)\}_{i=1}^{N_t},$$  
where $N_t$ grows linearly. We maintain $M$ cluster centers $\{(\bar K_j, \bar V_j)\}_{j=1}^M$. We compute soft assignments $A_{ij} = \mathrm{softmax}_j\bigl(\langle K_i, W_c K_j^{\mathrm{init}}\rangle\bigr)$, where $W_c$ is a learned projection and $K_j^{\mathrm{init}}$ are learnable prototypes.

Cluster updates:  
Block equation for new cluster centers:  
$$  
\bar K_j \leftarrow \frac{\sum_{i=1}^{N_t} A_{ij}\,K_i}{\sum_{i=1}^{N_t} A_{ij}},\quad  
\bar V_j \leftarrow \frac{\sum_{i=1}^{N_t} A_{ij}\,V_i}{\sum_{i=1}^{N_t} A_{ij}}.  
$$  
This operation is efficient $O(N_t M)$ and yields constant memory $O(M)$.

4.3 Adaptive Routing Network  
For each query vector $q\in\mathbb{R}^d$, we compute a gating distribution over three sources:  
$$g = \mathrm{softmax}\!\bigl(W_g q + b_g\bigr)\in\mathbb{R}^3.$$  
Denote $g_1,g_2,g_3$ as weights for raw KV, compressed super-KV, and retrieval. Attention outputs:  
- Raw KV attention: $\mathrm{Attn}_{\mathrm{raw}}(q) = \sum_{i=1}^{N_t} \alpha_i(q)\,V_i$, where $\alpha(q)=\mathrm{softmax}(K_i^\top q/\sqrt{d})$.  
- Compressed KV attention: $\mathrm{Attn}_{\mathrm{comp}}(q) = \sum_{j=1}^M \beta_j(q)\,\bar V_j$, where $\beta(q)=\mathrm{softmax}(\bar K_j^\top q/\sqrt{d})$.  
- Retrieval attention: retrieve top-$k$ docs via DPR, encode them with the backbone encoder to produce representations $\{U_\ell\}$, then $\mathrm{Attn}_{\mathrm{ret}}(q)=\sum_{\ell=1}^k \gamma_\ell(q)\,U_\ell$.  

Final attended value:  
$$  
\mathrm{Attn}(q)=g_1\,\mathrm{Attn}_{\mathrm{raw}}(q)+g_2\,\mathrm{Attn}_{\mathrm{comp}}(q)+g_3\,\mathrm{Attn}_{\mathrm{ret}}(q).  
$$  

4.4 Loss Functions and Joint Training  
We fine-tune a foundation model (e.g., LLaMA-style) end-to-end with three loss terms:  
- Task loss $L_{\mathrm{task}}$ (cross-entropy for generation or classification).  
- Compression fidelity loss  
  $$L_{\mathrm{comp}} = \frac{1}{N_t}\sum_{i,j} A_{ij}\|\;K_i - \bar K_j\|^2 + \|V_i - \bar V_j\|^2.$$  
- Retrieval relevance loss $L_{\mathrm{ret}}$ (contrastive loss ensuring retrieved docs are relevant).  

Total loss:  
$$  
L = L_{\mathrm{task}} + \lambda_c L_{\mathrm{comp}} + \lambda_r L_{\mathrm{ret}} + \lambda_g H(g),  
$$  
where $H(g)$ is entropy regularizer on gating to avoid degenerate routing, and $\lambda_c,\lambda_r,\lambda_g$ are hyperparameters tuned on validation data.

4.5 Experimental Design and Evaluation Metrics  
Baselines  
- Full attention transformer (quadratic).  
- Compressive Transformer with fixed heuristics.  
- Routing Transformer (sparse attention).  
- RAG-only (no compression).  

Metrics  
- Task performance:  
  - ROUGE-1/2/L for summarization.  
  - Exact match/F1 for QA.  
  - BLEU / ChrF for dialogue.  
- Efficiency:  
  - Inference speed (tokens/sec) on NVIDIA A100.  
  - Memory usage (GB) of KV cache.  
- Scalability: measure above metrics on contexts of length 8k, 16k, 32k.  
- Ablation: vary $M\in\{64,128,256\}$ and gating entropy weight $\lambda_g$.  

Implementation Details  
- Backbone: pretrained transformer of size 1.3B parameters.  
- Optimizer: AdamW, learning rate $3\times10^{-5}$, batch size 16 sequences, train for 20k steps.  
- Retrieval index updated every epoch.  
- Evaluation on held-out splits every 2k steps.

5 Expected Outcomes & Impact  
Expected Outcomes  
- Efficiency Gains: 2–5× speedups relative to full attention on 16k–32k contexts.  
- Memory Savings: 60–80% reduction in KV cache footprint at comparable performance.  
- Accuracy Retention: within 1–2% absolute drop (or even improvement via retrieval) in ROUGE/F1/BLEU versus full baseline.  
- Adaptive Behavior: learned gating will route most queries to compressed memory (>70%) and only fallback to raw KV for critical tokens, verified via routing statistics.

Broader Impact  
- Real-Time Deployment: enables streaming applications (news summarization, live dialogue) with bounded memory and latency.  
- Personalized Adaptation: compressive memory can store user-specific patterns in compact form, facilitating privacy-preserving on-device adaptation.  
- Foundation for Future Research: the integration of compression, routing, and retrieval in one framework opens new avenues for continual learning, sub-quadratic transformers, and hybrid memory architectures.  

By delivering a unified solution to efficient long-context processing, this work will significantly advance the state of the art in foundation model inference and adaptation for both academia and industry.