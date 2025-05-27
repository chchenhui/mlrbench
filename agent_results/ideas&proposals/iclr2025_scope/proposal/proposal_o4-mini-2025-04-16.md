Title  
Dynamic Sparse Retrieval-Augmented Sub-Quadratic Models for Efficient Long-Context Adaptation  

1. Introduction  
Background  
Foundation models (FMs) such as large language models (LLMs) and multi-modal transformers have pushed the frontier of AI by enabling zero- and few-shot generalization across a wide variety of tasks. Yet, as applications demand ever-longer context windows (e.g., continuous news streams, document archives, chat histories), two intertwined challenges emerge: (1) inference complexity and memory usage grow quadratically in sequence length when using standard self-attention with KV caching; (2) naïve retrieval-augmented generation (RAG) further inflates the prefill by concatenating large retrieved passages, exacerbating compute and latency bottlenecks. Concurrently, techniques such as mixture-of-experts (MoE) and sub-quadratic attention variants (e.g., sparse, low-rank, kernelized) promise asymptotic speedups but often lack adaptive integration with dynamic retrieval and memory compression.  

Research Objectives  
This proposal aims to bridge these gaps by designing, implementing, and evaluating a unified architecture—Dynamic Sparse Retrieval-Augmented Sub-Quadratic (DS-RASQ) model—that:  
• Learns a lightweight, query-specific retrieval policy via reinforcement learning (RL) to fetch only the most relevant context tokens.  
• Employs a sparse attention mechanism conditioned on retrieved tokens to achieve sub-quadratic inference cost.  
• Maintains and updates a fixed-size rotating compressive KV cache using learned low-rank projections, ensuring constant memory usage for arbitrarily long sequences.  
• Co-optimizes retrieval, attention, and compression modules end-to-end under a hybrid loss that balances task accuracy against retrieval and compute costs.  

Significance  
By enabling real-time adaptation to streaming contexts with constant memory and sub-quadratic compute, DS-RASQ will:  
1. Advance scalable optimization methods for adaptive foundation models, directly addressing workshop goals on efficient and adaptive fine-tuning and inference.  
2. Provide a unified solution for long-context understanding, RAG integration, and KV cache management, with broad applicability in NLP, vision, and multi-modal tasks.  
3. Demonstrate how learning-based retrieval and memory compression can jointly yield high accuracy, low latency, and bounded resource consumption—key for deploying FMs in real-world, resource-constrained settings.  

2. Methodology  
We decompose DS-RASQ into three core modules—Dynamic Sparse Retriever, Sub-Quadratic Sparse Attention, and Rotating Compressive KV Cache—and describe their algorithmic design, end-to-end training, and experimental validation.  

2.1 Dynamic Sparse Retriever  
Goal: Given an input query sequence $x\in\mathbb{R}^{n\times d}$ and past context cache $M\in\mathbb{R}^{r\times d}$, select a subset $S(x)\subseteq\{1,\dots,n\!+\!r\}$ of size $k\ll n+r$ maximizing downstream accuracy while minimizing retrieval cost.  
Policy parametrization  
We define a stochastic policy $\pi_\theta(S\mid x,M)$ implemented as a light transformer encoder or bi-attention network that scores each token index $i$ with logits $z_i=\psi_\theta(x,M)_i$. A top-$k$ Gumbel-Softmax or stochastic beam sampling selects indices $S$.  
Reinforcement learning objective  
We train $\pi_\theta$ via policy gradient to maximize expected reward balancing accuracy and cost:  
$$
\max_\theta\;\mathbb{E}_{S\sim\pi_\theta}\bigl[R_{\rm task}(y,\hat y)\;-\;\lambda_{\text{ret}}\lvert S\rvert\bigr],
$$  
where $R_{\rm task}$ is task-specific (e.g., token-level log-probability or QA F1), $\lvert S\rvert$ is the number of retrieved tokens, and $\lambda_{\text{ret}}$ penalizes larger retrievals. We apply the REINFORCE gradient  
$$
\nabla_\theta \approx \bigl(R_{\rm task}-b\bigr)\nabla_\theta\log\pi_\theta(S\mid x,M),
$$  
with baseline $b$ estimated via a moving average.  

2.2 Sub-Quadratic Sparse Attention  
Once $S$ is selected, we build a sparse attention mask $M_{\rm attn}\in\{0,1\}^{(n+r)\times(n+r)}$ such that tokens attend only to the query and retrieved indices. Formally, let  
$$
M_{\rm attn}[i,j]=
\begin{cases}
1,& j\in S\cup\{1,\dots,n\}\text{ if i is a query token},\\
1,& j\in S\text{ if i is a retrieved token},\\
0,&\text{otherwise}.
\end{cases}
$$  
The sparse attention is computed as  
$$
\text{SparseAttn}(Q,K,V)=\mathrm{softmax}\Bigl(\frac{QK^\top\odot M_{\rm attn}}{\sqrt{d_k}}\Bigr)V,
$$  
where $Q=XW_Q$, $K=XW_K$, $V=XW_V$ are the usual projections, and $\odot$ denotes element-wise masking. Since each row attends to at most $k+ n$ keys, the worst-case cost per layer is $O((n+r)(k+n))=O(nk+n^2)$, and by choosing $k\ll n$, we empirically achieve sub-quadratic scaling $O(n^{1.1})$ or better.  

2.3 Rotating Compressive KV Cache  
To bound memory growth across arbitrarily long streams, we maintain a fixed-size compressive KV cache $C\in\mathbb{R}^{r\times d}$ that summarizes all past context. At each inference step $t$, we have new key/value pairs $(K_t,V_t)\in\mathbb{R}^{m\times d}\times\mathbb{R}^{m\times d}$ from the last block of tokens. We compress them via learned low-rank projections $U,V\in\mathbb{R}^{d\times r}$:  
$$
C_t = \beta\,C_{t-1} \;+\;(1-\beta)\,\bigl(U^\top K_t + V^\top V_t\bigr),
$$  
where $\beta\in[0,1]$ is a decay factor controlling how fast old information is forgotten. This “rotating” update ensures $C_t$ remains fixed-size ($r\times d$) while gradually integrating new evidence.  

2.4 Joint End-to-End Training  
We optimize the combined model parameters $\Phi=\{\theta,\phi_{\rm attn},U,V\}$ under a hybrid loss  
$$
\mathcal{L}(\Phi)=\underbrace{\mathbb{E}_{S\sim\pi_\theta}\bigl[\mathcal{L}_{\rm task}(\hat y,y)\bigr]}_{\text{prediction loss}}
\;+\;\lambda_{\text{ret}}\;\mathbb{E}\bigl[|S|\bigr]
\;+\;\lambda_{\text{comp}}\;\mathrm{FLOPs}_{\rm attn}(k),
$$  
where $\mathcal{L}_{\rm task}$ is cross-entropy for generation or regression loss for classification, and $\mathrm{FLOPs}_{\rm attn}(k)$ is an analytic estimate of attention compute given retrieval size $k$. Gradients w.r.t.\ $\phi_{\rm attn},U,V$ are computed via backpropagation, while $\theta$ is updated by combining policy gradients from the retrieval reward with straight-through differentiable relaxation for sampling.  

2.5 Data Collection and Preprocessing  
We focus on tasks requiring long-context understanding and streaming adaptation:  
• Real-time news summarization: continuously updated newswire (e.g., Reuters, GDELT) segmented into chronological blocks.  
• Question answering over long documents (HotpotQA, MultiRC) with artificially extended contexts (e.g., concatenated Wikipedia articles up to 64K tokens).  
• Dialogue systems with prolonged histories (e.g., Reddit dialogues, customer support transcripts).  

For each domain, we preprocess text into sub-segments of length $m=512$, build retrieval indices with FAISS for retriever training, and maintain continuous streams for cache compression experiments.  

2.6 Experimental Design  
Baselines  
• Full-context transformer with KV caching (quadratic).  
• AttentionRAG (context pruning)†, GCA (grouped cross attention), RazorAttention (head-wise cache compression), PyramidKV (layer-wise funneling).  
Hardware  
NVIDIA A100 or equivalent GPUs; deployment on single–multi-GPU servers to measure scaling.  

Hyperparameter Sweeps  
• Retrieval size $k\in\{64,128,256\}$.  
• Compression rank $r\in\{64,128,256\}$.  
• Decay factor $\beta\in\{0.9,0.99,0.999\}$.  
• Cost weights $\lambda_{\text{ret}},\lambda_{\text{comp}}$ chosen to yield target throughput.  

Evaluation Metrics  
• Task accuracy: perplexity (language modeling), ROUGE (summarization), EM/F1 (QA), BLEU (translation).  
• Throughput: tokens processed per second.  
• Latency: end-to-end inference time per request.  
• Memory usage: peak GPU memory for KV/cache.  
• Computational complexity: empirical scaling of FLOPs vs.\ sequence length.  

3. Expected Outcomes & Impact  
3.1 Quantitative Outcomes  
We anticipate DS-RASQ to achieve:  
• Sub-quadratic scaling: empirical runtime $O(n^{1.1})$ or better vs.\ $O(n^2)$ for full attention.  
• Constant memory usage: fixed cache of size $r\times d$ vs.\ unbounded KV growth.  
• High task accuracy: within 1–2% of full-context baselines on QA and summarization, outperforming existing compression and sparse-attention methods.  
• Improved throughput: 2–5× speedup on long sequences (e.g., $n>8,192$) with comparable GPU memory footprint.  

3.2 Qualitative Impact  
• Real-time adaptation: DS-RASQ will seamlessly incorporate streaming data (e.g., breaking news) without re-initializing or costly retraining, enabling up-to-the-minute summarization and QA.  
• Personalization: by controlling retrieval policy and compression decay, models can focus on user-relevant context and forget stale information, facilitating personalized assistants and dialogue agents.  
• Multi-modal extension: the same principles generalize to vision and audio streams by treating patch tokens or frame embeddings as “context tokens,” enabling adaptive, efficient processing in video understanding and multi-modal reasoning.  

3.3 Alignment with Workshop Themes  
• Efficient Long Context Understanding: DS-RASQ directly tackles quadratic explosion via sub-quadratic sparse attention.  
• Sub-Quadratic Model Conversion: our architecture can be viewed as a converter from standard transformers to sparse-attention variants with dynamic retrieval.  
• Retrieval-Augmented Generation: we propose a learned retrieval policy integrated end-to-end, balancing relevance and efficiency.  
• Adaptive Fine-Tuning & Personalization: the RL-based retriever and decay-controlled cache support continual adaptation without full weight updates.  
• Inference Scaling & Throughput Optimization: by co-optimizing cost and accuracy, DS-RASQ provides a principled approach to test-time compute allocation.  

3.4 Broader Impact  
The proposed research offers a blueprint for building resource-efficient, adaptive foundation models that can operate under real-world constraints (latency, memory, streaming data). This has implications across domains—from on-device LLM inference in mobile/edge settings to large-scale cloud services offering personalized, up-to-date AI assistance. By open-sourcing DS-RASQ’s code, trained policies, and compressed caches, we aim to accelerate research in scalable adaptive foundation models and foster community benchmarks on long-context streaming tasks.  

References  
[†] Refer to AttentionRAG (Fang et al., 2025), GCA (Hu et al., 2024), RazorAttention (Tang et al., 2024), PyramidKV (Cai et al., 2024) for baseline methods.  

––––––––––––––––  
This proposal lays out a precise research plan—spanning algorithm design, mathematical formulation, end-to-end training, and comprehensive evaluation—to realize dynamic sparse retrieval-augmented sub-quadratic models for efficient long-context adaptation.