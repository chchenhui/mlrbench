Title  
Adaptive Dual-Memory State‐Space Models for Extreme‐Length Sequence Modeling  

1. Introduction  
Background  
Modern sequence models—transformers, recurrent neural networks (RNNs), and state‐space models (SSMs) such as S4, Mamba and their variants—have pushed the frontier of language, vision and biological‐sequence modeling. Yet despite their success on tasks of moderate length (up to 65K tokens), all suffer from a steep degradation in performance, stability or efficiency when faced with truly extreme contexts (100K+ tokens). Transformers’ quadratic attention becomes prohibitively expensive, RNNs struggle with gradient instability, and existing SSMs, while linear in time, cannot reliably “remember” or retrieve the right pieces of information once the sequence grows very long.  

Research Objectives  
This proposal aims to bridge the gap between theoretical memory capacity and practical memory utilization in sequence models by:  
• Designing a novel dual‐memory architecture that combines an SSM core with an external, differentiable working memory cache and a selective long‐term memory store.  
• Developing learnable “memory controllers” that dynamically decide what to store, compress, retrieve or evict from each memory tier based on contextual importance rather than simple recency.  
• Formulating a reinforcement‐learning (RL) objective to train the controllers so that memory management optimizes downstream task performance under tight compute budgets.  
• Evaluating the resulting model on extreme‐length benchmarks (100K–1M tokens) to validate improvements in long‐range dependency capture, reasoning, and computational efficiency.  

Significance  
Solving extreme‐length sequence modeling unlocks new capabilities in document‐level understanding, code reasoning, scientific literature synthesis and genome analysis. By shifting from monolithic attention or pure SSMs to an adaptive, hierarchical memory paradigm, we anticipate orders‐of‐magnitude improvements in both performance and efficiency—paving the way for next‐generation architectures at ICML 2024.  

2. Literature Review  
We briefly review key works and identify open gaps:  
• SMR (Qi et al., 2024) augments SSMs with a learnable replay buffer to stabilize state updates under non‐uniform sampling. It improves SSM stability but does not offer explicit long‐term retrieval.  
• Mamba and selective state spaces (Gu & Dao, 2023) use hardware‐aware parallelism to scale S4 but lack explicit external memory or importance‐based eviction.  
• Graph‐Mamba (Wang et al., 2024) adds node selection in graph sequence modeling but remains confined to structured data and does not dynamically compress past states.  
• Logarithmic Memory Networks (Taha, 2025) reduce footprint via a tree‐structured index for attention but impose rigid hierarchies and are designed for resource‐constrained settings rather than maximal performance on long sequences.  
• Spectral SSMs (Agarwal et al., 2023) learn linear dynamical systems via spectral filtering, offering efficiency but limited by linear memory dynamics.  
• MambaByte (Wang et al., 2024), MoE‐Mamba (Pióro et al., 2024) and Vision Mamba (Zhu et al., 2024) extend selective SSMs to raw‐bytes, mixture‐of‐experts and vision, respectively—but none explicitly handle dual‐tier, importance‐driven memory.  
• Jamba (AI21 Labs, 2024) combines transformers and Mamba at massive scale (52B parameters, 256K context) yet still relies on dense attention over its context window.  

Key Challenges  
1. Retaining and retrieving information across 100K+ tokens  
2. Balancing compute and memory efficiency  
3. Dynamically managing memory based on importance signals  
4. Scaling to diverse domains and sequence structures  

3. Methodology  
We propose the Adaptive Dual‐Memory State‐Space Model (ADM‐SSM), pictured in Figure 1.  

3.1 Model Architecture  
ADM‐SSM integrates three components:  
1. Core SSM Layer  
2. Fast Working Memory (WM) Cache  
3. Selective Long‐Term Memory (LTM) Store  
4. Learnable Memory Controllers  

3.1.1 Core SSM Dynamics  
We build on the structured state‐space layer (S4) for its linear runtime. At time step $t$, given input token embedding $x_t\in\mathbb{R}^d$ and previous hidden state $h_{t-1}\in\mathbb{R}^d$, the SSM update is:  
$$  
h_t = f_{\text{SSM}}(h_{t-1}, x_t)  
$$  
where $f_{\text{SSM}}$ is implemented via convolution in the transform domain as in Gu & Dao (2023).  

3.1.2 Working Memory Cache  
We maintain a fixed‐capacity cache $M^w_t\in\mathbb{R}^{K\times d}$ that holds the last $K$ selected state embeddings. Each slot $i\in[1,K]$ contains $(s^w_{t,i},\,c^w_{t,i})$ where $s^w_{t,i}\in\mathbb{R}^d$ is the stored vector and $c^w_{t,i}\in\mathbb{R}$ is an “importance score.”  

3.1.3 Long‐Term Memory Store  
The LTM $M^\ell_t\in\mathbb{R}^{L\times d}$ holds compressed summaries of older states, with $L\gg K$. Each slot holds $(s^\ell_{t,j},\,\sigma^\ell_{t,j})$ where $\sigma^\ell_{t,j}$ is a compressed importance metric.  

3.1.4 Memory Controllers  
We introduce two small neural modules:  
• Write Controller $g_w$: decides if $h_t$ enters WM.  
• Promote/Evict Controller $g_p$: decides which WM slots to evict to LTM and which LTM to purge.  

Formally, at each time step:  
1. Compute importance signal:  
   $$\alpha_t = \sigma\big(W_\alpha\,h_t + b_\alpha\big)\in(0,1)\,,$$  
   $$\beta_t = \sigma\big(W_\beta\,h_t + b_\beta\big)\in(0,1)\,.$$  
2. WM Write Decision:  
   With probability $\alpha_t$, write $h_t$ into WM by replacing the slot with lowest $c^w_{t-1,i}$. Set  
   $$s^w_{t,i^*} = h_t,\quad c^w_{t,i^*} = \alpha_t.$$  
3. Eviction to LTM:  
   For any WM slot $i$ with $c^w_{t-1,i} < \delta$, evict into LTM:  
   $$s^\ell_{t,j^*} = \text{Compress}(s^w_{t-1,i}),\quad \sigma^\ell_{t,j^*} = c^w_{t-1,i}.$$  
   Here Compress$(\cdot)$ can be a learned autoencoder or PCA.  
4. LTM Purge:  
   Periodically, remove LTM entries with $\sigma^\ell_{t,j}<\epsilon$ to bound memory size.  

3.1.5 Retrieval Mechanism  
To compute the context‐enhanced output $y_t$, we attend over both WM and LTM:  
$$  
r^w_t = \mathrm{Softmax}\Big(h_t^\top Q_w K^w_t\Big)V^w_t,\quad r^\ell_t = \mathrm{Softmax}\Big(h_t^\top Q_\ell K^\ell_t\Big)V^\ell_t,  
$$  
where $K^w_t,V^w_t$ are stacked keys and values from $M^w_t$, and likewise for LTM. The final output:  
$$  
y_t = U_h\,h_t + U_w\,r^w_t + U_\ell\,r^\ell_t\,.  
$$  

3.2 Reinforcement‐Learning Objective  
We formulate memory management as a Markov Decision Process (MDP). At time $t$, the state is $(h_t, M^w_{t-1},M^\ell_{t-1})$, actions are write/evict/purge decisions, and the reward $r_t$ is derived from downstream task loss improvement and compute penalty. We optimize the total expected discounted reward:  
$$  
J(\theta) = \mathbb{E}\big[\sum_{t=1}^T \gamma^{t-1}\,r_t\big]  
$$  
with PPO or A2C to train controller parameters $\theta = \{W_\alpha,b_\alpha,W_\beta,b_\beta,\dots\}$.  

3.3 Data Collection & Benchmarks  
We will use a combination of:  
• Long Range Arena (LRA) with sequences up to 16K tokens.  
• PG‐19 (50K tokens per document).  
• CodeXGLUE code modeling tasks (100K+ tokens).  
• Wikipedia Article Streams (1M tokens).  
• Synthetic reasoning tasks requiring retrieval of facts presented thousands of steps earlier (“Tempest Retrieval Tasks”).  

3.4 Experimental Design  
Baselines: Transformer‐XL, S4, Mamba, SMR, LMNs.  
Ablation studies:  
– Dual‐memory vs. single‐tier  
– With vs. without RL controller (random vs. learned eviction)  
– Varying $K,L,\delta,\epsilon$  
Evaluation Metrics:  
– Perplexity (language modeling)  
– Exact match / F1 (QA on LRA)  
– Retrieval‐accurate rate (synthetic tasks)  
– Compute cost: FLOPs, inference latency, memory footprint  
– Scaling curves: performance vs. sequence length ($16$K, $50$K, $100$K, $1$M).  

3.5 Algorithmic Summary  
Pseudocode (each time step $t$):  
```
1. h_t ← SSM_Update(h_{t-1}, x_t)
2. α_t, β_t ← Controller(h_t)
3. if rand() < α_t:  # Write to WM
      i* ← argmin_i c^w_{t-1,i}
      M^w_t[i*] ← (h_t, α_t)
   else:
      M^w_t ← M^w_{t-1}
4. Evict indices E ← {i | c^w_{t-1,i} < δ}
   for each i in E:
      j* ← argmin_j σ^ℓ_{t-1,j}
      M^ℓ_t[j*] ← (Compress(s^w_{t-1,i}), c^w_{t-1,i})
5. Purge LTM entries with σ^ℓ < ε
6. r^w_t, r^ℓ_t ← Attend(h_t, M^w_t, M^ℓ_t)
7. y_t ← U_h h_t + U_w r^w_t + U_ℓ r^ℓ_t
8. Compute task loss ℓ_t and reward r_t
9. Update SSM & controllers by combined gradient + RL step
```

4. Expected Outcomes & Impact  
We anticipate that ADM‐SSM will:  
• Achieve 10–20% lower perplexity on PG‐19 and LRA compared to strong SSM baselines at lengths ≥50K.  
• Double retrieval‐accuracy on synthetic long‐range QA tasks versus SMR or LMNs.  
• Reduce compute cost per token by 2× relative to transformer‐based long‐context methods at 100K+ tokens.  
• Demonstrate stable scaling behavior up to 1M‐token contexts with bounded memory footprint.  

Impact on the Field  
Our adaptive dual‐memory framework introduces a new paradigm for sequence modeling at scale. It provides:  
1. A principled mechanism for dynamic memory allocation that goes beyond recency or fixed hierarchies.  
2. An RL‐driven controller approach that directly ties memory decisions to downstream performance.  
3. Empirical and theoretical insights into trade‐offs between memory persistence, retrieval accuracy and compute efficiency.  

By open‐sourcing our code, pretrained models and extensive scaling curves, we aim to seed follow‐on work on hybrid memory models, hardware‐aware memory tiers and theoretical analyses of memory capacity in deep networks—paving the way for the next generation of sequence architectures.  

References (selected)  
[1] Qi et al. “SMR: State Memory Replay for Long Sequence Modeling.” arXiv:2405.17534.  
[2] Gu & Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” arXiv:2312.06837.  
[3] Taha. “Logarithmic Memory Networks.” arXiv:2501.07905.  
[4] Wang et al. “Graph-Mamba: Long-Range Graph Sequence Modeling.” arXiv:2402.00789.  
[5] Agarwal et al. “Spectral State Space Models.” arXiv:2312.06837.  