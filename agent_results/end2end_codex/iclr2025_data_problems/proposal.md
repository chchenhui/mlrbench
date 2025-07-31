Title  
MRIA: A Modular Framework for Scalable Influence Attribution in Retrieval-Augmented Generation Foundation Models  

Introduction  
Background  
Retrieval-Augmented Generation (RAG) has emerged as a cornerstone in modern large-scale language and multimodal modeling. By combining a powerful pretrained foundation model with an external knowledge retriever, RAG systems can ground generation in up-to-date facts, specialized corpora, or multimodal content streams. However, as RAG is adopted in high-stakes applications—legal research, scientific discovery assistants, personalized tutoring, or data marketplaces—the need for transparent, reproducible attribution of generated outputs to their underlying data sources becomes critical. Proper attribution fosters user trust, enables copyright compliance, and underpins economic models that reward data providers fairly.  

Literature from 2023–2025 has addressed context attribution in RAG (e.g., ARC-JSD based on Jensen-Shannon divergence), stochastic Shapley approximations (SIM-Shapley), and leverage-score sampling for efficient Shapley estimation. Surveys on RAG trustworthiness highlight gaps in scalable, end-to-end attribution and benchmarks for evaluating attribution fidelity and runtime overhead. Existing methods either incur prohibitive $\mathcal O(N)$ retriever queries (leave-one-out), disrupt inference latency, or lack integration with generation influences.  

Research Objectives  
1. Develop MRIA, a two-stage modular pipeline that jointly attributes retrieval and generation influences at scale.  
2. Leverage randomized Shapley estimators with sketching to approximate retrieval contributions in streaming settings.  
3. Introduce low-rank Jacobian sketching for efficient gradient-based generation attribution without full backpropagation.  
4. Implement MRIA end-to-end in a LLaMA-based RAG architecture with open-source components, ensuring practical integration.  
5. Rigorously evaluate trade-offs between attribution accuracy, latency, memory footprint, and scalability on text and multimodal benchmarks.  

Significance  
MRIA will be among the first frameworks to offer near-real-time, fine-grained attribution in RAG pipelines of realistic size (millions of documents and sub-second latency budgets). By doing so, it:  
• Enhances transparency and interpretability of RAG outputs, supporting debugging and compliance.  
• Enables data marketplace applications where data providers are compensated in proportion to actual usage.  
• Bridges retrieval and generation attribution, providing end-to-end source scoring rather than siloed metrics.  
• Opens avenues for research on fairness, privacy, and economic incentives in foundation model deployments.  

Methodology  
Overview  
MRIA consists of two interconnected modules: (1) Retrieval Attribution and (2) Generation Attribution. Each module is designed for streaming operation, sublinear space complexity, and tight integration with inference. The combined output is a ranked list of source contributions for each generated token or answer.  

1. Data Collection and Preprocessing  
– Corpora: Two classes of datasets will be used.  
  • Text benchmarks: Natural Questions, HotpotQA, TriviaQA; retrieval galleries of sizes 10^5, 10^6, 10^7 documents drawn from Wikipedia.  
  • Multimodal benchmarks: VQA v2, WebQA (images+text); galleries include image embeddings and associated metadata.  
– Preprocessing:  
  • Compute dense embeddings $e_i\in\mathbb R^d$ for each document or image using a dual-encoder (e.g., CLIP or LLaMA embedding head).  
  • Build an approximate nearest neighbor index (FAISS) supporting batched retrieval.  
  • Partition galleries into shards for streaming sketch updates.  

2. Retrieval Attribution Module  
2.1 Shapley-Value Definition  
Given a query $q$, a retriever scores a subset $S_q$ of top-$k$ items with similarity scores $\{s_i\}_{i\in S_q}$. We define a set function $f:\mathcal P(S_q)\to\mathbb R$ that measures retrieval utility (e.g., sum of normalized scores or retrieval recall on held-out gold). The true Shapley value of item $i$ is  
$$  
\phi_i = \sum_{R\subseteq S_q\setminus\{i\}} \frac{|R|!\,(|S_q|-|R|-1)!}{|S_q|!}\Bigl[f(R\cup\{i\})-f(R)\Bigr]\,.  
$$  
Direct computation is intractable for $|S_q|\gg 20$.  

2.2 Randomized Shapley Estimation with Sketching  
We approximate $\phi_i$ by randomized sampling of permutations and by sketching $f$ evaluations:  
Algorithm 1: Retrieval Attribution  
Input: scored set $S_q$, utility oracle $f$, sketch matrix $C\in\{0,1\}^{m\times n}$ (CountSketch), sample count $T$.  
Output: approximate Shapley $(\hat\phi_i)_{i\in S_q}$.  
1. Initialize $\hat\phi_i\leftarrow0$ for all $i$.  
2. For $t=1$ to $T$:  
   a. Sample a random permutation $\pi_t$ of $S_q$.  
   b. Maintain an $m$-dimensional CountSketch summary $u\leftarrow0$.  
   c. For each position $j$ in $\pi_t$:  
      i. Let $i=\pi_t[j]$. Update the sketch $u\leftarrow u + C\cdot e_i$.  
      ii. Estimate $f(R_{j})$ from $u$ as $\tilde f_j$.  
      iii. Compute marginal gain $\Delta_t(i)=\tilde f_j-\tilde f_{j-1}$.  
      iv. Accumulate $\hat\phi_i\leftarrow \hat\phi_i + \Delta_t(i)/T$.  
3. Return $(\hat\phi_i)$.  

Here, CountSketch compresses sum of embeddings with $O(m)$ memory, and $f$ is estimated as a linear function of the sketch: $\tilde f(u) = w^\top u$ for a learned weight vector $w$. The complexity per query is $O(T\,|S_q|\,m)$, with $m\ll |S_q|$.  

3. Generation Attribution Module  
3.1 Influence via Jacobian Sketching  
Given retrieved context tokens $X=(x_1,\dots,x_L)$ and a generation model that produces tokens $Y=(y_1,\dots,y_T)$, we define the influence of context token $x_j$ on output token $y_t$ by the gradient magnitude  
$$  
I_{j,t} = \bigl\lVert \tfrac{\partial \log p(y_t\mid y_{<t},X)}{\partial e_j} \bigr\rVert_2\,.  
$$  
Exact Jacobians are expensive for $L,T\sim10^3$. We employ random projection sketching:  
3.2 Low-Rank Jacobian Sketch  
Let $J_{t}\in\mathbb R^{d\times L}$ be the Jacobian of $\log p(y_t)$ wrt all embeddings. We draw a random Gaussian sketch matrix $S\in\mathbb R^{L\times r}$ with $r\ll L$. Compute $\tilde J_{t}=J_{t}S\in\mathbb R^{d\times r}$ via one backward pass:  
$$  
\tilde J_t = \frac{\partial \log p(y_t)}{\partial (X S)}\,.  
$$  
Then approximate per-token influence  
$$  
\hat I_{j,t} = \| \tilde J_t\,[S^\top]_{:,j}\|_2\,.  
$$  
Accumulating across all $t$ yields a generation attribution score $\gamma_j=\sum_{t=1}^T \hat I_{j,t}$.  

4. End-to-End Attribution Score  
For each source document $d$ contributing tokens $j\in d$, we combine retrieval and generation scores:  
$$  
\text{Score}(d) = \sum_{i\in S_q\cap d} \hat\phi_i \;+\;\lambda \sum_{j\in d} \gamma_j\,,  
$$  
where $\lambda$ trades off retrieval versus generation influence. Documents are then ranked by Score$(d)$ to produce final attribution.  

5. Experimental Design  
5.1 Baselines  
• Leave-one-out (LOO) attribution: remove each document and measure output difference.  
• KernelSHAP with $T=100$ samples.  
• SIM-Shapley and LeverageSHAP with recommended hyperparameters.  

5.2 Evaluation Metrics  
• Attribution Fidelity: Kendall’s $\tau$ and Spearman’s $\rho$ correlation between MRIA scores and ground-truth LOO scores.  
• Accuracy@k: fraction of true top-k influential sources recovered.  
• Latency Overhead: percentage increase in end-to-end RAG inference time.  
• Memory Footprint: additional peak RAM for sketches and buffers.  

5.3 Ablations and Sensitivity  
• Vary sketch dimension $m\in\{64,128,256\}$ and sample count $T\in\{50,100,200\}$.  
• Sweep generation sketch rank $r\in\{16,32,64\}$.  
• Explore $\lambda\in[0,1]$ to balance retrieval vs. generation.  

5.4 Reproducibility and Open Source  
All code, pretrained models, and evaluation scripts will be released under an open-source license. Detailed instructions will accompany the benchmark scripts to reproduce results on text and multimodal tasks.  

Expected Outcomes & Impact  
Outcomes  
• A fully documented implementation of MRIA integrated with LLaMA-2 (or similar open LLM) and FAISS retrieval.  
• Empirical evidence that MRIA attains $\geq$0.85 Kendall’s $\tau$ relative to LOO on text benchmarks with $<2\times$ overhead, and $\geq$0.80 on multimodal tasks with $<1.5\times$ overhead.  
• Comprehensive ablation charts illustrating trade-offs among sketch sizes, sample counts, and $\lambda$ settings.  
• A new standardized benchmark suite for RAG attribution, including dataset snapshots, evaluation metrics, and baseline results.  

Impact  
Transparency and Trust: MRIA will enable practitioners and end users to inspect which data sources most influenced a given answer, mitigating risks of hallucination and reinforcing factual integrity.  
Data Marketplaces: Fine-grained attribution scores provide a principled basis for compensating data providers according to actual usage, catalyzing sustainable data ecosystems.  
Legal Compliance: By tracing generated content back to specific copyrighted sources with quantifiable contributions, MRIA aids in auditing and regulatory reporting.  
Research Catalyst: The modular design invites extensions—fairness-aware weighting, privacy-preserving sketches, adaptive sampling schemes—and provides a shared evaluation platform.  
Broad Applicability: While focused on text and image QA, MRIA’s sketching and Shapley-based methods generalize to audio, video, and beyond, paving the way for attribution in truly multi-modal foundation models.  

In sum, MRIA is poised to address critical data challenges in RAG-based foundation models, offering a scalable, accurate, and open framework for end-to-end attribution. By delivering both methodological innovations and a practical toolkit, this work will shape future standards for transparency, fairness, and accountability in AI systems.