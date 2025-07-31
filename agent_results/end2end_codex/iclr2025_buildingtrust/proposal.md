Title  
Influence-Driven Selective Unlearning for Trustworthy Large Language Models

1. Introduction  
Background. Large language models (LLMs) trained on massive text corpora have demonstrated impressive capabilities in natural language understanding and generation. However, their reliance on memorized training data poses serious risks: the accidental leakage of personally identifiable information (PII), perpetuation of harmful biases, and the presence of undesirable or even copyrighted content. Full model retraining to remove specific memorized examples is computationally prohibitive, especially at the scale of modern LLMs (hundreds of millions to trillions of parameters). Consequently, there is a pressing need for efficient “unlearning” techniques that surgically remove unwanted knowledge while preserving the model’s general utility.  

Research Objectives. This proposal aims to design, implement, and evaluate a two-stage, influence-driven selective unlearning framework for LLMs. Our specific objectives are:  
• To develop a lightweight influence approximation module that identifies and ranks training samples most responsible for undesired behaviors (e.g., leaking “canary” sequences or biased completions).  
• To propose a gradient‐projection–based targeted unlearning algorithm that removes the influence of top-ranked samples from model parameters while minimizing collateral damage to general language performance.  
• To integrate a small distilled replay set and constrained fine-tuning to prevent catastrophic forgetting and ensure retention of previously learned capabilities.  
• To design comprehensive evaluation benchmarks and metrics covering privacy (canary removal, PII leakage), fairness (bias amplification), utility (perplexity, downstream task accuracy), and computational efficiency.  

Significance. Our framework addresses key challenges identified in recent literature (SOUL – arXiv:2404.18239; NAUF – arXiv:2407.10058; POP – arXiv:2406.14091; MOLLM – arXiv:2412.20412). By combining influence functions with gradient‐projection unlearning and replay, we strike a novel balance between unlearning efficacy, retention, and scalability. The outcome will be an open-source toolset and a set of best practices for practitioners in industry and academia who must comply with privacy regulations (e.g., GDPR “right to be forgotten”), mitigate bias, and maintain user trust in deployed LLM applications.

2. Methodology  
Overview. Our selective unlearning framework consists of four main components:  
  A. Data Collection & Canary Injection  
  B. Influence Approximation Module  
  C. Gradient Projection Unlearning & Constrained Fine-Tuning  
  D. Replay Set Construction and Regularization  

We detail each component below, followed by the experimental design and evaluation metrics.

2.1 Data Collection & Canary Injection  
• Privacy Benchmark: We follow standard canary‐based evaluation (Carlini et al., 2021). We inject synthetic “canary” sequences $c_i$ (e.g., “My credit card is 1234‐5678‐9012‐3456”) into a subset of training data. The goal is to verify complete removal of canaries post-unlearning.  
• Fairness Benchmark: We construct a balanced dataset of prompts requiring gendered continuations (e.g., “The nurse said that ___”). We insert a controlled gender bias by augmenting the training set with biased associations (e.g., “nurse → she,” “engineer → he”).  
• General Utility Data: We sample a held-out validation and test set from the original training distribution (e.g., wiki-text, Common Crawl subsets) to measure language modeling perplexity and downstream task metrics (e.g., GLUE average).

2.2 Influence Approximation Module  
We adopt a first‐order approximation of classical influence functions (Koh & Liang, 2017), similar in spirit to SOUL (Jia et al., 2024) and LIBU (Kudelya & Shirnin, 2025), but designed to scale to LLM‐sized models via Hessian‐vector‐product approximations.  

Let $\theta\in\mathbb{R}^d$ denote model parameters, and let $\ell(z;\theta)$ be the loss on a training example $z$. The exact influence of upweighting $z$ on loss at a test point $z'$ is:  
$$
I_\text{exact}(z,z') 
= -\nabla_\theta \ell(z';\theta)^\top H_\theta^{-1}\nabla_\theta \ell(z;\theta)\,,
$$  
where $H_\theta=\tfrac1n\sum_i\nabla^2_\theta \ell(z_i;\theta)$ is the empirical Hessian. Computing $H_\theta^{-1}$ directly is infeasible for large $d$. We approximate $v=H_\theta^{-1}u$ for $u=\nabla_\theta \ell(z;\theta)$ via conjugate‐gradient or LiSSA (Agarwal et al., 2017), truncated after $k$ iterations. This yields an approximate influence score:  
$$
\hat I(z,z') 
= -\nabla_\theta \ell(z';\theta)^\top v\,,
\quad 
v\approx H_\theta^{-1} \nabla_\theta \ell(z;\theta)\,.
$$  
We compute $\hat I(z,c_j)$ for each training sample $z$ and each canary $c_j$, then aggregate per‐sample influence by $\max_j|\hat I(z,c_j)|$ for privacy unlearning, and similarly for bias unlearning by selecting target prompts $p_b$ representing biased contexts.

Algorithm 1 Influence Scoring  
Input: Pretrained model $\theta^0$, training set $\mathcal{D}$, target set $\mathcal{T}$ (canaries or bias prompts), max CG iterations $k$.  
Output: Ranked list of samples by influence.  
1.  Compute $\nabla_\theta \ell(t;\theta^0)$ for each $t\in\mathcal{T}$.  
2.  For each $z\in\mathcal{D}$:  
    a.  Compute $u_z=\nabla_\theta \ell(z;\theta^0)$.  
    b.  Solve $H_{\theta^0}v_z = u_z$ approximately via $k$–step CG.  
    c.  Compute $\hat I(z) = -\max_{t\in\mathcal{T}} \nabla_\theta \ell(t;\theta^0)^\top v_z$.  
3.  Return top-$m$ samples with largest $|\hat I(z)|$.

2.3 Gradient Projection Unlearning & Constrained Fine-Tuning  
Once the $m$ most influential samples $\{z_{1},\dots,z_{m}\}$ are identified, we remove their influence by projecting out their gradient contributions from $\theta^0$. Let $G=[g_1,\dots,g_m]\in\mathbb{R}^{d\times m}$ with $g_i=\nabla_\theta \ell(z_i;\theta^0)$. We compute an orthonormal basis $Q\in\mathbb{R}^{d\times m}$ for $\mathrm{span}(G)$ via QR decomposition. Then define the projection matrix onto the orthogonal complement:  
$$
P = I_d - QQ^\top\,.  
$$  
We update parameters by:  
$$
\theta' = \theta^0 - \alpha\,P\Bigl(\sum_{i=1}^m g_i\Bigr)\,,
$$  
where $\alpha>0$ is a step‐size hyperparameter. This operation “pushes” $\theta^0$ away from the subspace spanned by the targeted gradients, effectively removing their first‐order influence.

Constrained Fine-Tuning. To recover any small utility losses, we perform a short fine-tuning stage on the remaining data $\mathcal{D}\setminus\{z_i\}$, with the constraint that updates remain approximately orthogonal to $Q$. Concretely, at each fine-tuning step $t$, given a mini-batch loss $\mathcal{L}_t$, we compute gradient $g_t=\nabla_\theta \mathcal{L}_t$. We then project $g_t\leftarrow Pg_t$ before applying the optimizer (e.g., Adam). This constraint prevents re-introduction of forgotten information.

2.4 Replay Set Construction and Regularization  
To prevent catastrophic forgetting of unrelated knowledge, we maintain a distilled replay set $\mathcal{R}$ of size $r\ll |\mathcal{D}|$. We construct $\mathcal{R}$ by:  
1.  Randomly sampling $r_1$ examples from $\mathcal{D}\setminus\{z_i\}$.  
2.  Adding $r_2$ exemplars selected via core‐set selection (e.g., K-Center Greedy) to maximize coverage of the feature (embedding) space.  

We interleave replay batches with fine-tuning batches, optimizing the loss:  
$$
\mathcal{L}_\text{total} 
= \mathcal{L}_\text{main}(\mathcal{D}\setminus\{z_i\})
+ \lambda\,\mathcal{L}_\text{replay}(\mathcal{R})\,,
$$  
where $\lambda$ governs the strength of retention regularization. Replay examples are also gradient‐projected by $P$ to ensure they do not reintroduce the unwanted influence.

2.5 Experimental Design  
Baselines. We compare against:  
• SOUL (Jia et al., 2024)  
• NAUF (Liu et al., 2024)  
• POP (Lee et al., 2024)  
• MOLLM (Pan et al., 2024)  

Models. We evaluate on GPT-2 medium (345M params) and a distilled T5 (220M params) to show scalability. All experiments run on 8×A100 GPUs.  

Hyperparameters. We grid‐search:  
• Number of top samples $m\in\{50,100,200\}$  
• CG iterations $k\in\{5,10,20\}$  
• Projection step‐size $\alpha\in\{1e^{-3},1e^{-4}\}$  
• Replay size $r\in\{500,1000,2000\}$  
• Regularization weight $\lambda\in\{0.1,1.0\}$  

Ablations. We perform ablations to isolate the effect of:  
• Influence ranking (vs. random ranking)  
• With and without gradient projection  
• With and without replay set  

2.6 Evaluation Metrics  
Privacy/Canary Removal.  
• Canary Extraction Rate (CER): fraction of canaries retrievable via beam search.  
• Nearest‐Neighbor Attack Success: proportion of test queries that match canary tokens.  

Fairness.  
• Bias Amplification Score (Park et al., 2018): measures shift in gender–occupation co-occurrence.  
• Stereotype Score (NLU fairness benchmarks).  

Utility.  
• Perplexity on held-out validation set.  
• Downstream task accuracy (GLUE average).  

Efficiency.  
• Total GPU hours and wall‐clock time compared to full retraining.  
• Memory footprint (peak GPU memory).  

3. Expected Outcomes & Impact  
We anticipate the following outcomes:

1. Unlearning Efficacy.  
   – Our method will achieve near‐zero CER (<1%) on canary removal while preserving a low nearest‐neighbor attack success rate.  
   – Bias metrics (bias amplification, stereotype scores) will reduce by ≥40% compared to the original model and outperform baselines by 10–20%.

2. Utility Retention.  
   – Perplexity degradation will be <5% relative to the original model.  
   – GLUE average score drop will be <2%, demonstrating minimal impact on general capabilities.  
   – Ablation studies will confirm that gradient projection and replay both contribute significantly to retention.

3. Computational Efficiency.  
   – Total GPU cost will be <30% of full retraining.  
   – Influence approximation and gradient projection will introduce at most a 1.5× slow-down over standard fine-tuning.

4. Theoretical Insights.  
   – Analysis of the projection subspace will shed light on the geometry of “forgetting” in high‐dimensional parameter space.  
   – We will derive bounds on residual influence after projection under mild smoothness assumptions.

Impact.  
• Academic. We will release our code and benchmarks to foster reproducible research in LLM unlearning and trustworthiness. Our theoretical and empirical findings will guide future unlearning algorithms that balance efficacy, retention, and efficiency.  
• Industrial. Practitioners deploying LLMs in sensitive settings (medical, legal, customer support) can use our toolkit to comply with “right to be forgotten” requests and mitigate bias without the cost of full retraining.  
• Regulatory. By providing transparent unlearning procedures and evaluation protocols, our work can inform policy discussions on LLM accountability, auditability, and compliance standards.  
• Societal. A reliable unlearning framework reduces risks of PII leakage and biased outputs, thereby strengthening user trust and safety in AI-driven systems.

4. Conclusion  
This proposal outlines a principled, practical framework for selective unlearning in large language models, combining influence estimation, gradient‐projection removal, constrained fine‐tuning, and replay regularization. By targeting the most responsible training samples and surgically eliminating their parameter influence, we aim to meet stringent privacy and fairness requirements while preserving model utility and reducing computational costs. Successful completion will yield both theoretical contributions on the geometry of unlearning and an open-source system ready for real-world deployment, ultimately advancing the trustworthiness of LLM‐driven applications.