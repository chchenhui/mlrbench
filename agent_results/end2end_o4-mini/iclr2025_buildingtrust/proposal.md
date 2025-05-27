1. Title  
Cluster-Driven Certified Unlearning for Large Language Models

2. Introduction  
Background. Large Language Models (LLMs) such as GPT-2, GPT-3 and their successors have demonstrated remarkable performance on a wide array of natural language tasks.  However, recent work has shown that LLMs can inadvertently memorize and regenerate sensitive, copyrighted, or outdated information from their training corpora.  As LLMs become integrated into real-world applications—chatbots, automated legal assistants, medical Q&A systems—the ability to efficiently and provably remove (“unlearn”) user- or compliance-sensitive data has become paramount for privacy, trust, and regulatory compliance.  

Key Challenges.  
  • Balancing targeted forgetting against overall model utility: naïve retraining or full fine-tuning wastes computation and may harm model accuracy.  
  • Computational efficiency and scalability: full retraining is often infeasible for billion-parameter models.  
  • Sequential unlearning requests: multiple, ongoing delete requests must not accumulate to degrade performance.  
  • Providing formal, statistical guarantees that unlearning has occurred, to satisfy privacy audits.  

Gap in Existing Work. Prior methods offer partial solutions: ReLearn (Xu et al., 2025) uses data augmentation and fine-tuning; “Unlearn What You Want” (Chen & Yang, 2023) inserts lightweight unlearning layers; CodeUnlearn (Wu et al., 2024) uses discrete concept codebooks; UNDIAL (Dong et al., 2024) applies self-distillation; O3 (Gao et al., 2024) employs orthogonal LoRA adapters.  However, none combine:  
  • A principled decomposition of model knowledge into clusters;  
  • Targeted interventions at the subspace level;  
  • A certification step via Fisher Information to quantify removed information.  

Research Objectives. This proposal aims to develop and evaluate a Cluster-Driven Certified Unlearning framework that:  
  1. Partitions a pretrained LLM’s latent representation space into semantically coherent clusters via hierarchical spectral clustering.  
  2. Identifies which clusters encode the concepts or examples to be removed, using efficient influence-score approximations.  
  3. Applies targeted low-rank gradient surgery within the affected subspaces to erase memorized data without retraining the full model.  
  4. Carries out a Fisher-information-based certification to bound the statistical divergence between the original and “unlearned” models, assuring auditors that no residual information remains.  

Significance. By avoiding full retraining, our method aims to reduce compute cost by over 60% (relative to full fine-tuning) on GPT-2 benchmarks, support real-time delete requests, maintain model utility (perplexity, downstream task performance), and provide provable guarantees for privacy and compliance.  This framework will strengthen trust in LLM deployments across industries.

3. Methodology  
Our approach comprises five key components: (1) Data Collection & Preprocessing, (2) Representation Clustering, (3) Influence‐Score Approximation, (4) Targeted Low-Rank Gradient Surgery, and (5) Fisher Information Certification. We then detail the experimental protocol and evaluation metrics.

3.1 Data Collection & Preprocessing  
  • Base Model: We begin with a pretrained GPT-2 Small or Medium model (Radford et al.).  
  • Training Corpus: Use the original WebText dataset and additional domain‐specific corpora (e.g., medical QA, legal documents) as needed.  
  • Deletion Requests: Simulate or collect real‐world delete sets $S_d$—for instance, privacy‐sensitive sentences, user chat logs, or copyrighted snippets.  
  • Validation & Test Sets: Hold out separate validation data $D_{val}$ (for tuning hyperparameters) and test data $D_{test}$ (for final evaluation).

3.2 Representation Clustering  
We cluster the model’s knowledge by segmenting hidden‐layer activations into $K$ clusters.  For a given layer $\ell$, let $h_i^\ell \in \mathbb{R}^d$ be the activation for token or span $i$.  

1. Similarity Matrix. Compute pairwise affinities  
   $$ S_{ij} = \exp\!\bigl(-\|h_i^\ell - h_j^\ell\|^2 / \sigma^2\bigr)\,. $$  
2. Graph Laplacian. Form degree matrix $D = \mathrm{diag}(\sum_j S_{ij})$ and Laplacian  
   $$ L = D - S\,. $$  
3. Spectral Embedding. Compute the first $m$ eigenvectors $U = [u_1,\dots,u_m]$ solving  
   $$ L\,u_k = \lambda_k D\,u_k,\quad k=1,\dots,m. $$  
4. Clustering. Apply hierarchical agglomerative clustering on rows of $U$ to yield $K$ clusters $\{C_1,\dots,C_K\}$.  Clusters are hierarchically organized to allow multi‐resolution deletion.

3.3 Influence‐Score Approximation  
To determine which clusters encode information about the deletion set $S_d$, we adapt influence functions (Koh & Liang, 2017) to the cluster subspaces.  For each cluster $C_k$, let $U_k\in \mathbb{R}^{d\times m_k}$ be the orthonormal basis of the span of $\{h_i^\ell : i\in C_k\}$.  Define the cluster‐level score  
   $$ I_k \;=\; \sum_{z\in S_d} \nabla_\theta\ell(z,\theta)^\top \,U_k U_k^\top\, \nabla_\theta R(\theta)\,, $$  
where $\ell(z,\theta)$ is the token‐level loss and $R(\theta)$ is a small ridge regularizer to stabilize Hessian inversion.  Practically, we approximate $H_\theta^{-1}\nabla_\theta\ell$ via a truncated Neumann series or stochastic Lanczos quadrature to avoid full Hessian inversion.

Clusters with $I_k$ above a threshold $\tau_I$ are marked for surgical intervention.

3.4 Targeted Low-Rank Gradient Surgery  
For each marked cluster $C_k$, we perform a low-rank gradient update that “erases” its contribution while minimally affecting orthogonal subspaces.  Let $g = \nabla_\theta \sum_{z\in S_d}\ell(z,\theta)$ be the aggregate gradient on the delete set.  We project $g$ onto the cluster subspace and scale:  
   $$ g_k = U_k U_k^\top\,g,  
      \quad  
      g_{\perp k} = g - g_k. $$  
We then update  
   $$ \theta' = \theta - \eta\,g_k, $$  
but with a carefully chosen learning rate $\eta_k$ for each cluster so as to remove memorized weights while preserving overall utility.  To handle sequential deletions $\{S_d^{(1)},S_d^{(2)},\dots\}$, we accumulate low-rank adapters in orthogonal subspaces, analogous to LoRA (Hu et al., 2021) but cluster‐specific, ensuring parameter disentanglement across requests.

3.5 Fisher Information Certification  
After gradient surgery, we quantify the divergence between the original model $p_\theta$ and the unlearned model $p_{\theta'}$ via a second‐order approximation:  
   $$ D_{\mathrm{KL}}\bigl(p_\theta\;\|\;p_{\theta'}\bigr)  
      \approx \tfrac12\Delta\theta^\top F(\theta)\,\Delta\theta,  
      \quad  
      \Delta\theta = \theta' - \theta, $$  
where $F(\theta)=\mathbb{E}_{x\sim p_\theta}[\nabla\log p(x;\theta)\nabla\log p(x;\theta)^\top]$ is the Fisher Information Matrix.  We bound this quantity by a user‐specified $\varepsilon$, guaranteeing that the unlearning operation does not introduce unexpected shifts.  A certificate is issued if $D_{\mathrm{KL}}\le\varepsilon$; otherwise, we iteratively refine $\eta_k$ or include neighboring clusters.

3.6 Experimental Design & Evaluation Metrics  
Datasets & Models:  
  – GPT-2 Small (117M) and Medium (345M) on WebText, plus domain‐specific corpora.  
  – Simulated deletion sets of varying sizes ($|S_d|=10$–1000 examples).  

Baselines:  
  • ReLearn (Xu et al., 2025)  
  • Unlearn What You Want (Chen & Yang, 2023)  
  • CodeUnlearn (Wu et al., 2024)  
  • UNDIAL (Dong et al., 2024)  
  • O3 framework (Gao et al., 2024)  

Metrics:  
  – Knowledge Forgetting Rate (KFR): fraction of targeted information removed, measured via membership inference or token‐reconstruction accuracy.  
  – Knowledge Retention Rate (KRR): relative performance on held‐out tasks not related to $S_d$.  
  – Perplexity ∆: change in model perplexity on $D_{test}$.  
  – Downstream Task Accuracy: e.g., GLUE benchmarks.  
  – Computational Cost: GPU hours, wall‐clock time vs. full retraining.  
  – Certified KL Bound: $D_{\mathrm{KL}}$ relative to $\varepsilon$.  

Ablations: vary number of clusters $K$, spectral embedding dimension $m$, influence threshold $\tau_I$, and adapter rank.  Statistical significance will be assessed via paired $t$-tests ($p<0.05$).

4. Expected Outcomes & Impact  
Expected Technical Outcomes.  
  1. A novel unlearning framework that segments model knowledge into clusters, enabling targeted interventions at subspace level.  
  2. Demonstration of 60–70% reduction in compute cost compared to full retraining or standard fine-tuning, across GPT-2 variants.  
  3. Maintenance of high utility: KRR ≥ 95% on held‐out tasks, perplexity increase ≤ 2%.  
  4. Robust handling of sequential unlearning requests without catastrophic forgetting or performance drift.  
  5. A provable $D_{\mathrm{KL}}\le\varepsilon$ guarantee, with auto-generated certificates for auditors.

Broader Impacts.  
  • Privacy & Compliance. Real-time compliance with “right to be forgotten” and data‐protection regulations (GDPR, CCPA).  
  • Trust in AI Systems. By offering statistical certificates, organizations can demonstrate due diligence in data removal, reducing legal and reputational risks.  
  • Deployment at Scale. The method’s efficiency and modularity make it suitable for integration into commercial APIs, on-device assistants, and regulated domains (healthcare, finance, legal).  
  • Open Source & Community. We will release code, pre-trained clustering adapters, and benchmarks, fostering reproducibility and further advances in machine unlearning.

5. References  
1. Xu, H., Zhao, N., Yang, L., et al.: ReLearn: Unlearning via Learning for Large Language Models. arXiv:2502.11190 (2025)  
2. Chen, J., Yang, D.: Unlearn What You Want to Forget: Efficient Unlearning for LLMs. arXiv:2310.20150 (2023)  
3. Wu, Y., Dossou, B. F. P., Liu, D.: CodeUnlearn: Amortized Zero-Shot Machine Unlearning in Language Models Using Discrete Concept. arXiv:2410.10866 (2024)  
4. Dong, Y. R., Lin, H., Belkin, M., et al.: UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in LLMs. arXiv:2402.10052 (2024)  
5. Liu, C. Y., Wang, Y., Flanigan, J., Liu, Y.: Large Language Model Unlearning via Embedding-Corrupted Prompts. arXiv:2406.07933 (2024)  
6. Pan, Z., Zhang, S., Zheng, Y., et al.: Multi-Objective Large Language Model Unlearning. arXiv:2412.20412 (2024)  
7. Du, H., Liu, S., Zheng, L., et al.: Privacy in Fine-tuning Large Language Models: Attacks, Defenses, and Future Directions. arXiv:2412.16504 (2024)  
8. Xiao, Y., Jin, Y., Bai, Y., et al.: PrivacyMind: Large Language Models Can Be Contextual Privacy Protection Learners. arXiv:2310.02469 (2023)  
9. Gao, C., Wang, L., Weng, C., et al.: Practical Unlearning for Large Language Models. arXiv:2407.10223 (2024)  
10. Geng, J., Li, Q., Woisetschlaeger, H., et al.: A Comprehensive Survey of Machine Unlearning Techniques for LLMs. arXiv:2503.01854 (2025)