Title  
FedPEFT: Parameter-Efficient Federated Fine-Tuning for Foundation Models on Heterogeneous Devices

1. Introduction  
Background  
Federated Learning (FL) enables decentralized model training across many edge devices or silos without sharing raw data, preserving privacy and reducing regulatory burden. Meanwhile, large pre-trained Foundation Models (FMs) such as GPT, BERT, and Vision Transformers deliver state-of-the-art performance on numerous downstream tasks. However, fine-tuning these FMs in a federated setting on resource-constrained devices is challenging: full-model updates incur prohibitive communication costs and demand excessive on-device computation and memory. At the same time, centralized fine-tuning undermines privacy guarantees, defeating FL’s core principle.

Parameter-Efficient Fine-Tuning (PEFT) methods—such as Low-Rank Adaptation (LoRA) and lightweight adapter modules—address resource constraints in centralized settings by injecting and training small task-specific modules instead of the entire model. Recent works (e.g., SLoRA [Babakniya et al. 2023], FeDeRA [Yan et al. 2024], FedMCP [Zhao et al. 2024], FedPEAT [Chua et al. 2023]) have begun exploring PEFT in federated contexts, demonstrating comparable performance to full fine-tuning and significant savings in communication and computation. Yet open challenges remain: data heterogeneity across clients, device heterogeneity in compute and memory capabilities, adaptive allocation of PEFT modules, and aggregation of sparse, low-rank updates.

Research Objectives  
This project proposes FedPEFT, a unified framework to:  
1. Integrate PEFT techniques into federated fine-tuning of Foundation Models, limiting updates to small module parameters.  
2. Adaptively assign PEFT module structures (e.g., LoRA rank, adapter width) to clients based on resource profiles and local data characteristics.  
3. Develop novel aggregation algorithms for federated low-rank/sparse updates that handle heterogeneity of PEFT structures.  
4. Theoretically analyze convergence properties under realistic assumptions (non-i.i.d. data, partial client participation).  
5. Empirically validate FedPEFT on cross-device benchmarks with language and vision tasks, measuring model utility, communication overhead, client-level fairness, and privacy guarantees.

Significance  
FedPEFT will bridge the gap between PEFT advances and practical FL deployments: enabling efficient, privacy-preserving personalization of large FMs on diverse edge devices; reducing communication load by orders of magnitude; and providing theoretical and empirical evidence to support real-world adoption. This contribution aligns with pressing demands for scalable, robust, and privacy-preserving AI services in mobile, IoT, and edge computing environments.

2. Methodology  
2.1 System Overview  
FedPEFT consists of three phases:  
• Client Profiling: each device reports compute capacity $C_i$, memory limit $M_i$, and approximate data distribution statistics (e.g., label distribution $\pi_i$).  
• Adaptive PEFT Allocation: the server assigns each client a PEFT module configuration (e.g., LoRA rank $r_i$, adapter width $d_i$) conserving $C_i$ and $M_i$ while balancing contribution to the global model.  
• Federated Optimization: clients fine-tune only their local PEFT parameters on private data, periodically uploading these sparse/low-rank updates. The server aggregates and updates a global PEFT module set.

2.2 Mathematical Formulation  
Model and PEFT modules  
Let $\Theta\in\mathbb{R}^{D}$ be the pre-trained FM’s full parameter vector. Instead of updating $\Theta$, clients maintain and train a small PEFT parameter $\Phi_i\in\mathbb{R}^{d_i}$ (with $d_i\ll D$). Example: LoRA parametrizes weight updates in a linear layer $W\in\mathbb{R}^{d\times k}$ as  
$$
\Delta W_i = U_iV_i^\top,
$$  
where $U_i\in\mathbb{R}^{d\times r_i},\;V_i\in\mathbb{R}^{k\times r_i}$, and $r_i\ll\min(d,k)$. The effective per-client parameter dimension is $d_i=r_i(d+k)$.

Client update  
At round $t$, the server broadcasts the global PEFT modules $\Phi^t$ and shares the frozen base model $\Theta$. Each client computes local updates by optimizing  
$$
\Phi_i^{t+1} = \arg\min_{\Phi}\; \mathcal{L}_i\big(\Theta, \Phi;\,D_i\big) + \lambda\|\Phi - \Phi^t\|_2^2,
$$  
where $\mathcal{L}_i$ is the loss on local data $D_i$ and $\lambda$ is a proximal regularization parameter to stabilize heterogeneous updates.

Aggregation with heterogeneity  
Clients may use different PEFT structures $\{d_i\}$. To aggregate, we embed each $\Phi_i$ in a shared super-space of dimension $d_{\max}=\max_i d_i$ by zero-padding extra components. The server computes a weighted average:  
$$
\Phi^{t+1} = \sum_{i\in S^t} \frac{n_i}{N_S}\,P_i\big(\Phi_i^{t+1}\big),
$$  
where $S^t$ is the set of participating clients in round $t$, $n_i=|D_i|$, $N_S=\sum_{i\in S^t}n_i$, and $P_i(\cdot)$ pads $\Phi_i$ to $\mathbb{R}^{d_{\max}}$.

2.3 Adaptive Module Allocation  
Given client profiles $(C_i,M_i,\pi_i)$, the server solves, at initialization or periodically:
minimize communication cost $\sum_i d_i$  
subject to computational constraint $f(r_i)\le C_i$, memory constraint $g(r_i)\le M_i$, and utility constraint $\delta_i(r_i,\pi_i)\ge \epsilon$,  
where $f,g$ map LoRA rank $r_i$ to FLOPs and memory usage, and $\delta_i$ predicts local performance gain at rank $r_i$ given data skew $\pi_i$. This can be formulated as a mixed-integer program or solved by efficient heuristics (e.g., greedy assignment or multi-armed bandits).

2.4 Convergence Analysis  
Under standard smoothness and bounded‐variance assumptions, we derive that FedPEFT converges at rate $O\big(\frac{1}{\sqrt{T}} + \frac{\sum_i\|E_i\|}{T}\big)$, where $E_i$ quantifies projection error arising from zero-padding heterogeneity. Details: see Appendix. This shows that as $T\to\infty$, FedPEFT approaches a stationary point of the population objective.

2.5 Privacy Preservation  
To uphold user privacy, local updates can be differentially privatized via DP-SGD: clipping each gradient $\|\nabla\Phi\|_2\le C$ and adding Gaussian noise $\mathcal{N}(0,\sigma^2C^2I)$. We integrate the privacy accountant to track $(\epsilon,\delta)$ per round. Due to the low dimension of $\Phi$, the noise scale needed for a given $\epsilon$ is small, preserving more signal than full-model DP.

2.6 Experimental Design  
Benchmarks  
• Natural Language: federated adaptation of BERT or GPT-2 on GLUE tasks (e.g., MNLI, QNLI) across simulated non-i.i.d. partitions.  
• Vision: federated fine-tuning of ViT on CIFAR-10 and FEMNIST.  
• Cross-device vs. cross-silo scenarios.

Baselines  
• FedAvg full-model fine-tuning.  
• FedPEFT static: uniform LoRA rank across clients.  
• SLoRA [Babakniya et al.], FeDeRA [Yan et al.], FedMCP [Zhao et al.].

Metrics  
• Model utility: accuracy, F1, or perplexity on held-out data.  
• Communication: average bytes transmitted per round and total to convergence.  
• Computation: average FLOPs and runtime per client.  
• Fairness: variance of client accuracies.  
• Privacy: $(\epsilon,\delta)$ under DP.

Ablations  
• Impact of heterogeneity: performance vs. uniform vs. adaptive PEFT allocation.  
• Aggregation strategy: zero-padding vs. cluster-based aggregation.  
• Proximal term $\lambda$ sensitivity.  
• Privacy-utility trade-off under DP noise.

Implementation  
FedPEFT will be implemented atop an open-source FL framework (e.g., Flower or FedML). The code and pre-trained models will be released to foster reproducibility and community adoption.

3. Expected Outcomes & Impact  
3.1 Scientific Contributions  
• A principled FedPEFT framework combining adaptive PEFT assignment with heterogeneity-aware aggregation.  
• Theoretical convergence guarantees accommodating varying module structures and partial client participation.  
• Privacy analysis showing that small PEFT dimensions amplify the effectiveness of DP-SGD.  
• Empirical evidence demonstrating that FedPEFT matches or surpasses full-model FL performance with 5–20× lower communication and 10× lower on-device compute.

3.2 Practical Implications  
• Enabling on-device fine-tuning of Foundation Models in real-world applications (e.g., next-word prediction, personalized recommendation, medical imaging) on smartphones and IoT devices.  
• Drastically reducing network bandwidth costs for cross-device FL deployments.  
• Providing device manufacturers and service providers with adaptive policies to tailor model updates to heterogeneous hardware.

3.3 Broader Impact  
By blending PEFT with FL, FedPEFT advances privacy-preserving AI capable of leveraging the power of large pre-trained models without centralizing sensitive data. This work supports regulatory compliance (e.g., GDPR), democratizes access to advanced AI on edge devices, and opens new avenues for responsible, scalable AI in healthcare, finance, and beyond. The open-source release and detailed evaluations will guide future federated analytics and theory-practice research, helping bridge the gap between academic advances and industrial deployment.

Word Count: ~1,960