Title  
InfluenceSpace: Hierarchical Influence–Driven Curation for Multi-Modal Foundation Models  

1. Introduction  

1.1 Background  
Foundation models (FMs) have achieved remarkable performance across vision, language, and cross-modal tasks by leveraging massive, heterogeneous datasets. However, the raw training corpora for multi-modal FMs typically contain billions of image–text pairs drawn from web sources, resulting in substantial redundancy, mislabeled or noisy samples, and pronounced representation biases. Naïve curation strategies—such as random filtering, simple heuristic thresholds on metadata, or uniform down-sampling—fail to quantify how each datum truly contributes to downstream model performance, robustness, and fairness. As FMs scale in parameter count and modality complexity, inefficient or biased data selection exacerbates environmental costs, training time, and societal harms (e.g., under-representation of minority groups).  

1.2 Research Objectives  
InfluenceSpace aims to fill this gap by developing a scalable, principled data-centric pipeline for multi-modal FM training. Specifically, our objectives are:  
•   To design a hierarchical two-stage curation framework that (a) clusters raw multi-modal data into semantically coherent groups and (b) computes amortized influence scores per cluster to identify harmful, redundant, or under-represented data.  
•   To formulate efficient influence estimation methods for large-scale multi-modal FMs, leveraging low-rank approximations of the Hessian and mini-batch gradient statistics.  
•   To develop reweighting and pruning strategies that optimize a global utility objective balancing accuracy, robustness, and fairness.  
•   To empirically validate the InfluenceSpace pipeline on standard vision–language benchmarks, measuring trade-offs between data reduction ratio, downstream performance (e.g., VQA accuracy, image–text retrieval mAP), fairness metrics, and training efficiency.  

1.3 Significance  
By quantifying and acting on each data cluster’s true impact on FM performance, InfluenceSpace can:  
•   Reduce the training corpus size by 20–50% with minimal degradation (≤1%) in downstream metrics, thereby lowering compute requirements and carbon footprint.  
•   Mitigate model bias by up-weighting high-influence, under-represented clusters, improving fairness (e.g., gap reduction between demographic subgroups).  
•   Provide a general, modality-agnostic curation framework compatible with Retrieval-Augmented Generation (RAG) systems and future multi-modal agents.  
•   Advance the state of the art in data marketplaces and attribution by estimating cluster-level influence, informing fair compensation and licensing decisions.  

2. Methodology  

Our pipeline consists of three main stages: (1) cross-modal embedding and clustering, (2) influence score estimation with low-rank Hessian approximations, and (3) iterative curation via pruning and reweighting. We summarize in Algorithm 1.  

2.1 Data Collection and Preprocessing  
We assume access to a raw multi-modal corpus $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, where $x_i$ denotes an image and $y_i$ its associated text caption. We perform the following preprocessing steps:  
•   Standard image normalization (resize, center crop, pixel scaling).  
•   Text cleaning (lowercasing, punctuation removal, subword tokenization).  
•   Removal of exact duplicates via hash tables.  
Resulting in a cleaned dataset $\mathcal{D}'$.  

2.2 Stage 1: Cross-Modal Embedding and Clustering  
To group semantically similar samples across modalities, we leverage a pre-trained cross-modal encoder (e.g., CLIP or FLAVA). For each $(x_i,y_i)\in\mathcal{D}'$, we compute embeddings  
$$  
v_i = f_\mathrm{img}(x_i)\in\mathbb R^d,\quad  
t_i = f_\mathrm{text}(y_i)\in\mathbb R^d,  
$$  
and form concatenated embeddings $e_i = [v_i; t_i]\in\mathbb R^{2d}$. We then apply mini-batch $k$-means to $\{e_i\}$ to obtain $K$ clusters $\{\mathcal{C}_1,\ldots,\mathcal{C}_K\}$. The choice of $K$ balances granularity and computational cost; we set $K\approx 10^3$ based on preliminary experiments.  

2.3 Stage 2: Influence Score Estimation  
Rather than compute influence at the individual sample level (prohibitively expensive), we amortize across clusters. Let $\theta\in\mathbb R^p$ denote the FM parameters. Define the per-sample loss as $\ell(z_i;\theta)$ where $z_i=(x_i,y_i)$. The classic influence function for a single sample $i$ on the loss at test point $z_\mathrm{test}$ is  
$$  
\mathcal I_{i\to \mathrm{test}} = -\nabla_\theta \ell(z_\mathrm{test};\theta)^\top  
H_\theta^{-1}\nabla_\theta \ell(z_i;\theta),  
$$  
where $H_\theta=\frac{1}{N}\sum_i\nabla^2_\theta\ell(z_i;\theta)$ is the empirical Hessian of the training loss. Computing $H_\theta^{-1}$ is intractable for large $p$.  

We propose to approximate cluster-level influence by:  
1.  Low-rank Hessian approximation. We compute the top $r$ eigenpairs of $H_\theta$ using stochastic Lanczos, yielding eigenvalues $\{\lambda_j\}_{j=1}^r$ and eigenvectors $\{u_j\}_{j=1}^r$. Denote $\Lambda_r=\mathrm{diag}(\lambda_1,\dots,\lambda_r)$, $U_r=[u_1,\dots,u_r]$. Then  
$$  
H_\theta^{-1}\approx U_r\Lambda_r^{-1}U_r^\top + \frac{1}{\lambda_{r+1}}\big(I-U_rU_r^\top\big).  
$$  
2.  Mini-batch gradient statistics. For each cluster $\mathcal{C}_k$, we sample a small mini-batch $B_k\subset\mathcal{C}_k$ of size $b\ll|\mathcal{C}_k|$. Compute average gradient  
$$g_k = \frac{1}{b}\sum_{i\in B_k}\nabla_\theta\ell(z_i;\theta).$$  
3.  Influence score for cluster $k$ as the dot-product with a validation gradient $g_\mathrm{val}$:  
$$  
I_k = -\,g_\mathrm{val}^\top\bigl(U_r\Lambda_r^{-1}U_r^\top + \tfrac{1}{\lambda_{r+1}}(I-U_rU_r^\top)\bigr)g_k.  
$$  
Here $g_\mathrm{val}=\frac{1}{|\mathcal V|}\sum_{z\in\mathcal V}\nabla_\theta\ell(z;\theta)$ is the gradient on a held-out validation set $\mathcal V$. This formulation generalizes DataInf [1] to the multi-modal, cluster-level setting.  

2.4 Stage 3: Pruning and Reweighting Strategy  
Given influence scores $\{I_k\}_{k=1}^K$, we classify clusters into three buckets:  
•   Harmful (low or negative $I_k$): prune entirely.  
•   Neutral (small positive $I_k$): retain with uniform weight.  
•   Beneficial but under-represented: up-weight.  

We solve a constrained optimization:  
$$  
\max_{w\in\mathbb R_+^K}\quad \sum_{k=1}^K w_k\,I_k  
\quad\text{s.t.}\quad \sum_{k=1}^K w_k|\mathcal{C}_k|\le B,\quad  
w_k\le w_{\max},  
$$  
where $B$ is the total desired corpus size after curation and $w_{\max}$ caps the up-weight to avoid overfitting. This is a convex quadratic program when influence scores are positive; we solve via projected gradient descent.  

2.5 Iterative Curation Loop  
We iterate stages 2–3 for $T$ rounds: at each round, we fine-tune the FM on the newly reweighted corpus, recompute $g_\mathrm{val}$, update low-rank Hessian approximations, and re-evaluate cluster influences. This yields a dynamic curriculum that adapts data selection as model parameters evolve.  

2.6 Experimental Design and Evaluation Metrics  
Datasets  
–   MS COCO [2]: standard image–captioning pairs.  
–   Visual Genome [3]: dense region captions.  
–   Conceptual Captions [4]: noisy web-scraped captions.  

Baselines  
–   Random sampling (uniform down-sampling).  
–   Heuristic filtering (e.g., CLIP score thresholding).  
–   DataInf-style individual influence estimation without clustering.  

Tasks & Metrics  
1.  Image–Caption Retrieval: Recall@1/5/10.  
2.  Image Captioning: BLEU-4, METEOR, CIDEr.  
3.  VQA (on VQAv2): overall accuracy, per-question type accuracy.  
4.  Fairness Metrics: For a set of demographic attributes (gender, ethnicity) inferred via metadata or attribute classifiers, compute performance gap $\Delta=\max_{g,g'}|\,\mathrm{acc}(g)-\mathrm{acc}(g')|$.  
5.  Efficiency:  
    –   Data reduction ratio $\rho=1-\frac{|\mathcal{D}_\mathrm{curated}|}{|\mathcal{D}|}$.  
    –   Training time and FLOPs.  
    –   Hessian eigenpair computation time.  

Ablation Studies  
•   Effect of cluster count $K$ (e.g., $10^2,10^3,10^4$).  
•   Low-rank dimension $r$ (e.g., 10, 50, 100).  
•   Batch size $b$ for gradient estimation.  
•   Up-weight cap $w_{\max}$.  
•   Number of iterative rounds $T$.  

Hardware & Reproducibility  
Experiments run on 8 NVIDIA A100 GPUs. Code, cluster assignments, and curated splits will be open-sourced to ensure reproducibility.  

3. Expected Outcomes & Impact  

3.1 Anticipated Outcomes  
1.  Data Efficiency: We expect InfluenceSpace to achieve $20$–$50\%$ reduction in training corpus while maintaining or improving baseline FM performance (≤1% relative drop in major metrics).  
2.  Robustness Enhancement: By pruning harmful clusters (e.g., noisy or conflicting samples), we anticipate reduced overfitting and improved generalization to out-of-distribution data.  
3.  Fairness Improvement: Up-weighting under-represented clusters should decrease performance gaps $\Delta$ by at least $30\%$ across demographic subgroups in tasks like VQA.  
4.  Computational Savings: Training time and FLOPs should drop proportionally to data reduction, yielding up to $40\%$ lower compute cost.  

3.2 Broader Impact  
–   Resource Reduction: Smaller, high-utility corpora directly cut carbon emissions and training expenses.  
–   Data Marketplace Insights: Cluster‐level influence scores enable more granular attribution, informing fair compensation for data contributors and guiding legal frameworks around data copyright.  
–   Scalability: The hierarchical approach generalizes to future multi-modal agents and RAG pipelines, providing a blueprint for data-centric FM development.  
–   Community Adoption: By releasing an open-source toolkit, we empower practitioners to curate custom corpora tailored to specific domains (medical imaging, remote sensing, accessibility).  
–   Ethical Considerations: We will conduct an ethics review to ensure pruning does not inadvertently remove critical minority-group content or perpetuate harms. Our fairness metrics and up‐weighting strategy explicitly mitigate such risks.  

3.3 Alignment with DATA-FM Workshop Themes  
InfluenceSpace directly addresses multiple themes of the DATA-FM workshop:  
•   Data Collection & Curation: We provide a principled pipeline for filtering and mixing multi-modal data.  
•   Data Attribution: Cluster-level influence scores offer efficient attribution methods.  
•   Legal & Technical Solutions: By quantifying harmful data, we lay groundwork for copyright mitigation.  
•   Synthetic Data & Model Collapse: Our pruning strategy can prevent model collapse by removing redundant or degenerate clusters.  
•   Fairness & Social Impact: Explicit up-weighting of under-represented clusters enhances fairness.  
•   Benchmarks & Evaluations: We propose new evaluation protocols measuring influence-driven curation efficacy.  

In summary, InfluenceSpace presents a novel, interdisciplinary framework that unifies theoretical influence estimation, practical clustering techniques, and rigorous empirical evaluation to tackle the pressing data challenges in multi-modal foundation model development. We believe this work will catalyze further research at the intersection of data curation, model interpretability, and AI ethics.  

References  
[1] Kwon, Y., Wu, E., Wu, K., & Zou, J. (2023). DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models. arXiv:2310.00902.  
[2] Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.  
[3] Krishna, R. et al. (2017). Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. IJCV.  
[4] Sharma, P. et al. (2018). Conceptual Captions: A Cleaned, Hypernymed, Image Captioning Dataset. ACL.