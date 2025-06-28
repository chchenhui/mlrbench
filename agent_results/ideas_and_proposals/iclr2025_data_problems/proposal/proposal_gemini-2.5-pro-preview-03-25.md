## 1. Title: InfluenceSpace: Scalable Hierarchical Influence-Driven Data Curation for Multi-Modal Foundation Models

## 2. Introduction

**2.1 Background**
Foundation Models (FMs), particularly large language models (LLMs) and increasingly multi-modal variants, have revolutionized machine learning, demonstrating remarkable capabilities across diverse tasks (Vaswani et al., 2017; Radford et al., 2021; Singh et al., 2021 - FLAVA). Their success is inextricably linked to the massive datasets used for their pre-training. However, the sheer scale and heterogeneity of these datasets present significant challenges. Raw web data, commonly used for training, is often replete with noise, redundancy, biases, and potentially copyrighted material (Birhane et al., 2021). As noted by the DATA-FM workshop, addressing these data-related challenges – covering collection, curation, attribution, legality, and societal impact – is paramount for the responsible and effective development of FMs.

Traditional data curation methods often rely on heuristics, such as simple filtering based on metadata, rule-based cleaning, or basic deduplication. While somewhat effective, these methods lack a principled way to quantify the actual contribution of individual data points or subsets to the model's learning objectives and downstream performance. This is especially critical for multi-modal FMs (e.g., vision-language models like CLIP, FLAVA) where the interplay between modalities adds complexity, and data imbalances can severely impact fairness and robustness (Liang et al., 2024 - HEMM). Poor curation can lead to wasted computational resources, slower convergence, models that inherit societal biases, and suboptimal performance on key tasks.

Influence functions (Koh & Liang, 2017) offer a theoretically grounded approach to estimate the effect of individual training points on model predictions or parameters. Recent work like DataInf (Kwon et al., 2023) has made progress in efficiently approximating influence scores for large models, particularly in parameter-efficient fine-tuning scenarios. However, applying point-wise influence estimation directly to the massive pre-training datasets of multi-modal FMs remains computationally prohibitive due to the scale ($N \sim 10^8 - 10^9$) and model size ($p \sim 10^9 - 10^{12}$). Furthermore, point-wise analysis may overlook the collective impact of semantically related data groups.

**2.2 Research Idea: InfluenceSpace**
We propose **InfluenceSpace**, a novel, scalable, hierarchical pipeline for influence-driven data curation specifically designed for large-scale multi-modal FM pre-training. Our core idea is to shift from computationally intractable point-wise influence estimation to a more manageable *cluster-level* influence analysis. InfluenceSpace operates in two main stages, potentially iterated:

1.  **Cross-Modal Semantic Clustering:** Leverage powerful pre-trained multi-modal embeddings (e.g., from CLIP or FLAVA) to project the raw data (e.g., image-text pairs) into a shared embedding space. Apply efficient clustering algorithms (e.g., Mini-Batch K-Means) to group data points into semantically coherent clusters. This stratification captures the thematic structure of the dataset.
2.  **Amortized Cluster Influence Scoring:** Compute an *influence score* for each cluster, estimating its aggregate impact on model performance or specific objectives (e.g., performance on a representative validation set, fairness metrics). To ensure scalability, we will employ computationally efficient techniques:
    *   **Amortized Computation:** Utilize gradient information readily available during standard training phases.
    *   **Approximation Techniques:** Employ methods like low-rank Hessian approximations (inspired by DataInf and similar works like KFAC) to drastically reduce the cost of influence calculation.
    *   **Mini-batch Sampling:** Estimate cluster gradients and influence using representative mini-batch samples from each cluster, avoiding full dataset passes.
3.  **Hierarchical Curation:** Based on these cluster influence scores, prune clusters identified as detrimental (negative influence) or redundant (near-zero influence). Conversely, up-weight or prioritize sampling from clusters exhibiting high positive influence, particularly those corresponding to under-represented but valuable semantic concepts, thereby potentially improving model robustness and fairness (inspired by Chameleon's goals, Erfanian et al., 2024).

This hierarchical approach – clustering followed by cluster-level influence assessment – provides a principled way to manage data complexity at scale. By operating on clusters, we significantly reduce the number of entities for which influence needs to be estimated, making the approach tractable for massive datasets.

**2.3 Research Objectives**
The primary objectives of this research are:

1.  To develop and implement the InfluenceSpace pipeline, integrating cross-modal clustering and efficient, amortized cluster-level influence estimation techniques for multi-modal data.
2.  To devise and validate scalable algorithms for approximating cluster influence scores using low-rank Hessian approximations and mini-batch gradient sampling within the context of multi-modal FM training.
3.  To experimentally evaluate the effectiveness of InfluenceSpace using standard vision-language benchmarks and FMs (e.g., fine-tuning CLIP/FLAVA or training smaller variants).
4.  To quantify the trade-offs achieved by InfluenceSpace in terms of:
    *   Data volume reduction (curation ratio).
    *   Model performance (accuracy, retrieval metrics, VQA scores, zero-shot capabilities).
    *   Training efficiency (convergence speed, computational cost reduction).
    *   Fairness and bias mitigation (performance across demographic groups, representation analysis).
5.  To compare InfluenceSpace against standard baseline curation strategies (e.g., random sampling, heuristic filtering) to demonstrate its advantages.

**2.4 Significance**
This research directly addresses several critical challenges highlighted by the DATA-FM workshop:

*   **Data Collection and Curation:** Provides practical, scalable strategies for curating massive multi-modal datasets tailored to FM training, moving beyond simple heuristics.
*   **Scalability:** Tackles the computational bottleneck of influence functions for web-scale data by introducing a hierarchical cluster-based approach with efficient approximations.
*   **Data and Society (Fairness):** Incorporates mechanisms to potentially mitigate bias by identifying and up-weighting high-influence clusters from under-represented groups.
*   **Benchmarks and Evaluations:** Contributes to understanding data value and provides a methodology that can be evaluated using standard FM benchmarks.

Successfully developing InfluenceSpace would offer a more principled, scalable, and potentially more effective framework for data curation in the era of large-scale multi-modal FMs. This could lead to significant reductions in the computational resources and time required for training state-of-the-art models, making FM development more accessible and sustainable. Furthermore, by enabling targeted curation based on influence, it holds the potential to produce models that are not only performant but also more robust and fair, contributing positively to the broader impact of AI.

## 3. Methodology

**3.1 Research Design Overview**
Our methodology follows a constructive and empirical approach. We will first develop the algorithmic components of the InfluenceSpace pipeline, integrating clustering and efficient influence estimation. We will then implement this pipeline and conduct extensive experiments on representative multi-modal datasets and foundation models to validate its effectiveness and compare it against relevant baselines.

**3.2 Data Collection and Preprocessing**

*   **Datasets:** We will primarily use large-scale, publicly available image-text datasets commonly employed for training multi-modal FMs. Potential candidates include subsets of LAION-400M/LAION-5B (Schuhmann et al., 2021; 2022), Conceptual Captions (CC3M, CC12M) (Sharma et al., 2018; Changpinyo et al., 2021), and potentially YFCC100M. Using subsets (e.g., 10M-100M scale) will allow for rigorous experimentation within reasonable resource constraints while still demonstrating scalability. We will use standard splits or create our own train/validation/test splits, ensuring no overlap, particularly guarding against test set contamination. For downstream evaluation, standard benchmarks like COCO Captions (Chen et al., 2015), Flickr30k (Young et al., 2014) for retrieval, VQA v2 (Goyal et al., 2017) for visual question answering, and ImageNet (Deng et al., 2009) for zero-shot classification will be used.
*   **Preprocessing:** Standard preprocessing will be applied: Images will be resized and normalized. Text captions will be tokenized using the tokenizer associated with the chosen foundation model. Basic filtering (e.g., removing entries with corrupt data, filtering based on image/text size) might be applied initially.

**3.3 InfluenceSpace Pipeline: Algorithmic Details**

Let $D = \{(x_i, y_i)\}_{i=1}^N$ be the initial large-scale multi-modal dataset, where $x_i$ is an image and $y_i$ is its corresponding text caption. Let $M(\theta)$ be the multi-modal foundation model with parameters $\theta$. Let $L(z, \theta)$ be the loss function for a data point $z=(x, y)$ (e.g., contrastive loss for CLIP-like models). Let $D_{val}$ be a smaller, clean validation set representative of downstream tasks or desired capabilities.

**Stage 1: Cross-Modal Embedding and Clustering**

1.  **Embedding:** Obtain joint or aligned embeddings for each pair $(x_i, y_i)$ using a pre-trained multi-modal encoder $E(\cdot)$, such as CLIP's image and text encoders. Let $e_i = [E_{img}(x_i); E_{txt}(y_i)]$ or use a method that yields a single joint embedding. This results in a set of embedding vectors $\{e_i\}_{i=1}^N$.
2.  **Clustering:** Apply a scalable clustering algorithm to the embeddings $e_i$. Mini-Batch K-Means is a suitable candidate due to its efficiency on large datasets.
    $$ \min \sum_{k=1}^K \sum_{e_i \in C_k} ||e_i - \mu_k||^2 $$
    where $K$ is the number of clusters (a hyperparameter, e.g., $K \sim \sqrt{N}$ or determined empirically) and $\mu_k$ is the centroid of cluster $C_k$. This partitions the dataset $D$ into $K$ clusters $C_1, \dots, C_K$.

**Stage 2: Amortized Cluster Influence Estimation**

Our goal is to estimate the influence of removing or re-weighting an entire cluster $C_k$ on the performance over the validation set $D_{val}$. Following Koh & Liang (2017), the influence of a training point $z_{train}$ on the loss at a test point $z_{test}$ is approximated by:
$$ \mathcal{I}_{loss}(z_{test}, z_{train}) = -\nabla L(z_{test}, \hat{\theta})^T H_{\hat{\theta}}^{-1} \nabla L(z_{train}, \hat{\theta}) $$
where $\hat{\theta}$ are the converged parameters and $H_{\hat{\theta}} = \frac{1}{N} \sum_{i=1}^N \nabla^2 L(z_i, \hat{\theta})$ is the Hessian of the training loss. Direct computation is infeasible ($O(Np^2 + p^3)$).

We adapt this to clusters and introduce approximations:

1.  **Cluster Gradient:** Represent the gradient impact of a cluster $C_k$ by the average gradient over a random mini-batch $B_k \subset C_k$:
    $$ \nabla L(C_k, \theta) \approx \frac{1}{|B_k|} \sum_{z \in B_k} \nabla L(z, \theta) $$
2.  **Validation Gradient:** Similarly, represent the validation objective direction using the average gradient over $D_{val}$ (or a mini-batch $B_{val} \subset D_{val}$):
    $$ \nabla L(D_{val}, \theta) \approx \frac{1}{|B_{val}|} \sum_{z \in B_{val}} \nabla L(z, \theta) $$
3.  **Hessian Approximation:** Avoid computing and inverting the full Hessian $H_{\theta}$. We will explore efficient approximations $\tilde{H}_{\theta}$ such that $\tilde{H}_{\theta}^{-1} v$ can be computed quickly:
    *   **Fisher Information Matrix (FIM) / Empirical FIM:** As used in KFAC (Martens & Grosse, 2015), which admits efficient inversion through block-diagonal or Kronecker-factored approximations. FIM is often used as an approximation of the Hessian.
    *   **Low-Rank Approximation:** Leverage techniques similar to DataInf (Kwon et al., 2023) if applicable, potentially focusing on approximations relevant to the LoRA parameters if fine-tuning, or using broader low-rank methods.
    *   **Diagonal Approximation:** A simpler baseline, approximating $H_{\theta}$ with its diagonal.
    Let $\tilde{H}_{\theta}^{-1}$ denote the efficiently computable inverse approximation.
4.  **Amortized Influence Score for Cluster $C_k$:** We define the influence score $I_k$ based on the alignment between the cluster's gradient and the validation gradient, mediated by the approximate inverse Hessian:
    $$ I_k \approx -\nabla L(D_{val}, \theta)^T \tilde{H}_{\theta}^{-1} \nabla L(C_k, \theta) $$
    These scores are computed "amortized" by using gradients ($\nabla L(z, \theta)$) and potentially Hessian information (e.g., activations needed for KFAC) collected during intermediate stages of model training (e.g., after a certain number of epochs or periodically). This reuses computation already performed for optimization.

**Stage 3: Hierarchical Curation**

Based on the computed influence scores $I_k$:

1.  **Pruning:** Define a threshold $\epsilon_{prune}$. Clusters $C_k$ with $I_k < \epsilon_{prune}$ (indicating negative or negligible influence on the validation objective) are removed from the training set.
2.  **Up-weighting/Sampling:** Define a threshold $\epsilon_{upweight}$. Clusters $C_k$ with $I_k > \epsilon_{upweight}$ (high positive influence Baskets) may have their sampling probability increased during subsequent training epochs.
3.  **Fairness Consideration:** (Optional but important) If metadata associated with clusters (e.g., inferred dominant concepts, or demographic attributes if available and ethically permissible) indicates potential under-representation of certain groups, we can adjust the thresholds or apply a bonus to the influence scores of high-performing clusters from these groups. For example, modify the up-weighting criterion to favor high-influence clusters representing rare concepts or minority groups. This aims to retain valuable data that might otherwise be sparsely represented.
4.  **Iteration:** The process (Train -> Cluster -> Score -> Curate -> Train) can be iterated. Re-clustering and re-scoring after further training might refine the curation process as the model's representation evolves.

**3.4 Experimental Design**

1.  **Model Architecture:** We will use established multi-modal architectures like CLIP (ViT-B/32 or similar) or potentially FLAVA. We may fine-tune publicly available checkpoints or train smaller versions from scratch on the curated vs. baseline datasets to isolate the effect of curation. Training will use standard optimizers (e.g., AdamW) and learning rate schedules.
2.  **Baselines:**
    *   **Full Dataset:** Train on the (subset of) the original dataset without specific curation (after basic preprocessing).
    *   **Random Pruning:** Randomly discard data points to match the data reduction M achieved by InfluenceSpace.
    *   **Heuristic Curation:** Implement common heuristics like:
        *   Embedding Deduplication: Remove near-duplicates based on embedding similarity.
        *   CLIP Score Filtering: Remove image-text pairs with low CLIP similarity scores.
3.  **Evaluation Tasks:**
    *   **Intrinsic:** Monitor training dynamics (loss curves, convergence speed). Measure final training time/FLOPs.
    *   **Downstream Performance:** Evaluate the trained/fine-tuned models on:
        *   Zero-Shot Image Classification (e.g., ImageNet accuracy).
        *   Image-Text Retrieval (e.g., Recall@K on Flickr30k, COCO).
        *   Visual Question Answering (e.g., VQA score on VQA v2, if applicable to the model).
    *   **Fairness/Bias:**
        *   Evaluate performance disparities on fairness benchmarks (e.g., FairFace attribute classification accuracy gaps, performance variations across concepts in retrieval tasks).
        *   Analyze the distribution of semantic concepts (possibly derived from cluster centroids or using external classifiers) in the curated dataset versus the original. Quantify representation bias reduction. Use evaluation frameworks like HEMM (Liang et al., 2024) as inspiration for comprehensive multi-modal fairness assessment.
4.  **Ablation Studies:** We will analyze the contribution of different components:
    *   Effect of different clustering algorithms or number of clusters $K$.
    *   Impact of different Hessian approximation methods ($\tilde{H}_{\theta}^{-1}$).
    *   Sensitivity to pruning/up-weighting thresholds ($\epsilon_{prune}, \epsilon_{upweight}$).
    *   Impact of the iteration frequency.

**3.5 Evaluation Metrics**

*   **Performance:** Accuracy, Top-k Accuracy, Recall@K, F1-score, VQA score (task-dependent).
*   **Efficiency:**
    *   Data Reduction Ratio: $1 - (|D_{curated}| / |D_{original}|)$.
    *   Training Time: Wall-clock time, steps/epochs to convergence.
    *   Computational Cost: Estimated FLOPs.
*   **Fairness:**
    *   Performance Gap: Difference in performance metrics between different demographic or semantic groups.
    *   Bias Metrics: Statistical measures of representation (e.g., KL divergence of concept distributions between curated and original data). Metrics like Equal Opportunity, Demographic Parity applied to downstream tasks.
*   **Influence Quality (Proxy):** Correlation between approximate cluster influence scores $I_k$ and the actual change in validation performance observed when removing/up-weighting the cluster (can be tested on smaller scale).

## 4. Expected Outcomes & Impact

**4.1 Expected Outcomes**

1.  **A Novel Curation Pipeline:** We expect to deliver a fully implemented and validated InfluenceSpace pipeline as a proof-of-concept for scalable, influence-driven multi-modal data curation.
2.  **Empirical Validation:** We anticipate demonstrating that InfluenceSpace can significantly reduce the size of large multi-modal training datasets (e.g., potentially 20-50% reduction or more) while maintaining or even improving downstream task performance compared to training on the full dataset or using baseline curation methods like random sampling.
3.  **Efficiency Gains:** We expect to show measurable improvements in training efficiency, reflected in reduced training time and potentially faster convergence due to focusing computation on higher-impact data.
4.  **Fairness Improvements:** We hypothesize that the targeted up-weighting mechanism within InfluenceSpace will lead to demonstrable improvements in fairness metrics, reducing performance gaps across different groups or concepts compared to baseline approaches.
5.  **Insights into Data Value:** The research will provide valuable insights into the relationship between semantic cluster properties, their influence scores, and their contribution to multi-modal FM performance and fairness. This contributes to a deeper understanding of "what data matters" in large-scale training.
6.  **Open Source Contribution (Potential):** We aim to release an implementation of the InfluenceSpace pipeline to facilitate reproducibility and further research by the community.

**4.2 Impact**

*   **Scientific Impact:** This research pushes the boundary of data-centric AI by proposing a scalable, principled method for data curation in the challenging domain of multi-modal FMs. It bridges the gap between theoretical influence function research and practical large-scale data management, offering a novel hierarchical approach. It directly addresses core themes of the DATA-FM workshop, contributing to the understanding and mitigation of data problems in FM development.
*   **Practical Impact:** If successful, InfluenceSpace could offer significant practical benefits to researchers and practitioners developing FMs. By reducing data requirements without sacrificing performance, it can lower the immense computational costs (and associated carbon footprint) of training these models. This could democratize FM development to some extent. Furthermore, the integrated approach to improving fairness through data curation provides a practical tool for building more responsible AI systems.
*   **Societal Impact:** By enabling the development of potentially fairer and more robust FMs with reduced resources, this work contributes to the positive societal impact of AI. Addressing bias through principled data curation is crucial as FMs become more integrated into real-world applications.
*   **Future Work:** This research opens avenues for future work, including extending InfluenceSpace to other modalities (e.g., audio, video), integrating it with synthetic data generation, exploring its use for data attribution at the cluster level, and developing more sophisticated methods for fairness-aware influence modulation.

In conclusion, InfluenceSpace offers a promising direction for tackling critical data curation challenges in the era of multi-modal foundation models. Its focus on scalable, influence-driven, and fairness-aware data management aligns perfectly with the goals of the DATA-FM workshop and holds significant potential for advancing the field.