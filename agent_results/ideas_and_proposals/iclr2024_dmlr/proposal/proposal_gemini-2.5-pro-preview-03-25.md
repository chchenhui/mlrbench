## 1. Title: Uncertainty-Driven Model-Assisted Curation (UMC) for Scalable Multi-Domain Foundation Model Datasets

## 2. Introduction

### Background
The advent of large-scale foundation models (FMs) has marked a paradigm shift in machine learning, demonstrating remarkable capabilities across domains like natural language processing (NLP) and computer vision (CV) (Bommasani et al., 2021). However, the performance and generalization ability of these models are critically dependent on the quality, size, diversity, and provenance of their training data. As research increasingly pivots from purely model-centric advancements towards data-centric AI (DCAI) (Zha et al., 2023a; Zha et al., 2023b; Xu et al., 2024), the challenges associated with creating and maintaining high-quality, large-scale datasets have come to the forefront. This is particularly true when aiming to expand FMs beyond traditional domains or to build models capable of functioning across multiple modalities and application areas (Oala et al., 2023).

Constructing such multi-domain datasets presents significant hurdles. Manual curation is prohibitively expensive and slow, often struggling to scale to the massive data volumes required by FMs. Existing automated or model-assisted approaches, while faster, frequently suffer from limitations. They may rely on simplistic heuristics, fail to capture domain-specific nuances, or inundate human curators with vast quantities of low-value or redundant data points. Furthermore, ensuring adequate coverage across diverse domains and effectively handling data quality issues, dataset drift, and ethical considerations remain open challenges (Zha et al., 2023b; Xu et al., 2024). The lack of robust, efficient, and targeted data curation pipelines hinders the development of truly versatile and reliable foundation models that can serve a wide array of applications beneficial to humanity.

### Problem Statement
The core problem addressed by this research is the inefficient and often inadequate process of curating large-scale datasets for training multi-domain foundation models. Current methods struggle to balance scalability with quality, often failing to identify and prioritize the most informative samples for human review, especially across diverse and potentially low-resource domains. This leads to high annotation costs, datasets with uneven quality or domain coverage, and foundation models that may lack robustness or fail to generalize effectively to new or underrepresented areas. There is a critical need for intelligent, human-in-the-loop curation strategies that optimize the use of limited human annotation resources by focusing efforts on data points most likely to improve model performance, broaden domain understanding, and enhance overall data quality.

### Proposed Solution: Uncertainty-Driven Model-Assisted Curation (UMC)
We propose **Uncertainty-Driven Model-Assisted Curation (UMC)**, an iterative framework designed to streamline and enhance the creation of large-scale, high-quality, multi-domain datasets for foundation models. The central idea is to leverage model-estimated uncertainty as a signal to guide human annotation efforts towards the most valuable data samples. UMC employs an ensemble of pre-trained domain-specific models (or a diverse generalist FM) to assess unlabeled data, identifying samples where the models exhibit low confidence or high disagreement. These uncertain samples, likely representing challenging instances, domain boundaries, or novel concepts, are clustered for contextual relevance and presented to human curators via an interactive interface. The curated data is then used to iteratively refine the foundation model(s) and update uncertainty estimates. A multi-armed bandit (MAB) framework dynamically allocates annotation resources, balancing the need to explore new or underrepresented domains (exploration) with the need to improve performance on known difficult examples (exploitation).

### Research Objectives
This research aims to achieve the following primary objectives:

1.  **Develop the UMC Framework:** Design, implement, and refine the iterative UMC pipeline, including the ensemble-based uncertainty estimation module, sample selection and clustering mechanisms, the human-in-the-loop curation interface, the model retraining loop, and the MAB-based resource allocation strategy.
2.  **Evaluate Curation Efficiency:** Quantitatively assess the efficiency gains of UMC compared to baseline curation strategies (e.g., random sampling, standard active learning) in terms of annotation cost reduction (e.g., number of labels required for target performance) and human effort.
3.  **Assess Impact on Model Performance and Robustness:** Evaluate the quality of datasets curated using UMC by measuring the performance (e.g., accuracy, F1-score, domain-specific metrics) and robustness (e.g., performance under dataset shift, out-of-distribution generalization) of foundation models trained on these datasets.
4.  **Analyze Domain Coverage and Exploration:** Investigate the effectiveness of the MAB allocator in ensuring broad and balanced domain coverage within the curated dataset and its ability to adaptively focus resources on emerging or challenging domains.
5.  **Investigate Uncertainty Metrics:** Explore and compare different uncertainty quantification techniques (e.g., predictive entropy, margin sampling, ensemble variance, Monte Carlo dropout) within the UMC framework for identifying informative samples across diverse data types and domains.

### Significance
This research directly addresses critical challenges highlighted in the data-centric AI literature (Zha et al., 2023a; Xu et al., 2024) and aligns with the goals of the Data-centric Machine Learning Research workshop. By focusing on model-assisted dataset construction guided by uncertainty, UMC offers a path towards more scalable, cost-effective, and high-quality data curation. Expected contributions include:
*   **Methodological Advancement:** Providing a novel, principled framework (UMC) that integrates uncertainty estimation, active learning, human-computer interaction (HCI), and dynamic resource allocation for multi-domain dataset curation.
*   **Practical Tooling:** The potential development of an open-source toolkit based on UMC could empower researchers and practitioners to build better datasets for foundation models across various fields, including those beyond traditional vision and language (e.g., healthcare, climate science, graph data (Zheng et al., 2023), Earth observation (Najjar et al., 2024)).
*   **Improved Foundation Models:** Enabling the creation of more robust, reliable, and versatile foundation models capable of handling diverse tasks and domains, ultimately accelerating AI progress in service of humanity.
*   **Cost Reduction:** Significantly reducing the human annotation effort and cost associated with large-scale dataset creation (targeting 30-50% reduction), making advanced AI more accessible.
*   **Enhanced Data Quality:** Improving dataset quality by focusing human expertise on the most ambiguous and challenging samples, potentially leading to better handling of data imperfections and biases.

## 3. Methodology

### Overall Framework
The UMC framework operates as an iterative pipeline, depicted conceptually below:

**(Conceptual Flow):**

```mermaid
graph LR
    A[Unlabeled Data Pool D_U] --> B{Ensemble Uncertainty Scoring};
    B --> C{Identify Uncertain Samples D_uncert};
    C --> D{Cluster & Prioritize Samples};
    D --> E[Interactive Curation Interface];
    E -- Labeled/Verified Data D_labeled --> F{Update Curated Dataset D_C};
    F --> G{Retrain/Fine-tune Foundation Model(s)};
    G --> B;
    H[Multi-Armed Bandit Allocator] -- Controls --> D;
    H -- Observes Performance/Uncertainty --> H;
    E -- Feedback/Flags --> H;
```

The core steps within each iteration are:

1.  **Ensemble Uncertainty Scoring:** An ensemble of $M$ models, $\{f_1, f_2, ..., f_M\}$, scores a batch of unlabeled samples $X_{batch} \subseteq D_U$. These models could be diverse pre-trained specialists for different domains or tasks, or obtained through techniques like checkpoint averaging or Monte Carlo dropout on a single large FM.
2.  **Uncertainty Quantification:** For each sample $x \in X_{batch}$, calculate uncertainty metrics based on the ensemble's predictions. Key metrics include:
    *   **Predictive Entropy (Confidence):** Measures the uncertainty within individual model predictions, averaged over the ensemble. For a classification task with $K$ classes and model $m$ predicting probabilities $p_m(y|x)$, the entropy is $H_m(x) = -\sum_{k=1}^K p_m(y=k|x) \log p_m(y=k|x)$. The average entropy is $\bar{H}(x) = \frac{1}{M} \sum_{m=1}^M H_m(x)$. High entropy indicates low confidence.
    *   **Ensemble Disagreement (Variance/Divergence):** Measures the inconsistency across ensemble members. For classification, this can be the variance of predicted probabilities for each class, averaged over classes, or metrics like the average Kullback-Leibler (KL) divergence between pairs of model predictive distributions. Let $\bar{p}(y|x) = \frac{1}{M}\sum_{m=1}^M p_m(y|x)$ be the average prediction. Disagreement can be captured by $D(x) = \frac{1}{M} \sum_{m=1}^M KL(p_m(y|x) || \bar{p}(y|x))$. High disagreement suggests ambiguity or domain boundaries.
3.  **Sample Selection:** Select a subset of samples $D_{uncert} \subseteq X_{batch}$ exhibiting high uncertainty, e.g., those exceeding a threshold on $\bar{H}(x)$ or $D(x)$, or samples falling within the top percentile of uncertainty scores.
4.  **Clustering and Prioritization:** Apply unsupervised clustering (e.g., k-means or DBSCAN on feature embeddings extracted by the FM or ensemble) to $D_{uncert}$. This groups similar uncertain samples together, allowing curators to review related items efficiently. The MAB allocator (detailed below) prioritizes which clusters or domains to present for curation.
5.  **Interactive Human Curation:** Present prioritized samples (or clusters) to human annotators via an interactive interface (inspired by Saveliev et al., 2025 principles). The interface will display the sample, relevant context, model predictions (potentially with confidence scores), and uncertainty information. Curators can:
    *   Verify or correct labels.
    *   Flag ambiguous, irrelevant, or problematic samples.
    *   Provide rationales or additional metadata.
    This human feedback ($D_{labeled}$) is crucial for improving data quality and model understanding.
6.  **Dataset Augmentation:** Add the newly labeled/verified samples $D_{labeled}$ to the curated dataset $D_C$. Remove them from the unlabeled pool $D_U$.
7.  **Model Retraining:** Periodically retrain or fine-tune the foundation model(s) $f_m$ (or a central FM) using the updated curated dataset $D_C$. Techniques like continual learning might be employed to prevent catastrophic forgetting.
8.  **MAB-based Resource Allocation:** A Multi-Armed Bandit (MAB) framework manages the allocation of the limited annotation budget across different 'arms'.
    *   **Arms:** Arms can represent different data domains (e.g., medical images, legal text, financial time series), different clusters of uncertain data identified in Step 4, or different uncertainty criteria.
    *   **Reward:** The reward for selecting an arm (e.g., annotating samples from a specific domain/cluster) needs to reflect the value generated. Potential reward signals include:
        *   Improvement in model performance on a held-out validation set specific to that arm/domain.
        *   Reduction in average uncertainty for samples similar to those just labeled.
        *   A measure of novelty or diversity introduced by the new labels.
        *   Direct feedback from curators on the usefulness of the presented samples.
    *   **Policy:** Use a standard MAB algorithm like Upper Confidence Bound (UCB1) or Thompson Sampling. For UCB1, at time step $t$, select the arm $a$ that maximizes:
        $$ a_t = \arg \max_{a} \left( \hat{\mu}_a(t-1) + c \sqrt{\frac{\log t}{N_a(t-1)}} \right) $$
        where $\hat{\mu}_a(t-1)$ is the estimated average reward from arm $a$ up to step $t-1$, $N_a(t-1)$ is the number of times arm $a$ has been pulled, and $c$ is an exploration parameter controlling the trade-off between exploiting high-reward arms and exploring less-sampled ones. This dynamically balances focusing on known difficult areas versus discovering new challenges or underrepresented domains.

### Data Collection and Preparation
We will primarily utilize large, publicly available, unlabeled or weakly labeled datasets as the initial pool $D_U$. Examples include subsets of Common Crawl for text, LAION for images, public code repositories, and potentially combining domain-specific public datasets (e.g., PubMed abstracts for biomedical text, PhysioNet challenges for time-series data, geospatial data archives). The key is to start with a diverse, multi-domain collection representative of real-world data challenges. If necessary, we may synthetically create multi-domain scenarios by combining existing labeled datasets and treating their labels as unavailable initially, allowing for controlled experiments.

### Experimental Design
To validate the UMC framework, we will conduct rigorous experiments comparing its performance against several baseline data curation strategies:

1.  **Random Sampling (RS):** Samples are randomly selected from $D_U$ for annotation. This serves as a basic benchmark.
2.  **Standard Uncertainty Sampling (US):** Samples are selected based on the uncertainty estimate of a single foundation model (not an ensemble), typically using predictive entropy or least confidence.
3.  **Ensemble Uncertainty Sampling (Ens-US):** Samples are selected based on ensemble uncertainty (e.g., average entropy or disagreement) but without the MAB allocation or clustering components.
4.  **UMC (Proposed Method):** The full framework incorporating ensemble uncertainty, clustering, MAB-based dynamic allocation, and the interactive loop.

**Setup:**
*   **Datasets:** We will use diverse multi-domain dataset configurations. For instance:
    *   A text-based setup combining news articles (multiple topics), scientific papers (multiple fields), social media posts, and code snippets.
    *   A vision-based setup using diverse image datasets like ImageNet subsets, COCO, medical imaging archives (e.g., ChestX-ray8), and satellite imagery.
    *   A mixed-modality setup if feasible, combining text and images (e.g., using LAION or similar).
*   **Foundation Models:** We will experiment with readily available pre-trained foundation models relevant to the chosen domains (e.g., BERT/RoBERTa/GPT variants for text, ResNet/ViT variants for vision). The ensemble $\{f_m\}$ can be created using different pre-trained models or fine-tuned versions.
*   **Simulation:** Human annotation will be simulated. Given a fixed annotation budget (e.g., $B$ labels per iteration, or a total budget $B_{total}$), each strategy selects samples, which are then "annotated" using ground-truth labels (if available from the original dataset) or by actual human annotators for a smaller-scale qualitative study.
*   **Iterations:** The curation process will run for multiple iterations, allowing for model retraining and adaptive sampling.

**Evaluation Metrics:**
*   **Curation Efficiency:**
    *   **Performance vs. Budget:** Plot model performance (on a held-out test set) as a function of the number of labels acquired. Compare the curves for different strategies.
    *   **Cost Reduction:** Calculate the percentage reduction in labels needed by UMC to reach a target performance level compared to baselines.
    *   **Time/Effort:** (If using real annotators) Measure time per annotation, potentially assess cognitive load via surveys.
*   **Model Performance:**
    *   **Overall Accuracy/F1/AUC:** Standard metrics averaged across all domains/tasks.
    *   **Per-Domain Performance:** Track performance on specific domains to assess balance.
    *   **Zero-Shot/Few-Shot Performance:** Evaluate generalization to unseen or low-resource tasks/domains within the dataset.
*   **Robustness:**
    *   **Out-of-Distribution (OOD) Generalization:** Evaluate model performance on related but distinct OOD datasets.
    *   **Dataset Shift:** Introduce synthetic or natural distribution shifts (e.g., temporal drift, new domains added mid-curation) and measure performance degradation.
*   **Domain Coverage & Data Quality:**
    *   **Domain Representation:** Measure the distribution of labeled samples across predefined domains. Use entropy or Gini index to quantify balance.
    *   **Uncertainty Reduction:** Track the average uncertainty of the remaining unlabeled pool $D_U$ over iterations.
    *   **Label Quality:** (If using real annotators) Measure inter-annotator agreement for samples selected by different strategies. Assess the rate of flagged/problematic samples.

### Implementation Details
The framework will be implemented using Python and standard ML libraries such as PyTorch/TensorFlow for models, Scikit-learn for clustering and metrics, and potentially libraries like modAL for active learning components. The MAB will be implemented based on standard algorithms. For the interactive interface, tools like Label Studio or custom-built interfaces using web frameworks (e.g., Flask/React) could be employed for pilot studies with human curators. Computation will leverage GPU resources for model training and inference.

## 4. Expected Outcomes & Impact

### Expected Outcomes
1.  **A Validated UMC Framework:** A robust and well-documented implementation of the Uncertainty-Driven Model-Assisted Curation pipeline applicable to multi-domain data.
2.  **Demonstrated Efficiency Gains:** Quantitative evidence showing that UMC achieves significant annotation cost reduction (potentially 30-50% or more, as hypothesized) compared to baseline methods for reaching target levels of foundation model performance across multiple domains.
3.  **Improved Model Quality:** Proof that models trained on UMC-curated data exhibit superior performance, better domain generalization, and increased robustness to dataset shifts compared to models trained using data curated via alternative methods under the same annotation budget.
4.  **Effective Resource Allocation:** Demonstration of the MAB component's ability to dynamically balance exploration of new/underrepresented domains and exploitation of known difficult areas, leading to more balanced and comprehensive datasets.
5.  **Insights into Uncertainty Metrics:** Comparative analysis of different uncertainty estimation techniques for guiding curation across diverse data types, providing guidance on selecting appropriate metrics for different scenarios.
6.  **Benchmark Datasets (Potential):** Creation of challenging multi-domain benchmark datasets curated using UMC, potentially released to the community to facilitate further research in data-centric AI and active learning.
7.  **Publications and Code Release:** Dissemination of findings through publications in top-tier machine learning conferences/journals and potentially releasing the UMC framework as open-source software.

### Impact
The successful completion of this research is expected to have a significant impact on the field of machine learning and its applications:

*   **Accelerating Foundation Model Development:** By drastically reducing the cost and effort required for high-quality data curation, UMC can accelerate the development and deployment of powerful foundation models in a wider range of scientific, industrial, and societal domains, including those currently hindered by data scarcity or poor data quality (e.g., personalized medicine, climate modeling, social sciences).
*   **Enhancing AI Reliability and Trustworthiness:** Datasets curated with UMC are expected to yield models that are more robust and generalize better, contributing to more reliable and trustworthy AI systems. The focus on uncertain samples may also help uncover and address hidden biases or edge cases more effectively.
*   **Advancing Data-Centric AI Methodologies:** This work will contribute novel techniques to the growing field of data-centric AI, particularly in model-assisted labeling, active learning for large models, and human-AI collaboration in data curation (Saveliev et al., 2025). The integration of ensemble uncertainty and MABs for dynamic, multi-domain curation represents a significant methodological step forward.
*   **Democratizing Access to High-Quality Data:** By making large-scale data curation more efficient, UMC could lower the barrier to entry for smaller research groups or organizations aiming to build specialized foundation models or adapt existing ones to specific domains.
*   **Informing Best Practices:** The findings will provide practical insights and guidelines for practitioners on how to best leverage model uncertainty and human expertise to build high-quality datasets efficiently, potentially influencing platforms and benchmarks like DataPerf and DataComp mentioned in the workshop description.
*   **Supporting Interdisciplinary Research:** The ability to build versatile multi-domain FMs more easily will foster interdisciplinary research by enabling AI tools that can effectively process and integrate information from diverse sources and modalities.

In summary, the UMC framework promises a principled and efficient approach to address the critical bottleneck of data curation for modern foundation models. By intelligently guiding human effort using model uncertainty and adaptive resource allocation, this research aims to significantly advance the scalability, quality, and domain-reach of large-scale AI systems.

### References (Implicitly used based on Literature Review)
*   Bommasani, R., et al. (2021). On the Opportunities and Risks of Foundation Models. arXiv:2108.07258.
*   Najjar, H., Nuske, M., & Dengel, A. (2024). Data-Centric Machine Learning for Earth Observation: Necessary and Sufficient Features. arXiv:2408.11384.
*   Oala, L., et al. (2023). DMLR: Data-centric Machine Learning Research -- Past, Present and Future. arXiv:2311.13028.
*   Saveliev, E., Liu, J., Seedat, N., Boyd, A., & van der Schaar, M. (2025). Towards Human-Guided, Data-Centric LLM Co-Pilots. arXiv:2501.10321. (*Note: Year adjusted based on typical arXiv pre-print conventions, assuming intended year*)
*   Xu, X., et al. (2024). Data-Centric AI in the Age of Large Language Models. arXiv:2406.14473.
*   Zha, D., Bhat, Z. P., Lai, K.-H., Yang, F., & Hu, X. (2023a). Data-centric AI: Perspectives and Challenges. arXiv:2301.04819.
*   Zha, D., Bhat, Z. P., Lai, K.-H., Yang, F., Jiang, Z., Zhong, S., & Hu, X. (2023b). Data-centric Artificial Intelligence: A Survey. arXiv:2303.10158.
*   Zheng, X., Liu, Y., Bao, Z., Fang, M., Hu, X., Liew, A. W.-C., & Pan, S. (2023). Towards Data-centric Graph Machine Learning: Review and Outlook. arXiv:2309.10979.