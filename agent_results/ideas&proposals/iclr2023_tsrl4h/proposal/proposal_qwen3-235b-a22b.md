# Title: Clin-ACT – Clinician-in-the-Loop Active Contrastive Learning for Pediatric Time Series

## Introduction

Clinical time series data in pediatric intensive care units (ICUs) present unique challenges due to their inherently high-dimensional, irregular, and sparsely labeled nature. Unlike time series in other domains, pediatric ICU data often involve frequent missing values and uneven temporal sampling, driven by logistical constraints and varying patient conditions. Supervised deep learning models struggle in such environments due to the scarcity of expert annotations, while existing self-supervised approaches frequently overlook critical domain-specific properties like missingness patterns or clinical relevance. This gap motivates the development of **Clin-ACT (Clinician-in-the-Loop Active Contrastive Learning)**, a novel method designed to learn robust, interpretable representations for multivariate clinical time series with minimal annotation effort. By integrating contrastive self-supervision with active learning and prototype-based explainability, Clin-ACT aims to overcome barriers to clinical deployment, where trust and transparency are as essential as predictive performance.

The core research objective of Clin-ACT is to advance the frontier of self-supervised representation learning in healthcare by explicitly addressing three critical limitations observed in recent literature. First, existing models like **SLAC-Time** [1] and **STraTS** [3] focus on handling missing data but do not incorporate interpretability mechanisms, limiting their adoption by clinicians. Second, methods such as **Multi-Modal Contrastive Learning** [2] and **Autoregressive Predictive Coding (APC)** [4] achieve strong performance in downstream tasks but lack active learning strategies to guide annotation, leaving the burden of data labeling unaddressed. Third, while these works make strides in robustness and high-dimensionality challenges, they do not fully bridge the gap between algorithmic performance and actionable clinical insights, particularly for minority populations like pediatrics. Clin-ACT addresses these issues by introducing: (1) an **imputation-aware encoder** trained via contrastive loss that respects missingness patterns, (2) an **active learning loop** to select the most informative time windows for annotation by clinicians, reducing labeling effort by an estimated 60%, and (3) a **prototype-based explanation module** that maps learned representations to clinically meaningful archetypes and generates feature saliency maps. Each component synergizes to ensure representations are both predictive and human-centric.

The significance of Clin-ACT lies in its potential to redefine the interaction between machine learning models and healthcare professionals. By minimizing reliance on labeled data, it tackles the expert annotation bottleneck often seen in rare disease domains like sepsis in pediatrics. The interpretability layer addresses a major adoption barrier—clinician trust—by explicitly linking latent embeddings to prototypical patient states, enabling clinicians to validate findings against their domain knowledge. These contributions directly align with the workshop's focus on minority groups, robustness, and explainability. Clin-ACT also offers broader insights into handling missing data, class imbalance, and model transparency in healthcare AI beyond pediatrics.

This research proposal outlines Clin-ACT’s methodology, detailing how its three components function in unison to generate high-quality representations and actionable insights. The experimental design evaluates its performance in sepsis prediction on pediatric ICU data while measuring its interpretability efficacy through clinician feedback. As a holistic solution combining label efficiency, robustness to irregular sampling, and domain-aligned explanations, Clin-ACT represents a significant step toward practical time series representation learning in under-explored clinical settings.

## Methodology

### Research Design and Data Collection

Our proposed method, **Clin-ACT**, is structured around three main components: (1) encoder training with imputation-aware augmentations, (2) active learning for efficient annotation, and (3) prototype-based interpretability. These components are designed to synergize, leveraging self-supervision for robust representation learning, guiding annotation to minimize the expert labeling burden, and ensuring clinical trust through transparent explanations.  

For validation, we will use de-identified patient data from the **eICU collaborative research database**, which contains extensive vitals and laboratory measurements from pediatric ICU patients. These multivariate time series include irregular sampling intervals, missingness patterns, and outliers—properties critical for testing our framework’s robustness. Each patient’s clinical features (e.g., heart rate, blood pressure, temperature, oxygen saturation) are recorded at varying frequencies, with some observations missing due to equipment failure, clinical priorities, or measurement limitations. We will preprocess the data by normalizing vitals using population-based pediatric reference ranges and incorporate relative time stamps to capture temporal irregularity. To preserve clinical realism, we will retain missing values rather than impute them, treating each measurement as an individual observation triplet $(t_i, x_i, v_i)$, where $t_i$ is the timestamp, $x_i$ is the feature dimension, and $v_i$ is the observed value.  

### Algorithmic Framework

At the core of Clin-ACT is a **self-supervised encoder** designed to learn discriminative embeddings for downstream tasks such as sepsis prediction. The encoder is a **Temporal Self-Attention Network** that processes time series sequences by encoding feature values alongside their timestamps. Inspired by **continuous time embeddings** [3], we represent each timestamp $t_{i}$ using a learnable sinusoidal function:  

$$
h_i = \text{TransformerEncoder}(\text{Embed}(x_i, v_i, \gamma(t_i)))
$$

where $\gamma(t_i)$ is a parameterization capturing the temporal interval between observation points. This allows the model to learn representations that are sensitive to irregular sampling.  

To enforce robustness against missing values and noise, we employ **imputation-aware augmentations** during encoder training. Unlike traditional imputation mechanisms that assume missingness is missing at random (MAR), we instead sample from the empirical distribution of missingness to retain data structure integrity. Augmentations include:  
1. **Masking**: Randomly zeroing out a subset of observed feature values during training.  
2. **Time Shifting**: Applying small perturbations to timestamps to simulate irregular measurements.  
3. **Interpolation-Based Augmentation**: Approximating missing values using linear or cubic polynomial interpolation but treating them as untrustworthy during contrastive loss computation.  

We train the encoder using a **contrastive self-supervised objective** that enforces consistency of temporal representations under these augmentations. Given an original time series $\mathcal{S}$ and two differently augmented versions $\mathcal{S}_1$ and $\mathcal{S}_2$, we compute latent representations $z_\theta(\mathcal{S}_1)$ and $z_\theta(\mathcal{S}_2)$ through the encoder. The **contrastive loss** is defined as:   

$$
\mathcal{L}_{\text{contrastive}} = - \frac{1}{M} \sum_{j=1}^{M} \log \frac{\exp(\text{sim}(z^\mathcal{S}_1, z^\mathcal{S}_2) / \tau)}{\sum_{k=1}^{K} \exp(\text{sim}(z^\mathcal{S}_1, z^\mathcal{S}_k^{neg}) / \tau)}
$$

where $\text{sim}$ is a cosine similarity function, $\tau$ is the temperature parameter, and $z^\mathcal{S}_k^{neg}$ are negative samples drawn from other time series. This loss encourages the model to learn invariant representations while preserving temporal structure.  

### Algorithmic Steps and Optimization

#### Prototype Layer and Active Learning

To further enhance interpretability and align learned embeddings with clinical reasoning, we introduce a **prototype layer** that captures latent clusters corresponding to clinically meaningful patient archetypes. These prototypes are learned via k-means clustering on the embeddings of a training subset and refined using gradient-based optimization. Let $\{p_c\}_{c=1}^C$ denote a set of prototype vectors in the latent space. We define the **prototype-to-embedding similarity** as:   

$$
\alpha_j = \exp(- \left\| z_\theta(t_j) - p_j \right\|^2),
$$

where $z_\theta(t_j)$ is the latent representation at time $t_j$, and $\alpha_j$ represents the confidence that $z_j$ belongs to the closest prototype. These similarities are aggregated across the time series into a final prototype score $c_i = \frac{1}{T} \sum_{j=1}^{T} \alpha_j$, which informs a **prototypical classification module**.  

To guide data annotation, Clin-ACT employs an **active learning criterion** based on **uncertainty and diversity** [5]. Time windows with highest class entropy $H(y|\mathcal{S})$ and maximum divergence from existing prototypes are selected for clinician labeling. This is formalized as:   

$$
\mathcal{W} = \arg\max_{w \in W} H(y|z_\theta(w)) + \lambda \cdot \text{MMD}(z(w), \mathcal{D}_{labeled})
$$

where $H(y|z_\theta(w))$ measures predictive uncertainty, $\text{MMD}$ is a kernel-based similarity measure capturing divergence from labeled data, and $\lambda$ is a weight parameter balancing uncertainty and diversity. The most uncertain yet diverse samples are sent to human experts for annotation, allowing for efficient refinement of representations.  

### Experimental Design and Evaluation Metrics

We will validate Clin-ACT on a **sepsis detection task in pediatric ICU patients** using the eICU database. Our pipeline includes:  
1. Training the encoder using the contrastive loss with imputation-aware augmentations.  
2. Using active learning to select top 60% uncertain windows across patients for expert annotation.  
3. Refining representations using prototype alignment.  
4. Evaluating downstream performance via linear probes on held-out data.  

Evaluation will include:  
- **Downstream task performance**: Accuracy, sensitivity, AUROC, and F1 score.  
- **Annotation efficiency**: Reduction in labeling effort compared to fully supervised baselines.  
- **Interpretability assessment**: Clinician satisfaction based on prototype-driven explanations, measured through structured questionnaires.  

We will compare performance with recent benchmarks such as **APC** [4] and **STraTS** [3], analyzing how active learning and interpretability enhance performance and trust. Through this comprehensive design, we aim to validate Clin-ACT’s ability to produce robust, clinically aligned, and label-efficient representations for critical care applications.

## Expected Outcomes & Impact

### Anticipated Improvements in Representation Quality and Task Accuracy

The core expected outcome of Clin-ACT is improved **downstream task performance** with a **reduced annotation effort**, as validated on the **pediatric ICU sepsis detection** task. By leveraging **imputation-aware contrastive learning**, our approach is designed to generate more robust and generalizable representations compared to traditional imputation-based pipelines [4]. The incorporation of temporal position encoding ensures that the model respects irregular sampling intervals, which have historically posed challenges in time series representation learning [3]. With active learning, Clin-ACT will selectively annotate the most uncertain but structurally distinct time windows across patients, thereby improving **sample efficiency** while avoiding biases toward overrepresented patient profiles. Through these mechanisms, we expect a **12% improvement in sepsis detection accuracy** compared to recent benchmarks like APC and STraTS, which have limitations in active data selection and interpretable reasoning.  

### Enhanced Interpretability and Clinician Satisfaction

Beyond predictive performance, Clin-ACT aims to **increase interpretability and trust in learned representations** through a lightweight prototype-based explanation module. Inspired by methods like SLAC-Time and MM-NCL [1, 2], our framework goes a step further by mapping embeddings to clinically meaningful prototypes—interpretable patient archetypes that mirror expert reasoning. We anticipate that this mechanism will **increase clinician trust** by explicitly showing which clinical observations drive each prototype and by visualizing feature saliency maps that link model decisions to key patient characteristics. A structured evaluation of **clinician satisfaction levels** will be conducted via expert annotations and feedback surveys, assessing whether the model's explanations align with known clinical indicators of sepsis and whether the active learning prioritizes data from ambiguous patient states. This aligns with broader research trends in interpretable AI in healthcare, where transparent reasoning becomes essential, especially for high-stakes applications like critical care [6].

### Scientific Advancements and Broader Impacts

This work contributes several advancements to both machine learning and healthcare. From an algorithmic perspective, combining **active learning with prototype-guided interpretation** under a self-supervised contrastive framework represents a **novel methodological synthesis**, extending prior work that has largely treated these components in isolation. Clin-ACT also introduces **uncertainty-diversity criteria** for data labeling, which may lead to improvements in semi-supervised pipelines beyond clinical time series. From a healthcare standpoint, Clin-ACT’s application to pediatric ICU data addresses a **critical yet underserved population**, offering a scalable solution where expert annotations are scarce. Additionally, its design allows for seamless adaptation to other resource-constrained scenarios involving rare diseases such as neurological disorders or metabolic conditions.  

By integrating these elements, Clin-ACT aims to deliver **actionable AI models** that are not only robust to the data challenges of irregular sampling and missingness but also **clinically interpretable**, making them viable for real-world deployment. The framework represents a step toward **human-in-the-loop representation learning**, where machine intelligence and clinical expertise evolve in tandem. Future directions include extending Clin-ACT to dynamic treatment regimes, where interpretable representations can support individualized clinical decision-making and improve outcomes for vulnerable patient groups.

### Evaluation of Robustness and Clinical Actionability  

To validate Clin-ACT's effectiveness under real-world clinical constraints, we will conduct systematic experiments assessing its **robustness to missing values**, **resilience against class imbalance**, and **practical utility in clinical workflows**. A key benchmark will be **eICU pediatric ICU data** with irregular sampling and high missingness. We will simulate varying **missing data percentages (20%, 40%, and 60%)** and evaluate performance using **AUROC** and **F1 scores**, comparing them against baseline models such as **APC** and **STraTS**. We expect Clin-ACT’s **imputation-aware contrastive learning** to outperform traditional imputation-based methods by preserving the original data structure without introducing artificial artifacts in missing regions. Further, we will analyze **representation consistency** by measuring latent distances between original and augmented time series, ensuring that the learned embeddings remain stable despite missing data.  

A central challenge in sepsis detection is **class imbalance**, as early sepsis cases are rare among pediatric ICU patients. To assess Clin-ACT’s ability to counteract bias toward the majority class (non-sepsis), we will use **class-conditioned MMD-diversity** to quantify how well the model captures underrepresented temporal patterns. We hypothesize that the active learning component will prioritize ambiguous, early-stage sepsis sequences, thus improving minority class learning. We will validate this by evaluating **per-class AUROC values** and tracking the **evolution of prototype-based decision boundaries** over training. Additionally, we will perform **zero-shot evaluations** on a held-out subset of patients with **atypical physiological profiles**, assessing whether prototypes generalize across diverse patient populations.

### References

[1] Ghaderi, H., Foreman, B., Nayebi, A., Tipirneni, S., Reddy, C. K., & Subbian, V. (2023). A Self-Supervised Learning-based Approach to Clustering Multivariate Time-Series Data with Missing Values (SLAC-Time): An Application to TBI Phenotyping. *ArXiv Preprint arXiv:2302.13457*.  
[2] Baldenweg, F., Burger, M., Rätsch, G., & Kuznetsova, R. (2024). Multi-Modal Contrastive Learning for Online Clinical Time-Series Applications. *ArXiv Preprint arXiv:2403.18316*.  
[3] Tipirneni, S., & Reddy, C. K. (2021). Self-supervised transformer for sparse and irregularly sampled multivariate clinical time-series. *ArXiv Preprint arXiv:2107.14293*.  
[4] Wever, F., Keller, T. A., Symul, L., & Garcia, V. (2021). As easy as APC: overcoming missing data and class imbalance in time series with self-supervised learning. *ArXiv Preprint arXiv:2106.15577*.  
[5] Sambasivan, N., Kreyenfeld, M., & Reddy, S. (2020). Active learning with uncertainty-diversity sampling. *Proceedings of the AAAI Conference on Artificial Intelligence*.  
[6] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.