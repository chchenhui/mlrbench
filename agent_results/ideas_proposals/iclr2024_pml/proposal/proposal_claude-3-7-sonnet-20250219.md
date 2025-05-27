# Regulation-Sensitive Dynamic Differential Privacy for Federated Learning: Balancing Legal Compliance with Model Utility

## 1. Introduction

The widespread adoption of machine learning models in sensitive domains such as healthcare, finance, and social services has raised significant privacy concerns. These concerns are reflected in modern privacy regulations like the General Data Protection Regulation (GDPR), which impose stringent requirements on the collection, processing, and retention of personal data. While machine learning systems offer tremendous benefits, their data-intensive nature often conflicts with these regulatory frameworks, creating a tension between model utility and privacy compliance.

Federated Learning (FL) has emerged as a promising approach for training machine learning models without centralizing sensitive data. In FL, model training occurs locally on client devices, with only model updates being shared with a central server. This aligns with the data minimization principle in regulations like GDPR. However, even sharing model updates can leak sensitive information, as demonstrated by various inference attacks in the literature.

Differential Privacy (DP) provides a formal privacy guarantee by adding calibrated noise to computations involving sensitive data. When applied to FL, DP typically treats all data dimensions uniformly, allocating a global privacy budget across all features. This approach, while mathematically sound, fails to account for the varying sensitivity levels of different data attributes under regulatory frameworks. For instance, GDPR classifies personal data into general and "special categories" (e.g., health, biometric data) with distinct protection requirements.

This mismatch between uniform DP application and regulation-based categorization creates two problems: (1) excessive noise on less sensitive features, unnecessarily degrading model utility, and (2) potentially insufficient protection for highly sensitive attributes, risking regulatory non-compliance. Recent work by Xu et al. (2023) on Gboard language models and Kiani et al. (2025) on time-adaptive privacy spending has explored non-uniform privacy allocation across training rounds, but feature-specific privacy budgeting aligned with regulatory sensitivity remains underexplored.

Our research aims to bridge this gap by developing a regulation-sensitive dynamic differential privacy framework for federated learning. The proposed approach automatically categorizes data features based on their regulatory sensitivity, dynamically allocates privacy budgets accordingly, and implements a transparent audit mechanism to demonstrate compliance. By aligning privacy protection with regulatory requirements, we aim to maximize model utility while ensuring legal compliance.

The significance of this research is threefold. First, it enhances the practical applicability of privacy-preserving FL in regulated domains by respecting feature-specific sensitivity. Second, it improves model utility compared to uniform DP approaches while maintaining equivalent privacy guarantees for sensitive attributes. Third, it provides a transparent and auditable mechanism for demonstrating regulatory compliance, addressing the growing demand for accountable AI systems.

## 2. Methodology

Our proposed methodology consists of four main components: (1) Regulatory Sensitivity Tagging, (2) Dynamic Privacy Budget Allocation, (3) Secure Aggregation with Tailored Privacy, and (4) Compliance Audit Logging. Figure 1 illustrates the overall architecture of our framework.

### 2.1 Regulatory Sensitivity Tagging

The first step of our approach is to automatically classify data features according to their regulatory sensitivity. We propose a two-level tagging system:

1. **Metadata-Based Tagging**: We utilize dataset schema information and associated metadata to perform initial classification. Features with names or descriptions that match predefined regulatory categories (e.g., "health_status," "religion," or "financial_account") are tagged accordingly.

2. **NLP-Based Classification**: For features without clear metadata, we employ a lightweight natural language processing classifier to analyze feature names, descriptions, and a sample of data values. The classifier is pre-trained on regulatory texts and examples of data categorized by privacy laws.

Formally, for each feature $f_i$ in the dataset, we compute a sensitivity score $S(f_i) \in [0,1]$ as:

$$S(f_i) = \alpha \cdot S_{meta}(f_i) + (1-\alpha) \cdot S_{nlp}(f_i)$$

where $S_{meta}(f_i)$ is the metadata-based score, $S_{nlp}(f_i)$ is the NLP-based score, and $\alpha \in [0,1]$ is a weighting parameter.

Based on these scores, features are assigned to sensitivity tiers $T = \{t_1, t_2, ..., t_k\}$, where each tier corresponds to a specific regulatory category with an associated base privacy requirement. These tiers are mapped to relative privacy budget multipliers $M = \{m_1, m_2, ..., m_k\}$, where $m_1 > m_2 > ... > m_k$ and $m_i \in (0,1]$. Higher sensitivity tiers receive smaller multipliers (resulting in stronger privacy protection).

### 2.2 Dynamic Privacy Budget Allocation

Given a global privacy budget $\varepsilon_{global}$ and $\delta_{global}$, we dynamically allocate feature-specific privacy parameters. For each feature $f_i$ in sensitivity tier $t_j$ with multiplier $m_j$, we compute its privacy budget as:

$$\varepsilon_i = \varepsilon_{global} \cdot m_j \cdot w_i$$

where $w_i$ is a feature-specific weighting factor determined by the feature's importance to the learning task, calculated using feature importance metrics from pre-training analysis.

To ensure that the total privacy budget does not exceed the global limit, we normalize these allocations:

$$\varepsilon_i^{norm} = \varepsilon_i \cdot \frac{\varepsilon_{global}}{\sum_{i=1}^{d} \varepsilon_i}$$

where $d$ is the total number of features. Similarly, we compute feature-specific $\delta_i$ values normalized to $\delta_{global}$.

This approach allows for adaptive privacy protection that respects both regulatory sensitivity and model utility requirements.

### 2.3 Secure Aggregation with Tailored Privacy

Our federated learning system implements secure aggregation with tailored differential privacy guarantees. The protocol operates as follows:

1. **Client-Side Processing**: Each client $c$ computes local model updates $\Delta w_c$ based on their private data. Before transmission, the client applies feature-specific clipping and noise addition:

   a. For each feature dimension $i$, clip the corresponding gradient component to a bound $B_i$:
   $$\Delta w_c^i = \text{clip}(\Delta w_c^i, -B_i, B_i)$$

   b. Add calibrated Gaussian noise to each dimension based on its allocated privacy budget:
   $$\tilde{\Delta} w_c^i = \Delta w_c^i + \mathcal{N}(0, \sigma_i^2)$$
   
   where $\sigma_i = \frac{2B_i\sqrt{2\ln(1.25/\delta_i)}}{\varepsilon_i}$ for $(\varepsilon_i, \delta_i)$-differential privacy.

2. **Secure Aggregation**: The server aggregates the noisy updates using a secure aggregation protocol (e.g., based on homomorphic encryption or secure multi-party computation) to protect individual client contributions:
   $$\Delta W = \frac{1}{|C|}\sum_{c \in C} \tilde{\Delta} w_c$$
   
   where $C$ is the set of participating clients.

3. **Model Update**: The global model is updated with the aggregated, privacy-preserved gradients:
   $$w_{t+1} = w_t + \eta \cdot \Delta W$$
   
   where $\eta$ is the learning rate at round $t$.

### 2.4 Compliance Audit Logging

To ensure transparency and accountability, our framework maintains an immutable audit log for each training round. The log includes:

1. Feature sensitivity categorizations and their justifications
2. Privacy budget allocations for each feature
3. Aggregate privacy guarantees achieved in each round
4. Cryptographic proofs of the secure aggregation process
5. Compliance certificates for relevant regulatory requirements

We implement the audit log using a tamper-proof data structure (e.g., a Merkle tree) with digital signatures to ensure integrity and non-repudiation. Privacy administrators and auditors can verify the log to ensure compliance with regulatory requirements.

### 2.5 Experimental Design

To evaluate our proposed approach, we will conduct experiments on two real-world datasets with varying regulatory sensitivity profiles:

1. **Healthcare Dataset**: The MIMIC-III clinical dataset, containing patient records with highly sensitive attributes (diagnoses, procedures) and less sensitive demographic information.

2. **Financial Dataset**: A credit scoring dataset containing financial transaction records, account information, and demographic data with varying sensitivity levels.

We will implement our approach using TensorFlow Federated and compare it against the following baselines:

1. Non-private federated learning (No DP)
2. Standard federated learning with uniform differential privacy (Uniform DP)
3. Group-based differential privacy approaches (Group DP)
4. Time-adaptive privacy budget allocation (Time-Adaptive DP)

For each dataset and method, we will train standard machine learning models (logistic regression, MLP, CNN) and evaluate them using the following metrics:

1. **Utility Metrics**:
   - Prediction accuracy, precision, recall, F1-score
   - Area under ROC curve (AUC)
   - Mean squared error (for regression tasks)

2. **Privacy Metrics**:
   - Effective epsilon per feature
   - Success rate of membership inference attacks
   - Success rate of attribute inference attacks
   - Reconstruction error for sensitive attributes

3. **Regulatory Compliance Metrics**:
   - Compliance with GDPR data minimization principle
   - Protection level for special category data
   - Transparency and auditability score
   - Legal expert assessment score (through collaboration with legal experts)

We will conduct an ablation study to analyze the contribution of each component of our framework and perform sensitivity analysis for key hyperparameters, including:

1. The weighting parameter $\alpha$ in sensitivity scoring
2. The number of sensitivity tiers $k$
3. The multiplier values $M = \{m_1, m_2, ..., m_k\}$

### 2.6 Implementation Details

Our implementation will use the following technologies and frameworks:

1. **Federated Learning Framework**: TensorFlow Federated or FATE (Federated AI Technology Enabler)
2. **Differential Privacy Library**: TensorFlow Privacy or OpenDP
3. **Secure Aggregation**: Implementation based on the protocol by Bonawitz et al. with adaptations for feature-specific privacy
4. **NLP Classification**: BERT-based model fine-tuned on regulatory texts
5. **Audit Logging**: Custom implementation using a Merkle tree structure with digital signatures

The experiment will simulate a federated learning environment with 100 clients for each dataset, with non-IID data distribution to reflect real-world scenarios. We will run each experiment for 200 communication rounds and repeat each setting 5 times with different random seeds to ensure statistical significance.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

Our research is expected to yield several significant outcomes:

1. **Improved Privacy-Utility Trade-off**: We anticipate that our regulation-sensitive dynamic differential privacy approach will achieve up to 30% improvement in model utility compared to uniform DP approaches while maintaining equivalent privacy guarantees for sensitive attributes. This improvement will be particularly pronounced for datasets with high variance in feature sensitivity.

2. **Feature-Specific Privacy Guarantees**: Our framework will provide mathematically sound privacy guarantees that are tailored to the regulatory sensitivity of each feature, ensuring that special category data receives stronger protection while less sensitive attributes are subject to less noise.

3. **Auditable Compliance Mechanism**: The proposed audit logging system will provide transparent and verifiable evidence of regulatory compliance, addressing a critical gap in current privacy-preserving machine learning approaches.

4. **Quantitative Insights on Feature Sensitivity**: Our experiments will generate valuable data on the relationship between feature sensitivity, privacy budget allocation, and model utility across different domains, contributing to the broader understanding of privacy-preserving machine learning.

5. **Open-Source Implementation**: We will release an open-source implementation of our framework, enabling researchers and practitioners to adopt regulation-sensitive differential privacy in their federated learning systems.

### 3.2 Impact

The impact of this research extends across multiple dimensions:

1. **Practical Adoption of Privacy-Preserving FL**: By aligning differential privacy mechanisms with regulatory requirements, our approach makes privacy-preserving federated learning more practical for deployment in highly regulated domains such as healthcare and finance.

2. **Regulatory Compliance**: Our framework provides a clear pathway for organizations to demonstrate compliance with data protection regulations like GDPR, reducing legal risks associated with machine learning deployment.

3. **Enhanced Data Protection**: By providing stronger privacy guarantees for sensitive attributes while relaxing constraints on less sensitive features, our approach enhances protection for individuals' most sensitive information.

4. **Interdisciplinary Bridge**: This research builds a bridge between legal/regulatory perspectives on data privacy and technical implementations of privacy-preserving machine learning, fostering more effective collaboration between these disciplines.

5. **Future Research Directions**: Our work opens up new avenues for research on context-aware privacy mechanisms that respect not only regulatory sensitivity but also individual privacy preferences and cultural differences in privacy expectations.

In summary, our research on regulation-sensitive dynamic differential privacy for federated learning addresses a critical gap in current privacy-preserving machine learning approaches. By aligning technical privacy mechanisms with regulatory requirements, we aim to enhance both privacy protection and model utility, making privacy-preserving federated learning more practical for real-world deployment in regulated domains.