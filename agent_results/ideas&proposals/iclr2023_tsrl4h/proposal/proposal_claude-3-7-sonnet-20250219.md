# Clin-ACT: Clinician-in-the-Loop Active Contrastive Learning for Pediatric Time Series Data

## 1. Introduction

### Background

Pediatric intensive care units (PICUs) generate vast amounts of multivariate time series data from vital signs, laboratory tests, and other clinical measurements. These data streams present unique challenges: they are high-dimensional, irregularly sampled, contain missing values, and are sparsely labeled due to the significant time constraints faced by clinical experts. Despite their potential value for critical tasks such as early sepsis detection, seizure prediction, and personalized treatment planning, the effective utilization of these data remains limited by several factors.

Traditional supervised learning approaches face significant barriers in this domain. First, they require substantial amounts of labeled data, which is particularly scarce in pediatric settings where annotation requires specialized expertise. Second, they often struggle with the irregular nature of clinical time series, where measurements may be taken at varying intervals depending on patient condition and clinical protocols. Third, they typically lack the transparency and interpretability necessary for clinical adoption, functioning as "black boxes" that provide predictions without explanation.

Self-supervised representation learning has emerged as a promising approach to address the label scarcity problem by leveraging unlabeled data. However, existing self-supervised methods for time series data often overlook the unique characteristics of clinical data, such as missingness patterns and outliers, which can contain important diagnostic information. Furthermore, these methods rarely incorporate clinician feedback or provide the level of interpretability required for clinical decision support.

### Research Objectives

This research proposes Clin-ACT (Clinician-in-the-Loop Active Contrastive Learning for Time Series), a novel framework designed to address the specific challenges of pediatric time series data. Our objectives are:

1. To develop a robust representation learning method for pediatric time series that explicitly accounts for irregularity, missingness, and outliers without requiring extensive preprocessing or imputation.

2. To design an active learning strategy that efficiently targets the most informative segments of time series data for clinical annotation, thereby maximizing the value of limited expert time.

3. To incorporate a prototype-based interpretability layer that maps learned representations to clinically meaningful archetypes and provides feature saliency maps to explain model decisions.

4. To validate the framework on real pediatric ICU data, demonstrating improvements in downstream task performance and clinician satisfaction compared to existing approaches.

### Significance

The proposed research addresses critical gaps in time series representation learning for pediatric healthcare. By combining contrastive self-supervision with active learning and interpretability mechanisms, Clin-ACT aims to deliver several significant contributions:

1. **Improved Label Efficiency**: We expect Clin-ACT to reduce the required annotation burden by approximately 60% while maintaining or improving model performance, making advanced analytics more feasible in resource-constrained clinical environments.

2. **Enhanced Interpretability**: The prototype-based approach will provide clinicians with transparent insights into model behavior, potentially increasing trust and adoption of machine learning systems in clinical practice.

3. **Pediatric-Specific Methodology**: By focusing on the unique challenges of pediatric data, this research addresses an underrepresented domain in healthcare AI, where models developed for adults may not transfer effectively.

4. **Broader Applicability**: While focusing on pediatric ICU data, the methods developed will be relevant to other clinical time series applications characterized by irregularity, sparsity, and limited expert availability.

## 2. Methodology

### Overview

Clin-ACT comprises three integrated components: (1) a contrastive self-supervised learning framework with specialized augmentations for clinical time series, (2) an active learning module that selects informative time windows for clinician annotation, and (3) a prototype-based interpretability layer that facilitates clinician understanding of the learned representations. Figure 1 illustrates the overall architecture of Clin-ACT.

### Data Collection and Preprocessing

We will utilize multivariate time series data from pediatric ICU patients, including:
- Vital signs (heart rate, respiratory rate, blood pressure, oxygen saturation)
- Laboratory results (complete blood count, chemistry panels)
- Medication administration records
- Clinical assessments and interventions

Data will be acquired from existing pediatric ICU databases, with appropriate IRB approval and de-identification procedures. We will include patients aged 0-18 years with ICU stays of at least 24 hours.

Minimal preprocessing will be applied to preserve authentic clinical patterns:
1. Normalization of each variable to zero mean and unit variance
2. Segmentation into overlapping windows of fixed duration (e.g., 6-hour windows with 1-hour step size)
3. Retention of explicit timestamps and missingness indicators rather than imputation

### Contrastive Self-Supervised Learning with Clinical Augmentations

The core of our representation learning approach is a contrastive framework that learns meaningful embeddings by enforcing consistency between differently augmented views of the same time series segment.

#### Encoder Architecture

We will employ a temporal encoder $f_θ$ consisting of:
1. A temporal embedding layer that accounts for irregular sampling
2. Stacked Transformer encoder blocks with multi-head attention
3. A global pooling operation followed by a projection head

The encoder maps a multivariate time series segment $x$ to a d-dimensional embedding vector $z = f_θ(x)$.

#### Clinically-Informed Augmentations

We introduce a set of augmentations specifically designed for clinical time series that preserve diagnostic information while creating diverse views:

1. **Missingness-Aware Masking**: Randomly mask values while accounting for existing missingness patterns:
   $$\tilde{x}_{i,t} = \begin{cases} 
   \text{missing}, & \text{with probability } p_{\text{mask}} \cdot (1 - m_{i,t}) \\
   x_{i,t}, & \text{otherwise}
   \end{cases}$$
   where $m_{i,t}$ is an indicator for whether feature $i$ at time $t$ is already missing.

2. **Clinical Range Preservation**: Apply jittering within clinically acceptable ranges:
   $$\tilde{x}_{i,t} = x_{i,t} + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim \mathcal{N}(0, \sigma_i^2)$$
   where $\sigma_i$ is set proportionally to the clinically acceptable range for feature $i$.

3. **Temporal Warping**: Apply non-uniform time warping that preserves critical events:
   $$\tilde{t} = t + \delta(t) \cdot (1 - c(t))$$
   where $\delta(t)$ is a random temporal offset and $c(t)$ is a clinical importance score derived from the rate of change and deviation from normal ranges.

4. **Modality Dropout**: Randomly drop entire measurement modalities (e.g., all lab values) to encourage robust cross-modal representations.

#### Contrastive Loss Function

We employ a modified InfoNCE loss that accounts for the temporal proximity of samples:

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\exp(\text{sim}(z_i, z_i^+)/\tau) + \sum_{j \neq i} w_{ij} \cdot \exp(\text{sim}(z_i, z_j)/\tau)}$$

where:
- $z_i$ and $z_i^+$ are embeddings of two augmented views of the same time window
- $\text{sim}(u, v) = u^T v / \|u\|\|v\|$ is the cosine similarity
- $\tau$ is a temperature parameter
- $w_{ij}$ is a weighting factor that decreases for negative samples that are temporally close to the anchor sample

### Active Learning for Efficient Clinical Annotation

To minimize clinician annotation burden while maximizing information gain, we develop an active learning strategy that selects the most informative time windows for expert review.

#### Uncertainty-Diversity Criterion

We propose a hybrid selection criterion that balances representation uncertainty with diversity:

$$\text{score}(x_i) = \alpha \cdot \text{uncertainty}(x_i) + (1-\alpha) \cdot \text{diversity}(x_i)$$

where:
- $\text{uncertainty}(x_i) = 1 - \max_k P(y_i = k | x_i)$ for classification tasks
- $\text{diversity}(x_i) = \min_{j \in \mathcal{L}} d(z_i, z_j)$, where $\mathcal{L}$ is the set of already labeled samples and $d$ is a distance function in the embedding space
- $\alpha \in [0,1]$ is a balancing hyperparameter

For regression tasks, uncertainty can be estimated using ensemble variance or estimated prediction intervals.

#### Batch Selection Process

Active learning proceeds in batches:
1. Train initial representation model on unlabeled data
2. Select top-k windows according to the selection criterion
3. Present selected windows to clinicians for annotation
4. Fine-tune model with newly labeled data
5. Repeat steps 2-4 until annotation budget is exhausted or performance plateaus

### Prototype-Based Interpretability Layer

To enhance interpretability, we develop a prototype layer that maps learned representations to clinically meaningful concepts.

#### Prototype Learning

We learn a set of $M$ prototypes $\{p_1, p_2, ..., p_M\}$ that represent archetypal patterns in the embedding space. The similarity between an embedding $z$ and prototype $p_m$ is given by:

$$s_m(z) = \exp\left(-\frac{\|z - p_m\|^2}{\sigma^2}\right)$$

where $\sigma$ is a scaling parameter.

For classification tasks, the class probabilities are computed as:

$$P(y = k | x) = \frac{\sum_{m \in \mathcal{M}_k} s_m(f_θ(x))}{\sum_{m=1}^M s_m(f_θ(x))}$$

where $\mathcal{M}_k$ is the set of prototypes associated with class $k$.

#### Feature Attribution

For each prototype, we generate feature attribution maps that indicate which input features contribute most to the similarity with that prototype. We adapt Integrated Gradients for this purpose:

$$\text{attr}_i(x, p_m) = (x_i - x'_i) \cdot \int_{\alpha=0}^1 \frac{\partial s_m(f_θ(x' + \alpha(x-x')))}{\partial x_i} d\alpha$$

where $x'$ is a baseline input (e.g., all zeros or population means).

#### Clinical Prototype Refinement

To ensure prototypes align with clinical understanding, we incorporate clinician feedback:
1. Present prototypes and their feature attributions to clinicians
2. Allow clinicians to refine prototype definitions and associations
3. Update the prototype layer with clinician input
4. Repeat refinement process iteratively

### Experimental Design and Evaluation

#### Datasets

We will evaluate Clin-ACT on two pediatric ICU datasets:
1. A proprietary dataset from a large children's hospital (with IRB approval)
2. The publicly available MIMIC-IV pediatric subset

#### Downstream Tasks

We will assess the learned representations on several clinically relevant tasks:
1. Early sepsis prediction (binary classification)
2. Length of stay prediction (regression)
3. Mortality risk assessment (binary classification)
4. Physiological deterioration prediction (binary classification)

#### Evaluation Protocol

For each downstream task:
1. Split data into training (60%), validation (20%), and test (20%) sets, stratified by outcome
2. Train Clin-ACT on the training set (self-supervised phase)
3. Apply active learning to select samples for annotation
4. Fine-tune on labeled examples for the specific task
5. Evaluate performance on the test set

#### Evaluation Metrics

For classification tasks:
- Area Under the Receiver Operating Characteristic curve (AUROC)
- Area Under the Precision-Recall Curve (AUPRC)
- F1-score
- Sensitivity and specificity at clinically relevant thresholds

For regression tasks:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Concordance index

For active learning efficiency:
- Learning curve showing performance vs. number of labels
- Annotation time reduction compared to random selection

For interpretability:
- Clinician rating of prototype relevance (1-5 scale)
- Clinician rating of feature attribution accuracy (1-5 scale)
- Time required for clinicians to understand model decisions

#### Baseline Methods

We will compare Clin-ACT against:
1. Fully supervised learning with various model architectures
2. Standard self-supervised approaches (e.g., SimCLR, BYOL adapted for time series)
3. Imputation-based methods (e.g., using GRU-D or BRITS for imputation before modeling)
4. Random selection for active learning comparison

#### Clinical Evaluation

To assess real-world utility:
1. Conduct a user study with 10-15 pediatric intensivists
2. Present cases with model predictions and explanations
3. Collect feedback on interpretability, trust, and potential clinical impact
4. Measure agreement between model-identified important features and clinician judgment

## 3. Expected Outcomes & Impact

### Expected Outcomes

1. **Technical Advances**:
   - A novel contrastive learning framework with clinical augmentations for pediatric time series data
   - An effective active learning strategy that reduces annotation requirements by approximately 60%
   - A prototype-based interpretability layer with clinically meaningful explanations

2. **Performance Improvements**:
   - Target improvement of 12-15% in early sepsis detection compared to standard supervised learning
   - Expected 8-10% improvement in physiological deterioration prediction
   - Significant reduction in the number of false alarms while maintaining high sensitivity

3. **Practical Outputs**:
   - An open-source implementation of the Clin-ACT framework
   - A set of validated clinical prototypes for common pediatric ICU patterns
   - Guidelines for integrating active learning into clinical annotation workflows

4. **Knowledge Contributions**:
   - Insights into the specific challenges of pediatric time series compared to adult data
   - Understanding of which time series features and patterns are most informative for different clinical predictions
   - Assessment of clinician preferences for model interpretability in time-critical settings

### Impact

1. **Clinical Impact**:
   - Enhanced decision support tools for pediatric intensivists
   - More efficient use of clinician time through targeted annotation
   - Potential for earlier detection of critical conditions, leading to improved patient outcomes

2. **Methodological Impact**:
   - Advancement of representation learning techniques for irregular, sparse clinical time series
   - New approaches to incorporating clinical domain knowledge into self-supervised learning
   - Bridge between purely data-driven and clinician-guided methods

3. **Broader Impact**:
   - Extension of methods to other underrepresented healthcare domains
   - Potential application to other data-limited areas beyond pediatrics (e.g., rare diseases)
   - Contribution to the broader goal of trustworthy AI in healthcare

4. **Ethical Considerations**:
   - Improved model transparency addressing ethical concerns about black-box models in healthcare
   - Potential to reduce healthcare disparities by enabling advanced analytics in resource-limited settings
   - Framework for responsible deployment of AI systems in sensitive pediatric care contexts

By developing a clinician-centered approach to time series representation learning, Clin-ACT aims to advance both the technical capabilities and practical applicability of machine learning in pediatric critical care, ultimately supporting better clinical decisions and improved patient outcomes.