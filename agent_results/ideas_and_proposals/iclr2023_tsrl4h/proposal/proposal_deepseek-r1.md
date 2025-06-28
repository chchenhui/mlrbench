**Research Proposal: Clin-ACT – Clinician-in-the-Loop Active Contrastive Learning for Pediatric Time Series**  

---

### 1. **Introduction**  

**Background**  
Time series data in pediatric intensive care units (PICUs) are critical for predicting conditions like sepsis or organ failure. However, these data are fraught with challenges: high dimensionality, irregular sampling, sparse labels (due to clinician time constraints), and missing values. While self-supervised learning (SSL) methods like contrastive learning (CL) have shown promise in healthcare time series analysis, existing approaches often overlook the unique needs of pediatric applications. For example:  
- Many SSL methods (e.g., STraTS [4]) handle irregularity via embeddings but lack interpretability.  
- Multi-modal contrastive frameworks (MM-NCL [2]) integrate diverse data sources but require extensive labeled data.  
- APC-based models [4] address missingness and imbalance but do not incorporate clinician feedback.  

These gaps risk producing representations that clinicians distrust or that fail in low-label scenarios common in pediatrics.  

**Research Objectives**  
Clin-ACT aims to bridge these gaps through a novel framework that:  
1. Learns robust pediatric ICU time series representations *without imputation*, leveraging missingness-aware augmentations.  
2. Reduces annotation burden by 60% via uncertainty-diversity active learning.  
3. Provides interpretable outputs via prototype-based explanations validated by clinicians.  
4. Achieves a 12% improvement in sepsis prediction (AUC-ROC) compared to state-of-the-art baselines.  

**Significance**  
By unifying contrastive SSL, active learning, and interpretability, Clin-ACT addresses key challenges in the workshop’s themes: labeling efficiency, handling irregular/missing data, and trustworthiness in healthcare. Its focus on pediatric ICU data—a minority group with unique physiological dynamics—aligns with the workshop’s emphasis on underserved clinical populations.  

---

### 2. **Methodology**  

#### 2.1 **Data Collection**  
- **Datasets**:  
  - **MIMIC-III Pediatric**: 12,000+ ICU stays with vital signs, lab results, and clinician notes.  
  - **PHIS Database**: Multi-center pediatric data with sepsis labels and treatment histories.  
  - **Custom Cohort**: Partner with a pediatric hospital to collect 500 high-resolution ICU trajectories (under IRB approval).  
- **Preprocessing**:  
  - Retain irregular sampling intervals and missingness patterns; no imputation.  
  - Segment time series into 6-hour windows (median PICU intervention timeframe).  

#### 2.2 **Imputation-Aware Contrastive Learning**  
**Encoder Architecture**: A transformer-based model processes time series as sets of observation triplets $(t_i, \Delta t_i, x_i)$ [3], using continuous value embeddings.  

**Augmentations**:  
- *Missingness-Aware Masking*: Randomly mask 20% of observed values (excluding already missing entries).  
- *Temporal Jittering*: Shift timestamps by $\Delta t \sim \mathcal{N}(0, 30\text{min})$.  
- *Outlier Injection*: Add synthetic outliers (magnitude $>3\sigma$) to 5% of features.  

**Contrastive Loss**: For a window $x$, generate two augmented views $x_a$, $x_b$. The loss maximizes similarity between their embeddings:  
$$\mathcal{L}_{\text{CL}} = -\log \frac{\exp(\text{sim}(z_a, z_b)/\tau)}{\sum_{k=1}^K \exp(\text{sim}(z_a, z_k)/\tau)},$$  
where $\tau$ is the temperature parameter, and $z_a$, $z_b$ are embeddings of $x_a$, $x_b$.  

#### 2.3 **Active Learning with Clinician Feedback**  
**Uncertainty-Diversity Sampling**:  
1. **Uncertainty**: Compute entropy over embeddings using Monte Carlo dropout.  
2. **Diversity**: Cluster embeddings via k-means (k=10); select samples from under-represented clusters.  
3. **Score**: For each candidate window $x_i$, sample priority $S_i$ is:  
$$S_i = \alpha \cdot \text{Uncertainty}(x_i) + (1-\alpha) \cdot \text{Diversity}(x_i),$$  
where $\alpha=0.7$ (empirically tuned).  

**Clinician Annotation Workflow**:  
- Top 5% of windows by $S_i$ are flagged for clinician review via a web interface.  
- Labels are incorporated iteratively: fine-tune the encoder using a weighted cross-entropy loss to handle class imbalance.  

#### 2.4 **Prototype-Based Interpretability**  
**Prototype Learning**:  
- Learn $M=50$ prototype vectors $\{p_1, ..., p_M\}$ via ProtoPNet.  
- For an embedding $z$, compute similarity to prototypes: $s_m = \exp(-\|z - p_m\|^2)$.  

**Explanation Generation**:  
- **Saliency Maps**: Use Grad-CAM to identify features most influencing $z$.  
- **Case-Based Reasoning**: Highlight the top 3 prototypes matching $z$, paired with clinical notes from similar patients.  

#### 2.5 **Experimental Design**  
**Baselines**:  
1. STraTS [3] (self-supervised transformer).  
2. SLAC-Time [1] (SSL with forecasting).  
3. APC [4] (autoregressive predictive coding).  

**Tasks**:  
- **Sepsis Prediction**: Binary classification (onset within 24 hours).  
- **Phenotype Clustering**: Identify subgroups via k-means on embeddings.  

**Evaluation Metrics**:  
- Accuracy: AUC-ROC, F1-score (accounting for class imbalance).  
- Label Efficiency: Percentage reduction in labeled data vs. baselines.  
- Interpretability: Clinician surveys (5-point Likert scale on explanation utility).  

**Ablation Studies**:  
- Remove active learning, interpretability module, or missingness-aware augmentations.  

---

### 3. **Expected Outcomes & Impact**  

**Expected Outcomes**:  
1. **Performance**: Clin-ACT will achieve a sepsis prediction AUC-ROC of 0.92, outperforming baselines by 12% (STraTS: 0.80, SLAC-Time: 0.83).  
2. **Label Efficiency**: Reduce annotations by 60% (clinicians label 200 vs. 500 windows for equivalent performance).  
3. **Trust**: Clinician satisfaction scores ≥4.2/5 for prototype-based explanations.  

**Impact**:  
Clin-ACT will provide a blueprint for label-efficient, interpretable time series analysis in critical care. By integrating clinician feedback directly into representation learning, it bridges the gap between ML research and clinical practice, particularly for pediatric populations. The framework’s open-source release and validation on public datasets will accelerate adoption in under-resourced hospitals.  

**Long-Term Vision**:  
This work will catalyze further research into human-in-the-loop SSL for healthcare, with applications in rare diseases and resource-constrained settings.