# SynthTab – LLM-Driven Synthetic Tabular Data with Constraint-Aware Generation

## Introduction  
### Background  
Synthetic data generation has emerged as a critical solution to address data scarcity, privacy limitations, and bias mitigation in machine learning (ML) pipelines. However, existing approaches for tabular data often fail to capture intricate schema constraints (e.g., data types, uniqueness, referential integrity) and domain-specific semantics. For instance, autoregressive large language models (LLMs) struggle to enforce functional dependencies between columns due to their sequential generation process, while generative adversarial networks (GANs) and diffusion models sacrifice data utility when imposing privacy guarantees. Recent studies (e.g., TabuLa, HARMONIC) highlight these gaps: TabuLa (Zhao et al., 2023) improves training efficiency via token compression but lacks schema validation, while HARMONIC (Wang et al., 2024) introduces privacy metrics but does not enforce relational constraints.  

### Research Objectives  
This work proposes **SynthTab**, a multi-agent framework that combines LLMs with schema-aware validation and privacy-preserving mechanisms to generate high-fidelity synthetic tabular data. Specifically, we aim to:  
1. **Ensure schema compliance**: Generate data that adheres to domain-specific constraints (e.g., foreign key relationships, data type validity, business rules).  
2. **Preserve statistical properties**: Maintain feature distributions, correlations, and dependencies observed in real data.  
3. **Guarantee privacy**: Integrate differential privacy (DP) to bound information leakage while minimizing utility loss.  
4. **Enable iterative refinement**: Use feedback loops for error correction, improving synthetic data quality over iterations.  

### Significance  
Current synthetic data tools often produce invalid records (e.g., mismatched data types) or violate enterprise policies (e.g., leaking sensitive information). For domains like healthcare and finance, where data integrity and compliance are non-negotiable, such failures hinder adoption. SynthTab bridges this gap by:  
- **Bridging modality gaps**: Incorporating retrieval-augmented prompts (Adams et al., 2024) to inject domain semantics (e.g., SQL keywords, medical terminologies).  
- **Advancing constraint-aware generation**: Building on Schema-Constrained GM (Johnson et al., 2023) but integrating multi-agent verification for complex dependencies.  
- **Balancing privacy and utility**: Extending DP mechanisms from Doe and Smith (2023) to tabular data while preserving critical relationships (e.g., patient-disease associations).  

## Methodology  
### Research Design  

#### Data Collection and Preprocessing  
1. **Schema Input**: Collect metadata for the target table, including column names, data types, uniqueness constraints, reference keys (foreign keys), and business rules (e.g., "Age ≥ 18 for employment contracts").  
2. **Statistical Profiles**: Compute column-wise statistics:  
   - **Numerical**: Mean $ \mu_j $, standard deviation $ \sigma_j $, min/max values for column $ j $.  
   - **Categorical**: Frequency distributions $ P(c_{jk}) $ for category $ k $ in column $ j $.  
3. **Reference Corpus**: Curate a domain-specific knowledge corpus (e.g., medical guidelines, financial reports) to enable retrieval-augmented prompts.  

#### Algorithmic Framework  
SynthTab operates via three agents (Figure 1):  
1. **LLM Generator**: A fine-tuned LLM (e.g., Llama-3-8B) proposes candidate rows.  
2. **Schema Validator**: Enforces constraints using logical rules and chain-of-thought reasoning.  
3. **Quality Assessor**: Evaluates data fidelity and privacy, providing corrective feedback.  

**Algorithm 1: Iterative Constraint-Aware Generation**  
```python  
def synthtab(T_target, Schema, Stats, Corpus):  
    Initialize LLM with synthetic row generation instruction  
    Initialize Validator with Schema constraints  
    Initialize DP budget ϵ_total  
    Generate dataset D ← ∅  
    while D.size < T_target.size:  
        prompt_prompt ← RetrieveSimilarExamples(Corpus, Schema)  # Equation (1)  
        row_candidate ← LLM(prompt_prompt)  
        if Validator.check(row_candidate):  # Equation (2)  
            row_dp ← ApplyDP(row_candidate, ϵ_total)  
            D.add(row_dp)  
    return D  
```  

**Figure 1**: SynthTab Multi-Agent Framework  
![SynthTab Framework](synthtab_framework.png)  

#### Component Details  

##### 1. LLM Generator  
- **Fine-Tuning**: Use a hybrid dataset with rows from real tables and their schema statistics. Train objective:  
  $$  
  \mathcal{L}_{\text{LLM}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}, \text{Schema}, \text{Stats})  
  $$  
  where $ x_t $ is the token at position $ t $, and the context includes schema rules.  
- **Retrieval-Augmented Prompting**: Embed the target schema into vector space using a vision-language model and retrieve the $ K $-nearest examples from the reference corpus:  
  $$  
  \text{Prompt} = \text{Schema} \cup \text{Stats} \cup \{E_k\}_{k=1}^K  
  $$  
  where $ E_k $ are retrieved examples (Adams et al., 2024).  

##### 2. Schema Validator  
Enforces four constraint types:  
- **Data Type Validation**: Ensure values match data types (e.g., `DATE`, `VARCHAR(n)`) using regex patterns.  
- **Uniqueness**: Check keys (e.g., `customer_id`) are unique across rows.  
- **Referential Integrity**: Verify foreign keys exist in referenced tables.  
- **Business Rules**: Parse natural language constraints using LLM-driven logic (e.g., converting "Age ≥ 18" to a lambda function).  

The Validator checks each candidate row $ r $ as:  
$$  
\text{Validator}(r) = \bigwedge_{c \in \text{Constraints}} \text{Eval}(c, r)  
$$  
where $ \text{Eval}(c, r) $ returns `True` if constraint $ c $ holds for row $ r $.  

##### 3. Differential Privacy Mechanism  
Add DP noise to numerical columns using the Laplace mechanism:  
$$  
v' = v + \mathcal{L}(\Delta f / \epsilon)  
$$  
where $ \Delta f $ is the sensitivity of column $ v $, and $ \epsilon $ is the privacy budget allocated per row. For categorical features, use the exponential mechanism (Doe and Smith, 2023).  

#### Experimental Design  
1. **Datasets**:  
   - **Real-world**: 50+ tables from UCI Machine Learning Repository, Kaggle competitions, and enterprise financial/medical datasets.  
   - **Synthetic Benchmarks**: Generate tables with intentional constraint violations (e.g., mismatched dates, orphaned foreign keys) to test robustness.  

2. **Baselines**:  
   - TabuLa (Zhao et al., 2023)  
   - DoppelGANger (Privacy-GAN by Brown et al., 2024)  
   - Constraint-Aware GM (Lee et al., 2024)  

3. **Metrics**:  
   - **Schema Compliance**: \% of rows satisfying all constraints.  
   - **Statistical Fidelity**:  
     - Wasserstein distance $ W_1 $ between real/synthetic distributions.  
     - KL-divergence $ D_{\text{KL}}(P_{\text{real}} || P_{\text{synth}}) $.  
   - **Privacy**: Membership inference success rate (via Melis et al., 2023 attack).  
   - **Utility**: Accuracy of ML models (XGBoost, BERT) trained on synthetic data for classification tasks.  

4. **Ablation Studies**:  
   - Impact of LLM prompting strategies (with/without retrieval augmentation).  
   - Effectiveness of Validator in correcting errors (compare pre-validation vs. post-validation data quality).  

## Expected Outcomes & Impact  
### Anticipated Contributions  
1. **SynthTab Framework**: A generalizable pipeline for synthetic tabular data generation that:  
   - Enforces complex schema constraints (e.g., referential integrity, business rules).  
   - Integrates DP via gradient-based noise allocation.  
   - Uses feedback loops to iteratively refine synthetic samples.  
2. **Benchmark Evaluation**: Extensive experiments on 20+ real-world datasets, showing improvements in schema compliance (+25% vs. GANs) and downstream ML performance.  
3. **Open-Source Tools**: Release SynthTab as an open-access library with pre-trained LLMs for healthcare, finance, and enterprise domains.  

### Practical Implications  
- **Data Augmentation**: Enable training of ML models in low-data regimes (e.g., rare medical conditions) by generating valid synthetic samples.  
- **Safe Data Sharing**: Allow enterprises to share datasets with vendors while satisfying GDPR and HIPAA requirements.  
- **Domain-Specific Applications**: Tailor synthetic data to sectors like finance (generating transaction logs with audit constraints) and healthcare (preserving diagnosis-patient relationships).  

### Theoretical and Broader Impact  
- Advance understanding of how LLMs can encode and enforce structured constraints, addressing limitations in their autoregressive generation process (Xu et al., 2024).  
- Provide insights into the trade-off between DP noise calibration and statistical fidelity in structured data.  
- Influence policy frameworks for synthetic data governance by demonstrating practical compliance verification techniques.  

By solving the critical problem of constraint- and privacy-aware generation, SynthTab has the potential to accelerate adoption of synthetic data in regulated industries and enable safer exploration of large-scale ML on sensitive datasets.