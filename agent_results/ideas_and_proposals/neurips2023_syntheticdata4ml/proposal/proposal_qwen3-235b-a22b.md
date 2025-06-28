# Differentially Private and Fair Tabular Data Synthesis via Constrained Large Language Models  

---

## 1. Introduction  

### Background  
Synthetic data generation has emerged as a transformative solution to address critical challenges in machine learning (ML) development: **data scarcity**, **privacy risks**, and **bias amplification**. In high-stakes domains like healthcare and finance, access to high-quality training data is often restricted due to legal, ethical, or logistical barriers. Even when data exists, biases in representation (e.g., underrepresentation of minority groups) can perpetuate inequities in ML-driven decision-making systems. Large language models (LLMs), with their capacity to generate contextually rich and semantically coherent outputs, offer a promising avenue for synthetic tabular data generation. However, naively applying LLMs to this task risks leaking sensitive information from training data and reinforcing biases present in the original datasets.  

Differential privacy (DP) provides a formal framework for quantifying privacy guarantees by limiting the influence of individual data points on model outputs. Despite recent advancements in DP tabular data synthesis—such as DP-TBART (2023) and DP-LLMTGen (2024)—these methods often neglect fairness constraints. Conversely, fairness-aware synthesis techniques (e.g., [6]) rarely incorporate DP, leaving gaps in holistic, trustworthy ML pipelines. Meanwhile, the unique capabilities of LLMs for tabular generation remain underexplored in the context of these dual requirements.  

### Research Objectives  
This work aims to bridge these gaps by developing a novel framework for generating tabular data that satisfies *both* differential privacy and fairness constraints while maintaining high utility. The objectives are:  
1. To design a DP mechanism that integrates noise injection into LLM fine-tuning and decoding processes.  
2. To embed fairness constraints (e.g., demographic parity, equalized odds) directly into the LLM’s generation pipeline.  
3. To validate the method’s effectiveness against state-of-the-art baselines (e.g., TableDiffusion [3], DP-2Stage [4]) across multiple datasets and evaluation metrics.  

### Significance  
This research will advance three key aspects of trustworthy ML:  
1. **Privacy Preservation**: Enable DP guarantees for synthetic tabular data, facilitating regulatory compliance (e.g., GDPR, HIPAA).  
2. **Bias Mitigation**: Address representation disparities through explicit fairness constraints in synthetic data curation.  
3. **Scalability**: Leverage pre-trained LLMs to handle complex, high-dimensional tabular data efficiently.  

---  

## 2. Methodology  

### Data Collection and Preprocessing  
We will use publicly available tabular datasets with known privacy/fairness risks, including:  
- **Adult Income** (predicting income ≥$50K, sensitive attribute: gender/race)  
- **German Credit** (credit risk, sensitive attribute: age)  
- **MIMIC-III** (medical data, sensitive attributes: ethnicity, insurance type).  

**Preprocessing Steps**:  
1. **Tokenization**: Convert numerical and categorical features into text sequences using templates (e.g., `"Age: 35, Gender: Female, Income: ≤50K"`).  
2. **Normalization**: Scale numerical features to [0,1] and encode categoricals with label encoding.  
3. **Splitting**: Divide datasets into training (70%), validation (15%), and test (15%) sets.  

### Model Architecture and Training  

#### a. LLM Selection and Fine-Tuning  
We will use **Llama-3-8B** as the base model, selected for its strong generalization and interpretability. Key steps:  

1. **Domain Adaptation**:  
   - Fine-tune Llama-3-8B on a publicly available synthetic medical dataset (MedSynth) to adapt to tabular generation.  

2. **Differential Privacy Integration**:  
   - **DP-SGD Training**: Apply differentially private stochastic gradient descent (DP-SGD) during fine-tuning on the target dataset. Gradients are clipped with bound $C$ and Gaussian noise added:  
     $$
     \nabla_{\theta} \mathcal{L}_{\text{DP}} = \frac{1}{m} \sum_{i=1}^m \text{clip}(\nabla_{\theta} \ell(x_i, \theta), C) + \mathcal{N}(0, \sigma^2 C^2 I),
     $$  
     where $m$ is the batch size, $\ell$ is the loss per sample, $\sigma$ is the noise multiplier, and $C$ ensures bounded sensitivity.  

   - **Privacy Accounting**: Compute the $(\epsilon, \delta)$-DP guarantees using the moments accountant method [1].  

3. **Fairness Constraints**:  
   - **Training Objective**: Introduce a fairness-aware loss function combining task accuracy and fairness penalties:  
     $$
     \mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{fairness}},
     $$  
     where $\mathcal{L}_{\text{task}}$ is cross-entropy for categorical features, and $\mathcal{L}_{\text{fairness}}$ penalizes deviations from target demographic parity:  
     $$
     \mathcal{L}_{\text{fairness}} = \left| \text{Demographic Parity Ratio} - 1 \right|, \quad \text{Demographic Parity Ratio} = \frac{\Pr(\hat{Y}=1 | A=0)}{\Pr(\hat{Y}=1 | A=1)},
     $$  
     with $A$ as a sensitive attribute and $\hat{Y}$ as the synthetic outcome.  

   - **Constrained Decoding**: Use beam search with fairness-aware masking during generation. For example, if a candidate output violates fairness constraints, downweight its probability.  

#### b. Baseline Models  
Compare with:  
1. **DP-TBART** (transformer-based autoregressive model with DP [1]).  
2. **TableDiffusion** (diffusion model with DP [3]).  
3. **DP-2Stage** (two-stage fine-tuning [4]).  

### Experimental Design  

#### Evaluation Metrics  
1. **Utility**:  
   - **Prediction Accuracy**: Train a gradient-boosted tree (XGBoost) on synthetic data and evaluate AUC-ROC on real data.  
   - **Statistical Fidelity**: Compute Wasserstein distance between real and synthetic feature distributions.  

2. **Privacy**: Report $(\epsilon, \delta)$ values for all DP methods.  

3. **Fairness**:  
   - **Disparate Impact (DI)**: $\Pr(\hat{Y}=1 | A=0)/\Pr(\hat{Y}=1 | A=1)$. A value ≥0.8 is considered fair [6].  
   - **Equalized Odds**: $\Pr(\hat{Y}=1 | Y=1, A=0) - \Pr(\hat{Y}=1 | Y=1, A=1)$.  

4. **Data Quality**:  
   - **Mode Coverage**: Ratio of unique feature combinations in synthetic vs. real data.  
   - **Proximity**: Average cosine similarity between synthetic and real instances.  

#### Ablation Studies  
- **Component Analysis**: Test the impact of DP only, fairness only, and combined constraints.  
- **Noise Sensitivity**: Vary $\epsilon$ (from 1 to 10) and measure trade-offs with utility/fairness.  

#### Computational Infrastructure  
Train on NVIDIA A100 GPUs with PyTorch+DeepSpeed. Use the **opacus** library for DP-SGD.  

---  

## 3. Expected Outcomes & Impact  

### Anticipated Results  
1. **Quantifiable Privacy**: Our method will achieve $\epsilon \leq 2$ (strict DP) while maintaining ≥85% of the utility of non-private baselines.  
2. **Improved Fairness**: The Disparate Impact ratio will exceed 0.85 on Adult and German Credit datasets, outperforming DP-TBART (DI ≈0.78) and TableDiffusion (DI ≈0.75).  
3. **High-Fidelity Data**: Wasserstein distances between synthetic and real data will be within 5% of non-private methods.  

### Broader Implications  
1. **Policy and Regulation**: Enable compliant use of synthetic data in regulated sectors by offering formal DP guarantees.  
2. **Fair Algorithm Development**: Provide benchmark datasets that reduce representational harm in high-stakes ML applications.  
3. **Scientific Contribution**: Demonstrate the viability of LLMs for DP/fair tabular synthesis, paving the way for multi-modal (e.g., text + tables) extensions.  

### Limitations and Mitigation  
- **Utility Trade-offs**: Aggressive DP noise may degrade data quality. We will use adaptive clipping and noise optimization [5].  
- **Bias Propagation**: LLMs may inherit biases from pre-training data. Regular audits during fine-tuning will monitor for spurious correlations.  

---  

This work directly addresses the workshop’s goals by advancing generative AI techniques that holistically tackle privacy, fairness, and utility—critical steps toward trustworthy ML deployment. By integrating constraints into LLMs, we aim to set a new standard for synthetic data generation in sensitive domains.