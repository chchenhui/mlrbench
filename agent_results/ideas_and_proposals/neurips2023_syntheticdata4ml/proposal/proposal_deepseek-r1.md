**Research Proposal**

**1. Title**  
"Differentially Private and Fair Synthetic Tabular Data Generation Using Constrained Large Language Models"

---

**2. Introduction**  
**Background**  
The development of trustworthy machine learning (ML) systems in high-stakes domains like healthcare, finance, and education is hindered by three interrelated challenges: data scarcity, privacy concerns, and inherent biases in training datasets. Synthetic data generation offers a promising solution by enabling the creation of realistic, privacy-preserving, and bias-corrected datasets. While generative models—particularly Large Language Models (LLMs)—have demonstrated exceptional capabilities in text and image generation, their application to structured tabular data remains underexplored, especially in scenarios requiring formal privacy guarantees (e.g., differential privacy) and fairness constraints. Existing approaches either prioritize fidelity at the expense of privacy and fairness or focus narrowly on one constraint without harmonizing multi-objective optimization. This gap limits their utility in sensitive real-world applications.

**Research Objectives**  
1. Develop a framework for training LLMs to generate synthetic tabular data that satisfies both differential privacy (DP) and group fairness constraints.  
2. Validate the utility, privacy, and fairness of the synthetic data through rigorous empirical evaluation.  
3. Characterize the trade-offs between these objectives and propose methods to balance them.  

**Significance**  
This work addresses critical challenges in synthetic data generation by unifying privacy and fairness constraints into a single LLM-based pipeline. By enabling the generation of datasets that are (1) DP-protected against membership inference attacks and (2) bias-mitigated for specified sensitive attributes (e.g., race, gender), the proposed method will support the development of trustworthy ML models in domains where ethical and legal constraints are paramount. The integration of LLMs further leverages their ability to model complex tabular data distributions while retaining scalability across dataset sizes.

---

**3. Methodology**  
**3.1. Data Collection and Preprocessing**  
- **Datasets**: Focus on benchmarking with healthcare (MIMIC-IV), finance (Adult Income Census), and education (Student Performance) datasets containing sensitive attributes.  
- **Preprocessing**:  
  - Perform feature engineering to convert categorical variables into LLM-compatible token sequences.  
  - Partition datasets into training (for synthetic data generation) and validation (for downstream ML evaluation).  
  - Identify sensitive attributes (e.g., gender, race) for fairness constraints.  

**3.2. Model Architecture and Training**  
Utilize a pre-trained LLM (e.g., GPT-2 or TabTransformer) adapted for tabular data generation. The model is fine-tuned with two critical modifications:  

**A. Differential Privacy Mechanism**  
Apply **DP-SGD (Differentially Private Stochastic Gradient Descent)** during fine-tuning to ensure $(\epsilon, \delta)$-DP guarantees:  
1. Compute per-example gradients $\nabla_\theta \mathcal{L}(x_i, y_i)$ for the loss $\mathcal{L}$.  
2. Clip gradients to a maximum $L_2$-norm $C$:  
   $$\tilde{g}_i = \frac{\nabla_\theta \mathcal{L}(x_i, y_i)}{\max\left(1, \frac{\|\nabla_\theta \mathcal{L}(x_i, y_i)\|_2}{C}\right)}.$$  
3. Add Gaussian noise $\mathcal{N}(0, C^2 \sigma^2 I)$ to the aggregated batch gradient:  
   $$g_{\text{DP}} = \frac{1}{B} \left(\sum_{i=1}^B \tilde{g}_i + \mathcal{N}(0, C^2 \sigma^2 I)\right).$$  

**B. Fairness-Aware Training**  
Incorporate a fairness regularizer into the loss function to minimize disparities across sensitive groups. For a binary sensitive attribute $S \in \{0, 1\}$ and generated feature distribution $P_\theta$, enforce demographic parity divergence:  
$$\mathcal{L}_{\text{fair}} = \lambda \cdot \text{KL}\left(P_\theta(X | S=0) \| P_\theta(X | S=1)\right),$$  
where $\lambda$ controls the fairness-utility trade-off. The total loss becomes:  
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLE}} + \mathcal{L}_{\text{fair}}.$$  

**3.3. Decoding with Constraints**  
During inference, use constrained decoding to ensure synthetically generated rows adhere to fairness and privacy:  
1. **Privacy**: Inject calibrated noise into the LLM’s logits before sampling.  
2. **Fairness**: Apply a fairness-aware masking strategy that suppresses biased token sequences.  

**3.4. Experimental Design**  
- **Baselines**: Compare against DP-TBART [1], DP-LLMTGen [2], TableDiffusion [3], and non-private GAN/VAE frameworks.  
- **Evaluation Metrics**:  
  - **Utility**:  
    - *Fréchet Distance (FD)*: Measure distributional similarity between real and synthetic data.  
    - *Downstream ML Performance*: Train classifiers (e.g., XGBoost) on synthetic data and test on real validation data.  
  - **Privacy**:  
    - $\epsilon$-*Privacy Budget*: Quantify formal DP guarantees.  
    - *Membership Inference Attack Success Rate*: Use ML models to predict if a sample was in the training set.  
  - **Fairness**:  
    - *Demographic Parity Difference*: $\left|P(\hat{Y}=1 | S=0) - P(\hat{Y}=1 | S=1)\right|$.  
    - *Equalized Odds Difference*: $\left|P(\hat{Y}=1 | S=0, Y=y) - P(\hat{Y}=1 | S=1, Y=y)\right|$ for $y \in \{0, 1\}$.  

- **Ablation Studies**: Isolate the effects of DP noise, fairness constraints, and LLM architecture choices.  

---

**4. Expected Outcomes & Impact**  
**Expected Outcomes**  
1. A novel framework for generating tabular data that achieves:  
   - $\epsilon \leq 2$ with $\delta = 10^{-5}$ (strong DP guarantees).  
   - Demographic parity difference reduced by ≥40% compared to DP-LLMTGen [2].  
   - Comparable or superior downstream ML performance (e.g., AUC-ROC within 5% of non-private baselines).  
2. Insights into the interaction between DP noise injection and fairness constraints, including empirical trade-off curves.  
3. Open-source implementation and benchmark datasets for reproducibility.  

**Impact**  
The proposed method will enable organizations in healthcare, finance, and education to:  
1. Share synthetic datasets without compromising individual privacy.  
2. Mitigate biases in ML systems trained on synthetic data.  
3. Accelerate research in domains hampered by data scarcity.  
The integration of LLMs into tabular data synthesis will also advance generative modeling research, providing a pathway to unify privacy, fairness, and fidelity in multi-modal settings.  

---

**5. Conclusion**  
This proposal addresses a critical gap in synthetic data generation by developing a constrained LLM framework that harmonizes differential privacy and fairness. By advancing both algorithmic innovation and empirical benchmarking, the work aims to establish a new standard for trustworthy synthetic data generation, fostering ethical ML deployment in high-stakes domains. Future work will extend the framework to time-series and multimodal data, further bridging the gap between generative AI and real-world societal needs.