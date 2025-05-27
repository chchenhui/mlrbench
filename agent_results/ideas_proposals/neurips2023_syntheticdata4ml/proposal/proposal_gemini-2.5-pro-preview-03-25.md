## 1. Title: Differentially Private and Fair Tabular Data Synthesis via Constrained Large Language Models

## 2. Introduction

### 2.1 Background
The advancement of machine learning (ML) is intrinsically linked to the availability of large, high-quality datasets. However, obtaining such datasets is often infeasible, particularly in sensitive domains like healthcare, finance, and education. Key challenges include **data scarcity**, where sufficient samples, especially for rare events or underrepresented groups, are difficult or costly to collect; **privacy concerns**, where using real individual-level data poses significant ethical and legal risks, hindering data sharing; and **inherent bias**, where datasets reflect historical or societal biases, leading to ML models that perpetuate or amplify unfairness against specific demographic groups. These challenges critically limit the development and deployment of trustworthy ML systems in high-stakes applications.

Synthetic data generation has emerged as a promising paradigm to mitigate these issues. By creating artificial data that mimics the statistical properties of real data, synthetic data can potentially augment limited datasets, enable data sharing without exposing real individuals' information, and allow for the curation of more balanced and fair datasets. Generative Artificial Intelligence (AI), particularly recent advances in generative models, offers powerful tools for this task.

Among generative models, Large Language Models (LLMs), pre-trained on vast amounts of text data, have shown remarkable capabilities in understanding context, capturing complex dependencies, and generating coherent sequences. Their potential application extends beyond natural language to structured data, including tabular data, which is prevalent in many critical domains. Representing tabular data as sequences allows LLMs to learn the joint distribution of features and generate new, realistic data rows.

However, naively applying standard LLMs for tabular data synthesis presents significant risks. Without explicit safeguards, LLMs can inadvertently memorize and reproduce sensitive information from their training data, leading to privacy breaches. Furthermore, if the original data contains biases (e.g., underrepresentation or skewed associations related to sensitive attributes like race or gender), the generated synthetic data will likely inherit and potentially amplify these biases, undermining fairness goals. Existing research often focuses on either high-fidelity generation, neglecting privacy and fairness, or addresses privacy/fairness primarily in discriminative modeling tasks, leaving a gap in generative settings. While recent works (Castellon et al., 2023; Tran & Xiong, 2024; Truda, 2023; Afonja et al., 2024; Johnson & Lee, 2024; Grey & Yellow, 2024) have started exploring differentially private (DP) or fair synthetic data generation, often using GANs, VAEs, or specialized transformers, the combined challenge of achieving high utility, strong privacy guarantees, *and* demonstrable fairness simultaneously using the power of pre-trained LLMs remains an active and crucial area of research.

### 2.2 Problem Statement
The core problem addressed by this research is the lack of robust methods for generating synthetic tabular data using Large Language Models that simultaneously satisfy three critical requirements: (1) **High Data Utility:** The synthetic data should accurately capture the statistical properties and complex relationships present in the real data to be useful for downstream ML tasks. (2) **Formal Privacy Guarantees:** The generation process must adhere to Differential Privacy (DP), providing mathematical guarantees against the leakage of individual information from the original dataset. (3) **Provable Fairness:** The synthetic data should mitigate biases present in the original data concerning specified sensitive attributes, leading to fairer downstream ML models according to pre-defined fairness metrics.

### 2.3 Research Objectives
This research aims to develop and evaluate a novel framework, "DP-Fair-TabLLM," for generating differentially private and fair synthetic tabular data by fine-tuning pre-trained LLMs with explicit constraints. The specific objectives are:

1.  **Develop a Constrained LLM Fine-tuning Framework:** Design and implement a methodology to fine-tune pre-trained LLMs (e.g., GPT-style, T5) on tabular data, incorporating mechanisms for both differential privacy and fairness.
2.  **Integrate Differential Privacy:** Implement and evaluate state-of-the-art DP mechanisms (e.g., DP-SGD) during the fine-tuning process to provide rigorous $(\epsilon, \delta)$-DP guarantees for the generated data. Analyze the impact of privacy budget $\epsilon$ on data utility and fairness.
3.  **Incorporate Fairness Constraints:** Formulate and integrate fairness constraints directly into the LLM's training objective or decoding strategy. Focus on group fairness metrics such as Demographic Parity (DP) and Equalized Odds (EO) with respect to specified sensitive attributes.
4.  **Optimize the Utility-Privacy-Fairness Trade-off:** Investigate techniques to effectively balance the inherent trade-offs between data utility, privacy loss, and fairness improvements during the generation process.
5.  **Comprehensive Empirical Evaluation:** Rigorously evaluate the proposed DP-Fair-TabLLM framework on diverse benchmark tabular datasets from domains susceptible to privacy and fairness concerns. Compare its performance against state-of-the-art baseline methods across multiple metrics for utility, privacy, and fairness.

### 2.4 Significance
This research holds significant potential for advancing trustworthy machine learning. By enabling the generation of high-quality, private, and fair synthetic tabular data, this work can:

*   **Unlock ML in Sensitive Domains:** Provide organizations in healthcare, finance, etc., with a mechanism to leverage sensitive data for model development and analysis without compromising individual privacy or perpetuating systemic biases.
*   **Promote Data Sharing and Collaboration:** Facilitate the sharing of realistic, privacy-preserving datasets, accelerating research and development in various fields.
*   **Address Data Scarcity and Bias:** Offer a principled way to augment datasets, particularly for underrepresented groups, leading to more robust and equitable ML models.
*   **Advance Generative Modeling Research:** Contribute novel techniques for controlling the output of LLMs beyond simple text generation, specifically incorporating complex constraints like DP and fairness for structured data.
*   **Provide Practical Tools:** Develop open-source implementations and guidelines enabling practitioners to apply the proposed methods in real-world scenarios.

Addressing the limitations highlighted in the workshop description, this research directly tackles the intersection of generative models, privacy, and fairness, aiming to provide a unified solution and contribute to establishing consistent benchmarking practices.

## 3. Methodology

### 3.1 Conceptual Framework
The proposed DP-Fair-TabLLM framework leverages a pre-trained LLM as the backbone for tabular data generation. The core idea is to fine-tune this LLM on a sensitive real tabular dataset using an objective function and training procedure specifically designed to incorporate differential privacy and fairness constraints. The process involves serializing tabular data into a sequence format suitable for LLMs, applying DP mechanisms during fine-tuning, and integrating fairness objectives into the learning process.

### 3.2 Data Representation and Preprocessing
Tabular data, consisting of rows (records) and columns (features), needs to be converted into a sequence format that an LLM can process.
1.  **Serialization:** Each row of the table will be converted into a string. A common format is `Col1Name = Value1 ; Col2Name = Value2 ; ... ; ColNName = ValueN`. Categorical features are used directly. Numerical features may require tokenization strategies, such as discretization/binning (e.g., using quantiles) or representing digits as individual tokens. The choice of serialization and numerical handling strategy will be explored as part of the research.
2.  **Special Tokens:** Special tokens like `<bos>` (beginning of sequence), `<eos>` (end of sequence), and `<pad>` (padding) will be used as standard in LLM training.
3.  **Data Splitting:** The real dataset $D_{real}$ will be used solely for fine-tuning the generator. Evaluation will primarily use the generated synthetic data $D_{synth}$ and hold-out test sets from $D_{real}$ for downstream task evaluation (TSTR paradigm).

### 3.3 LLM Architecture
We will primarily leverage existing powerful pre-trained autoregressive LLMs, such as variants of GPT (Generative Pre-trained Transformer) or T5 (Text-to-Text Transfer Transformer), known for their strong generative capabilities. The choice will depend on preliminary experiments assessing suitability for structured data generation and compatibility with DP fine-tuning libraries. The model takes a sequence prefix (potentially just `<bos>`) and autoregressively generates the token sequence representing a tabular row.

### 3.4 Differential Privacy Integration
We will ensure the synthetic data generation process satisfies $(\epsilon, \delta)$-Differential Privacy. The primary mechanism will be Differentially Private Stochastic Gradient Descent (DP-SGD) applied during the fine-tuning phase.
1.  **DP-SGD:** During each training step, DP-SGD involves:
    *   **Gradient Clipping:** Computing per-sample gradients and clipping their L2 norm to a predefined threshold $C$. This bounds the sensitivity of each sample's contribution.
    *   **Noise Addition:** Adding Gaussian noise scaled by the sensitivity $C$ and the privacy budget $(\epsilon, \delta)$ to the aggregated clipped gradients before updating the model parameters. The noise variance is typically $\sigma^2 \propto C^2 / \epsilon$.
    $$ \tilde{g}_t = \frac{1}{B} \sum_{i=1}^{B} \text{clip}(\nabla_{\theta} L(x_i; \theta_t), C) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) $$
    where $B$ is the batch size, $L$ is the loss function, $x_i$ is the $i$-th sample, $\theta_t$ are model parameters at step $t$, and $\text{clip}(\cdot, C)$ clips the L2 norm.
2.  **Privacy Accounting:** The cumulative privacy loss $(\epsilon, \delta)$ over the entire fine-tuning process will be tracked using established techniques like Rényi Differential Privacy (RDP) accountants (e.g., `tensorflow_privacy`, `opacus`), which provide tighter bounds compared to traditional moments accountants, especially for low values of $\epsilon$. Different values of $\epsilon$ (e.g., 1, 5, 10) will be explored to study the privacy-utility trade-off.

### 3.5 Fairness Integration
Fairness constraints will be incorporated to mitigate bias related to pre-defined sensitive attributes $A$ (e.g., race, gender) with respect to a target variable $Y$ (if applicable for fairness definition) or the generation process itself. We will focus on group fairness notions:
1.  **Demographic Parity (DP):** Aims for the predicted outcome (or generated feature distribution) to be independent of the sensitive attribute. For a classifier $h$ trained on synthetic data, DP requires $P(h(X)=1 | A=a) \approx P(h(X)=1 | A=b)$ for different sensitive groups $a, b$.
2.  **Equalized Odds (EO):** Aims for the prediction rates (True Positive Rate, False Positive Rate) to be equal across groups. Requires $P(h(X)=1 | A=a, Y=y) \approx P(h(X)=1 | A=b, Y=y)$ for $y \in \{0, 1\}$.

**Integration Strategies:**
*   **Fairness Regularization:** Add a fairness-promoting regularization term $L_{fair}$ to the main LLM loss function $L_{LM}$ (e.g., cross-entropy for next token prediction).
    $$ L_{total} = L_{LM} + \lambda_{fair} L_{fair} $$
    $L_{fair}$ could be designed to penalize fairness violations measured on generated mini-batches during training. For instance, it could measure the statistical distance (e.g., KL divergence, Wasserstein distance) between the distributions of generated features conditional on different sensitive attribute values, or approximate DP/EO violations using differentiable proxies. The hyperparameter $\lambda_{fair}$ controls the strength of the fairness constraint.
*   **Constrained Decoding:** Modify the LLM's decoding process (e.g., beam search) at generation time to explicitly promote fairness criteria. This might involve re-ranking generated sequences based on their contribution to overall dataset fairness or guiding the generation towards fairer distributions. This decouples fairness enforcement from the DP fine-tuning, potentially offering more flexibility but possibly lower utility. We will primarily focus on integrating fairness into the training objective via $L_{fair}$.

### 3.6 Combined Training and Generation
The LLM will be fine-tuned using DP-SGD on the serialized real data $D_{real}$ using the combined loss $L_{total}$. The fine-tuning process carefully balances learning the data distribution ($L_{LM}$), ensuring privacy (via DP-SGD noise and clipping), and promoting fairness ($L_{fair}$). Once trained, the DP-Fair-TabLLM model can generate arbitrarily many synthetic tabular rows by sampling from its learned distribution, starting from the `<bos>` token and autoregressively predicting tokens until `<eos>` is generated. These generated sequences are then de-serialized back into tabular format to form $D_{synth}$.

### 3.7 Experimental Design

1.  **Datasets:** We will use publicly available benchmark datasets commonly used in fairness and privacy research, containing sensitive attributes. Examples include:
    *   **Adult Income:** Predict income >\$50K based on demographic features (sensitive attributes: race, sex).
    *   **COMPAS:** Predict recidivism based on criminal history and demographics (sensitive attribute: race).
    *   **Bank Marketing:** Predict term deposit subscription (sensitive attribute: age group).
    *   (Potentially) A healthcare-related dataset where privacy and fairness are paramount, subject to ethical approvals and data use agreements (e.g., MIMIC-III demo subset focusing on specific prediction tasks with demographic attributes).
    Sensitive attributes and target variables for fairness evaluation will be clearly defined for each dataset.

2.  **Baselines:** The proposed DP-Fair-TabLLM will be compared against several baselines:
    *   **REAL:** Models trained directly on the real data (upper bound for utility, non-private, potentially unfair).
    *   **LLM-Base:** Fine-tuned LLM without DP or fairness constraints.
    *   **LLM-DP:** Fine-tuned LLM with only DP-SGD (similar to Tran & Xiong, 2024; Afonja et al., 2024, but using our specific setup).
    *   **LLM-Fair:** Fine-tuned LLM with only fairness constraints (non-private).
    *   **Existing DP Synthesizers:** DP-TBART (Castellon et al., 2023), TableDiffusion (Truda, 2023), CTGAN+DP (applying DP-SGD to GANs).
    *   **Existing Fair Synthesizers:** Methods designed only for fairness (if applicable, less common than DP or DP+Fair).
    *   **Existing DP+Fair Synthesizers:** GAN/VAE/Transformer-based methods incorporating both (e.g., Johnson & Lee, 2024; White & Brown, 2023; Green & Black, 2024; Grey & Yellow, 2024 - implemented based on paper descriptions if code unavailable).

3.  **Evaluation Metrics:** We will evaluate the generated synthetic data $D_{synth}$ based on three pillars:

    *   **Data Utility:**
        *   **Statistical Similarity:** Compare marginal distributions (e.g., using Jensen-Shannon Divergence - JSD, or Wasserstein distance) and feature correlations (e.g., difference in pairwise correlation matrices) between $D_{real}$ and $D_{synth}$. Propensity Mean Squared Error (pMSE) can also assess distributional similarity.
        *   **Machine Learning Efficacy (TSTR - Train-Synthetic-Test-Real):** Train standard ML models (e.g., Logistic Regression, Random Forest, MLP) on $D_{synth}$ and evaluate their performance (e.g., Accuracy, F1-score, AUC) on a held-out test set from $D_{real}$. Compare results with models trained on $D_{real}$.

    *   **Privacy:**
        *   **Formal Guarantee:** Report the achieved $(\epsilon, \delta)$-DP guarantee calculated via the RDP accountant based on the DP-SGD parameters (noise multiplier $\sigma$, clipping norm $C$, number of steps, sampling rate).
        *   **Empirical Privacy (Optional):** Potentially run Membership Inference Attacks (MIAs) as a secondary, empirical measure of privacy leakage, acknowledging their limitations and dependence on attacker assumptions.

    *   **Fairness:**
        *   **Downstream Model Fairness:** Train ML models on $D_{synth}$ and evaluate their fairness on the real test set using standard group fairness metrics:
            *   **Demographic Parity Difference (DPD):** $|P(\hat{Y}=1|A=a) - P(\hat{Y}=1|A=b)|$
            *   **Equalized Odds Difference (EOD):** Maximum of the absolute difference in True Positive Rates (TPR) and False Positive Rates (FPR) across groups: $\max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)$
            *   **Equal Opportunity Difference (EoppD):** $|TPR_a - TPR_b|$
        *   **Intrinsic Data Fairness:** Analyze fairness properties directly within $D_{synth}$, such as representation parity or statistical associations between sensitive attributes and other features, compared to $D_{real}$.

4.  **Ablation Studies:** Perform ablation studies to understand the contribution of different components: varying the privacy budget $\epsilon$, the fairness constraint weight $\lambda_{fair}$, the choice of LLM architecture, and the data serialization method. Analyze the trade-offs explicitly (e.g., plot Utility vs. $\epsilon$, Fairness vs. $\epsilon$, Utility vs. Fairness at fixed $\epsilon$).

## 4. Expected Outcomes & Impact

### 4.1 Expected Outcomes
We anticipate the following outcomes from this research:

1.  **A Novel DP-Fair-TabLLM Framework:** A fully developed and implemented framework for generating synthetic tabular data using LLMs that satisfies differential privacy and promotes group fairness.
2.  **High-Quality Synthetic Data:** Demonstration that the proposed framework can generate synthetic tabular data exhibiting high utility (comparable ML efficacy to non-private baselines, high statistical similarity), strong privacy guarantees (formal $(\epsilon, \delta)$-DP), and improved fairness (reduced DPD/EOD in downstream tasks compared to models trained on real or naively generated data).
3.  **Understanding of Trade-offs:** Quantitative insights into the complex interplay and trade-offs between data utility, privacy budget $\epsilon$, and fairness levels achievable with the proposed method. This will include Pareto frontiers illustrating achievable combinations.
4.  **Comparative Analysis:** Rigorous benchmarking results comparing DP-Fair-TabLLM against state-of-the-art methods, clearly highlighting its advantages and limitations across different datasets and evaluation metrics.
5.  **Open-Source Implementation:** A publicly available code repository implementing the DP-Fair-TabLLM framework and evaluation scripts to facilitate reproducibility and adoption by the research community and practitioners.
6.  **Peer-Reviewed Publications:** Dissemination of findings through high-impact publications in leading ML conferences (e.g., NeurIPS, ICML, ICLR) and journals.

### 4.2 Impact
The successful completion of this research is expected to have a significant impact:

*   **Enhancing Trustworthy AI:** By providing a practical method for generating private and fair synthetic data, this work will contribute directly to the development of more trustworthy and responsible AI systems, particularly crucial for high-stakes applications.
*   **Enabling Data-Driven Innovation:** Lowering the barriers related to privacy and fairness will enable wider use of sensitive data for research, development, and innovation in fields like healthcare (e.g., analyzing patient data for treatment efficacy without privacy risks), finance (e.g., building fair credit scoring models), and social science research.
*   **Improving Fairness and Equity:** The ability to generate fair synthetic data can be used to audit and mitigate bias in existing datasets and models, potentially leading to more equitable outcomes from deployed ML systems. It provides a tool to proactively shape data towards desired fairness properties.
*   **Advancing Synthetic Data Generation:** This research pushes the boundaries of generative modeling by showing how large pre-trained models can be adapted and constrained for complex, multi-objective generation tasks beyond natural language, specifically addressing the unique challenges of tabular data.
*   **Informing Policy and Practice:** The findings and tools developed can inform organizational policies regarding data sharing and use, and provide practitioners with concrete methods to operationalize privacy and fairness principles in their ML workflows.

By addressing the critical need for synthetic data that is simultaneously useful, private, and fair, this research aligns perfectly with the goals of the Workshop on Synthetic Data Generation, aiming to bridge existing gaps and empower the development of reliable and ethical ML applications.