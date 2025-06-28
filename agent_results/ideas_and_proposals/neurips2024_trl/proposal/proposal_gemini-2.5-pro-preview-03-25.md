Okay, here is a detailed research proposal for "SynthTab – LLM-Driven Synthetic Tabular Data with Constraint-Aware Generation", structured according to your requirements and incorporating the provided information.

---

# **Research Proposal**

## **1. Title:**

**SynthTab: A Constraint-Aware Multi-Agent Framework using Large Language Models for High-Fidelity Synthetic Tabular Data Generation**

## **2. Introduction**

### **Background**

Tabular data remains the backbone of data management and analysis across countless domains, underpinning critical operations in finance, healthcare, enterprise resource planning, scientific research, and more. Despite its ubiquity, the effective application of machine learning (ML) on tabular data often faces significant hurdles, primarily related to data accessibility. Data scarcity is common in niche domains or for specific rare events, hindering the training of robust ML models. Furthermore, privacy regulations (like GDPR, HIPAA) and concerns severely restrict the sharing and usage of sensitive tabular datasets, particularly in fields like medicine and finance.

Synthetic data generation (SDG) has emerged as a promising solution to mitigate these challenges. By creating artificial data that mimics the statistical properties and structure of real data, SDG can facilitate data augmentation, enable privacy-preserving data sharing, and serve as a basis for software testing and system benchmarking. Traditional statistical methods (e.g., sampling from learned distributions) and early deep learning approaches like Generative Adversarial Networks (GANs) (e.g., CTGAN, TVAE) and Variational Autoencoders (VAEs) have shown success but often struggle with capturing complex column dependencies, maintaining semantic consistency, and strictly adhering to schema constraints (data types, uniqueness, referential integrity, business rules). Violating these constraints renders the synthetic data unrealistic and potentially unusable for downstream tasks or database loading.

Recently, Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human language, and their potential for structured data is increasingly being explored (Nguyen et al., 2024; Zhao et al., 2023). LLMs inherently model sequential dependencies and can capture complex patterns, making them attractive candidates for generating coherent rows in a table. However, applying LLMs directly to tabular data generation presents unique challenges. Standard autoregressive generation may fail to capture non-sequential dependencies typical in tables and often overlooks explicit schema constraints (Xu et al., 2024). Ensuring the generated data is not only statistically similar but also *valid* according to domain-specific rules and privacy-compliant requires specialized techniques. Recent works like HARMONIC (Wang et al., 2024) and TabuLa (Zhao et al., 2023) have explored LLM fine-tuning and architectural modifications for tabular synthesis, while others have focused specifically on schema constraints (Johnson & Williams, 2023) or privacy (Doe & Smith, 2023; Brown & Green, 2024). However, integrating these aspects—high-fidelity generation, strict constraint adherence, privacy preservation, and domain semantics—within a single, robust framework remains an open research problem.

### **Research Objectives**

This research proposes **SynthTab**, a novel multi-agent framework designed to generate high-fidelity, constraint-aware, and privacy-preserving synthetic tabular data. SynthTab leverages the generative power of LLMs while incorporating explicit mechanisms for schema validation, quality assessment, and privacy protection through coordinated agent interactions.

The primary objectives of this research are:

1.  **Design and Implement the SynthTab Framework:** Develop a multi-agent system comprising:
    *   An LLM-based **Generator Agent** fine-tuned for tabular data structures and enhanced with Retrieval-Augmented Generation (RAG) to incorporate domain knowledge and real data patterns.
    *   A **Schema Validator Agent** employing rule-based checks and potentially chain-of-thought reasoning to enforce data types, uniqueness, referential integrity, and custom business rules.
    *   A **Quality Assessor Agent** evaluating the statistical fidelity and downstream utility of generated data, providing feedback for iterative refinement.
    *   An integrated **Privacy Module** applying differential privacy techniques to bound information leakage.
2.  **Enforce Strict Schema Compliance:** Develop and evaluate mechanisms within the Schema Validator agent to ensure generated data strictly adheres to predefined database schemas and business logic, a critical limitation in many existing SDG methods.
3.  **Enhance Generation Fidelity via RAG and Iterative Refinement:** Investigate the effectiveness of RAG in improving the semantic realism of generated data by grounding the LLM in relevant examples. Implement an iterative feedback loop where quality assessment guides subsequent generation steps.
4.  **Incorporate Tunable Privacy Guarantees:** Integrate differential privacy mechanisms (e.g., DP-SGD during fine-tuning, output perturbation) into the framework, allowing users to balance data utility and privacy levels (controlled by $\epsilon, \delta$).
5.  **Comprehensive Evaluation:** Rigorously evaluate SynthTab against state-of-the-art tabular SDG baselines across diverse datasets. Assessment criteria will include statistical fidelity, constraint violation rates, downstream ML task performance (utility), and privacy leakage (e.g., via Membership Inference Attacks).

### **Significance**

This research addresses critical needs in the field of tabular data management and machine learning. By developing SynthTab, we aim to provide a practical and effective solution for generating high-quality synthetic data that is both realistic and trustworthy.

*   **Enabling ML in Data-Scarce/Private Settings:** SynthTab can unlock the potential of ML in domains where real data is limited or cannot be easily shared, such as healthcare, finance, and emerging scientific areas.
*   **Improving Data Augmentation:** High-fidelity, constraint-aware synthetic data can serve as a powerful tool for augmenting training datasets, potentially improving the robustness and generalization of ML models.
*   **Facilitating Safer Data Sharing:** By incorporating differential privacy, SynthTab offers a mechanism for sharing insights from sensitive datasets without exposing individual records, fostering collaboration and research reproducibility.
*   **Advancing Generative Models for Structured Data:** This work contributes to the growing research area of applying LLMs and multi-agent systems to structured data challenges, pushing the boundaries beyond text generation. It specifically addresses the weaknesses of naive LLM application by adding explicit validation and quality control loops.
*   **Providing a Practical Tool:** If successful, SynthTab can be developed into a reusable tool or library for researchers and practitioners needing high-quality synthetic tabular data tailored to specific schemas and requirements.

This research directly aligns with the workshop themes of "Representation Learning for (semi-)Structured Data", "Generative Models and LLMs for Structured Data", "Applications of TRL models" (specifically tabular data generation, data augmentation), and "Challenges of TRL models" (privacy, data quality).

## **3. Methodology**

### **Overall Framework**

SynthTab operates as a multi-agent system where specialized agents collaborate to generate synthetic tabular data iteratively. The core components are:

1.  **LLM Generator Agent:** Responsible for proposing candidate data rows.
2.  **Schema Validator Agent:** Responsible for verifying schema compliance of proposed rows.
3.  **Quality Assessor Agent:** Responsible for evaluating the statistical quality and utility of generated data batches and providing feedback.
4.  **Privacy Module:** Integrated to enforce differential privacy guarantees.

The overall workflow is envisioned as follows (Figure 1 - Conceptual):

*   **Input:** Target table schema (column names, data types, constraints like UNIQUE, NOT NULL, FOREIGN KEY, CHECK constraints/business rules), optionally column statistics (marginals, correlations) derived from a real dataset, and potentially a small seed dataset for RAG/fine-tuning. Privacy parameters ($\epsilon, \delta$) are also provided.
*   **Initialization:** The LLM Generator is initialized (potentially fine-tuned). The RAG knowledge base is built (if seed data is provided).
*   **Iterative Generation Loop:**
    a.  **Generation:** The LLM Generator, guided by the schema, statistics, and potentially retrieved examples (via RAG), generates a batch of candidate rows.
    b.  **Validation:** The Schema Validator checks each candidate row against all specified constraints. Invalid rows are discarded or flagged for correction.
    c.  **Accumulation:** Valid rows are added to the pool of synthetic data.
    d.  **Assessment & Feedback (Periodic):** The Quality Assessor periodically evaluates the accumulated synthetic data pool against target statistics and potentially downstream task performance. Feedback (e.g., areas of poor statistical match, common validation errors) is relayed to the LLM Generator.
    e.  **Refinement:** The LLM Generator adjusts its strategy based on the feedback (e.g., modifying prompts, generation temperature, or re-ranking retrieved examples).
*   **Termination:** The loop continues until a desired number of valid rows are generated or quality metrics plateau/converge.
*   **Output:** A synthetic tabular dataset adhering to the schema and privacy constraints.

```mermaid
graph LR
    A[Input: Schema, Stats, Seed Data?, Privacy Params] --> B(Initialize LLM Generator + RAG KB);
    B --> C{Iterative Loop};
    C --> D[LLM Generator: Propose Candidate Rows (using RAG)];
    D --> E[Schema Validator: Check Constraints];
    E -- Valid Rows --> F[Accumulate Synthetic Data];
    E -- Invalid Rows --> G(Discard/Flag);
    F --> H{Assess Quality?};
    H -- Yes --> I[Quality Assessor: Evaluate Stats & Utility];
    I --> J[Provide Feedback to LLM Generator];
    J --> C;
    H -- No --> C;
    C -- Termination Condition Met --> K[Output: Final Synthetic Dataset];

    style G fill:#f9f,stroke:#333,stroke-width:2px
```
*Figure 1: Conceptual Workflow of the SynthTab Multi-Agent System*

### **Data Collection / Input Specification**

*   **Schema Definition:** The framework requires a precise definition of the target table schema, including:
    *   Column names and data types (e.g., INTEGER, VARCHAR, FLOAT, DATE, BOOLEAN).
    *   Constraints: PRIMARY KEY, UNIQUE constraints, FOREIGN KEY references (if generating related tables), NOT NULL constraints, CHECK constraints (representing simple business rules, e.g., `age > 0`, `status IN ('active', 'inactive')`).
*   **Statistical Information (Optional but Recommended):** To guide generation towards realistic distributions, summary statistics from the original data (if available and permissible) will be used. This includes:
    *   Marginal distributions for each column (histograms for categorical/discrete, density estimates or key percentiles for continuous).
    *   Pairwise correlation matrix or mutual information scores between columns.
    *   These statistics may need to be computed under differential privacy if derived from sensitive data (see Privacy Module).
*   **Seed Data (Optional):** A small, potentially anonymized or publicly available sample of real data can be used to:
    *   Fine-tune the LLM Generator.
    *   Populate the knowledge base for the RAG component.

### **Agent 1: LLM Generator**

*   **Model Choice:** We plan to experiment with pre-trained LLMs known for strong reasoning and instruction-following capabilities (e.g., Llama-3, Mistral, potentially smaller fine-tuned models for efficiency). The choice will depend on performance and computational feasibility.
*   **Input Representation:** Tabular rows will be serialized into a text format suitable for the LLM, potentially as comma-separated values (CSV-like strings), JSON objects, or descriptive key-value pairs (e.g., "Column `Name`: `Value`, Column `Age`: `Value`..."). We will investigate different serialization formats for effectiveness. Permutation strategies explored by Nguyen et al. (2024) and Xu et al. (2024) during fine-tuning might be adapted to help the LLM capture column dependencies regardless of order.
*   **Fine-tuning:** If seed data is available, the base LLM will be fine-tuned using instruction-based tuning. Instructions will prompt the model to generate rows compliant with the given schema and reflecting provided statistical hints. Techniques like those in HARMONIC (Wang et al., 2024) using k-NN inspired instructions or TabuLa's (Zhao et al., 2023) specialized tokenization/padding might be explored if vanilla fine-tuning proves insufficient. If using DP-SGD, privacy costs will be tracked.
*   **Retrieval-Augmented Generation (RAG):**
    *   **Knowledge Base (KB):** Constructed from the (potentially privacy-preserving) seed data or an external corpus of domain-relevant tables/text. Rows/patterns are embedded using a suitable sentence transformer or specialized tabular embedding model.
    *   **Retrieval:** Given generation context (schema, target statistics, partially generated row), the system retrieves the top-k most similar entries from the KB using vector similarity search (e.g., cosine similarity on embeddings).
    *   **Prompting:** The retrieved examples are incorporated into the LLM prompt, guiding it to generate data consistent with observed patterns and domain vocabulary (similar to Adams & Brown, 2024, but adapted for structured data generation). Example Prompt Fragment:
        ```
        Schema: User(ID INT PK, Name TEXT, Age INT CHECK(Age>18), City TEXT)
        Stats Hint: Age distribution peaks around 35. Common cities: ['New York', 'London', 'Tokyo'].
        Retrieved Similar Examples: 
        - ID: 101, Name: Alice, Age: 32, City: New York
        - ID: 205, Name: Bob, Age: 45, City: London
        Generate the next row, ensuring ID is unique and Age > 18:
        ID: [Unique ID], Name: [Name], Age: [Age > 18 reflecting stats], City: [City reflecting stats/examples] 
        ```
*   **Generation Strategy:** The LLM will generate candidate rows autoregressively based on the augmented prompt. Sampling parameters (temperature, top-k, top-p) will be tuned. For generating multiple rows, batch generation or sequential generation (conditioning on previously generated valid rows) will be explored.

### **Agent 2: Schema Validator**

*   **Function:** To rigorously enforce all specified schema constraints on candidate rows generated by the LLM.
*   **Mechanism:** Primarily rule-based, but potentially augmented with simple LLM reasoning for complex/natural language business rules.
    *   **Data Type Validation:** Checks if values match column types (e.g., `isdigit()` for INT, date parsing for DATE).
    *   **Uniqueness Validation:** Checks PRIMARY KEY and UNIQUE constraints against previously generated *valid* rows within the current batch and potentially across the entire synthetic dataset being built. Efficient lookups (e.g., hash sets) will be used.
    *   **Referential Integrity (Foreign Keys):** If generating multiple related tables, ensures foreign key values exist in the referenced primary key column of the parent table (requires coordinated generation or post-processing).
    *   **NOT NULL Validation:** Checks for missing values in required columns.
    *   **CHECK Constraints & Business Rules:** Executes simple SQL-like CHECK constraints (e.g., `Age > 18`, `Salary BETWEEN 30000 AND 100000`). Custom rules expressed programmatically or potentially via natural language (parsed by a dedicated LLM call or rule engine) can be integrated.
*   **Interaction:** Receives a candidate row (or batch). Returns a boolean validation status per row. For invalid rows, it should ideally provide specific error messages (e.g., "Violates UNIQUE constraint on column 'ID'", "Value 'XYZ' invalid for INT column 'Age'") which can be used as feedback to the Generator Agent (e.g., added to the negative constraints in the next prompt).

### **Agent 3: Quality Assessor**

*   **Function:** Evaluate the quality of the generated synthetic data pool and provide corrective feedback. Assessment occurs periodically (e.g., every N valid rows generated).
*   **Metrics:**
    *   **Statistical Fidelity:**
        *   *Marginal Distributions:* Compares the distribution of values in each synthetic column to the target statistics (or empirical distributions from real data, if available) using metrics like Jensen-Shannon Divergence (JSD) for categorical data or Wasserstein distance / Kolmogorov-Smirnov test for continuous data.
        *   *Pairwise Correlations:* Compares the correlation matrix (e.g., Pearson for continuous, Cramer's V for categorical) of the synthetic data to the target correlations. Differences measured using Frobenius norm or element-wise differences.
    *   **Downstream Utility (Train-Synthetic-Test-Real - TSTR):** Train standard ML models (e.g., Logistic Regression, Gradient Boosting, MLP) on the generated synthetic data and evaluate their performance on a held-out set of *real* data. Compare metrics like Accuracy, F1-score, AUC (for classification) or RMSE, MAE (for regression) against models trained on real data (if available) and data generated by baselines.
    *   **Constraint Adherence Rate (Sanity Check):** Although the Validator aims for 100% adherence post-filtering, monitoring the *pre-filtering* acceptance rate from the LLM Generator provides insight into its ability to learn constraints.
*   **Feedback Loop:** The assessment results are summarized and fed back to the LLM Generator. Example feedback: "The distribution for column 'Age' is skewed too young compared to target stats.", "Correlation between 'Income' and 'EducationLevel' is weaker than expected.", "High rejection rate due to UNIQUE constraint violation on 'Email'." This feedback informs prompt refinement or adjustments to the generation strategy in the next iteration.

### **Privacy Module**

*   **Goal:** Provide formal privacy guarantees, primarily through Differential Privacy (DP).
*   **Mechanisms (to be explored and selected based on effectiveness and applicability):**
    1.  **DP Statistics:** If input statistics are derived from sensitive data, compute them using DP mechanisms (e.g., Laplace or Gaussian mechanism for counts/histograms, DP covariance matrix estimation). The $\epsilon_{stats}$ budget must be accounted for.
    2.  **DP Fine-Tuning (DP-SGD):** If fine-tuning the LLM Generator on sensitive seed data, use DP-Stochastic Gradient Descent. This adds noise during training to provide DP for the model parameters, which translates to privacy for the training data. The $\epsilon_{ft}$ budget must be tracked.
    3.  **DP Output Perturbation:** Add calibrated noise (e.g., Laplace or Gaussian) directly to the generated *valid* rows. This is simpler but may degrade utility significantly, especially for discrete data or data requiring high precision. Requires careful parameter tuning.
    4.  **Privacy Filtering in RAG:** Ensure the RAG retrieval mechanism doesn't leak sensitive information from the knowledge base, potentially by using DP embeddings or limiting retrieved content detail.
*   **Privacy Budget:** The total privacy loss ($\epsilon_{total}$) will be composed of the budgets spent in different stages (e.g., $\epsilon_{total} = \epsilon_{stats} + \epsilon_{ft}$ using privacy composition theorems). Users can specify a target $\epsilon$ and $\delta$, and the framework will allocate/adjust mechanisms to stay within budget.

### **Experimental Design**

*   **Datasets:** We will use a combination of publicly available benchmark datasets with varying characteristics (size, number of columns, data types, constraint complexity):
    *   *General Benchmarks:* UCI Adult Census Income, Covertype, Credit Card Fraud dataset.
    *   *Domain-Specific (Simulated/Public):* Potentially simulate a simple relational schema (e.g., `Orders` and `Customers` with foreign keys) or use public datasets from finance (e.g., LendingClub - requires careful handling of sensitive fields) or a simplified healthcare context (if suitable public data exists).
*   **Baselines:** We will compare SynthTab against a comprehensive set of baselines:
    *   *Statistical Methods:* Independent sampling based on marginals.
    *   *GAN-based:* CTGAN, TVAE.
    *   *LLM-based (No Constraints):* Fine-tuned LLM without explicit validation/feedback loop.
    *   *LLM-based (State-of-the-Art):* HARMONIC (Wang et al., 2024), TabuLa (Zhao et al., 2023), potentially the method from Nguyen et al. (2024) if code is available.
    *   *Constraint-Aware (Non-LLM):* Methods focusing purely on constraints if available (e.g., rule-based generation or specific models like Johnson & Williams, 2023).
*   **Evaluation Metrics:**
    *   **Statistical Fidelity:** JSD, Wasserstein distance, correlation matrix difference (as described in Quality Assessor). Visualizations (histograms, pair plots).
    *   **Constraint Compliance:** Percentage of final generated rows satisfying all schema constraints (should be 100% by design due to Validator, but we'll report the pre-filtering acceptance rate). We will specifically test effectiveness on UNIQUE, FK, and CHECK constraints.
    *   **Downstream Utility (TSTR):** Performance (Accuracy, F1, AUC / RMSE, MAE) of standard ML models trained on synthetic data and tested on real data, compared across different SDG methods. We will also evaluate in low-data scenarios (training ML models on small real subset + synthetic data).
    *   **Privacy Assessment:**
        *   *Membership Inference Attack (MIA):* Train an attacker model to distinguish between records used for training/statistics/fine-tuning and holdout records, using the synthetic data. Lower attack accuracy indicates better privacy.
        *   *Formal DP Guarantees:* Report the theoretical ($\epsilon, \delta$) values if DP mechanisms are employed.
*   **Ablation Studies:** To understand the contribution of each component, we will evaluate SynthTab variants:
    *   Without the RAG component.
    *   Without the Schema Validator (relying only on LLM prompting for constraints).
    *   Without the Quality Assessor feedback loop (single-pass generation).
    *   Without the Privacy Module vs. with varying levels of $\epsilon$.

## **4. Expected Outcomes & Impact**

### **Expected Outcomes**

1.  **A Functional SynthTab Framework:** The primary outcome will be a working prototype of the SynthTab multi-agent system capable of generating synthetic tabular data based on input schemas, statistics, and privacy requirements.
2.  **Superior Constraint Adherence:** We expect SynthTab to significantly outperform baseline methods (especially naive LLM generation and traditional GANs/VAEs) in adhering to complex schema constraints, including data types, uniqueness, referential integrity, and business rules, owing to the dedicated Schema Validator agent. The pre-filtering acceptance rate should improve over iterations due to feedback.
3.  **High Statistical Fidelity and Utility:** We anticipate that the combination of LLM generation, RAG for domain grounding, and iterative refinement driven by the Quality Assessor will lead to synthetic data with high statistical similarity to real data (low JSD/Wasserstein distances, accurate correlations) and strong performance in downstream ML tasks (TSTR metrics comparable to or potentially improving upon baselines, especially in low-data regimes).
4.  **Effective Privacy-Utility Trade-off:** The integrated Privacy Module is expected to provide measurable privacy guarantees (quantified by MIA resistance and/or formal $\epsilon, \delta$ values) while maintaining reasonable data utility. We aim to demonstrate how users can tune the privacy level and observe the corresponding impact on data quality.
5.  **Demonstration of Multi-Agent Synergy:** The results should highlight the benefits of the multi-agent approach, where specialized agents (Generator, Validator, Assessor) collaborate to overcome the limitations of monolithic generative models. Ablation studies will quantify the contribution of each agent.
6.  **Insights into LLMs for Structured Data:** The research will provide valuable insights into the challenges and effective strategies for applying LLMs to structured, constraint-heavy data generation tasks.

### **Deliverables**

*   **Source Code:** A publicly released codebase implementing the SynthTab framework and the experimental setup.
*   **Generated Datasets:** Examples of synthetic datasets generated by SynthTab for the benchmark scenarios.
*   **Publications:** At least one peer-reviewed publication detailing the framework, methodology, and experimental results, targeted at relevant ML conferences/workshops (like the Table Representation Learning Workshop) or journals.
*   **Technical Report:** A detailed report documenting the architecture, algorithms, and findings.

### **Broader Impact**

The SynthTab project has the potential for significant broader impact:

*   **Democratizing Access to Data:** By providing a reliable way to generate realistic synthetic data, SynthTab can lower barriers for researchers and small organizations that lack access to large, proprietary datasets.
*   **Enhancing Privacy Protection:** Offering a tool that incorporates rigorous privacy mechanisms like differential privacy can promote responsible data handling and enable safer sharing of insights derived from sensitive information in fields like healthcare and finance.
*   **Improving ML Model Robustness:** Better synthetic data for augmentation can lead to more robust and generalizable ML models, particularly when real data is scarce or imbalanced.
*   **Advancing AI for Structured Data:** This work contributes to the fundamental understanding of how advanced AI models like LLMs can be effectively adapted and controlled for tasks involving structured data, moving beyond their primary domain of natural language.
*   **Establishing Best Practices:** The comprehensive evaluation framework and focus on constraint adherence aim to set higher standards for the quality and trustworthiness required of synthetic tabular data.

Ultimately, SynthTab aims to provide a valuable contribution to the machine learning community by addressing a persistent challenge in working with tabular data, potentially unlocking new applications and fostering innovation in data-driven fields.

---
**References** (Based on provided literature review)

1.  Wang, Y., Feng, D., Dai, Y., Chen, Z., Huang, J., Ananiadou, S., Xie, Q., & Wang, H. (2024). *HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection*. arXiv:2408.02927.
2.  Zhao, Z., Birke, R., & Chen, L. (2023). *TabuLa: Harnessing Language Models for Tabular Data Synthesis*. arXiv:2310.12746.
3.  Nguyen, D., Gupta, S., Do, K., Nguyen, T., & Venkatesh, S. (2024). *Generating Realistic Tabular Data with Large Language Models*. arXiv:2410.21717.
4.  Xu, S., Lee, C.-T., Sharma, M., Yousuf, R. B., Muralidhar, N., & Ramakrishnan, N. (2024). *Are LLMs Naturally Good at Synthetic Tabular Data Generation?* arXiv:2406.14541.
5.  Doe, J., & Smith, J. (2023). *Differentially Private Synthetic Data Generation for Tabular Data*. arXiv:2305.12345. (Placeholder reference)
6.  Johnson, A., & Williams, B. (2023). *Schema-Constrained Generative Models for Tabular Data*. arXiv:2311.23456. (Placeholder reference)
7.  Brown, E., & Green, M. (2024). *Privacy-Preserving Tabular Data Synthesis Using GANs*. arXiv:2402.34567. (Placeholder reference)
8.  Lee, D., & Kim, S. (2024). *Constraint-Aware Data Augmentation for Tabular Data*. arXiv:2404.45678. (Placeholder reference)
9.  White, K., & Black, L. (2024). *Multi-Agent Systems for Synthetic Data Generation*. arXiv:2407.56789. (Placeholder reference)
10. Adams, R., & Brown, T. (2024). *Retrieval-Augmented Generation for Tabular Data Synthesis*. arXiv:2409.67890. (Placeholder reference)