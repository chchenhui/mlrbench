# SynthTab: A Schema-Aware Multi-Agent Framework for High-Fidelity Synthetic Tabular Data Generation

## 1. Introduction

### Background

Tabular data forms the backbone of modern data management and analysis pipelines, with structured data representing a significant portion of the world's information assets. Despite its ubiquity in real-world applications—from healthcare records to financial transactions—tabular data suffers from unique challenges that limit its utility in machine learning applications. Data scarcity, privacy concerns, and regulatory restrictions frequently constrain the availability of high-quality datasets for developing robust models.

The generation of synthetic tabular data has emerged as a promising solution to these challenges. By creating artificial data that maintains the statistical properties and semantic relationships of real data without exposing sensitive information, synthetic data generation can enable innovation while preserving privacy. Recent advances in large language models (LLMs) have demonstrated remarkable capabilities for understanding and generating text, code, and other modalities, suggesting potential for tabular data synthesis as well.

However, existing approaches to synthetic tabular data generation face significant limitations. Traditional statistical methods often struggle to capture complex relationships between columns. Deep learning approaches like GANs may generate unrealistic values or violate essential domain constraints. Furthermore, most current LLM-based solutions treat tabular data generation as a simplified text generation problem, ignoring the rich structural constraints and semantic relationships inherent in real-world schemas.

### Research Objectives

This research proposes SynthTab, a novel multi-agent framework for synthetic tabular data generation that addresses these limitations through a constraint-aware approach. The primary objectives of this research are:

1. Develop a schema-aware tabular data generation system that leverages the contextual understanding capabilities of LLMs while enforcing structural and semantic constraints.

2. Create a multi-agent architecture that separates concerns between data generation, constraint validation, quality assessment, and privacy preservation.

3. Implement and evaluate retrieval-augmented generation techniques to enhance the realism of synthetic data through access to domain-specific knowledge.

4. Design mechanisms for privacy-preserving synthetic data generation that provide formal guarantees against information leakage.

5. Validate the effectiveness of SynthTab across diverse domains and data schemas, demonstrating improvements in downstream model performance and data utility.

### Significance

The significance of this research extends across several dimensions:

**Data Augmentation**: In domains where data is scarce or expensive to collect, SynthTab can generate additional training samples that improve model performance, particularly for minority classes or rare phenomena.

**Privacy-Preserving Data Sharing**: Organizations can share synthetic versions of sensitive datasets for research or collaboration without compromising individual privacy or violating regulations like GDPR or HIPAA.

**Benchmark Creation**: The ability to generate large volumes of realistic tabular data enables the creation of diverse benchmarks for evaluating machine learning algorithms.

**Schema-Aware Generation**: Unlike approaches that treat tabular data as unstructured text, SynthTab explicitly models and preserves the complex interrelationships defined by database schemas, ensuring generated data maintains referential integrity, uniqueness constraints, and domain-specific rules.

**Production Readiness**: By addressing validation, quality assessment, and privacy concerns within a unified framework, SynthTab aims to bridge the gap between research prototypes and production-ready synthetic data generation systems.

## 2. Methodology

### 2.1 System Architecture

SynthTab employs a multi-agent architecture consisting of four specialized components working in concert:

1. **Data Generation Agent**: A fine-tuned LLM responsible for proposing candidate synthetic rows based on schema understanding and retrieval-augmented prompts.

2. **Schema Validator Agent**: Enforces structural constraints including data types, range limits, uniqueness, and referential integrity.

3. **Quality Assessor Agent**: Evaluates the statistical fidelity of generated data and its utility for downstream tasks.

4. **Privacy Guardian Agent**: Applies differential privacy mechanisms to bound information leakage.

These agents interact through a coordinated workflow, with feedback loops enabling iterative refinement of the synthetic data generation process. Figure 1 illustrates this architecture.

### 2.2 Data Preparation and Schema Understanding

Prior to generation, SynthTab processes the input schema and representative examples through the following steps:

1. **Schema Parsing**: Extract column definitions, data types, constraints (PRIMARY KEY, FOREIGN KEY, UNIQUE, NOT NULL), and value ranges from database schema definitions or infer them from example data.

2. **Statistical Analysis**: For each column, compute descriptive statistics including distribution type, central tendency measures, variance, and outlier information.

3. **Cross-Column Analysis**: Identify correlations, mutual information, and potential functional dependencies between columns.

4. **Knowledge Extraction**: For categorical columns, extract domain vocabulary and valid value sets.

These insights are encoded into a schema representation $S = \{C_1, C_2, ..., C_n\}$ where each column $C_i$ is defined by its properties:

$$C_i = \{name_i, type_i, constraints_i, statistics_i, relationships_i\}$$

### 2.3 Data Generation Agent

The Data Generation Agent utilizes a fine-tuned LLM to propose candidate rows for the synthetic dataset. We fine-tune a base LLM (e.g., Llama-3-70B) on a diverse collection of tabular datasets to enhance its understanding of tabular structures.

#### 2.3.1 Retrieval-Augmented Generation

To improve the realism and domain relevance of generated data, we implement a retrieval-augmented generation approach:

1. For each generation request, retrieve relevant examples from a reference dataset or knowledge base.
2. Construct a prompt that includes:
   - The formal schema definition
   - Column-specific statistics and constraints
   - Retrieved examples that illustrate typical patterns
   - Instructions for generating data that maintains relationships

The prompt template follows this structure:

```
You are tasked with generating synthetic tabular data that adheres to the following schema:
[SCHEMA_DEFINITION]

Column statistics:
[COLUMN_STATISTICS]

Schema constraints:
[SCHEMA_CONSTRAINTS]

Here are some example rows from similar data:
[RETRIEVED_EXAMPLES]

Generate [N] new rows that maintain the relationships between columns and adhere to all constraints. 
Each row should be realistic and consistent with the statistical properties described above.
```

#### 2.3.2 Generation Strategy

Rather than generating entire tables at once, SynthTab employs a hierarchical generation strategy:

1. **Column Ordering Determination**: Identify an optimal generation order for columns based on dependency analysis, starting with independent columns and progressing to dependent ones.

2. **Progressive Generation**: For each row, generate values column-by-column according to the determined order, conditioning each new column on previously generated values.

3. **Batched Processing**: Generate data in small batches (e.g., 10-50 rows) to enable efficient validation and feedback incorporation.

The mathematical formulation for this process is:

$$P(x_i | S, x_1, x_2, ..., x_{i-1})$$

where $x_i$ is the value for the $i$-th column in the generation order, $S$ is the schema representation, and $x_1, x_2, ..., x_{i-1}$ are previously generated values in the current row.

### 2.4 Schema Validator Agent

The Schema Validator Agent ensures that generated data adheres to all specified constraints through a chain-of-thought verification process:

#### 2.4.1 Constraint Types

The validator checks multiple constraint categories:

1. **Type Constraints**: Ensures values match their specified data types (numeric, date, categorical, etc.).

2. **Range Constraints**: Verifies values fall within allowed ranges (e.g., 0-100 for percentage fields).

3. **Uniqueness Constraints**: Checks that columns marked as UNIQUE or PRIMARY KEY contain no duplicates.

4. **Nullability Constraints**: Ensures NOT NULL columns always contain values.

5. **Referential Integrity**: Validates FOREIGN KEY relationships by confirming values exist in referenced tables.

6. **Custom Business Rules**: Applies domain-specific validation rules (e.g., "end_date must be after start_date").

#### 2.4.2 Validation Algorithm

The validation process occurs through a multi-stage pipeline:

1. **Individual Value Validation**: Check each value against its column-specific constraints.

2. **Row-Level Validation**: Verify intra-row relationships and business rules.

3. **Cross-Row Validation**: Ensure uniqueness and referential integrity across the entire dataset.

4. **Feedback Generation**: For invalid data, produce detailed error messages with suggested corrections.

The validator implements a formal verification function $V(D, S) \rightarrow \{valid, invalid\}$ where $D$ is the generated dataset and $S$ is the schema. For invalid data, it returns specific violations and correction suggestions.

### 2.5 Quality Assessor Agent

The Quality Assessor Agent evaluates the fidelity and utility of the generated data through multiple dimensions:

#### 2.5.1 Statistical Similarity Metrics

To measure how well synthetic data preserves the statistical properties of real data, we employ:

1. **Univariate Distribution Comparison**: Compare distributions of individual columns using metrics such as Jensen-Shannon divergence or Kolmogorov-Smirnov tests:

$$D_{KS}(F_1, F_2) = \sup_x |F_1(x) - F_2(x)|$$

2. **Correlation Preservation**: Measure preservation of correlation structures using matrix distance metrics:

$$d_{corr}(R_{real}, R_{synth}) = \|R_{real} - R_{synth}\|_F$$

where $R_{real}$ and $R_{synth}$ are correlation matrices for real and synthetic data, and $\|\cdot\|_F$ is the Frobenius norm.

3. **Principal Component Analysis**: Compare data variance along principal components.

#### 2.5.2 Machine Learning Utility

To assess whether synthetic data serves as an effective substitute for real data in machine learning tasks:

1. **Train-on-Synthetic, Test-on-Real (TSTR)**: Train models on synthetic data and evaluate on real test data.

2. **Train-on-Real, Test-on-Synthetic (TRTS)**: Train models on real data and evaluate on synthetic test data.

3. **Performance Gap Analysis**: Measure the performance difference between models trained on real versus synthetic data:

$$\Delta_{perf} = |Performance_{real} - Performance_{synth}|$$

#### 2.5.3 Feedback Loop Implementation

The Quality Assessor provides structured feedback to the Data Generation Agent to improve subsequent generations:

1. Identify columns or relationships with poor statistical fidelity.
2. Highlight under-represented patterns or classes.
3. Suggest specific adjustments to generation parameters.

This feedback is formalized as a set of adjustments $A = \{a_1, a_2, ..., a_m\}$ where each $a_j$ is a specific recommendation for improving generation quality.

### 2.6 Privacy Guardian Agent

The Privacy Guardian Agent ensures that synthetic data generation does not leak sensitive information about individuals in the original dataset:

#### 2.6.1 Differential Privacy Implementation

We implement differential privacy guarantees through:

1. **Noisy Aggregation**: Apply calibrated noise to statistical aggregates used during the generation process:

$$\tilde{q}(D) = q(D) + \text{Lap}\left(\frac{\Delta q}{\epsilon}\right)$$

where $q(D)$ is a query on the dataset, $\Delta q$ is the sensitivity of the query, and $\epsilon$ is the privacy budget.

2. **DP-Compliant Training**: When fine-tuning LLMs on sensitive data, employ differentially private optimization algorithms like DP-SGD:

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{1}{B} \sum_{i \in B_t} \text{clip}(\nabla \ell(x_i, \theta_t), C) + \text{Noise} \right)$$

where $\text{clip}(\cdot, C)$ clips gradients to norm $C$ and $\text{Noise}$ is calibrated to the sensitivity.

3. **Membership Inference Protection**: Apply specific techniques to reduce membership inference risks.

#### 2.6.2 Privacy Risk Assessment

The Privacy Guardian continuously evaluates potential privacy risks:

1. **Memorization Detection**: Test for verbatim copying of rare patterns from training data.

2. **Attribute Inference Analysis**: Assess whether sensitive attributes can be inferred from synthetic data.

3. **Privacy Budget Tracking**: Monitor cumulative privacy budget consumption to ensure bounds are maintained.

The overall privacy guarantee is formalized as $(\epsilon, \delta)$-differential privacy, where:

$$\Pr[M(D) \in S] \leq e^\epsilon \cdot \Pr[M(D') \in S] + \delta$$

for any two adjacent datasets $D$ and $D'$ differing in one record, any subset of outputs $S$, and the mechanism $M$ representing our synthetic data generation process.

### 2.7 Experimental Design

We will evaluate SynthTab's performance through comprehensive experiments across diverse datasets and scenarios:

#### 2.7.1 Datasets

We will use a diverse collection of tabular datasets spanning multiple domains:

1. **Financial**: Credit scoring datasets (e.g., German Credit, FICO)
2. **Healthcare**: Medical records (e.g., MIMIC-III, synthetically modified for privacy)
3. **E-commerce**: Customer transaction records
4. **Relational**: Multi-table datasets with complex relationships
5. **Time-series**: Temporal datasets with sequential dependencies

#### 2.7.2 Evaluation Protocols

For each dataset, we will conduct the following evaluations:

1. **Constraint Satisfaction**: Measure the percentage of generated rows that satisfy all schema constraints.

2. **Statistical Fidelity**: Compare statistical properties between real and synthetic data using the metrics described in Section 2.5.1.

3. **Machine Learning Utility**: Evaluate downstream task performance using the protocols from Section 2.5.2, employing multiple model types (decision trees, neural networks, etc.).

4. **Privacy Assessment**: Conduct membership inference and attribute inference attacks to evaluate privacy protection.

5. **Ablation Studies**: Evaluate the contribution of individual components by selectively disabling them.

#### 2.7.3 Baselines

We will compare SynthTab against state-of-the-art approaches:

1. **Traditional Methods**: SMOTE, Gaussian Copulas
2. **Deep Learning Methods**: CTGAN, TVAE
3. **LLM-Based Approaches**: HARMONIC, TabuLa, and direct prompting of foundation models
4. **Commercial Solutions**: Synthetic data platforms like Mostly.ai, Gretel.ai

For each baseline, we will use the same evaluation metrics, enabling direct comparisons of performance across all dimensions.

## 3. Expected Outcomes & Impact

### 3.1 Technical Contributions

SynthTab is expected to advance the state of the art in synthetic tabular data generation through several key contributions:

1. **Schema-Aware Generation Framework**: A novel architecture that explicitly incorporates schema constraints and relationships into the generation process, ensuring that synthetic data maintains structural integrity.

2. **Multi-Agent Validation System**: A comprehensive approach to validating and refining synthetic data across multiple dimensions (structural, statistical, utility, privacy) through specialized agents.

3. **Retrieval-Enhanced Tabular Generation**: New techniques for incorporating domain knowledge into LLM-based tabular data generation through retrieval-augmented prompting.

4. **Privacy-Utility Balancing Mechanisms**: Methods for controlling the trade-off between data utility and privacy protection with formal guarantees.

5. **Comprehensive Evaluation Framework**: A standardized approach to assessing synthetic tabular data quality across multiple dimensions.

### 3.2 Practical Impact

Beyond its technical contributions, SynthTab is expected to enable several practical applications:

1. **Enhanced Machine Learning in Low-Data Regimes**: Organizations will be able to augment limited datasets with high-quality synthetic samples, improving model performance where data collection is expensive or restricted.

2. **Privacy-Preserving Data Sharing**: SynthTab will enable sharing of realistic synthetic versions of sensitive datasets for research, development, and collaboration while protecting individual privacy.

3. **Improved Testing and Development**: Software engineers and data scientists will gain access to realistic test data that maintains the complexity of production environments without exposing sensitive information.

4. **Regulatory Compliance**: Organizations subject to data protection regulations can leverage SynthTab to create compliant synthetic datasets for use cases that would otherwise be restricted.

5. **Equitable Access to Benchmark Data**: The research community will benefit from broader access to realistic tabular datasets across domains, potentially addressing disparities in access to high-quality training data.

### 3.3 Future Research Directions

The development of SynthTab opens several promising avenues for future research:

1. **Cross-Modal Synthesis**: Extending the framework to handle mixed-modality datasets that combine tabular data with text, images, or time series.

2. **Hierarchical and Temporal Data**: Adapting the approach to more complex data structures like hierarchical records or temporal sequences with causal dependencies.

3. **Interactive Data Generation**: Developing interfaces that allow users to iteratively refine synthetic data generation through natural language feedback.

4. **Domain Adaptation**: Creating specialized versions of SynthTab for high-value domains like healthcare, finance, and scientific research.

5. **Federated Synthetic Data**: Exploring techniques for generating synthetic data from multiple sources without centralizing the original sensitive data.

By addressing the fundamental challenges of constraint-aware synthetic tabular data generation, SynthTab aims to bridge an important gap in the current landscape of AI tools, enabling broader and safer use of machine learning in domains where data availability and privacy concerns have previously limited adoption.