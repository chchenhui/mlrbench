# Benchmark Cards: Standardizing Context and Holistic Evaluation for ML Benchmarks

## Introduction

### Background
Machine learning (ML) research critically relies on benchmarks to evaluate model performance, drive algorithmic innovation, and compare advancements. However, the current ecosystem faces significant challenges. Benchmarks often prioritize single aggregate metrics (e.g., accuracy on ImageNet), incentivizing leaderboard optimization over comprehensive model understanding. This reductionist approach overlooks critical factors such as fairness across subgroups, robustness to distribution shifts, computational efficiency, and ethical considerations. For instance, models excelling in traditional benchmarks may perpetuate biases or fail under real-world conditions, leading to unreliable deployments in high-stakes domains like healthcare and criminal justice.

The lack of standardized benchmark documentation exacerbates these issues. Unlike datasets and models, which have seen governance frameworks (e.g., FAIR principles for datasets and Model Cards for algorithmic transparency), benchmarks often lack explicit guidance on scope, holistic evaluation criteria, and limitations. This gap hampers reproducibility, contextualization, and responsible deployment of ML systems. The ML community has recognized this concern: proposals like HELM (Holistic Evaluation of Language Models) advocate multi-metric evaluation for language models, while federated learning communities have pioneered use-case-specific holistic frameworks. Yet, no universal standard exists to operationalize holistic evaluation for benchmarks *across disciplines*.

### Research Objectives
This research proposes **Benchmark Cards**, a structured documentation framework designed to:
1. Standardize *contextual* reporting of benchmark intentions and constraints.
2. Promote *multimetric evaluation* that captures fairness, robustness, efficiency, and other domain-specific concerns.
3. Mitigate misuse risks through explicit documentation of limitations and edge cases.

### Research Significance
The proposed work aligns directly with the workshop's goals to restructure the ML data ecosystem. By addressing overfitting through alternative evaluation paradigms, improving dataset reproducibility via documentation standards, and fostering responsible leaderboard design, Benchmark Cards could catalyze cultural shifts in:
- **Benchmark creation**: Encouraging designers to define "success" in nuanced, domain-specific terms.
- **Model selection**: Enabling practitioners to identify models truly suited to their use cases, not just leaderboard winners.
- **Evaluation practices**: Embedding accountability norms in ML research beyond univariate comparisons.

This framework fills a critical gap highlighted in prior work: while Model Cards and Datasheets standardize model and dataset documentation, benchmarks remain under-specified, perpetuating flawed evaluation cycles.

## Methodology

### Framework Design for Benchmark Cards
We propose a standardized Benchmark Card template with six key components:
1. **Intended Use & Scope**: 
   - Primary deployment scenarios (e.g., resource-constrained environments, medical diagnosis).
   - Exclusion criteria (e.g., inapplicable domains, cultural contexts).
2. **Dataset Composition**:
   - Summary statistics of underlying data (e.g., demographics, sampling biases).
   - Ethical considerations (sensitive attributes, data provenance issues).
3. **Evaluation Metrics**:
   - Core metrics (e.g., accuracy) used for primary leaderboards.
   - Contextual metrics (e.g., F1-score per subgroup, computational cost).
   - Optional weighted combination equation:
     $$
     \text{Composite Score} = \sum_{i=1}^{n} w_i \cdot \frac{\text{Metric}_i}{\tau_i}
     $$
     where $w_i$ represents use-case-specific weights normalized to $\sum w_i = 1$, and $\tau_i$ defines threshold penalties for minimal acceptable performance (reflected as constraint violation penalties in $\text{Composite Score}$).
4. **Robustness & Sensitivity Analysis**:
   - Performance under distribution shifts (e.g., adversarial examples, out-of-domain samples).
5. **Known Limitations**:
   - Identified failure cases, overfitting risks, and misuse scenarios.
6. **Version Control & Dependencies**:
   - Required software/hardware configurations and update history.

### Data Collection & Benchmark Selection
The framework will be piloted on 5-7 widely-used benchmarks spanning modalities (vision, NLP, tabular data) and paradigms (supervised, foundation models), including:
1. ImageNet (vision classification)
2. GLUE (NLP understanding)
3. CodeSearchNet (code generation)
4. PhysioNet (healthcare time-series)
5. OpenML-100 (tabular baselines)

Each Benchmark Card will be collaboratively developed using:
- Literature surveys of existing evaluations and critiques for the benchmark.
- Semi-structured interviews with 5-10 developers/maintainers of each benchmark.
- Delphi method iterations involving ML fairness researchers to settle on contextual metric weights.

### Algorithmic Implementation
To operationalize the composite scoring formula, we propose an adversarial weight-rebalancing process:
1. For each benchmark, experts define $k$ representative use cases $u_1,...,u_k$.
2. For each use case $u_j$, collect importance weights $w_{ij} \in [0,1]$ for metrics $i=1,...,n$ with $\sum_i w_{ij}=1$.
3. For any candidate model $m$, compute utility vector $v_m = [\text{Metric}_1,...,\text{Metric}_n]$.
4. Determine dominant use case $j^* = \arg\max_j w_j^T v_m$.
5. Compute robustness score $\rho$ penalizing extreme deviations from metric thresholds $\tau_i$.

This forces benchmark designers to preemptively define acceptable trade-offs, producing scores that align with contextual priorities.

### Experimental Validation
Validation will proceed in three phases:

#### Phase 1: Template Fidelity
Objective: Ensure cards accurately reflect benchmark designers' intentions.
- **Metrics**: Inter-rater agreement (Fleiss’ κ) between card authors and original benchmark developers across 20+ key properties (e.g., intended use cases, critical metrics).

#### Phase 2: Adoption Impact
Objective: Measure whether using Benchmark Cards improves evaluation practices.
- **Experimental Design**: 
  - Split 200+ ML researchers into a control group (given standard leaderboard) and intervention group (given the same leaderboard + Benchmark Card).
  - Both groups select "best models" for 5 fabricated (but realistic) deployment scenarios.
- **Hypotheses**:
  - **H1**: Intervention group picks models from different leaderboard positions in >60% of cases.
  - **H2**: Intervention group exhibits higher between-subject agreement on optimal choices.

#### Phase 3: Longitudinal Evaluation
Objective: Track real-world impact on ML publications.
- **Protocol**: Monitor changes in evaluation metrics reported in ICLR/NeurIPS papers over 24 months.
- **Metrics**: 
  - Percentage of papers using ≥3 contextual metrics vs. a single primary metric.
  - Citation rates of Benchmark Cards in datasets/methods sections.

### Ethical Considerations
The research involves minimal human subject risk as interviews with benchmark creators and crowdsourced surveys will be optional and anonymized. To address potential biases in expert weight elicitation, we ensure diverse representation across industry, academia, and geographic regions (n≥30 participants, 30% underrepresented in ML).

## Expected Outcomes & Impact

### Principal Deliverables
1. **Benchmark Card Template v1.0**: 
   - YAML-based schema with required/optional fields.
   - Python-based validation toolkit for repositories (e.g., HuggingFace Datasets).
2. **Initial Card Catalog**: 
   - 5-7 human-audited Benchmark Cards (ImageNet v2.0, GLUE+FAIR, CodeSearchNet2024, etc.).
   - GitHub-based interactive gallery with faceted search by application domain/typical metrics.
3. **Robust Decision Framework**:
   - Open-source implementation of weighted evaluation formulas with visualization dashboards.

### Anticipated Changes in Research Practice
1. **Cultural**: Shift from pursuing single "SOTA numbers" toward trade-off-aware model selection—as seen in HELM’s language model evaluations.
2. **Technical**: Proliferation of adversarial robustness and subgroup fairness metrics as routine evaluation standards.
3. **Institutional**: Integration with major repositories:
   - **OpenML**: Require Benchmark Cards for new dataset submissions.
   - **HuggingFace**: Banner alerts warning "This model was tested outside its benchmarked scope."
   - **NeurIPS**: Mandatory "Benchmarks Used" section in paper submissions.

### Measurable Impact Targets (2027)
1. Adoption: 50% of ICLR 2027 benchmarks cite Benchmark Cards, up from 20% at baseline (2024).
2. Evaluation Practices: Increase in multi-metric reporting: 70% (2027) vs. 35% (2024).
3. Misuse Reduction: 40% decrease in model reuses outside documented scopes (based on audit trails in repositories).

### Broader Implications
By institutionalizing contextual evaluation, this work could:
- Reduce deployment risks in safety-critical fields (e.g., healthcare diagnostics).
- Mitigate harms from overused benchmarks by expiring outdated artifacts (per documented deprecation criteria).
- Create feedback pathways: Models documented with Model Cards can now be evaluated against benchmarks designed for their intended context, closing loops in responsible ML development.

This proposal directly tackles the workshop’s themes—rethinking repositories as governance tools, standardizing dataset curation/metrics, and establishing alternative benchmarking paradigms. Success would mark a paradigm shift from viewing benchmarks as "leaderboards" to defining them as *"responsible evaluation contracts"*, binding research to real-world relevance.