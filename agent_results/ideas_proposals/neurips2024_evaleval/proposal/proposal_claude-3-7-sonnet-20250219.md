# CoEval: A Collaborative Multi-Stakeholder Framework for Standardized Societal Impact Assessment of Generative AI

## 1. Introduction

### Background

The rapid advancement and widespread deployment of generative artificial intelligence (AI) systems have led to profound societal implications across domains including text, image, audio, and video generation. These systems are increasingly influencing multiple facets of human life, from education and healthcare to media production and information dissemination. While the NeurIPS Broader Impact statement requirement has successfully shifted norms for AI publications to consider potential negative societal impacts, the field lacks standardized, systematic approaches to impact evaluation (Solaiman et al., 2023). This gap is particularly concerning as generative AI systems scale and become more pervasive in society.

Current evaluation practices suffer from several critical limitations. First, they are predominantly expert-centric, often excluding the perspectives of end-users, marginalized communities, and other stakeholders directly affected by these technologies. Second, assessments tend to be ad hoc, lacking methodological consistency and reproducibility. Third, there is insufficient integration between technical performance metrics and social impact considerations, creating a disconnect between what is measured and what truly matters for societal well-being (Chouldechova et al., 2024).

Recent research has begun addressing these challenges. Mun et al. (2024) demonstrated the value of democratic assessment through their PARTICIP-AI framework, which involves laypeople in speculating about AI use cases and impacts. Solaiman et al. (2023) proposed categories for evaluating social impacts across generative modalities. Chouldechova et al. (2024) introduced measurement theory from social sciences to improve evaluation validity. Parthasarathy et al. (2024) advocated for participatory approaches throughout the AI lifecycle. Despite these advances, a comprehensive, standardized framework that integrates these insights and can be widely adopted is still lacking.

### Research Objectives

This research proposes CoEval, a Collaborative Multi-Stakeholder Framework for Standardized Societal Impact Assessment of Generative AI. The primary objectives of this research are to:

1. Develop and validate a participatory framework (CoEval) that systematically incorporates diverse stakeholder perspectives in generative AI impact assessments
2. Create standardized, modular evaluation protocols that balance rigor with practical implementation
3. Design and test a mixed-methods toolkit for measuring impacts across different generative AI modalities (text, image, audio)
4. Establish a living repository and knowledge-sharing platform to facilitate broader adoption of impact evaluation best practices
5. Formulate evidence-based policy recommendations for the responsible development and governance of generative AI systems

### Significance

This research addresses a critical gap in the field by transforming how generative AI impacts are evaluated and by democratizing the assessment process. Its significance lies in several key contributions:

1. **Methodological Innovation**: CoEval bridges evaluation science with participatory methods to create a novel approach that ensures technical rigor while incorporating diverse perspectives.

2. **Standardization**: By providing structured protocols and metrics, this research will enable more consistent and comparable impact assessments across different generative AI systems.

3. **Inclusivity**: The framework intentionally centers historically underrepresented stakeholders in the evaluation process, ensuring more equitable consideration of AI's societal effects.

4. **Practical Utility**: CoEval's modular design allows for adaptability across contexts while maintaining core methodological principles, facilitating broader adoption by researchers, industry practitioners, and policymakers.

5. **Policy Relevance**: The findings will directly inform evidence-based policy recommendations and governance frameworks for generative AI development and deployment.

By addressing these challenges, CoEval will contribute to more responsible, transparent, and socially beneficial generative AI systems, ultimately helping to align technological advancement with broader societal values and needs.

## 2. Methodology

The CoEval framework employs a mixed-methods, participatory approach structured in three interconnected phases. Each phase builds upon established methodologies while introducing innovations specific to generative AI evaluation.

### 2.1. Phase 1: Multi-Stakeholder Co-Design Workshops

The first phase establishes context-specific impact criteria through structured co-design workshops with diverse stakeholders.

#### 2.1.1. Stakeholder Identification and Recruitment

We will employ a modified Stakeholder Analysis Matrix (SAM) to systematically identify and recruit participants across four key stakeholder categories:

1. **Technical Experts**: AI researchers, developers, and engineers
2. **Domain Experts**: Specialists from relevant application domains (e.g., healthcare, education, journalism)
3. **End Users**: Direct and indirect users of generative AI systems
4. **Governance Actors**: Policymakers, regulators, and ethics board members

For each stakeholder category, we will assess:
- Power/influence level (1-5 scale)
- Interest/impact level (1-5 scale)
- Representation of marginalized or vulnerable communities (priority weighting)

Recruitment will target a minimum of 20 participants per workshop, with deliberate oversampling of underrepresented stakeholders to ensure diversity.

#### 2.1.2. Workshop Structure and Facilitation

Workshops follow a structured three-stage protocol:

1. **Divergent Exploration** (90 minutes): Using modified nominal group technique, participants individually identify potential impacts of the generative AI system under evaluation across seven categories from Solaiman et al. (2023):
   - Bias and fairness
   - Cultural values and norms
   - Economic impacts
   - Environmental costs
   - Information integrity
   - Privacy and surveillance
   - Psychological effects

2. **Convergent Prioritization** (90 minutes): Through a facilitated card-sorting exercise, participants collectively cluster and prioritize impacts using a consensus-based approach. The prioritization formula integrates:

$$P_i = \frac{1}{N} \sum_{j=1}^{N} (w_j \times S_{ij})$$

Where:
- $P_i$ is the priority score for impact $i$
- $N$ is the number of participants
- $w_j$ is the weighting factor for stakeholder $j$ (adjusted to amplify underrepresented voices)
- $S_{ij}$ is the score (1-5) assigned by stakeholder $j$ to impact $i$

3. **Metrics Development** (120 minutes): Using a modified Delphi method, participants collaboratively develop evaluation metrics for the top-priority impacts. For each metric, they determine:
   - Operational definition
   - Measurement approach (qualitative, quantitative, or mixed)
   - Threshold values for acceptable/concerning outcomes
   - Contextual considerations for interpretation

#### 2.1.3. Workshop Analysis and Documentation

Workshop outputs will be systematically documented using a standardized protocol:

1. **Impact Map**: Visual representation of identified impacts and their relationships
2. **Metrics Specification Document**: Detailed description of each prioritized metric
3. **Stakeholder Value Alignment Matrix**: Documentation of differing perspectives and priorities across stakeholder groups
4. **Procedural Validity Assessment**: Evaluation of the workshop process using established participatory research criteria

### 2.2. Phase 2: Mixed-Methods Assessment Toolkit

Phase 2 involves developing and implementing a modular assessment toolkit based on workshop outputs.

#### 2.2.1. Toolkit Development

The assessment toolkit will include four components:

1. **Survey Instruments**: Standardized questionnaires for measuring user perceptions and experiences, adapted from established instruments in social psychology and human-computer interaction. Survey design will follow best practices including:
   - Cognitive pre-testing with diverse respondents
   - Psychometric validation (reliability and validity testing)
   - Adaptive question routing to reduce respondent burden

2. **Qualitative Assessment Protocols**: Semi-structured interview guides and focus group scripts to capture nuanced stakeholder perspectives. These will include:
   - Core question sets for cross-study comparability
   - Optional modules for context-specific exploration
   - Structured coding frameworks for analysis

3. **Computational Metrics**: Algorithmic measures of system behavior and outputs, including:
   - Bias detection algorithms for different modalities
   - Information quality assessments
   - Representation and diversity metrics
   - Automated documentation of system capabilities and limitations

4. **Scenario Simulations**: Structured protocols for testing system behavior in high-stakes or edge cases, including:
   - Adversarial testing scenarios
   - Context-shift robustness evaluation
   - Longitudinal performance tracking

#### 2.2.2. Implementation Process

The assessment process follows a systematic workflow:

1. **Preparation**: Customization of toolkit components based on workshop outputs and system characteristics
2. **Sampling Strategy**: Development of stakeholder sampling plan using:

$$n_i = N \times \frac{w_i \times p_i}{\sum_{j=1}^{k} (w_j \times p_j)}$$

Where:
- $n_i$ is the sample size for stakeholder group $i$
- $N$ is the total sample size
- $w_i$ is the importance weight for group $i$
- $p_i$ is the proportion of the population in group $i$
- $k$ is the number of stakeholder groups

3. **Data Collection**: Parallel implementation of multiple assessment methods:
   - Surveys (minimum n=200 per stakeholder category)
   - Qualitative interviews (n=15-20 per stakeholder category)
   - Focus groups (3-5 groups of 6-8 participants each)
   - Computational analysis of system outputs
   - Scenario testing (minimum 20 scenarios)

4. **Integrated Analysis**: Mixed-methods analysis using explanatory sequential design:
   - Quantitative analysis of survey and computational data
   - Qualitative analysis of interviews and focus groups
   - Integration of findings through joint displays and meta-matrices
   - Identification of convergence and divergence across methods

#### 2.2.3. Measurement Validation

Following Chouldechova et al. (2024), we will rigorously validate all measurements through:

1. **Content Validity**: Expert review of metrics to ensure comprehensive coverage
2. **Construct Validity**: Statistical validation of measurement constructs:
   - Factor analysis for survey instruments
   - Inter-rater reliability for qualitative coding (Cohen's κ ≥ 0.8)
   - Algorithmic validation for computational metrics

3. **Criterion Validity**: Correlation with established benchmarks where available
4. **Ecological Validity**: Assessment of real-world applicability and generalizability

### 2.3. Phase 3: Knowledge Sharing and Policy Translation

The final phase focuses on documenting, disseminating, and implementing findings.

#### 2.3.1. Living Repository Development

We will create an open-source repository with:

1. **Evaluation Protocols**: Step-by-step guidelines for implementing CoEval
2. **Data Library**: Anonymized datasets from pilot implementations
3. **Metrics Database**: Searchable archive of validated metrics across domains
4. **Case Studies**: Documented examples of CoEval applications
5. **Implementation Tools**: Code, templates, and facilitation resources

The repository will be built on FAIR principles (Findable, Accessible, Interoperable, Reusable) and include version control and community contribution mechanisms.

#### 2.3.2. Policy Recommendation Development

We will translate evaluation findings into policy guidance through:

1. **Impact Classification Framework**: Standardized taxonomy for categorizing and communicating impacts:

$$I = f(S, C, M, T)$$

Where:
- $I$ represents impact category
- $S$ represents severity (1-5 scale)
- $C$ represents certainty of evidence (1-5 scale)
- $M$ represents modality (text, image, audio, video)
- $T$ represents timeframe (immediate, short-term, long-term)

2. **Governance Model Templates**: Adaptable frameworks for oversight mechanisms
3. **Regulatory Guidance**: Technical specifications for compliance assessments
4. **Public Communication Materials**: Accessible summaries for broader dissemination

#### 2.3.3. Validation and Refinement

We will validate the framework through:

1. **Pilot Testing**: Implementation across three generative AI domains:
   - Large language models (text generation)
   - Image generation systems
   - Audio synthesis platforms

2. **Comparative Analysis**: Assessment of CoEval against existing frameworks:

$$E = \sum_{i=1}^{n} w_i \times \frac{CoEval_i - Baseline_i}{Baseline_i}$$

Where:
- $E$ is the overall effectiveness score
- $w_i$ is the weight for evaluation criterion $i$
- $CoEval_i$ is the performance of CoEval on criterion $i$
- $Baseline_i$ is the performance of baseline methods on criterion $i$

3. **Iterative Refinement**: Continuous improvement based on implementation feedback:
   - Quarterly review and update cycles
   - Stakeholder feedback integration
   - Adaptation to emerging challenges and technologies

### 2.4. Experimental Design for Framework Validation

To rigorously validate the CoEval framework, we will conduct a multi-phase experimental evaluation:

#### 2.4.1. Comparative Study Design

We will evaluate CoEval against two baseline approaches:

1. **Expert-only assessment** (current standard practice)
2. **PARTICIP-AI framework** (Mun et al., 2024)

For each approach, we will assess the same three generative AI systems (one text-based, one image-based, and one audio-based) using a within-subjects design.

#### 2.4.2. Evaluation Metrics

Performance will be measured across four dimensions:

1. **Comprehensiveness**: Breadth and depth of identified impacts
   - Number of unique impacts identified
   - Coverage across the seven impact categories
   - Depth of analysis for each impact

2. **Inclusivity**: Representation of diverse perspectives
   - Diversity index of stakeholder participation
   - Equity of voice across stakeholder groups
   - Representation of marginalized communities

3. **Practicality**: Implementation feasibility
   - Time and resource requirements
   - Ease of adaptation across contexts
   - Scalability potential

4. **Actionability**: Utility for decision-making
   - Specificity of recommendations
   - Alignment between metrics and organizational actions
   - Stakeholder satisfaction with outputs

#### 2.4.3. Statistical Analysis

We will employ mixed-effects models to account for nested data structures:

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \gamma_k + \epsilon_{ijk}$$

Where:
- $Y_{ijk}$ is the outcome for framework $i$ on system $j$ for metric $k$
- $\mu$ is the overall mean
- $\alpha_i$ is the effect of framework $i$
- $\beta_j$ is the effect of system $j$
- $(\alpha\beta)_{ij}$ is the interaction effect
- $\gamma_k$ is the random effect for metric $k$
- $\epsilon_{ijk}$ is the error term

## 3. Expected Outcomes & Impact

The CoEval research project is expected to produce significant contributions across four key domains:

### 3.1. Methodological Contributions

1. **Validated Participatory Framework**: A comprehensive, tested framework for collaborative generative AI impact assessment that balances scientific rigor with inclusive participation.

2. **Standardized Metrics and Protocols**: A suite of validated assessment tools that enable consistent, comparable evaluations across systems and contexts.

3. **Integration Model**: A systematic approach to integrating qualitative and quantitative data from diverse stakeholders into coherent impact assessments.

4. **Measurement Theory Application**: Practical demonstration of how social science measurement principles can strengthen AI evaluation practices.

### 3.2. Knowledge Resources

1. **Open-Source Repository**: A comprehensive, accessible collection of evaluation tools, protocols, and datasets that will serve as a community resource.

2. **Case Study Library**: Documented examples of framework implementation across different generative AI systems and contexts, providing practical guidance for future applications.

3. **Metrics Database**: A searchable collection of validated metrics for assessing different dimensions of generative AI impact, supporting evidence-based evaluation practices.

4. **Best Practices Guide**: A detailed handbook for conducting collaborative impact assessments, addressing common challenges and providing practical solutions.

### 3.3. Policy Implications

1. **Governance Recommendations**: Evidence-based proposals for governance structures that incorporate multi-stakeholder perspectives in generative AI oversight.

2. **Regulatory Frameworks**: Technical specifications and standards that can inform regulatory approaches to generative AI assessment and monitoring.

3. **Implementation Guidance**: Practical guidelines for organizations seeking to implement responsible AI practices, tailored to different organizational contexts and capacities.

4. **Public Engagement Model**: A tested approach for meaningful public participation in AI governance that goes beyond superficial consultation.

### 3.4. Societal Impact

1. **Democratized Evaluation**: Broader participation in assessing generative AI impacts, ensuring that historically marginalized perspectives shape AI accountability.

2. **Enhanced Transparency**: Improved documentation and communication of generative AI capabilities, limitations, and potential impacts.

3. **Risk Mitigation**: More comprehensive identification and addressing of potential harms before systems are widely deployed.

4. **Trust Building**: Stronger foundations for public trust in generative AI through inclusive, transparent evaluation processes.

The broader impact of this work extends beyond academic contributions to practical change in how generative AI systems are developed, evaluated, and governed. By creating a standardized yet flexible framework that centers diverse stakeholder perspectives, CoEval will help shift industry practices toward more responsible innovation. The open-source nature of all project outputs will democratize access to high-quality evaluation tools, enabling broader participation in shaping the future of generative AI.

Furthermore, the policy recommendations emerging from this research will provide evidence-based guidance for regulators and policymakers seeking to develop appropriate governance mechanisms for rapidly evolving generative AI technologies. This contribution is particularly timely given increasing calls for regulatory frameworks that effectively balance innovation with protection against potential harms.

Ultimately, CoEval aims to transform generative AI evaluation from a primarily technical exercise conducted by experts to a collaborative process that meaningfully incorporates the perspectives of all those affected by these powerful technologies. In doing so, it will help ensure that generative AI development aligns with broader societal values and contributes positively to human flourishing.