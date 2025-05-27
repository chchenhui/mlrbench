# SecureED: A Contrastive Learning Framework for Detecting AI-Generated Responses in Educational Assessments

## 1. Introduction

### Background

The emergence of large foundation models (LFMs) such as ChatGPT, GPT-4, Llama, and Gemini has revolutionized numerous sectors, with education being particularly impacted. These sophisticated models can generate human-like text across various domains, creating both opportunities and challenges for educational assessment. While these technologies hold promise for enhancing assessment development and administration, they simultaneously pose a significant threat to assessment integrity when misused by students to generate responses to assignments and examinations.

Recent studies have highlighted the increasing sophistication of AI-generated content and the corresponding difficulty in distinguishing it from human-authored work. Weber-Wulff et al. (2023) and Elkhatat et al. (2023) have demonstrated that existing detection tools often struggle with reliability and accuracy, particularly when content is paraphrased or manipulated. This challenge is compounded in educational contexts where assessments span diverse subjects and question types, requiring detection methods to generalize across domains while maintaining high precision.

The reliability gap in current detection systems raises critical concerns about false accusations and threatens to erode trust in educational assessment processes. Axios (2024) reported that educators face mounting difficulties in identifying AI-generated student work, with existing tools exhibiting significant limitations. This situation creates an urgent need for more robust, adaptable detection frameworks specifically designed for educational contexts, where assessment integrity directly impacts learning outcomes and credential validity.

### Research Objectives

This research proposes SecureED, a novel contrastive learning framework designed to address the critical challenges of AI-generated content detection in educational assessments. The primary objectives of this research are:

1. To develop a robust, multimodal detection system capable of accurately distinguishing between human-authored and AI-generated responses across diverse subject areas and question types.

2. To implement a contrastive learning approach that captures distinctive features of human versus AI-generated educational responses, with emphasis on high-order thinking tasks.

3. To enhance model generalizability through domain adaptation techniques that allow effective detection across varying assessment contexts without requiring extensive domain-specific training.

4. To establish resistance against adversarial evasion tactics, including paraphrasing and other manipulation strategies.

5. To design an explainable detection framework that provides transparent reasoning for its classifications, supporting fair implementation in educational settings.

### Significance

The significance of this research lies in its potential to safeguard the integrity of educational assessments in an era of increasingly accessible and sophisticated AI tools. By developing reliable detection methods, SecureED will:

1. Preserve the validity and fairness of educational assessments, ensuring that credentials accurately reflect student knowledge and skills.

2. Enable educators to confidently integrate generative AI into educational practices while maintaining assessment security.

3. Provide a technical foundation for establishing ethical norms around AI use in education, distinguishing between appropriate learning applications and inappropriate substitution of student work.

4. Advance the technical capabilities of contrastive learning approaches for style and authorship distinction in educational contexts.

5. Create an open-source resource that can be adapted and integrated across educational platforms, democratizing access to detection capabilities.

As educational institutions globally grapple with the implications of generative AI for assessment integrity, SecureED represents a timely and essential contribution to maintaining the credibility of educational evaluation while embracing the benefits of AI in education.

## 2. Methodology

### 2.1 Data Collection and Preparation

The development of SecureED requires a comprehensive multimodal dataset encompassing diverse educational assessment contexts. The data collection process will involve:

1. **Dataset Construction**: We will create a balanced dataset of paired human-authored and AI-generated responses across multiple domains:
   - Text-based responses (essays, short answers, explanations)
   - Mathematical solutions and proofs
   - Programming code and algorithms
   - Scientific explanations and hypotheses
   - Visual reasoning tasks (diagrams, charts interpretations)

2. **Assessment Types**: Responses will span various assessment types including:
   - Factual recall questions
   - Conceptual understanding assessments
   - Problem-solving tasks
   - Critical analysis questions
   - Creative tasks
   - High-order reasoning problems

3. **Data Sources**:
   - Collaboration with educational institutions to collect anonymized human responses
   - Generation of responses using multiple LFMs (GPT-4, Llama, Gemini, Claude) with varying prompts and parameters
   - Collection of human responses attempting to mimic AI writing styles
   - Collection of AI responses that have been human-edited

4. **Preprocessing Pipeline**:
   - Anonymization of all human-authored content
   - Standardization of formatting
   - Creation of metadata tags for domain, question type, and complexity level
   - Verification of authentic authorship through controlled collection processes

The final training dataset will consist of at least 100,000 paired responses across multiple domains, with a validation set of 20,000 pairs and a held-out test set of 20,000 pairs.

### 2.2 Model Architecture

SecureED employs a contrastive learning architecture designed to capture subtle differences between human and AI-generated responses while generalizing across domains:

1. **Encoder Architecture**: We utilize a dual-encoder approach with:
   - A text encoder based on a pre-trained language model (e.g., RoBERTa-large)
   - Domain-specific encoders for mathematical expressions, code, and visual elements
   - A fusion mechanism to integrate multimodal information

2. **Multi-Level Feature Extraction**: Drawing from DeTeCtive (Guo et al., 2024), we implement a multi-level feature extraction approach:
   - Token-level features capturing word choice and lexical patterns
   - Sentence-level features for syntactic structures and transitions
   - Document-level features for organizational patterns and coherence
   - Task-specific features aligned with educational assessment types

3. **Domain Adaptation Module**: Inspired by ConDA (Bhattacharjee et al., 2023), we implement:
   - A domain adversarial training component
   - Cross-domain representation alignment
   - Subject-specific adaptation layers

The core architecture is illustrated in Figure 1 (not shown in this proposal but would depict the dual-encoder structure with contrastive learning components).

### 2.3 Contrastive Learning Framework

The contrastive learning approach forms the foundation of SecureED's detection capabilities:

1. **Training Objective**: For a response pair $(x_h, x_a)$ where $x_h$ is human-authored and $x_a$ is AI-generated, we minimize the following contrastive loss:

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(s(f(x_h), g(x_h))/\tau)}{\exp(s(f(x_h), g(x_h))/\tau) + \sum_{j=1}^{N} \exp(s(f(x_h), g(x_a^j))/\tau)}$$

Where:
- $f(\cdot)$ and $g(\cdot)$ are the encoders for different views of the input
- $s(\cdot,\cdot)$ is the similarity function (cosine similarity)
- $\tau$ is the temperature parameter
- $N$ is the number of negative samples

2. **Triplet Network Implementation**: Extending WhosAI (La Cava et al., 2024), we implement a triplet network approach:
   - Anchor: Human response to a question
   - Positive: Different human response to the same question
   - Negative: AI-generated response to the same question

3. **Educational Context Integration**: We enhance the contrastive learning with education-specific features:
   - Response coherence with the question
   - Reasoning pattern consistency
   - Domain-specific knowledge application
   - Error pattern analysis

The training process can be formalized as minimizing the combined loss:

$$\mathcal{L} = \mathcal{L}_{\text{contrast}} + \lambda_1 \mathcal{L}_{\text{domain}} + \lambda_2 \mathcal{L}_{\text{task}} + \lambda_3 \mathcal{L}_{\text{adv}}$$

Where $\lambda_1, \lambda_2, \lambda_3$ are weighting coefficients for the domain adaptation, task-specific, and adversarial components respectively.

### 2.4 Advanced Features for Educational Assessment

To specifically address educational assessment contexts, SecureED incorporates specialized features:

1. **Reasoning Coherence Analysis**: We implement a module that traces the logical progression of responses to identify patterns consistent with human versus AI reasoning:

$$C_r = \frac{1}{n-1} \sum_{i=1}^{n-1} \text{sim}(s_i, s_{i+1}) \cdot \text{rel}(s_i, s_{i+1}, q)$$

Where:
- $s_i$ represents the $i$-th statement in the response
- $\text{sim}(\cdot,\cdot)$ measures semantic similarity
- $\text{rel}(\cdot,\cdot,\cdot)$ assesses relevance to the question $q$

2. **Knowledge Consistency Verification**: This component examines whether the response demonstrates consistent knowledge depth:

$$K_c = \frac{1}{m} \sum_{j=1}^{m} \text{cons}(k_j, K_{\text{domain}})$$

Where:
- $k_j$ represents a knowledge claim in the response
- $K_{\text{domain}}$ is the domain knowledge reference
- $\text{cons}(\cdot,\cdot)$ measures consistency

3. **Creativity and Originality Metrics**: For assessments requiring creative thinking:

$$O_s = \alpha \cdot \text{uniqueness}(r) + \beta \cdot \text{divergence}(r) + \gamma \cdot \text{elaboration}(r)$$

Where $r$ is the response and $\alpha, \beta, \gamma$ are weighting parameters.

### 2.5 Adversarial Training and Evasion Resistance

To enhance robustness against evasion tactics, we implement:

1. **Adversarial Sample Generation**: Creating challenging examples through:
   - Paraphrasing AI-generated content
   - Human editing of AI-generated responses
   - Instruction-tuned AI responses designed to evade detection
   - Style transfer techniques applied to AI content

2. **Adversarial Training Process**: The model is trained with an adversarial objective:

$$\min_{\theta_D} \max_{\theta_A} \mathbb{E}_{x \sim P_{\text{data}}}[\log D_{\theta_D}(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D_{\theta_D}(A_{\theta_A}(z)))]$$

Where:
- $D_{\theta_D}$ is the detector with parameters $\theta_D$
- $A_{\theta_A}$ is the adversarial generator with parameters $\theta_A$

3. **Watermark Detection Integration**: Drawing from Kirchenbauer et al. (2023), we incorporate complementary detection of embedded watermarks when present in AI-generated text:

$$W_d = \text{KL}(P_{\text{observed}}(t) || P_{\text{expected}}(t))$$

Where $P_{\text{observed}}(t)$ and $P_{\text{expected}}(t)$ are the observed and expected token distributions.

### 2.6 Experimental Design and Evaluation

The evaluation of SecureED will be comprehensive and rigorous:

1. **Core Detection Performance Metrics**:
   - Accuracy, Precision, Recall, F1-score
   - Area Under the ROC Curve (AUC)
   - False positive and false negative rates
   - Confidence calibration

2. **Cross-Domain Generalization Tests**:
   - Training on certain subjects and testing on others
   - K-fold cross-validation across domains
   - Few-shot adaptation to new assessment types

3. **Robustness Testing Against Evasion**:
   - Evaluation against paraphrased AI content
   - Testing with human-edited AI responses
   - Assessment of detection capability against specialized evasion techniques

4. **Comparative Evaluation**:
   - Benchmarking against commercial detectors (GPTZero, Originality.AI, Copyleaks)
   - Comparison with academic detection frameworks (ConDA, DeTeCtive, WhosAI)
   - Ablation studies isolating the contribution of each component

5. **Educational Context Evaluation**:
   - Real-world testing in educational settings
   - User studies with educators
   - Assessment of integration feasibility with learning management systems

6. **Explainability Assessment**:
   - Quantitative evaluation of feature importance
   - Human evaluation of explanation quality
   - Assessment of decision transparency

The evaluation matrix will include diverse test sets encompassing different subject areas, question complexities, and response lengths to ensure comprehensive assessment of the model's performance across educational contexts.

## 3. Expected Outcomes & Impact

### 3.1 Technical Outcomes

The successful implementation of SecureED is expected to yield several significant technical outcomes:

1. **High-Performance Detection Framework**: A detection system achieving significantly higher accuracy (target >95%) and lower false positive rates (<2%) compared to existing tools, particularly for educational responses requiring higher-order thinking.

2. **Cross-Domain Generalization**: A model demonstrating robust performance across different subject domains without requiring extensive retraining, with less than 10% performance degradation when transferring to new domains.

3. **Resistance to Evasion Tactics**: Maintained detection accuracy (>85%) even when faced with sophisticated evasion strategies such as paraphrasing, human editing, or adversarial perturbations.

4. **Explainable Detection**: A transparent system that provides human-interpretable explanations for its classifications, highlighting the specific patterns that contributed to the detection decision.

5. **Educational Assessment Integration**: Technical APIs and integration frameworks allowing SecureED to be incorporated into various educational platforms and learning management systems.

6. **Open-Source Resources**: Publication of code, pretrained models, and integration documentation to support widespread adoption across educational contexts.

### 3.2 Scientific Contributions

SecureED will advance scientific knowledge in several key areas:

1. **Contrastive Learning Advances**: New insights into applying contrastive learning for stylistic and cognitive pattern differentiation, extending beyond current applications in the literature.

2. **Domain Adaptation Techniques**: Novel approaches to cross-domain generalization for text classification tasks, particularly in educational contexts with diverse subject matter.

3. **Educational Assessment Patterns**: Deeper understanding of the distinctive patterns that differentiate human and AI cognitive approaches to educational tasks, potentially informing cognitive science and educational research.

4. **Adversarial Robustness**: New methodologies for creating detection systems robust against sophisticated evasion techniques, contributing to the broader field of adversarial machine learning.

5. **Multimodal Integration**: Advanced techniques for fusing information across modalities (text, code, mathematical expressions) in detection frameworks.

### 3.3 Educational Impact

The broad impacts of SecureED on educational assessment and practice include:

1. **Assessment Integrity**: Preservation of the validity and reliability of educational assessments by providing educators with tools to identify AI-generated content reliably.

2. **Balanced AI Integration**: Supporting a balanced approach to AI in education, where AI tools can be used constructively for learning while maintaining assessment security.

3. **Reduced Assessment Anxiety**: Alleviating concerns among educators about the reliability of assessments in the age of generative AI, potentially reducing reactive policies like returning to in-person proctored examinations.

4. **Educational Equity**: Contributing to educational equity by reducing the advantage gained by students with greater access to and proficiency with AI tools in assessment contexts.

5. **Detection Standardization**: Establishing technical standards and best practices for AI-generated content detection in educational settings, potentially informing policy and practice.

6. **Trust in Educational Credentials**: Maintaining societal trust in educational credentials by ensuring that they continue to reflect authentic student abilities and knowledge.

### 3.4 Practical Applications

The practical applications of SecureED will extend to various educational contexts:

1. **Higher Education**: Integration with university learning management systems to support academic integrity in remote and online learning environments.

2. **K-12 Education**: Scaled implementations appropriate for primary and secondary education, helping teachers maintain assessment integrity while teaching appropriate AI use.

3. **Professional Certification**: Applications in professional credentialing and certification programs where credential validity is essential.

4. **Educational Technology**: Integration into existing educational technology platforms, providing developers with robust detection capabilities.

5. **Assessment Development**: Tools for assessment creators to evaluate the "AI-solvability" of their questions, potentially leading to more thoughtful assessment design.

The SecureED framework represents a significant advancement in our ability to maintain educational assessment integrity in the age of large foundation models. By leveraging contrastive learning approaches and focusing specifically on educational contexts, this research addresses a critical need in modern education systems while contributing valuable scientific insights to multiple fields of study.