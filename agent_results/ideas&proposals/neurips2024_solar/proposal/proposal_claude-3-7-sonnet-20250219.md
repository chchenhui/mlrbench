# Interpretable Language Models for Low-Resource Languages: A Culturally-Grounded Approach to Transparency and Equity

## 1. Introduction

Language Models (LMs) have become integral to natural language processing applications, from machine translation to content generation. However, their benefits remain largely inaccessible to speakers of low-resource languages. These languages—often spoken by marginalized communities—lack the extensive digital corpora that high-resource languages enjoy, resulting in poorer model performance and reliability. Recent developments like InkubaLM (Tonja et al., 2024) and Glot500 (Imani et al., 2023) have demonstrated progress in addressing data scarcity issues, yet a critical gap remains: interpretability.

The opacity of language models poses particular risks for low-resource language communities. When models fail or exhibit biases, users lack the tools to understand these failures, making it impossible to contest or rectify problematic outputs. This opacity undermines trust in these technologies and can perpetuate inequities. As Bansal et al. (2021) demonstrated with Sumerian cuneiform, extremely low-resource languages present unique challenges that require specialized approaches to interpretability and evaluation.

The problem is twofold: technical and cultural. First, existing interpretability methods are designed for high-resource languages with abundant training data and well-understood linguistic features, making them ill-suited for the morphological complexity and code-switching patterns common in many low-resource languages (Tanaka et al., 2024; Sharma et al., 2023). Second, explanation interfaces rarely consider the cultural communication norms of target communities, creating disconnects between model explanations and user understanding (Gonzalez et al., 2023).

This research aims to address these gaps by developing a culturally-grounded interpretability framework for low-resource language models. Our approach integrates technical innovations with community participation to create explanation methods that are both technically sound and culturally appropriate. By extending local explanation techniques to accommodate linguistic diversity while co-designing interfaces with native speakers, we aim to enhance both model transparency and community agency.

Our research objectives are:

1. To develop and adapt explanation techniques that accurately capture the linguistic features unique to low-resource languages
2. To co-design culturally appropriate explanation interfaces with native speaker communities
3. To establish evaluation metrics that assess both technical robustness and user-perceived trustworthiness
4. To create open-source tools that enable community-led auditing and refinement of language models

This work contributes to the broader goal of socially responsible language modeling by addressing the equity and transparency concerns highlighted in the literature on bias and exclusion in LMs (Blodgett et al., 2020) and the need for interpretable AI systems (Lipton, 2018). By focusing on low-resource languages, we target a critical area where technical innovation can directly promote social inclusion and linguistic diversity in AI development.

## 2. Methodology

Our research methodology encompasses three interconnected components: (1) adaptation of explanation techniques for linguistic diversity, (2) community-centered interface design, and (3) comprehensive evaluation. Each component addresses different aspects of the interpretability challenge for low-resource language models.

### 2.1 Adapting Explanation Techniques for Linguistic Diversity

Existing local explanation methods like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) typically operate at the token level, assuming linguistic units that may not align with the morphological structures of low-resource languages. We will extend these methods to accommodate language-specific features through the following steps:

#### 2.1.1 Morphological Adaptation

We will modify feature attribution methods to account for complex morphological structures through:

1. **Morpheme-based segmentation**: Instead of token-level attributions, we will implement morpheme-level analysis using language-specific segmentation tools where available, or unsupervised morphological segmentation algorithms (Morfessor, Byte-Pair Encoding) where specialized tools are lacking.

2. **Hierarchical feature attribution**: We will develop a hierarchical attribution system that can assign importance at multiple linguistic levels (morpheme, word, phrase) using the following formula:

$$Attr(x_i) = \sum_{j \in \mathcal{M}(i)} w_j \cdot \phi_j$$

Where $Attr(x_i)$ is the attribution for linguistic unit $x_i$, $\mathcal{M}(i)$ is the set of morphemes comprising $x_i$, $\phi_j$ is the base attribution value for morpheme $j$, and $w_j$ is a weight reflecting morphological importance within the unit.

#### 2.1.2 Code-Switching Handling

To address code-switching (the alternation between multiple languages), we will:

1. **Language identification**: Implement a fine-grained language identification system based on GlotLID (Kargaran et al., 2023) to identify language boundaries within mixed text.

2. **Cross-lingual attribution alignment**: Develop a method to normalize attribution values across languages with the following formulation:

$$\hat{\phi}_i = \frac{\phi_i - \mu_{L(i)}}{\sigma_{L(i)}} \cdot \sigma_{ref} + \mu_{ref}$$

Where $\phi_i$ is the raw attribution for unit $i$, $L(i)$ is the language of unit $i$, $\mu_{L(i)}$ and $\sigma_{L(i)}$ are the mean and standard deviation of attributions in language $L(i)$, and $\mu_{ref}$ and $\sigma_{ref}$ are reference values for normalization.

#### 2.1.3 Contextual and Cultural References

We will enhance explanation systems to identify and explicate cultural references by:

1. **Cultural knowledge base**: Develop language-specific knowledge bases containing cultural references, idioms, and context-dependent expressions.

2. **Contextual relevance scoring**: Implement a scoring function that identifies when cultural knowledge is necessary for interpretation:

$$CR(x, c) = cos(emb(x), emb(c)) \cdot W_{cultural}$$

Where $CR(x, c)$ is the cultural relevance score for text unit $x$ in context $c$, $emb()$ produces embeddings, $cos()$ is cosine similarity, and $W_{cultural}$ is a learned weight matrix for cultural relevance.

### 2.2 Community-Centered Interface Design

To ensure explanations align with cultural communication norms, we will employ participatory design methods:

#### 2.2.1 Community Recruitment and Engagement

We will engage with native speaker communities through:

1. **Diverse participant recruitment**: Collaborate with language advocacy organizations to recruit 15-20 participants per language, ensuring diversity in age, gender, education, and technological experience.

2. **Participatory workshops**: Conduct a series of 3-5 workshops per community focusing on:
   - Current challenges with language technologies
   - Cultural norms for explanation and justification
   - Feedback on prototype interfaces

#### 2.2.2 Interface Prototyping and Iteration

Based on community input, we will develop:

1. **Explanation prototype portfolio**: Create multiple prototype interfaces demonstrating different explanation paradigms (visual, textual, comparative, contrastive).

2. **Iterative refinement**: Conduct at least three rounds of design, feedback, and refinement with community participants.

3. **Cultural alignment assessment**: Develop metrics to evaluate how well interfaces align with cultural expectations:

$$CA_i = \frac{1}{N} \sum_{j=1}^{N} (w_j \cdot S_{ij})$$

Where $CA_i$ is the cultural alignment score for interface $i$, $N$ is the number of cultural dimensions evaluated, $w_j$ is the importance weight for dimension $j$, and $S_{ij}$ is the alignment score for interface $i$ on dimension $j$.

#### 2.2.3 Accessibility and Deployment Considerations

To ensure explanations are accessible to all community members, we will:

1. **Technical requirement minimization**: Design interfaces that function on low-bandwidth connections and basic devices.

2. **Multimodal explanations**: Develop options for visual, textual, and audio-based explanations based on user preferences and needs.

3. **Local deployment options**: Create solutions for offline or locally-hosted explanations to accommodate limited internet access.

### 2.3 Comprehensive Evaluation Framework

We will evaluate our interpretability framework using both technical and user-centered metrics:

#### 2.3.1 Technical Robustness Evaluation

To assess the technical quality of explanations, we will:

1. **Faithfulness testing**: Measure how accurately explanations reflect model behavior using input perturbation:

$$Faith(E) = corr(\Delta pred(x, x'), \Delta attr(x, x'))$$

Where $Faith(E)$ is the faithfulness score for explanation method $E$, $\Delta pred(x, x')$ is the change in model prediction between original input $x$ and perturbed input $x'$, and $\Delta attr(x, x')$ is the change in attribution values.

2. **Consistency evaluation**: Assess explanation stability under similar inputs:

$$Cons(E) = 1 - \frac{1}{|\mathcal{P}|} \sum_{(x_i, x_j) \in \mathcal{P}} d(E(x_i), E(x_j))$$

Where $Cons(E)$ is the consistency score, $\mathcal{P}$ is the set of similar input pairs, and $d(E(x_i), E(x_j))$ is a distance measure between explanations.

3. **Cross-lingual evaluation**: For languages with code-switching, evaluate explanation consistency across language boundaries within the same text.

#### 2.3.2 User-Perceived Trust Assessment

To evaluate user trust and understanding, we will:

1. **Task-based evaluation**: Measure users' ability to predict model behavior, detect errors, and correct mistakes using explanations.

2. **Trust scale surveys**: Adapt validated trust measurement instruments to assess:
   - Perceived competence
   - Perceived benevolence
   - Perceived integrity
   - Calibrated trust (alignment between trust and system capabilities)

3. **Qualitative feedback**: Conduct semi-structured interviews to capture nuanced perceptions of explanation quality and cultural appropriateness.

#### 2.3.3 Community Impact Evaluation

To assess broader impact on language communities, we will:

1. **Technology adoption metrics**: Track usage patterns and sustained engagement with explanatory tools.

2. **Community agency assessment**: Measure instances of community-led model assessment and refinement.

3. **Linguistic inclusion perception**: Evaluate changes in perceptions of technological inclusivity before and after deployment.

### 2.4 Experimental Protocol

We will validate our approach across four low-resource languages chosen to represent diverse linguistic families and features:

1. A language with complex morphology (e.g., Amharic)
2. A language with frequent code-switching (e.g., Wolof)
3. A language with limited digital presence (e.g., Quechua)
4. A language with significant dialectal variation (e.g., Kurdish)

For each language, we will:

1. **Baseline model development**: Fine-tune existing multilingual models (e.g., InkubaLM, Glot500) on available language-specific data.

2. **Explanation generation**: Apply both standard and our adapted explanation methods to model outputs.

3. **Community evaluation**: Engage 15-20 native speakers per language in both technical evaluation and interface assessment.

4. **Comparative analysis**: Measure the performance gap between standard and adapted explanation methods across all evaluation metrics.

The experimental timeline will span 18 months, with 3-4 months dedicated to each language, allowing for sequential improvement of methods based on earlier findings.

## 3. Expected Outcomes & Impact

This research project is expected to yield several significant outcomes with important implications for socially responsible language modeling research:

### 3.1 Technical Outcomes

1. **Adapted Explanation Algorithms**: Novel algorithms extending SHAP and LIME for morphologically rich and code-switching languages, accounting for linguistic features often overlooked in current explainability methods.

2. **Open-source Interpretability Toolkit**: A comprehensive, accessible toolkit for generating explanations in low-resource language models, including:
   - Language-specific morphological processors
   - Code-switching detection and normalization tools
   - Cultural reference databases for contextual explanations
   - Configurable explanation interfaces designed with community input

3. **Evaluation Benchmarks**: New benchmarks for assessing explanation quality in low-resource settings, incorporating both technical metrics and user-centered evaluations.

4. **Implementation Case Studies**: Detailed documentation of implementation and results across the four studied languages, providing templates for expansion to other low-resource languages.

### 3.2 Community and Social Impact

1. **Increased Transparency and Trust**: By making language model behavior more transparent, we expect to build greater trust in these technologies among low-resource language communities, potentially increasing adoption and benefit.

2. **Community Empowerment**: The project will equip language communities with tools to audit, critique, and improve language models, shifting power dynamics in technology development toward greater inclusivity.

3. **Cultural Preservation**: By emphasizing the importance of cultural context in explanations, the project contributes to preserving cultural knowledge and nuance in digital spaces.

4. **Educational Resources**: The explanatory interfaces will serve as educational tools, helping users understand both how language models work and how computational approaches interpret their languages.

### 3.3 Research Community Impact

1. **Methodological Innovation**: The interdisciplinary approach combining technical innovation with community participation establishes a new paradigm for responsible AI development, particularly for underserved populations.

2. **Ethics and Fairness**: By addressing the equity gap in language technology access, this work contributes to broader discussions on fairness and justice in AI systems.

3. **Standards Development**: The evaluation metrics and cultural alignment assessments developed in this project can inform industry and academic standards for responsible development of language technologies.

4. **Future Research Directions**: This work will open new research avenues in culturally-informed AI, community-centered design, and responsible innovation for marginalized communities.

### 3.4 Long-term Vision

In the longer term, this research aims to contribute to a fundamental shift in how language technologies are developed for marginalized communities—moving from the current model where communities are passive recipients of technologies developed elsewhere to one where they are active participants in technology development and assessment. The tools and methodologies developed here will enable:

1. **Community-Led Model Development**: Communities can use explanation tools to identify areas where models need improvement, guiding future data collection and model refinement.

2. **Responsible Deployment Guidelines**: The insights gained will inform best practices for deploying language technologies in diverse cultural contexts.

3. **Educational Initiatives**: The frameworks developed can support educational programs that build local capacity for language technology development.

4. **Policy Recommendations**: Evidence gathered on the impact of interpretable models can inform policy decisions about language technology deployment and regulation.

Through these outcomes, this research addresses critical gaps in socially responsible AI by making language models both more accessible and more accountable to underrepresented linguistic communities, advancing the dual goals of transparency and inclusivity in global NLP applications.