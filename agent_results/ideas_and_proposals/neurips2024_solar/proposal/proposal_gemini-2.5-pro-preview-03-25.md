## 1. Title: Interpretable and Community-Centered Language Models for Low-Resource Languages: Enhancing Transparency, Equity, and Accountability

## 2. Introduction

**Background:** Large Language Models (LMs) have demonstrated remarkable capabilities across a wide range of natural language tasks, transforming fields from information retrieval to creative writing. However, their rapid development and deployment raise significant societal concerns, including the potential for perpetuating bias, excluding marginalized communities, ensuring security and privacy, and maintaining safety and alignment [13, 30, 2, 53, 51]. The Socially Responsible Language Modelling Research (SoLaR) workshop highlights the critical need to address these risks proactively, embedding principles of fairness, equity, accountability, transparency, and safety into the core of LM research and development.

A particularly acute set of challenges arises in the context of low-resource languages (LRLs). These languages, often spoken by marginalized communities, typically lack the vast datasets used to train state-of-the-art models [4]. While recent efforts have shown promise in developing effective LMs for LRLs with limited data (e.g., InkubaLM [1], Glot500 [2]), these models inherit and potentially amplify the opacity characteristic of deep learning systems. The combination of data scarcity, which can lead to brittle performance and unpredictable failures, and the inherent "black box" nature of LMs creates a significant barrier to trust and adoption. For speakers of LRLs, opaque models risk delivering unreliable or biased outputs without recourse or understanding, potentially exacerbating existing digital divides and reinforcing societal inequities [5]. Without mechanisms for transparency and interpretation, it becomes difficult to audit these models for fairness, identify failure modes, or empower communities to participate in their refinement and governance.

**Problem Statement:** The central problem this research addresses is the critical lack of transparency and interpretability in LMs specifically designed for or applied to low-resource languages. This opacity hinders the ability of both researchers and end-user communities to understand model behavior, diagnose biases, build trust, and ensure equitable deployment. Existing interpretability techniques (e.g., LIME, SHAP), largely developed for high-resource languages like English, often fail to adequately capture the unique linguistic characteristics of LRLs, such as complex morphology or prevalent code-switching [8, 9]. Furthermore, standard explanation interfaces may not align with the cultural or communicative norms of diverse linguistic communities, limiting their practical utility [6]. This confluence of technical and socio-cultural challenges necessitates a tailored approach to interpretability for LRLs, one that is both methodologically sound and community-grounded.

**Research Objectives:** This research aims to develop and evaluate a novel interpretability framework specifically designed for LMs operating in low-resource language contexts. Our primary objectives are:

1.  **Adapt Existing Interpretability Methods:** To modify and extend established local explanation techniques (e.g., SHAP, LIME) to effectively handle the unique linguistic features prevalent in LRLs, including complex morphology and code-switching patterns.
2.  **Co-design Culturally Grounded Interfaces:** To collaborate directly with native speakers of selected LRLs to design and develop explanation interfaces that are intuitive, understandable, and aligned with their cultural and communicative preferences.
3.  **Develop Comprehensive Evaluation Metrics:** To establish a robust evaluation methodology that assesses both the technical fidelity and robustness of the adapted interpretability methods and the user-perceived trust, usefulness, and understandability of the co-designed explanations.
4.  **Produce Open-Source Tools and Guidelines:** To release the developed interpretability methods and interface components as open-source software, accompanied by best-practice guidelines for implementing culturally grounded explainability in LRL settings, fostering wider adoption and research [10].

**Significance:** This research directly addresses multiple core themes of the SoLaR workshop. By focusing on interpretability for LRLs, we aim to enhance **transparency** and **accountability** in models serving underrepresented communities. Adapting methods to linguistic diversity and co-designing interfaces with native speakers promotes **fairness** and **equity**, mitigating the risk of exclusion [2, 53]. Providing tools for communities to audit and understand models fosters **safety** by enabling the identification of potentially harmful biases or failure modes [41, 15]. Ultimately, this work seeks to empower marginalized linguistic communities, enabling them to engage more meaningfully with AI technologies, build trust in useful applications (e.g., LMs for social good [9, 31]), and contribute to the development of more socially responsible language technology worldwide. This research will provide valuable insights and practical tools for researchers, developers, and policymakers working towards inclusive and ethical AI.

## 3. Methodology

**Research Design Overview:** We propose a mixed-methods research design that integrates technical development with human-centered design and evaluation. The methodology comprises four main phases: (1) Data Collection and Preparation, (2) Adaptation of Interpretability Techniques, (3) Community-Driven Interface Co-Design, and (4) Experimental Design and Evaluation. This phased approach allows for rigorous technical development informed by continuous community engagement and feedback.

**Phase 1: Data Collection and Preparation**

*   **Language Selection:** We will initially focus on 2-3 low-resource languages exhibiting diverse linguistic characteristics relevant to interpretability challenges. Potential candidates could be drawn from initiatives like InkubaLM (e.g., Yoruba, Swahili [1]) or Glot500 [2], considering factors like data availability (even if limited), presence of complex morphology or code-switching, and the feasibility of collaborating with speaker communities.
*   **Corpus Curation:** We will leverage existing LRL corpora (e.g., datasets associated with [1, 2]) and potentially augment them through web scraping or partnerships with linguistic organizations, strictly adhering to ethical data sourcing practices. Data will be cleaned and preprocessed.
*   **Annotation (if necessary):** For specific evaluations (e.g., identifying code-switching points, morphological segmentation), we may perform limited manual annotation in collaboration with native speakers or linguistic experts, ensuring fair compensation. Tools like GlotLID [3] might assist in verifying language labels within potentially noisy web-scraped data.
*   **Model Training/Selection:** We will utilize existing pre-trained models for LRLs (e.g., InkubaLM [1]) or fine-tune larger multilingual models (e.g., Glot500-base [2]) on our curated corpora for specific downstream tasks (e.g., text classification, masked language modeling) relevant to interpretability evaluation. This ensures our interpretability methods are tested on realistic LRL models.

**Phase 2: Adapting Interpretability Techniques**

This phase focuses on adapting local, model-agnostic explanation methods like LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations), chosen for their widespread use and flexibility. The core idea of these methods is to approximate the local behavior of a complex model $f$ with a simpler, interpretable model $g$. For an input instance $x$, LIME generates explanations by sampling perturbations $z'$ in the vicinity of $x$ in a simplified feature space, weighting them by proximity, and fitting $g$ to $f(h_x(z'))$, where $h_x$ maps simplified features back to the original space. SHAP uses Shapley values from cooperative game theory to attribute the prediction outcome fairly among input features.

*   **Adaptation for Morphology [8]:** Many LRLs are morphologically rich, where single words carry significant grammatical information. Standard token-based explanations are insufficient.
    *   *Approach:* We will adapt LIME/SHAP to operate at the morpheme level. Input text will first be segmented into morphemes using language-specific morphological analyzers (or unsupervised methods if analyzers are unavailable). Perturbations (for LIME) or feature coalitions (for SHAP) will be defined over morphemes instead of tokens.
    *   *Formalization (Conceptual):* Let $x = (m_1, m_2, ..., m_k)$ be the morpheme sequence. The explanation model $g$ will assign importance scores $\phi_i$ to each morpheme $m_i$. For SHAP, the contribution of a morpheme $m_i$ would be its Shapley value:
        $$ \phi_i(f, x) = \sum_{S \subseteq M \setminus \{m_i\}} \frac{|S|! (k - |S| - 1)!}{k!} [f_S(S \cup \{m_i\}) - f_S(S)] $$
        where $M$ is the set of all morphemes, $S$ is a subset of morphemes, and $f_S(S)$ is the model prediction using only morphemes in $S$. Implementing this efficiently requires approximations (e.g., KernelSHAP) adapted to morpheme-level features.
*   **Adaptation for Code-Switching [9]:** Code-switching (mixing languages) is common in LRL communities. Explanations need to account for this phenomenon.
    *   *Approach:* We will integrate language identification (potentially using models like GlotLID [3] at the segment level) into the explanation pipeline. Explanations can then be generated conditionally based on the language of the segment or token being explained. Perturbations for LIME could involve substituting words/phrases with equivalents in the *other* language present in the context or replacing them with language-specific neutral placeholders. Feature importance for SHAP could be computed considering the language context.
    *   *Implementation:* Develop wrappers around LIME/SHAP that preprocess input, detect code-switching points, and guide the perturbation or feature attribution process based on language tags associated with tokens or segments.

**Phase 3: Community-Driven Interface Co-Design**

This phase directly addresses the need for explanations to be understandable and useful to the target community [6].

*   **Participant Recruitment:** We will recruit native speakers from the selected LRL communities through partnerships with cultural organizations, universities, or online community groups. Participants will represent diverse backgrounds (age, gender, technical literacy). Ethical protocols, including informed consent and fair compensation, will be strictly followed.
*   **Co-Design Workshops:** We will conduct a series of participatory design workshops.
    *   *Needs Assessment:* Initial sessions will explore participants' understanding of AI/LMs, their information needs regarding model behavior, and culturally relevant ways of explaining decisions or reasoning.
    *   *Prototyping & Iteration:* Based on initial feedback, we will develop low-fidelity prototypes of explanation interfaces (e.g., sketches, wireframes). These prototypes will visualize the adapted LIME/SHAP explanations (e.g., highlighting important morphemes, indicating language switches influencing predictions) in different formats (textual summaries, interactive visualizations).
    *   *Feedback Cycles:* Prototypes will be iteratively refined based on participant feedback gathered through think-aloud protocols, focus groups, and usability testing. We aim to identify interface designs that are perceived as intuitive, trustworthy, and useful.

**Phase 4: Experimental Design and Evaluation**

We will employ a comprehensive evaluation strategy targeting both technical quality and user-centric effectiveness [7].

*   **Technical Evaluation:**
    *   *Metrics:*
        *   *Faithfulness:* How accurately do the explanations reflect the model's internal reasoning? Measured using local fidelity scores (correlation between explanation model $g$ and LM $f$ on local perturbations) and metrics based on feature removal/addition (e.g., evaluating prediction change when removing top-k features identified by the explanation).
        *   *Robustness:* How stable are explanations under minor input perturbations? Measured by comparing explanations (e.g., using Jaccard index or cosine similarity on feature importance vectors) for slightly modified inputs.
    *   *Experiments:* Compare the faithfulness and robustness of our adapted LIME/SHAP methods (morpheme-aware, code-switch-aware) against baseline implementations (standard token-based LIME/SHAP) on benchmark tasks (e.g., sentiment analysis, NLI) using the LRL models from Phase 1. Ablation studies will assess the contribution of each adaptation (morphology, code-switching).
*   **User-Centric Evaluation:**
    *   *Metrics:*
        *   *Perceived Trust:* Measured using validated psychometric scales (e.g., Trust in Automation scales) adapted to the LM context [7].
        *   *Understandability:* Assessed through participant ratings and comprehension questions about specific explanations.
        *   *Usefulness:* Evaluated based on participants' ability to use explanations for specific tasks (see below) and subjective ratings.
        *   *Satisfaction:* Overall user satisfaction with the explanation interface.
    *   *Experiments:* Conduct controlled user studies with native speakers recruited in Phase 3.
        *   *Task-Based Evaluation [7]:* Participants will perform tasks using the LM system equipped with different explanation interfaces (e.g., baseline vs. co-designed interface showing adapted explanations). Tasks could include:
            *   Debugging biased predictions: Identifying why the model produced a potentially unfair output.
            *   Comparative understanding: Determining why the model gave different predictions for similar inputs.
            *   Confidence assessment: Using explanations to judge the reliability of a model's prediction for a critical decision.
        *   *Surveys and Interviews:* Collect quantitative data via questionnaires (using the metrics above) and qualitative insights via semi-structured interviews to understand the nuances of user experience and perceived value.
        *   *Comparison:* Compare user performance and subjective ratings between (a) models with no explanations, (b) models with baseline explanations/interfaces, and (c) models with our adapted methods and co-designed interfaces.

**Ethical Considerations:** Throughout the research, we will adhere to strict ethical guidelines. This includes obtaining Institutional Review Board (IRB) approval, ensuring informed consent from all participants (explaining data usage, anonymization, and withdrawal rights), providing fair compensation for community members' time and expertise, ensuring data privacy and security, and committing to transparently sharing anonymized results and open-source artifacts back with the participating communities and the wider research field.

## 4. Expected Outcomes & Impact

**Expected Outcomes:** This research is expected to produce several concrete outputs:

1.  **Adapted Interpretability Algorithms:** Novel adaptations of LIME and SHAP specifically designed to handle morphological complexity and code-switching in low-resource languages.
2.  **Culturally Grounded Explanation Interfaces:** Prototypes and design principles for user interfaces that present model explanations in ways that are intuitive and meaningful to specific LRL communities, developed through a co-design process.
3.  **Comprehensive Evaluation Framework:** A validated set of metrics and protocols for evaluating both the technical quality (faithfulness, robustness) and user-centric effectiveness (trust, understandability, usefulness) of interpretability methods in LRL contexts [7].
4.  **Open-Source Toolkit:** A publicly available software library implementing the adapted interpretability methods and example interface components, facilitating adoption and further research by others [10].
5.  **Best-Practice Guidelines:** Recommendations for researchers and practitioners on how to develop and evaluate interpretable LMs for LRLs in a socially responsible and community-centered manner [5, 6].
6.  **Empirical Findings:** Research publications detailing the performance of the adapted methods, the results of the user studies, and insights into the challenges and opportunities of explainable AI for LRLs [4, 5].

**Impact:** The proposed research holds significant potential for positive impact aligned with the goals of the SoLaR workshop:

*   **Enhanced Transparency and Accountability:** By providing methods to understand *why* LRL models make certain predictions, our work will increase transparency, enabling better auditing for bias and performance issues, thus fostering greater accountability [41, 15].
*   **Increased Equity and Reduced Exclusion:** Tailoring interpretability to the linguistic realities of LRLs [8, 9] and co-designing interfaces with communities [6] directly addresses the risk of excluding non-English speakers from the benefits of understandable AI. This promotes linguistic diversity and digital equity.
*   **Fostering Trust and Adoption:** Understandable models are more likely to be trusted and appropriately adopted by end-users [7]. This is crucial for the responsible deployment of LMs in sensitive LRL application areas, such as education, healthcare information dissemination, or access to services [9, 31].
*   **Empowering Communities:** Providing tools and guidelines for interpretation empowers LRL communities to participate more actively in the AI lifecycle. They can better assess whether models align with their values, identify harmful outputs, and advocate for improvements, shifting power towards those most affected by the technology.
*   **Advancing Socially Responsible AI Research:** This project contributes directly to the body of knowledge on socially responsible LM development. It provides concrete methodologies for integrating interpretability and community engagement into the workflow for LRLs, offering a model that can inform research in other under-resourced or vulnerable contexts. The focus on LRLs addresses a critical gap often overlooked in mainstream AI ethics discussions.
*   **Informing Policy and Practice:** The findings and guidelines can inform policymakers and organizations deploying LMs in multilingual contexts about the importance and practicalities of implementing transparent and equitable systems.

In conclusion, this research tackles the dual challenges of performance and opacity in low-resource language models by developing technically sound and community-grounded interpretability solutions. By prioritizing transparency, equity, and collaboration, we aim to contribute significantly to the development of more socially responsible and trustworthy language technologies for all linguistic communities.