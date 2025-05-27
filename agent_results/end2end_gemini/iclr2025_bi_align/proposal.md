**1. Title: Fostering Human Cognitive Alignment in Complex AI Partnerships: An Adaptive AI Tutoring Framework**

**2. Introduction**

The rapid proliferation of sophisticated Artificial Intelligence (AI) systems across critical domains, such as scientific discovery, medical diagnosis, and autonomous decision-making, has brought the challenge of human-AI alignment to the forefront. As these systems become more autonomous and their internal workings more opaque, ensuring they operate in a manner consistent with human values, goals, and expectations is paramount (Russell et al., 2015). Traditionally, AI alignment has predominantly focused on a unidirectional approach: shaping AI systems to conform to human specifications (the "Aligning AI with Humans" perspective). However, as highlighted by Shen et al. (2024) in their systematic review, this perspective is insufficient to address the dynamic and evolving nature of human-AI interactions. They propose a "Bidirectional Human-AI Alignment" framework, emphasizing the equal importance of "Aligning Humans with AI"—empowering humans to understand, critically evaluate, and effectively collaborate with AI systems.

This research addresses the "Aligning Humans with AI" dimension, motivated by the observation that the increasing complexity of AI systems often outpaces human users' ability to form accurate mental models of their reasoning processes, capabilities, and limitations. This cognitive misalignment can lead to misuse, disuse, or over-reliance on AI, undermining the potential benefits of human-AI collaboration and potentially leading to errors with significant consequences (Li et al., 2025). Current AI interfaces often lack robust mechanisms to proactively support users in developing this crucial understanding, thereby hindering the preservation of human agency and the realization of truly synergistic partnerships.

The central idea of this research is to develop and evaluate an "AI Cognitive Tutor" module designed to be integrated within complex AI systems. This tutor will adaptively intervene when it detects potential user misunderstanding or misalignment with the AI's operations. Such interventions could include tailored explanations of the AI’s logic, simplified analogies, visualizations of data assumptions, or clarifications regarding the AI’s uncertainty. This concept draws inspiration from adaptive tutoring systems in education (Baradari et al., 2025; Dong et al., 2023) and the principles of Reciprocal Human Machine Learning (Te'eni et al., 2023), aiming to create a learning loop where humans improve their understanding of AI, and their interactions can, in turn, inform refinements in AI communication or even the tutor itself.

**Research Objectives:**
The primary objectives of this research are:
1.  To identify and characterize common triggers indicating user cognitive misalignment or misunderstanding when interacting with complex AI systems (e.g., in a medical diagnosis support tool).
2.  To design and develop a modular AI Cognitive Tutor capable of delivering adaptive, context-sensitive educational interventions based on detected triggers.
3.  To curate and implement a library of diverse tutoring strategies (e.g., simplified explanations, analogies, interactive queries, counterfactuals) tailored to bridge gaps in user understanding of AI reasoning, data use, and uncertainty.
4.  To empirically evaluate the effectiveness of the AI Cognitive Tutor in enhancing users' mental model accuracy of the AI, improving human-AI task performance, fostering appropriate reliance, and increasing subjective understanding and confidence.
5.  To explore mechanisms through which user interaction and feedback with the tutor can inform the refinement of the tutor itself and potentially the explanatory capabilities of the primary AI system, thus contributing to the bidirectional alignment loop.

**Significance:**
This research holds significant potential for advancing the field of human-AI interaction and alignment. By focusing on the human aspect of the alignment equation, it directly contributes to preserving human agency in an increasingly AI-driven world. An effective AI Cognitive Tutor can empower users to move from being passive consumers of AI outputs to active, critical collaborators, leading to more effective, safe, and trustworthy human-AI partnerships. This work addresses key challenges identified in the literature, such as real-time cognitive state assessment and the complexity of bidirectional alignment (Shen et al., 2024). The findings will offer practical design guidelines for creating AI systems that are not only intelligent but also intelligible and teachable to their human partners, ultimately fostering a more inclusive and beneficial human-AI ecosystem. This aligns well with the workshop's goal of broadening the understanding of AI alignment and fostering interdisciplinary collaboration.

**3. Methodology**

This research will be conducted in three primary phases: (1) Identification of User Misunderstanding Triggers and Tutoring Needs, (2) Design and Development of the Adaptive AI Cognitive Tutor, and (3) Experimental Validation of the Tutor. We will focus on the context of a complex AI system, specifically a hypothetical AI-powered medical diagnostic support tool that analyzes medical images and patient data to suggest potential diagnoses and treatment options. This domain is chosen due to its complexity, critical nature, and the significant impact of human (clinician) understanding on decision-making.

**Phase 1: Identification of User Misunderstanding Triggers and Tutoring Needs**

*   **Objective:** To understand how clinicians interact with a complex AI diagnostic tool, identify common points of confusion or misinterpretation, and determine situations where a cognitive tutor could be beneficial.
*   **Data Collection & Participants:**
    *   We will recruit 15-20 medical professionals (e.g., radiologists, oncologists) with varying levels of experience with AI tools.
    *   Participants will engage in simulated diagnostic tasks using a wizard-of-oz prototype of an AI diagnostic tool. The AI's behavior (including deliberate introduction of ambiguous or complex scenarios) will be partially controlled by researchers to elicit a range of user responses.
    *   Data will be collected through:
        *   **Think-aloud protocols:** Participants will verbalize their thought processes while interacting with the system.
        *   **Interaction logs:** Keystrokes, clicks, time spent on different information panels, queries made.
        *   **Eye-tracking data:** To infer attention and information processing patterns.
        *   **Post-task interviews:** To probe specific instances of confusion or unexpected actions.
*   **Analysis:**
    *   Qualitative analysis (thematic analysis) of think-aloud protocols and interview data to identify recurring themes of misunderstanding related to AI reasoning, data interpretation, confidence scores, and limitations.
    *   Quantitative analysis of interaction logs and eye-tracking data to identify behavioral patterns correlated with self-reported confusion or observable errors. For example, repeated requests for the same information, prolonged hesitation before action, or direct expressions of uncertainty (e.g., "I don't understand why it suggested X").
    *   These analyses will help define a set of **Misunderstanding Triggers ($T$)**. A trigger $T_i$ could be a specific user action sequence, a query pattern, a physiological signal (if available and ethically appropriate, though initially we focus on behavioral cues), or a direct input indicating confusion.

**Phase 2: Design and Development of the Adaptive AI Cognitive Tutor**

*   **Objective:** To design and implement the AI Cognitive Tutor module, including its trigger detection mechanism, library of tutoring strategies, and adaptivity logic.
*   **Tutor Architecture:**
    1.  **Monitoring Component:** Continuously observes user interaction $B_u = \{b_1, b_2, ..., b_n\}$ with the main AI system (e.g., GUI interactions, queries).
    2.  **Trigger Detection Module:** Identifies occurrences of predefined misunderstanding triggers $T_i$. The probability of a misunderstanding $M$ given user behavior $B_u$ can be modeled, $P(M|B_u)$. A trigger is activated if $P(M|B_u) > \theta_M$, where $\theta_M$ is a dynamically adjustable threshold. Initially, this will be rule-based, derived from Phase 1 findings, with potential for future ML-based classification.
    3.  **Tutoring Strategy Library ($R_t$):** A repository of various intervention types, $R_t = \{strat_1, strat_2, ..., strat_k\}$. Examples include:
        *   **Simplified Explanations:** Rephrasing complex AI outputs or model justifications (e.g., LIME/SHAP explanations) in simpler terms.
        *   **Analogies:** Relating AI concepts to familiar medical diagnostic processes.
        *   **Interactive Q&A:** Allowing users to ask clarifying questions about a specific AI suggestion, with the tutor providing targeted answers sourced from a knowledge base about the AI model (inspired by KG-RAG, Dong et al., 2023).
        *   **Visualizations:** Graphical representations of data distributions the AI considered, or how its confidence level was derived.
        *   **Micro-learning Snippets:** Short, focused content (e.g., a brief explanation of "model uncertainty" or "feature weighting" in the context of the current AI output).
        *   **Contrastive Explanations:** Explaining why the AI chose option A instead of a plausible option B.
    4.  **Adaptivity Logic ($\pi_{tutor}$):** A policy that selects an appropriate tutoring strategy $strat_j \in R_t$ based on the detected trigger $T_i$, user's interaction history $H_u$, and potentially a simple user model (e.g., novice/expert, learning progress).
        $$ strat_{selected} = \pi_{tutor}(T_i, H_u, \text{UserProfile}) $$
        Initially, this will be a heuristic-based selection (e.g., if trigger is "repeatedly ignoring high-uncertainty warnings," strategy might be a micro-learning snippet on AI uncertainty). Future work could explore reinforcement learning to optimize this policy based on tutor effectiveness.
    5.  **Feedback Mechanism:** The tutor will solicit feedback on its interventions (e.g., "Was this explanation helpful? Yes/No/Partially"). This feedback will be used for:
        *   Iterative refinement of the tutoring strategies.
        *   Logging effectiveness, potentially for adapting $\pi_{tutor}$ over time.
        *   Potentially, this feedback, aggregated over many users, could inform the primary AI's own explanation generation methods if certain concepts are consistently difficult for users to grasp despite tutoring. This forms a part of the bidirectional learning loop.

*   **Implementation:** The tutor will be developed as a software module, potentially a sidebar or a pop-up interface, integrated with the prototype AI diagnostic tool. Technologies will include Python for backend logic, and web technologies (HTML, CSS, JavaScript) for the user interface components.

**Phase 3: Experimental Validation**

*   **Objective:** To empirically evaluate the AI Cognitive Tutor's impact on user understanding, human-AI task performance, trust calibration, and overall user experience.
*   **Research Design:** A between-subjects experimental design will be employed.
    *   **Participants:** 60 medical professionals (or senior medical students as a proxy if access to professionals is limited, ensuring clear reporting of this choice) who were not involved in Phase 1. Participants will be randomly assigned to one of two groups (30 per group).
    *   **Independent Variable:** Presence of the AI Cognitive Tutor.
        *   **Control Group (CG):** Participants use the AI diagnostic tool without the Cognitive Tutor. Standard explainability features of the AI (if any) will be active.
        *   **Treatment Group (TG):** Participants use the AI diagnostic tool integrated with the active AI Cognitive Tutor.
    *   **Tasks:** Participants in both groups will perform a series of standardized medical diagnostic tasks using the AI tool. Tasks will be designed to cover scenarios where AI reasoning might be complex, uncertain, or counter-intuitive, based on insights from Phase 1.
    *   **Procedure:**
        1.  Pre-experiment questionnaire: Demographics, prior experience with AI, baseline knowledge test about AI concepts relevant to the tool.
        2.  Training session: Introduction to the AI diagnostic tool (and the tutor for TG).
        3.  Experimental tasks: Participants perform diagnostic tasks. Interactions, tutor activations (for TG), and task outcomes are logged.
        4.  Post-task questionnaires:
            *   Mental Model Assessment: Using methods like knowledge mapping or scenario-based questions to assess their understanding of how the AI works (e.g., "If patient data X changes, how would the AI's prediction likely change and why?").
            *   Subjective Understanding: Self-reported understanding of AI suggestions.
            *   Cognitive Load: NASA-TLX or Subjective Workload Assessment Technique (SWAT).
            *   Trust in AI: Using established trust scales (e.g., Madsen & Gregor, 2000).
            *   Perceived Usefulness and Usability of the Tutor (for TG): Using SUS or custom scales.
        5.  Semi-structured interviews: With a subset of participants from both groups to gather qualitative insights.
*   **Evaluation Metrics:**
    *   **Primary Metrics:**
        *   **Mental Model Accuracy Score:** Quantified score based on post-task assessment. Hypothesis: $Score_{TG} > Score_{CG}$.
        *   **Diagnostic Task Performance:** Accuracy of final diagnoses, agreement with AI when AI is correct, appropriate disagreement when AI is incorrect, time to diagnosis. Hypothesis: $Performance_{TG}$ will show more appropriate reliance and potentially better accuracy in complex cases.
    *   **Secondary Metrics:**
        *   **User-AI Misalignment Incidents:** Frequency of errors attributable to misunderstanding the AI (e.g., misinterpreting confidence scores). Hypothesis: $Incidents_{TG} < Incidents_{CG}$.
        *   **Cognitive Load:** Hypothesis: $Load_{TG}$ might be initially higher due to engagement with tutor but could lead to lower load in later tasks due to better understanding, or overall optimized.
        *   **Trust Calibration:** Correlation between user trust and AI reliability on a per-case basis. Hypothesis: TG will show better trust calibration.
        *   **Tutor Effectiveness (for TG):** Frequency of tutor activation, user ratings of tutor interventions, correlation between tutor use and improved performance/understanding on subsequent similar tasks.
*   **Data Analysis:**
    *   Quantitative data will be analyzed using appropriate statistical tests (e.g., t-tests, ANOVA, regression models) to compare outcomes between the Control and Treatment groups.
    *   Qualitative data from interviews and open-ended responses will be analyzed using thematic analysis to provide richer context and explanations for quantitative findings.
    *   Interaction logs with the tutor will be analyzed to understand which strategies are most frequently used and rated as helpful.

**Ethical Considerations:**
All studies involving human participants will undergo ethical review and approval by an Institutional Review Board (IRB). Participants will provide informed consent, and their data will be anonymized and handled confidentially. The AI diagnostic tool is a prototype for research purposes only and will not be used for actual clinical decision-making.

**4. Expected Outcomes & Impact**

**Expected Outcomes:**
This research is expected to yield several key outcomes:
1.  **A Catalog of User Misunderstanding Triggers:** A systematically identified set of behavioral and contextual cues indicating when users (specifically clinicians in our testbed) struggle to comprehend or appropriately align with a complex AI system. This will be valuable for designing proactive support mechanisms.
2.  **A Novel AI Cognitive Tutor Framework:** A functional prototype of an adaptive AI Cognitive Tutor module, including its architecture, a library of diverse tutoring strategies, and an initial adaptivity logic. This framework will demonstrate the feasibility of embedding such tutors within complex AI interfaces.
3.  **Empirical Evidence of Effectiveness:** Quantitative and qualitative evidence regarding the AI Cognitive Tutor's impact on:
    *   **Improved User Comprehension:** Users in the treatment group are expected to develop significantly more accurate and nuanced mental models of the AI's reasoning, capabilities, data dependencies, and uncertainty representation compared to the control group.
    *   **Enhanced Human-AI Collaboration:** We anticipate that users interacting with the tutor will exhibit more effective collaboration patterns, such as more appropriate reliance on AI suggestions (i.e., higher acceptance when AI is correct, more critical questioning when AI is potentially flawed or uncertain), leading to improved joint task performance in complex diagnostic scenarios.
    *   **Reduced Misalignment-Driven Errors:** A decrease in errors stemming from misinterpretation of AI outputs or a flawed understanding of its operational logic.
4.  **Design Principles for Human-Centric AI Explanations:** Insights into which tutoring strategies are most effective for different types of AI complexities and user knowledge gaps, contributing to design guidelines for more human-centric AI explanations and support systems.
5.  **Refined Understanding of Bidirectional Alignment Dynamics:** Preliminary insights into how feedback from human interactions with the tutor can be leveraged to improve the tutor itself and potentially the primary AI's communication strategies, demonstrating a practical step towards realizing bidirectional alignment.

**Impact:**
The successful completion of this research will have a multifaceted impact:
1.  **Advancing Human-AI Alignment Research:** This work will make a direct contribution to the "Aligning Humans with AI" aspect of the bidirectional human-AI alignment paradigm, as advocated by Shen et al. (2024) and the workshop's theme. It will provide a concrete methodology and empirical findings for an under-explored yet critical dimension of alignment.
2.  **Promoting Human Agency and Trust:** By empowering users with better understanding, the AI Cognitive Tutor can help preserve human agency in decision-making processes involving AI. This, in turn, can foster more calibrated trust, where users neither blindly accept nor unfairly dismiss AI contributions.
3.  **Enhancing Safety and Efficacy of AI in Critical Domains:** In fields like medicine, where misunderstanding AI can have severe consequences, improving clinician comprehension is crucial for patient safety and optimal diagnostic outcomes. The principles developed here can be extended to other high-stakes domains (e.g., finance, autonomous driving).
4.  **Informing the Design of Future AI Systems:** The findings will offer actionable design principles for AI developers and HCI researchers to create systems that are not just powerful but also more transparent, understandable, and supportive of human cognitive needs. This aligns with the need for "steerability" and "interpretability" in AI deployment.
5.  **Fostering Interdisciplinary Collaboration:** This research inherently bridges AI, HCI, cognitive science, and domain-specific expertise (e.g., medicine), promoting the interdisciplinary approach called for by the workshop.
6.  **Societal Benefit:** Ultimately, by enabling humans to better understand and collaborate with increasingly complex AI, this research can contribute to ensuring that AI technologies are used responsibly and effectively for the benefit of society, mitigating risks associated with cognitive misalignment between humans and AI. It will also highlight the importance of continuous learning for humans in an AI-augmented world, akin to the reciprocal learning discussed by Te'eni et al. (2023).

By addressing the critical need to enhance human understanding of complex AI systems, this research aims to pave the way for more productive, safe, and meaningful human-AI partnerships, contributing significantly to the evolving landscape of bidirectional human-AI alignment.