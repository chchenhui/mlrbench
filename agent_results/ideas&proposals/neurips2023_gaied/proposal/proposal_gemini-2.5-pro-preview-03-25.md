Okay, here is a research proposal based on the provided task description, research idea, and literature review.

---

**1. Title:** **Enhancing Collaborative Inquiry in Education: Developing and Evaluating a Generative AI Socratic Learning Partner**

**2. Introduction**

**2.1 Background**
The landscape of education is undergoing a rapid transformation driven by advances in Artificial Intelligence (AI), particularly Generative AI (GenAI) and Large Language Models (LLMs) like ChatGPT (OpenAI, 2023). These technologies offer unprecedented opportunities to reshape educational paradigms, falling under the GAI→ED thrust identified by the GAIED workshop. Current applications often focus on AI as informational tutors or automated assistants for tasks like grading or content summarization (Dan et al., 2023; Jabbour et al., 2025). While valuable, these applications often mirror traditional didactic teaching methods, primarily delivering information or providing corrective feedback.

Simultaneously, established pedagogical principles emphasize the importance of active and collaborative learning for developing deeper understanding and critical thinking skills (Vygotsky, 1978; Chi & Wylie, 2014). Peer interaction, when effective, encourages students to articulate their reasoning, confront misconceptions, consider alternative perspectives, and engage in reflective thinking. The Socratic method, characterized by probing questions and guided inquiry, is a powerful technique for fostering such cognitive processes (Paul & Elder, 2007). However, facilitating high-quality peer collaboration or individualized Socratic dialogue at scale remains a significant challenge in many educational settings due to resource constraints and variability in student interaction skills.

Existing AI tutoring systems, even those employing Socratic elements like SocratiQ (Jabbour et al., 2025) and EduChat (Dan et al., 2023), or those focusing on Socratic dialogue generation like SPL (Zhang et al., 2024), often retain a tutor-centric dynamic. They guide students towards correct answers or predefined learning objectives but may fall short of simulating the emergent, sometimes challenging, and often reciprocal nature of peer-to-peer inquiry. There is a gap in leveraging GenAI to specifically emulate the role of a *learning partner* – an entity that collaborates, questions, and challenges *alongside* the student, fostering the cognitive benefits of collaborative inquiry without necessarily possessing or providing definitive answers.

**2.2 Problem Statement**
While GenAI offers potent tools for education, current implementations often fail to fully harness its potential for fostering deep critical thinking and metacognitive skills typically cultivated through effective Socratic dialogue and collaborative peer inquiry. Standard AI tutors tend towards information delivery or corrective feedback, whereas human peer interactions, though beneficial, can be inconsistent, unavailable, or lack pedagogical structure. There is a need for AI systems that can act as consistent, scalable Socratic *partners*, stimulating students' reasoning, self-explanation, and reflective processes in a manner akin to productive peer collaboration, thereby addressing a key opportunity within the GAI→ED framework.

**2.3 Proposed Research: The Socratic Learning Partner (SLP)**
This research proposes the development and evaluation of a novel LLM-based agent: the Socratic Learning Partner (SLP). Unlike conventional AI tutors that provide answers or direct guidance, the SLP is designed to simulate a peer engaged in collaborative inquiry. Its primary function is not to teach facts but to stimulate the student's own thinking processes through carefully crafted Socratic interactions. This will be achieved through:
*   **Specialized Prompting Strategies:** Designing prompts that instruct the LLM to adopt a peer persona focused on inquiry, asking open-ended and probing questions, encouraging articulation of reasoning, constructively challenging assumptions, and requesting clarification, while actively avoiding providing direct solutions or declarative statements.
*   **Fine-tuning on Curated Dialogues:** Fine-tuning a base LLM on a dataset of high-quality Socratic dialogues and examples of effective peer learning interactions, specifically emphasizing questioning techniques, turn-taking patterns of inquiry, and scaffolding without giving away answers.

The SLP aims to serve as a readily available, consistent partner that can engage students in the kind of deep cognitive processing associated with robust collaborative learning.

**2.4 Research Objectives**
The primary objectives of this research are:
1.  **Design and Implement the SLP:** Develop the core architecture of the SLP, including the selection of a suitable base LLM and the implementation of specialized prompting frameworks and dialogue management strategies.
2.  **Curate/Generate Fine-tuning Data:** Collect, filter, and potentially generate a dataset of dialogues embodying Socratic inquiry and effective peer collaboration principles suitable for fine-tuning the SLP.
3.  **Develop Fine-tuning Methodology:** Fine-tune the selected LLM using the curated dataset to enhance its ability to perform Socratic questioning and emulate peer inquiry dynamics.
4.  **Evaluate SLP Effectiveness:** Conduct rigorous user studies to compare the impact of interacting with the SLP on students' learning processes (e.g., depth of explanation, self-correction, reflective thinking) against interactions with a conventional answer-focused AI tutor and a control group.
5.  **Analyze Interaction Patterns:** Qualitatively and quantitatively analyze the dialogue logs to understand how students interact with the SLP and identify the specific conversational features that correlate with desired learning outcomes.

**2.5 Significance**
This research holds significant potential contributions:
*   **Pedagogical Innovation (GAI→ED):** It explores a novel role for GenAI in education – the AI as a collaborative inquiry partner – moving beyond the traditional tutor/assistant paradigm and directly addressing the GAIED theme of exploring new human-machine collaborative systems.
*   **Enhanced Learning Outcomes:** By fostering critical thinking, self-explanation, and metacognitive skills, the SLP could lead to deeper and more transferable learning compared to passive information consumption or answer-driven tutoring.
*   **Scalability and Accessibility:** An effective SLP could provide scalable opportunities for students to engage in Socratic inquiry and collaborative learning experiences, regardless of the availability of trained human facilitators or suitable peers.
*   **Contribution to AI Research:** This work will contribute insights into advanced prompting and fine-tuning techniques for guiding LLM behavior towards complex pedagogical goals, addressing challenges in generating authentic Socratic dialogue (Challenge 1 from literature review).
*   **Informing Educational Practice:** Findings will provide educators and instructional designers with evidence regarding the potential and limitations of using GenAI as a Socratic partner, guiding its integration into learning environments. The evaluation framework directly addresses the challenge of assessing pedagogical effectiveness highlighted by EducationQ (Shi et al., 2025) (Challenge 3).

**3. Methodology**

**3.1 Research Design**
This research will employ a mixed-methods approach, combining system development with rigorous experimental evaluation. The methodology is structured into two main phases: (1) System Development and (2) Experimental Evaluation.

**3.2 Phase 1: System Development**

*   **3.2.1 Data Collection and Curation:**
    *   **Objective:** To assemble a high-quality dataset for fine-tuning the SLP, emphasizing Socratic questioning and collaborative inquiry patterns.
    *   **Sources:** Potential sources include:
        *   Existing educational dialogue corpora (e.g., transcripts from tutoring sessions, classroom discussions), filtered for Socratic interactions.
        *   Philosophical Socratic dialogues.
        *   Transcripts of high-quality peer tutoring or collaborative problem-solving sessions.
        *   Synthetically generated dialogues using powerful LLMs (e.g., GPT-4) prompted to exemplify Socratic peer interactions, followed by human review and refinement.
    *   **Curation Criteria:** Dialogues will be selected/filtered/rated based on criteria such as: frequency of open-ended/probing questions, evidence of challenging assumptions, focus on student reasoning (not answers), constructive feedback, and collaborative turn-taking structure. We will develop a rubric for this curation process.

*   **3.2.2 LLM Selection and Architecture:**
    *   **Model Choice:** We will explore state-of-the-art LLMs, considering factors like underlying capabilities, context window size, fine-tuning accessibility, and cost. Potential candidates include models from the GPT series (OpenAI), Llama series (Meta), Mistral series, or other suitable open-source models that permit fine-tuning. The choice will be justified based on preliminary tests of Socratic capability via prompting and fine-tuning documentation.
    *   **System Architecture:** The SLP will consist of a core LLM integrated with a dialogue management module. This module will maintain basic state information (e.g., recent topics, student's expressed confusion points, key assumptions identified) to ensure conversational coherence and guide the questioning strategy over multiple turns.

*   **3.2.3 Algorithmic Steps: Prompt Engineering and Fine-tuning:**
    *   **Prompt Engineering:** We will design a structured meta-prompt to guide the LLM's behavior. This prompt will define the SLP's persona (a curious peer, not an expert tutor), its core objective (to stimulate student thinking through questions), and its constraints (e.g., avoid providing direct answers, prioritise questions that elicit reasoning, gently challenge inconsistencies). Example prompt fragment:
        ```
        You are Alex, a learning partner working with a student on [Topic]. Your goal is NOT to give answers but to help the student think deeply by asking probing questions. Act like a curious peer. Ask things like: 'That's interesting, can you explain why you think that?', 'What assumptions are we making here?', 'Is there another way to look at this?', 'What if [counter-example]?'. If the student is stuck, ask them to break down the problem or explain what they *do* understand. Do not provide solutions or confirm correctness directly. Focus on their reasoning process.
        ```
    *   **Fine-tuning:** The selected base LLM will be fine-tuned on the curated dialogue dataset.
        *   *Objective Function:* Standard supervised fine-tuning will be employed initially, maximizing the likelihood of generating responses that align with the Socratic/collaborative style in the dataset. The loss function is typically the cross-entropy loss:
        $$
        L_{SFT} = - \sum_{i=1}^{N} \sum_{j=1}^{|y_i|} \log P(y_{i,j} | y_{i,<j}, x_i; \theta)
        $$
        where $N$ is the number of examples, $x_i$ is the dialogue history (prompt), $y_i$ is the target Socratic response, $y_{i,j}$ is the $j$-th token of the response, and $\theta$ represents the model parameters.
        *   *Refinement (Optional):* Depending on initial results, we may explore Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO). This would involve collecting human preferences between pairs of potential SLP responses (e.g., rating which response is more Socratic or peer-like) and optimizing the model based on this feedback, potentially improving nuance and alignment beyond supervised fine-tuning. This directly addresses the challenge of simulating authentic Socratic dialogue (Challenge 1).

**3.3 Phase 2: Experimental Evaluation**

*   **3.3.1 Participants:** We aim to recruit N=60-90 undergraduate students (e.g., from an introductory university course in a domain requiring conceptual reasoning, such as physics, ethics, or programming). Participants will be randomly assigned to one of three conditions. Recruitment will be via university channels, with appropriate IRB approval and informed consent.

*   **3.3.2 Study Design:** A between-subjects experimental design will be used to compare three conditions:
    1.  **SLP Condition:** Students interact with the Socratic Learning Partner.
    2.  **Conventional AI Tutor (Baseline):** Students interact with an AI tutor designed to provide more direct explanations and answers (e.g., using a standard educational chatbot setup or a highly scaffolded version of the base LLM).
    3.  **Control Condition:** Students work on the problems independently using only static resources (e.g., textbook/notes), or potentially interact with a non-interactive logging system to control for interaction time.

*   **3.3.3 Task:** Participants will engage in one or two problem-solving tasks within the chosen domain. The tasks will be designed to require conceptual understanding and reasoning, rather than simple fact recall, making them suitable for Socratic inquiry (e.g., explaining a physics concept and applying it to a novel scenario, analyzing an ethical dilemma, debugging a complex piece of code conceptually).

*   **3.3.4 Procedure:**
    1.  **Pre-Test:** Assess baseline knowledge and potentially critical thinking skills relevant to the task domain.
    2.  **Introduction:** Brief participants on the task and how to interact with their assigned system (or work independently).
    3.  **Interaction Phase:** Participants engage with the problem-solving task using their assigned condition for a fixed duration (e.g., 30-45 minutes). All interactions (dialogue logs, actions) will be recorded. Think-aloud protocol may be used for a subset of participants to capture real-time reasoning.
    4.  **Post-Test:** Assess knowledge gain and understanding after the interaction phase.
    5.  **Survey:** Administer questionnaires to measure perceived learning experience, engagement, cognitive load, usefulness of the AI (if applicable), and perceived quality of interaction (e.g., using scales like SUS or custom Likert items).

*   **3.4 Data Collection Methods:**
    *   **Dialogue Logs:** Full transcripts of student-AI interactions (Conditions 1 & 2).
    *   **Performance Data:** Accuracy/quality of solutions to the problem-solving tasks.
    *   **Pre/Post Test Scores:** Measure learning gains using validated or carefully designed instruments.
    *   **Survey Responses:** Quantitative ratings and qualitative feedback on user experience.
    *   **(Optional) Think-Aloud Recordings:** Audio/video recordings of participants verbalizing their thoughts during the task.

*   **3.5 Evaluation Metrics:** Addressing evaluation complexity (Challenge 3) requires a multi-faceted approach:
    *   **Learning Outcomes (Quantitative):**
        *   Normalized learning gain: $(PostTest - PreTest) / (MaxScore - PreTest)$.
        *   Task performance scores (rubric-based assessment of solutions).
    *   **Interaction Quality & Process (Quantitative & Qualitative):**
        *   *Dialogue Analysis:* Code dialogue logs (manual coding with established inter-rater reliability or potentially automated methods) for:
            *   Frequency and type of AI questions (e.g., probing vs. clarifying vs. leading).
            *   Frequency and depth of student explanations (e.g., using word count, rubric for explanation quality).
            *   Instances of student self-correction or expressed changes in understanding.
            *   Ratio of AI questions to AI statements.
            *   Student question-asking frequency.
        *   *Survey Metrics:* Perceived helpfulness, engagement, ease of use, Socratic nature (for SLP), cognitive load.
    *   **Metacognition/Reflection (Qualitative):**
        *   Thematic analysis of think-aloud protocols and open-ended survey responses to identify evidence of reflection, strategy shifts, and articulation of reasoning stimulated by the interaction.

*   **3.6 Statistical Analysis:** ANOVA or equivalent non-parametric tests will be used to compare quantitative metrics (learning gain, survey scores, dialogue metrics) across the three conditions. Correlation analyses will explore relationships between interaction patterns and learning outcomes. Qualitative data will be analyzed using established thematic analysis techniques.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
1.  **A Functional SLP Prototype:** A demonstrable LLM-based agent capable of engaging students in Socratic dialogue aimed at collaborative inquiry.
2.  **Curated Dialogue Dataset:** A dataset useful for training AI models for Socratic interaction, potentially released to the research community.
3.  **Effective Prompting/Fine-tuning Strategies:** Identification and documentation of specific prompt structures and fine-tuning approaches that successfully elicit Socratic, peer-like behavior from LLMs.
4.  **Empirical Evaluation Results:** Comparative data demonstrating the effectiveness of the SLP in fostering student explanation, self-correction, and potentially learning gains, compared to conventional AI tutoring and independent learning. This will provide insights into how SLP addresses personalization (Challenge 2) through responsive questioning.
5.  **Analysis of Interaction Dynamics:** Detailed understanding of how students interact with the SLP, including successful patterns and potential pitfalls (e.g., student frustration, conversational dead-ends).
6.  **Methodological Contributions:** A validated mixed-methods framework for evaluating AI systems designed for complex pedagogical roles like Socratic partnership.

**4.2 Impact**
This research aims to make significant impacts aligned with the goals of the GAIED workshop:
*   **Advancing GAI→ED:** It directly contributes to the GAI→ED thrust by designing and evaluating a novel educational technology application of GenAI that moves beyond information delivery towards fostering higher-order thinking skills through human-AI collaboration.
*   **Enhancing Educational Practice:** If successful, the SLP concept offers a scalable tool to supplement traditional teaching, providing students with readily available opportunities for deep cognitive engagement and potentially improving critical thinking and metacognitive skills crucial for lifelong learning.
*   **Informing AI Development:** The project will push the boundaries of controlling LLM behavior for specific, nuanced pedagogical interactions, contributing knowledge about fine-tuning for role-playing and complex dialogue acts. It also addresses the ethical dimension (Challenge 5) implicitly by designing for supportive inquiry rather than potentially biased knowledge dissemination.
*   **Stimulating Further Research:** By demonstrating the potential (or limitations) of AI as a Socratic partner, this research will likely stimulate further investigation into AI roles in collaborative learning, adaptive inquiry systems, and the evaluation of AI pedagogical effectiveness.
*   **Building the GAIED Community:** We plan to disseminate findings through publications, presentations at workshops like GAIED, and potentially open-sourcing code or datasets, fostering collaboration among researchers, educators, and practitioners interested in the intersection of GenAI and education.

**5. References**

*   Chi, M. T., & Wylie, R. (2014). The ICAP Framework: Linking Cognitive Activities to Active Learning. *Educational Psychologist*, 49(4), 219–243.
*   Dan, Y., Lei, Z., Gu, Y., Li, Y., Yin, J., Lin, J., Ye, L., Tie, Z., Zhou, Y., Wang, Y., Zhou, A., Chen, Q., Zhou, J., He, L., & Qiu, X. (2023). *EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education*. arXiv preprint arXiv:2308.02773.
*   Jabbour, J., Kleinbard, K., Miller, O., Haussman, R., & Janapa Reddi, V. (2025). *SocratiQ: A Generative AI-Powered Learning Companion for Personalized Education and Broader Accessibility*. arXiv preprint arXiv:2502.00341. (Note: Year adjusted for consistency, assuming typo in original provided list).
*   OpenAI. (2023). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.
*   Paul, R., & Elder, L. (2007). Critical Thinking: The Art of Socratic Questioning. *Journal of Developmental Education*, 31(1), 36-37.
*   Shi, Y., Liang, R., & Xu, Y. (2025). *EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework*. arXiv preprint arXiv:2504.14928. (Note: Year adjusted for consistency).
*   Vygotsky, L. S. (1978). *Mind in society: The development of higher psychological processes*. Harvard University Press.
*   Zhang, L., Lin, J., Kuang, Z., Xu, S., & Hu, X. (2024). *SPL: A Socratic Playground for Learning Powered by Large Language Model*. arXiv preprint arXiv:2406.13919.

---