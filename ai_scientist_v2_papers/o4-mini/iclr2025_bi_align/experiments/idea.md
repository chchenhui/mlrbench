## Name

coadaptive_explanation_alignment

## Title

Co-Adaptive Explanation Interfaces: Aligning AI and Human Reasoning through Dual-Channel Feedback

## Short Hypothesis

We hypothesize that a co-adaptive explanation interface which (1) actively models individual users' cognitive biases and (2) provides dual-channel explanations (content justification plus bias-awareness signals) will yield faster, more robust mutual alignment than static or single-channel XAI approaches. This setting isolates the dynamic feedback loop between AI explanation adaptation and human reinterpretation, a process not captured by standard offline explainability evaluations.

## Related Work

Traditional XAI methods (e.g., LIME, SHAP) focus on static post hoc explanations (Ribeiro et al. 2016; Lundberg & Lee 2017). Human‐in‐the‐loop RL and interactive machine teaching papers (e.g., Kulesza et al. 2015; Amershi et al. 2014) explore customizing model behavior but do not explicitly address human cognitive biases in the explanation loop. Recent personalized explanation work (e.g., Poursabzi‐Sangdeh et al. 2021) adjusts explanations to expertise level but lacks real‐time user bias modeling and bidirectional adaptation. Our proposal uniquely integrates a dynamic bias detector into the XAI interface and closes the loop by feeding insights back into the explanation generation process, enabling true co-adaptation.

## Abstract

As AI systems assume ever more complex decision roles, static post hoc explanations fail to ensure lasting trust and correct mental models. We introduce Co-Adaptive Explanation Interfaces, an interactive framework in which AI continually models each user’s cognitive biases and adapts its explanations through two channels: (a) a content justification channel that conveys the model’s reasoning, and (b) a bias-awareness channel that alerts users when their inferred biases may misalign with the model. Corrections from users are treated as feedback to update both the AI’s internal decision model and its user bias estimates. We study this bidirectional alignment process in a controlled labeling task, comparing our approach to baseline static and single-channel dynamic explainers. We demonstrate that dual-channel co-adaptation leads to faster convergence of human trust calibration, improved task accuracy, and better alignment of human mental models with AI reasoning. Our results highlight the importance of modeling human cognitive factors in XAI and provide a blueprint for building co-adaptive human-AI systems that evolve together.

## Experiments

- 1. System Implementation: Build a simulated classification task (e.g., image labeling) with an underlying neural model. Implement three interfaces: (a) static LIME‐style explanations, (b) single‐channel dynamic explanations that adapt to user corrections, and (c) our dual‐channel co-adaptive interface that also models and signals user bias.
- 2. User Study: Recruit N=60 participants, randomly assign to the three conditions. Each participant completes 100 labeling trials. Collect per‐trial: (i) user decisions, (ii) self‐reported trust score, (iii) response times, and (iv) bias indicators (e.g., tendency to overweight certain features).
- 3. Metrics: Measure (i) trust calibration error (|reported trust – model accuracy|), (ii) labeling accuracy, (iii) convergence speed of bias estimates (via Kullback–Leibler divergence to ground‐truth simulated bias), and (iv) mental‐model alignment (via post‐task questionnaire).
- 4. Ablation Study: Disable either the bias detector or the dynamic explanation update to quantify each component’s contribution to overall alignment.
- 5. Simulation Experiments: Simulate synthetic users with known cognitive bias profiles to test scalability and robustness of the bias estimation submodule under varying noise levels.

## Risk Factors And Limitations

- Modeling real human biases accurately is challenging; supplanting real users with simulated bias profiles may oversimplify complexities.
- Dual-channel explanations may overwhelm some users or introduce cognitive load, reducing overall performance in complex tasks.
- Findings in controlled labeling tasks may not directly generalize to high-stakes domains (e.g., medical diagnosis) without further domain adaptation.
- Dependence on self-reported trust scores can introduce subjectivity; alternative implicit measures may be needed for validation.

