Title  
SecureED: A Contrastive Learning Framework for Robust Detection and Prevention of AI-Generated Responses in Educational Assessments  

1. Introduction  
Background  
The rapid emergence and widespread adoption of large foundation models (LFMs), such as GPT-4, Llama, and Gemini, have transformed educational practice by enabling unprecedented levels of content generation. At the same time, students may misuse these models to generate homework answers, test responses, or essays, thereby undermining academic integrity and the validity of high-stakes assessments. Current AI-generated content detectors (e.g., GPTZero, Originality.AI) show promising performance on narrow benchmarks but suffer from limited generalizability across subjects, question types (e.g., text, code, mathematics), and stylistic domains. Moreover, simple paraphrasing or adversarial editing can evade existing detectors, resulting in false negatives, while overly sensitive tools produce false positives that risk eroding trust among educators and learners.  

Research Objectives  
SecureED aims to fill these gaps by developing a robust, domain-aware, and explainable detection framework for AI-generated educational responses. Our core objectives are:  
• To design a multimodal contrastive learning model that learns domain-invariant representations of human-written versus LFM-generated answers across text, code, mathematical expressions, and diagrams.  
• To incorporate adversarial training strategies that harden the detector against paraphrasing, style transfer, and other evasion tactics.  
• To integrate explainability mechanisms that provide transparent feature attributions, thereby increasing stakeholder trust and facilitating fair use.  
• To release an open-source detection API and integration guidelines for seamless deployment in existing assessment platforms.  

Significance  
By delivering a high-fidelity, generalizable, and interpretable detection system, SecureED will restore confidence in automated and high-stakes assessments, enable educators to responsibly adopt generative AI tools for learning support, and set a new standard for accountability in educational AI.  

2. Methodology  
2.1 Overview  
SecureED combines contrastive representation learning, domain adaptation, adversarial training, and explainable AI. The architecture comprises three main components:  
  1. A dual-encoder contrastive network that maps responses to an embedding space.  
  2. A domain discriminator that enforces subject- and question-type invariance via adversarial objectives.  
  3. An explainability module that produces feature attributions and summary rationales.  

2.2 Data Collection and Preprocessing  
Datasets  
• Human-authored responses: We collect real student answers and expert model responses across four modalities—free text, programming code (Python, Java), mathematical derivations (LaTeX format), and diagram descriptions. Data sources include open educational repositories (e.g., Stanford Question Answering Dataset, CodeSearchNet, MATH dataset) and institution-provided assignments.  
• LFM-generated responses: For each human-authored item, we prompt multiple LFMs (GPT-4, Llama 2, Gemini) to generate answers under controlled temperature and max-token settings. We also include paraphrased and adversarially modified versions by (1) prompting LFMs to paraphrase, (2) back-translation, and (3) synonym substitution.  
Data Labels  
Each response is labeled as “human” or “AI,” with sublabels indicating the generator model and the evasion tactic used.  

Preprocessing  
• Tokenization: We apply a unified tokenizer (Byte-Pair Encoding) for textual and code data. Mathematical LaTeX is tokenized at the symbol level.  
• Feature augmentation: We extract domain features such as average sentence length, code cyclomatic complexity, mathematical symbol distribution, and reasoning coherence metrics (e.g., proof step count).  
• Normalization: Continuous features are standardized; categorical features use one-hot encoding.  

2.3 Model Architecture  
2.3.1 Contrastive Encoder  
We employ two encoders, $f_\theta(\cdot)$ (query) and $f_\phi(\cdot)$ (key), parameterized by Transformers. Following MoCo-style momentum updates, $\phi$ is updated by a moving average of $\theta$.  

Block equation for the contrastive loss (InfoNCE):  
$$  
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp\big(\mathrm{sim}(f_\theta(x_i),\,f_\phi(x_i^+)) / \tau\big)}{\exp\big(\mathrm{sim}(f_\theta(x_i),\,f_\phi(x_i^+)) / \tau\big) + \sum_{j=1}^K \exp\big(\mathrm{sim}(f_\theta(x_i),\,f_\phi(x_j^-)) / \tau\big)}  
$$  
where:  
• $x_i$ is a query response, $x_i^+$ is its positive pair (same label, e.g., “human”), and $\{x_j^-\}_{j=1}^K$ are negatives (opposite label).  
• $\mathrm{sim}(u,v)=u^\top v / (\|u\|\|v\|)$ is cosine similarity, and $\tau$ is a temperature hyperparameter.  
• $N$ is the batch size and $K$ is the number of negatives maintained in a dynamic queue.  

2.3.2 Domain Adaptation via Adversarial Discriminator  
To ensure the encoder’s representations are subject- and question-type invariant, we add a domain discriminator $D_\psi$ that predicts the domain label $d$ (e.g., mathematics, code, text). We optimize a gradient reversal layer (GRL) objective:  

$$  
\min_{\theta}\max_{\psi}\, \mathcal{L}_{\text{InfoNCE}}(f_\theta) - \lambda_{\text{adv}}\;\mathbb{E}_{x,d}\big[\log D_\psi(f_\theta(x))_d\big]  
$$  

Here $\lambda_{\text{adv}}$ balances contrastive and adversarial losses. The encoder seeks to maximize domain confusion while preserving class separation.  

2.3.3 Adversarial Training Module  
We incorporate hard negatives generated by:  
  a. Paraphrase LFM: use an auxiliary LFM to produce paraphrases of human answers, labeling them as negative samples.  
  b. Back-translation: translate to a pivot language and back.  
  c. Synonym and syntax substitutions.  
At each training epoch, we update the negative queue with these adversarial samples to increase robustness.  

2.3.4 Explainability Module  
We apply SHAP (SHapley Additive exPlanations) on the frozen encoder to compute feature attributions for each input token or feature vector. For long documents, we aggregate top-k tokens with highest absolute SHAP values and provide a human-readable rationale describing which phrases or structural patterns drove the “AI” or “human” decision.  

2.4 Training Protocol  
• Optimization: We use AdamW with learning rate $1e\!-\!5$, weight decay $0.01$. Momentum for key encoder is $0.995$.  
• Batch size: 128 queries + 128 keys, with a queue size of 16,384 negatives.  
• Hyperparameter tuning: We sweep $\tau\in\{0.05,0.1,0.2\}$, $\lambda_{\text{adv}}\in\{0.1,1.0,5.0\}$.  
• Early stopping: based on validation AUC on a held-out cross-domain set.  

2.5 Experimental Design  
Datasets Splits  
• In-domain test: 20% of data from seen subjects.  
• Cross-domain test: 20% from novel subjects and question types not used in training (e.g., art history, programming puzzles).  
• Adversarial test: a curated set of paraphrased, back-translated, and watermark-removed samples.  

Baseline Methods  
• GPTZero, Originality.AI, Copyleaks API.  
• Supervised classifiers (XGBoost, Random Forest) on engineered features.  
• ConDA (Bhattacharjee et al., 2023) and DeTeCtive (Guo et al., 2024).  

Evaluation Metrics  
• Detection accuracy, precision, recall, F1-score.  
• ROC AUC and Precision-Recall AUC.  
• Robustness: drop in F1 when paraphrasing adversarial set is applied.  
• Fairness measures: equality of error rates across demographic subgroups (when available).  
• Explainability quality: human-rated trustworthiness scores on a sample of 200 explanations.  

2.6 Integration and API Deployment  
We will package SecureED as a RESTful API with endpoints for batch and real-time scoring. The API returns:  
• Label: {“human”,”AI”} with confidence score.  
• Top-k contributing tokens or features and textual rationale.  
• Suggested remediation steps (e.g., “Flag for educator review”).  
We will provide Docker containers and client libraries (Python, JavaScript) for seamless integration into LMS platforms (Moodle, Canvas) and assessment engines.  

3. Expected Outcomes & Impact  
3.1 Expected Outcomes  
• A state-of-the-art detector that outperforms baselines by ≥10% in F1 on in-domain tests and ≥15% on cross-domain and adversarial sets.  
• A publicly released multimodal dataset of human and LFM responses with adversarial variants, under a permissive research license.  
• An open-source Python library and hosted API, complete with documentation, integration guides, and example notebooks.  
• A technical report and best-practice guidelines for educators and assessment designers on deploying AI detection responsibly.  

3.2 Impact  
Preserving Assessment Integrity  
SecureED will equip educators and institutions with a reliable tool to detect unauthorized AI assistance, thus upholding academic standards in an era of pervasive generative AI.  

Facilitating Safe AI Adoption  
By providing transparent explanations of detection decisions and clear integration pathways, SecureED lowers the barrier for responsible incorporation of generative AI into formative assessments and learning aids.  

Advancing Research in Trustworthy AI  
Our multimodal contrastive framework and adversarial training insights will contribute to the broader fields of AI safety, domain adaptation, and explainable machine learning, inspiring future work on protecting integrity in digital contexts beyond education.  

Equity and Fairness  
Through fairness evaluations and demographic error-rate monitoring, SecureED will set a precedent for equitable AI detection, ensuring that no student group is unfairly flagged due to language, cultural, or stylistic differences.  

Open Science and Community Engagement  
By open-sourcing our methods, data, and code, we promote transparency, reproducibility, and collaborative improvement, fostering a community around trustworthy, accountable AI in education.  

Conclusion  
SecureED addresses one of the most pressing challenges at the intersection of large foundation models and educational assessment: how to detect and prevent misuse of AI-generated content. By combining multimodal contrastive learning, adversarial robustness, domain adaptation, and explainability, this research offers a comprehensive solution that advances both theory and practice. The proposed datasets, open-source tools, and deployment guidelines will empower educators to safeguard assessment validity while harnessing the pedagogical potential of generative AI.