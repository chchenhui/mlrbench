**Research Proposal: SecureED: Generative AI for Detecting and Preventing AI-Generated Responses in Educational Assessments**

---

### 1. **Title**  
*SecureED: A Contrastive Learning Framework for Robust Detection of AI-Generated Content in High-Stakes Educational Assessments*

---

### 2. **Introduction**  
**Background**  
The rapid adoption of large foundation models (LFMs) like GPT-4 and Gemini has introduced transformative opportunities for educational assessment, including automated scoring and adaptive testing. However, this progress is accompanied by significant risks: students increasingly misuse LFMs to generate responses, undermining the validity of assessments. Current detection tools, such as GPTZero, suffer from poor generalizability across domains (e.g., mathematics vs. essays), susceptibility to adversarial attacks (e.g., paraphrasing), and limited ability to detect high-order reasoning tasks. These shortcomings erode trust in assessment systems and hinder the ethical integration of AI in education. Existing approaches, including watermarking and statistical analysis of perplexity, fail to address the dynamic nature of AI-generated text and lack explainability for educators.  

**Research Objectives**  
This study aims to:  
1. Develop *SecureED*, a contrastive learning framework optimized to distinguish AI-generated from human responses in educational contexts.  
2. Enhance cross-domain robustness by incorporating multimodal data (text, code, math) and adversarial training.  
3. Evaluate the framework’s ability to detect AI-generated content in high-order thinking tasks, such as creative problem-solving and deductive reasoning.  
4. Provide interpretable explanations of detection outcomes to foster trust among educators and students.  

**Significance**  
*SecureED* addresses critical gaps in the intersection of AI and educational assessment by:  
- **Preserving academic integrity** through reliable detection of AI misuse.  
- **Enabling safe adoption of generative AI** in assessment platforms without compromising validity.  
- **Advancing contrastive learning techniques** for domain-agnostic detection, setting a foundation for future research on trustworthy AI in education.  

---

### 3. **Methodology**  
#### **3.1 Research Design**  
The proposed framework combines contrastive learning, adversarial training, and domain-specific feature engineering. A phased approach includes:  
1. **Dataset Construction**: Curate a multimodal dataset of human and AI-generated responses across subjects (STEM, humanities) and question types (essays, code, math proofs).  
2. **Model Development**: Train *SecureED* using triplet networks to learn discriminative features between human and AI-generated content.  
3. **Adversarial Robustness Testing**: Simulate evasion tactics (e.g., paraphrasing) and refine the model iteratively.  
4. **Evaluation**: Benchmark against state-of-the-art detectors (GPTZero, ConDA) using accuracy, explainability, and cross-domain performance metrics.  

#### **3.2 Data Collection and Preprocessing**  
- **Sources**:  
  - **Human responses**: Collect essays, code submissions, and math problem-solving steps from public educational datasets (e.g., CodeComment, MATH dataset).  
  - **AI-generated responses**: Use diverse LFMs (GPT-4, Gemini, Llama-3) to generate answers for paired questions. Adversarial samples will include paraphrased outputs via tools like QuillBot.  
  - **Domain adaptation**: Integrate existing datasets (e.g., CyberHumanAI) for cross-validation.  
- **Preprocessing**:  
  - **Tokenization**: Use domain-specific tokenizers (e.g., CodeBERT for programming tasks).  
  - **Alignment**: Normalize text length and structure using sequence padding.  
  - **Labeling**: Annotate pairs as "human" or "AI-generated," with metadata for question types and subject domains.  

#### **3.3 Model Architecture**  
*SecureED* employs a **triplet contrastive learning framework** (Figure 1) with three key components:  
1. **Feature Encoder**: A transformer-based backbone (e.g., DeBERTa) to embed input text into latent representations.  
2. **Contrastive Loss**: Minimize distance between human samples (**H**) and maximize distance between human and AI-generated samples (**A**):  
   $$\mathcal{L}_{\text{cont}} = \sum_{i=1}^N \left[ \| f(H_i) - f(H_i^+) \|^2 - \| f(H_i) - f(A_i^-) \|^2 + \alpha \right]_+$$  
   where $f(\cdot)$ is the encoder, $H_i^+$ is a positive pair (same domain), $A_i^-$ is a negative pair (AI-generated), and $\alpha$ is a margin hyperparameter.  
3. **Domain Adversarial Network**: A gradient reversal layer (GRL) to learn domain-invariant features, enhancing generalizability.  

#### **3.4 Adversarial Training**  
To counter evasion tactics (e.g., paraphrasing):  
1. Generate perturbed AI samples using:  
   - **Lexical substitutions**: Replace keywords via synonym dictionaries.  
   - **Syntax restructuring**: Use bidirectional transformers to rephrase sentences.  
2. Train *SecureED* iteratively using a GAN-style setup:  
   - **Generator**: Perturbs AI-generated text to evade detection.  
   - **Discriminator**: *SecureED* classifies perturbed samples.  

#### **3.5 Feature Engineering for Educational Context**  
- **Domain-Specific Features**:  
  - **Reasoning coherence**: Measure logical consistency via graph-based semantic analysis.  
  - **Creativity patterns**: Quantify idea density using TF-IDF divergence between student and LFM responses.  
  - **Error distributions**: Track common mistakes (e.g., math miscalculations) unique to humans.  
- **Explainability**: Integrate SHAP values to highlight linguistic features (e.g., repetition, rigidity) indicative of AI generation.  

#### **3.6 Experimental Design**  
- **Baselines**: Compare against GPTZero, ConDA, and DeTeCtive.  
- **Metrics**:  
  - **Accuracy/F1-score**: Standard classification performance.  
  - **Cross-domain AUC**: Performance on unseen subjects/question types.  
  - **Robustness**: Success rate against paraphrasing attacks.  
  - **Explainability**: User trust score (survey of educators).  
- **Validation**:  
  1. In-distribution test: 80% of dataset for training, 20% for validation.  
  2. Cross-domain test: Evaluate on STEM vs. humanities tasks not seen during training.  
  3. Human evaluation: 50 educators rate detection explanations for clarity and usefulness.  

---

### 4. **Expected Outcomes & Impact**  
**Expected Outcomes**  
1. *SecureED* will achieve **≥90% F1-score** in detecting AI-generated content across domains, outperforming GPTZero (reported 78% F1) and ConDA (85% F1).  
2. The framework will demonstrate **<5% performance drop** under adversarial attacks, compared to 20–40% degradation in current tools.  
3. Cross-domain generalization will reduce the need for subject-specific retraining, enabling scalable deployment.  
4. Explanations via SHAP values will achieve a **≥4.0/5.0 user trust score** from educators.  

**Impact**  
- **Educational Assessment**: Preserve the validity of high-stakes exams by mitigating AI misuse, enabling institutions to adopt generative AI tools responsibly.  
- **AI Research**: Advance contrastive learning techniques for multimodal, domain-agnostic detection, with applications beyond education (e.g., journalism, legal documents).  
- **Policy & Practice**: Provide open-source detection APIs and integration guidelines for platforms like Canvas and Moodle, fostering equitable access to AI accountability tools.  

---

This proposal addresses the urgent need for robust, interpretable solutions to safeguard educational integrity while harnessing the potential of large foundation models. By bridging gaps in detection accuracy, generalizability, and trustworthiness, *SecureED* aims to set a new standard for AI accountability in education.