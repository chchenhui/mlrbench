1. Title  
Cognitive Architecture-Guided Training for Human-Like Reasoning in Language Models

2. Introduction  
Background. Large language models (LLMs) such as GPT-3, PaLM and LLaMA have demonstrated impressive proficiency across a wide range of natural language tasks. However, their internal reasoning processes remain opaque and often diverge from how humans think. Outputs may be factually incorrect, logically inconsistent, or lack transparency. From the behavioral sciences we know rich, formalized theories of human cognition—cognitive architectures like ACT-R, SOAR or CLARION—that specify how memory retrieval, attention, and rule-based reasoning interact to produce human decisions. To build AI systems that are both high-performing and psychologically interpretable we must bridge these cognitive theories with modern LLM practice.

Research Objectives. This proposal aims to:  
• Develop a hybrid training framework that aligns LLM internal representations with step-by-step traces generated by a cognitive architecture.  
• Design a constrained decoding mechanism that biases generation toward sequences consistent with cognitive-model predicted reasoning steps.  
• Evaluate the resulting “cognitively guided” LLM on standard reasoning benchmarks and human behavioral data, measuring both task performance and congruence with human step-wise reasoning.

Significance. By integrating validated cognitive architectures into LLM training and inference, we expect to 1) improve transparency—users can inspect intermediate reasoning traces; 2) enhance trust and alignment—outputs will follow psychologically plausible steps; and 3) open new interdisciplinary pathways between AI and behavioral science. Applications include educational tutoring (where model reasoning must mirror human pedagogical steps), healthcare decision support (transparent diagnostic chains), and collaborative human-AI work (where partners require human-like explanations).

3. Methodology  

3.1 Overview  
Our approach consists of two major innovations:  
  • A hybrid loss function combining standard language modeling loss with a cognitive-alignment penalty.  
  • A decoding algorithm that ranks candidate continuations by a joint LLM–cognitive-model score.

3.2 Data Collection and Preprocessing  
We will construct two corpora:  

  1. Human-reasoning transcripts. We collect protocols from published syllogistic reasoning studies (e.g. Wason selection task, transitive inference tasks) augmented with think-aloud chains of thought annotated at the level of cognitive operations (e.g., “retrieve premise,” “apply rule,” “generate conclusion”).  
  2. Standard reasoning benchmarks. Public datasets such as GSM8K (grade-school math word problems), CLUTRR (family relations), and ProofWriter (logical proof generation).  
   
Each chain-of-thought (CoT) is tokenized, and each reasoning step is assigned a discrete label $s_t\in\mathcal{S}$ drawn from a fixed cognitive taxonomy derived from an ACT-R module (e.g. retrieval, goal selection, production firing).

3.3 Cognitive Architecture Simulation  
We adopt ACT-R as our reference cognitive model. For each reasoning problem $x$, the ACT-R simulator produces a sequence of cognitive states and operations:  
$$ (s_1^{\text{ACTR}}, s_2^{\text{ACTR}}, \dots, s_{T_x}^{\text{ACTR}})\,. $$  
We extract from each state:  
  – Buffers content (goal, retrieval).  
  – Production rule identifiers.  
  – Salience measures.  

This trace provides a probabilistic model $P_{\mathrm{ACTR}}(s_t\mid s_{<t},x)$ over the next cognitive operation.

3.4 Hybrid Training Objective  
We fine-tune a pretrained LLM parameterized by $\theta$ on paired $(x,y,\mathbf{s}^{\mathrm{ACTR}})$, where $y=(y_1,\dots,y_T)$ is the target CoT or final answer, and $\mathbf{s}^{\mathrm{ACTR}}=(s_1^{\mathrm{ACTR}},\dots,s_T^{\mathrm{ACTR}})$ the cognitive trace. Our total loss is:  

Block equation for hybrid loss  
$$
\mathcal{L}(\theta) \;=\;  
-\sum_{(x,y,\mathbf{s})\in\mathcal{D}} \Biggl[ 
\sum_{t=1}^T \log p_\theta(y_t\mid y_{<t},x)
\;+\;\lambda\,\sum_{t=1}^T D_{\mathrm{KL}}\!\bigl(
P_{\mathrm{ACTR}}(s_t\mid s_{<t},x)
\;\|\;
P_\theta(s_t\mid y_{<t},x)\bigr)\Biggr].
$$

Here:  
  • $p_\theta(y_t\mid y_{<t},x)$ is the standard next-token probability.  
  • $P_\theta(s_t\mid y_{<t},x)$ is a small classification head on the LLM hidden state that predicts the cognitive operation at step $t$.  
  • $D_{\mathrm{KL}}(\cdot\|\cdot)$ is the Kullback–Leibler divergence.  
  • $\lambda>0$ trades off language modeling vs. cognitive alignment.

3.5 Constrained Decoding Mechanism  
At inference we generate a chain of thought by beam search (beam size $B$), but each partial hypothesis $h = (y_1,\dots,y_t)$ carries a combined score:  
$$
\mathrm{score}(h) \;=\;
\sum_{i=1}^t \Bigl[\log p_\theta(y_i\mid y_{<i},x)
\;+\;\beta\,\log P_{\mathrm{ACTR}}\bigl(\hat s_i\mid \hat s_{<i},x\bigr)\Bigr]\,,
$$  
where $\hat s_i=\arg\max_s P_\theta(s\mid y_{<i},x)$ is the most likely predicted operation. The hyperparameter $\beta$ controls how strongly decoding adheres to the cognitive model.

3.6 Experimental Design  

Datasets & Tasks  
  • Syllogistic reasoning (4-term and 5-term syllogisms).  
  • GSM8K for arithmetic reasoning.  
  • CLUTRR for commonsense relational reasoning.  
  • Human evaluation tasks: users rate explanations on naturalness and trust.  

Baselines  
  1. Off-the-shelf LLM with chain-of-thought prompting.  
  2. LLM fine-tuned on CoTs without cognitive alignment (i.e. $\lambda=0$).  
  3. LLM fine-tuned with RLHF.  

Ablations  
  • Vary $\lambda\in\{0,0.1,1.0\}$ and $\beta\in\{0,0.5,1.0\}$.  
  • Compare ACT-R vs. CLARION as the cognitive guide.  

Evaluation Metrics  
  – Task Accuracy: final answer correctness.  
  – CoT Fidelity: proportion of steps where $\hat s_t=s_t^{\mathrm{ACTR}}$ (precision, recall, F1).  
  – Cognitive Distance: average KL divergence $\frac1T\sum_t D_{\mathrm{KL}}(P_{\mathrm{ACTR}}\|P_\theta)$.  
  – Interpretability Score: crowd-sourced rating on a 5-point scale for “How human-like is this reasoning?”  
  – Human Trust Score: “How much do you trust the model’s conclusion given this explanation?”

Computational Setup  
  – Base model: GPT-2 Medium or LLaMA-7B for proof-of-concept, scaling to GPT-3.5 in later phases.  
  – ACT-R version 7.9 implementation.  
  – Training on 8×A100 GPUs, early stopping on a held-out human reasoning set.

4. Expected Outcomes & Impact  

4.1 Expected Outcomes  
  • Improved Interpretability: We anticipate a large increase in CoT Fidelity (from ~30% in standard LLMs to >70% under cognitive alignment).  
  • Competitive Performance: Task accuracy on GSM8K and CLUTRR should match or exceed standard fine-tuned LLMs (±1–2%), mitigating the typical performance–interpretability trade-off.  
  • Human Preference Gains: In user studies, we expect ≥80% of participants to prefer cognitively guided explanations for clarity and trustworthiness.  
  • Generalization: Models trained under this regime should transfer better to unseen reasoning formats (e.g. novel syllogism patterns), as cognitive constraints regularize reasoning strategies.

4.2 Broader Impact  
  • Alignment and Safety: By rooting model decisions in human-validated cognitive processes, we create a path toward safer, more aligned AI systems that can be audited step by step.  
  • Interdisciplinary Bridges: This project forges collaboration between AI researchers, cognitive scientists, and psychologists—validating psychological theories at scale and enriching AI with human behavioral insights.  
  • Applications in Education: Tutors that present human-style step-by-step solutions can adapt to students’ cognitive load and pedagogical best practices.  
  • Healthcare & Law: Decision-support tools with transparent reasoning chains will foster acceptance among practitioners concerned about black-box diagnostics.  
  • Ethical Considerations: We will monitor for biases in cognitive models (e.g. framing effects) and ensure alignment processes do not inadvertently amplify undesirable human biases.

5. Conclusion & Future Work  
We propose a novel framework that tightly integrates cognitive architectures into both the training and inference of language models, yielding systems whose reasoning closely mirrors human psychological processes. Our hybrid loss and constrained decoding mechanisms are designed to be modular—applicable across LLM families and cognitive theories. Future directions include: extending beyond ACT-R to dynamic models of emotion and motivation; applying the framework to multimodal reasoning (vision + language); and deploying in interactive environments (e.g. tutoring systems, collaborative robots). This research paves the way toward truly human-aligned AI that reasons, explains, and collaborates just as people do.