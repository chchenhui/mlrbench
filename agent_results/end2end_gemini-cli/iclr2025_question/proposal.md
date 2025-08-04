### **1. Title: Disentangling Epistemic and Aleatoric Uncertainty for Hallucination-Aware Generation in Large Language Models**

---

### **2. Introduction**

**2.1 Background and Motivation**
Large Language Models (LLMs) and other foundation models have demonstrated unprecedented capabilities in generating human-like text, translating languages, and answering complex questions. Their integration into high-stakes domains such as healthcare, finance, and law is rapidly accelerating. However, a significant barrier to their reliable deployment is the phenomenon of "hallucination," where models generate factually incorrect, nonsensical, or unfaithful information with unwavering confidence (A Survey on Uncertainty Quantification of Large Language Models, 2024). This unreliability poses substantial risks, as end-users may place undue trust in flawed outputs.

To address this, Uncertainty Quantification (UQ) has emerged as a crucial field of study. UQ aims to provide a measure of a model's confidence in its predictions, enabling a more nuanced assessment of when an LLM's output can be trusted and when human oversight is necessary. Recent research has produced a variety of UQ techniques, including ensemble-based methods like Uncertainty-Aware Fusion (UAF) (Dey et al., 2025), computationally efficient single-pass methods such as Semantic Entropy Probes (SEPs) (Kossen et al., 2024), and framework-specific approaches for Retrieval-Augmented Generation (FRANQ) (Fadeeva et al., 2025). These methods have shown promise in identifying hallucinations by associating them with high uncertainty scores (Bouchard & Chauhan, 2025; Shelmanov et al., 2025).

However, a fundamental limitation persists in the current UQ landscape: the conflation of different sources of uncertainty. LLMs operate in both constrained, factual domains and open-ended, creative ones. The uncertainty in answering "What is the capital of France?" is fundamentally different from the uncertainty in "Write a poem about the sea." The former involves a single correct answer, and any deviation signals a model's lack of knowledge. The latter invites infinite valid, creative responses. Current UQ methods often treat all sources of high textual variance as indicative of potential error, which can stifle the model's desirable creative capabilities. This creates a critical trade-off: mitigating hallucinations often comes at the cost of making models overly cautious and less useful for generative tasks.

This proposal argues for a more nuanced approach. The key to building truly reliable and versatile LLMs lies in **disentangling uncertainty**. We must distinguish between:
1.  **Epistemic Uncertainty ($U_E$)**: This is *model uncertainty* arising from a lack of knowledge. It is reducible with more data or better model parameters and is the primary indicator of factual hallucinations. When an LLM does not know the answer to a factual question, its epistemic uncertainty should be high.
2.  **Aleatoric Uncertainty ($U_A$)**: This is *data uncertainty* inherent in the task itself. It is irreducible and reflects the natural ambiguity or variability in the data. Creative writing, open-ended ideation, and summarizing subjective opinions are tasks with high aleatoric uncertainty. This uncertainty is a feature, not a bug.

By disentangling these two sources, we can develop a system that selectively flags outputs with high epistemic uncertainty as potential factual errors requiring verification, while permitting and even encouraging outputs with high aleatoric uncertainty in creative contexts. This would allow us to mitigate harmful hallucinations without sacrificing the generative power that makes LLMs so transformative.

**2.2 Research Objectives**
This research aims to develop and validate a novel framework for disentangling epistemic and aleatoric uncertainty in LLMs to enable hallucination-aware generation. Our specific objectives are:

1.  **To develop a theoretical and practical framework for disentangling $U_E$ and $U_A$ in autoregressive text generation.** This involves proposing a model architecture and a specialized training objective designed to explicitly separate these two uncertainty types.
2.  **To construct a specialized, multi-domain dataset for training and evaluating uncertainty disentanglement.** This dataset will contain a carefully balanced mix of tasks with low aleatoric uncertainty (e.g., factual QA) and high aleatoric uncertainty (e.g., creative writing), enabling the model to learn the distinction.
3.  **To implement and train an LLM using the proposed framework.** We will fine-tune a pre-trained foundation model with a composite loss function that encourages the model to predict distinct scores for epistemic and aleatoric uncertainty based on the context.
4.  **To rigorously evaluate the proposed method against state-of-the-art UQ baselines.** We will assess its ability to (a) accurately detect factual hallucinations, (b) preserve generative quality and diversity in creative tasks, and (c) provide well-calibrated and interpretable uncertainty estimates.

**2.3 Significance**
The successful completion of this research will have significant scientific and practical impacts. Scientifically, it will advance the theoretical understanding of uncertainty in generative models, moving beyond monolithic uncertainty scores to a more principled, disentangled representation. It directly addresses the key challenges of differentiating uncertainty types and balancing creativity with accuracy, identified as critical open problems in recent literature.

Practically, this work will pave the way for more reliable, controllable, and trustworthy AI systems. A model equipped with our framework could dynamically adjust its behavior:
-   In a medical report summarization task, a high $U_E$ score would trigger a request for human review or flag a statement as "unverified."
-   In a brainstorming session with a marketing team, high $U_A$ would be interpreted as a signal of creative exploration, allowing the model to generate diverse ideas without being penalized.

This capability would enhance user trust, reduce the risks of deploying LLMs in critical applications, and ultimately unlock their full potential as both powerful knowledge engines and versatile creative partners.

---

### **3. Methodology**

Our research is structured into three main phases: (1) Dataset Construction, (2) Model Development and Training, and (3) Experimental Validation.

**3.1 Phase 1: Dataset Construction for Uncertainty Disentanglement (DUnD)**
A core hypothesis of this work is that an LLM can learn to disentangle uncertainties if trained on a dataset that makes the distinction explicit. We will construct the **Disentangled Uncertainty Dataset (DUnD)**, a multi-source corpus where each instance is labeled with a context type indicating its inherent aleatoric uncertainty.

*   **Low Aleatoric Uncertainty (Factual) Component:** This partition will consist of tasks with a narrow, well-defined solution space.
    *   **Sources:** We will use established question-answering and fact-checking datasets such as Natural Questions, TriviaQA, and BoolQ.
    *   **Data Format:** Each instance will be a `(prompt, reference_answer)` pair. The prompt is the question, and the `reference_answer` is the ground-truth factual response. These instances will be labeled with `context_type = FACTUAL`.

*   **High Aleatoric Uncertainty (Creative) Component:** This partition will feature open-ended tasks where a wide range of outputs are valid and desirable.
    *   **Sources:** We will draw from datasets like the Stanford Alpaca dataset (for creative instruction-following), the Story-Cloze Test (for narrative generation), and prompts from creative writing forums.
    *   **Data Format:** Each instance will be a `(prompt, [example_completion_1, ..., example_completion_N])` pair. The prompts will ask for stories, poems, ideas, or dialogues. We will not have a single "correct" answer, but rather a set of diverse, high-quality examples to illustrate the desired variability. These instances will be labeled with `context_type = CREATIVE`.

*   **Out-of-Distribution (OOD) Factual Component:** To train the model to recognize its knowledge boundaries (i.e., high epistemic uncertainty), we will curate a subset of factual questions that are intentionally obscure, recent, or from highly specialized domains (e.g., advanced theoretical physics, niche legal precedents) that are unlikely to be well-represented in the base model's pre-training data. These will also be labeled `context_type = FACTUAL`, but will serve as a proxy for high-$U_E$ scenarios during training and evaluation.

The final DUnD dataset will comprise approximately 200,000 instances, balanced between the factual and creative components, with a dedicated split for training, validation, and testing.

**3.2 Phase 2: Model Architecture and Disentangled Training Objective**
We propose a novel fine-tuning approach that equips a standard pre-trained LLM with the ability to predict disentangled uncertainty scores.

*   **Base Model:** We will use a powerful, publicly available foundation model, such as Llama-3-8B or Mistral-7B, as our starting point. The core weights of the LLM will be fine-tuned.

*   **Model Architecture:** The architecture consists of the base LLM followed by two output heads that share the final hidden state $h_t$ at each decoding step $t$:
    1.  **Token Prediction Head:** A standard linear layer that maps $h_t$ to a probability distribution over the vocabulary, $p(y_t | y_{<t}, x)$.
    2.  **Uncertainty Prediction Head:** A small Multi-Layer Perceptron (MLP) that maps $h_t$ to a two-dimensional vector representing the predicted uncertainties: $[\hat{U}_{E,t}, \hat{U}_{A,t}]$. These are scalar values, activated with a sigmoid or softplus function to ensure they are non-negative.

*   **Disentangled Uncertainty Loss Function ($L_{DU}$):** Our primary contribution is a composite loss function designed to force the model to learn the desired separation. The total loss for a sequence is:
    $$
    L_{total} = L_{NLL} + \lambda_{U} L_{DU}
    $$
    where $L_{NLL}$ is the standard negative log-likelihood loss for next-token prediction, and $L_{DU}$ is our novel disentanglement loss, weighted by a hyperparameter $\lambda_U$.

The disentanglement loss $L_{DU}$ is defined as follows:
$$
L_{DU} = \frac{1}{T} \sum_{t=1}^{T} \left( L_E(\hat{U}_{E,t}, \text{context}) + L_A(\hat{U}_{A,t}, \text{context}) \right)
$$
where $T$ is the sequence length. The component losses $L_E$ and $L_A$ are defined based on the `context_type` from our DUnD dataset. We use a Mean Squared Error (MSE) loss to push the predicted uncertainties towards target values (0 for low, 1 for high):

*   **For `context_type = FACTUAL` (low aleatoric uncertainty):**
    *   We want low aleatoric uncertainty prediction: $L_A = (\hat{U}_{A,t} - 0)^2$.
    *   Epistemic uncertainty should be low for in-domain facts and high for OOD facts. For simplicity in the initial training stage, we enforce low epistemic uncertainty on our training data, assuming it is "known": $L_E = (\hat{U}_{E,t} - 0)^2$. The high-$U_E$ signal for OOD data will be learned implicitly as these inputs will generate higher token-level entropy and model disagreement, which the uncertainty head can be trained to capture during a second-stage fine-tuning or through contrastive learning.

*   **For `context_type = CREATIVE` (high aleatoric uncertainty):**
    *   We want high aleatoric uncertainty prediction: $L_A = (\hat{U}_{A,t} - 1)^2$.
    *   We want low epistemic uncertainty, as the model should be "confident" in its creative process, not "ignorant": $L_E = (\hat{U}_{E,t} - 0)^2$.

By minimizing this composite loss, the model learns not only to generate coherent text but also to self-report its uncertainty type at each step, associating high $\hat{U}_A$ with creative contexts and reserving high $\hat{U}_E$ for situations where it lacks factual knowledge.

**3.3 Phase 3: Experimental Design and Validation**
We will conduct a comprehensive evaluation to validate our framework's effectiveness.

*   **Baselines:** Our proposed model, which we'll call **DUnE-LLM** (Disentangled Uncertainty Estimation LLM), will be compared against several state-of-the-art baselines:
    1.  **Token-level Entropy:** The Shannon entropy of the next-token prediction distribution. This is a simple, common UQ measure.
    2.  **Monte Carlo (MC) Dropout:** A Bayesian approximation method where multiple forward passes are performed with dropout enabled at inference time. The variance of the predictions serves as an uncertainty estimate.
    3.  **Semantic Entropy Probes (SEPs):** A state-of-the-art single-pass method that approximates semantic entropy from hidden states (Kossen et al., 2024). This represents a strong, computationally efficient baseline.
    4.  **Ensemble Methods:** A small ensemble of models to capture model uncertainty, similar in spirit to UAF (Dey et al., 2025).

*   **Evaluation Tasks and Metrics:**
    1.  **Uncertainty Disentanglement Evaluation:**
        *   **Task:** Using the test split of our DUnD dataset, we will classify each generated token as originating from a `FACTUAL` or `CREATIVE` context.
        *   **Metric:** We will use the Area Under the Receiver Operating Characteristic Curve (AUROC) to measure how well our predicted $\hat{U}_A$ score can distinguish between the two contexts. We expect DUnE-LLM to significantly outperform baselines, whose monolithic uncertainty scores should poorly discriminate between the contexts.

    2.  **Hallucination Detection Evaluation:**
        *   **Task:** We will evaluate on established hallucination detection benchmarks like **TruthfulQA** and **HaluEval**. We will task the models with answering factual questions and use their uncertainty scores to predict whether a generated answer is correct or a hallucination.
        *   **Metrics:** We will measure the AUROC for detecting hallucinated responses using the predicted $\hat{U}_E$ score from DUnE-LLM versus the general uncertainty scores from baselines. We will also report Precision, Recall, and F1-score for a binary hallucination classification task by thresholding the uncertainty scores.

    3.  **Creativity and Generation Quality Preservation:**
        *   **Task:** We will use prompts from creative writing datasets and evaluate the generated outputs.
        *   **Metrics:**
            *   **Quality:** Human evaluation will be conducted on a 5-point Likert scale for coherence, relevance, and creativity.
            *   **Diversity:** We will measure semantic diversity by calculating the average pairwise cosine distance between embeddings of multiple generations for the same prompt. We will also measure lexical diversity using `distinct-n` scores. We hypothesize that baselines, when coupled with a rejection sampling mechanism to filter high-uncertainty outputs, will show reduced diversity compared to our DUnE-LLM, which should remain creative.

    4.  **Uncertainty Calibration:**
        *   **Task:** We will assess how well the predicted uncertainty scores reflect the true likelihood of error.
        *   **Metrics:** We will compute the **Expected Calibration Error (ECE)** for our epistemic uncertainty score $\hat{U}_E$. A well-calibrated model's confidence should align with its accuracy. We will plot reliability diagrams to visualize calibration.

---

### **4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**
This research is expected to produce several key outcomes:

1.  **A Novel Framework for Uncertainty Disentanglement:** The primary outcome will be the DUnE-LLM methodology, including the model architecture modifications and the novel Disentangled Uncertainty Loss function. This will be the first framework, to our knowledge, that explicitly trains an LLM to separate epistemic and aleatoric uncertainty in text generation.

2.  **A Publicly Available Benchmark Dataset (DUnD):** We will release the DUnD dataset to the research community. This resource will facilitate future research on nuanced uncertainty quantification, model introspection, and the trade-offs between factuality and creativity.

3.  **A Fine-tuned, Hallucination-Aware LLM:** We will release the weights of our best-performing DUnE-LLM model, providing a practical, off-the-shelf tool for researchers and practitioners interested in deploying more reliable LLMs.

4.  **Comprehensive Empirical Validation:** Our work will provide strong empirical evidence demonstrating the superiority of a disentangled approach to uncertainty over existing monolithic methods. We expect to show that our $\hat{U}_E$ score is a better predictor of factual hallucinations than general uncertainty, and that our framework preserves creative capacity far better than traditional uncertainty-based filtering.

**4.2 Broader Impact**
The impact of this research extends beyond the academic community. By enabling LLMs to distinguish between "I don't know" (high $U_E$) and "Let's be creative" (high $U_A$), our work will contribute to a new generation of more trustworthy and context-aware AI.

*   **For High-Stakes Applications:** In domains like law, medicine, and engineering, our framework can act as a critical safety layer. A high epistemic uncertainty score can automatically trigger fallback mechanisms, such as consulting a database, performing a web search via a RAG module, or escalating the query to a human expert. This will reduce the risk of catastrophic failures due to model hallucination and build the necessary trust for wider adoption.

*   **For Creative and Collaborative Applications:** The framework will unshackle LLMs from the overly conservative behavior induced by naive hallucination mitigation. Content creators, writers, and designers will be able to leverage the full creative potential of LLMs, confident that the model's exploratory generation will not be misconstrued as factual error.

*   **For Human-AI Interaction:** Communicating uncertainty is a key challenge. Our disentangled approach provides a more intuitive way to convey a model's state to a user. Instead of a generic "I am 70% confident," the model could respond, "This is a factual question and I am not certain of the answer" (high $U_E$) or "There are many creative ways to answer this; here is one idea" (high $U_A$). This enhanced transparency will foster more effective and trusting human-AI collaboration.

In conclusion, this research tackles a fundamental, pressing challenge at the frontier of reliable AI. By moving from quantifying uncertainty to *understanding* it, we aim to build a foundation for LLMs that are not only powerful but also self-aware, safe, and truly aligned with diverse human needs.