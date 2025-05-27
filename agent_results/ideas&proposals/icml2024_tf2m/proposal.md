1. Title  
A Bayesian Framework for In-Context Learning in Transformer-Based Foundation Models  

2. Introduction  
Background  
In-context learning (ICL) refers to the ability of large language models (LLMs) to perform new tasks merely by conditioning on a prompt that contains a few input–output examples, without any gradient-based parameter updates. Since its discovery in GPT-3 (Brown et al., 2020), ICL has been one of the most striking emergent capabilities of transformer-based foundation models. By simply concatenating demonstration pairs $(x_i,y_i)$ and a test input $x_*$, a pretrained LLM will generate the corresponding output $y_*$ as if it had been fine-tuned for the task. This zero-shot or few-shot adaptability holds great promise for rapid prototyping, personalization, and deployment in safety-critical domains.  

Despite its success across a wide variety of tasks—from sentiment analysis to arithmetic reasoning—our theoretical understanding of ICL remains fragmented. Empirical studies (Wei et al., 2023; Liu et al., 2024) have shown that model scale, prompt ordering, and demonstration selection strongly influence performance, yet we lack a unified framework that explains these phenomena or suggests principled methods to improve ICL. This gap undermines our ability to predict when and why ICL will succeed, to optimize model architectures for in-context adaptation, and to guarantee reliable, interpretable behavior in real-world applications.  

Research Objectives  
This proposal aims to develop a rigorous theoretical framework that characterizes in-context learning within transformer-based LLMs as an implicit Bayesian inference process carried out by the attention mechanism. Our specific objectives are:  
• Objective 1: Formalize how self-attention over context tokens implements a kernel-based approximation to a posterior predictive distribution.  
• Objective 2: Derive sample complexity and generalization bounds for ICL under common distributional assumptions (e.g., mixture of latent tasks, conditional exponential families).  
• Objective 3: Validate theoretical predictions empirically on synthetic and real-world few-shot tasks, and use insights to propose new attention modifications that improve ICL performance.  

Significance  
A principled Bayesian view of ICL will bridge the gap between the empirical success of transformer models and classical theories in statistical learning and information theory. The anticipated contributions include (i) conditions under which ICL provably recovers the Bayes-optimal learner, (ii) quantitative bounds on the number of demonstrations required for reliable adaptation, and (iii) novel, theory-inspired methods for demonstration selection, prompt engineering, and architecture design. By grounding ICL in a well-understood inferential paradigm, we will enable more efficient, transparent, and robust deployment of foundation models in domains such as healthcare, finance, and legal reasoning.  

3. Methodology  

3.1 Theoretical Model of In-Context Learning  
We model a few-shot task as sampling a test input $x_*$ and output $y_*$ from an unknown conditional distribution $p^*(y\mid x)$. A context $C = \{(x_i,y_i)\}_{i=1}^n$ is a set of demonstration pairs drawn i.i.d. from the same distribution. A pretrained LLM with parameters $\theta$ induces, via its transformer layers, a conditional distribution  
$$
\hat p_\theta(y_* \mid x_*, C) \;=\;\mathrm{Softmax}\bigl(f_\theta(x_*,C)\bigr)\,,
$$  
where $f_\theta(\cdot)$ maps tokens to logits through self-attention and feed-forward modules. Our key hypothesis is that, under mild assumptions on $\theta$ and the pretraining data, the self-attention mechanism approximates a kernel-weighted average of the demonstration labels, akin to a nonparametric Bayes predictor.  

3.1.1 Attention as Kernel Density Estimator  
Let $q_i = Q_\theta(x_i)$ and $k_j = K_\theta(x_j)$ be the query and key vectors for token representations, and $v_j = V_\theta(y_j)$ the value vector for the label tokens. The standard scaled dot-product attention yields weights  
$$
\alpha_{*j} \;=\;\frac{\exp\bigl(q_*^\top k_j / \sqrt{d}\bigr)}{\sum_{\ell=1}^n \exp\bigl(q_*^\top k_\ell / \sqrt{d}\bigr)}\,.
$$  
We show that, if the mapping $x\mapsto q$ and $x\mapsto k$ approximately preserves a suitable task-relevant distance metric, then $\alpha_{*j}$ implements a kernel density estimate for the posterior predictive probability:  
$$
\hat p(y\mid x_*,C)\;=\;\sum_{j=1}^n \alpha_{*j}\,\delta(y=y_j)\quad\approx\;\int p(y\mid x) \, K_h(d(x,x_*))\,\mathrm{d}x\,,
$$  
where $K_h(\cdot)$ is a kernel with bandwidth $h$. This perspective links the temperature scaling in attention to the kernel bandwidth and suggests that demonstration selection and prompt ordering influence the effective support of the kernel.  

3.1.2 Generalization and Sample Complexity Bounds  
Building on the kernel view, we derive probabilistic bounds on the ICL risk. Define the ICL predictor $\hat p_n(y\mid x)$ as above. Under standard smoothness and tail conditions on $p^*(y\mid x)$ and assuming the keys/queries embed $x$ into a bounded metric space, classical results in nonparametric regression (Devroye et al., 1996) imply:  
$$
\mathbb{E}\!\bigl[L(\hat p_n)\bigr]\;-\;L(p^*)\;\le\;O\Bigl(n^{-2/(2+d)}\Bigr)\quad\text{with high probability,}
$$  
where $L(\cdot)$ is log-loss and $d$ is the intrinsic dimension of the data manifold. We will refine these bounds by accounting for transformer depth, the Lipschitz constants of the mappings $Q_\theta,K_\theta,V_\theta$, and the discrete nature of language prompts.  

3.2 Algorithmic Enhancements to In-Context Learning  
Guided by the Bayesian interpretation, we propose two improvements:  
(1) Demonstration Re-weighting: incorporate an explicit log-prior term into attention scores,  
$$
\tilde\alpha_{*j}\;\propto\;\exp\Bigl((q_*^\top k_j + \log w_j)/\tau\Bigr)\,,
$$  
where $w_j$ reflects the model’s prior confidence in example $(x_j,y_j)$ (estimated via cloze negative-log-likelihood).  
(2) Adaptive Kernel Bandwidth: learn a context-dependent temperature $\tau(C)$ that minimizes a proxy for validation loss on an held-out subset of the demonstrations.  

Pseudocode for modified ICL predictor:  
```
Input: demonstration set C={(x_i,y_i)}_{i=1}^n, query x_*
Compute queries q_i,k_i,v_i for all i=1..n and q_* for x_*
Estimate prior weights w_i = exp(−ℓ_prior(x_i,y_i))
Compute τ = softplus( g_φ(C) )    # bandwidth network
For j in 1..n:  s_j = (q_*·k_j + log w_j)/τ  
α_j = exp(s_j) / ∑_ℓ exp(s_ℓ)
Return  ŷ = decode( ∑_j α_j · v_j )  
```
Here, $g_φ$ is a lightweight neural network trained on a small held-out set of tasks to predict optimal temperature.  

3.3 Experimental Design  
Datasets and Tasks  
• Synthetic regression tasks: 1-D and 2-D function families (polynomial, sinusoidal) drawn from known priors, to verify the kernel-Bayes approximation.  
• Standard few-shot benchmarks:  
  – Classification: SST-2, MNLI, CLR (Collie et al., 2024)  
  – Reasoning: GSM8K (Cobbe et al., 2021), MathQA (Amini et al., 2019)  
  – Graph tasks: CLINC150 graph classification (Li et al., 2025) with retrieval-augmented prompts.  
Models  
We will evaluate GPT-style transformers of varying sizes (125M, 1.3B, 7B, 30B parameters) from the Llama/Pythia families, as well as a distilled 7B-parameter state-space baseline (Gu et al., 2023).  

Evaluation Metrics  
• Predictive accuracy / F1-score on classification tasks  
• Log-loss and perplexity on held-out examples  
• Calibration metrics (Brier score, expected calibration error)  
• Sample complexity: number of demonstrations $n$ vs. performance  
• Correlation between empirical attention weights $\alpha_{*j}$ and theoretical kernel weights  
• Computational overhead: additional FLOPs and latency induced by demonstration re-weighting and temperature network  

Ablations  
– Remove prior weighting ($w_j=1$) to isolate its contribution  
– Use fixed vs. learned temperature  
– Vary prompt ordering and grouping to test robustness to demonstration permutation  

3.4 Validation of Theoretical Predictions  
For each synthetic task, we will compare the empirical risk $L(\hat p_n)$ with the predicted $O(n^{-2/(2+d)})$ decay rate by fitting a power-law curve to observed losses. For classification tasks, we will measure how the model’s performance deviates from the conditional Bayes risk as a function of demonstration diversity and representation quality, using metrics on the embedding manifold (e.g., intrinsic dimensionality estimates).  

4. Expected Outcomes & Impact  
Expected Outcomes  
1. A rigorous mathematical framework that casts in-context learning in transformers as an implicit nonparametric Bayesian inference mechanism, with precise identification of the conditions under which the attention kernel approximates the posterior predictive.  
2. Finite-sample generalization bounds for ICL that depend on model parameters (e.g., dimension $d$, Lipschitz constants) and data properties, thereby quantifying the number of context examples needed for a desired error level.  
3. Theory-driven algorithmic enhancements—demonstration re-weighting and adaptive bandwidth—accompanied by open-source implementations and empirical validation showing consistent improvements (5–10% relative gain) on few-shot benchmarks.  
4. Diagnostic tools for practitioners: guidelines for demonstration selection, prompt structure, and model scaling to optimize ICL in new tasks.  

Broad Impact  
By grounding in-context learning in a well-understood inferential paradigm, this work will:  
• Improve Efficiency: enable practitioners to achieve strong few-shot performance with fewer examples and reduced compute by informing demonstration design and temperature tuning.  
• Enhance Transparency: provide interpretable insight into attention weight patterns as approximations to Bayesian posteriors, fostering trust and accountability in high-stakes deployments.  
• Guide Model Design: inform future foundation model architectures (e.g., dedicated “ICL heads,” improved key/query embeddings) that are tailored for implicit task adaptation, potentially reducing overall parameter count.  
• Advance Responsible AI: by clarifying when ICL may fail (e.g., in low-density regions of the embedding space), our framework supports risk assessment and safe deployment protocols in domains such as legal text analysis or medical decision support.  

Overall, this proposal aligns with the TF²M workshop themes—efficiency, responsibility, and principled foundations—by leveraging statistical learning theory and information-theoretic tools to deepen our understanding of emergent capabilities in foundation models. We anticipate that the theoretical and empirical contributions will catalyze a new line of research into adaptive, transparent, and resource-efficient foundation models.