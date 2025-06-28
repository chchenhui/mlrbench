1. Title  
Probing Pre-Training Data Influence on Emergent Abilities via Representation Perturbation  

2. Introduction  
Background  
Foundation models (FMs) such as GPT-3, BERT, SimCLR and CLIP have exhibited a range of surprising “emergent” capabilities—chain-of-thought reasoning, in-context learning, and few-shot generalization—that are not present in smaller scale models. While scale, data volume, and architecture each play a role, we still lack a principled understanding of how *specific* subsets of the pre-training corpus contribute to these emergent phenomena. Recent work (Du et al. 2024; Wei et al. 2022) suggests that certain loss thresholds or data types may be critical triggers for these abilities, but we do not yet know *which* data are most responsible or *where* in the model those capabilities reside.  

Research Objectives  
1. Identify cohesive subsets of pre-training data (e.g., code, mathematical proofs, dialogues).  
2. Quantify the causal influence of each subset on emergent reasoning capabilities.  
3. Map those influences to subspaces of the model’s hidden representations.  
4. Derive practical guidelines for data curation to amplify or suppress specific emergent skills.  

Significance  
Understanding the data-to-capability pathway is crucial for:  
• Efficient Model Development—focus on high-value data and avoid wasteful scale-only regimes.  
• Risk Mitigation—identify and remove the data responsible for undesirable behaviors (e.g., misinformation).  
• Theoretical Insight—bridge the gap between empirical scaling laws and mechanistic explanations of emergent phenomena.  

3. Literature Review  
CHORUS (Kayali et al., 2023) demonstrates that foundation models can unify diverse data-management tasks, underscoring the power of large pre-training corpora. However, it does not dissect which data subsets drive particular capabilities.  
Du et al. (2024) analyze emergent abilities through the pre-training loss perspective, finding thresholds below which new skills appear. Their approach, while insightful, treats the loss holistically and does not isolate the role of distinct data types.  
Wei et al. (2022) catalogue a variety of emergent capabilities as scale increases, but they attribute them broadly to “scale” without a fine-grained account of data composition.  
Muppet (Aghajanyan et al., 2021) introduces pre-finetuning over multiple tasks to improve generalization. This highlights that *what* the model sees influences emergent representations, yet the specific impact of raw pre-training slices on emergent reasoning is still unknown.  
Key Gaps  
• No prior work has *causally* linked pre-training data subsets to emergent skills.  
• Representation-level perturbation methods have been applied to linguistic features but not to data-cluster subspaces.  
• A unified experimental framework for measuring the data-capability relationship is lacking.  

4. Methodology  
We propose a three-phase approach: (A) data clustering, (B) subspace construction & perturbation, and (C) empirical evaluation using causal mediation analysis.  

4.1 Data Clustering  
• Data Sources: Assemble representative corpora for each candidate subset:  
  – Code (GitHub Python+Java)  
  – Mathematical texts (arXiv math, StackExchange Math)  
  – Dialogues (OpenSubtitles, Reddit conversations)  
  – Prose (Wikipedia, newswire)  
• Embedding Extraction: For each document $d_i$ in subset $j$, compute a pooled sentence embedding via a pre-trained encoder $g(\cdot)$:  
  $$z_i^{(j)} = g(d_i)\in\mathbb{R}^p$$  
• Clustering Validation: Verify that subsets form well-separated clusters in the $z$-space via silhouette score and pairwise distance statistics.  

4.2 Representation Subspace Construction  
We target a pre-trained FM $f$ (e.g., LLaMA-7B). Let $h_\ell(x)\in\mathbb{R}^d$ denote the hidden activation at layer $\ell$ for input $x$.  
1. **Collect Activations**  
   For each subset $j$ and each $d_i\in D_j$, sample a set of prompts $x_{i,k}$ and record  
   $$H_j = \bigl[h_\ell(x_{i,k})\bigr]_{i,k}\in\mathbb{R}^{d\times N_j}$$  
2. **Principal Component Analysis (PCA)**  
   Compute the top $k$ components of $H_j$:  
   $$H_j = U_j \Sigma_j V_j^\top,\quad U_j\in\mathbb{R}^{d\times k}$$  
3. **Projection Operators**  
   Define the cluster subspace projector  
   $$P_j = U_j U_j^\top,\quad Q_j = I_d - P_j$$  
   where $P_j h$ captures the component of $h$ aligned with data subset $j$.  

4.3 Representation Perturbation & Causal Mediation  
We adapt *amnesic probing* and causal mediation analysis to our setting. For an emergent-task input $x$:  
1. **Baseline Activation**: $h_\ell = h_\ell(x)$.  
2. **Ablation**:  
   $$h_\ell^{(-j)} = Q_j\,h_\ell$$  
3. **Amplification** (optional):  
   $$h_\ell^{(+j)} = h_\ell + \alpha\,P_j\,h_\ell,\quad \alpha>0$$  
4. **Forward Pass**  
   Feed $h_\ell^{(\star)}$ through the remaining layers to produce output logits and predictions.  
5. **Outcome Measure**  
   On an emergent reasoning dataset $\mathcal{T}$ (e.g., GSM8K, BIG-Bench), define performance metric $m(\cdot)$ (accuracy, chain-of-thought success) and compute  
   $$\Delta m_j = m\bigl(f(x;\,h_\ell)\bigr) - m\bigl(f(x;\,h_\ell^{(-j)})\bigr)\,.$$  
6. **Causal Mediation**  
   Treat subset alignment $M_j(x)=P_j h_\ell(x)$ as a mediator between “treatment” $T_j$ (presence of data subset in pre-training) and outcome $Y=m(f(x))$. We estimate the *average causal mediation effect* (ACME) using standard approaches (Imai et al., 2010).  

4.4 Experimental Setup  
Datasets  
• Reasoning: GSM8K (grade school math), BIG-Bench logical and symbolic reasoning tasks.  
• Control tasks: commonsense QA (ARC), language modeling perplexity.  
Models  
• Primary: LLaMA-7B (open weights, mixed text-code pre-training).  
• Secondary: GPT-2 XL (smaller backbone for sanity check).  
Protocols  
1. For each task, sample $n$ inputs $\{x_i\}$.  
2. For each subset $j$, compute ablated outputs and record performance.  
3. Compare to a random‐subspace ablation of equal dimension $k$ as a control.  
4. Repeat with $\alpha$‐amplification experiments.  

4.5 Evaluation Metrics & Statistical Analysis  
• **Performance Drop**: $\Delta m_j$ as above.  
• **Effect Size**: Cohen’s $d_j = \frac{\Delta m_j}{\sigma(\Delta m_j)}$.  
• **Calibration Change**: Expected Calibration Error (ECE) difference.  
• **Statistical Significance**: Bootstrap 95% confidence intervals; paired t-tests with Bonferroni correction over $j$.  
• **Correlation Analysis**: Pearson correlation between subset size $|D_j|$ and importance $\Delta m_j$.  

5. Expected Outcomes & Impact  
Expected Outcomes  
• A *ranking* of pre-training data subsets by their causal contribution to emergent reasoning.  
• Characterization of *where* in the network these capabilities reside (layers $\ell$ with largest $\Delta m_j$).  
• Quantitative guidelines for *data curation*: e.g., “Including an extra 10M tokens of mathematical text yields a 2–3% absolute gain on GSM8K.”  
• Insights into amplification: whether selectively boosting subspace activations can *enhance* emergent skills without full re-training.  

Broader Impact  
• **Efficient FM Development**: Train smaller models on a targeted corpus to achieve desired reasoning capabilities, reducing compute and carbon footprint.  
• **Safety & Alignment**: Identify and remove data subsets that drive harmful or biased emergent behaviors, facilitating safer deployment.  
• **Theoretical Advancement**: Bridge empirical scaling laws and mechanistic theories by showing how data shapes representation geometry to induce capabilities.  
• **Tooling for Practitioners**: Open‐source code for subspace construction and perturbation, enabling the community to probe their own models.  

In sum, this proposal lays out a rigorous, causally grounded framework to illuminate the data roots of emergent abilities in foundation models. By perturbing learned representations along interpretable, data-derived axes, we will advance both theory and practice toward more efficient, controllable, and safe AI systems.