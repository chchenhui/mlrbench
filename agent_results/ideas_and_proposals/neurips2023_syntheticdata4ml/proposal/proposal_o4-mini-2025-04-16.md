Title: PrivFairGen: Constrained LLM-Based Differentially Private and Fair Synthetic Tabular Data Generation

1. Introduction  
Background  
High-quality training data is the cornerstone of modern machine learning. Yet for many high-stakes domains—healthcare, finance, criminal justice—data scarcity, privacy regulations (e.g. HIPAA, GDPR), and entrenched biases pose severe barriers to dataset collection and open sharing. Tabular data in particular often contains sensitive attributes (age, race, medical codes) that cannot be published wholesale. At the same time, under-representation of minority groups in real datasets can lead to models that perpetuate or amplify societal inequities. Synthetic data generation offers a flexible remedy: once a generative model is trained, one can sample arbitrarily many “fake” records to augment training sets, without exposing individual records.  

Recent advances in deep generative modeling—for example DP-TBART (Castellon et al. 2023), TableDiffusion (Truda 2023), and DP-2Stage (Afonja et al. 2024)—demonstrate that both transformer-based and diffusion-based methods can achieve strong data utility under a differential privacy (DP) guarantee. Concurrently, fairness-aware approaches such as Fairness-Aware Synthetic Data Generation (Johnson & Lee 2024) and fair VAE methods (Green & Black 2024) show that conditional generation can mitigate biases. However, most existing work tackles privacy and fairness separately: generative models are often optimized purely for fidelity, with privacy or fairness injected only as a post-hoc constraint. Moreover, large language models (LLMs) have demonstrated remarkable capacity for high-fidelity sequence generation and controllable sampling, but few studies (Tran & Xiong 2024; Red & Blue 2023) have fully exploited LLMs for tabular data under joint DP and fairness constraints.  

Research Objectives  
We propose PrivFairGen, a unified framework for synthesizing tabular data that simultaneously satisfies rigorous differential privacy guarantees and formal group fairness constraints by leveraging a pre-trained LLM. Our key objectives are:  
• To design a two-stage fine-tuning procedure that (a) adapts an LLM’s text modeling capability to tabular data syntax, then (b) applies DP-SGD with precise privacy accounting.  
• To incorporate fairness penalties—targeting demographic parity and equalized odds—directly into the LLM’s fine-tuning loss or decoding phase, yielding synthetic data that corrects under-representation.  
• To develop a decoding-time constraint mechanism that enforces statistical parity across sensitive groups without harming overall data utility.  
• To empirically validate the method on several benchmark tabular datasets, comparing against state-of-the-art DP- and fairness-aware generators across utility, privacy, and fairness metrics.  

Significance  
PrivFairGen bridges the gap between high-fidelity LLM-based synthesis and the ethical imperatives of privacy and fairness. By developing an end-to-end framework that enforces constraints throughout training and sampling, we enable practitioners and data stewards in regulated domains to generate large-scale synthetic datasets that:  
• Protect individual privacy with quantified DP guarantees (ε,δ).  
• Mitigate bias against protected subgroups.  
• Maintain downstream ML performance comparable to non-private, non-fair generative baselines.  
This work promises to accelerate trustworthy ML adoption in domains where data sharing is currently infeasible.  

2. Methodology  
Overview  
PrivFairGen proceeds in three phases: (1) Tabular formatting and pseudo-data pre-training, (2) DP-SGD fine-tuning with fairness-aware loss, (3) Constraint-guided decoding for final data generation. Figure 1 (omitted) depicts the pipeline.  

2.1 Data Representation & Preprocessing  
• Input data D_real consists of n records, each a vector x∈ℝ^d with m continuous features and k categorical features (including a sensitive attribute S∈{0,1}).  
• We convert each record to a token sequence “field1:val1 | … | field_d:val_d” using delimiter tokens. Continuous features are quantized or represented via a fixed‐precision string. Categorical values map to tokens.  
• Define a vocabulary V sufficiently large to encode all attribute names and values. Add special tokens <BOS>, <EOS>.  

2.2 Stage 1: Non-Private Pseudo-Data Pre-Training  
• To accelerate adaptation, we construct a large pseudo‐tabular corpus D_pseudo by sampling marginal distributions and feature pairwise copulas from D_real. D_pseudo contains N≫n synthetic records without privacy concerns.  
• Fine-tune the pre-trained LLM (e.g. GPT-2 small) on D_pseudo with the negative log-likelihood objective:  
  $$  
  \mathcal{L}_{\mathrm{NLL}}(\theta) = -\frac{1}{N}\sum_{x\in D_{\mathrm{pseudo}}}\sum_{t=1}^{T}\log p_\theta(x_t\,|\,x_{<t})\,.  
  $$  
• Use standard AdamW optimization for 1–2 epochs to teach the model tabular syntax.  

2.3 Stage 2: DP-SGD Fine-Tuning with Fairness Penalty  
We now fine-tune on D_real under DP constraints and fairness objectives.  

2.3.1 Differential Privacy Mechanism  
• At each iteration t with mini‐batch B_t of size b, compute per‐example gradients g_i = ∇_θℓ(θ; x_i).  
• Clip each gradient:  
  $$  
  \bar g_i = \frac{g_i}{\max\bigl(1,\tfrac{\|g_i\|_2}{C}\bigr)}\,,  
  $$  
  where C is the clipping norm.  
• Aggregate and add noise:  
  $$  
  \tilde g_t = \frac{1}{b}\sum_{i\in B_t}\bar g_i \;+\;\mathcal{N}\bigl(0,\,\sigma^2C^2I\bigr)\,.  
  $$  
• Update parameters: θ_{t+1} = θ_t −η\tilde g_t.  
• Track the privacy loss (ε,δ) using the moments accountant (Abadi et al., 2016).  

2.3.2 Fairness-Aware Loss  
We incorporate a penalty term targeting demographic parity:  
• Let Ŷ be the predicted label for a downstream classifier f trained on synthetic data; let S∈{0,1} be the sensitive attribute.  
• Define demographic parity gap:  
  $$  
  \Delta_{DP}(\tilde D) = \Bigl|P_{x\sim\tilde D}[f(x)=1\mid S=0] - P_{x\sim\tilde D}[f(x)=1\mid S=1]\Bigr|\,.  
  $$  
• Similarly, equalized odds gap Δ_EO measures difference in true positive rates across groups.  
• During fine-tuning, approximate Δ_DP on the current batch by sampling k rows per group and computing a differentiable estimate using the model’s next‐token probabilities or a small classifier head attached to the LLM.  
• Total loss per batch:  
  $$  
  \mathcal{L}_t = \underbrace{\tfrac{1}{b}\sum_{i\in B_t}\mathcal{L}_{\mathrm{NLL}}(x_i)}_{\text{fidelity}} \;+\;\lambda_{\mathrm{fair}}\;\max\bigl(\widehat\Delta_{DP},\,\tau\bigr)\,,  
  $$  
  where λ_fair controls the trade-off and τ is a small slack threshold.  
• Backpropagate through this combined loss under the DP-SGD mechanism above.  

2.4 Stage 3: Constraint-Guided Decoding  
Once fine-tuning completes, we sample synthetic records via a customized decoding algorithm:  
1. At time step t, compute next‐token logits ℓ_t = f_θ(x_{<t}).  
2. Add calibrated Gaussian noise to logits: ℓ'_t = ℓ_t + 𝒩(0,σ_dec^2I) to further obscure memorized patterns.  
3. Perform top-k filtering, retaining the k most likely tokens.  
4. Sample a candidate continuation token x_t from the filtered distribution.  
5. If the partial sequence so far would, upon completion, violate a fairness constraint (e.g. generate too many records of S=1 relative to S=0), discard and resample up to R times.  
6. End when <EOS> is generated or maximum length reached.  

By combining noise injection and constrained sampling, we ensure the final dataset both upholds DP properties and meets the desired parity constraints.  

2.5 Experimental Design  
Datasets  
• Adult Income (UCI), COMPAS recidivism, MIMIC-III clinical records (after de-identification), and a synthetic healthcare claims dataset.  
Metrics  
1. Utility  
  – Classification accuracy, F1-score on a downstream target (e.g. income>50K, recidivism) for models trained on synthetic vs. real data.  
  – Statistical similarity: Kolmogorov–Smirnov distance for continuous features, Jaccard index for categorical marginals.  
2. Privacy  
  – (ε,δ) from the moments accountant.  
  – Empirical membership inference attack accuracy on held-out real records.  
3. Fairness  
  – Demographic parity difference Δ_DP, equalized odds difference Δ_EO on the synthetic dataset.  
Baselines  
• DP-TBART, DP-LLMTGen, DP-2Stage, TableDiffusion, fairness-aware VAE, DP-GAN (White & Brown 2023).  
Experimental Protocol  
• Vary privacy budgets ε∈{0.5,1,3,8} with δ=10⁻⁵.  
• Vary fairness weights λ_fair∈{0,0.1,1.0}.  
• Perform 5 random splits for train/test and report mean±std.  
Implementation  
• PyTorch + HuggingFace Transformers.  
• Opacus for DP-SGD.  
• Code and synthetic data released under an open-source license.  

3. Expected Outcomes & Impact  
We anticipate that PrivFairGen will deliver:  
• High-utility synthetic tabular datasets whose downstream classification accuracy is within 5% of models trained on the real data, even under ε=1.  
• Rigorous DP guarantees (e.g. (ε=1, δ=10⁻⁵)) validated by both theoretical accounting and empirical attacks showing near‐random membership inference.  
• Marked reduction in group bias, achieving Δ_DP<0.02 and Δ_EO<0.03, outperforming unconstrained DP generators by 30–50% on fairness metrics.  

Impact  
By unifying privacy and fairness in an LLM-based synthesis framework, PrivFairGen will:  
• Enable researchers and institutions to share high-fidelity synthetic data without exposing sensitive records.  
• Provide a reproducible baseline for integrating fairness constraints into generative AI.  
• Facilitate the development of equitable ML systems in regulated domains, accelerating trust and adoption.  

In sum, PrivFairGen represents a significant step toward ethical, privacy-preserving, and unbiased synthetic data generation for tabular domains.