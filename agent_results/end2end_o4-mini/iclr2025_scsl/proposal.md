1. Title  
SpurGen: A Synthetic Multimodal Benchmark for Detecting and Mitigating Spurious Correlations  

2. Introduction  

Background  
Deep learning models often achieve high in-distribution accuracy by latching onto spurious correlations—“shortcuts” in the data that correlate with labels but do not reflect underlying causal structure. Such reliance on spurious signals undermines model robustness, generalization to out-of-distribution (OOD) settings, and fairness, especially when under-represented groups are involved. Existing benchmarks for spurious correlation detection and robustness (e.g., Waterbirds, CivilComments-WILDS) rely on human-annotated group labels or naturally occurring biases. These approaches are limited in scope: they cover only a few known correlations, require expensive annotation, and cannot explore unknown or controllable spurious channels systematically.

Research Objectives  
This proposal introduces SpurGen, a *modular synthetic benchmark* that:  
• Generates paired data (images, captions, and—optionally—audio, time series, graphs) with *configurable* spurious channels.  
• Enables systematic stress-testing of pre-trained and fine-tuned models by controlling spurious strength and interaction.  
• Defines rigorous evaluation metrics—*Spurious Sensitivity Score* and *Invariance Gap*—to quantify a model’s reliance on each spurious channel.  
• Ships evaluation scripts for popular robustification methods (e.g., Invariant Risk Minimization, contrastive augmentation, adversarial debiasing).  

Significance  
By moving beyond naturally occurring biases and manual group labels, SpurGen will:  
• Provide a *unified* platform to evaluate and compare robustness across modalities and model types (vision, vision–language, audio, etc.).  
• Accelerate progress on methods that truly learn causal relationships rather than statistical shortcuts.  
• Offer insights into which architectural or optimization choices exacerbate or mitigate shortcut learning, guiding future model design.

3. Methodology  

3.1 Overview of SpurGen Generator  
SpurGen is a *parameterized data generator* that synthesizes multimodal examples $(x, y)$ with multiple orthogonal spurious channels $c \in \mathcal{C}$. For each class $y \in \{1,\dots,K\}$, we assign each channel $c$ a spurious attribute $s_c\in S_c$ with probability $p_c$. The generator outputs:  
  – An image $x_{\text{img}}$ rendered via procedural graphics.  
  – A caption $x_{\text{text}}$ constructed from templates.  
  – (Optional) A synthetic audio clip $x_{\text{audio}}$ or time-series $x_{\text{ts}}$.  

3.1.1 Image Synthesis  
• Base objects: $K$ 3D CAD models or vector shapes.  
• Spurious channels:  
  – Background texture (e.g., stripes vs. polka-dots).  
  – Object color hue (e.g., red vs. blue).  
  – Lighting direction or intensity.  
• Implementation: Blender or Unity with scripting API.  
• Control variable: for channel $c$, choose attribute $s_c$ with probability $p_c(y)$, allowing alignment (high $p_c$) or misalignment (low $p_c$).

3.1.2 Caption Generation  
• Template grammar: “A photo of a [color] [object] with a [background] background.”  
• Spurious textual channels:  
  – Sentence template choice (e.g., active vs. passive voice).  
  – Synonym usage (e.g., “image” vs. “picture”).  
• Implementation: simple fill-in templates or lightweight GPT prompting with fixed seeds for reproducibility.

3.1.3 Extending to Additional Modalities  
We plan two pilot extensions:  
 1. Audio: generate sine waves representing class labels, add spurious white-noise patterns or environmental sounds.  
 2. Time series: generate base sinusoidal signals and overlay spurious correlated wavelets.  

3.2 Formalizing Spurious Control  
For each example $i$, let $s_{i,c}$ denote the attribute of channel $c$. We define a *spurious alignment function*  
$$
p_c(y) = \Pr(s_{i,c}=s^+ \mid y_i=y)\,,
$$  
where $s^+$ is the “positive” spurious attribute. By varying $p_c(y)$ across $\{0.1,0.5,0.9\}$, we create regimes of weak, medium, and strong spurious alignment.

3.3 Evaluation Metrics  
3.3.1 Standard and Worst-Group Accuracy  
Let $\hat y_i = \arg\max_y f_\theta(x_i)[y]$. We compute:  
  • Overall accuracy  
$$
\mathrm{Acc} = \frac1N\sum_{i=1}^N \mathbf{1}(\hat y_i = y_i)\,.
$$  
  • Worst-group accuracy over groups defined by joint spurious assignments (e.g., $(s_{c1},s_{c2})$).  

3.3.2 Spurious Sensitivity Score (SSS)  
For channel $c$, we define  
$$
\mathrm{SSS}_c = \mathbb{E}_{(x,y)}\Bigl|f_\theta(x)[y] - f_\theta\bigl(\mathrm{Shuffle}_c(x)\bigr)[y]\Bigr|\!,
$$  
where $\mathrm{Shuffle}_c(x)$ replaces $s_{c}$ in $x$ with an attribute sampled uniformly from $S_c$. A high $\mathrm{SSS}_c$ indicates strong reliance on $c$.

3.3.3 Invariance Gap (IG)  
We measure the loss gap between controlled and uncontrolled spurious settings:  
$$
\mathrm{IG} = \mathbb{E}_{(x_{\mathrm{ctrl}},y)}\bigl[\ell(f_\theta(x_{\mathrm{ctrl}}),y)\bigr] \;-\; \mathbb{E}_{(x_{\mathrm{unc}},y)}\bigl[\ell(f_\theta(x_{\mathrm{unc}}),y)\bigr].
$$  
Here, $x_{\mathrm{ctrl}}$ has fixed spurious attributes and $x_{\mathrm{unc}}$ has randomized ones. Lower IG implies greater invariance.

3.4 Robustification Baselines  
We will implement and benchmark:  
 1. Invariant Risk Minimization (IRM)  
    – Objective:  
    $$
    \min_{\theta,w}\;\sum_{e\in\mathcal{E}}R^e(\theta)\quad\text{s.t.}\quad w\in\arg\min_{w'}\sum_{e\in\mathcal{E}}R^e(w'\cdot\theta)\,.
    $$  
    – Practical penalty form (Arjovsky et al., 2019).  
 2. Group Distributionally Robust Optimization (Group-DRO)  
    – Minimize worst-group risk: $\min_\theta\max_{g} R^g(\theta)$.  
 3. Adversarial Feature Debiasing  
    – Train a feature extractor $h_\theta$ and adversary $a_\phi$ to predict spurious $s_c$.  
    – Minimize  
    $$
    \min_{\theta}\max_{\phi}\;\sum\nolimits_i\ell_{\mathrm{cls}}(f_\theta(x_i),y_i)\;-\;\lambda\;\ell_{\mathrm{adv}}\bigl(a_\phi(h_\theta(x_i)),s_{i,c}\bigr).
    $$  
 4. Contrastive Augmentation  
    – Generate positive pairs by permuting spurious channels, apply contrastive loss to align object-centric representations.

3.5 Experimental Protocol  
Datasets  
  • SpurGen-Image: 10 classes, 3 channels, $20\,000$ samples.  
  • SpurGen-Text: same classes with templated captions.  
  • Pilot SpurGen-Audio (optional).  

Models  
  • CLIP ViT-B/16 (pre-trained, zero-shot).  
  • BLIP-2 (fine-tuned).  
  • LLaVA (multimodal LLM).  

Training & Evaluation  
  – Split: 60% train, 20% validation, 20% test, ensuring each spurious regime appears equally.  
  – Hyperparameters: batch size 64, AdamW, learning rate $10^{-5}$–$10^{-4}$, 10 epochs.  
  – For each method, sweep $\lambda\in\{0.1,1.0,10\}$ for adversarial and IRM penalties.  
  – Report mean $\pm$ std over 5 seeds.  
  – Statistical significance: paired t-test ($p<0.05$) comparing each method to ERM baseline.

Ablations  
  • Vary spurious alignment $p_c(y)$ to assess method sensitivity.  
  • Remove one channel at a time to isolate effects.  
  • Compare synthetic spur vs. real-data spur (e.g., Waterbirds).

4. Expected Outcomes & Impact  

Expected Outcomes  
• A public release of the *SpurGen* codebase and data generator, supporting images, captions, and optional audio/time-series.  
• A suite of evaluation scripts computing accuracy, worst-group accuracy, SSS, and IG.  
• Benchmarks of ERM, IRM, Group-DRO, adversarial debiasing, and contrastive augmentation on SpurGen, with detailed ablations.  
• Insights into which spurious channels are most pernicious for current vision–language and multimodal models.  

Broader Impact  
SpurGen will:  
• Fill a critical gap by providing *controllable, extensible* benchmarks that go beyond hand-annotated biases.  
• Enable consistent comparison and stress-testing of new robustification methods across research groups.  
• Illuminate fundamental questions about how deep models learn shortcuts, guiding future architecture and optimizer design.  
• Lower the barrier to entry for researchers studying spurious correlation, thanks to a turnkey, open-source toolkit.  

Long-Term Vision  
By demonstrating the efficacy (or failure) of various defenses in a controlled setting, this work will steer the community toward methods that generalize *causally* rather than statistically. SpurGen’s synthetic approach paves the way for *adaptive benchmarks* that evolve as new spurious channels or modalities emerge, ensuring sustained progress in building robust, reliable AI systems.