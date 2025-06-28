### Title  
**Iterative Generative Design with Active Learning for Optimized Antibody Affinity Maturation**

---

### Introduction  
**Background**  
Antibodies play a pivotal role in therapeutics, diagnostics, and industrial applications due to their ability to bind specific antigens with high affinity. However, traditional methods for antibody affinity maturation, such as directed evolution and combinatorial screening, are laborious and costly. These approaches often explore vast sequence spaces inefficiently, relying on random mutagenesis and high-throughput screening. While advancements in deep learning have enabled promising computational tools for antibody design, such as generative models that propose affinity-improved variants, their integration with experimental validation remains underdeveloped. Current in silico methods often prioritize benchmark performance over real-world applicability, leading to a disconnect between computational predictions and wet-lab feasibility.  

The emergence of generative machine learning models—such as ProteinMPNN for sequence design and ESM-IF for evolutionary modeling—has demonstrated the ability to generate functional antibody sequences. However, unguided sampling of variants frequently results in low experimental success rates, as freshly generated candidates may lack physicochemical stability or fail to express in living systems. Concomitantly, active learning frameworks have shown promise in optimizing small-molecule drug discovery by iteratively refining models using targeted experimental feedback. Yet, their application to antibody affinity maturation remains limited, with few studies evaluating workflows that tightly integrate generative design, predictive modeling, and adaptive experimentation.  

**Research Objectives**  
This research aims to bridge the gap between computational antibody design and experimental validation by developing an iterative active learning framework that combines:  
1. **Generative sequence modeling** to propose high-quality antibody variants.  
2. **Predictive models** for binding affinity and structural consistency.  
3. **Active learning strategies** to prioritize experiments on the most promising variants.  
4. **Closed-loop feedback** to refine models using experimental results, iteratively improving design accuracy.  

The primary goal is to maximize binding affinity while minimizing experimental costs, achieving a 2–3-fold reduction in required measurements compared to traditional methods like yeast display or phage display screening.  

**Significance**  
This work addresses critical challenges in therapeutic development, where rapid design of high-affinity antibodies is essential for combating diseases such as cancer and infectious agents. By integrating generative models with experimental validation, the proposed framework will:  
- Accelerate antibody discovery pipelines, reducing timelines from months to weeks.  
- Lower costs by focusing experiments on the most probable candidates.  
- Enable systematic exploration of sequence-structure-function relationships in antibodies.  
- Provide a reusable blueprint for other biomolecular design tasks, such as enzyme optimization or vaccine development.  
Moreover, bridging computational and experimental paradigms aligns with the GEM workshop’s mission to advance rationally designed biomolecules with real-world impact.  

---

### Methodology  
**Overview**  
The proposed framework operates as a closed-loop system (Figure 1), integrating four stages:  
1. **Generative Sequence Modeling**: Propose candidate antibody variants.  
2. **Predictive Modeling**: Estimate binding affinity and structural quality.  
3. **Active Learning**: Select variants for experimental validation.  
4. **Experimental Validation**: Quantify binding affinity using lab assays.  
5. **Feedback Loop**: Retrain models using newly acquired experimental data.  

This iterative process repeats until affinity plateaus or experimental budgets are exhausted.  

**Generative Sequence Modeling**  
Two complementary models generate plausible antibody variants:  
1. **ProteinMPNN**: A graph neural network (GNN) that designs sequences optimized for a target structure by minimizing side-chain repacking energy:  
   $$\mathcal{L}_{\text{structure}} = \frac{1}{N} \sum_{i=1}^{N} \left\| E_{\text{pred}}(s_i | x) - E_{\text{native}} \right\|_2^2$$
   Here, $E_{\text{pred}}$ is the predicted energy for sequence $s_i$ given structure $x$, and $E_{\text{native}}$ is the native stability target.  
2. **ESM-IF**: A transformer-based model that infers evolutionary constraints to ensure sequence diversity:  
   $$P(s | x) = \prod_{i=1}^{L} P(s_i | x, s_1, \dots, s_{i-1})$$
   This autoregressive formulation ensures variants adhere to evolutionary patterns observed in antibody repertoires.  

Candidate variants are resampled until structural validity (via DBD4 similarity) and sequence diversity thresholds (CDR-H3 RMSD > 0.5 Å) are satisfied.  

**Predictive Modeling**  
Two models score candidates:  
1. **Affinity Predictor**: A geometric deep learning model (e.g., DeepHIT or ABlooper) that estimates binding free energy ($\Delta G$) using antibody-antigen complex structures.  
   $$\hat{\Delta G} = f_{\theta}(x_{\text{AB}}, x_{\text{Ag}})$$ where $x_{\text{AB}}$ and $x_{\text{Ag}}$ are structural coordinates.  
2. **Uncertainty Quantifier**: An ensemble of sequence-based convolutional neural networks (CNNs) estimates prediction confidence:  
   $$\hat{u} = \frac{1}{T} \sum_{t=1}^{T} \sigma_t(f_t(s))$$  
   With $T=5$ bootstrap models, $\hat{u}$ represents epistemic uncertainty.  

**Active Learning with Hybrid Acquisition**  
To balance **exploration** and **exploitation**, variants are selected via a weighted acquisition function:  
$$a(x) = \alpha \cdot \text{EI}(x) + (1-\alpha) \cdot u(x)$$  
where:  
- $ \text{EI}(x) = (y_{\text{top}} - \mu(x)) \Phi(z) + \sigma(x) \phi(z) $ is expected improvement ($z = \frac{y_{\text{top}} - \mu}{\sigma}$),  
- $u(x)$ is predictive uncertainty (network-based),  
- $\alpha \in [0,1]$ tunes exploration-exploitation trade-offs.  

Candidates are optimized using beam search to maximize $a(x)$ while satisfying physicochemical constraints (e.g., charge, hydrophobicity within native ranges).  

**Experimental Validation**  
Chosen variants undergo yeast display followed by surface plasmon resonance (SPR) to quantify $K_D$ (dissociation constant). Key steps:  
1. **Library Construction**: Generated sequences are cloned into a yeast display vector (pCTCON2), expressed on cell surfaces.  
2. **FACS Sorting**: Cells are labeled with fluorescently tagged antigen and sorted into bins based on fluorescence intensity (5 bins, FACS Aria III).  
3. **SPR Validation**: Top 100 FACS candidates are expressed in E. coli and tested on a Biacore T200 instrument (3 cycles, 25°C).  

**Training Protocol**  
1. **Initialization**:  
   - Start with 200 pre-trained variants from the SabDab database (filtered to $K_D < 100$ nM).  
   - Pre-train ProteinMPNN and ESM-IF on 10,000 human VDJ sequences.  
2. **Iteration Loop**:  
   - At each round $t$, generate 1000 variants.  
   - Select 50 via acquisition function.  
   - Acquire SPR labels for 30 high-confidence candidates ($\alpha=0.7$) and 20 exploratory variants ($\alpha=0.3$).  
   - Append labeled data to $\mathcal{D}_t$.  
   - Retrain $f_{\theta}$ and $f_t(s)$ on $\mathcal{D}_t$ using transfer learning.  

**Evaluation Metrics**  
1. **Computational Metrics**:  
   - **Sequence Designability (Q)**: $Q = 1 - \frac{1}{L} \sum_{i=1}^{L} \| \hat{x}_i - x_i \|_2$ (structural similarity to target).  
   - **Novelty**: $\text{Coverage}_{\text{CDR-H3}} = \frac{\text{Unique Motifs in Top-K}}{\text{All Motifs in Training Set}}$.  
   - **Diversity**: Average Hamming distance between top 100 variants.  
2. **Experimental Metrics**:  
   - **Median $K_D$ Improvement**: $\Delta K_D = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} (K_D^{(t)} - K_D^{(0)})$.  
   - **Hit Rate**: \% of variants with $K_D < 1$ nM.  
3. **Benchmarking**:  
   - Against RosettaDesign and single-step DeepHIT baselines.  
   - Using 5-fold cross-validation on PDBbind data (2020 curated set).  

---

### Expected Outcomes & Impact  
**Expected Outcomes**  
1. **Improved Affinity**: Achieve $K_D < 0.1$ nM for ≥70% of antibody-antigen pairs after 5 iterations, outperforming single-step ML baselines (Rosetta: 30%, DeepHIT: 50%).  
2. **Experimental Efficiency**: Reduce the number of required measurements by 60% compared to naive screening (e.g., 250 labeled variants vs. 1000).  
3. **Generalization**: Demonstrate robustness across diverse antigens (e.g., SARS-CoV-2 Spike, HER2, hIgE).  
4. **Model Insights**: Reveal design patterns (e.g., CDR-H3 tyrosine clusters, framework stability hotspots).  

**Validation Plan**  
- **Blind Experiments**: Holdout testing on anti-dsDNA antibodies (known for polyreactivity).  
- **X-ray Crystallography**: Validate structural congruence of top variants using Synchrotron facilities (e.g., SPring-8).  
- **Benchmark Release**: Publish an open-source antibody affinity maturation benchmark with experimental data.  

**Long-Term Impact**  
- **Therapeutic Development**: Accelerate discovery of antibodies targeting challenging pathogens (e.g., HIV, malaria).  
- **Model Development**: Inspire generalized frameworks for protein-antibody codesign using latent space prioritization.  
- **Closing the Loop**: Establish a paradigm where ML actively guides hypothesis formation, improving both prediction accuracy and experimental utility.  

This work aligns with the GEM workshop’s goals by validating that generative models, when coupled with iterative experimental feedback, can produce actionable insights for real-world biology. The proposed methodology directly addresses literature gaps in data scarcity (via iterative retraining) and model generalization (by preserving evolutionary realism), while pioneering active learning in antibody design—an area where systematic exploration remains scarce. Future work will extend this framework to bispecific antibodies and multi-target vaccines.  

---  

**Word Count**: ~1,950