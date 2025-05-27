Title  
Physics-Constrained Multimodal Transformer for Robust Materials Property Prediction from Sparse Data  

1. Introduction  
1.1 Background  
Recent advances in artificial intelligence (AI) have revolutionized fields such as natural language processing, computer vision and drug discovery. Yet, AI-driven materials science lags behind these adjacent areas, despite its enormous potential to accelerate the design and discovery of new materials. A key bottleneck is the multimodal and often incomplete nature of materials data: synthesis recipes, characterization images (SEM, TEM), diffraction patterns (XRD), spectroscopy (IR, Raman), and computed properties (DFT, MD) must be fused. Moreover, fundamental physical laws (e.g., phase equilibria, conservation laws, crystallographic symmetry) are only partially encoded in existing data‐driven models, leading to poor generalization when data are sparse.  

1.2 Research Objectives  
We propose to develop a Physics-Constrained Multimodal Transformer (PC-MMT) that  
• tokenizes and embeds multiple modalities in a shared latent space,  
• employs cross-attention to fuse modalities even when some are missing,  
• injects known physical constraints as soft biases in the attention mechanism and loss function, and  
• generates physically plausible property predictions and material candidates under data sparsity.  

1.3 Significance  
By explicitly integrating domain knowledge into a flexible transformer architecture, PC-MMT addresses two critical “AI4Mat” challenges:  
• Managing multimodal, incomplete data via modality-aware embeddings and missing‐data‐robust cross-attention.  
• Enforcing physical plausibility through constraint-informed attention layers and regularization.  
This approach promises improved generalization, interpretability and reliable hypothesis generation—key requirements for real-world materials discovery pipelines.  

2. Methodology  
Our research design comprises five main components: data collection and preprocessing, model architecture, physics-constrained learning objectives, training and optimization, and experimental validation.  

2.1 Data Collection & Preprocessing  
Datasets:  
• Public databases: Materials Project, AFLOW, OQMD—tabular formation energies, band gaps, elastic constants.  
• Characterization repositories: NOMAD, open XRD, SEM/TEM image archives.  
• In-house curated data: synthesis parameters (temperature, time, precursors), XRD patterns and corresponding micrographs.  

Preprocessing steps:  
1. Standardize units and nomenclature.  
2. Align records across modalities via unique sample IDs.  
3. Impute trivial missing values (e.g. default synthesis temperature) when appropriate; mark hard‐missing entries.  
4. Augment the training set with synthetic missing‐modality scenarios: randomly drop modalities at rates $p\in\{0.0,0.3,0.5,0.7\}$ to evaluate robustness.  

2.2 Model Architecture  
2.2.1 Modality-Specific Tokenization  
Each input sample consists of $M$ modalities. For modality $m$, we define a tokenizer $T_m(\cdot)$ that converts raw data into a sequence of tokens:  
• Tabular data (composition, processing): split into element‐property pairs, then map to embedding vectors of dimension $d$.  
• Spectra (XRD, Raman): segment continuous curves into windows, encode intensities as tokens.  
• Images (SEM/TEM): patchify as in Vision Transformer (ViT) [Vaswani et al., 2017; Dosovitskiy et al., 2021].  

2.2.2 Shared Embedding Space  
All tokens, regardless of modality, are projected into a common $d$-dimensional space via learned linear layers $E_m$. Positional encodings are added to maintain order in sequences.  

2.2.3 Cross-Attention Fusion with Missing-Modality Handling  
We stack $L$ transformer layers. In layer $\ell$, for each head $h\in\{1,\dots,H\}$ we compute:  
$$Q^{(\ell,h)} = Z^{(\ell-1)}W_Q^{(\ell,h)},\quad K^{(\ell,h)} = Z^{(\ell-1)}W_K^{(\ell,h)},\quad V^{(\ell,h)} = Z^{(\ell-1)}W_V^{(\ell,h)},$$  
where $Z^{(\ell-1)}\in\mathbb R^{N\times d}$ is the token matrix from the previous layer ($N$ total tokens across all available modalities) and $W_{Q,K,V}^{(\ell,h)}\in\mathbb R^{d\times d_h}$.  

To gracefully handle missing modalities, we introduce a binary mask $M_{\text{miss}}\in\{0,-\infty\}^{N\times N}$ that zeros out attention scores corresponding to absent tokens. The standard scaled dot-product attention is modified to incorporate both missing‐modality and physics constraints (see Section 2.3):  
$$
\mathrm{Attention}^{(\ell,h)}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_h}} + M_{\text{miss}} + M_{\text{phys}}\Bigr)V,
$$  
with $M_{\text{phys}}$ described below. The multi-head output is concatenated and projected:  
$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}_{h=1}^H\bigl(\mathrm{Attention}^{(\ell,h)}\bigr)W_O^{(\ell)},\quad W_O^{(\ell)}\in\mathbb R^{Hd_h\times d}.
$$  

2.2.4 Physics-Informed Attention Bias ($M_{\text{phys}}$)  
We encode known physical constraints as soft biases in the attention logits. Examples:  
• Phase diagram compatibility: pairs of composition tokens that violate known phase stability carry a negative bias $\beta_{\text{phase}}<0$.  
• Conservation laws: tokens representing input and output species in synthesis are encouraged to attend via positive bias $\beta_{\text{cons}}$.  
• Crystallographic symmetries: spatial‐token attention is aligned with known lattice symmetry operations.  

Formally, if tokens $i,j$ should be discouraged (resp. encouraged) to interact, we set  
$$
(M_{\text{phys}})_{ij} = 
\begin{cases}
\beta_-<0, & (i,j)\in\mathcal{D}_{\text{violate}},\\
\beta_+>0, & (i,j)\in\mathcal{E}_{\text{enforce}},\\
0,        & \text{otherwise.}
\end{cases}
$$  

2.3 Learning Objective  
We train PC-MMT end-to-end to predict a set of target material properties $y\in\mathbb R^T$ (e.g., formation energy, band gap, elastic modulus). Let $\hat y$ be the model output. The loss comprises:  
1. Prediction loss (regression or classification):  
   $$\mathcal{L}_{\mathrm{pred}} = \frac1T\sum_{t=1}^T \ell\bigl(\hat y_t,y_t\bigr),$$  
   where $\ell$ is mean squared error (MSE) for regression or cross-entropy for classification tasks.  
2. Physics constraint penalty: encourage satisfaction of soft constraints beyond attention bias. For each constraint $c$, define a differentiable measure $g_c(Z)$ (e.g., deviation from stoichiometry balance):  
   $$\mathcal{L}_{\mathrm{phys}} = \sum_{c}\alpha_c\,g_c(Z),$$  
   with weights $\alpha_c>0$.  

Overall:  
$$
\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \lambda\,\mathcal{L}_{\mathrm{phys}},
$$  
where $\lambda$ balances predictive accuracy and physical plausibility.  

2.4 Training & Optimization  
• Initialization: Transformer weights are initialized as in [Vaswani et al., 2017].  
• Optimizer: AdamW with learning rate warm-up and cosine decay.  
• Mini-batching: Group samples by similar modality availability to reduce padding overhead.  
• Early stopping on validation RMSE and physical plausibility ratio.  
• Hyperparameter sweep: number of layers $L\in\{4,6,8\}$, heads $H\in\{4,8\}$, embedding size $d\in\{128,256\}$, $\lambda\in[0.1,10]$.  

2.5 Experimental Design & Evaluation  
We design experiments to answer:  
• Q1: How does PC-MMT perform under varying missing‐modality rates compared to baselines?  
• Q2: Does physics-informed attention improve physical plausibility without degrading accuracy?  
• Q3: Can PC-MMT generalize to unseen compositions or processing conditions?  

Baselines:  
1. Concatenation Transformer (no physics bias).  
2. Meta-Transformer [Zhang et al., 2023] (unified multimodal, no physics).  
3. Graph neural network with late fusion of modalities.  
4. Variational multimodal autoencoder (no constraints).  

Datasets & Splits:  
• Random split (80/10/10) on Materials Project entries with complete modalities.  
• Grouped split by composition families to test generalization.  
• Real-world split by experimental batch to test domain shift.  

Evaluation Metrics:  
• Regression: RMSE, MAE, $R^2$.  
• Classification: accuracy, precision, recall, F1, ROC‐AUC.  
• Physical plausibility ratio: fraction of predictions satisfying hard physics checks (e.g., charge neutrality).  
• Robustness under missing data: metric degradation as a function of drop rate.  
• Computational cost: training time, inference time per sample.  

Validation via DFT follow-up: For top-K novel candidate predictions (e.g., band gap > target), perform DFT calculations to verify formation energy and stability.  

3. Expected Outcomes & Impact  
We anticipate that PC-MMT will:  
• Significantly outperform baselines on property prediction accuracy, especially under high missing‐modality scenarios (e.g., >30% drop rate).  
• Yield a higher physical plausibility ratio (e.g., >95%) compared to un-constrained models (<80%).  
• Demonstrate the utility of attention‐level physics biases in guiding representation learning.  
• Generalize to novel composition families with minimal fine-tuning.  

Impact on Materials Discovery:  
• Accelerated hypothesis generation: reliable predictions from fragmented data reduce wasted experiments.  
• Enhanced interpretability: attention maps reveal modality interactions and physics compliance.  
• Broader adoption: a framework transferrable to other scientific domains with multimodal data and known constraints (e.g., chemistry, biology, geoscience).  
• Progress on “Why Isn’t It Real Yet?”: by addressing data heterogeneity and physics integration, PC-MMT closes key gaps hindering AI’s exponential growth in materials science.  

4. References  
[1] Dosovitskiy, A., et al. “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.” arXiv:2010.11929 (2021).  
[2] Gao, Z.-F., et al. “AI-Accelerated Discovery of Altermagnetic Materials.” arXiv:2311.04418 (2023).  
[3] Zhang, Y., et al. “Meta-Transformer: A Unified Framework for Multimodal Learning.” arXiv:2307.10802 (2023).  
[4] Chen, C., et al. “Accelerating Computational Materials Discovery with AI and Cloud HPC.” arXiv:2401.04070 (2024).  
[5] Takahara, I., et al. “Accelerated Inorganic Materials Design with Generative AI Agents.” arXiv:2504.00741 (2025).  
[6] Vaswani, A., et al. “Attention is All You Need.” NeurIPS (2017).  