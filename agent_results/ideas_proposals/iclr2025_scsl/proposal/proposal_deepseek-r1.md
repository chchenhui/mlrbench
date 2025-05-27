# Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS): A Framework for Countering Spurious Correlations in Deep Learning  

## 1. Introduction  
### Background  
Deep learning models frequently exploit spurious correlations—statistical shortcuts that hold inconsistently across environments—to make predictions. These shortcuts arise from biases in training data (e.g., penguins frequently appearing against snowy backgrounds in "bird" classifications) and are exacerbated by the tendency of gradient-based optimization to prioritize easily learnable features. While methods like group robust optimization [1], causal representation learning [6], and last-layer reweighting [9] mitigate *known* spurious correlations, they struggle when spurious features are unlabeled, latent, or modality-specific. Recent works such as SPUME [3] and RaVL [5] leverage vision-language models or region-aware losses but require task-specific prior knowledge, limiting scalability.  

### Research Objectives  
This work proposes **Adaptive Invariant Feature Extraction using Synthetic Interventions (AIFS)**, a framework that automates the discovery and suppression of spurious correlations during training. The key objectives are:  
1. **Unsupervised Identification**: Automatically detect latent spurious features without relying on group annotations.  
2. **Dynamic Intervention**: Synthesize controlled perturbations in latent subspaces to simulate distribution shifts.  
3. **Invariant Learning**: Enforce prediction consistency under synthetic interventions while penalizing reliance on perturbed features.  
4. **Modality Agnosticism**: Ensure applicability across image, tabular, and multimodal data.  

### Significance  
AIFS addresses critical gaps identified in the literature:  
- **Scalability**: Unlike SPUME [3] or RaVL [5], AIFS requires no attribute annotations or region-level supervision.  
- **Generality**: By operating on latent representations, it extends beyond vision tasks to tabular and multimodal settings.  
- **Foundational Insights**: The framework provides empirical insights into how optimization dynamics and intervention-based training alter feature dependence in deep models.  

---  

## 2. Methodology  
### Framework Overview  
AIFS integrates a generative intervention loop (Figure 1) with a dual-objective loss to guide models toward invariant features:  
1. **Pretrained Encoder**: A backbone network (e.g., ResNet, Transformer) maps inputs $x$ to latent representations $z = E(x)$.  
2. **Intervention Module**: Applies stochastic perturbations $\Delta z$ to subspaces of $z$ via learned masks $M \in [0, 1]^d$:  
   $$z_{\text{pert}} = z + M \odot \Delta z, \quad \Delta z \sim \mathcal{N}(0, \sigma^2I)$$  
   where $M$ prioritizes dimensions with high *sensitivity scores* (see §2.2).  
3. **Dual-Objective Loss**: Trains the classifier $C$ to (i) preserve prediction invariance under interventions and (ii) reduce dependence on perturbed features.  

![AIFS Architecture](figures/architecture.png)  
*Figure 1: AIFS pipeline. Latent perturbations are guided by sensitivity-based masks, and the dual loss enforces invariance.*  

### Algorithmic Details  
**Step 1: Sensitivity-Aware Masking**  
Sensitivity scores $S \in \mathbb{R}^d$ for latent dimensions are computed via gradient attribution:  
$$S_i = \mathbb{E}_{x \sim \mathcal{D}} \left[ \left\| \frac{\partial \mathcal{L}(C(z), y)}{\partial z_i} \right\|_2 \right]$$  
Dimensions with higher $S_i$ are hypothesized to encode spurious features. A differentiable Gumbel-Softmax sampler selects the top-$k$ dimensions for masking $M$.  

**Step 2: Dual-Objective Optimization**  
The classifier is trained with a joint loss:  
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{inv}} + \lambda \mathcal{L}_{\text{sens}}$$  
- **Invariance Loss**: Encourages consistency between original and perturbed predictions:  
  $$\mathcal{L}_{\text{inv}} = \mathbb{E}_{x, \Delta z} \left[ \text{KL}\left( C(z) \parallel C(z_{\text{pert}}) \right) \right]$$  
- **Sensitivity Loss**: Penalizes confidence drops when high-sensitivity dimensions are perturbed:  
  $$\mathcal{L}_{\text{sens}} = \mathbb{E}_{x, \Delta z} \left[ \max\left(0, \log C(z)_y - \log C(z_{\text{pert}})_y \right) \right]$$  
  This forces the model to rely less on perturbed (spurious) features.  

**Step 3: Iterative Adaptation**  
Every $T$ epochs, sensitivity scores $S$ are recomputed, and masks $M$ are updated to target the latest high-sensitivity dimensions. This closed-loop process enables progressive refinement of invariant features.  

### Experimental Design  
**Datasets**  
- **Image**: Waterbirds (spurious: background) [9], CelebA (spurious: hairstyle/gender) [3].  
- **Tabular**: Synthetic datasets with confounders (e.g., income prediction with spurious ZIP code correlations).  
- **Multimodal**: MMIMDb (movie genre prediction with text-image spurious links) [5].  

**Baselines**  
- GroupDRO [1], SPUME [3], ElRep [2], ULE [4], and standard ERM.  

**Evaluation Metrics**  
- **Worst-Group Accuracy (WGA)**: Accuracy on minority subgroups.  
- **Robustness Gap (RG)**: Difference between in-distribution and out-of-distribution (perturbed) accuracy.  
- **Sensitivity Sparsity**: Fraction of latent dimensions with near-zero sensitivity scores (measures feature selectivity).  

**Implementation**  
- **Encoder**: ResNet-50 (image), BERT (text), MLP (tabular).  
- **Training**: AdamW optimizer, $\lambda = 0.5$, $k = 20\%$ of latent dimensions perturbed.  

---  

## 3. Expected Outcomes & Impact  
### Expected Outcomes  
1. **Improved Robustness**: AIFS is anticipated to achieve ≥5% higher WGA than GroupDRO and SPUME on Waterbirds and CelebA by dynamically suppressing latent spurious features.  
2. **Generalization**: The framework will demonstrate consistent gains across modalities, including a 10–15% reduction in RG for tabular data with synthetic confounders.  
3. **Interpretability**: Sensitivity maps will reveal known spurious features (e.g., backgrounds in Waterbirds) without supervision, validated via saliency analysis.  

### Broader Impact  
- **Reliable AI Systems**: By mitigating reliance on spurious cues, AIFS can enhance the safety of medical diagnosis models (e.g., avoiding demographic biases in X-ray analysis) and autonomous systems (e.g., robust sensor fusion in self-driving cars).  
- **Methodological Advancements**: The intervention paradigm bridges causal representation learning [6,7] and gradient-based attribution [8], offering insights into the interplay between optimization and invariance.  

---  

## 4. Conclusion  
AIFS pioneers a closed-loop, modality-agnostic approach to counter spurious correlations by synthesizing latent interventions and adaptively suppressing shortcut features. By unifying gradient attribution with invariance learning, the framework addresses critical limitations of supervised robustification methods while providing insights into the mechanistic origins of shortcut learning. Successful validation of AIFS will advance the deployment of reliable AI systems in high-stakes applications.  

---  
**References**  
[1] Sagawa et al., *ICML 2020*; [2] Wen et al., *arXiv:2502.09850*; [3] Zheng et al., *arXiv:2406.10742*; [4] Mitchell et al., *arXiv:2409.02792*; [5] Varma et al., *arXiv:2411.04097*; [6] Yao et al., *arXiv:2409.02772*; [7] Chen et al., *arXiv:2305.02640*; [8] Sun et al., *arXiv:2307.12344*; [9] Izmailov et al., *NeurIPS 2022*.