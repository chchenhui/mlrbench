# Causal Diffusion Models: Disentangling Latent Causal Factors in Generative AI  

## 1. Introduction  

### Background  
Generative AI systems, such as diffusion models and large language models, have achieved unprecedented performance in tasks involving image synthesis, text generation, and data augmentation. These models excel at capturing complex data distributions through deep neural networks. However, their reliance on statistical associations—rather than causal relationships—often leads to spurious correlations and poor generalization under distributional shifts. For instance, a medical imaging generative model might associate a specific imaging artifact with a disease diagnosis, inadvertently amplifying biases rather than modeling the true underlying pathology.  

Causal representation learning (CRL) has emerged as a critical area of research to address these limitations. CRL seeks to identify disentangled latent variables corresponding to *interventional* causal factors, enabling controllable and interpretable generation. Recent advancements, such as DeCaFlow (2025) and CausalBGM (2025), demonstrate the feasibility of integrating causal inference into generative frameworks. However, these methods often assume simplified causal structures or rely on restrictive assumptions like fully observed confounders. Diffusion models, with their iterative denoising process, present a unique opportunity to embed causal reasoning into every generation step, aligning synthetic data with causal mechanisms.  

### Research Objectives  
This proposal aims to develop **Causal Diffusion Models (CDMs)**, a novel class of generative models that explicitly incorporate causal graphs into their latent space. Key objectives include:  
1. Designing a causal discovery module to infer directed causal relationships among latent variables using observational or interventional data.  
2. Integrating causal constraints into the diffusion process to enforce alignment between denoising steps and the causal graph.  
3. Evaluating the model’s ability to perform counterfactual editing, robustness to confounding variables, and generalization across domains.  

### Significance  
CDMs will advance both causal inference and generative modeling by:  
- **Improving Trustworthiness**: Reducing spurious correlations through explicit modeling of causal mechanisms.  
- **Enabling Controllability**: Allowing users to manipulate specific causal factors (e.g., varying "disease severity" while preserving anatomical structures).  
- **Facilitating Scientific Discovery**: Supporting hypothesis testing in domains like healthcare (e.g., predicting treatment effects) and social sciences.  
This work addresses critical gaps in current generative models, aligning them with the principles of causality to enhance interpretability and fairness.  

---

## 2. Methodology  

Our approach combines causal discovery techniques with the architectural strengths of diffusion models. The methodology is divided into three phases: (1) causal graph learning, (2) integration into the diffusion process, and (3) experimental validation.  

### 2.1 Causal Graph Learning  

We propose a **causal discovery module** that infers directional relationships among latent variables using a hybrid approach:  

#### Structural Causal Model (SCM)  
Let $\mathbf{z} = [z_1, z_2, ..., z_d] \in \mathbb{R}^d$ represent latent variables, and $G = (V, E)$ denote the causal graph where $V$ corresponds to variables and $E$ represents directed edges. The SCM is defined as:  
$$
z_i = f_i(\mathrm{pa}(z_i), \epsilon_i), \quad \forall i \in \{1, ..., d\}
$$  
where $\mathrm{pa}(z_i)$ denotes parents of $z_i$ in $G$, and $\epsilon_i$ are independent exogenous noise terms.  

#### Hybrid Causal Discovery  
The module combines:  
1. **Score-based methods**: Using likelihood scores (e.g., BIC or AIC) to evaluate candidate graphs.  
2. **Interventional data**: If available, leveraging interventions to resolve causal directionality ambiguities.  
3. **Domain constraints**: Incorporating expert knowledge (e.g., "disease causes symptom" in healthcare) via constraint-based approaches like the PC algorithm.  

**Implementation**:  
- Train an encoder $E: \mathbf{x} \mapsto \mathbf{z}$ mapping observations $\mathbf{x}$ to latents.  
- Optimize $G$ using a variational objective:  
$$
\mathcal{L}_{\text{causal}} = \sum_{i=1}^d \log p(z_i | \mathrm{pa}(z_i)) - \lambda \cdot \Omega(G),
$$  
where $\Omega(G)$ penalizes cyclic graphs and encourages sparse edges via an $L_1$-norm.  

### 2.2 Integration into Diffusion Process  

Diffusion models generate data through a forward process of noise addition followed by a reverse process of denoising. We modify the reverse process to align with the inferred causal graph $G$.  

#### Forward Process  
Given data $\mathbf{x}_0$, the forward process adds Gaussian noise iteratively:  
$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_{t-1}, (1-\alpha_t)\mathbf{I})
$$  
for $t = 1$ to $T$.  

#### Reverse Process with Causal Alignment  
The denoising step at time $t$ predicts $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$ using a neural network $D_\theta$. To enforce causal constraints:  
1. **Causal masking**: During denoising, apply a masking matrix $M_G$ derived from $G$ to suppress non-causal dependencies. For example, if $z_i \rightarrow z_j$, then $z_j$'s update depends on $z_i$ but not vice versa.  
2. **Causal consistency loss**: Add a term penalizing deviations from the SCM:  
$$
\mathcal{L}_{\text{consistency}} = \sum_{i=1}^d \mathbb{E}_{q(\mathbf{z})} \left[ \left(z_i - \hat{f}_i(\mathrm{pa}(z_i)) \right)^2 \right],
$$  
where $\hat{f}_i$ is a parameterized function approximating $f_i$.  

**Total Loss Function**:  
$$
\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \beta \cdot \mathcal{L}_{\text{causal}} + \gamma \cdot \mathcal{L}_{\text{consistency}},
$$  
with hyperparameters $\beta, \gamma$ controlling regularization strength.  

### 2.3 Experimental Design  

#### Datasets  
1. **Synthetic**: Controlled environments with ground-truth causal graphs (e.g., linear Gaussian models).  
2. **Medical Imaging**: Chest X-rays with annotated pathologies (e.g., NIH ChestX-ray14).  
3. **Tabular Data**: Economic datasets (e.g., UCI Adult) for counterfactual fairness analysis.  

#### Baselines  
- **VAE-based CRL**: C2VAE (2024), β-TCVAE.  
- **Causal Flow**: DeCaFlow (2025), CausalBGM (2025).  
- **Standard Diffusion**: DDPM (2020), DDIM (2021).  

#### Evaluation Metrics  
1. **Counterfactual Accuracy**:  
   $$
   \text{CA} = \frac{1}{N} \sum_{i=1}^N \left\| \mathcal{T}(z_i) - \mathcal{G}(z_i) \right\|_2,
   $$  
   where $\mathcal{T}$ is the true counterfactual, and $\mathcal{G}$ is the generated one.  
2. **Disentanglement Score (MIG)**: Measures mutual information gap between latents and attributes.  
3. **Causal Discovery Accuracy (SHD)**: Structural Hamming distance between inferred and true graphs.  
4. **Confounder Robustness**: Performance degradation under synthetic confounding.  

#### Ablation Studies  
- Impact of interventional data on causal graph recovery.  
- Effect of masking strategies on denoising alignment.  
- Trade-offs between causal accuracy and generation quality (FID score).  

---

## 3. Expected Outcomes & Impact  

### Expected Outcomes  
1. **Causal Alignment in Latent Space**:  
   - Demonstration of CDMs’ ability to recover ground-truth causal graphs on synthetic data (SHD < 0.15).  
   - Improved MIG scores (e.g., >0.6) compared to baselines, indicating better disentanglement.  
2. **Controllable Generation**:  
   - Counterfactual accuracy (CA) within 10% of true effects in medical imaging benchmarks.  
3. **Robustness**:  
   - 20% lower performance drop under confounding shifts compared to standard diffusion models.  

### Broader Impact  
1. **Healthcare**:  
   - Enable generation of synthetic medical images where disease attributes (e.g., tumor size) are independently adjustable, aiding diagnostic model training.  
2. **Economics**:  
   - Test policy interventions (e.g., tax changes) by generating counterfactual economic scenarios.  
3. **Fairness**:  
   - Mitigate algorithmic bias by disentangling sensitive attributes from other factors (e.g., gender in facial recognition).  

### Long-Term Vision  
This work lays the foundation for **Causal Foundation Models**, combining the scalability of large-scale diffusion models with the interpretability of causal reasoning. Future directions include:  
- Developing scalable algorithms for causal discovery in billion-dimensional latent spaces.  
- Integrating CDMs with large language models (LLMs) to enforce causal coherence in text generation.  
- Creating open-source benchmarks (e.g., CausalImageNet) to drive progress in CRL.  

By bridging the gap between generative AI and causal inference, CDMs promise to transform applications where trustworthiness and controllability are paramount.