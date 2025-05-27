# Physics-Constrained Generative Models for Realistic High-Impact Climate Extremes

## **Introduction**

### **Background**  
Climate change poses existential risks to ecosystems, economies, and human well-being, necessitating accurate climate projections to guide adaptation and mitigation strategies. While physics-based numerical Earth system models remain the gold standard for simulating climate dynamics, they struggle to reliably capture High Impact-Low Likelihood (HILL) events such as unprecedented heatwaves, floods, and droughts. These extremes are underrepresented in observational and reanalysis datasets like ERA5 due to their historical rarity, limiting models' ability to simulate tail risks critical for policymaking. Recent advancements in artificial intelligence (AI) have demonstrated promise in weather forecasting, but extrapolating into under-sampled regions of climate variability—such as mid-latitude dynamics or decadal shifts in the El-Niño Southern Oscillation—remains a challenge. Hybrid approaches that merge machine learning (ML) with physical principles offer a pathway to address this gap.

Generative models, particularly Physics-Informed Neural Networks (PINNs), Generative Adversarial Networks (GANs), and diffusion models, have shown success in simulating realistic spatio-temporal data while embedding physical constraints. For example, Tretiak et al. (2022) demonstrated physics-constrained GANs for 3D turbulence simulation, while Yin et al. (2024) integrated physics-informed discriminators into precipitation nowcasting. However, existing methods often prioritize statistical realism over strict adherence to fundamental laws, risking physically implausible generations. This proposal addresses this challenge by developing a scalable framework for generating physically consistent climate extremes, augmented with domain-specific priors.

### **Research Objectives**  
This work aims to:  
1. **Design physics-constrained generative models** that embed conservation laws (e.g., energy, mass) and thermodynamic principles into loss functions or architectural constraints, ensuring physical plausibility.  
2. **Generate high-resolution spatio-temporal HILL event data** (e.g., heatwaves over Europe, hurricanes over the Atlantic) that obey known climate dynamics.  
3. **Validate generated events** against physical consistency metrics and compare their performance to state-of-the-art baselines like evtGAN (Boulaguiem et al., 2021) and conditional GAN-based downscaling (Li & Cao, 2024).  
4. **Quantify uncertainties** in generated samples and propose calibration techniques to enhance reliability for risk assessment.  

The project's significance lies in bridging the gap between data-driven ML and physics-based modeling, providing tools to populate the "tail-risk" sectors of climate projections crucial for building resilient infrastructure.

---

## **Methodology**

### **1. Data Collection and Preprocessing**  
**Data Sources:**  
- **Observational datasets**:  
  - ERA5 reanalysis (spatial resolution: 0.25°, hourly temporal resolution; Hersbach et al., 2020).  
  - Satellite-based precipitation data (IMERG; Hou et al., 2014).  
- **Climate model outputs**: CMIP6 simulations for synthetic HILL events under high-emissions scenarios (RCP8.5).  
- **Historical extreme events**: Compilations of past HILL events (e.g., European heatwaves of 2003 and 2022).  

**Preprocessing:**  
- **Spatio-temporal alignment**: Resample data to uniform 0.5°×0.5° grids and 6-hourly intervals.  
- **Extreme value extraction**: Apply block maxima and peak-over-threshold methods from extreme value theory to isolate extreme events.  
- **Normalization**: Standardize variables (e.g., temperature, precipitation) using domain-specific statistics (mean, standard deviation).  

### **2. Model Architecture**  
We propose a hybrid generative framework combining physics constraints with adversarial training or diffusion processes. Two architectures are considered:

#### **A. Physics-Informed GAN (PI-GAN)**  
**Generator (G):**  
A conditional 3D Convolutional Neural Network (CNN) augmented with self-attention modules to capture long-range spatio-temporal dependencies. Input includes latent noise vectors ($z \in \mathbb{R}^d$) and climate covariates (e.g., sea surface temperatures).  

**Discriminator (D):**  
A physics-informed critic network with two heads:  
1. Adversarial branch: Discriminates simulated vs. real data.  
2. Physics branch: Enforces conservation laws via a parametric loss:  
   $$L_{\text{physics}} = \lambda_1 \cdot \|\nabla \cdot \mathbf{u}\|^2 + \lambda_2 \cdot \left\|\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u})\right\|^2,$$  
   where $\mathbf{u}$ is wind velocity, $\rho$ is air density, and $\lambda_1, \lambda_2$ are balancing weights.  

**Training:**  
- Minimax game with a combined loss:  
  $$\min_G \max_D \left[L_{\text{adv}} + \gamma \cdot L_{\text{physics}}\right],$$  
  where $L_{\text{adv}} = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$ and $\gamma$ controls physics constraints.  
- **Regularization**: Gradient penalty (Gulrajani et al., 2017) ensures Lipschitz continuity.  

#### **B. Physics-Constrained Diffusion Model**  
**Forward Process:**  
Add noise to training data over $T$ steps:  
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I).$$  

**Reverse Process:**  
A 3D U-Net predicts noise residuals $\epsilon_\theta(x_t, t)$ while incorporating physical constraints via a weighted loss:  
$$L = \mathbb{E}_{x_0, t, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2] + \beta \cdot \|\text{div}(\tilde{\mathbf{u}})\|^2,$$  
where $\tilde{\mathbf{u}}$ is the reconstructed wind field.  

### **3. Experimental Design**  
**Training Protocol:**  
- **Datasplits**: 70% training, 15% validation, 15% test. Stratify training data by event severity.  
- **Hyperparameters**: Adam optimizer (learning rate $=10^{-4}$), batch size $=32$ for 500 epochs.  
- **Transfer Learning**: Pre-train on CMIP6 data before fine-tuning on ERA5 to mitigate observational data scarcity.  

**Baselines for Comparison:**  
1. **Unconditional GAN** (baseline for statistical realism, ignores physics).  
2. **evtGAN** (Boulaguiem et al., 2021; specializes in spatial extremes).  
3. **NeuralGCM** (Stephan et al., 2024; hybrid physics-ML model).  
4. **Physics-Informed PINN Reducer** (hypothetical baseline enforcing constraints via training interpolation).  

### **4. Evaluation Metrics**  
**A. Statistical Realism:**  
- **Fréchet Inception Distance (FID):** Compares spatial patterns of temperature/precipitation fields.  
- **Extreme Value Consistency (EVC):** Quantifies how well generated events adhere to theoretical extreme value distributions (e.g., Generalized Extreme Value).  

**B. Physical Consistency:**  
- **Divergence Error (DE):** Measures adherence to mass conservation:  
  $$\text{DE} = \frac{1}{N} \sum_{i=1}^N \left|\nabla \cdot \mathbf{u}_i\right|,$$  
  where $\mathbf{u}_i$ is wind velocity in the $i$-th pixel.  
- **Energy Conservation Test:** For each generated event, compute kinetic energy dissipation rate $\epsilon = \nu \|\nabla \mathbf{u}\|^2$ (where $\nu$ is viscosity) and compare to CMIP6 simulations.  

**C. Downstream Utility:**  
- **Impact Model Benchmark**: Train a deep learning model for crop yield prediction using generated vs. real data and compare root mean square error (RMSE).  

### **5. Uncertainty Quantification**  
- **Ensemble Generation**: Sample 100 latent vectors per test case to model aleatoric uncertainty.  
- **Bayesian PINN**: Embed Gaussian priors over weights to quantify epistemic uncertainty (Gal et al., 2016).  

---

## **Expected Outcomes & Impact**

### **1. Technical Contributions**  
- **First-Of-Its-Kind Framework**: A scalable generative model architecture that tightly integrates hard physics constraints (e.g., conservation laws) with adversarial/diffusion-based learning.  
- **Domain-Specific Metrics**: Novel evaluation metrics like DE and EVC to assess physical plausibility of synthesized extremes.  

### **2. Scientific Contributions**  
- **Enhanced Climate Risk Assessment**: By populating HILL event distributions (e.g., 1-in-1000-year heatwaves), the framework will enable policymakers to quantify systemic risks in marginalized regions.  
- **Validation of Physical Constraints**: Empirical evidence on the trade-offs between physics fidelity and statistical realism, addressing debates around "constraint rigidity" in generative modeling.  

### **3. Societal Impact**  
- **Improved Adaptation Plans**: Application to flood zone mapping or dry/wet season hobbying can inform resilient infrastructure planning.  
- **Open-Source Tools**: Release of training code, pre-trained models, and datasets for generating synthetic climate extremes under CMIP6 scenarios.  

### **4. Limitations and Mitigation**  
- **Computational Feasibility**: Training 3D CNNs on spatio-temporal data is resource-intensive; we plan to optimize training using mixed-precision techniques and cloud GPUs.  
- **Generalization Across Regions**: Domain adaptation techniques (e.g., cyclical GANs) will be explored to transfer models trained on one region (e.g., Europe) to others (e.g., South Asia).  

---

This proposal combines cutting-edge ML with climate science to tackle three critical challenges: data scarcity of HILL events, physical plausibility of AI-generated scenarios, and quantitative uncertainty estimation. By addressing these, our work will advance the frontier of hybrid AI-climate modeling, providing actionable tools for climate resilience.