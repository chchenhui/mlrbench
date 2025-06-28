**Title:** Causality-Aware World Models via Counterfactual Latent State Prediction

---

### **Introduction**

World models—computational frameworks that enable intelligent agents to simulate, predict, and interact with their environment—are foundational to advancements in artificial intelligence (AI), robotics, and cognitive science. Historically rooted in sequential modeling techniques such as recurrent neural networks (RNNs) and state-space models (SSMs), world models have evolved to encompass transformer architectures, diffusion models, and hybrid approaches tailored for high-dimensional and multimodal environments. However, a critical limitation persists: most world models learn correlations rather than causal relationships governing environmental dynamics. This restricts their ability to generalize to unseen interventions (e.g., novel actions or external perturbations) or anticipate counterfactual outcomes crucial for decision-making in domains like healthcare and embodiment.

**Research Objectives**  
This proposal aims to design **causality-aware world models** that explicitly encode causal mechanisms through **counterfactual latent state prediction**. Our objectives are threefold:  
1. To develop a hybrid architecture combining temporal modeling (e.g., transformers or SSMs) with counterfactual reasoning mechanisms that condition latent state transitions on hypothetical interventions.  
2. To formulate a training paradigm where models simultaneously predict factual (observed) and counterfactual (hypothetical) trajectories, regularizing the latent space to capture causal structures.  
3. To validate the model’s capacity to generalize to unseen interventions, interpret learned causal relationships, and improve robustness in complex, high-dimensional environments.

**Significance**  
Existing world models often degrade under distributional shifts caused by interventions (e.g., altering physical properties of objects or deploying new policies in robotics). By integrating counterfactual learning, our approach aligns with human-like causal reasoning, enabling agents to ask *“What would happen if...?”* and make decisions based on underlying mechanisms rather than surface-level patterns. This work addresses key challenges identified in recent literature, including the lack of **zero-shot generalization to unseen interventions** (Baradel et al., 2019; Melnychuk et al., 2022) and the **need for causal discovery in latent representations** (White & Green, 2023). Success would advance domains ranging from robotics to medical simulation, where understanding intervention effects is paramount.

---

### **Methodology**

#### **1. Data Collection and Environment Design**  
**Simulated Causal Environments**:  
We train and evaluate models on synthetic and semi-simulated environments with explicit causal structures:  
- **Physics Worlds**: Grid-based environments inspired by CoPhy (Baradel et al., 2019), where agents interact with objects whose dynamics are governed by Newtonian mechanics. Interventions include altering initial positions, applying forces, or modifying mass.  
- **Disentangled VAE Spaces**: Environments like *3D Identifiable VAE* (Kumar et al., 2017), where object attributes (color, shape, position) are independently controllable. Interventions manipulate specific latent factors.  
- **Custom Environments**: Extend MuJoCo or PyBullet with scripted intervention protocols (e.g., disabling actuators).  

**Counterfactual Dataset Generation**:  
For each environment, we collect triplets:  
- Historical context $ \mathcal{H}_t = \{(s_{t-k}, a_{t-k})\}_{k=1}^K $ (states $ s $, actions $ a $)  
- Factual outcome: $ s_{t+1} = f_{\text{env}}(s_t, a_t) $  
- Counterfactual outcome: $ s_{t+1}^{\text{count}} = f_{\text{env}}(s_t, a_t^{\text{count}}) $, where $ a_t^{\text{count}} $ is a perturbed action (e.g., random direction or magnitude) or state perturbation.  

#### **2. Model Architecture**  

We propose a **Causal Hybrid Transformer-SSM (CHyTS)** framework (Figure 1):  

**Encoder**: Maps observations $ s_t $ to latent states $ z_t \in \mathbb{R}^d $ via a convolutional or linear network.  

**Temporal Prior** (S4 SSM): Models latent state transitions using a state-space model with parameters conditioned on interventions:  
$$
z_{t+1} = \mathcal{T}_{\theta}(z_t, a_t, c_t) = A(c_t)z_t + B(c_t)a_t + \epsilon_t,
$$  
where $ c_t \in \mathbb{R}^m $ is a counterfactual context vector (described below), $ \epsilon_t \sim \mathcal{N}(0, \sigma I) $, and $ A, B $ are parameter matrices modulated by $ c_t $.  

**Intervention Encoder**: Transforms the perturbed action/state $ a_t^{\text{count}} $ or $ s_t^{\text{count}} $ into a context vector $ c_t \in \mathbb{R}^m $ via:  
$$
c_t = \text{Enc}_{\phi}(a_t^{\text{count}}, a_t) \quad \text{or} \quad c_t = \text{Enc}_{\phi}(s_t^{\text{count}}, s_t),
$$  
contrasting the intervened input with the factual input.  

**Attention Modulation**: To inject counterfactual context into multi-head attention:  
$$
\text{Attn}(Q, K, V, c) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \beta \cdot \phi(c)\right)V,
$$  
where $ \phi(c) \in \mathbb{R}^{n_{\text{heads}}} $ is a learned projection and $ \beta $ is a scaling coefficient.  

**Decoder**: Reconstructs $ \hat{s}_{t+1} $ from $ z_{t+1} $ and predicts the reward $ \hat{r}_{t+1} $.  

#### **3. Training Procedure**  
**Objective Function**: Joint optimization of factual and counterfactual trajectories:  
$$
\mathcal{L} = \underbrace{\sum_{t} \|s_{t+1} - \hat{s}_{t+1}\|_2^2}_{\text{Factual Loss}} + \lambda \cdot \underbrace{\sum_{t} \|s_{t+1}^{\text{count}} - \hat{s}_{t+1}^{\text{count}}\|_2^2}_{\text{Counterfactual Loss}}.
$$  
**Curriculum Learning**: Progressively introduce complex interventions:  
1. Phase 1: Train on small state perturbations (e.g., minor position changes).  
2. Phase 2: Introduce large interventions and action-level modifications.  

**Regularization**:  
- **Causal Structure Prior**: Penalize latent space deviations via KL-divergence from a causal graph estimated using PC-algorithm (Kalainathan et al., 2020).  
- **Domain Confusion Loss** (Melnychuk et al., 2022): Use a gradient reversal layer to ensure invariance of latent states to spurious non-causal factors.  

#### **4. Experimental Design**  

**Baselines**:  
- **DeepMDP** (Grimm et al., 2019): Contrastive learning of latent states without counterfactuals.  
- **Causal InfoGAN** (Li et al., 2022): Infers causal factors via adversarial training but lacks explicit intervention modeling.  
- **Counterfactual MDPs** (Thomas et al., 2017): Requires manual specification of causal graphs.  

**Evaluation Metrics**:  
- **Prediction Accuracy**: MSE for $ s_{t+1} $, $ s_{t+1}^{\text{count}} $.  
- **Causal Fidelity**: Structural Hamming Distance (SHD) between learned and true causal graphs.  
- **Zero-Shot Generalization**: Accuracy on interventions not seen during training (e.g., removing an object entirely).  
- **Task-Based Transfer**: Downstream performance in reinforcement learning (e.g., MuJoCo locomotion after training on perturbed transitions).  

**Ablation Studies**:  
- Impact of $ \lambda $ on trade-offs between factual/counterfactual accuracy.  
- Effectiveness of attention modulation vs. SSM conditioning.  
- Influence of dataset quality (e.g., noisy vs. clean interventions).  

---

### **Expected Outcomes & Impact**

#### **1. Technical Advancements**  
- **Zero-Shot Interventions**: Our model will outperform baselines by ≥30% in predicting outcomes of unseen interventions on synthetic environments (e.g., CoPhy variants).  
- **Causal Latent Structure**: Learned representations will exhibit higher disentanglement scores (e.g., MIG ≥ 0.7) and lower SHD (≤ 2 errors) compared to non-counterfactual models.  
- **Scalability**: CHyTS will efficiently model high-dimensional visual sequences (e.g., 64×64 image frames) while maintaining counterfactual fidelity.  

#### **2. Scientific Insight**  
- **Latent Causality**: Visualization of the latent space will reveal axes corresponding to physical quantities (mass, velocity) or task-specific invariants, validated via probe-based interpretability (Alain & Bengio, 2017).  
- **Robust Policy Planning**: Integration with model-based RL will demonstrate improved performance in distribution-shifted scenarios (e.g., robots adapting to actuator failures).  

#### **3. Domain Applications**  
- **Robotics**: Enhanced simulation for predicting effects of tool modifications or terrain changes.  
- **Healthcare**: Simulating treatment outcomes under hypothetical dosages or interventions, aligning with frameworks like CausaLM (Feder et al., 2020).  
- **Climate Modeling**: Forecasting long-term climate effects of geoengineering scenarios.  

#### **4. Broader Impact**  
This work addresses the critical challenge of **causal generalization in world models**, fostering trust in AI systems operating in high-stakes environments. Open-sourcing code and benchmarks will accelerate research in causal AI, complementing efforts in diffusion-based causal models (Chao et al., 2023) and interventional world models (Purple & Orange, 2023). Future research could extend counterfactual learning to multi-agent systems or incorporate real-world observational datasets.

---

### **Conclusion**  
By unifying counterfactual reasoning with scalable temporal models, this proposal tackles a core bottleneck in world modeling: extracting actionable causal knowledge from data. The CHyTS framework bridges the gap between correlation-driven AI and human-like causal understanding, with transformative implications for simulation, robotics, and scientific discovery.