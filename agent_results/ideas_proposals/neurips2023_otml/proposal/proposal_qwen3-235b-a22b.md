# **Adaptive Unbalanced Optimal Transport for Robust Domain Adaptation under Label Shift**  

---

## **1. Introduction**  

### **Background**  
Optimal Transport (OT) has emerged as a powerful tool in machine learning (ML) for aligning distributions across domains by finding minimal transportation cost plans $^{1}$. In domain adaptation (DA), OT enables the alignment of source (labeled) and target (unlabeled) distributions, particularly when their marginal distributions differ (covariate shift) or class distributions vary (label shift). However, classical OT assumes *balanced distributions* (equal total mass in source and target domains) and perfect one-to-one correspondences, which often fail in real-world scenarios due to label shifts. For instance, autonomous vehicles may encounter new environmental conditions with shifted class distributions, or medical imaging datasets may underrepresent rare diseases compared to training data.  

Unbalanced Optimal Transport (UOT) relaxes the strict mass conservation constraint of OT by introducing *marginal relaxation penalties*, such as Kullback–Leibler (KL) divergence $^{1}$. This allows for partial matching of distributions with differing masses. However, existing UOT methods require *fixed hyperparameters* to control the degree of marginal relaxation (e.g., penalty strength λ). This manual tuning is impractical in realistic DA settings where label shifts vary unpredictably.  

### **Research Objectives**  
This study proposes **Adaptive Unbalanced Optimal Transport (A-UOT)**, a novel framework that:  
1. **Automatically adapts** UOT’s marginal relaxation parameters to unknown label shifts by learning them during training.  
2. **Integrates** A-UOT into a deep domain adaptation pipeline, enabling end-to-end optimization of both feature alignment and label shift compensation.  
3. **Empirically validates** the framework on cross-domain benchmarks with synthetic and real-world label shifts.  

### **Significance**  
1. **Practical Impact**: Robust DA under label shifts is critical for applications with distributional shifts, such as autonomous systems, healthcare diagnostics, and cross-lingual NLP tasks.  
2. **Technical Contribution**: A-UOT addresses UOT’s key limitation—manual parameter tuning—by learning the optimal degree of mass variation, closing the gap between theory and real-world deployment.  
3. **Theoretical Advancement**: By combining UOT with dynamic class proportion estimation, the framework bridges DA theory (e.g., label shift compensation $^{2}$) with robust OT formulations $^{1}$.  

---

## **2. Methodology**  

### **2.1. Problem Formulation and Assumptions**  
We consider a domain adaptation setting where:  
- **Source domain** $ \mathcal{S} = \{(x_i^s, y_i^s)\}_{i=1}^{n_s} $: Labeled data drawn from distribution $ P_{s}(X, Y) $.  
- **Target domain** $ \mathcal{T} = \{x_j^t\}_{j=1}^{n_t} $: Unlabeled data drawn from $ P_t(X, Y) $.  
- **Label Shift**: $ P_s(Y) \neq P_t(Y) $, but $ P_s(X|Y) = P_t(X|Y) $ (class-conditional feature alignment).  

The goal is to learn a feature extractor $ f: \mathcal{X} \rightarrow \mathcal{Z} $ and classifier $ g: \mathcal{Z} \rightarrow \mathcal{Y} $ such that $ g(f(x_j^t)) $ minimizes the target domain’s prediction loss.  

### **2.2. A-UOT Framework**  

#### **2.2.1 Mathematical Formulation**  
Let $ \mu = \{\mu_k\}_{k=1}^K $ and $ \nu = \{\nu_l\}_{l=1}^L $ denote source and target *class-specific feature distributions* in the learned latent space $ \mathcal{Z} $. A-UOT learns transport plans $ \pi_{kl} $ that solve:  
$$
\min_{\pi \in \mathbb{R}_{+}^{K \times L}} \sum_{k=1}^{K} \sum_{l=1}^{L} \pi_{kl} \cdot C_{kl} + \alpha D_{\text{KL}}(\pi \mathbf{1} \parallel \mu) + \beta D_{\text{KL}}(\pi^T \mathbf{1} \parallel \nu),
$$
where $ C_{kl} = \mathbb{E}[\text{MSE}(z_k^s, z_l^t)] $ is the cost matrix between source and target features, and $ \alpha, \beta \in \mathbb{R}_+ $ are adaptive marginal relaxation parameters.  

**Key Innovation**: Unlike fixed UOT $^{1}$, we learn $ \alpha $ and $ \beta $ as functions of *pseudo-label statistics* (Section 2.3) via gradient-based optimization:  
$$
\alpha = \psi_{\text{CNN}}(\nu), \quad \beta = \phi_{\text{CNN}}(\mu),
$$
where $ \psi $ and $ \phi $ are shallow neural networks that map target/class-conditional statistics to parameter values.  

#### **2.2.2 Algorithmic Steps**  
1. **Feature Learning**: Train a shared encoder $ f $ to extract features $ z_i^s = f(x_i^s) $, $ z_j^t = f(x_j^t) $.  
2. **Pseudo-Labeling**: Initialize $ g $ on $ \mathcal{S} $, then generate pseudo-labels $ \hat{y}_j^t = g(x_j^t) $ for $ \mathcal{T} $. Compute target class proportions $ \hat{\nu}_l $ using soft argmax over $ \hat{y}_j^t $.  
3. **Adaptive UOT**: Compute $ \pi $ using accelerated Sinkhorn iterations $^{1}$ with learnable $ \alpha $ and $ \beta $.  
4. **OT-Distance Loss**: Define the alignment loss as $ \mathcal{L}_{\text{OT}} = \sum_{k,l} \pi_{kl} \cdot C_{kl} $.  
5. **Classifier Update**: Update $ g $ using source labels and pseudo-label entropy:  
$$
\mathcal{L}_{\text{cls}} = \frac{1}{n_s} \sum_{i} \ell(y_i^s, g(z_i^s)) - \gamma \frac{1}{n_t} \sum_{j} H(g(z_j^t)),
$$
where $ \ell $ is cross-entropy, $ H $ is entropy regularization $^{3}$, and $ \gamma $ controls weight.  
6. **Joint Optimization**: Minimize $ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{OT}} $ using stochastic gradient descent.  

### **2.3. Adaptive Parameter Learning**  
We learn $ \alpha $ and $ \beta $ to reflect *pseudo-label uncertainty*:  
- **Target Uncertainty**: If pseudo-labels are confident (low entropy), $ \beta $ increases to relax the target marginal constraint, allowing the model to adapt to the estimated $ \nu $.  
- **Source Class Balance**: If source class proportions are skewed, $ \alpha $ dynamically adjusts to prevent over-alignment.  

This is formalized via:  
$$
\alpha = \text{Softplus}(\mathbf{w}_\alpha^T \cdot \mu + b_\alpha), \quad \beta = \text{Softplus}(\mathbf{w}_\beta^T \cdot \hat{\nu} + b_\beta),
$$
where $ \mathbf{w}_\alpha, \mathbf{w}_\beta $ are learnable parameters.  

### **2.4. Computational Enhancements**  
- **Minibatch UOT**: To handle large-scale data, we extend A-UOT to minibatch settings as in $^{1}$, ensuring scalability.  
- **Sinkhorn Acceleration**: Use log-domain stabilization and dynamic regularization to reduce Sinkhorn iterations.  

### **2.5. Experimental Design**  

#### **Datasets**  
1. **MNIST-USPS**: Handwritten digit domain shift with synthetic label imbalances (e.g., overrepresenting "0" in target).  
2. **PACS**: Multi-domain art/painting/animal photo dataset with spurious correlations and label shifts.  
3. **Camelyon17**: Histopathology dataset with domain (hospital) and class (cancer subtypes) shifts.  

#### **Baselines**  
- **OT-based**: DeepJDOT $^{4}$, Target-OT $^{2}$.  
- **UOT-based**: Fixed-UOT with grid search on $ \alpha, \beta $.  
- **Non-OT**: DANN $^{3}$, MixUp-based UOT $^{3}$.  

#### **Evaluation Metrics**  
- **Primary**: Target domain accuracy, H-divergence between source and target embeddings.  
- **Secondary**: OT cost, Sinkhorn divergence, and pseudo-label calibration error (Brier score).  

#### **Implementation Details**  
- **Architecture**: ResNet-18 backbone with domain-adversarial classifier for $ f $ and $ g $.  
- **Hyperparameters**: Stochastic gradient descent (0.9 momentum, 0.001 learning rate), $ \lambda \in [0.1, 1.0] $.  
- **Ablation**: Study impact of $ \alpha, \beta $ adaptation using label shift severity as a covariate.  

---

## **3. Expected Outcomes & Impact**  

### **3.1. Theoretical and Empirical Advancements**  
1. **Adaptive Parameter Learning**: Demonstrate that $ \alpha, \beta $ can be learned end-to-end without manual tuning, outperforming fixed UOT by >10% on PACS and Camelyon17 with extreme label shifts (e.g., 1:9 vs. 9:1).  
2. **Robustness**: Achieve near-oracle performance when true label shifts are known (oracle baseline), validated via synthetic experiments.  
3. **Generalization**: Improve accuracy on standard DA benchmarks like MNIST-USPS (target accuracy >95%) while maintaining robustness to outliers.  

### **3.2. Societal and Industrial Impact**  
- **Autonomous Systems**: Enable safer deployment in unseen environments (e.g., self-driving cars adapting to new weather conditions).  
- **Healthcare**: Facilitate medical AI models that generalize across patient populations with varying disease prevalence.  
- **NLP**: Enhance cross-lingual models that adapt to low-resource languages with label shifts.  

### **3.3. Addressing Literature Challenges**  
| **Challenge** | **This Study** | **Relevant Baseline** |  
|----------------|----------------|------------------------|  
| Label Shift | Adaptive parameter estimation via pseudo-labels | Fixed-UOT (manual λ) $^{1}$ |  
| Parameter Selection | Learned via gradient descent | Grid search $^{1}$ |  
| Negative Transfer | Class-conditional OT + entropy regularization | MixUp-based UOT $^{3}$ |  
| Scalability | Adaptive UOT + minibatch Sinkhorn | Full-batch OT $^{2}$ |  

By integrating A-UOT into a unified framework, we advance the state-of-the-art by addressing all five challenges identified in prior work.  

---

**References**  
[1] Kilian Fatras et al. "Unbalanced minibatch Optimal Transport" (2021)  
[2] Alain Rakotomamonjy et al. "Optimal Transport for Conditional Domain Matching and Label Shift" (2020)  
[3] Kilian Fatras et al. "Optimal transport meets noisy label robust loss" (2022)  
[4] Quang Huy Tran et al. "Unbalanced CO-Optimal Transport" (2022)