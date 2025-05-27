**Research Proposal: Adaptive Unbalanced Optimal Transport for Robust Domain Adaptation under Label Shift**  

---

### 1. **Introduction**  

**Background**  
Domain adaptation (DA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain with differing data distributions. Optimal transport (OT) has emerged as a powerful tool for DA, leveraging Wasserstein distances to align feature distributions across domains. However, standard OT methods assume balanced class distributions (i.e., equal label proportions) between domains, which is often violated in real-world scenarios due to *label shift*—a mismatch in class priors. For instance, medical imaging datasets may exhibit varying disease prevalence across hospitals, or autonomous vehicles may encounter dynamic class distributions (e.g., pedestrians vs. vehicles) across environments.  

Unbalanced OT (UOT) relaxes the strict marginal constraints of OT, allowing mass creation/destruction to handle distribution mismatches. Yet, existing UOT-based DA methods rely on predefined parameters to control marginal relaxation, which limits their adaptability to unknown label shifts. This work addresses this gap by proposing an *adaptive* UOT framework that dynamically learns relaxation parameters during training, enabling robust DA under label shift.  

**Research Objectives**  
1. Develop an **Adaptive Unbalanced Optimal Transport (A-UOT)** framework that learns marginal relaxation parameters directly from data.  
2. Integrate A-UOT into a deep DA model to align feature distributions while compensating for label shifts.  
3. Validate the framework on benchmarks with synthetic and real-world label shifts, demonstrating improved robustness over fixed UOT and OT baselines.  

**Significance**  
By automating the tuning of UOT parameters, this work eliminates a key bottleneck in applying OT to real-world DA problems. The proposed method will enhance the reliability of DA in critical applications such as medical diagnosis (where label shifts are common) and autonomous systems (where environments evolve dynamically).  

---

### 2. **Methodology**  

#### **2.1 Data Collection**  
- **Datasets**: Use standard DA benchmarks (Office-31, Office-Home, VisDA) with synthetic label shifts induced by subsampling classes in the target domain. For example, reduce the proportion of "monitor" class in Office-31’s *Amazon*→*Webcam* task to simulate label shift.  
- **Real-World Data**: Include datasets with inherent label shifts, such as medical imaging data (e.g., Camelyon17 for tumor classification across hospitals) or autonomous driving datasets (e.g., SHIFT for weather/lighting variations).  
- **Preprocessing**: Extract features using pretrained CNNs (ResNet-50) and normalize to unit variance.  

#### **2.2 Adaptive Unbalanced OT Framework**  

**Mathematical Formulation**  
Let $\mu = \sum_{i=1}^n a_i \delta_{x_i}$ (source) and $\nu = \sum_{j=1}^m b_j \delta_{y_j}$ (target) be discrete measures. The UOT problem relaxes marginal constraints via divergences:  
$$
\text{UOT}(\mu, \nu) = \min_{\pi \geq 0} \sum_{i,j} C_{i,j} \pi_{i,j} + \lambda_1 D_\phi(\pi \mathbf{1}, \mathbf{a}) + \lambda_2 D_\phi(\pi^\top \mathbf{1}, \mathbf{b}),
$$  
where $C_{i,j}$ is the cost matrix, $D_\phi$ is a divergence (e.g., KL divergence), and $\lambda_1, \lambda_2$ control marginal relaxation.  

**Adaptive Parameter Learning**  
Instead of fixing $\lambda_1, \lambda_2$, we propose to learn them via:  
1. **Neural Prediction**: Use a lightweight network $g_\theta$ to predict $\lambda_1, \lambda_2$ from target domain statistics (e.g., feature moments) or pseudo-labels.  
2. **Joint Optimization**: Integrate UOT into a deep DA model (Figure 1) by minimizing:  
$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \gamma \cdot \text{UOT}(\mu, \nu; \lambda_1, \lambda_2),
$$  
where $\mathcal{L}_{\text{cls}}$ is the source classification loss, and $\gamma$ balances alignment and task performance.  

**Algorithmic Steps**  
1. **Input**: Source features $\{x_i^s\}$, labels $\{y_i^s\}$; target features $\{x_j^t\}$.  
2. **Step 1**: Compute initial pseudo-labels $\hat{y}_j^t$ for the target using a source-trained classifier.  
3. **Step 2**: Estimate target label proportions $\hat{\mathbf{b}}$ via $\hat{b}_c = \frac{1}{m} \sum_j \mathbb{I}(\hat{y}_j^t = c)$.  
4. **Step 3**: Train $g_\theta$ to predict $\lambda_1, \lambda_2$ using $\hat{\mathbf{b}}$ and feature statistics.  
5. **Step 4**: Solve UOT with adaptive $\lambda_1, \lambda_2$ using Sinkhorn-Knopp iterations.  
6. **Step 5**: Update feature extractor $f_\phi$ and classifier $h_\psi$ using $\mathcal{L}$.  

#### **2.3 Experimental Design**  

**Baselines**  
- Standard OT (Cuturi, 2013)  
- Fixed UOT (Fatras et al., 2021)  
- COOT (Tran et al., 2022)  
- LabelShift (Rakotomamonjy et al., 2020)  

**Evaluation Metrics**  
1. **Target Accuracy**: Classification accuracy on the target domain.  
2. **Wasserstein Distance**: $\mathcal{W}_2(\mu, \nu)$ between aligned features.  
3. **Parameter Sensitivity**: Ablation on fixed vs. learned $\lambda_1, \lambda_2$.  
4. **Convergence Analysis**: Training stability and time complexity.  

**Implementation Details**  
- **Network Architecture**: ResNet-50 backbone, 2-layer MLP for $g_\theta$.  
- **Optimization**: Adam with learning rate $10^{-4}$, $\gamma=0.1$.  
- **UOT Solver**: Shorn-Knopp with 100 iterations, entropy regularization $\epsilon=0.1$.  

---

### 3. **Expected Outcomes**  

1. **Improved DA Performance**: A-UOT will outperform fixed UOT and OT baselines on label-shifted benchmarks (e.g., +5% accuracy on Office-Home).  
2. **Robustness to Label Shifts**: The learned $\lambda_1, \lambda_2$ will correlate with the severity of label shift, enabling automatic adaptation.  
3. **Theoretical Insights**: Analysis of convergence guarantees for the joint optimization of $\lambda_1, \lambda_2$ and model parameters.  
4. **Open-Source Toolbox**: Release code and pretrained models for reproducibility.  

---

### 4. **Impact**  

This work will advance DA by providing a principled solution to label shift, a pervasive challenge in real-world applications. By integrating adaptive UOT into deep learning, the framework will:  
- **Enhance Medical Diagnostics**: Improve model generalization across hospitals with varying disease prevalence.  
- **Enable Safer Autonomous Systems**: Adapt to dynamic environments (e.g., sudden changes in pedestrian density).  
- **Reduce Manual Tuning**: Automate parameter selection, making OT more accessible to non-experts.  

The proposed method will also inspire future research on *dynamic OT*, where cost functions and constraints evolve with data.  

---

**Proposed Timeline**  
- **Months 1–3**: Implement baseline methods and A-UOT prototype.  
- **Months 4–6**: Conduct ablation studies on synthetic label shifts.  
- **Months 7–9**: Evaluate on real-world datasets (Camelyon17, SHIFT).  
- **Months 10–12**: Theoretical analysis and code/documentation release.  

**Budget Overview**  
- **Computational Resources**: $5,000 for GPU clusters.  
- **Datasets**: $2,000 for licensing medical imaging data.  
- **Personnel**: $50,000 for graduate researcher support.  

---

**Conclusion**  
By bridging the gap between unbalanced OT and adaptive parameter learning, this research will deliver a robust, scalable framework for domain adaptation under label shift, with broad implications for machine learning in non-stationary environments.