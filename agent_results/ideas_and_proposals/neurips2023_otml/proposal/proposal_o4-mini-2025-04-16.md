1. Title  
Adaptive Unbalanced Optimal Transport for Robust Domain Adaptation under Label Shift  

2. Introduction  
Background  
Domain adaptation seeks to transfer knowledge from a labeled source domain to an unlabeled (or sparsely labeled) target domain. Optimal transport (OT) has emerged as a powerful tool for domain adaptation by aligning source and target distributions in feature space. In its classical formulation, OT enforces mass‐conservation constraints, implicitly assuming that the marginal distributions (including class‐conditional and label marginals) of source and target are identical or balanced. In practice, real‐world domain shifts often involve label shifts—that is, changes in the target’s class proportions relative to the source. Standard OT is ill‐suited to handle such label shifts: it enforces the wrong marginal and thus may misalign samples across classes, leading to negative transfer or degraded performance.

Unbalanced OT (UOT) generalizes OT by relaxing the marginal constraints via divergence penalties, allowing source and target marginals to differ. However, existing UOT‐based adaptation methods require the user to pre‐specify relaxation parameters that govern how much mass may be created or destroyed on each side. In the context of label shift, these relaxation parameters have a direct impact on the inferred target label proportions, yet in most situations the true shift is unknown and difficult to tune manually.

Research Objectives  
This proposal aims to develop an Adaptive Unbalanced Optimal Transport (A-UOT) framework that automatically learns the degree of marginal relaxation needed per class from data. We will integrate A-UOT into an end-to-end deep domain adaptation pipeline. Specifically, our objectives are:  
• Formulate a class-wise UOT problem with learnable relaxation parameters that capture label‐shift effects.  
• Embed this adaptive UOT layer within a deep feature extractor and classifier, optimizing jointly for feature alignment and parameter estimation.  
• Theoretically analyze the consistency and generalization properties of joint feature adaptation and relaxation‐parameter learning.  
• Empirically demonstrate robustness and improved performance across benchmarks exhibiting label imbalance or shift.  

Significance  
A-UOT will relieve practitioners from laborious hyperparameter tuning of marginal relaxation and will enable domain adaptation methods that are robust to unknown label shifts. This is critical for applications in medical imaging, natural language processing, and other high‐stakes domains where class proportions often vary in deployment. By learning the relaxation parameters, the model implicitly infers target label proportions, offering insights into the nature of the domain shift itself.

3. Methodology  

3.1 Problem Setup  
Let 𝒟ₛ={ (xᵢˢ,yᵢˢ) }_{i=1}^{nₛ} denote the source dataset with feature vectors xᵢˢ∈ℝᵈ and labels yᵢˢ∈{1,…,C}. Let 𝒟ₜ={ xⱼᵗ }_{j=1}^{nₜ} be the unlabeled target dataset. Denote by μₛ and μₜ the empirical feature distributions in the source and target domains over ℝᵈ. Under label shift, the target label marginal b=(b₁,…,b_C), b_c=Pₜ(y=c), differs from the source marginal a=(a₁,…,a_C), a_c=Pₛ(y=c). We seek a coupling γ∈ℝ_{+}^{nₛ×nₜ} aligning μₛ to μₜ while adapting for label mass differences across classes.

3.2 Adaptive Unbalanced OT Formulation  
We adopt an entropic UOT formulation with divergence penalties on the marginals and introduce class‐wise relaxation parameters τ=(τ₁,…,τ_C):  

Block equation start  
$$  
\min_{γ\in\mathbb R_{+}^{nₛ\times nₜ}}  
\;  
\langle γ,\,C\rangle  
\;+\;\sum_{c=1}^{C}  
\bigl[  
τ_c \,D_{\mathrm{KL}}(γ\,1_{nₜ}\bigm\|a_c\,e_{c}^{(ₛ)})  
\;+\;  
τ_c \,D_{\mathrm{KL}}(γ^{\top}1_{nₛ}\bigm\|b_c\,e_{c}^{(ₜ)})  
\bigr]  
\;-\;  
ε\,H(γ)\,.  
$$  
Block equation end  

Here:  
• C_{ij} = ∥f_θ(xᵢˢ) − f_θ(xⱼᵗ)∥²₂ is the squared‐Euclidean cost in the learned feature space f_θ:ℝᵈ→ℝᵏ.  
• D_{KL}(p‖q)=∑_i p_i log(p_i/q_i) is the Kullback–Leibler divergence.  
• a_c e_c^{(ₛ)} is a degenerate distribution placing mass a_c on source samples of class c (similarly for b_c e_c^{(ₜ)} on target pseudo‐class c).  
• H(γ)=−∑_{i,j}γ_{ij}log γ_{ij} is the entropy.  
• ε>0 is the entropic regularization weight.  
• τ_c>0 (c=1…C) are learned relaxation weights controlling the allowed class‐wise mass variation.  

By making τ learnable, the coupling γ can create or destroy mass per class in accordance with the true (unknown) shift, rather than a single global slack parameter.

3.3 End-to-End Learning  
We integrate the above A-UOT layer into a deep network comprising a feature extractor f_θ and classifier g_φ. The joint objective is:  

Inline equation  
$  
\mathcal{L}(θ,φ,τ)  
=  
\underbrace{\frac{1}{nₛ}\sum_{i=1}^{nₛ}\ell_{\mathrm{CE}}\bigl(g_φ(f_θ(xᵢˢ)),yᵢˢ\bigr)}_{\text{source classification}}  
\;+\;  
\lambda_{\mathrm{OT}}\;\mathcal{L}_{\mathrm{A\text‐UOT}}(θ,τ)  
\;+\;  
\lambda_{\mathrm{ent}}\;\mathcal{L}_{\mathrm{ent}}(θ,φ)  
\;+\;  
\lambda_{τ}\,\|τ\|_{2}^{2}\,.  
$  

Components:  
1. Source classification loss: standard cross‐entropy ℓ_{CE}.  
2. A-UOT loss: the optimal value of the UOT problem defined above, parameterized by θ and τ. We denote this value by 𝓛_{A-UOT}(θ,τ).  
3. Entropy minimization on target classifier outputs:  
Inline equation  
$  
\mathcal{L}_{\mathrm{ent}}  
=  
-\frac{1}{nₜ}\sum_{j=1}^{nₜ}\sum_{c=1}^{C}p_{jc}\log p_{jc},  
\quad  
p_{jc}= \bigl[g_φ(f_θ(xⱼᵗ))\bigr]_{c}.  
$  
This encourages confident predictions on target.  
4. Regularization on τ to avoid degenerate solutions.

Optimization proceeds by alternating (or jointly) updating θ, φ, and τ via stochastic gradient descent. The A-UOT loss is computed via a differentiable Sinkhorn‐like algorithm with class‐wise KL penalties. Gradients ∂𝓛_{A-UOT}/∂τ are obtained by automatic differentiation through the Sinkhorn iterations or via implicit differentiation of the optimality conditions.

3.4 Algorithmic Details  
1. Parameterization of τ: we set τ_c=exp(α_c) to ensure positivity, and learn α∈ℝ^C.  
2. Mini-batch UOT: at each iteration we sample mini‐batches of source and target samples. We estimate source marginals â_c from true labels and target marginals b̂_c from classifier probabilities p_{jc}.  
3. Sinkhorn with class‐wise margins: we build cost matrix C and perform the generalized Sinkhorn–Knopp scaling:  

Block equation  
$$  
u \leftarrow \frac{ â}{K\,v},  
\quad  
v \leftarrow \frac{ b̂}{K^{\top}\,u},  
\quad  
K_{ij} = \exp\bigl(-C_{ij}/ε\bigr).  
$$  
Here â∈ℝ^{batchₛ} collects weighted source marginals (duplicating τ_c factors per sample).  
4. Convergence criterion: a fixed number T of Sinkhorn iterations (e.g. T=50) or until marginals match within tolerance.  
5. Complexity: each Sinkhorn iteration costs O(m n) per mini‐batch (m source, n target). Learning τ adds O(C) overhead.  

3.5 Theoretical Analysis  
We will analyze the following aspects:  
• Consistency of τ‐estimation: under mild assumptions on feature separation, the learned τ converges to values proportional to true target‐source label‐ratio b_c/a_c.  
• Generalization bound: extending the domain adaptation bound  
Block equation  
$$  
εₜ(h)  
\le  
εₛ(h)  
+  
W_{A\text‐UOT}(Pₛ^f,Pₜ^f;\tau)  
+  
\Lambda( h ),  
$$  
where εₛ,εₜ are source/target risk; W_{A-UOT} is the optimal transport cost with learned τ, and Λ(h) is the joint optimal risk on the two domains.  
We will show that adapting τ tightens the bound in the presence of label shift.

3.6 Experimental Design  
Datasets  
• Office‐31, Office‐Home, VisDA‐2017: classical domain adaptation benchmarks. We will induce synthetic label shift by re‐sampling target classes (e.g. reduce some classes to 10% frequency).  
• Digits (MNIST→USPS, SVHN→MNIST): model low‐to‐low domain shifts with class imbalance.  
• Realistic medical imaging tasks (e.g. chest X-ray classification with varying disease prevalence).  
Baselines  
• Standard OTDA (Ganin et al. 2016; Courty et al. 2017).  
• Unbalanced OT with fixed relaxation (Fatras et al. 2021).  
• Importance‐weighted OT (Rakotomamonjy et al. 2020).  
• MixUp + OT methods (\textsc{mixunbot}, Fatras et al. 2022).  
Metrics  
• Target classification accuracy and per‐class accuracy.  
• H‐score: harmonic mean between source and target accuracies.  
• Label‐proportion estimation error: MSE between estimated b̂ and true b.  
• Computational cost: runtime per epoch.  
Ablations  
• Compare class‐wise learnable τ vs single global τ.  
• Impact of entropy loss weight λ_{ent} and τ‐regularization λ_τ.  
• Sensitivity to Sinkhorn regularization ε.  

4. Expected Outcomes & Impact  
Expected Outcomes  
• A novel A-UOT algorithm that automatically adapts marginal‐relaxation weights per class, leading to improved domain adaptation under label shifts.  
• Theoretical guarantees on consistency of τ‐learning and tighter generalization bounds.  
• A publicly released PyTorch implementation, enabling reproducibility and adoption.  
• Empirical evidence of performance gains (5–10% absolute improvement in target accuracy) over fixed‐parameter UOT and other state‐of‐the‐art methods, especially in high‐imbalance regimes.  
Scientific Impact  
• Provides a principled way to handle unknown label shift in OT‐based adaptation, filling a gap in the literature.  
• Advances understanding of how marginal relaxation interacts with feature learning in deep models.  
Practical Impact  
• Facilitates robust domain adaptation in medical diagnostics, remote sensing, and NLP tasks where class prevalence may vary unpredictably.  
• Reduces hyperparameter tuning burden for practitioners by learning relaxation automatically.  
Broader Impacts  
• Encourages fairer models by compensating for under‐represented classes.  
• Opens a pathway to joint estimation of domain shift statistics and adaptation, with applications beyond classification (e.g. object detection, segmentation).  

In summary, this proposal outlines a comprehensive plan to develop, analyze, and validate an Adaptive Unbalanced Optimal Transport framework that learns how much mass to create or destroy per class, thereby automatically correcting for label shifts in domain adaptation. The combination of theoretical analysis, algorithmic innovation, and extensive empirical validation will advance both the theory and practice of OT in machine learning.