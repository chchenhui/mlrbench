Title:  
Causal Structure‐Aware Domain Generalization via Invariant Mechanism Learning

1. Introduction  
Background  
In real‐world applications, machine learning models often face data distributions at test time that differ from those observed during training. This phenomenon, known as distribution shift, can dramatically degrade performance in safety‐critical domains such as medical imaging, autonomous driving, and remote sensing. Domain generalization (DG) addresses this challenge by training models on multiple source domains and evaluating on unseen target domains. Despite extensive work—ranging from domain‐invariant feature learning (IRM, CORAL, DANN) to meta‐learning frameworks (MLDG, Reptile)—empirical risk minimization (ERM) remains surprisingly hard to beat in many benchmarks (e.g., DomainBed). A core reason is that existing DG methods often latch on to spurious correlations that vary across domains, rather than truly causal, invariant mechanisms of data generation.

Causal mechanisms, by contrast, govern the true data‐generating process and are stable under interventions or shifts that do not alter the underlying causal graph. If we can discover and learn representations aligned with these invariant causal structures, we can hope to build models that generalize robustly to new domains. However, two major challenges arise: (1) inferring causal graphs from purely observational, multi‐domain data with unknown interventions; (2) integrating learned causal structure into deep representation learning in a differentiable, scalable manner.

Research Objectives  
We propose a novel framework—Causal Structure‐Aware Domain Generalization (CSADG)—that jointly performs causal discovery and invariant representation learning. Our objectives are:  
• Infer a domain‐agnostic causal graph over latent variables using domain‐level metadata and multi‐domain observations.  
• Learn a feature extractor whose representation aligns with the inferred causal mechanisms, penalizing dependencies on non‐causal, domain‐specific factors.  
• Provide theoretical guarantees on invariance under domain shifts and derive generalization bounds.  
• Validate empirically on standard DG benchmarks and a real‐world medical imaging dataset, demonstrating improvements over ERM and state‐of‐the‐art DG methods.

Significance  
By explicitly modeling and exploiting causal mechanisms, our work aims to overcome the brittleness of existing DG techniques. In applications like autonomous driving—where encountering novel weather or lighting conditions is inevitable—or multi‐center medical diagnosis—where scanner brands and patient demographics shift—our approach can yield reliable and interpretable predictions. Moreover, integrating causal discovery with deep learning contributes to bridging the gap between causality theory and practical representation learning.

2. Methodology  

2.1 Problem Setup and Notation  
Let 𝔈={e₁,…,eₖ} denote K source domains. From each domain e we have data D_e={(x_i^e,y_i^e)}_{i=1}^{n_e}, where x_i^e∈ℝ^d is an input vector, y_i^e∈{1,…,C} is a label, and domain metadata d_i^e=e. Our goal is to learn a predictor f:ℝ^d→{1,…,C} from ⋃_e D_e that performs well on unseen domains e* not in 𝔈. We posit there exists a latent structural causal model (SCM) over latent variables h∈ℝ^m and label y:

  h = g(h, ε)  
  y = π(h, ζ)  

where g encodes causal mechanisms among components of h, ε,ζ are noise variables. The true causal graph G (with adjacency matrix A∈{0,1}^{m×m}) is invariant across domains, whereas observational distributions of x and y may shift due to interventions on non‐causal factors or changes in noise distributions.

2.2 Overview of the Framework  
Our CSADG framework comprises three main components:

1. Latent Causal Graph Inference  
2. Invariant Representation Learning  
3. Joint Optimization with Theoretical Regularization  

We describe each in detail below.

2.3 Latent Causal Graph Inference  
Goal: Infer the adjacency matrix A of the latent SCM under observational multi‐domain data.  
Approach: We adopt a differentiable extension of NOTEARS (Zheng et al., 2018), constrained by domain‐level metadata and conditional independence tests.

2.3.1 Structural Equation Model  
Assume each latent h_j is generated as a linear combination plus noise:

  $$h = Ah + ε,\quad ε\sim \mathcal{N}(0, I_m).$$

We enforce acyclicity with the constraint:

  $$h(A) = \mathrm{tr}\big(e^{A\circ A}\big) - m = 0.$$

We learn A by minimizing a reconstruction loss of inferred latents plus an ℓ₁ sparsity penalty:

  $$\mathcal{L}_{\mathrm{graph}}(A, H) = \frac{1}{N}\sum_{i,e}\big\lVert H_i^e - A\,H_i^e\big\rVert_2^2 + \lambda_{\ell_1}\lVert A\rVert_1$$  
subject to $h(A)=0$.

2.3.2 Incorporating Domain Metadata  
To reduce spurious edges driven by domain‐specific effects, we weight the reconstruction loss by domain consistency:

  $$\mathcal{L}_{\mathrm{meta}} = \frac{1}{K}\sum_{e\in\mathcal{E}}\sum_{i=1}^{n_e} w_e\big\lVert H_i^e - A\,H_i^e\big\rVert_2^2$$

where $w_e$ reflects confidence in domain e (e.g., based on sample size or known intervention strengths). This encourages edges consistent across domains.

2.4 Invariant Representation Learning  
Once A is inferred, we learn a feature extractor $f_\theta:\,x\mapsto H\in\mathbb{R}^m$ that aligns H with the SCM and emphasizes causal components.

2.4.1 Encoder–Decoder Architecture  
We employ an autoencoder structure:

• Encoder: $H = f_\theta(x)$  
• Decoder: $\hat{x} = g_\phi(H)$  

Reconstruction loss:

  $$\mathcal{L}_{\mathrm{rec}}(\theta,\phi)=\frac{1}{N}\sum_{i,e}\lVert x_i^e - g_\phi\big(f_\theta(x_i^e)\big)\rVert_2^2.$$

2.4.2 Causal Invariance Regularization  
We partition latent units into “causal” indices $\mathcal{C}$ (those with outgoing edges in A) and “non‐causal” indices $\bar{\mathcal{C}}$. To enforce invariance, we penalize dependence between non‐causal latents $H_{\bar{\mathcal{C}}}$ and domain labels. We use the Hilbert–Schmidt Independence Criterion (HSIC) as a differentiable dependence measure:

  $$\mathcal{L}_{\mathrm{inv}}(\theta)=\mathrm{HSIC}\big(H_{\bar{\mathcal{C}}}, D\big)$$

where D denotes domain metadata. Minimizing $\mathcal{L}_{inv}$ encourages $H_{\bar{\mathcal{C}}}\perp D$.

2.4.3 Prediction Loss  
We train a classifier $c_\psi:\,H_{\mathcal{C}}\mapsto \hat{y}$ on the causal latents:

  $$\mathcal{L}_{\mathrm{cls}}(\theta,\psi)=\frac{1}{N}\sum_{i,e}\ell_{\mathrm{CE}}\big(c_\psi\big(H_{i,\mathcal{C}}^e\big),y_i^e\big)$$

where $\ell_{CE}$ is cross‐entropy.

2.5 Joint Optimization  
We combine all losses into a unified objective:

  $$\min_{\theta,\phi,\psi,A}\;\mathcal{L}_{\mathrm{cls}}+\alpha\,\mathcal{L}_{\mathrm{rec}}+\beta\,\mathcal{L}_{\mathrm{inv}}+\gamma\,\mathcal{L}_{\mathrm{graph}}  
\quad\text{s.t.}\quad h(A)=0$$

Hyperparameters $\alpha,\beta,\gamma,\lambda_{\ell_1}$ trade off accuracy, reconstruction fidelity, invariance strength, and graph sparsity.

2.6 Theoretical Analysis  
Under mild assumptions (linear SCM, Gaussian noise), we show that minimizing $\mathcal{L}_{\mathrm{graph}}+\mathcal{L}_{\mathrm{inv}}$ yields representations $H_{\mathcal{C}}$ whose conditional distribution $P(y\mid H_{\mathcal{C}})$ is invariant across domains. We derive a generalization bound for unseen domain e*:

  $$\Big|R_{e^*}(f)-R_{e}(f)\Big|\leq O\big(\mathrm{TV}(P_{e^*}(H_{\bar{\mathcal{C}}}),P_{e}(H_{\bar{\mathcal{C}}}))\big)$$

where $R_e$ is the risk in domain e, showing that controlling dependence on non‐causal latents limits risk disparity.

2.7 Experimental Design  

Datasets  
• DomainBed suite: PACS, OfficeHome, TerraIncognita, DomainNet.  
• Medical Imaging: Multi‐center chest X‐ray dataset with hospital‐level domain labels.

Protocol  
• Leave‐one‐domain‐out cross‐validation: train on K–1 domains, test on held‐out domain.  
• 5 random seeds per split.  
• Baselines: ERM, IRM, GroupDRO, CORAL, DANN, Mixup, MLDG.  

Metrics  
• Average accuracy across target domains.  
• Worst‐case (min‐domain) accuracy.  
• HSIC measure of residual dependence.  
• Sparsity and fitting error of inferred graph.  

Implementation Details  
• Encoder/decoder: 4‐layer CNN for images, ReLU activations.  
• Optimizer: Adam, learning rate 1e‐4.  
• Batch size: 64.  
• Hyperparameter search: grid search on validation splits.  

Ablation Studies  
• Remove $\mathcal{L}_{\mathrm{graph}}$ to test effect of causal discovery.  
• Remove $\mathcal{L}_{\mathrm{inv}}$ to test invariance penalty.  
• Vary latent dimension m and regularization weights.

3. Expected Outcomes & Impact  

Expected Outcomes  
1. Improved Domain Generalization: We anticipate that CSADG will outperform ERM and leading DG methods by 3–7% average accuracy and reduce worst‐case drop.  
2. Interpretable Causal Graphs: The learned adjacency matrices A should recover known causal relations (validated on synthetic data with known structure) and remain stable across source domains.  
3. Theoretical Insights: Our generalization bound will formalize the benefit of invariance regularization, contributing to the theory of DG under causal assumptions.  
4. Scalability: We will demonstrate that our differentiable causal discovery scales to medium‐sized neural networks and image datasets.

Impact  
• Reliability in Safety‐Critical Systems: By focusing on invariant causal features, CSADG paves the way for robust perception in autonomous vehicles under novel weather or lighting conditions.  
• Medical Diagnostics: In multi‐center studies, models trained with CSADG should generalize across hospitals with different scanners or patient populations, reducing the risk of misdiagnosis due to dataset shift.  
• Causality and Deep Learning Integration: Our framework demonstrates a practical method to embed causal discovery in deep architectures, inspiring further research in causal representation learning.  
• Open‐Source Resources: We will release code, pretrained models, and synthetic datasets to foster reproducibility and accelerate follow‐on work.

In summary, this proposal advances domain generalization by marrying causal inference and representation learning. By discovering and leveraging invariant mechanisms, CSADG addresses the root causes of distribution shift, offering both theoretical guarantees and practical robustness for real‐world applications.