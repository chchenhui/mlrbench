1. Title  
Causal Diffusion Models: Disentangling Latent Causal Factors in Generative AI

2. Introduction  
Background  
Over the past decade, deep generative models—particularly diffusion-based frameworks—have achieved remarkable success in modeling high-dimensional data distributions for images, video, and text. Frameworks such as Denoising Diffusion Probabilistic Models (DDPMs) and their variants capture intricate data dependencies by gradually perturbing and denoising samples. However, these architectures predominantly learn statistical associations rather than true causal relationships, making them vulnerable to spurious correlations and confounding biases. In safety-critical domains like healthcare or economics, the inability to intervene on specific causal factors limits the model’s interpretability, trustworthiness, and reliability under distributional shifts.

Causal representation learning (CRL) seeks to identify latent factors that correspond to true causal variables and infer the structural relationships among them. Recent advances in CRL (e.g., DeCaFlow, CausalBGM, C2VAE) demonstrate that coupling deep generative models with structural causal models helps mitigate confounding and enables counterfactual reasoning. Yet, most existing proposals handle causal discovery and generative modeling in isolation or under restrictive assumptions (fully observed causal graph, limited confounders, low‐dimensional data).

Research Objectives  
This proposal aims to develop a unified framework, called Causal Diffusion Models (CDMs), that:  
• Discovers a latent structural causal graph among disentangled latent factors.  
• Integrates causal structure directly into the diffusion denoising process.  
• Enables controlled and counterfactual generation by intervening on specific latent causes.  
• Demonstrates improved robustness to confounding, distribution shifts, and enhanced interpretability in real-world applications.

Significance  
By embedding causal graph constraints within a state-of-the-art diffusion architecture, CDMs will advance the frontier of trustworthy generative AI. In biomedical imaging, for example, users could systematically vary disease severity while preserving patient anatomy—opening new avenues for data augmentation, hypothesis testing, and fair synthetic data generation. More broadly, CDMs promise to set a new paradigm for deep generative models that respect underlying causal mechanisms.

3. Literature Review  
1. DeCaFlow (Almodóvar et al., 2025)  
   – Introduces a causal generative model leveraging observational data and known graph structure to adjust for hidden confounders via proxy variables.  
   – Demonstrates identifiability of causal queries under weak assumptions and improved sample quality.  
2. CausalBGM (Liu & Wong, 2025)  
   – A Bayesian generative model for estimating individualized treatment effects in high-dimensional observational data.  
   – Utilizes latent features to disentangle treatment and outcome drivers, providing robust uncertainty quantification.  
3. C2VAE (Zhao et al., 2024)  
   – Combines structural causal models with a variational autoencoder to learn both correlation and causation among latent properties.  
   – Employs a novel pooling mechanism to capture inter-property correlations, enabling property-controlled generation.  
4. Survey on Causal Generative Modeling (Komanduri et al., 2023)  
   – Reviews theoretical foundations and applications of causal representation learning in deep generative models.  
   – Identifies open challenges: latent variable identifiability, confounder handling, interpretability, robustness, and scalability.

Key challenges emerging from this literature include:  
– Identifying and disentangling true latent causal variables in high dimensions.  
– Accounting for unobserved confounders to avoid biased generation.  
– Embedding causal structure in continuous‐time generative processes.  
– Designing efficient training algorithms that scale to large datasets.

4. Methodology  
We propose CDMs with three core components: (A) a latent encoder/decoder pair, (B) a causal discovery module, and (C) a causally-guided diffusion process.  

4.1 Model Architecture  
– Encoder $E_\phi$: maps observed data $x\in\mathbb{R}^D$ to a $d$-dimensional latent code $z=E_\phi(x)$.  
– Decoder $D_\psi$: reconstructs data $\hat x=D_\psi(z)$.  
– Diffusion noise‐predictor $\epsilon_\theta$: parameterizes the backward denoising steps conditioned on latent code $z$ and causal graph $G(A)$.

4.2 Causal Discovery Module  
We model the latent causal variables $z\in\mathbb{R}^d$ via a linear Structural Equation Model (SEM):  
$$z = A^\top z + n,\quad n\sim\mathcal{N}(0,I_d)\,. \tag{1}$$  
Here, $A\in\mathbb{R}^{d\times d}$ is a weighted adjacency matrix with zeros on the diagonal and encodes a directed acyclic graph (DAG). We adopt a NOTEARS‐style formulation (Zheng et al., 2018) to learn $A$:  
Minimize  
$$\mathcal{L}_{\mathrm{causal}}(\phi,A)\;=\;\frac{1}{N}\sum_{i=1}^N\|z^{(i)}-A^\top z^{(i)}\|_2^2 \;+\;\lambda\|A\|_1$$  
subject to the acyclicity constraint  
$$h(A)=\operatorname{tr}\big(e^{A\circ A}\big)-d=0\,, \tag{2}$$  
where $\circ$ denotes the Hadamard product. We optimize jointly over $(\phi,A)$ by alternating updates:  
1. Encode minibatch $\{x^{(i)}\}$ to $\{z^{(i)}\}$ via $E_\phi$.  
2. Update $A$ with projected gradient descent to satisfy $h(A)=0$.  
3. Backpropagate causal loss into encoder parameters $\phi$.

If interventional or domain‐knowledge constraints are available (e.g., known edges or partial ordering), they are incorporated as hard constraints on $A_{ij}$.

4.3 Causal Diffusion Process  
We extend standard DDPM denoising to respect causal dependencies among latent variables. A vanilla diffusion step is:  
$$z_t = \sqrt{1-\beta_t} \,z_{t-1} + \sqrt{\beta_t}\,\eta_t,\quad \eta_t\sim\mathcal{N}(0,I)\,. $$  
In CDMs, we modify the reverse process to incorporate $A$:  
$$\hat z_{t-1} \;=\;\frac{1}{\sqrt{\alpha_t}}\Bigl(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(z_t,t\,|\,G)\Bigr)\;+\;\kappa\,A^\top z_t\,, \tag{3}$$  
where $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. The term $\kappa\,A^\top z_t$ injects causal structural corrections, encouraging removal or addition of noise according to causal parents.  

The diffusion loss is:  
$$\mathcal{L}_{\mathrm{diff}}(\theta)=\mathbb{E}_{t,x,\epsilon}\Bigl[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x+\sqrt{1-\bar\alpha_t}\epsilon,t\,|\,G)\|_2^2\Bigr]\,. \tag{4}$$  

Overall objective:  
$$\min_{\phi,\theta,A}\;\mathcal{L}_{\mathrm{diff}}+\lambda_1\,\mathcal{L}_{\mathrm{causal}}+\lambda_2\,h(A)\,. \tag{5}$$  

4.4 Data Collection and Preprocessing  
We will evaluate CDMs on both synthetic benchmarks and real-world datasets:  
• Synthetic disentanglement data (e.g., dSprites, MPI3D) with known causal factors.  
• Biomedical imaging: chest X-ray datasets (ChestX-ray14 with disease severity labels), single-cell microscopy with gene knockout interventions.  
• Video benchmarks where latent dynamics follow known transition graphs.  
Data are normalized, cropped to fixed resolution, and partitioned into observational and (where available) interventional subsets.

4.5 Experimental Design  
Baselines  
– Standard DDPM without causal guidance.  
– C2VAE (Zhao et al., 2024) and DeCaFlow (Almodóvar et al., 2025).  
– CausalBGM (Liu & Wong, 2025) adapted to continuous data.

Evaluation Metrics  
1. Sample Quality: Fréchet Inception Distance (FID) on generated images/videos.  
2. Disentanglement & Identifiability: Mutual Information Gap (MIG) between true factors and learned $z$.  
3. Causal Discovery Accuracy: Structural Hamming Distance (SHD) between inferred $A$ and ground-truth graph.  
4. Counterfactual Fidelity: Given an intervention $\mathrm{do}(z_j=\tilde z)$, measure change in output via classifier consistency and average causal effect (ACE) error.  
5. Robustness to Distribution Shift: Evaluate FID and ACE under domain-shifted test sets.  
6. Efficiency & Scalability: Training time per epoch, memory footprint, and convergence rate.

Ablation Studies  
– Effect of causal injection weight $\kappa$.  
– Impact of interventional data fraction.  
– Role of acyclicity penalty $\lambda_2$.

Implementation Details  
We will implement CDMs in PyTorch, leveraging distributed GPU training. Hyperparameters $(\lambda_1,\lambda_2,\kappa)$ will be selected via grid search on validation sets. Each experiment will be repeated across five random seeds to report mean±std.

5. Expected Outcomes & Impact  
We anticipate that CDMs will:  
• Achieve lower FID scores compared to non-causal diffusion and existing causal generative baselines.  
• Recover latent causal graphs with SHD significantly below chance on both synthetic and real tasks.  
• Enable high-fidelity counterfactual generation (low ACE error) and precise interventions on latent disease factors in medical images.  
• Exhibit robust performance under distribution shifts and confounding settings.

Scientific Impact  
CDMs will bridge the gap between modern diffusion models and causal representation learning, providing a principled framework for disentangling and manipulating latent causes. Methodologically, our approach generalizes SEM-based causal discovery to continuous-time generative processes, opening new research directions in causally-aware generative modeling.

Societal Impact  
In healthcare, CDMs could generate realistic synthetic data for rare conditions, augmenting training sets while ensuring patient privacy. They enable clinicians to test “what-if” scenarios (e.g., progressive disease interventions) on augmentations without risking real patients. More broadly, causally-informed generative models promise fairer and more transparent AI systems in finance, economics, and social sciences by avoiding spurious correlations.

Future Work  
Beyond this proposal, CDMs can be extended to incorporate non-linear SEMs, richer intervention schemes, and large-scale foundation models (text or multimodal). We will release code, pretrained models, and benchmark suites to foster reproducibility and community collaboration.

In conclusion, Causal Diffusion Models represent a novel synthesis of diffusion-based generative modeling and causal representation learning, with the potential to transform how we build, interpret, and trust deep generative AI.