Title: Counterfactual-Augmented Contrastive Causal Representation Learning via VAE and Normalizing Flows

1. Introduction  
Background  
Modern deep models achieve impressive performance by exploiting statistical regularities in large datasets, yet they often fail under domain shift, adversarial perturbations or when asked to plan under novel configurations. This limitation stems from representations that capture correlations but do not reflect underlying causal factors. Causal representation learning (CRL) seeks to remedy this by discovering low-dimensional, high-level variables and their causal relations directly from raw observations. By integrating interventions and counterfactual reasoning into representation learning, we can hope to learn features that generalize across environments, resist adversarial attacks and support planning and transfer.  

Research Objectives  
- Develop an unsupervised learning framework that enforces each latent coordinate to correspond to a distinct causal factor, through on-the-fly simulated interventions and contrastive training.  
- Design a generative architecture combining a Variational Autoencoder (VAE) encoder, a conditional normalizing flow decoder, and a latent intervention module to produce realistic counterfactual samples.  
- Formulate a contrastive objective that (i) pulls together representations differing only along one intervened dimension, and (ii) pushes apart representations intervened along different axes.  
- Empirically validate the method on synthetic benchmarks (dSprites, CLEVR) and real-world domain-shift tasks, measuring disentanglement, out-of-distribution (OOD) robustness and downstream planning performance.  

Significance  
By marrying counterfactual interventions with contrastive learning, our approach promises to achieve unsupervised discovery of causal factors without task-specific labels. This will yield representations that are interpretable, robust to domain shifts and amenable to high-level reasoning and control. Beyond academic benchmarks, such representations could improve generalization in robotics, healthcare imaging and other safety-critical applications.  

2. Related Work  
1. Causally Disentangled Generative VAE (An et al., 2023) introduces supervised regularization for causal disentanglement, but requires labels for causal factors.  
2. Disentangled Causal VAE (Fan et al., 2023) uses supervised flows to model inter-factor dependencies, enhancing interpretability at the cost of supervision.  
3. CaD-VAE for Recommendations (Wang et al., 2023) considers causal relationships among user factors but is tailored to tabular interaction data.  
4. Causal Representation via Counterfactual Intervention (Li et al., AAAI 2024) employs reweighted losses to reduce biases but remains largely graph-based and supervised.  
5. Interventional Causal Representation Learning (Ahuja et al., 2022) proves latent identifiability under perfect interventions but assumes curated interventional data.  
6. Causal Contrastive Learning for Time (El Bouchattaoui et al., 2024) integrates CPC in temporal settings but does not address image-based unsupervised disentanglement.  

Gaps: existing methods often rely on supervision, handcrafted interventions or known causal graphs. Few combine on-the-fly latent interventions, normalizing-flow decoders and contrastive objectives in a fully unsupervised image-based CRL setting.  

3. Methodology  
3.1 Model Overview  
Our architecture consists of:  
 • Encoder $q_\phi(z\mid x)$: a standard convolutional network producing approximate posterior means $\mu(x)$ and variances $\sigma^2(x)$ for $d$ latent dimensions.  
 • Latent Intervention Module: randomly selects one coordinate $k\in\{1,\dots,d\}$ and replaces $z_k$ by an independent sample $z'_k\sim p(z_k)$, yielding a counterfactual code $z_{\text{cf}}=(z_1,\dots,z_{k-1},z'_k,z_{k+1},\dots,z_d)$.  
 • Decoder Flow $p_\psi(x\mid z)$: a conditional normalizing flow that maps $(z,u)\mapsto x$, where $u\sim\mathcal{N}(0,I_u)$ is an auxiliary variable. The log-likelihood is  
$$
\log p_\psi(x\mid z)=\log p_u(u)+\log\Bigl|\det\frac{\partial g_\psi(z,u)}{\partial u}\Bigr|\quad\text{with }x=g_\psi(z,u).
$$  

3.2 Generative and Inference Objectives  
We optimize a combined objective:  
$$
\mathcal{L}=\mathcal{L}_{\rm ELBO}+\lambda_{\rm flow}\,\mathcal{L}_{\rm flow}+\lambda_{\rm ctr}\,\mathcal{L}_{\rm ctr}.
$$  
(1) ELBO on original data:  
$$
\mathcal{L}_{\rm ELBO}=-\mathbb{E}_{q_\phi(z\mid x)}\bigl[\log p_\psi(x\mid z)\bigr]+\mathrm{KL}\bigl(q_\phi(z\mid x)\parallel p(z)\bigr).
$$  
(2) Flow reconstruction loss on counterfactuals:  
$$
\mathcal{L}_{\rm flow}=-\mathbb{E}_{q_\phi(z\mid x)}\bigl[\log p_\psi(x_{\rm cf}\mid z_{\rm cf})\bigr].
$$  
(3) Contrastive causal loss: we encode each latent code $z$ (and $z_{\rm cf}$) through a projection head $h_\omega:\mathbb{R}^d\to\mathbb{R}^m$ yielding representations $r=h_\omega(z)$, $r'=h_\omega(z_{\rm cf})$. Let $\mathrm{sim}(r_i,r'_j)=r_i^\top r'_j/\|r_i\|\|r'_j\|$. For each intervened axis $k$, the per-sample loss is  
$$
\ell_{\rm ctr}(k)= -\log\frac{\exp(\mathrm{sim}(r_k,r'_k)/\tau)}{\sum_{j=1}^d\exp(\mathrm{sim}(r_k,r'_j)/\tau)}.
$$  
Hence  
$$
\mathcal{L}_{\rm ctr}=\mathbb{E}_{x,k\sim\mathrm{Unif}[1,d]}\bigl[\ell_{\rm ctr}(k)\bigr].
$$  

3.3 Training Procedure  
For each minibatch $\{x^{(i)}\}_{i=1}^N$:  
1. Encode $q_\phi(z^{(i)}\mid x^{(i)})$ and sample $z^{(i)}$.  
2. Sample $k^{(i)}\sim\mathrm{Unif}\{1,\dots,d\}$ and $z'^{(i)}_{k^{(i)}}\sim p(z_{k})$, form $z_{\rm cf}^{(i)}$.  
3. Decode $x_{\rm rec}^{(i)}\sim p_\psi(x\mid z^{(i)})$ and $x_{\rm cf}^{(i)}\sim p_\psi(x\mid z_{\rm cf}^{(i)})$, compute log-likelihoods via normalizing flow.  
4. Compute $\mathcal{L}_{\rm ELBO},\mathcal{L}_{\rm flow},\mathcal{L}_{\rm ctr}$ and backpropagate gradients to update $\{\phi,\psi,\omega\}$.  

3.4 Experimental Design  
Datasets & Domain Shifts  
• Synthetic: dSprites (shape, scale, orientation) and CLEVR (object color, shape, size). Simulate domain shifts by altering marginal distributions of one factor at test time (e.g., unseen scale range or color combinations).  
• Real-world: RotatedMNIST and ColoredMNIST for digit domain variables; robotic simulation (e.g., CARLA driving scenarios) with changes in lighting, weather or camera angle.  

Evaluation Metrics  
• Disentanglement: Mutual Information Gap (MIG), SAP score and DCI Disentanglement.  
• Reconstruction: Mean squared error and Negative Log-Likelihood.  
• OOD Robustness: classification/regression accuracy drop between training and shifted test distributions.  
• Downstream Planning: integrate learned encoder into a simple planner or RL agent (e.g., PPO) that receives $z$ as state; measure cumulative reward on novel tasks (e.g., object manipulation with unseen object shapes).  

Baselines  
• $\beta$-VAE, FactorVAE for unsupervised disentanglement.  
• Interventional CRL (Ahuja et al.) with supervised interventions.  
• Causal Flow VAE (Fan et al.).  
• CPC-based contrastive representation learning (Oord et al.).  

Hyperparameters & Implementation  
• Latent dimension $d\in\{6,10\}$, projection head dimension $m=128$, temperature $\tau=0.1$.  
• Weights $\lambda_{\rm flow},\lambda_{\rm ctr}$ tuned on held-out validation set.  
• Encoder/decoder networks implemented in PyTorch; normalizing flows via RealNVP.  
• Training with Adam, learning rate $1\mathrm{e}{-4}$, batch size 128, for 200 epochs.  

4. Expected Outcomes & Impact  
We anticipate that Counterfactual-Augmented Contrastive CRL will:  
• Yield latent variables that accurately align with true generative factors, achieving higher MIG/SAP/DCI scores than baselines by at least 10 %.  
• Produce robust reconstructions and generative samples, with negligible likelihood degradation under domain shifts.  
• Demonstrate markedly smaller performance drops (e.g., <5 % vs. >20 % for baselines) when transferring to shifted test domains.  
• Empower downstream planners to generalize to novel environments, showing >15 % gain in cumulative reward over representation baselines.  

Broader Impact  
Our fully unsupervised approach reduces reliance on labels or pre-specified interventions, facilitating scalable discovery of causal features in visual data. The learned representations can improve safety and reliability in robotics, autonomous driving and medical imaging by ensuring robust generalization and interpretable latent factors. By releasing code and pretrained models, we aim to provide a benchmark for future causal representation learning research.