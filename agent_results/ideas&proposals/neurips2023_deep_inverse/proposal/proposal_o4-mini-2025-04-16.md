Title: Meta-Learning Robust Deep Solvers for Inverse Problems under Forward Model Uncertainty

1. Introduction  
Inverse problems—recovering unknown signals or images $x$ from indirect measurements $y$—arise in diverse applications such as medical tomography, seismic imaging, and computational photography. Formally, we model measurements as  
$$y = \mathcal{F}(x; \theta^*) + \varepsilon,$$  
where $\mathcal{F}(\cdot; \theta^*)$ is a (possibly nonlinear) forward operator parameterized by true but unknown parameters $\theta^*$, and $\varepsilon$ is measurement noise. Classical reconstruction methods exploit known physics of $\mathcal{F}$ and hand‐crafted regularization, while recent deep learning (DL) solvers learn rich data priors and inversion maps, achieving dramatic improvements in speed and quality. However, DL solvers typically rely on precise knowledge of the forward operator and its noise model. In real‐world settings—calibrated imaging devices, geophysical surveys, or industrial sensors—model parameters can drift due to temperature, mechanical wear, or simplified modeling assumptions. Such forward‐model mismatch degrades DL reconstructions and undermines reliability in high‐stakes domains.

Research Objectives  
This proposal aims to develop a meta‐learning framework that trains DL‐based inverse solvers to be robust across a distribution of forward models. Instead of optimizing for a single nominal operator $\mathcal{F}(\cdot;\theta_0)$, we treat $\theta$ as a random variable drawn from a known uncertainty distribution $p(\theta)$ (reflecting, e.g., small calibration shifts or unmodeled physics). Our objectives are:  
1. Formulate a meta‐learning objective that encourages fast adaptation and good average‐case performance across $\theta\sim p(\theta)$.  
2. Design neural architectures—e.g., unrolled proximal gradient networks or diffusion‐prior‐based reconstructor—that efficiently share statistical strength across tasks while allowing task‐specific fine‐tuning.  
3. Theoretically analyze convergence and generalization under model uncertainty.  
4. Empirically validate on simulated and real‐data inverse problems (e.g., CT tomography with angle perturbations, seismic imaging with velocity uncertainty), comparing to standard single‐model training and uncertainty‐aware baselines.

Significance  
A robust inverse solver that degrades gracefully under forward‐model perturbations will increase trust and reliability in DL‐based reconstruction, enabling broader deployment in medicine, nondestructive evaluation, and beyond. By unifying meta‐learning with inverse‐problem solvers, we bridge communities in computational imaging, machine learning, and applied mathematics, advancing both theory and practice.

2. Related Work and Literature Review  
Recent literature has begun to address model mismatch and uncertainty in DL solvers for inverse problems:

– Guan et al. (2024) “Solving Inverse Problems with Model Mismatch using Untrained Neural Networks within Model‐based Architectures” propose embedding an untrained residual block to jointly fit unknown forward‐model errors and the reconstruction. This model‐based adaptation improves robustness but lacks task‐agnostic meta‐learning.  
– Wu et al. (2024) “Uncertainty Quantification for Forward and Inverse Problems of PDEs via Latent Global Evolution (LE‐PDE‐UQ)” integrate UQ into DL surrogates using latent latent trajectories; they quantify predictive uncertainty but do not meta‐train across model distributions.  
– Khorashadizadeh et al. (2022) “Deep Variational Inverse Scattering” introduce a Bayesian U‐Net with normalizing flows to sample posterior reconstructions, addressing uncertainty but assuming a fixed forward operator.  
– Barbano et al. (2020) “Quantifying Model Uncertainty in Inverse Problems via Bayesian Deep Gradient Descent” extend gradient descent networks in a Bayesian framework, yielding uncertainty estimates for fixed models.  
– Physics‐Informed Neural Networks (PINNs, 2025) incorporate PDE constraints into network loss functions, improving robustness to noisy data but again assume known physics.  

Key Gaps  
1. None of these methods meta‐learn across a *distribution* of forward models.  
2. Existing UQ approaches quantify uncertainty post-hoc rather than shape the reconstructor’s training for robust adaptation.  
3. Computational cost of per‐task retraining or Bayesian inference remains high; meta‐training promises efficiency.  

3. Methodology  
We propose a meta‐learning pipeline that trains a shared initialization of network parameters to optimize for rapid adaptation to any forward model sampled from $p(\theta)$.  

3.1 Problem Setting and Notation  
– Let $\mathcal{T} = \{T_i\}$ be tasks indexed by forward‐model parameter $\theta_i$.  
– For each task $T_i$, we have a (synthetic or real) dataset of paired examples  
  $$\mathcal{D}_i = \{(x_{ij}, y_{ij}): y_{ij} = \mathcal{F}(x_{ij};\theta_i) + \varepsilon_{ij}\}_{j=1}^{N}.$$  
– We assume a known or estimated distribution $p(\theta)$ over $\theta$.  

Our goal is to learn a neural reconstructor $R(x;\phi)$ parameterized by $\phi$ that achieves low expected reconstruction loss  
$$\mathbb{E}_{\theta\sim p(\theta)}\left[\mathcal{L}_{\theta}(\phi)\right],$$  
where  
$$\mathcal{L}_{\theta}(\phi) \;=\; \frac{1}{N_{\text{test}}}\sum_{(x,y)\sim \mathcal{D}_{\theta}^{\text{test}}}\ell\bigl(R(y;\phi),\,x\bigr)$$  
and $\ell(\cdot,\cdot)$ is a per‐sample loss (e.g., mean squared error).

3.2 Meta‐Learning Objective  
We adopt a Model‐Agnostic Meta‐Learning (MAML) style formulation. Each meta‐training iteration proceeds as follows:

1. Sample a batch of tasks $\{\theta_i\}_{i=1}^B$ from $p(\theta)$.  
2. For each task $\theta_i$:  
   a. Split $\mathcal{D}_i$ into support set $S_i$ and query set $Q_i$.  
   b. Compute task‐specific adaptation by one or more gradient steps:  
      $$\phi_i' = \phi - \alpha \nabla_{\phi} \Bigl[\mathcal{L}_{\theta_i}^S(\phi)\Bigr],$$  
      where $\mathcal{L}_{\theta_i}^S(\phi)$ is the loss on $S_i$.  
   c. Evaluate adapted parameters on $Q_i$: compute $\mathcal{L}_{\theta_i}^Q(\phi_i')$.  
3. Update the meta‐parameters $\phi$ by minimizing the aggregate query loss:  
   $$\phi \leftarrow \phi - \beta \nabla_\phi \sum_{i=1}^B \mathcal{L}_{\theta_i}^Q\bigl(\phi - \alpha \nabla_{\phi}\mathcal{L}_{\theta_i}^S(\phi)\bigr).$$  

Here, $\alpha$ and $\beta$ are inner‐ and outer‐loop learning rates. We may extend to multiple inner steps or higher‐order extensions.  

3.3 Network Architectures  
We will explore two families of networks as the meta‐learner:  
A. Unrolled Gradient Networks  
   – Based on $K$ iterations of a learned proximal gradient method.  
   – At iteration $k$:  
     $$x^{(k+1)} = x^{(k)} - \eta_k \nabla_{x}\bigl\|\mathcal{F}(x^{(k)};\theta_i)-y\bigr\|^2_2 + \mathcal{P}_{\psi_k}\bigl(x^{(k)}\bigr),$$  
     where $\mathcal{P}_{\psi_k}$ is a CNN‐based regularizer with parameters $\psi_k$.  
   – All $\{\eta_k,\psi_k\}_{k=1}^K$ are meta‐learned.  

B. Diffusion‐Model Priors  
   – We pretrain a diffusion model $D_{\omega}(x)$ on clean data to serve as a learned prior.  
   – Inversion solves  
     $$\min_x \|y - \mathcal{F}(x;\theta)\|^2_2 + \lambda\,\mathrm{KL}\bigl(q_t(x)\,\|\,p_t(x)\bigr),$$  
     using annealed projections guided by $D_{\omega}$.  
   – The meta‐learner will adapt step sizes and noise schedules for robust inversion across $\theta$.  

3.4 Dataset and Forward‐Model Distribution  
We will conduct experiments on:  
1. CT Tomography  
   – Clean images: Shepp–Logan phantoms, clinical CT images (public CT datasets).  
   – Forward operator: Radon transform with perturbed angle distributions, calibration‐offset generation.  
   – Noise: Gaussian with SNRs in [20,40] dB and Poisson noise.  
2. Seismic Imaging  
   – Synthetic subsurface models from Marmousi benchmark; real field data.  
   – Forward PDE solver parameterized by velocity models perturbed by random velocity shifts ($\pm5$–10%).  
   – Noise: realistic measurement noise models.  

We sample $\theta$ by drawing calibration parameters (angles, offsets, velocity multipliers) from a multivariate Gaussian $p(\theta)=\mathcal{N}(\theta_0,\Sigma)$.  

3.5 Experimental Design and Evaluation Metrics  
We will compare:  
– Baseline DL model trained on nominal $\theta_0$ (no meta‐training).  
– Joint‐adaptation approach (Guan et al. untrained residual block).  
– Bayesian DL gradient descent (Barbano et al.).  
– Our meta‐learning solvers (unrolled network, diffusion‐prior network).  

Evaluation metrics include:  
– Peak Signal‐to‐Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Normalized Mean Squared Error (NMSE).  
– Calibration of uncertainty estimates: Expected Calibration Error (ECE).  
– Adaptation speed: number of gradient steps vs performance on new $\theta$.  
– Runtime and memory overhead.  

Statistical Validation  
For each model and task distribution, we will run $n=30$ random trials, report mean±std of metrics, and conduct paired t‐tests to assess significance ($p<0.05$).  

Implementation Details  
– Meta‐training in PyTorch on multi‐GPU clusters; learning rates $\alpha\in\{10^{-3},10^{-4}\}$, $\beta\in\{10^{-4},10^{-5}\}$.  
– Inner‐loop batch size $|S_i|=16$, query size $|Q_i|=16$, tasks per batch $B=8$.  
– Total meta‐iterations $\sim20{,}000$.  
– Code and preprocessed data to be released under an open‐source license.  

4. Expected Outcomes & Impact  
We anticipate the following outcomes:  
1. A meta‐learning procedure that yields DL inverse solvers with substantially improved robustness to forward‐model perturbations, demonstrated by higher PSNR/SSIM on unseen $\theta$ and fewer adaptation steps.  
2. Theoretical insights into the convergence behavior of meta‐training under model uncertainty, including generalization bounds that scale with the variance of $p(\theta)$.  
3. A comprehensive benchmark suite—datasets, forward‐model generators, and code—for future evaluations of robust inverse‐problem solvers.  
4. Open‐source release of trained models and code, fostering reproducible research and adoption by applied communities.  

Broader Impact  
– Medical imaging: more reliable reconstructions in low‐resource clinics with imperfect calibration.  
– Geophysics: stable inversion across variable subsurface properties, improving exploration and monitoring.  
– Scientific computing: a blueprint for meta‐learning under physical model uncertainty applicable to PDE‐constrained inverse problems.  

In summary, this research will deliver both novel algorithms and practical tools to make DL‐based inverse solving trustworthy in the face of inevitable model uncertainties, accelerating real‐world impact across science and engineering.