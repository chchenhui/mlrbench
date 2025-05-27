Title:
Conditional Neural Operator for Probabilistic Inverse Modeling in Turbulent Flows

Introduction:
Background
Accurate inference of hidden parameters in high‐dimensional partial differential equation (PDE) systems is a core challenge in computational science and engineering. Turbulent flow fields, governed by the Navier–Stokes equations, exhibit complex spatiotemporal interactions across scales. Inverse problems—recovering initial or boundary conditions, forcing terms, or model parameters from sparse observations—are critical for flow control, design optimization, and uncertainty quantification. Conventional methods rely on repeated calls to expensive high‐fidelity solvers or sampling‐based inference (e.g., MCMC), leading to high computational costs and limited applicability in real‐time settings. Moreover, they often provide only point estimates or poorly calibrated uncertainty measures under data scarcity.

Research Objectives
1. Develop a conditional neural operator (CNO) framework that jointly learns:
   • A differentiable surrogate for the forward Navier–Stokes solution map.  
   • An amortized posterior over input parameters given sparse flow observations.  
2. Incorporate fast Fourier Neural Operators (FNOs) to encode the PDE structure and leverage spectral representations.  
3. Employ conditional normalizing flows (CNFs) for flexible, tractable posterior modeling capable of capturing multimodal and non‐Gaussian distributions.  
4. Design an end‐to‐end training scheme based on variational inference that aligns the surrogate and posterior modules.  
5. Rigorously validate the CNO on synthetic turbulent flow datasets and benchmark against state‐of‐the‐art inverse methods.

Significance
This project will bridge the gap between simulation and real‐time inverse modeling by offering a fast, differentiable, and uncertainty‐aware inference engine. The CNO’s ability to produce calibrated posterior samples in milliseconds enables:
  • Real‐time decision making in flow control (e.g., drag reduction).  
  • Gradient‐based design optimization through backpropagation in the surrogate.  
  • Quantification and control of epistemic and aleatoric uncertainties in complex physical systems.  

Literature Review:
Recent work on neural surrogates and probabilistic inverse problems has laid the foundation for our approach:

1. CoNFiLD (Du et al., 2024) demonstrated rapid probabilistic generation of 3D turbulent flows in irregular domains via conditional neural field encoding and latent diffusion. However, their focus is generative simulation rather than posterior inference under scarce observations.

2. IUFNO (Wang et al., 2024) introduced an implicit U‐Net enhanced Fourier Neural Operator for long‐term prediction of channel flows. They achieved high accuracy and efficiency but did not address inverse modeling or uncertainty quantification.

3. Oommen et al. (2024) integrated neural operators with diffusion models to improve spectral fidelity in turbulence. While this enhances forward modeling, it lacks end‐to‐end posterior inference capabilities.

4. Haitsiukevich et al. (2024) used diffusion‐based generative models as neural operators for recovering unobserved states in dynamical systems. Their results point to the power of diffusion processes but do not directly tackle high‐dimensional PDE parameter inversion under uncertainty.

Key challenges emerging from these studies include high‐dimensional inverse inference, data scarcity, rigorous uncertainty quantification, simulation‐to‐real generalization, and balancing accuracy with computational efficiency. Our proposed CNO unites FNO and CNF modules in a variational framework to address these challenges comprehensively.

Methodology:
Overview
We propose a two‐component architecture: (1) a forward surrogate $\mathcal{F}_\theta$ parameterized by an FNO, mapping input parameters $\kappa$ (e.g., viscosity field, boundary forcing) to flow fields $u=(u_x,u_y,p)$; (2) a conditional normalizing flow $q_\phi(z\,|\,\kappa_{\mathrm{obs}})$ that models the posterior over latent codes $z$, which map back to $\kappa$. Observations $\kappa_{\mathrm{obs}}$ are sparse measurements of $u$ at sensor locations.

1. Data Generation and Preprocessing
  • Generate a synthetic dataset $\mathcal{D}=\{(\kappa^{(i)},u^{(i)})\}_{i=1}^N$ via a high‐fidelity Navier–Stokes solver over a rectangular domain $\Omega$.  
  • Parameter sampling: draw $\kappa^{(i)}$ from a prior distribution $p(\kappa)$ (e.g., Gaussian random fields with varying correlation length scales and amplitude).  
  • Solve $\mathrm{NS}(u;\kappa)=0$ with standard DNS. Store full‐field solutions on a $64\times64$ grid for $T$ time steps.  
  • Extract sparse observations $\kappa_{\mathrm{obs}}^{(i)}$ by sampling $m$ sensor points $(x_j,y_j)$ per snapshot:  
    $$\kappa_{\mathrm{obs}}^{(i)} = \{u^{(i)}(x_j,y_j)\}_{j=1}^m + \varepsilon, \quad \varepsilon\sim\mathcal{N}(0,\sigma_{\mathrm{obs}}^2).$$  

2. Forward Surrogate: Fourier Neural Operator
  • Input: discretized parameter field $\kappa\in\mathbb{R}^{64\times64\times d}$, where $d$ channels may include boundary conditions.  
  • Apply $L$ iterative FNO layers: each layer lifts to latent space, applies Fourier transform, multiplies by learnable spectral weights, and applies inverse transform:  
    $$
    v^{\ell+1}(x) = \sigma\Big(W v^\ell(x) + \mathcal{F}^{-1}\big(R^\ell \cdot \mathcal{F}(v^\ell)\big)(x)\Big), 
    $$
    where $v^0=\kappa$, $W$ is a local linear transform, $R^\ell$ are complex spectral weights, and $\sigma$ is a nonlinearity.  
  • The final output $v^L(x)$ is projected to $u(x)$ via a linear layer.  
  • Loss for forward surrogate: mean squared error (MSE) over training snapshots:  
    $$
    \mathcal{L}_{\mathrm{F}}(\theta) = \frac{1}{N\,T\,n_x\,n_y}\sum_{i,t}\|u^{(i)}_t - \mathcal{F}_\theta(\kappa^{(i)})_t\|_2^2.
    $$  

3. Posterior Modeling: Conditional Normalizing Flow
  • Introduce latent variable $z\sim p(z)=\mathcal{N}(0,I)$ of dimension $d_z$.  
  • Define a sequence of invertible transforms $f_{\phi}$ conditioned on $\kappa_{\mathrm{obs}}$:  
    $$
    \kappa = f_{\phi}(z;\kappa_{\mathrm{obs}}),\quad z = f^{-1}_{\phi}(\kappa;\kappa_{\mathrm{obs}}).
    $$
  • The conditional density is  
    $$
    q_{\phi}(\kappa\,|\,\kappa_{\mathrm{obs}}) = p(z)\,\Big|\det\,\frac{\partial z}{\partial \kappa}\Big|.
    $$
  • We parameterize $f_{\phi}$ via coupling layers or masked autoregressive flows, with conditioning networks that embed $\kappa_{\mathrm{obs}}$ into feature vectors.  

4. Joint Training via Amortized Variational Inference
  • Objective: maximize the evidence lower bound (ELBO) for each training pair:  
    $$
    \mathcal{L}_{\mathrm{ELBO}}(\theta,\phi) = \mathbb{E}_{q_{\phi}(z|\kappa,\kappa_{\mathrm{obs}})}\big[\log p_\theta(\kappa|\kappa_{\mathrm{obs}},z)\big] - \mathrm{KL}\big(q_{\phi}(z|\kappa,\kappa_{\mathrm{obs}})\,\|\,p(z)\big).
    $$
  • Here $p_\theta(\kappa|\kappa_{\mathrm{obs}},z)$ is implicitly defined by requiring $\mathcal{F}_\theta(\kappa)$ to match observed $\kappa_{\mathrm{obs}}$. We implement this via a reconstruction loss on the sur­ro­gate’s predictions at sensor locations:  
    $$
    \log p_\theta(\kappa_{\mathrm{obs}}\,|\,\kappa) \approx -\frac{1}{2\sigma_{\mathrm{obs}}^2}\|\kappa_{\mathrm{obs}} - \mathcal{F}_\theta(\kappa)|_{\mathrm{sensors}}\|_2^2 + C.
    $$
  • Full training loss:  
    $$
    \mathcal{L}(\theta,\phi) = \mathbb{E}_{(\kappa,\kappa_{\mathrm{obs}})\sim\mathcal{D}}\Big[ \mathrm{KL}\big(q_{\phi}(z|\kappa,\kappa_{\mathrm{obs}})\,\|\,p(z)\big) + \frac{1}{2\sigma_{\mathrm{obs}}^2}\|\kappa_{\mathrm{obs}} - \mathcal{F}_\theta(f_{\phi}(z;\kappa_{\mathrm{obs}}))|_{\mathrm{sensors}}\|^2\Big] + \lambda\,\mathcal{L}_\mathrm{F}(\theta).
    $$
  • We optimize $\theta,\phi$ end‐to‐end using Adam with learning rate warm‐up and gradient clipping.  

5. Inference and Backpropagation
  • Given new sparse observations $\kappa_{\mathrm{obs}}^*$, draw posterior samples:  
    $$
    z^{(s)}\sim p(z),\quad\kappa^{(s)} = f_{\phi}(z^{(s)};\kappa_{\mathrm{obs}}^*).
    $$
  • Compute full fields $\hat u^{(s)} = \mathcal{F}_\theta(\kappa^{(s)})$.  
  • Use gradient $\nabla_\kappa J(u)$ via backprop through $\mathcal{F}_\theta$ for design objectives $J$ (e.g., minimize drag coefficient).  
  • Obtain uncertainty estimates from sample ensemble $\{\hat u^{(s)}\}$.  

6. Experimental Design and Evaluation
  • Datasets:
     – Synthetic 2D Kolmogorov flow and 3D channel flow at Reynolds numbers $Re\in[100,10^4]$.  
     – Real‐world PIV (particle image velocimetry) data for channel flow benchmarks.  
  • Baselines:
     – MCMC inversion with full Navier–Stokes solver.  
     – Ensemble Kalman Inversion (EKI).  
     – FNO‐only with Gaussian posterior (amortized Gaussian variational inference).  
     – PINN‐based inverse approach.  
  • Metrics:
     – Reconstruction error: $\mathrm{MSE}(\kappa,\hat\kappa)$ and relative $L_2$ error.  
     – Forward prediction error at unobserved locations.  
     – Negative log‐likelihood (NLL) of true parameters under predicted posterior.  
     – Continuous Ranked Probability Score (CRPS) for uncertainty calibration.  
     – Coverage of $95\%$ credible intervals.  
     – Wall‐clock runtime for training and inference.  
     – Gradient quality: cosine similarity between true and surrogate gradients for a set of test loss functions.  
  • Ablations:
     – Impact of number of sensors $m$.  
     – Posterior flow depth and latent dimension $d_z$.  
     – Weighting factor $\lambda$ in joint loss.  
     – Comparison of spectral vs. spatial FNO variants.  

Expected Outcomes & Impact:
1. Real‐time Posterior Sampling
   • Our CNO will generate calibrated posterior samples in $<10\,$ms per example, enabling real‐time decision support in control and design tasks.

2. Improved Inversion Accuracy
   • We expect reconstruction errors to improve by $20\%$–$50\%$ over EKI and amortized Gaussian baselines, especially under sensor scarcity ($m<10$).

3. Uncertainty Quantification
   • Posterior distributions will exhibit well‐calibrated credible intervals (empirical coverage within $\pm 2\%$ of nominal levels) and low CRPS relative to baselines.

4. Gradient‐Based Design Applications
   • Backpropagation through the surrogate will yield gradients that align with high‐fidelity solver gradients with cosine similarity $>0.9$, facilitating gradient‐based optimization for drag reduction and flow shaping.

5. Simulation‐to‐Real Bridging
   • When fine‐tuned on limited real‐world PIV data, our model will demonstrate robust generalization, reducing the sim2real gap in inverse tasks by at least $30\%$.

6. Open Source Release
   • We will release code, pretrained models, and a dataset of synthetic turbulent flows to catalyze research in differentiable surrogates and probabilistic inverse modeling.

Broader Impact
By delivering a fast, differentiable, and uncertainty‐aware inversion engine, this research will accelerate scientific discovery and engineering design across disciplines such as aerospace engineering (flow control), climate science (data assimilation), and biomedical flows (blood flow inference). The conditional neural operator framework is extensible to other PDE systems (e.g., elasticity, electromagnetics), paving the way for a new generation of data‐driven, probabilistic simulations and solvers.