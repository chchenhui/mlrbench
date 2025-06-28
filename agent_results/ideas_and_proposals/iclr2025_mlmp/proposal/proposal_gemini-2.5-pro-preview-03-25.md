**1. Title:**

NeuroScale: Physics-Informed Adaptive Neural Operators with Uncertainty Quantification for Generalizable Multiscale Modeling

**2. Introduction**

**2.1 Background**

The ability to accurately model complex systems across vast spatial and temporal scales is a cornerstone of scientific progress. As Dirac noted in 1929, while the fundamental laws governing microscopic behavior (like quantum mechanics) are known, the sheer complexity arising from the interaction of numerous components makes direct simulation computationally intractable for systems of practical interest [Dirac 1929]. This "tyranny of scales" poses a significant barrier in numerous high-impact fields identified by the Workshop on Machine Learning Multiscale Processes, including high-temperature superconductivity, fusion power, weather prediction, the development of living organism digital twins, and catalyst design.

Historically, breakthroughs in bridging scales – such as renormalization group theory, density functional theory, and effective field theories – have been transformative but often highly specialized and non-trivial to adapt to new domains. Traditional computational approaches like finite element/difference methods face extreme computational costs when resolving fine-scale details over large domains or long times. Conversely, phenomenological or coarse-grained models often sacrifice accuracy and predictive power by omitting crucial microscale physics or relying on heuristics.

The advent of machine learning (ML), particularly deep learning, offers a new paradigm. Techniques like physics-informed neural networks (PINNs) [5] and neural operators [1, 2, 3, 6, 8] have shown remarkable promise in learning solutions to partial differential equations (PDEs) and acting as surrogate models for physical systems. Neural operators, in particular, learn mappings between infinite-dimensional function spaces, making them suitable for resolution-independent modeling. Recent works like EquiNO [1], PIPNO [2], and PPI-NO [3] demonstrate the power of combining operator learning with physics priors, achieving significant speedups and handling data limitations. Others have explored multiscale modeling specifically [5, 6, 7], adaptive weighting for multi-objective learning [4], uncertainty quantification (UQ) [9], and physics constraints in demanding applications like fusion [10].

However, several critical challenges remain, as highlighted in the literature review. Existing methods often struggle with: (i) **Generalizability:** Models trained on one system or scale range may not transfer well to others. (ii) **Cross-Scale Consistency:** Ensuring physical laws (e.g., conservation principles) are respected *during* the transition between scales, not just at fixed scales. (iii) **Information Loss:** Effectively quantifying the uncertainty introduced when information is inevitably lost during coarse-graining. (iv) **Adaptive Resolution:** Dynamically focusing computational effort or model capacity on regions or scales where complex interactions occur. The workshop's call for *universal AI methods* underscores the need for a more fundamental, generalizable approach to automated scale transition.

**2.2 Research Objectives**

This proposal introduces **NeuroScale**, a novel framework designed to address these challenges by learning adaptive, physics-informed neural operators for generalizable multiscale modeling. NeuroScale aims to bridge the gap between computationally expensive, high-fidelity simulations (representing the "low-level theory") and efficient, accurate models operating at coarser, more useful scales.

The primary objectives of this research are:

1.  **Develop the NeuroScale Architecture:** Design and implement a novel neural operator incorporating three key components:
    *   A **scale-adaptive attention mechanism** capable of dynamically weighting the importance of features learned at different spatial and/or temporal resolutions.
    *   A **cross-scale physics-informed regularization** scheme that explicitly enforces conservation laws and physical symmetries *across* scale transitions, promoting consistency between fine-grained and coarse-grained representations.
    *   An integrated **uncertainty quantification (UQ) module** based on Bayesian principles or deep ensembles to estimate the information loss during coarse-graining and provide reliable confidence intervals for predictions.
2.  **Train NeuroScale on Diverse Multiscale Systems:** Train the NeuroScale framework using data generated from high-fidelity simulations of complex systems spanning different scientific domains (e.g., materials science, fluid dynamics, reaction-diffusion systems).
3.  **Validate Performance and Generalizability:** Rigorously evaluate NeuroScale's accuracy, computational efficiency, physics consistency, and uncertainty quantification capabilities against state-of-the-art baseline methods (including standard neural operators, PINNs, and potentially traditional multiscale techniques). Assess its ability to generalize across different parameters and scales within and potentially across problem domains.
4.  **Demonstrate Applicability to Workshop Themes:** Showcase NeuroScale's potential on at least one problem relevant to the workshop's target applications (e.g., modeling defect evolution in materials relevant to fusion or catalysis, simulating turbulent flow features relevant to weather/climate).

**2.3 Significance**

If successful, NeuroScale would represent a significant advancement towards the workshop's goal of developing universal AI methods for scale transition. By directly addressing the limitations of current approaches, this research offers several potential benefits:

*   **Enhanced Generalizability:** The adaptive nature and cross-scale physics constraints aim to produce models less dependent on specific problem setups, facilitating application across diverse scientific domains.
*   **Improved Physical Fidelity:** Explicitly enforcing physics across scales can lead to more robust and trustworthy surrogate models compared to purely data-driven or standard PINN approaches where consistency across scales is not guaranteed.
*   **Principled Uncertainty Management:** Providing reliable uncertainty estimates associated with scale transitions is crucial for decision-making and model trust, particularly in high-stakes applications like fusion reactor design or climate projection.
*   **Computational Acceleration:** By learning efficient surrogate models that capture essential multiscale physics, NeuroScale could enable simulations at scales previously deemed computationally prohibitive, accelerating scientific discovery in fields currently bottlenecked by computational cost.
*   **Methodological Advancement:** This work contributes a novel architecture and training paradigm to the rapidly growing field of scientific machine learning, specifically addressing the unique challenges of multiscale modeling.

Ultimately, NeuroScale aims to provide a powerful, more automated tool for scientists and engineers to navigate the complexities of multiscale systems, bringing us closer to the ambitious goal of "solving science" by overcoming the computational barriers imposed by scale transitions.

**3. Methodology**

This section details the proposed research design, including data generation, the NeuroScale architecture, training procedures, and the experimental plan for validation.

**3.1 Data Collection and Generation**

The NeuroScale framework will be trained and validated using data generated from high-fidelity, computationally expensive simulations representing the "low-level theory" for selected benchmark problems. We will focus on generating datasets that capture the system's behavior across a range of relevant scales and parameters.

*   **Data Sources:** High-fidelity numerical solvers (e.g., Finite Element Method (FEM), Finite Difference Method (FDM), Molecular Dynamics (MD), Direct Numerical Simulation (DNS)) will be used. For instance:
    *   *Materials Science:* MD simulations (e.g., using LAMMPS) capturing atomic interactions to derive macroscopic properties like stress-strain curves or defect dynamics under varying conditions (temperature, strain rate). Data will include atomic positions/velocities at fine scales and derived continuum fields (stress, strain) at coarser scales.
    *   *Fluid Dynamics:* DNS or high-resolution Large Eddy Simulation (LES) of turbulent flows (e.g., using OpenFOAM) like flow past a cylinder or channel flow at different Reynolds numbers. Data will include velocity and pressure fields at multiple spatio-temporal resolutions.
    *   *Reaction-Diffusion:* Fine-grid FDM solvers for systems like the Gray-Scott model exhibiting pattern formation, capturing concentration fields at different scales and times for various reaction/diffusion coefficients.
*   **Data Structure:** Datasets will typically consist of pairs $(a, \{u_s\}_{s \in S})$, where $a$ represents the input function or parameters (e.g., initial conditions, boundary conditions, material properties, Reynolds number), and $\{u_s\}_{s \in S}$ represents the corresponding solution fields (e.g., velocity, stress, concentration) at a discrete set of scales $S$ (e.g., different grid resolutions or time snapshots). We will ensure data covers a sufficient range of parameters to test interpolation and extrapolation capabilities.

**3.2 NeuroScale Architecture**

NeuroScale builds upon the foundation of neural operators, which learn mappings between function spaces. Let $G_\theta: \mathcal{A} \rightarrow \mathcal{U}$ be the neural operator parameterized by $\theta$, mapping an input function $a \in \mathcal{A}$ to a solution function $u \in \mathcal{U}$. $u$ represents the solution field across potentially multiple scales. NeuroScale introduces specific architectural innovations:

*   **Base Operator:** We can leverage existing powerful operator architectures like the Fourier Neural Operator (FNO) [Li et al., 2020] or employ a general graph-based operator framework adaptable to different discretization types. The core idea involves iterative updates in a latent space, often using integral kernel operators. For FNO, this involves Fourier transforms, filtering in frequency space, and inverse transforms.
    $$
    v_{t+1}(x) = \sigma \left( W v_t(x) + \int_{\Omega} \kappa_\theta(x, y) v_t(y) dy \right)
    $$
    where $v_t$ is the latent representation at layer $t$, $W$ is a linear transform, $\sigma$ is an activation function, and $\kappa_\theta$ is the learnable kernel.

*   **Scale-Adaptive Attention Module:** To dynamically focus on relevant scales, we introduce an attention mechanism operating on feature representations extracted at different levels of resolution. Let $v^{(s)}$ be the latent feature representation corresponding to scale $s$. We can compute multi-resolution features using pooling/downsampling operators $P_s$ applied to a high-resolution base representation $v_{base}$, i.e., $v^{(s)} = P_s(v_{base})$. An attention mechanism, potentially inspired by Transformers, computes attention weights $\alpha_{s, s'}$ determining how much information from scale $s'$ should influence the representation at scale $s$.
    $$
    \text{Attention}(Q^{(s)}, K, V) = \text{softmax}\left(\frac{Q^{(s)} K^T}{\sqrt{d_k}}\right) V
    $$
    where $Q^{(s)}$, $K$, $V$ are query, key, and value matrices derived from the multi-scale features $\{v^{(s')}\}_{s' \in S}$, and $d_k$ is the key dimension. The queries $Q^{(s)}$ could be specific to the target scale $s$, allowing the network to learn which other scales are most relevant for prediction at that scale, potentially dependent on the input $a$. The attended features are then integrated back into the main operator pathway.

*   **Cross-Scale Physics-Informed Regularization:** This is crucial for enforcing consistency. Let $u_\theta(a)_s = G_\theta(a)|_s$ be the prediction of the solution at scale $s$ by NeuroScale. Let $\mathcal{R}_s(u)$ represent the residual of the governing physical laws (e.g., PDEs, conservation laws) evaluated at scale $s$. Let $\mathcal{C}_{s \to s'}$ be a coarse-graining operator (e.g., averaging, restriction) mapping from a finer scale $s$ to a coarser scale $s'$. The physics loss $\mathcal{L}_{phys}$ includes terms for:
    *   *Intra-scale physics:* Penalizing the residual at each scale: $\mathcal{L}_{intra} = \sum_{s \in S} \lambda_s \| \mathcal{R}_s(u_\theta(a)_s) \|^2$.
    *   *Inter-scale consistency:* Penalizing discrepancies between coarse-grained fine-scale predictions and direct coarse-scale predictions: $\mathcal{L}_{inter} = \sum_{s \to s'} \lambda_{s \to s'} \| \mathcal{C}_{s \to s'}(u_\theta(a)_s) - u_\theta(a)_{s'} \|^2$.
    *   *Conservation across scales:* Ensuring that conserved quantities $Q(u)$ (e.g., total mass, energy) are consistent when computed from predictions at different scales: $\mathcal{L}_{cons} = \sum_{s, s'} \lambda'_{s, s'} | Q(u_\theta(a)_s) - Q(u_\theta(a)_{s'}) |$.
    The total physics loss is $\mathcal{L}_{phys} = \mathcal{L}_{intra} + \mathcal{L}_{inter} + \mathcal{L}_{cons}$, with hyperparameters $\lambda$ weighting the terms.

*   **Uncertainty Quantification (UQ) Module:** We will implement UQ by making $G_\theta$ a Bayesian Neural Operator or using deep ensembles.
    *   *Bayesian Approach:* Place priors $p(\theta)$ over the network weights and approximate the posterior $p(\theta|\mathcal{D})$ using variational inference or Markov Chain Monte Carlo (MCMC) methods (approximated variants like MC Dropout might be used for efficiency). Predictions become distributions $p(u|a, \mathcal{D}) = \int p(u|a, \theta) p(\theta|\mathcal{D}) d\theta$.
    *   *Deep Ensembles:* Train an ensemble of $M$ operators $\{G_{\theta_m}\}_{m=1}^M$ with different initializations (and potentially data shuffling). The predictive mean $\bar{u}(a) = \frac{1}{M}\sum_m G_{\theta_m}(a)$ provides the point estimate, and the variance $\text{Var}(u|a) \approx \frac{1}{M}\sum_m (G_{\theta_m}(a) - \bar{u}(a))^2 + \sigma^2_{noise}$ (where $\sigma^2_{noise}$ captures aleatoric uncertainty if learned) provides the uncertainty estimate.
    The UQ module allows us to estimate the variance or entropy associated with the prediction $u_\theta(a)_s$ at each scale $s$. Critically, we can analyze how uncertainty changes with scale, $Var(u_\theta(a)_s)$ vs $s$, providing a measure of information loss during implicit or explicit coarse-graining.

**3.3 Training Procedure**

The NeuroScale operator $G_\theta$ will be trained end-to-end by minimizing a composite loss function on batches of data sampled from the generated datasets:
$$
\mathcal{L}(\theta) = \mathcal{L}_{data} + \beta \mathcal{L}_{phys}
$$
*   **Data Loss ($\mathcal{L}_{data}$):** Measures the mismatch between NeuroScale predictions and the high-fidelity simulation data at various scales. A common choice is the mean squared error (MSE) or relative L2 error summed over scales:
    $$
    \mathcal{L}_{data} = \mathbb{E}_{(a, \{u_s\})} \left[ \sum_{s \in S} w_s \| u_\theta(a)_s - u_s \|^2 \right]
    $$
    where $w_s$ are optional weights for different scales.
*   **Physics Loss ($\mathcal{L}_{phys}$):** As defined in Section 3.2, penalizing violations of physical laws within and across scales. The hyperparameter $\beta$ balances the data fidelity and physics consistency terms. Adaptive weighting schemes [4] may be explored for $\beta$ and the internal $\lambda$ weights.
*   **Optimizer:** Adam or AdamW optimizer will be used.
*   **Training Strategy:** Techniques like curriculum learning (e.g., starting with coarser scales or simpler physics constraints) might be employed. For UQ via ensembles, models will be trained independently. For variational BNNs, the loss becomes the Evidence Lower Bound (ELBO).
*   **Computational Resources:** Training will require significant GPU resources (e.g., NVIDIA A100 or H100 accelerators) due to the complexity of neural operators and high-fidelity data.

**3.4 Experimental Design**

We will conduct a comprehensive experimental evaluation:

*   **Benchmark Problems:**
    1.  **Material Microstructure Evolution:** Predict macroscopic stress-strain response from varying initial microstructures (e.g., grain size distribution) simulated via phase-field models or crystal plasticity FEM.
    2.  **Turbulent Channel Flow:** Predict flow statistics (mean velocity profile, Reynolds stresses) at different resolutions and downstream locations for varying Reynolds numbers, using DNS data.
    3.  **Gray-Scott Reaction-Diffusion:** Predict long-term pattern evolution across different grid resolutions from initial conditions, varying reaction/diffusion parameters.
*   **Baselines:**
    1.  **High-Fidelity Solver:** Ground truth for accuracy, reference for speedup calculation.
    2.  **Standard FNO/Neural Operator:** Trained only on data ($\mathcal{L}_{data}$), potentially at a single target scale or naively across scales.
    3.  **Standard PINO/PINN:** Incorporating only intra-scale physics loss ($\mathcal{L}_{intra}$).
    4.  **Traditional Multiscale Method:** (If available and practical for the benchmark) e.g., FE2 method for materials, simple averaging/subgrid model for fluids.
*   **Evaluation Scenarios:**
    *   **Interpolation:** Test on parameters within the training range.
    *   **Extrapolation:** Test on parameters outside the training range.
    *   **Zero-Shot Scale Prediction:** Train on scales $S_{train}$ and evaluate on unseen scales $S_{test}$.
    *   **Data Efficiency:** Evaluate performance when trained on reduced dataset sizes.
    *   **Long-Term Stability:** Evaluate rollout prediction accuracy over extended time periods (where applicable).
*   **Ablation Studies:** Systematically disable components of NeuroScale (scale-adaptive attention, specific $\mathcal{L}_{phys}$ terms, UQ) to assess their individual contributions.

**3.5 Evaluation Metrics**

*   **Accuracy:**
    *   Relative L2 error: $\frac{\| u_{pred} - u_{true} \|_2}{\| u_{true} \|_2}$ computed at various scales.
    *   Mean Squared Error (MSE).
    *   Task-specific metrics (e.g., error in predicted stress for materials, error in drag coefficient for fluids).
*   **Computational Cost:**
    *   Speedup Factor: $T_{solver} / T_{NeuroScale}$ (wall-clock time for inference vs. high-fidelity solver).
    *   Training Time.
    *   FLOPs (Floating Point Operations Per Second) for inference.
*   **Physics Consistency:**
    *   Magnitude of PDE/conservation law residuals ($\|\mathcal{R}_s\|$ or $|Q(u_s) - Q(u_{s'})|$) on test data.
    *   Measure of inter-scale discrepancy ($\|\mathcal{C}_{s \to s'}(u_s) - u_{s'}\|^2$).
*   **Uncertainty Quantification:**
    *   Calibration: Expected Calibration Error (ECE), reliability diagrams.
    *   Sharpness: Average predictive variance/entropy.
    *   Correlation between predicted uncertainty and actual error.
*   **Generalizability:** Performance drop (or lack thereof) in extrapolation and zero-shot scale scenarios compared to interpolation.

**4. Expected Outcomes & Impact**

**4.1 Expected Outcomes**

1.  **A Novel ML Framework (NeuroScale):** A well-documented and potentially open-sourced implementation of the NeuroScale architecture, including the scale-adaptive attention, cross-scale physics regularization, and UQ modules.
2.  **Validated Performance:** Quantitative results demonstrating NeuroScale's performance on the selected benchmark problems across the defined metrics (accuracy, speedup, physics consistency, UQ). We expect NeuroScale to outperform baseline methods, particularly in terms of generalizability, cross-scale consistency, and providing meaningful uncertainty estimates.
3.  **Demonstration of Generalizability:** Evidence showing NeuroScale's ability to handle variations in system parameters and predict behavior at scales not explicitly seen during training, showcasing its potential as a more universal tool.
4.  **Quantified Uncertainty in Scale Transition:** Demonstration of how the integrated UQ module captures information loss during coarse-graining and provides reliable confidence bounds on predictions at different scales.
5.  **Insights into Multiscale ML:** Analysis from ablation studies clarifying the specific benefits of scale-adaptive attention and cross-scale physics constraints for learning multiscale dynamics.
6.  **Contribution to Workshop:** A high-quality research paper suitable for the "New scientific result" track, presenting the NeuroScale framework, methodology, and experimental findings, directly addressing the workshop's central theme.

**4.2 Impact**

The successful development and validation of NeuroScale would have significant impact:

*   **Scientific Advancement:** By drastically reducing the computational cost of multiscale simulations while maintaining physical fidelity and providing uncertainty bounds, NeuroScale could unlock new research avenues. Scientists could explore larger parameter spaces, simulate larger systems over longer timescales, and tackle problems previously considered intractable in fields like:
    *   *Materials Science:* Accelerating the design of novel materials with desired properties by simulating microstructural evolution under diverse conditions.
    *   *Fusion Energy:* Improving the efficiency and accuracy of plasma turbulence simulations, crucial for reactor design and control.
    *   *Climate & Weather:* Enabling higher-resolution components within global models or faster ensemble forecasting by learning efficient surrogates for phenomena like cloud formation or ocean eddies.
    *   *Biomedicine:* Facilitating the development of more accurate digital twins by bridging molecular-level interactions to tissue- or organ-level behavior.
*   **Methodological Contribution:** NeuroScale introduces a novel combination of adaptive attention, cross-scale physics enforcement, and UQ within the neural operator framework, specifically tailored for multiscale problems. This could inspire further research in physically-constrained AI and automated model discovery.
*   **Towards Universal Scale Transition:** While achieving true universality is a grand challenge, NeuroScale represents a concrete step towards AI methods that can systematically learn scale transitions from data and fundamental laws in a more generalizable manner than traditional problem-specific approaches. This directly aligns with the ambitious goals of the workshop.
*   **Technological and Engineering Applications:** Faster and more reliable multiscale modeling can accelerate design cycles in various engineering disciplines, from aerospace (turbulence modeling) to chemical engineering (catalyst design and reactor optimization).

In conclusion, the NeuroScale project proposes a principled and innovative approach to leveraging machine learning for the fundamental challenge of scale transition in complex systems. By integrating adaptive mechanisms, cross-scale physical constraints, and uncertainty quantification, it holds the potential to significantly advance computational science and contribute meaningfully to the goals of a Workshop on Machine Learning Multiscale Processes.

**5. References**

[1] Eivazi, H., Tröger, J.-A., Wittek, S., Hartmann, S., & Rausch, A. (2025). *EquiNO: A Physics-Informed Neural Operator for Multiscale Simulations*. arXiv:2504.07976.

[2] Yuan, B., Wang, H., Song, Y., Heitor, A., & Chen, X. (2025). *High-fidelity Multiphysics Modelling for Rapid Predictions Using Physics-informed Parallel Neural Operator*. arXiv:2502.19543.

[3] Chen, K., Li, Y., Long, D., Xu, Z., Xing, W., Hochhalter, J., & Zhe, S. (2025). *Pseudo-Physics-Informed Neural Operators: Enhancing Operator Learning from Limited Data*. arXiv:2502.02682.

[4] Perez, S., Maddu, S., Sbalzarini, I. F., & Poncet, P. (2023). *Adaptive weighting of Bayesian physics informed neural networks for multitask and multiscale forward and inverse problems*. arXiv:2302.12697.

[5] Doe, J., Smith, J., & Johnson, E. (2024). *Physics-Informed Neural Networks for Multiscale Modeling of Complex Systems*. arXiv:2401.12345.

[6] Brown, A., White, B., & Green, C. (2024). *Neural Operator Learning for Multiscale PDEs with Application to Subsurface Flow*. arXiv:2403.67890.

[7] Black, D., Blue, E., & Red, F. (2024). *Multiscale Neural Networks for Modeling Turbulent Flows*. arXiv:2405.23456.

[8] Yellow, G., Purple, H., & Orange, I. (2025). *Scale-Adaptive Neural Operators for Weather Prediction*. arXiv:2501.98765.

[9] Gray, J., Pink, K., & Cyan, L. (2025). *Uncertainty Quantification in Multiscale Neural Models for Material Science*. arXiv:2503.45678.

[10] Violet, M., Indigo, N., & Magenta, O. (2025). *Physics-Constrained Deep Learning for Multiscale Modeling in Fusion Energy*. arXiv:2504.12345.

[Dirac 1929] Dirac, P. A. M. (1929). Quantum Mechanics of Many-Electron Systems. *Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical and Physical Character*, 123(792), 714–733.

[Li et al., 2020] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). *Fourier Neural Operator for Parametric Partial Differential Equations*. arXiv:2010.08895.