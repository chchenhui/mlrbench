# Physics-Constrained Active Learning for Accelerated Materials Discovery

## Introduction

### Background
The discovery of novel materials with targeted properties is a cornerstone of advancements in energy, electronics, and medicine, yet traditional experimental and computational approaches remain prohibitively expensive and time-consuming. For example, synthesizing and testing a new alloy composition can require weeks of lab work, while high-fidelity physics simulations may run for days on supercomputers. Bayesian Optimization (BO)—a leading active learning framework—has emerged as a promising solution, iteratively selecting experiments that balance exploration of uncharted design spaces and exploitation of known performance trends. However, standard BO frameworks often propose candidates that violate fundamental physical principles such as thermodynamic stability, charge neutrality, or synthesis feasibility, leading to wasted resources on non-viable proposals. Recent works (e.g., Smith et al., 2023; Garcia et al., 2023) have highlighted the urgency of integrating domain-specific constraints into BO, but these efforts have primarily addressed isolated constraints (e.g., thermodynamic stability alone) or imposed penalties heuristically. Key challenges remain in systematically encoding multi-faceted physical laws while preserving BO’s data efficiency.

### Research Objectives
This research proposes **Physics-Constrained Bayesian Optimization (PC-BO)**, a framework that embeds diverse physical constraints into two critical components of BO: (1) the surrogate model and (2) the acquisition function. The methodology will address four interlinked goals:  
1. **Develop constrained Gaussian Process (GP) models** that inherently exclude unphysical regions in the design space by incorporating equality/inequality constraints derived from thermodynamics, crystallography, and synthetic chemistry principles.  
2. **Design acquisition functions** (e.g., Expected Improvement) that penalize constraint violations via probabilistic safety assessments, enabling adaptive trade-offs between performance optimization and physical plausibility.  
3. **Establish scalable algorithms** for optimizing the constrained acquisition function in high-dimensional spaces (e.g., 10–100 design variables representing elemental ratios or molecular configurations).  
4. **Validate the framework** on synthetic benchmarks and real-world materials discovery tasks, demonstrating improvements in discovery speed, validity rates, and robustness compared to unconstrained baselines.  

### Significance
A successful PC-BO framework will bridge the gap between theoretical active learning research and practical materials engineering, offering:  
- **Reduced experimental costs** by focusing evaluations on physically valid candidates.  
- **Accelerated discovery** of materials with targeted properties (e.g., high superconductivity critical temperature).  
- **Cross-domain reusability** through a modular constraint-encoding module applicable to chemistry, catalysis, and battery design.

---

## Methodology

### Data Collection and Preprocessing  
We will evaluate PC-BO on two material design tasks:  

1. **Ternary Alloy Optimization**: Predicting composition ratios $(x_{1}, x_{2}, x_{3})$ of Al–Zn–Mg alloys to maximize tensile strength, subject to the physical constraints:  
   - **Charge neutrality**: $x_{1} \cdot (+3) + x_{2} \cdot (+2) + x_{3} \cdot (-1) = 0$.  
   - **Thermodynamic stability**: Free energy of mixing $\Delta G_{mix} < 0$ computed via Miedema’s model.  
   - **Synthesis feasibility**: Elemental ratios must conform to industrial vacuum-arc melting protocols (e.g., $x_{3} \leq 0.2$).  

2. **Metal–Organic Framework (MOF) Design**: Optimizing 10-dimensional descriptors of MOFs (e.g., pore volume, linker types) to maximize CO₂ adsorption, constrained by crystal structure symmetry rules and density requirements ($0.3 \leq \rho \leq 1.2 \, \text{g/cm}^3$).  

For both tasks, benchmark data will be synthesized using physics-guided simulators (e.g., CALPHAD for alloys) and augmented with low-noise experimental data from the Materials Project and Citrine Informatics Platform. Missing values will be imputed via graph neural networks that respect crystallographic symmetries (Wu et al., 2020).

### Physics-Informed Surrogate Models  
The constrained GP surrogate model will encode physical laws via **hard** and **soft constraints**:  
1. **Hard Constraints**: Enforced through the GP kernel by embedding Lagrangian multipliers. For example, to restrict predictions to the plane defined by charge neutrality ($a \cdot x = 0$), we use the **SIREN kernel** (Sitzmann et al., 2020) with modified Fourier features:  
   $$
   k_{\text{SIREN-hard}}(x, x') = \sum_{i=1}^N w_i \sin\left( \omega_i (a^\top x) + \phi_i \right) \sin\left( \omega_i (a^\top x') + \phi_i \right),
   $$
   where $a = [3, 2, -1]$ for Al–Zn–Mg. This ensures all GP realizations satisfy $a^\top x = 0$.  

2. **Soft Constraints**: Modeled probabilistically as likelihood penalizations for violations of $\Delta G_{mix}$ or $\rho$. For example, the GP likelihood function becomes:
   $$
   \log p(Y|f) = -\frac{1}{2} \left( Y - f(X) \right)^\top K^{-1} \left( Y - f(X) \right) - \lambda \sum_{i=1}^{N} \max\left(0, \Delta G_{mix}(x_i)\right),
   $$
   where $\lambda$ weighs constraint importance. This formulation, inspired by Martinez et al. (2023), allows tunable strictness.  

The GP posterior is computed using variational inference to handle non-conjugate likelihoods, with gradients optimized via Adam (Kingma & Ba, 2015).

### Constraint-Aware Acquisition Function  
The Expected Improvement (EI) acquisition function is modified to incorporate a physics-compliance score $P_c(x)$, representing the probability of $x$ satisfying all constraints. For equality constraints (e.g., $a \cdot x = 0$), $P_c(x)$ evaluates the GP posterior’s alignment with the constraint subspace. For inequality constraints (e.g., $\Delta G_{mix} < 0$):  
$$
P_c(x) = \mathbb{E}_{f \sim GP}\left[ \mathbb{I}\left( f_{\Delta G}(x) < 0 \right) \right] \approx \Phi\left( \frac{-\mu_{\Delta G}(x)}{\sigma_{\Delta G}(x)} \right),
$$
where $\Phi$ is the standard normal CDF, and $\mu_{\Delta G}/\sigma_{\Delta G}$ are the GP’s mean/prediction uncertainty for $\Delta G_{mix}$. The **Physics-Constrained Expected Improvement (PC-EI)** is then:
$$
\text{PC-EI}(x) = \text{EI}(x) \cdot P_c(x).
$$
Optimization of PC-EI is performed via Sequential Quadratic Programming (SQP) to handle the constrained search space.

### Experimental Design  
#### Baselines  
- **Unconstrained BO (U-BO)**: Standard GP-UCB framework.  
- **Penalty BO (P-BO)**: Adds a quadratic penalty to the reward for constraint violations (Maringer & Schurle, 2022).  
- **Feasibility BO (F-BO)**: Two-stage approach: first learns a feasibility classifier, then optimizes objective within predicted feasible regions (Garrido et al., 2021).  

#### Metrics  
- **Discovery Rate**: Proportion of trials with $f(x) \geq \gamma f_{\text{opt}}$ ($\gamma = 0.9$).  
- **Validity Rate**: Proportion of queried points violating no constraints.  
- **Cumulative Regret**: $\sum_{t=1}^{T} (f_{\text{opt}} - f(x_t))$.  
- **Optimization Efficiency**: Mean iterations to converge (1% change in PC-EI over 5 iterations).  

#### Ablation Studies  
- **Kernel Ablation**: Replace SIREN-hard with standard RBF to test hard constraint efficacy.  
- **Acquisition Ablation**: Compare PC-EI with variants that discard $P_c(x)$ or use deterministic hard thresholds.  

---

## Expected Outcomes & Impact

### Novel Contributions  
1. **First-Order Physics Constraint Integration**: Demonstrates how equality/inequality constraints can be systematically embedded into both GP kernels and acquisition functions.  
2. **Modular Framework**: Enables drop-in replacement of constraint models, allowing generalization across materials systems (e.g., MOFs vs. intermetallics).  
3. **Benchmark Datasets**: Releases two annotated datasets with physical constraint annotations for reproducibility.  

### Empirical Advancements  
- **2–3x Faster Discovery**: PC-BO will reduce iterations needed to reach optimal materials by 2–3 times compared to P-BO and U-BO (hypothesis: validated by 95% confidence intervals in convergence curves).  
- **>95% Validity Rate**: Constraints will prune over 95% of unphysical candidates, outperforming F-BO’s 80–85% baseline.  
- **Robustness to Noise**: Physics-based priors will maintain validity rates under synthetic corruption (e.g., 20% Gaussian noise in $\Delta G_{mix}$ outputs).  

### Societal and Industrial Impact  
- **Materials Engineering**: Accelerates development of carbon capture materials and low-co2 metallurgy.  
- **Algorithm Design**: Inspires physics-constrained BO methods in robotics (safety constraints) and drug discovery (Lipinski’s rules).  
- **Cross-Disciplinary Synergy**: Bridges computational materials science and machine learning research communities via open-source implementations in GPyTorch and Ax.  

By addressing the missing link between theoretical BO frameworks and practical experimental reality, this work will democratize data-efficient discovery systems for real-world scientific innovation.