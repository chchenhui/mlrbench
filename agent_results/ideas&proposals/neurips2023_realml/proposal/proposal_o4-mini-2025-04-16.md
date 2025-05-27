1. Title  
Physics-Constrained Bayesian Optimization for Accelerated Materials Discovery  

2. Introduction  
2.1 Background  
Discovering novel materials with target properties is central to advances in energy storage, catalysis, and electronics. Traditional “one‐at‐a‐time” experimental or high‐fidelity simulation campaigns are prohibitively expensive and slow: each candidate evaluation (synthesis + characterization or density‐functional‐theory simulation) can take hours to days. Bayesian Optimization (BO) and other active‐learning strategies have emerged as powerful wrappers around expensive experiments, trading off exploration of unknown regions and exploitation of promising leads to minimize the total number of queries required to find high‐performance materials.  

However, off‐the‐shelf BO often proposes candidates that violate well‐known physical laws (e.g., thermodynamic instability, charge‐neutrality violations or infeasible synthesis conditions). Evaluating or synthesizing such infeasible candidates wastes valuable resources and slows down discovery. A recent literature (Smith et al., 2023; Kim et al., 2023; Patel et al., 2023; Brown et al., 2023; Garcia et al., 2023; White et al., 2023; Allen et al., 2023; Adams et al., 2023; Evans et al., 2023; Hall et al., 2023) has begun to incorporate physics–based constraints into either the surrogate model or the acquisition function. Yet, a general framework that unifies constrained Gaussian‐process surrogates, constraint‐aware acquisition design, and multi‐fidelity experimentation remains to be developed and validated across synthetic benchmarks and real‐world materials tasks.  

2.2 Research Objectives  
This proposal aims to develop, implement, and validate a **Physics‐Constrained Bayesian Optimization** (PC‐BO) framework for materials discovery. The specific objectives are:  
  • Formulate a constrained BO problem that explicitly models multiple physical constraints (e.g., thermodynamic stability, charge neutrality, synthesis feasibility).  
  • Design a surrogate modeling approach based on constrained Gaussian Processes (cGPs) that enforces known constraints in the posterior predictive.  
  • Derive and implement a constraint‐aware acquisition function (Constrained Expected Improvement and physics‐penalized EI).  
  • Incorporate multi‐fidelity data sources (low‐cost physics simulations vs. high‐cost experiments) into the PC‐BO loop.  
  • Empirically evaluate PC‐BO against standard BO, unconstrained cGP‐BO, and random sampling on both synthetic constrained optimization benchmarks and real materials‐discovery tasks (e.g., perovskite photovoltaics, polymer electrolytes).  

2.3 Significance  
By integrating domain knowledge directly into the active learning loop, PC‐BO is expected to:  
  • Substantially reduce the number of invalid or infeasible experiments, cutting cost and time by up to 50–70%.  
  • Accelerate convergence to high‐performance materials in both synthetic and real tasks (improving sample efficiency by 2–5×).  
  • Provide a general and extensible blueprint for other scientific domains (robotics, drug design, causal discovery) where physical or domain constraints are critical.  

3. Methodology  
3.1 Problem Formulation  
Let X⊂ℝ^d be the continuous design space of candidate materials parameters (e.g., composition ratios, processing temperatures). We seek to maximize an unknown objective function f:X→ℝ (e.g., power conversion efficiency) under m known inequality constraints c_j:X→ℝ, j=1…m, each representing a physical law or feasibility rule. Formally:  
$$
\max_{x\in X_{\rm feas}} f(x), 
\quad X_{\rm feas} = \{\,x\in X: c_j(x)\le 0,\;\forall j=1\ldots m\}.
$$  
Each evaluation at x yields a noisy observation y=f(x)+ε and constraint‐noisy measures z_j=c_j(x)+η_j, where ε,η_j are zero‐mean Gaussian noise. We allow multi‐fidelity: a cheap approximation f^L (e.g., coarse simulation) and an expensive true f^H (e.g., experiment).  

3.2 Physics‐Constrained Surrogate Modeling  
We place Gaussian‐process priors on both f and each constraint c_j:  
$$
f\sim \mathcal{GP}(\mu_f(x),k_f(x,x')), 
\quad c_j\sim \mathcal{GP}(\mu_{c_j}(x),k_{c_j}(x,x')).
$$  
To encode physical knowledge, we adopt the following strategies:  
 1. **Mean‐function encoding:** If an analytical form of a physical constraint is known (e.g. an approximate free‐energy surface g(x) with g(x)≤0 for stable compounds), we set μ_{c_j}(x)=g(x).  
 2. **Virtual observations (hard constraints):** We add pseudo‐data {(x_v,0)} at virtual points x_v known to satisfy c_j(x_v)=0, enforcing the GP posterior to respect these anchors.  
 3. **Constraint co‐kriging:** When multiple constraints are correlated (e.g., stability and charge neutrality), we use a multi‐output GP with a block covariance K_c combining cross‐covariances between c_j and c_{j'}.  

Given n_t high‐fidelity observations D^H_t={(x_i,y_i)}_{i=1…n_t} and n^L_t low‐fidelity D^L_t, we use an auto‐regressive multi‐fidelity GP (Kennedy–O’Hagan) to fuse them. Let K denote the joint covariance matrix and Y the stacked observations; the posterior mean and variance at a test point x are:  
$$
\mu_f(x)=k_{t}^\top(K+\sigma^2 I)^{-1}Y,\quad
\sigma^2_f(x)=k_{tt}-k_{t}^\top(K+\sigma^2 I)^{-1}k_{t},
$$  
and similarly for each constraint GP.  

3.3 Constraint‐Aware Acquisition Function  
We propose two variants of physics‐aware acquisition:  

1. **Constrained Expected Improvement (CEI)**  
   $$ 
   \alpha_{\rm CEI}(x) = \mathrm{EI}(x)\;\prod_{j=1}^m P\bigl(c_j(x)\le 0\bigr), 
   $$  
   where  
   $$
   \mathrm{EI}(x)=\mathbb{E}\bigl[\max\{0,\,f(x)-f(x^+)\}\bigr],\quad
   P(c_j(x)\le0)=\Phi\!\Bigl(\frac{-\mu_{c_j}(x)}{\sigma_{c_j}(x)}\Bigr),
   $$  
   f(x^+) is the current best feasible value and Φ is the standard Gaussian CDF.  

2. **Physics‐Penalized Expected Improvement (PPEI)**  
   $$ 
   \alpha_{\rm PPEI}(x) = \mathrm{EI}(x)\;-\;\lambda\sum_{j=1}^m\max\{0,\mu_{c_j}(x)\},
   $$  
   where λ>0 is a tunable penalty coefficient. This form softly discourages candidate points with high predicted constraint violations.  

We will compare both variants and tune λ by warm‐up experiments on synthetic constrained functions.  

3.4 PC‐BO Algorithm  
Pseudocode:  
```
Input: initial design X₀ via Latin Hypercube, budget T, penalty λ 
Initialize: 
  Fit multi‐fidelity, constrained GPs on D₀
for t = 1…T do 
  For each x∈X (via continuous optimization of α): 
    Compute μ_f(x),σ_f(x), {μ_{c_j}(x),σ_{c_j}(x)}
    Compute α_PC−BO(x) := α_CEI(x) or α_PPEI(x)
  Select xₜ = argmax_x α_PC−BO(x)
  Query low‐fidelity f^L(xₜ); optionally update GP with novel D^L 
  If uncertainty / predicted yield high, perform high‐fidelity f^H(xₜ) & measure constraints 
  Update GP priors with new data 
end for
Output: x* = argmax (observed feasible f)
```  

3.5 Data Collection & Experimental Design  
We will validate PC‐BO on two fronts:

A. Synthetic Benchmarks  
  • Constrained Branin‐Hoo and Hartmann‐3D with analytic constraint surfaces.  
  • Randomly generated feasible region shapes to test robustness to non‐convex constraints.  

B. Real Materials Discovery Tasks  
  1. **Perovskite Photovoltaic Efficiency**  
     – Design variables: A‐site cation ratio, B‐site alloying fraction, synthesis temperature.  
     – Objective: simulated power conversion efficiency via surrogate DFT code (low‐fidelity) and laboratory measurements (high‐fidelity).  
     – Constraints: Goldschmidt tolerance factor (stability), charge neutrality, synthesis temperature window.  
  2. **Polymer Electrolyte Ionic Conductivity**  
     – Variables: monomer composition, backbone rigidity parameter.  
     – Objective: ionic conductivity from coarse molecular dynamics (low‐fidelity) and impedance spectroscopy (high‐fidelity).  
     – Constraints: glass transition temperature limits, solubility boundaries.  

For each task:  
  • Initialize with 10 randomly sampled feasible points.  
  • Run PC‐BO for 50 additional evaluations (budget chosen to reflect typical lab throughput).  
  • Compare against baselines: Unconstrained BO, Constrained GP + standard EI, Random Search.  

3.6 Evaluation Metrics  
We will quantify performance by:  
  • **Feasible‐Success Rate (FSR):** fraction of proposed points that satisfy true constraints.  
  • **Sample Efficiency (SE):** number of high‐fidelity evaluations to reach within 95% of the optimum (estimated by large‐scale grid search).  
  • **Cumulative Regret (R_T):**  
    $$
    R_T = \sum_{t=1}^T \bigl(f(x^+) - f(x_t)\bigr),
    $$  
    where x^+=best true feasible point.  
  • **Computational Overhead:** time per iteration for surrogate updates and acquisition optimization.  

Statistical significance will be assessed over 20 independent random seeds; we will report mean ± standard error.  

4. Expected Outcomes & Impact  
4.1 Expected Outcomes  
  • **Improved Sample Efficiency:** We expect PC‐BO to require 2–5× fewer high‐fidelity evaluations than baseline BO to reach target performance.  
  • **Near‐Zero Constraint Violations:** CEI‐based selection should maintain FSR>95%, while unconstrained BO often achieves <50%.  
  • **Generality Across Domains:** We anticipate consistent gains on both synthetic benchmarks and real materials tasks. Ablation studies will reveal the relative contributions of constrained surrogates vs. penalized acquisition.  
  • **Open‐Source Release:** We will publish our PC‐BO implementation, benchmark scripts, and processed materials datasets to accelerate adoption in the community.  

4.2 Broader Impact  
  • **Accelerating Sustainable Materials Development:** By focusing experimental resources on physically feasible candidates, PC‐BO can dramatically shorten R&D cycles in areas such as photovoltaics, batteries, and catalysis—ultimately contributing to lower‐carbon technologies.  
  • **Cross‐Disciplinary Applicability:** The same framework can be adapted to other domains—robotics (physical safety constraints), drug design (toxicity and pharmacokinetic constraints), and causal‐inference experiments (ethical or budgetary constraints).  
  • **Educational Value:** Through thorough documentation and tutorials, we will lower the barrier for experimental scientists to adopt principled active‐learning methods in their laboratories.  

In summary, the proposed Physics‐Constrained Bayesian Optimization framework offers a unified, general, and practically relevant solution to the pressing challenge of efficient, physically guided exploration in high‐impact scientific domains. By tightly integrating known constraints into both the surrogate model and acquisition strategy—and validating across synthetic and real materials systems—PC‐BO stands to set a new standard for domain‐aware active learning in the real world.