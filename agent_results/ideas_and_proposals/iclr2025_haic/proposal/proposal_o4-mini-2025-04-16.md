1. Title  
Unraveling Long-Term Coevolution: A Framework for Mitigating Bias in Dynamic Human-AI Feedback Loops in Healthcare

2. Introduction  
Background  
AI‐driven clinical decision support systems are rapidly being integrated into healthcare workflows to recommend treatment plans, monitor patient adherence, and predict health outcomes. As these systems interact continuously with patients and clinicians, bidirectional feedback loops emerge: AI recommendations shape patient behavior (e.g., medication adherence, lifestyle changes), and those behaviors in turn alter the data on which the AI retrains. While reinforcement‐learning‐based approaches can optimize for clinical performance, they often overlook how biases—rooted in socio‐economic disparities, measurement artifacts, or incomplete data—can be amplified through repeated human‐AI coadaptation.

Problem Statement  
Static bias mitigation techniques (for example, data reweighting or one‐time fairness constraints) treat each deployment as a fixed batch problem. They fail to capture how an AI system’s evolving policy interacts over time with heterogeneous patient populations, potentially reinforcing or manufacturing new inequities. In high‐stakes settings such as diabetes management—where social determinants of health already drive disparate outcomes—the danger is especially acute: small initial biases can compound across multiple feedback cycles, degrading both fairness and efficacy.

Research Objectives  
Our goal is to develop, analyze, and validate a dynamic, bias‐aware human‐AI coevolution framework for healthcare. We will:  
a. Construct a simulation environment modeling longitudinal patient‐AI interactions in chronic disease management.  
b. Design and implement a bias‐aware “co‐correction” mechanism that (i) detects shifts in equity over time via causal mediation analysis and (ii) generates patient‐facing explanations to recalibrate trust.  
c. Propose and compute a novel “looping inequity” metric to quantify how long‐term human‐AI interaction widens or narrows disparities.  
d. Validate our approach on a case study in diabetes management, comparing against static fairness and performance‐only baselines.

Significance  
By explicitly modeling and correcting feedback‐loop‐induced bias, this research bridges algorithmic fairness, causal inference, and participatory AI design. It yields:  
• A generalizable simulation toolkit for human‐AI coevolution research.  
• A bias‐aware reinforcement‐learning algorithm suitable for high‐stakes domains.  
• Empirically‐validated guidelines for deploying equitable, trustworthy AI in healthcare.  
These contributions speak directly to HAIC 2025’s call for work on long‐term human‐AI interaction, dynamic feedback loops, and socio‐technical bias.

3. Methodology  
Overview  
Our methodology has three core components: (A) a longitudinal simulation environment for patient–AI coevolution; (B) a bias‐aware co‐correction algorithm that integrates reinforcement learning with causal mediation analysis and explainability; (C) an experimental design for validation on diabetes management data.

A. Data Collection and Preprocessing  
• Data Sources  
  – Real‐world EHR data (e.g., MIMIC‐IV subset) augmented with published longitudinal cohort statistics for Type 2 diabetes patients.  
  – Demographic attributes $Z \in \{0,1\}$ indicating a protected group (e.g., socio‐economic status).  
  – Time‐indexed features: clinical state $X_t$ (e.g., HbA1c, blood pressure), environmental stressors $E_t$, treatment adherence signals $A_t$.  
• Preprocessing  
  – Imputation of missing values via multiple imputation by chained equations (MICE).  
  – Normalization of continuous features to $[0,1]$.  
  – Stratified sampling to preserve protected‐group proportions in training/validation/test splits.

B. Simulation Framework  
We formalize patient–AI coevolution as a Markov decision process (MDP) 
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma),$$
where:  
$\mathcal{S}\subseteq\mathbb{R}^d$ is the patient health state (clinical + environmental),  
$\mathcal{A}$ is the set of AI‐recommended interventions (e.g., medication dose adjustments, lifestyle prompts),  
$P(s' \mid s,a)$ is the transition kernel modeling how patients respond to treatment and external factors,  
$R(s,a,s')$ is the reward reflecting clinical improvement minus a fairness penalty,  
$\gamma\in(0,1)$ is the discount factor over a horizon $T$.  

State Dynamics  
We let the next‐state function be  
$$s_{t+1} = f(s_t, a_t) + \varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\Sigma)$$  
where $f$ is estimated from historical trajectories via Gaussian process regression.

Reward Design  
The instantaneous reward combines clinical utility and an equity penalty:  
$$r_t = U(s_{t+1}) - \lambda\,\Delta_t,$$  
where $U(s)$ is a scalar measure of expected health improvement (e.g., negative HbA1c change),  
$\Delta_t = \bigl|\mathbb{E}[U(s_{t+1})\mid Z=1] - \mathbb{E}[U(s_{t+1})\mid Z=0]\bigr|$ quantifies disparity at time $t$,  
and $\lambda>0$ balances performance versus fairness.

C. Bias‐Aware Co‐Correction Algorithm  
We employ a policy $\pi_\theta(a\mid s)$ parameterized by neural network weights $\theta$. The algorithm proceeds as follows:

Pseudocode:  
```
Initialize θ ← θ₀, patient‐behavior model φ ← φ₀
for each episode do
  reset s₁ ∼ initial state distribution
  for t = 1 to T do
    aₜ ∼ π_θ(·∣sₜ)
    bₜ ∼ P_φ(behavior ∣ sₜ,aₜ)                   # patient response
    sₜ₊₁ = f(sₜ,aₜ) + εₜ                   
    compute Uₜ₊₁ = U(sₜ₊₁)
    compute disparity Δₜ = ∣E[Uₜ₊₁∣Z=1] − E[Uₜ₊₁∣Z=0]∣
    rₜ = Uₜ₊₁ − λ Δₜ
    # Policy update (policy‐gradient)
    θ ← θ + α ∇_θ log π_θ(aₜ∣sₜ) Gₜ
    # Update bias tracking via causal mediation
    if Δₜ > τ then
      perform mediation analysis and generate explanation
      adjust θ via fairness‐correction term Δ_corr
    end if
  end for
end for
```
Here, $G_t = \sum_{k=t}^T \gamma^{k-t}r_k$ is the return, $\alpha$ is learning rate, and $\tau$ a disparity threshold.

C1. Causal Mediation Analysis  
We decompose the total effect (TE) of an AI action $A$ on outcome $Y$ into direct and indirect components via patient behavior $B$:  
$$\text{TE} = \mathbb{E}[Y\mid do(A=1)] - \mathbb{E}[Y\mid do(A=0)],$$  
$$\text{IE} = \mathbb{E}[Y\mid do(A=1,B(A=1))] - \mathbb{E}[Y\mid do(A=1,B(A=0))],$$  
$$\text{DE} = \text{TE} - \text{IE}.$$  
We estimate these using front‐door adjustment and Monte Carlo sampling inside the simulator. When $\text{IE}$ is large and favoring one protected group, we flag the loop as generating bias through behavior mediation.

C2. Explainability and Trust Calibration  
When the disparity $\Delta_t$ exceeds $\tau$, we produce patient‐facing, counterfactual explanations using a local surrogate model (e.g., SHAP). For a given patient $i$, we compute feature‐attribution values $\phi_j$ satisfying  
$$Y_i(a | s_i) = \phi_0 + \sum_j \phi_j,$$  
and convey “if you had chosen a slightly higher/lower dose, your expected outcome difference would be….” We track patient trust $T_i$ via an internal feedback model:  
$$T_i^{new} = T_i^{old} + \eta\,(E_i – T_i^{old}),$$  
where $E_i$ is an elicited trust score (e.g., via a three‐question survey scaled to $[0,1]$), and $\eta\in(0,1)$ a calibration rate. High trust restores the reliance on AI suggestions; low trust triggers fallback to conservative policies.

D. Experimental Design and Evaluation Metrics  
D1. Validation Scenario: Diabetes Management  
We simulate $N=1{,}000$ virtual Type 2 diabetes patients over a $T=52$‐week horizon. Two demographic cohorts ($Z=0$ majority, $Z=1$ minority) each of size 500 are constructed to reflect real‐world differences in baseline HbA1c and adherence rates.  

D2. Baseline Methods  
1. Standard RL (maximize $U$ only).  
2. Static fairness RL (add one‐time fairness constraint in initial training).  
3. RL + static explainability (produce explanations but no dynamic co‐correction).  

D3. Evaluation Metrics  
1. Clinical performance: mean reduction in HbA1c, RMSE of glycemic prediction.  
2. Equity metrics:  
   • Disparity over time $\Delta_t$.  
   • Looping inequity  
     $$L = \frac{1}{T}\sum_{t=1}^T\Bigl[\mathbb{E}[Y_t\mid \text{with AI}] - \mathbb{E}[Y_t\mid \text{without AI}]\Bigr].$$  
3. Trust calibration: average trust $T$ by cohort.  
4. Robustness: sensitivity of outcomes to noise $\varepsilon_t$ and to hyperparameters $(\gamma,\lambda,\tau)$.  
5. Statistical significance: paired $t$‐tests and Wilcoxon signed‐rank tests on disparity reduction.

D4. Implementation Details  
• Neural policy: two‐layer feedforward network with ReLU activations, 64 units each.  
• Optimizer: Adam with learning rate $1e^{-4}$.  
• Simulator: implemented in Python, PyTorch for RL, DoWhy for causal mediation, SHAP for explanations.  
• Reproducibility: code and simulated data will be open‐sourced under MIT license.

4. Expected Outcomes & Impact  
Expected Outcomes  
a. A validated coevolutionary simulation toolkit that can be extended to other chronic diseases or high‐stakes domains (e.g., mental health, oncology).  
b. Empirical evidence that our bias‐aware co‐correction reduces long‐term disparities by $\mathbf{25\%\sim40\%}$ relative to static baselines, while maintaining or improving clinical performance within a $95\%$ confidence interval.  
c. A new, operational “looping inequity” metric that quantifies how human‐AI feedback loops contribute to inequities over time.  
d. A set of best practices and hyperparameter guidelines for healthcare practitioners and AI developers to monitor and adjust human‐AI systems in production.

Broader Impact  
Our research advances the HAIC agenda by:  
• Moving beyond one‐shot fairness fixes to continuous co‐adaptation analysis.  
• Integrating causal inference and explainable AI to keep AI systems aligned with evolving human values and trust dynamics.  
• Providing policy‐relevant insights about governance of adaptive AI in healthcare: for instance, triggering audits when $\Delta_t$ or $L$ exceed regulatory thresholds.  
• Promoting participatory design by modeling patient trust and feedback as integral to algorithm updates, thereby strengthening human autonomy.  

By illustrating a clear path from theory to deployable systems, this proposal not only contributes to the academic understanding of human‐AI coevolution but also equips stakeholders—clinicians, ethicists, regulators—with actionable tools to ensure that adaptive AI in healthcare remains equitable, robust, and trustworthy over the long run.