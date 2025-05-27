Okay, here is a research proposal based on the provided task description, idea, and literature review.

---

**1. Title:** **Accelerating Real-World Materials Discovery via Physics-Constrained Bayesian Optimization**

**2. Introduction**

*   **Background:** The discovery and design of novel materials with tailored properties is a cornerstone of technological advancement, impacting fields ranging from renewable energy (photovoltaics, catalysts) and electronics (semiconductors) to medicine (biomaterials) and structural engineering. Traditionally, materials discovery relies heavily on intuition-driven Edisonian approaches or computationally intensive high-throughput screening, both of which are often prohibitively expensive and time-consuming. A typical materials development cycle can take decades and cost millions of dollars. Machine learning (ML), particularly active learning strategies like Bayesian Optimization (BO), offers a promising paradigm shift towards data-driven, accelerated discovery [Shahriari et al., 2016]. BO iteratively builds a probabilistic surrogate model (commonly a Gaussian Process, GP) of the material property landscape (e.g., mapping chemical composition or structure to stability or conductivity) and uses an acquisition function (e.g., Expected Improvement, EI) to intelligently select the next most informative experiment (physical synthesis or computational simulation) to perform. This allows for efficient exploration and exploitation of the vast materials design space with significantly fewer evaluations compared to brute-force methods.

    However, a critical limitation hinders the direct application of standard BO in real-world scientific discovery, particularly in materials science. BO algorithms, optimizing solely based on the surrogate model and acquisition function, often explore regions of the design space corresponding to physically implausible or unstable materials. For instance, they might suggest compounds violating fundamental principles like charge neutrality, thermodynamic stability criteria (e.g., predicting highly unstable phases), or known synthesis constraints. Evaluating these physically unrealistic candidates constitutes a significant waste of precious experimental resources (time, consumables, instrument access) or expensive computational cycles (e.g., Density Functional Theory calculations). This inefficiency gap underscores the need highlighted by the Workshop on Adaptive Experimental Design and Active Learning in the Real World: bridging the gap between principled ML algorithms and practical, resource-constrained scientific applications. Integrating domain knowledge, specifically physical laws and constraints, directly into the active learning loop is paramount for developing truly effective and resource-efficient materials discovery pipelines. Recent studies, including those by Smith et al. (2023), Kim et al. (2023), and Patel et al. (2023), have begun exploring this direction, demonstrating the potential of physics-informed ML.

*   **Research Objectives:** This research aims to develop, implement, and rigorously evaluate a **Physics-Constrained Bayesian Optimization (PC-BO)** framework designed to accelerate the discovery of *valid* and high-performing materials. The specific objectives are:
    1.  **Develop the PC-BO Framework:** Formalize a BO framework that explicitly incorporates known physical constraints relevant to materials science (e.g., thermodynamic stability, charge neutrality, structural rules, synthesis feasibility heuristics).
    2.  **Implement Constraint Handling Mechanisms:** Implement and compare two primary strategies for constraint integration within the PC-BO framework:
        *   **Physics-Informed Surrogate Models:** Utilize constrained Gaussian Processes (cGP) that inherently encode physical feasibility into the probabilistic model of the material property landscape, potentially building on work like Kim et al. (2023).
        *   **Constraint-Aware Acquisition Functions:** Modify standard acquisition functions (e.g., Expected Improvement, Upper Confidence Bound) to explicitly penalize or discard candidate points predicted to violate physical constraints, extending ideas from Patel et al. (2023) and Garcia et al. (2023).
    3.  **Evaluate Performance on Realistic Materials Discovery Tasks:** Quantitatively assess the performance of the PC-BO framework against standard BO and random sampling baselines on representative materials discovery problems using simulated (e.g., DFT-based) and potentially publicly available experimental datasets. Evaluation will focus on efficiency (speed of discovering high-performing *valid* materials), validity rate of suggestions, and computational overhead.
    4.  **Analyze the Impact of Different Constraint Types:** Investigate the relative importance and synergistic effects of incorporating different types of physical constraints (e.g., stability vs. synthesis rules) on the efficiency and outcome of the materials discovery process.

*   **Significance:** This research directly addresses the core themes of the workshop by focusing on **real-world experimental design** in materials science, integrating **domain knowledge** (physics) into **efficient active learning** (BO), and aiming for **sample-efficient** discovery. By guiding the search towards physically plausible regions, the proposed PC-BO framework is expected to significantly reduce the number of costly failed experiments or simulations. This translates to:
    *   **Accelerated Scientific Discovery:** Faster identification of novel materials with desired properties.
    *   **Reduced Research Costs:** More efficient utilization of experimental and computational resources.
    *   **Enhanced Reliability:** Increased confidence in the candidates proposed by the ML model, fostering trust between ML tools and domain scientists.
    *   **Broader Applicability:** Providing a robust and adaptable framework that can potentially be applied to other scientific domains facing similar challenges (e.g., drug design, protein engineering) where domain constraints are critical.
    This work aims to make a tangible contribution towards making active learning a more practical and impactful tool for real-world scientific discovery, directly addressing the workshop's goal of identifying solutions that bridge theory and practice.

**3. Methodology**

*   **Research Design:** The research will follow a computational methodology involving algorithm development, implementation, and simulation-based validation on materials science tasks. We will compare the proposed PC-BO approach against relevant baselines in controlled settings.

*   **PC-BO Framework Overview:**
    The core of the PC-BO framework is an iterative loop:
    1.  **Initialization:** Start with a small dataset $D_0 = \{(\mathbf{x}_i, y_i, \mathbf{c}_i)\}_{i=1}^{N_0}$, where $\mathbf{x}_i \in \mathcal{X}$ is the material descriptor (e.g., composition vector, structural parameters), $y_i = f(\mathbf{x}_i) + \epsilon_i$ is the observed target property (e.g., formation energy, band gap), $\epsilon_i$ is observation noise, and $\mathbf{c}_i$ represents whether known physical constraints are satisfied at $\mathbf{x}_i$.
    2.  **Surrogate Modeling:** Fit a probabilistic surrogate model (typically a GP) to the current data $D_t$ to approximate the objective function $f(\mathbf{x})$ and potentially the constraint functions $g_j(\mathbf{x})$.
    3.  **Constraint Handling:** Incorporate physical constraints using one of the proposed mechanisms (detailed below).
    4.  **Acquisition Function Optimization:** Select the next point $\mathbf{x}_{t+1}$ to evaluate by maximizing a (potentially constraint-aware) acquisition function $\alpha(\mathbf{x})$ over the design space $\mathcal{X}$:
        $$ \mathbf{x}_{t+1} = \arg \max_{\mathbf{x} \in \mathcal{X}} \alpha(\mathbf{x} | D_t) $$
    5.  **Evaluation:** Perform the "experiment" (e.g., run a DFT calculation or query a database/oracle) to obtain $y_{t+1} = f(\mathbf{x}_{t+1}) + \epsilon_{t+1}$ and determine constraint satisfaction $\mathbf{c}_{t+1}$.
    6.  **Update Data:** Augment the dataset $D_{t+1} = D_t \cup \{(\mathbf{x}_{t+1}, y_{t+1}, \mathbf{c}_{t+1})\}$.
    7.  **Iteration:** Repeat steps 2-6 until a stopping criterion is met (e.g., budget exhausted, target property achieved).

*   **Data Collection / Simulation:**
    *   **Domain:** We will focus on specific materials discovery tasks, such as finding stable ternary compounds with low formation energy within a defined chemical subspace (e.g., exploring combinations of elements A, B, C) or optimizing a functional property like the thermoelectric figure of merit (ZT).
    *   **Source:** Primary data will be generated using well-established computational methods like Density Functional Theory (DFT) via packages like VASP or Quantum Espresso. This provides a controlled environment where the "ground truth" property $f(\mathbf{x})$ and constraint satisfaction $g_j(\mathbf{x})$ can be evaluated on demand, simulating the experimental process. We may also leverage existing large-scale materials databases (e.g., Materials Project, OQMD) as sources of initial data or for broader validation.
    *   **Representation $\mathbf{x}$:** Material compositions will be represented as fractional vectors. If structures are considered, appropriate structural descriptors (e.g., based on symmetry, bond distances/angles) will be used.

*   **Algorithmic Details:**

    1.  **Surrogate Model (Gaussian Process):** We will use GPs as the primary surrogate model due to their ability to provide uncertainty estimates, crucial for BO. A GP defines a distribution over functions:
        $$ f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) $$
        where $m(\mathbf{x})$ is the mean function (often set to zero or a simple polynomial) and $k(\mathbf{x}, \mathbf{x}')$ is the covariance (kernel) function, capturing the similarity between points. We will explore standard kernels like the Radial Basis Function (RBF) or Matérn kernels, potentially tailored to materials representations. Hyperparameters will be optimized by maximizing the marginal likelihood on the observed data $D_t$.

    2.  **Physics Constraints Formulation:** We will mathematically formulate relevant physical constraints:
        *   **Thermodynamic Stability:** Often approximated by the energy above the convex hull ($E_{hull}$). A material is considered stable or metastable if $g_{stab}(\mathbf{x}) = E_{hull}(\mathbf{x}) \le \delta$, where $\delta$ is a small energy threshold (e.g., 0-50 meV/atom). Calculating $E_{hull}$ requires knowing the energies of competing phases, which can sometimes be estimated or looked up.
        *   **Charge Neutrality:** For ionic compounds, the sum of formal charges must be zero: $g_{charge}(\mathbf{x}) = \sum_k n_k q_k = 0$, where $n_k$ is the number of atoms of type $k$ with charge $q_k$ in the predicted stoichiometry $\mathbf{x}$. This is often a hard deterministic constraint.
        *   **Synthesis Feasibility Rules (Heuristic):** Incorporate domain-specific rules, e.g., certain element combinations rarely form stable compounds, or specific structural motifs are unlikely. These might be represented as boolean functions $g_{synth}(\mathbf{x}) \in \{0, 1\}$.

    3.  **Constraint Handling Mechanisms:**
        *   **Mechanism A: Constrained Gaussian Processes (cGP):** This involves modifying the GP itself to respect the constraints. For instance, if a constraint is defined as $g(\mathbf{x}) \le 0$, methods like those described by Gardner et al. (2014) or the approach in Kim et al. (2023) can be used. This might involve treating the constraint function $g(\mathbf{x})$ with another GP (if it's unknown and noisy) and modifying the posterior predictive distribution of $f(\mathbf{x})$ to only have mass in the feasible region, or using basis function approaches that satisfy the constraints by construction. This approach internalizes the constraints within the model's predictions and uncertainty.
        *   **Mechanism B: Constraint-Aware Acquisition Function:** This approach uses a standard GP for $f(\mathbf{x})$ but modifies the acquisition function $\alpha(\mathbf{x})$ to account for constraints.
            *   We may need separate GPs $h_j(\mathbf{x}) \sim \mathcal{GP}(m_j(\mathbf{x}), k_j(\mathbf{x}, \mathbf{x}'))$ to model unknown constraint functions $g_j(\mathbf{x})$ based on observed constraint satisfaction data $\mathbf{c}_i$.
            *   The probability of satisfying constraint $j$ at point $\mathbf{x}$ can be estimated from its GP model, e.g., $P(g_j(\mathbf{x}) \le \delta | D_t) = \Phi\left(\frac{\delta - \mu_{h_j}(\mathbf{x})}{\sigma_{h_j}(\mathbf{x})}\right)$, where $\Phi$ is the standard normal CDF, and $\mu_{h_j}, \sigma_{h_j}$ are the posterior mean and standard deviation of $h_j(\mathbf{x})$.
            *   For deterministic constraints (like charge neutrality), $P(g_j(\mathbf{x}) \le \delta)$ is either 0 or 1.
            *   The overall probability of satisfying all $M$ constraints is $P(\text{feasible}(\mathbf{x}) | D_t) = \prod_{j=1}^M P(g_j(\mathbf{x}) \le \delta_j | D_t)$.
            *   A constrained acquisition function, e.g., Constrained Expected Improvement (CEI), can be defined as:
                $$ \alpha_{CEI}(\mathbf{x}) = \alpha_{EI}(\mathbf{x}) \times P(\text{feasible}(\mathbf{x}) | D_t) $$
                where $\alpha_{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f(\mathbf{x}^*), 0) | D_t]$ is the standard Expected Improvement over the current best *observed feasible* value $f(\mathbf{x}^*)$. Similar modifications can be applied to UCB (Constrained UCB). This approach decouples the objective modeling from constraint handling, potentially offering more flexibility.

*   **Experimental Design for Validation:**
    *   **Baselines:**
        1.  Standard Bayesian Optimization (using GP + EI/UCB without constraints).
        2.  Random Sampling (within the design space $\mathcal{X}$).
        3.  (Optional) Another relevant physics-informed method from the literature (e.g., specific approach from Smith et al. 2023 if implementation details are available).
    *   **Tasks:**
        1.  *DFT Simulation Task:* Find the ternary compound (e.g., in the Li-Mn-O system) with the minimum formation energy (maximizing stability) below the convex hull, subject to charge neutrality. The design space $\mathcal{X}$ will be the compositional space (e.g., $Li_x Mn_y O_z$ with $x+y+z=1$). Function evaluations involve running DFT calculations. Constraints: $E_{hull} \le \delta$ (thermodynamic stability, evaluated via DFT) and $\sum n_k q_k = 0$ (charge neutrality, deterministic).
        2.  *(Potential) Database Task:* Optimize a known property (e.g., band gap) within a subset of a large materials database (e.g., Materials Project), using the database as an oracle. Constraints (e.g., stability $E_{hull} \le \delta$ ) would also be queried from the database. This allows testing scalability on larger candidate pools.
    *   **Procedure:**
        *   For each task and each method (PC-BO with Mechanism A, PC-BO with Mechanism B, Standard BO, Random), run $K=20-30$ independent trials with different random initializations ($N_0=5-10$ points).
        *   Each trial runs for a fixed budget of $T$ total evaluations (e.g., $T=200$ DFT calculations).
        *   Track the performance metrics at each iteration $t=1, ..., T$.
    *   **Evaluation Metrics:**
        1.  **Best Feasible Value Found:** Track the minimum (or maximum) objective value $y^*$ found among *valid* candidates suggested up to iteration $t$, averaged over the $K$ trials. $y^*(t) = \min \{ y_i \mid i \le t, \mathbf{c}_i \text{ is feasible} \}$. Plot $y^*(t)$ vs. $t$.
        2.  **Cumulative Regret:** If the true optimal feasible value $y_{opt}$ is known or can be estimated, measure the cumulative regret.
        3.  **Convergence Speed:** Number of evaluations $t$ required to find a valid material with property $y \le y_{target}$ (for minimization).
        4.  **Validity Rate:** Percentage of suggested candidates $\mathbf{x}_t$ (for $t > N_0$) that satisfy all physical constraints. Plot this rate vs. $t$.
        5.  **Computational Cost:** Measure the average wall-clock time required per iteration (model fitting + acquisition function optimization) for each method.

**4. Expected Outcomes & Impact**

*   **Expected Outcomes:**
    1.  **A validated PC-BO framework:** A robust implementation of the Physics-Constrained Bayesian Optimization framework, incorporating selectable constraint handling mechanisms (cGP and constraint-aware acquisition functions).
    2.  **Quantitative performance evaluation:** Clear empirical results demonstrating the effectiveness of PC-BO compared to standard BO and random search on representative materials discovery tasks. We expect PC-BO to find high-performing, valid materials significantly faster (i.e., requiring fewer expensive evaluations) and to propose far fewer physically implausible candidates.
    3.  **Comparative analysis of constraint handling:** Insights into the relative merits, computational costs, and applicability domains of using constrained GPs versus modifying the acquisition function for incorporating physical knowledge.
    4.  **Understanding of constraint impact:** Evidence on how different types of constraints (e.g., thermodynamic vs. charge neutrality vs. heuristic synthesis rules) individually and collectively influence the search efficiency and the quality of discovered materials.
    5.  **Open-source contribution (potential):** Release of the PC-BO implementation as a software package to benefit the broader materials science and ML communities.

*   **Impact:** This research will contribute directly to the goals of the workshop by demonstrating a concrete methodology for integrating crucial domain knowledge into active learning for enhanced real-world performance. The primary impact will be the acceleration of the materials discovery cycle, enabling scientists to identify promising candidate materials with reduced experimental or computational cost. By explicitly handling physical constraints, the PC-BO framework will generate more reliable and actionable suggestions, fostering greater adoption of ML techniques in the physical sciences. This work will address key challenges identified in the literature, such as balancing exploration/exploitation under constraints and improving sample efficiency, while providing insights into managing the computational trade-offs involved. If successful, the PC-BO framework could serve as a blueprint for similar constrained optimization problems in drug design, protein engineering, and other scientific domains where experiments are costly and domain constraints are essential. Ultimately, this research aims to bridge the gap between theoretical active learning algorithms and their practical deployment in high-impact scientific applications, leading to faster innovation.

**5. References** (Illustrative - including provided literature and key BO/GP papers)

*   Gardner, J. R., Kusner, M. J., Xu, Z. E., Weinberger, K. Q., & Cunningham, J. P. (2014). Bayesian Optimization with Inequality Constraints. *Proceedings of the 31st International Conference on Machine Learning (ICML)*.
*   Garcia, M., Thompson, N., & Zhang, O. (2023). Incorporating Thermodynamic Constraints in Bayesian Optimization for Materials Design. *arXiv preprint arXiv:2305.56789*.
*   Kim, D., Martinez, E., & Wang, F. (2023). Constrained Gaussian Processes for Materials Design. *arXiv preprint arXiv:2302.23456*.
*   Patel, G., Nguyen, H., & Chen, I. (2023). Active Learning with Physical Constraints in Materials Science. *arXiv preprint arXiv:2303.34567*.
*   Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. *Proceedings of the IEEE, 104*(1), 148-175.
*   Smith, A., Johnson, B., & Lee, C. (2023). Physics-Informed Bayesian Optimization for Accelerated Materials Discovery. *arXiv preprint arXiv:2301.12345*.
*   (Include other relevant papers from the provided literature list and standard GP/BO references as needed).

---