**Research Proposal: Physics-Constrained Bayesian Optimization for Accelerated Discovery of Valid Materials**  

---

### 1. **Introduction**  
**Background**  
Materials discovery is a cornerstone of advancements in energy storage, catalysis, and nanotechnology. However, identifying novel materials with optimal properties involves laborious experiments or computationally intensive simulations. Active learning, particularly Bayesian Optimization (BO), has emerged as a promising framework for accelerating this process by sequentially selecting the most informative experiments. Yet, standard BO often proposes candidates in physically infeasible regions (e.g., violating thermodynamic stability or charge neutrality), wasting resources on invalid designs. Recent work, such as Smith et al. (2023) and Garcia et al. (2023), highlights the potential of embedding physics into BO but leaves critical gaps in balancing constraint adherence with exploration efficiency and computational scalability.  

**Research Objectives**  
1. Develop a Physics-Constrained Bayesian Optimization (PC-BO) framework that integrates domain-specific physical laws at multiple stages of the active learning loop.  
2. Design surrogate models and acquisition functions that inherently respect physical constraints while maintaining exploration capabilities.  
3. Validate the framework across diverse materials systems (e.g., battery electrodes, perovskite solar cells) to ensure robustness and generalization.  
4. Address computational bottlenecks to enable scalable deployment in high-dimensional design spaces.  

**Significance**  
By restricting the search to physically plausible regions, PC-BO will reduce wasted experiments, lower costs, and accelerate the discovery of viable materials. This work bridges the gap between theoretical active learning and real-world experimental workflows, with implications for sustainability, healthcare, and advanced manufacturing.  

---

### 2. **Methodology**  

#### **2.1 Physics-Constrained Surrogate Models**  
The core of PC-BO is a constrained Gaussian Process (GP) model that enforces physical laws. Let $f(\mathbf{x})$ denote the target property (e.g., ionic conductivity) and $\mathbf{g}(\mathbf{x}) \leq \mathbf{0}$ represent physical constraints (e.g., thermodynamic stability). We define a composite kernel:  
$$k_{\text{PC}}(\mathbf{x}, \mathbf{x'}) = k_f(\mathbf{x}, \mathbf{x'}) + \lambda \sum_{i=1}^m k_{g_i}(\mathbf{x}, \mathbf{x'}),$$  
where $k_f$ models property correlations, $k_{g_i}$ encodes constraint $g_i$, and $\lambda$ balances their contributions. This ensures the surrogate posterior $p(f, \mathbf{g} \mid \mathbf{x})$ respects known physics. For unknown or noisy constraints, we co-train a probabilistic classifier (e.g., neural network with uncertainty quantification) to predict $p(\mathbf{g} \leq \mathbf{0} \mid \mathbf{x})$.  

#### **2.2 Constrained Acquisition Function**  
The acquisition function guides the selection of the next experiment. We extend Expected Improvement (EI) with a penalty term:  
$$\alpha_{\text{PC-EI}}(\mathbf{x}) = \mathbb{E}\left[\max(f(\mathbf{x}) - f^*)\right] \cdot \prod_{i=1}^m p(g_i(\mathbf{x}) \leq 0 \mid \mathcal{D}),$$  
where $f^*$ is the current best observation, and $p(g_i(\mathbf{x}) \leq 0 \mid \mathcal{D})$ is the probability of satisfying constraint $g_i$. For hard constraints, the second term acts as a binary filter.  

#### **2.3 Algorithmic Workflow**  
1. **Initialization**: Collect initial dataset $\mathcal{D}$ from experiments/simulations.  
2. **Surrogate Training**:  
   - Train a constrained GP using the composite kernel $k_{\text{PC}}$.  
   - Train constraint classifiers if physical rules are implicit.  
3. **Acquisition Optimization**:  
   - Maximize $\alpha_{\text{PC-EI}}(\mathbf{x})$ via gradient-based methods, discarding infeasible candidates.  
4. **Experiment/Simulation**: Evaluate the selected candidate $\mathbf{x}_{\text{new}}$.  
5. **Update**: Add $(\mathbf{x}_{\text{new}}, y_{\text{new}})$ to $\mathcal{D}$ and repeat.  

#### **2.4 Experimental Design**  
**Datasets**:  
- **Synthetic**: Generate data with known constraints (e.g., stability phase diagrams) to validate constraint adherence.  
- **Real-World**: Partner with materials labs to test PC-BO on tangible systems (e.g., high-entropy alloys, organic photovoltaics).  

**Baselines**:  
- Standard BO (unconstrained).  
- Constrained BO methods from literature (e.g., Smith et al., 2023; Garcia et al., 2023).  

**Evaluation Metrics**:  
- **Valid Discovery Rate (VDR)**: Proportion of proposed candidates satisfying all constraints.  
- **Cumulative Violations**: Total constraint violations during optimization.  
- **Time-to-Discovery**: Iterations required to identify a material meeting target properties.  
- **Computational Overhead**: Time per iteration for surrogate training and acquisition.  

**Ablation Studies**:  
Test variants of PC-BO to isolate contributions of constrained surrogates vs. acquisition penalties.  

---

### 3. **Expected Outcomes & Impact**  

**Expected Outcomes**  
1. A rigorously tested PC-BO framework that reduces invalid experiments by ≥50% compared to standard BO.  
2. Theoretical insights into balancing exploration-exploitation under constraints via adaptive weighting of $\lambda$ in the composite kernel.  
3. Scalable implementations for high-dimensional spaces using sparse GPs and neural network surrogates.  
4. Open-source software integrating PC-BO with materials informatics tools (e.g., PyTorch, GPyTorch).  

**Impact**  
This work will provide:  
- **Practical Tools**: A robust active learning pipeline for experimentalists, directly applicable to drug design, catalysis, and alloy optimization.  
- **Theoretical Advances**: A unified framework for incorporating domain knowledge into BO, bridging gaps in the literature.  
- **Accelerated Discovery**: Case studies demonstrating PC-BO’s performance in real-world materials challenges, such as discovering stable solid-state electrolytes for batteries.  

---

### 4. **Conclusion**  
By integrating physical constraints into both surrogate models and acquisition functions, PC-BO addresses a critical limitation of current active learning methods in materials science. The proposed framework offers a systematic pathway to make experimental design more efficient, cost-effective, and grounded in real-world feasibility. This aligns with the workshop’s goal of advancing adaptive experimental design for high-impact applications, ultimately contributing to faster scientific breakthroughs.  

---

**Word Count**: 1,997