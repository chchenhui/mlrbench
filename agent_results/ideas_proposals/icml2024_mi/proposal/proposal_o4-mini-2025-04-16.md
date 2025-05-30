1. Title  
Modeling Cognitive Effort in Human Feedback for Robust AI Alignment  

2. Introduction  
Background  
Aligning AI agents with human intentions and values is critical for deploying safe, reliable systems in domains ranging from recommender systems and autonomous vehicles to large language models. Existing human‐in‐the‐loop methods—such as Reinforcement Learning with Human Feedback (RLHF) or Learning from Demonstrations (LfD)—typically assume that human feedback directly reflects stable, unbiased preferences and that humans are fully rational. In practice, however, providing feedback incurs cognitive cost: humans simplify choices under time pressure or fatigue and rely on heuristics that introduce systematic biases. Ignoring these factors leads AI systems to misinterpret signals of true preferences, undermining alignment.  

Research Objectives  
This proposal aims to develop an effort‐aware framework for inferring human preferences under bounded rationality. Our specific objectives are:  
• Formalize a cognitive effort‐aware feedback model that jointly represents true preferences and mental effort costs.  
• Design a hierarchical Bayesian inference algorithm that recovers both preference parameters and effort levels from observed feedback.  
• Collect and curate behavioral datasets capturing human feedback under controlled task complexities and time constraints.  
• Rigorously evaluate how accounting for cognitive effort improves preference inference accuracy and reduces misalignment in downstream tasks.  

Significance  
By explicitly modeling the trade‐off between decision accuracy and mental effort, we will enable AI systems to distinguish between intended preferences and effort‐induced noise. This advance is vital for applications where feedback is effortful—such as healthcare decision support, educational tutoring systems, and high‐stakes robotics—thus improving robustness, user trust, and ethical alignment.  

3. Methodology  
Our methodology comprises four components: (A) Model Formalization, (B) Hierarchical Inference Algorithm, (C) Data Collection Protocol, and (D) Experimental Validation.  

A. Model Formalization  
We represent each human feedback instance as a choice or ranking action $a$ in state $s$ (e.g., a ranking prompt over $n$ items), generated by an unknown reward parameter vector $\theta\in\mathbb{R}^d$ and a cognitive effort level $e\ge0$.  
1. Utility with Effort Cost  
   We posit that a human’s net utility for action $a$ is  
   $$U(a\mid s,\theta,e)\;=\;R(a\mid s,\theta)\;-\;\lambda(e)\,C(a\mid s)\,,$$  
   where  
   • $R(a\mid s,\theta)$ is the reward modeled by a linear or nonlinear mapping $R(a\mid s,\theta)=\theta^\top\phi(s,a)$, with feature map $\phi(s,a)\in\mathbb{R}^d$.  
   • $C(a\mid s)$ quantifies the cognitive cost of evaluating or enumerating action $a$ (e.g., number of pairwise comparisons in a ranking task).  
   • $\lambda(e)$ is an increasing function modeling sensitivity to effort (e.g., $\lambda(e)=\log(1+e)$).  

2. Stochastic Choice Model  
   Under bounded rationality, choices are noisy. We adopt a softmax‐style likelihood:  
   $$P(a\mid s,\theta,e)\;=\;\frac{\exp\bigl(U(a\mid s,\theta,e)/\tau\bigr)}{\sum_{a'}\exp\bigl(U(a'\mid s,\theta,e)/\tau\bigr)}\,, $$  
   with temperature $\tau>0$ controlling randomness.  

3. Hierarchical Priors  
   We impose priors on individual parameters and population‐level hyperpriors:  
   $$\theta_i\sim\mathcal{N}(\mu_\theta,\Sigma_\theta),\quad e_i\sim\mathrm{Gamma}(\alpha_e,\beta_e),$$  
   and  
   $$\mu_\theta\sim\mathcal{N}(0,\sigma^2 I),\quad \Sigma_\theta\sim\mathrm{InvWishart}(W_0,\nu_0),$$  
   $$\alpha_e,\beta_e\sim\mathrm{Gamma}(a_0,b_0)\,,$$  
   for each participant $i$.  

4. Posterior Inference Objective  
   Given dataset $\mathcal{D}=\{(s_{i,j},a_{i,j})\}$, the joint posterior is  
   $$p(\Theta,E,\mu_\theta,\Sigma_\theta,\alpha_e,\beta_e\mid\mathcal{D})\propto p(\mathcal{D}\mid\Theta,E)\,p(\Theta\mid\mu_\theta,\Sigma_\theta)\,p(E\mid\alpha_e,\beta_e)\,\pi(\mu_\theta,\Sigma_\theta)\,\pi(\alpha_e,\beta_e),$$  
   where  
   $$p(\mathcal{D}\mid\Theta,E)=\prod_{i,j}P(a_{i,j}\mid s_{i,j},\theta_i,e_i)\,. $$  

B. Hierarchical Inference Algorithm  
We will implement a scalable variational inference (VI) scheme with the following steps:  
1. Variational Factorization  
   Assume a mean‐field factorization  
   $$q(\Theta,E,\mu_\theta,\Sigma_\theta,\alpha_e,\beta_e)=q(\Theta)\,q(E)\,q(\mu_\theta)\,q(\Sigma_\theta)\,q(\alpha_e)\,q(\beta_e)\,. $$  
2. Evidence Lower Bound (ELBO)  
   Derive the ELBO  
   $$\mathcal{L}(q) = \mathbb{E}_{q}\Bigl[\log p(\mathcal{D},\Theta,E,\dots)\Bigr] - \mathbb{E}_{q}\Bigl[\log q(\cdot)\Bigr]\,, $$  
   and compute gradients via the reparameterization trick for continuous variables.  
3. Optimization  
   Use stochastic gradient ascent (Adam optimizer) with mini‐batches over participants and feedback instances.  
4. Posterior Samples for Uncertainty  
   After convergence, draw samples from each factor $q$ to quantify uncertainty in $\theta_i$ and $e_i$.  

We will also implement a baseline MCMC sampler (NUTS) for smaller datasets to validate the VI approximation.  

C. Data Collection Protocol  
We will collect two complementary datasets: synthetic and human.  
1. Synthetic Data  
   • Generate $M$ simulated agents with known $(\theta_i^*,e_i^*)$ drawn from prior.  
   • Simulate feedback over $T$ states per agent under varying cost functions (e.g., different $n$).  
   • Purpose: validate inference accuracy where ground truth is known.  
2. Human Behavioral Study  
   • Participants: $N\approx100$ recruited via online platforms (e.g., Mechanical Turk).  
   • Tasks:  
     – Pairwise comparison tasks (choose preferred image or sentence) with time limits (5s, 10s, 20s).  
     – $k$‐way ranking tasks over $k\in\{5,10,15\}$ items, under no‐time, moderate‐time, and low‐time conditions.  
   • Measures: recorded actions $a_{i,j}$, response times, self‐reported effort.  
   • Cost Estimation: set $C(a\mid s)$ proportional to number of pairwise evaluations implied by $a$.  

All protocols will follow institutional review board (IRB) guidelines, with informed consent and minimal risk.  

D. Experimental Validation  
We will evaluate on both synthetic and human data with the following metrics and baselines.  

1. Evaluation Metrics  
   • Preference Inference Accuracy  
     – Root Mean Square Error: $\mathrm{RMSE}(\theta)=\sqrt{\frac{1}{d}\|\hat\theta-\theta^*\|^2}$.  
     – Kendall’s Tau correlation between true and inferred item utilities in ranking tasks.  
   • Effort Estimation Accuracy (synthetic)  
     – Mean Absolute Percentage Error: $\mathrm{MAPE}(e)=\frac{1}{N}\sum_i|\hat e_i-e_i^*|/e_i^*$.  
   • Predictive Log‐Likelihood (human)  
     – Average $\frac{1}{|\mathcal{D}_{\mathrm{test}}|}\sum\log P(a_{i,j}\mid s_{i,j};\hat\theta_i,\hat e_i)$.  
   • Downstream Regret  
     – When substituting inferred $\hat\theta$ into a policy, measure worst‐case regret over held‐out states.  

2. Baseline Methods  
   • Standard IRL (no effort): treat all feedback as noise from rational agent.  
   • Hybrid IRL (Ren et al., 2024): integrates expert and online data but ignores effort.  
   • Inverse Decision Modeling (Jarrett et al., 2023): learns biased beliefs but omits explicit effort cost.  

3. Ablation Studies  
   • Fixed vs. Learned Cost Sensitivity: compare models where $\lambda(e)$ is fixed or parametrized.  
   • Prior Sensitivity: vary hyperpriors on $(\alpha_e,\beta_e)$ and $(\mu_\theta,\Sigma_\theta)$.  
   • Task Complexity: analyze performance decay as $k$ in ranking increases.  

4. Statistical Analysis  
   • Use paired t‐tests or Wilcoxon signed‐rank tests to compare our model against baselines.  
   • Report 95% credible intervals from variational posterior for key parameters.  

Implementation Details  
• Platform: Python with PyTorch for VI, Pyro for probabilistic modeling.  
• Code & Data Release: All code, synthetic data generators, and anonymized human datasets will be open‐sourced upon publication.  

4. Expected Outcomes & Impact  
Expected Outcomes  
• A novel effort‐aware IRL model that quantifies cognitive costs in human feedback.  
• A scalable hierarchical Bayesian inference algorithm yielding accurate joint estimates of preferences and effort.  
• A publicly available dataset of human feedback under varying effort conditions, along with analysis scripts.  
• Empirical demonstration that accounting for cognitive effort reduces preference inference error by 15–30% under high‐complexity tasks, compared to state‐of‐the‐art baselines.  
• Identification of systematic biases (e.g., under‐weighing low‐effort options) that arise purely from effort constraints.  

Impact  
By integrating effort dynamics, AI systems can:  
• Distinguish true preferences from effort‐induced shortcuts, reducing misalignment in critical domains (e.g., medical treatment recommendation, where patients may simplify complex trade‐offs).  
• Adapt feedback interfaces dynamically (e.g., reduce choice set size when users exhibit high fatigue) to elicit more reliable signals.  
• Guide future RLHF protocols by calibrating human feedback models with realistic bounded rationality considerations.  
• Contribute to the theoretical foundations of Human‐AI Alignment by bridging cognitive science (bounded rationality) and machine learning (IRL).  

Ultimately, our work advances both the safety and usability of AI agents interacting with humans, paving the way for more trustworthy, user‐centric systems.  

5. Conclusion and Future Directions  
We have outlined a comprehensive research plan to model cognitive effort in human feedback, address key challenges in data collection and inference, and validate the approach in both synthetic and real‐world settings. Future extensions could explore dynamic effort modeling over time (learning fatigue trajectories), multi‐modal feedback (combining verbal and behavioral cues), and integration with deep neural architectures for large‐scale RLHF in language models. This effort‐aware framework promises to be a foundational step toward robust, ethically aligned AI systems.