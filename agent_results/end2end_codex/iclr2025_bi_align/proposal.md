1. Title  
Uncertainty-Driven Co-Adaptation: A Bayesian Framework for Bidirectional Human-AI Alignment  

2. Introduction  
Background  
Traditional AI alignment primarily focuses on a unidirectional process—shaping AI behavior to satisfy fixed human specifications (AI-centered alignment). Recent surveys (e.g., “Towards Bidirectional Human-AI Alignment,” 2024) argue that human preferences evolve over time and that humans themselves must adapt to AI capabilities to maintain effective collaboration (human-centered alignment). Static one-way approaches can lead to misalignment when user needs shift or when the AI’s decision logic becomes opaque, undermining trust and human agency.  

Bidirectional human-AI alignment reframes the interaction as a co-adaptation loop: the AI refines its internal model of user values through targeted feedback, while the user continuously updates their mental model of the AI by observing uncertainty cues and alternative suggestions. This dynamic paradigm promises more robust, transparent, and scalable alignment in real-world settings such as recommendation, planning, and assistive decision systems.  

Research Objectives  
This proposal aims to develop, formalize, and empirically validate an Uncertainty-Driven Co-Adaptation (UDCA) framework that:  
1. Builds a Bayesian user-preference model to quantify uncertainty over latent human values.  
2. Actively queries users when the AI’s “value ambiguity” exceeds a dynamic threshold, minimizing unnecessary interruptions.  
3. Presents alternative action paths with confidence indicators to empower users to critique and refine AI suggestions.  
4. Demonstrates effectiveness on recommendation and planning benchmarks, measuring alignment quality, query efficiency, and user trust.  

Significance  
By tightly integrating uncertainty estimation, active preference elicitation, and transparent suggestion visualization, UDCA addresses key challenges in bidirectional alignment: accurate uncertainty quantification, evolving preferences, intuitive human-AI communication, scalability, and trust. The outcomes will offer a generalizable algorithmic and interface blueprint for co-adaptive systems that preserve human agency while ensuring AI systems respect dynamically shifting human values.  

3. Methodology  
Our methodology comprises four components: (A) Bayesian User-Preference Modeling, (B) Uncertainty-Driven Query Selection, (C) Bidirectional Interaction Protocol & UI Design, and (D) Experimental Evaluation.  

A. Bayesian User-Preference Modeling  
We assume each user has an underlying preference parameter $\theta\in\mathbb{R}^d$ governing utility over candidate actions $a\in\mathcal{A}$. We place a prior $p(\theta)=\mathcal{N}(\mu_0,\Sigma_0)$. For any pair of actions $(a_i,a_j)$ with feature embeddings $\phi(a)\in\mathbb{R}^d$, we model pairwise feedback $y\in\{0,1\}$ (“$a_i$ preferred to $a_j$”) via a logistic likelihood:  
$$
p(y=1\mid a_i,a_j,\theta)=\sigma\bigl(\theta^\top(\phi(a_i)-\phi(a_j))\bigr),
\quad\sigma(z)=\frac{1}{1+e^{-z}}.
$$  
Upon receiving feedback $y_t$ on query $(a_i^t,a_j^t)$, we update the posterior by Bayes’ rule:  
$$
p(\theta\mid \mathcal{D}_{t})\propto p(y_t\mid a_i^t,a_j^t,\theta)\,p(\theta\mid \mathcal{D}_{t-1}).
$$  
In practice we maintain a Laplace approximation $\mathcal{N}(\mu_t,\Sigma_t)$ via iterative Newton–Raphson.  

B. Uncertainty-Driven Query Selection  
At interaction step $t$, in context $s_t$, the AI proposes a candidate action set $\{a_1,\dots,a_K\}$. For each $a_k$, we compute the posterior predictive mean utility  
$$
\bar u(a_k)\;=\;\mathbb{E}_{\theta\sim\mathcal{N}(\mu_{t-1},\Sigma_{t-1})}\bigl[\theta^\top\phi(a_k)\bigr]
\;=\;\mu_{t-1}^\top\phi(a_k),
$$  
and variance  
$$
\sigma^2(a_k)\;=\;\phi(a_k)^\top\Sigma_{t-1}\,\phi(a_k).
$$  
We define value ambiguity 
$$
U_{\max} = \max_k \sigma(a_k).
$$  
If $U_{\max}>\tau_t$, where $\tau_t$ is an adaptive threshold (initially set by percentile of prior variances and decayed over time), the system issues an active query. We select the most informative comparison pair $(a_p,a_q)$ by maximizing expected information gain:  
$$
(a_p,a_q)
=\arg\max_{i<j}\;\mathbb{E}_{y}\bigl[\,
D_{KL}\bigl(p(\theta\mid\mathcal{D}_{t-1},(a_i,a_j,y))\;\|\;p(\theta\mid\mathcal{D}_{t-1})\bigr)\bigr].
$$  
We approximate this expectation via two-point Monte Carlo on $y\in\{0,1\}$.  

C. Bidirectional Interaction Protocol & UI Design  
Our co-adaptive loop proceeds:  
1. Observe context $s_t$ (e.g., user history in a recommendation task).  
2. Generate $K$ candidate actions $\{a_k\}$. Compute $\bar u(a_k)$ and $\sigma(a_k)$.  
3. If $\max_k\sigma(a_k)>\tau_t$, issue a query $(a_p,a_q)$ for user preference. Record feedback $y_t$ and update $(\mu_t,\Sigma_t)$.  
4. Choose action $\hat a_t=\arg\max_k \bar u(a_k)$ and present it to the user along with:  
   a. A confidence interval $[\bar u(\hat a_t)\pm c\cdot\sigma(\hat a_t)]$.  
   b. Two or three alternative suggestions ranked by $\bar u(a)$, each with a small icon depicting relative confidence.  
5. User either accepts or modifies the action; log that choice as implicit feedback and update posterior if appropriate.  
6. Adapt threshold $\tau_{t+1}=\alpha\,\tau_t + (1-\alpha)\,\tau_{\min}$ to gradually reduce query frequency.  

We will build a lightweight web UI prototype where uncertainty is visualized via error bars and colored glyphs. User feedback modalities include pairwise ranking, choose-one-out demonstrations, or counterfactual “What if?” scenarios.  

D. Experimental Evaluation  
We validate UDCA on two domains:  

1. Recommendation Benchmark  
– Dataset: MovieLens-1M with simulated users whose preference parameter $\theta^*$ is randomly sampled.  
– Baselines: Standard RLHF loop, passive Bayesian updating, rule-based static alignment.  
– Metrics:  
   • Preference Calibration Error:  
     $$\mathrm{PCE}=\frac{1}{N}\sum_{i=1}^N\bigl(\theta^{*\top}\phi(a_i)-\mu_T^\top\phi(a_i)\bigr)^2.$$  
   • Decision Quality: average true utility $\frac{1}{T}\sum_{t}\theta^{*\top}\phi(\hat a_t)$.  
   • Query Efficiency: number of queries until PCE $\le\epsilon$.  

2. Interactive Planning Task  
– Environment: Grid-world navigation with goal states. Users provide occasional steering preferences (e.g., avoid certain regions).  
– Real-user Study: N=30 participants recruited via crowd-platform.  
– Conditions: UDCA vs static alignment vs explain-only baseline.  
– Human-centered Metrics:  
   • User Trust: post-experiment trust scale (Likert).  
   • Mental Model Accuracy: questionnaire on AI’s decision logic.  
   • Perceived Workload: NASA-TLX.  

Ablation Studies  
– No uncertainty visualization  
– No bidirectional suggestions (AI-only updates)  
– Fixed vs adaptive threshold $\tau_t$  

Statistical Analysis  
We will use paired t-tests and ANOVA (p<0.05) to compare methods, and regression analysis to correlate uncertainty reduction with trust improvements.  

4. Expected Outcomes & Impact  
Expected Outcomes  
– A formal UDCA algorithm that integrates Bayesian preference modeling, active query selection, and bidirectional suggestion visualization.  
– A lightweight UI toolkit demonstrating co-adaptive interactions with uncertainty cues.  
– Empirical evidence that UDCA outperforms static or unidirectional alignment baselines in calibration error (reduction by ≥20%), decision quality (increase by ≥10%), and user trust (increase by ≥15% on Likert scale).  
– Open-source implementation and benchmark suite for future research.  

Impact  
Algorithmic Advancement: UDCA introduces a principled information-theoretic query strategy within a dynamic co-adaptation loop, bridging gaps identified in Deep Bayesian Active Learning (Melo et al., 2024), CoExBO (Adachi et al., 2023), and MAPLE (Mahmud et al., 2024).  
Human-Centered Design: By surfacing alternative action paths and uncertainty indicators, UDCA empowers users to form accurate mental models, fostering agency and trust—key goals of bidirectional alignment.  
Scalability & Generality: The framework applies to diverse domains (recommendation, planning, clinical decision support) with minimal domain-specific tuning, addressing the scalability challenge noted in “Negotiative Alignment” (Doo et al., 2025).  
Societal Benefit: Improved alignment processes can reduce negative side effects of AI deployment, support inclusive decision-making, and inform policy on transparent, user-empowered AI systems.  

5. References  
[1] Melo, L. C., Tigas, P., Abate, A., & Gal, Y. (2024). Deep Bayesian Active Learning for Preference Modeling in Large Language Models. arXiv:2406.10023.  
[2] Mahmud, S., Nakamura, M., & Zilberstein, S. (2024). MAPLE: A Framework for Active Preference Learning Guided by Large Language Models. arXiv:2412.07207.  
[3] Adachi, M., Planden, B., Howey, D. A., Osborne, M. A., et al. (2023). Looping in the Human: Collaborative and Explainable Bayesian Optimization. arXiv:2310.17273.  
[4] Wang, J., Wang, H., Sun, S., & Li, W. (2023). Aligning Language Models with Human Preferences via a Bayesian Approach. arXiv:2310.05782.  
[5] Luo, H., Zhou, Z., Shu, Z., et al. (2025). On the Interplay of Human-AI Alignment, Fairness, and Performance Trade-offs in Medical Imaging. arXiv:2505.10231.  
[6] “Beyond Preferences in AI Alignment” (2024). Unspecified authors.  
[7] Zhou, Z., Liu, J., Yang, C., et al. (2023). Beyond One-Preference-for-All: Multi-Objective Direct Preference Optimization.  
[8] Zhao, S., Dang, J., & Grover, A. (2023). Group Preference Optimization: Few-Shot Alignment of Large Language Models.  
[9] “Towards Bidirectional Human-AI Alignment: A Systematic Review” (2024). Unspecified authors.  
[10] Doo, F. X., Shah, N., Kulkarni, P., et al. (2025). Negotiative Alignment: An Interactive Approach to Human-AI Co-Adaptation for Clinical Applications.  
