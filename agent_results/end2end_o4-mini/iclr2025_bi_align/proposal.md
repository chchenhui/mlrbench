1. Title  
Uncertainty-Driven Reciprocal Alignment: A Bidirectional Framework for Dynamic Human–AI Collaboration  

2. Introduction  
Background  
As AI systems assume increasingly complex, safety-critical decision-making roles, classical one-way alignment methods—where AI purely adapts to a static model of human preferences—prove inadequate. Humans’ goals, risk tolerances, and ethical considerations evolve in response to context and AI behavior; likewise, AI uncertainty (both model- and data-driven) directly impacts human trust and willingness to collaborate. The emerging paradigm of bidirectional human-AI alignment (Pyae, 2025) emphasizes a continuous, mutual feedback loop: AI systems not only learn from human corrections but also expose their own uncertainties to guide human updates of their mental models.  

Research Objectives  
• Design an interactive alignment framework that (a) detects and communicates AI uncertainty at decision points and (b) incorporates real-time human feedback to update both the AI policy and its model of human preferences.  
• Formulate a multi-objective reinforcement learning (MORL) algorithm that optimizes task performance while minimizing misalignment, calibrated by a Bayesian user model.  
• Implement a lightweight HCI interface for visualizing outcome predictions and confidence intervals, enabling users to correct AI actions or adjust preference weights on the fly.  
• Empirically validate the approach on simulated decision-making tasks via controlled user studies, measuring alignment error, task efficiency, and trust calibration.  

Significance  
By uniting uncertainty quantification, Bayesian preference modeling, and interactive design, we pioneer a truly reciprocal alignment loop. This advances AI-centered alignment by improving confidence calibration (Li et al., 2024; Papantonis & Belle, 2023) and supports human-centered alignment by preserving agency through clear, targeted feedback opportunities. The framework promises safer deployment in healthcare triage, autonomous vehicles, and resource allocation, where stakes are high and human oversight is paramount.  

3. Related Work  
1. The Human-AI Handshake Framework (Pyae, 2025)  
   • Introduces mutual learning, validation, and feedback.  
   • Lacks a concrete uncertainty-driven mechanism for surfacing AI confidence to users.  
2. Complementing explanations with uncertainty (Papantonis & Belle, 2023)  
   • Demonstrates that pairing explanations with calibrated uncertainty improves trust.  
   • Does not address dynamic preference updates or reciprocal policy adaptation.  
3. Overconfident and Underconfident AI Hinder Collaboration (Li et al., 2024)  
   • Shows miscalibration leads to misuse/disuse.  
   • Calls for alignment methods that jointly calibrate confidence and update policies.  
4. Human Uncertainty in Concept-Based Systems (Collins et al., 2023)  
   • Provides datasets/tools for human feedback with inherent uncertainty.  
   • Focused on static concept systems rather than online, bidirectional loops.  
5. AI Alignment (Wikipedia, 2025)  
   • Surveys alignment challenges and possibilities.  
   • Highlights the gap in dynamic, bidirectional processes.  

Gap Analysis  
Existing work underscores the importance of uncertainty calibration and user agency but stops short of integrating:  
(a) real-time surfacing of AI uncertainty,  
(b) Bayesian inference over evolving human preferences, and  
(c) multi-objective policy updates that explicitly trade off performance and alignment divergence.  

4. Methodology  
We propose Uncertainty-Driven Reciprocal Alignment (UDRA), comprised of the following components:  

4.1. Problem Setting  
Let $\mathcal S$ be the state space, $\mathcal A$ the action space, and $w\in\mathbb R^k$ a latent human preference vector. At each timestep $t$, the AI observes $s_t$, selects $a_t\sim\pi_\theta(a\mid s,w)$, and incurs a task reward $r_{\rm task}(s_t,a_t)$. The human holds an internal utility $u_w(a_t,s_t)=w^\top\phi(s_t,a_t)$ for feature map $\phi$. We define an alignment loss  
$$
\mathcal L_{\rm align}(s_t,a_t,w)\;=\;\lVert w - \hat w_t\rVert_2^2\,,  
$$  
where $\hat w_t$ is the AI’s current estimate of user preferences.  

4.2. Bayesian User Modeling  
We maintain a posterior over $w$ given correction data $\mathcal D_t=\{(s_i,a_i^{\rm cor})\}_{i=1}^t$. Assuming a Gaussian prior $w\sim\mathcal N(\mu_0,\Sigma_0)$ and linear feedback likelihood  
$$
p(a_i^{\rm cor}\mid s_i,w)\;\propto\;\exp\bigl(w^\top\phi(s_i,a_i^{\rm cor})\bigr)\,,
$$  
the posterior is updated via  
$$
p(w\mid\mathcal D_t)\;\propto\;p(w\mid\mathcal D_{t-1})\;\prod_{i=1}^t p(a_i^{\rm cor}\mid s_i,w)\,.
$$  
Closed-form updates for $\mu_t,\Sigma_t$ are available under Laplace approximations or assumed density filtering.  

4.3. Multi-Objective Reinforcement Learning  
We define a scalarized reward at time $t$:  
$$
r_t(\theta,w) \;=\; r_{\mathrm{task}}(s_t,a_t)\;-\;\lambda\,\lVert w - \hat w_t\rVert_2^2\;,
$$  
where $\lambda>0$ balances task performance against alignment loss. The policy $\pi_\theta$ is trained to maximize the expected discounted return  
$$
J(\theta)\;=\; \mathbb{E}_{\pi_\theta}\Bigl[\sum_{t=0}^T\gamma^t\,r_t(\theta,w)\Bigr]\,,
$$  
via policy gradients or entropy-regularized actor–critic.  

4.4. Uncertainty Estimation & Visualization  
At each decision step, we compute:  
• Task outcome distribution: predict next-state outcomes using an ensemble of dynamics or a dropout-based model.  
• Confidence interval on $Q$-values: derive $\sigma_Q(s_t,a)$ from ensemble variance.  
We present a concise dashboard:  
• Bar plots of top-k action utilities with shaded confidence bands.  
• Probability mass functions over salient features (e.g.\ cost, time).  
This exposes AI’s epistemic uncertainty to the user.  

4.5. Interactive HCI Interface  
Users interact via a web interface that:  
1. Highlights actions with high uncertainty or high misalignment risk.  
2. Allows users to:  
   a. Select a corrective action $a_t^{\rm cor}$, or  
   b. Adjust preference weights in $\hat w_t$ via sliders.  
3. Provides just‐in‐time explanations for why the AI favored certain actions (contrastive “why‐this‐not‐that” queries).  

4.6. Policy and Preference Updates  
Upon receiving human input at time $t$, we:  
1. Update the Bayesian posterior $p(w\mid\mathcal D_t)\to (\mu_t,\Sigma_t)$.  
2. Recompute the alignment term $\lVert w - \hat w_t\rVert_2^2$ in the reward.  
3. Perform a policy gradient step:  
   $$  
   \theta_{t+1}\;=\;\theta_t \;+\;\alpha\,\widehat\nabla_\theta J(\theta_t)\,,  
   $$  
   where the gradient is estimated via REINFORCE or an actor–critic critic $V_\psi$.  

Algorithm 1 (Uncertainty-Driven Reciprocal Alignment)  
1. Initialize $\theta_0,\mu_0,\Sigma_0$.  
2. For each episode:  
   a. For $t=0\ldots T$:  
      i. Observe $s_t$.  
      ii. Compute $Q$-estimate and uncertainty $\sigma_Q(s_t,a)$.  
      iii. Display top actions & confidences.  
      iv. User provides $a_t^{\rm cor}$ or weight adjustment.  
      v. Update posterior $(\mu_t,\Sigma_t)\leftarrow p(w\mid\mathcal D_t)$.  
      vi. Compute $r_t = r_{\rm task} - \lambda\lVert w - \mu_t\rVert^2$.  
      vii. Update $\theta$ via policy gradient.  
3. Return final policy $\pi_{\theta^*}$.  

4.7. Experimental Design  
Environments  
• Simulated resource allocation tasks (e.g.\ supply chain routing).  
• Safety-critical mock scenarios (e.g.\ autonomous driving intersections).  
Participants  
• 30–40 lay users in a within-subjects design. Each user experiences:  
  – Baseline: standard RL agent with static alignment (RLHF).  
  – UDRA agent.  
Evaluation Metrics  
1. Alignment Error: $\frac1T\sum_t\lVert a_t^{\rm ai}-a_t^{\rm cor}\rVert_2$.  
2. Task Efficiency: average cumulative task reward per episode.  
3. Trust Calibration: Spearman’s $\rho$ between AI confidence and observed accuracy.  
4. User Satisfaction: post-task questionnaire (Likert scales on control, transparency, trust).  

Data Analysis  
Use paired $t$-tests and repeated-measures ANOVA to compare conditions; confidence intervals via bootstrapping.  

5. Expected Outcomes & Impact  
Anticipated Results  
• Alignment error under UDRA will decrease by at least 20% relative to baseline.  
• Task efficiency will be maintained or improved, demonstrating that alignment need not trade off performance.  
• Trust calibration (Spearman’s $\rho$) will increase significantly, indicating better user confidence in AI cues.  
• User satisfaction scores on agency and transparency will outperform static-alignment controls (p<0.05).  

Theoretical Contributions  
• A formal MORL formulation that integrates Bayesian user modeling into the reward function.  
• Proof-of-concept algorithm showing convergence properties under mild assumptions on posterior update stability.  

Practical Impact  
• A reusable software toolkit for real-time uncertainty visualization and feedback collection.  
• Guidelines for HCI practitioners on designing minimal, actionable explanation dashboards.  
• Applicability to domains requiring tight human oversight (e.g.\ medical decision support, autonomous vehicles, disaster response).  

Broader Societal Benefits  
By embedding reciprocity and uncertainty awareness into alignment, UDRA fosters more transparent, trustworthy AI. It preserves human agency, mitigates risks from miscalibrated models, and lays groundwork for ethically aligned, collaborative AI systems in society at large.