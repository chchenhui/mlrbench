Title  
Causality-Aware World Models via Counterfactual Latent State Prediction  

1. Introduction  
Background  
World models—learned simulators of an agent’s environment—have been highly successful at predicting sequences of observations and supporting planning (e.g. Ha and Schmidhuber, 2018; Hafner et al., 2019). Classical approaches rely on recurrent neural networks (RNNs), Transformers or state-space models (SSMs) to capture transition dynamics in latent space. Yet, such models often learn purely correlational structure: they predict what usually happens next, but not what would happen under novel interventions. Without causal understanding, agents may fail when deployed in settings that differ from training or when asked to reason about “what if” scenarios—key requirements in robotics, healthcare, and safety-critical applications.  

Motivation & Significance  
We propose to endow world models with causal reasoning by explicitly training them to predict counterfactual latent states under hypothetical interventions. By combining sequence prediction with intervention-driven trajectory perturbations, our model will learn latent representations that capture cause-effect relationships. This will (1) improve zero-shot generalization to unseen interventions, (2) yield more robust planning under distribution shift, and (3) produce interpretable latent factors aligned with causal mechanisms. Such capabilities are crucial for reliable decision-making in embodied AI, healthcare simulations, and adaptive control.  

Research Objectives  
1. Develop an architecture that jointly predicts factual and counterfactual latent state trajectories under sampled interventions.  
2. Formulate training objectives that align latent space with causal factors by penalizing counterfactual prediction error.  
3. Design benchmarks and evaluation metrics to quantify zero-shot intervention generalization, causal representation quality, and downstream planning performance.  

2. Methodology  
We outline (A) data collection and intervention protocol, (B) model architecture and learning objectives, and (C) experimental design for evaluation.  

2.1 Data Collection & Intervention Protocol  
We will build on simulated environments with known causal structures:  

• Physical simulation (e.g. MuJoCo block stacking, OpenAI Gym CartPole, SSV2).  
• Grid-world navigation with movable objects.  
• Simulated healthcare scenarios (e.g. patient vital signs calibrated from MIMIC-III).  

For each environment we generate:  
1. Factual trajectories: sequences of observations $o_{1:T}$ and actions $a_{1:T-1}$.  
2. Intervention trajectories: at random time steps $t_i$, replace action $a_{t_i}$ with a perturbed action $\tilde a_{t_i} = a_{t_i} + \delta_i$, or alter initial state $s_1$. We record the resulting observation sequence $\tilde o_{t_i+1:t_i+H}$.  

Dataset Construction  
• N₁ = 50 000 factual trajectories of length T=50 per environment.  
• N₂ = 20 000 intervention events, each with horizon H=10.  
• Each intervention record: $(o_{1:t_i},\,\tilde a_{t_i},\,\tilde o_{t_i+1:t_i+H})$.  

2.2 Model Architecture  
Our model extends a variational state-space framework with dual decoders for factual and counterfactual predictions.  

2.2.1 Latent Transition Model  
We denote by $z_t\in\mathbb{R}^d$ the latent state at time $t$ and by $a_t$ the action. We posit a probabilistic transition:  
$$
q_\phi(z_t \mid z_{t-1}, a_{t-1}) = \mathcal{N}\bigl(\mu_\phi(z_{t-1},a_{t-1}),\,\Sigma_\phi(z_{t-1},a_{t-1})\bigr).
$$  
An RNN, Transformer encoder, or continuous-time SSM implements $\mu_\phi,\Sigma_\phi$.  

2.2.2 Observation Decoder  
We reconstruct observations via  
$$
p_\theta(o_t\mid z_t)=\mathrm{Bernoulli}(f_\theta(z_t))\quad\text{or}\quad\mathcal{N}\bigl(g_\theta(z_t),\,\sigma^2 I\bigr),
$$  
depending on the modality (discrete pixels or continuous sensor readings).  

2.2.3 Intervention Encoder & Counterfactual Decoder  
To model interventions, we embed the perturbation signal $(\delta_i,a_{t_i})$ into a vector $e_i\in\mathbb{R}^{d_e}$ via a small MLP. At each intervention time $t_i$ we branch off a counterfactual latent trajectory $\{z^{\mathrm{cf}}_{t}\}_{t=t_i+1}^{t_i+H}$ defined by:  
```
for τ = t_i+1 … t_i+H:  
    z^{cf}_τ ~ q_\phi(z | z^{cf}_{τ-1},\,\tilde a_{τ-1},\,e_i)  
```
where $\tilde a_{τ-1}=a_{τ-1}$ for τ>t_i+1, and $\tilde a_{t_i}=a_{t_i} + \delta_i$. The above can be implemented by augmenting the transition network to accept $e_i$ after $t_i$.  

2.2.4 Training Objective  
Our loss combines three terms:  
1. Factual ELBO ($\mathcal{L}_{\mathrm{fact}}$):  
$$
\mathcal{L}_{\mathrm{fact}} = \sum_{t=1}^T \Bigl[\mathbb{E}_{q_\phi(z_t\mid z_{t-1},a_{t-1})}\bigl[-\log p_\theta(o_t\mid z_t)\bigr] + \mathrm{KL}\bigl(q_\phi(z_t\mid z_{t-1},a_{t-1}) \parallel p(z_t)\bigr)\Bigr].
$$  
2. Counterfactual prediction loss ($\mathcal{L}_{\mathrm{cf}}$):  
Measure discrepancy between predicted counterfactual latents $z^{\mathrm{cf}}_{t_i+1:t_i+H}$ and “ground-truth” latents inferred from the interventional trajectory:  
$$
\mathcal{L}_{\mathrm{cf}} = \sum_{i=1}^{N_2}\sum_{τ=t_i+1}^{t_i+H} \bigl\| \mu_\phi(z^{\mathrm{cf}}_{τ}\mid z^{\mathrm{cf}}_{τ-1},\tilde a_{τ-1},e_i) - \hat z_{τ}\bigr\|_2^2,
$$  
where $\hat z_{τ}$ is the encoder’s posterior mean from the observed interventional frame $\tilde o_{τ}$.  
3. Regularizer for causal disentanglement ($\mathcal{L}_{\mathrm{reg}}$):  
Encourage $z_t$ to factorize into causal components by penalizing total correlation (Chen et al., 2018):  
$$
\mathcal{L}_{\mathrm{reg}} = \mathrm{TC}(z_{1:T}) \approx \sum_{j=1}^d \mathrm{KL}\bigl(q(z^j_{1:T})\|\,q(z^j_{1:T}\mid z^{-j}_{1:T})\bigr).
$$  

Overall loss:  
$$
\mathcal{L} = \mathcal{L}_{\mathrm{fact}} + \lambda_{\mathrm{cf}}\,\mathcal{L}_{\mathrm{cf}} + \lambda_{\mathrm{reg}}\,\mathcal{L}_{\mathrm{reg}},
$$  
with tunable weights $\lambda_{\mathrm{cf}},\lambda_{\mathrm{reg}}>0$.  

2.2.5 Algorithmic Steps  
1. Initialize model parameters $(\phi,\theta)$.  
2. For each minibatch:  
   a. Sample B factual sequences; compute $\mathcal{L}_{\mathrm{fact}}$.  
   b. Sample B₁ intervention events; compute $\mathcal{L}_{\mathrm{cf}}$ using encoder posteriors on interventional observations.  
   c. Compute $\mathcal{L}_{\mathrm{reg}}$ on latent encodings.  
   d. Backpropagate $\nabla_\phi,\nabla_\theta\,\mathcal{L}$ and update via Adam.  
3. Repeat until convergence.  

2.3 Experimental Design & Evaluation Metrics  
Benchmarks & Baselines  
• Standard world model (without counterfactual head).  
• Causal Transformer (Melnychuk et al., 2022).  
• Diffusion-based causal model (Chao et al., 2023).  

Evaluation Protocol  
1. Factual prediction error: RMSE and negative log-likelihood on held-out factual trajectories.  
2. Counterfactual generalization: sample novel interventions (actions/displacements not seen during training) and measure latent MSE and observation reconstruction error over horizon H.  
3. Average Treatment Effect (ATE) estimation: compare predicted and true mean shift in a target variable under intervention.  
4. Latent causal alignment: measure correlation between individual latent dimensions and known ground-truth causal factors using mutual information estimation (e.g. InfoGAN metric).  
5. Planning performance: embed model in a model-based RL loop (e.g. Dreamer agent); compare cumulative reward under distributional shift.  

3. Expected Outcomes & Impact  
We anticipate that integrating counterfactual latent state prediction will yield:  
• Improved zero-shot generalization: lower counterfactual prediction error on novel interventions compared to baselines.  
• More disentangled, causally aligned latent factors, facilitating interpretability and model auditing.  
• Enhanced planning robustness: model-based agents using our causal world model will achieve higher rewards under shifted dynamics or in “what-if” tasks.  
• Broader applicability in domains requiring safe intervention reasoning, such as adaptive healthcare planning or robotic manipulation under uncertainty.  

Impact  
This work advances the theory and practice of world models by bridging the gap between correlational sequence modeling and causal reasoning. By providing an end-to-end framework for learning causally informed latent dynamics, our approach will:  
1. Enable agents to predict the outcome of unseen interventions, critical for safe decision-making.  
2. Offer a blueprint for integrating counterfactual reasoning into large-scale generative world models (video/text).  
3. Stimulate new benchmarks for causal world-model evaluation across language, vision, and control.  

In summary, our proposal aims to deliver a principled, scalable method for causality-aware world modelling, with rigorous empirical validation and potential for high-impact applications in embodied AI, healthcare, and beyond.