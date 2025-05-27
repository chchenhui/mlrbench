Title: Diffusion-Based Inference-Time Alignment for Large Language Models via Target Density Sampling

1. Introduction  
Large language models (LLMs) such as GPT and PaLM have demonstrated impressive fluency and adaptability, yet aligning their generation to downstream objectives (e.g. safety, factuality, style) typically requires expensive fine-tuning or reinforcement-learning-based pipelines (RLHF). These methods suffer from high computation cost, instability, and over-optimization (reward hacking). An emerging alternative is inference-time alignment: steering a pre-trained model’s output on the fly by sampling from a target density $$\pi(x)\propto p_\theta(x)\exp(\lambda\,r(x))$$ without updating model weights. Here $p_\theta(x)$ is the base LLM’s sequence distribution, $r(x)$ a task-specific reward function, and $\lambda>0$ a temperature. 

Diffusion and score-based methods have recently reshaped probabilistic inference by framing sampling as solving a stochastic differential equation or a discrete Markov chain guided by learned score networks. Classifier-guidance and MCMC methods (e.g. Langevin dynamics) can inject external reward gradients into the diffusion process at inference time. However, most prior work has focused on image diffusion or sentence-level adjustments; token-level diffusion in LLMs remains under-explored.

This proposal introduces a novel inference-time alignment algorithm—Diffusion-Based Inference-Time Alignment (DIA)—that (1) learns a token-level diffusion sampler to approximate $p_\theta(x)$ via continuous embeddings, (2) incorporates reward guidance akin to Langevin/MCMC to sample from $\pi(x)$, and (3) uses a lightweight reward-aware proposal to accelerate mixing. Our objectives are:  
• To design and implement a diffusion-inspired generative sampler for discrete text guided by arbitrary differentiable (or surrogate) reward functions.  
• To derive theoretical guarantees of convergence to the target density under mild conditions.  
• To empirically evaluate DIA on safety filtering, style transfer, and factuality tasks, comparing against baselines such as RLHF, classifier guidance, and SMC methods.

By enabling real-time, sample-efficient alignment without retraining, DIA promises to dramatically reduce deployment costs, improve model safety and controllability, and open new applications for user-defined, context-aware LLM customization.

2. Methodology  

2.1 Problem Formulation  
Let $x=(x_1,\dots,x_T)$ denote a token sequence. The base LLM defines $$p_\theta(x)=\prod_{t=1}^T p_\theta(x_t\mid x_{<t}).$$ We define a target density  
$$\pi(x)\;\propto\;p_\theta(x)\,\exp\bigl(\lambda\,r(x)\bigr),$$  
where $r(x)\in\mathbb{R}$ is a reward function (e.g. safety score, style classifier log-probability). Our goal is to draw approximate samples from $\pi(x)$ at inference time, using only black-box access to $p_\theta$ and a differentiable or surrogate differentiable approximation $\hat r(x)$.

2.2 Continuous Embedding and Diffusion Model  
Because diffusion samplers operate in continuous space, we embed tokens via a fixed, pre-trained embedding matrix $E:\{1,\dots,V\}\to\mathbb{R}^d$. Let $$z_0 = E(x) = [E(x_1),\dots,E(x_T)]\in\mathbb{R}^{T\times d}.$$ We train a denoising diffusion probabilistic model (DDPM) to approximate $p_\theta(x)$ by modeling the marginal $p(z_0)$. Following standard discrete diffusion approaches, we define a forward noising process for $t=1,\dots,T_{\max}$:  
$$z_t = \alpha_t z_{t-1} + \sigma_t\,\epsilon_t,\quad \epsilon_t\sim\mathcal{N}(0,I),$$  
where schedules $\{\alpha_t,\sigma_t\}$ are learned or set to ensure well-conditioned transitions. A neural network $\epsilon_\psi(z_t,t)$ is trained to predict the added noise by minimizing  
$$\mathcal{L}_{\rm DDPM} = \mathbb{E}_{z_0,t,\epsilon}\bigl\|\,\epsilon - \epsilon_\psi(\alpha_t z_0 + \sigma_t\epsilon,\;t)\bigr\|^2.$$

At inference, the reverse diffusion dynamics without reward would sample $$z_{t-1} = \frac{1}{\alpha_t}\Bigl(z_t - \sigma_t\,\epsilon_\psi(z_t,t)\Bigr) + \tilde\sigma_t\,\tilde\epsilon_t.$$

2.3 Reward-Guided Reverse Process  
To steer toward high-reward regions, we adapt Langevin dynamics in latent space. Define the continuous target density on $z$:  
$$\pi_z(z_0)\;\propto\;\underbrace{p_{\rm DDPM}(z_0)}_{\approx\,p_\theta(x)}\;\exp\!\bigl(\lambda\,\hat r(G(z_0))\bigr),$$  
where $G(z)$ maps a continuous latent back to a discrete token sequence via $$G_i(z)=\arg\max_{v\in V}\langle E(v),\,z_i\rangle.$$ Under score-based theory, the score is  
$$\nabla_{z_t}\log\pi_z(z_t) = \underbrace{\nabla_{z_t}\log p_{\rm DDPM}(z_t)}_{=\,s_\psi(z_t,t)}\;+\;\lambda\,\nabla_{z_t}\hat r\bigl(G(z_t)\bigr).$$  
We approximate $s_\psi(z_t,t)= -\tfrac{1}{\sigma_t}\,\epsilon_\psi(z_t,t)$ (standard score estimate). The full guided reverse-diffusion update with step size $\eta_t$ is:  
$$
\begin{aligned}  
\mu_t &= \frac{1}{\alpha_t}\Bigl(z_t - \sigma_t\,\epsilon_\psi(z_t,t)\Bigr),\\  
z_{t-1} &= \mu_t \;+\;\eta_t\,\Sigma_t^2\,\underbrace{\bigl[-\tfrac{1}{\sigma_t}\,\epsilon_\psi(z_t,t)+\lambda\,\nabla_{z_t}\hat r(G(z_t))\bigr]}_{ \approx \nabla_{z_t}\log\pi_z(z_t)} \;+\;\Sigma_t\,\sqrt{2\eta_t}\,\zeta_t,
\end{aligned}
$$  
where $\Sigma_t^2=1-\alpha_t^2$ and $\zeta_t\sim\mathcal{N}(0,I)$. This scheme blends denoising diffusion with ULA-style reward gradients.

2.4 Reward Surrogate and Gradient Estimation  
Many practical rewards (toxicity classifiers, style models) may be non-differentiable w.r.t. embeddings. We train a lightweight differentiable critic $h_\phi(z)$ to regress $\hat r(G(z))$ on samples from $p_\theta$, minimizing  
$$\mathcal{L}_{\rm critic}=\mathbb{E}_{x\sim p_\theta}\bigl(\,h_\phi(E(x))-r(x)\bigr)^2.$$  
At inference we use $\nabla_z h_\phi(z)$ as a surrogate for $\nabla_z\hat r(G(z))$.

2.5 Algorithm Summary  
Algorithm: Inference-Time Diffusion Alignment (DIA)  
Input: base model $p_\theta$, reward critic $h_\phi$, diffusion net $\epsilon_\psi$, schedules $\{\alpha_t,\sigma_t,\eta_t\}$, temperature $\lambda$.  
1. Sample $z_{T_{\max}}\sim\mathcal{N}(0,I)$  
2. For $t=T_{\max}$ down to $1$:  
   a. Compute denoised mean: $\mu_t=\frac{1}{\alpha_t}(z_t-\sigma_t\epsilon_\psi(z_t,t))$  
   b. Compute guidance term: $g_t=-\tfrac{1}{\sigma_t}\epsilon_\psi(z_t,t)+\lambda\,\nabla_{z_t}h_\phi(z_t)$  
   c. Draw noise $\zeta_t\sim\mathcal{N}(0,I)$  
   d. Update  
       $$z_{t-1} = \mu_t + \eta_t\,\Sigma_t^2\,g_t + \Sigma_t\sqrt{2\eta_t}\,\zeta_t$$  
3. Decode $x=G(z_0)$

2.6 Theoretical Analysis  
Under regularity of $\epsilon_\psi$ and Lipschitz continuity of $h_\phi$, the above guided reverse process is an instance of the Unadjusted Langevin Algorithm on the latent target $\pi_z$. Provided step sizes $\eta_t$ are sufficiently small and $\sum_t\eta_t=\infty,\sum_t\eta_t^2<\infty$, standard results (e.g. Raginsky et al. 2017) guarantee convergence in Wasserstein distance to $\pi_z$. Discretization bias can be reduced by Metropolis-Hastings correction if desired.

2.7 Experimental Design  
Datasets & Tasks:  
• Safety alignment on OpenAI WebText‐like language generation with a pretrained GPT-2 small. Reward: Perspective API toxicity score.  
• Style transfer: formal ↔ informal on GYAFC benchmark, reward from a pretrained style classifier.  
• Factuality improvement in summarization on CNN/DailyMail: reward from an entailment model (RoBERTa).  

Baselines:  
• Zero-shot sampling from $p_\theta$  
• Classifier guidance (Ho & Salimans 2022)  
• SMC-based inference alignment (Sunwoo et al. 2025)  
• RLHF (PPO) fine-tuning  

Metrics:  
• Reward value $\mathbb{E}[r(x)]$  
• Language quality: BLEU, ROUGE, perplexity, human preference via crowdsourcing  
• Diversity: self-BLEU, distinct-n  
• Inference latency (seconds per 100 tokens)  
• Failure modes: reward hacking frequency  

Ablations:  
• Varying $\lambda$ to control trade-off  
• Effect of number of diffusion steps $T_{\max}$ and step sizes $\eta_t$  
• With vs. without critic correction (i.e. true $\nabla\hat r$ vs. surrogate)  
• MH correction vs. ULA only  

3. Expected Outcomes & Impact  
We anticipate that DIA will:  
1. Achieve alignment quality (reward) on par with or exceeding RLHF-trained models, while preserving fluency and diversity.  
2. Reduce end-to-end compute cost by 50–80% compared to PPO/Fine-tuning, since no gradient through the LLM is needed.  
3. Offer stable, controllable inference with fewer reward-hacking failures due to the smoothing effect of diffusion and the explicit exploration–exploitation trade-off in Langevin sampling.  
4. Demonstrate broad applicability across safety, style, and factuality tasks without per-task fine-tuning.  

Broader Impacts:  
• Systems adopting DIA can support on-demand, user-specific constraints (e.g. personal style, domain lexicons) without retraining.  
• The framework unifies sampling, diffusion, and reinforcement perspectives, advancing the theoretical understanding of inference-time alignment.  
• By open-sourcing code, pretrained critics, and diffusion schedules, we contribute reusable benchmarks and tools to the community.  

4. Timeline & Milestones  
(1) Month 1–2: Prepare data pipelines; implement base DDPM on GPT-2 embeddings.  
(2) Month 3–4: Train reward critic networks; integrate reward guidance into reverse diffusion.  
(3) Month 5–6: Conduct experiments on safety and style benchmarks; compare to baselines.  
(4) Month 7–8: Extend to factuality summarization; perform ablations and MH corrections.  
(5) Month 9: Analyze results, prepare code release, write paper for FPI workshop.  

This proposal addresses core FPI themes: it bridges diffusion-based samplers and sampling from target densities, leverages optimal transport/control views of guided sampling, and applies to LLM fine-tuning/inference-time alignment in natural language. We believe the DIA framework will stimulate further research on scalable, learning-based sampling methods for probabilistic inference across domains.