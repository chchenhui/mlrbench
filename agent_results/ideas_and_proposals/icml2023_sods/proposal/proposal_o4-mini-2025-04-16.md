Title  
Graph Neural Surrogate–Driven GFlowNets for Efficient Black-Box Discrete Sampling

Introduction  
Background. Sampling and optimization in high-dimensional discrete spaces arise in numerous domains—combinatorial optimization, protein and molecular design, structured prediction, and conditional generation in large language models. Unlike continuous spaces, where gradient-based methods such as Langevin Monte Carlo or Hamiltonian Monte Carlo exploit smoothness, discrete spaces lack readily available gradients and often exhibit long-range, high-order correlations that render naive local proposals inefficient. Black-box objectives compound the difficulty: each function evaluation is costly or accessible only via an oracle (e.g., a deep language model or physics simulator).

Generative Flow Networks (GFlowNets) have emerged as a powerful framework for sampling from complex discrete distributions by amortizing sampling policies via a novel flow-matching or trajectory balance objective. However, existing GFlowNet methods typically assume direct access to oracles for reward evaluation and struggle when each evaluation incurs high cost. Surrogate-driven approaches promise query efficiency but have been underexplored in combination with GFlowNets.

Research Objectives. We propose to develop an iterative, active-learning framework—Graph Neural Surrogate–Driven GFlowNets (G²Flow)—that couples a trainable Graph Neural Network (GNN) surrogate model with a GFlowNet sampler to dramatically reduce black-box query complexity while maintaining high-quality, diverse samples. The key objectives are:  
1. Design a robust GNN surrogate that approximates the true discrete energy $E(x)$ and supplies pseudo-gradients for guiding proposals.  
2. Integrate the surrogate into a GFlowNet sampler, defining surrogate-based rewards $r_\theta(x)\!=\!\exp(-E_\theta(x))$ and combining these with importance-weighted corrections from occasional true evaluations.  
3. Develop an active-learning acquisition strategy that selects the most informative proposals—balancing high surrogate uncertainty and high potential reward—for true objective evaluation.  
4. Theoretically and empirically validate that G²Flow converges to the target distribution with substantially fewer oracle queries than baseline discrete samplers and surrogate-only methods.

Significance. By reducing the number of expensive evaluations by an order of magnitude or more, G²Flow makes practical a new class of applications in black-box discrete inference and optimization: constrained generation with large language models, protein sequence design under structural constraints, and combinatorial engineering problems where each simulation or wet-lab experiment is costly. The proposed active coupling of surrogate learning and flow-based sampling establishes a general paradigm for efficient exploration of complex discrete landscapes.

Methodology  
1. Problem Formulation  
We consider a discrete domain $\mathcal{X}$ (e.g., sequences, graphs, or combinatorial objects) and a black-box energy function $E:\mathcal{X}\to\mathbb{R}$. Our goal is to sample from the target Gibbs distribution  
$$
P(x)\;=\;\frac{\exp\bigl(-E(x)\bigr)}{Z},\quad Z=\sum_{x\in\mathcal{X}}\exp\bigl(-E(x)\bigr),
$$  
or to identify high-probability modes for optimization. Each evaluation of $E(x)$ is expensive, so we aim to minimize the number of true evaluations while approximating $P(x)$ closely.

2. Surrogate Model  
We parameterize a surrogate energy $E_\theta(x)$ via a Graph Neural Network (GNN). If $x$ is a sequence, we treat it as a path graph; for molecular graphs, we directly use the molecular graph. The surrogate is trained on a dataset $\mathcal{D}=\{(x_i,E(x_i))\}$ to minimize the mean-squared error:  
$$
\mathcal{L}_\mathrm{surr}(\theta)\;=\;\frac{1}{|\mathcal{D}|}\sum_{(x_i,y_i)\in\mathcal{D}}\bigl(E_\theta(x_i)-y_i\bigr)^2\;+\;\lambda\|\theta\|^2.
$$  
We maintain an ensemble of $K$ surrogate models $\{E_{\theta_k}\}_{k=1}^K$ to quantify epistemic uncertainty:  
$$
u(x)\;=\;\mathrm{Var}_{k}\bigl[E_{\theta_k}(x)\bigr].
$$

3. Generative Flow Network Sampler  
We define a Markov decision process (MDP) whose states $s\in\mathcal{S}$ represent partial constructions of $x$ and whose terminal states correspond to complete objects in $\mathcal{X}$. The action space $A(s)$ extends the partial object by one discrete step. A GFlowNet is trained to sample trajectories $\tau=(s_0\to s_1\to\cdots\to s_T=x)$ with probability proportional to a terminal reward. We use the Trajectory Balance (TB) objective:  
$$
\mathcal{L}_\mathrm{TB}(\phi)\;=\;\Bigl(\log F_\phi(s_0)\;+\!\sum_{t=0}^{T-1}\log P_{F,\phi}(s_{t+1}|s_t)\;-\;\log Z_\phi\;-\;\log r_\theta(x)\Bigr)^2,
$$  
where $P_{F,\phi}$ is the forward policy, $F_\phi(s)$ is the state flow, $Z_\phi$ is a learned normalizing constant, and the reward $r_\theta(x)=\exp(-E_\theta(x))$ is given by the surrogate.

4. Active Surrogate–GFlowNet Loop  
We propose the following iterative algorithm:

Algorithm G²Flow  
Input: Budget $B$ of true evaluations, initial seed set $\mathcal{D}$, ensemble size $K$, batch size $M$, acquisition sizes $m_u,m_r$  
1. Train ensemble $\{E_{\theta_k}\}$ on $\mathcal{D}$ by minimizing $\mathcal{L}_\mathrm{surr}$.  
2. Train GFlowNet parameters $\phi$ by minimizing $\mathcal{L}_\mathrm{TB}$ using surrogate rewards.  
3. For $i=1,\dots,M$: sample $x_i\sim\mathrm{GFlowNet}_\phi$.  
4. Compute uncertainty $u(x_i)$ and surrogate reward $r_\theta(x_i)$.  
5. Select $m_u$ points with highest $u(x_i)$ and $m_r$ points with highest surrogate reward. Let $S$ be this acquisition set.  
6. Evaluate $E(x)$ for $x\in S$ (true oracle calls).  
7. Augment dataset: $\mathcal{D}\leftarrow \mathcal{D}\cup\{(x,E(x)):\,x\in S\}$.  
8. Optionally apply importance weights for reward recalibration: for each new $x$, weight its contribution in $\mathcal{L}_\mathrm{TB}$ by  
$$
w(x)\;=\;\exp\bigl(-E(x)\bigr)\;\big/\;\exp\bigl(-E_\theta(x)\bigr).
$$  
9. Update budget $B\leftarrow B-|S|$. If $B>0$, return to Step 1; else terminate.

5. Experimental Design  
Benchmarks  
1. Synthetic energy landscapes. High-order Ising models on lattices with known partition functions to measure sampling accuracy via KL divergence.  
2. Constrained language model sampling. Given a pre-trained GPT-2 with hidden Bayesian posterior constraints (e.g., must contain keywords), sample sentences under oracle evaluation of constraint satisfaction.  
3. Protein sequence design. Use a differentiable proxy for folding stability (e.g., a fast structural predictor) as $E(x)$ and evaluate on an expensive physics-based simulator for true target.  

Baselines  
• Discrete MCMC: Gibbs, Metropolis–Hastings with random local proposals.  
• Gradient-based discrete MCMC: discrete Langevin analogues.  
• Embedding methods: continuous relaxations + rounding (e.g., Gumbel-Softmax).  
• Standard GFlowNet without surrogate.  
• Stein variational methods in discrete space.  
• Deep Bayesian optimization with Gaussian process surrogates.  

Metrics  
• Query efficiency: number of true $E(x)$ evaluations vs. average sample quality $\mathbb{E}[f(x)]$ or negative energy $-E(x)$.  
• Distributional accuracy: KL$(\widehat P\|P)$ where $\widehat P$ is the empirical distribution from samples.  
• Diversity: average pairwise Hamming distance (or graph edit distance)  
$$
\bar D\;=\;\frac{2}{N(N-1)}\sum_{i<j}d(x_i,x_j).
$$  
• Surrogate accuracy: RMSE$_\mathrm{surr}=\sqrt{\frac{1}{|\mathcal{D}|}\sum_i (E_\theta(x_i)-E(x_i))^2}$.  
• Computation time: wall-clock time per oracle evaluation saved.  

Evaluation Protocol  
We vary total oracle budget $B$ and report performance curves (reward vs. evaluations). Ablations include: ensemble size $K$, acquisition split $(m_u,m_r)$, frequency of surrogate retraining, and reward recalibration on/off.

Expected Outcomes & Impact  
We expect that G²Flow will:  
1. Achieve comparable or better sampling quality than baseline GFlowNet and MCMC methods using 5–10× fewer oracle evaluations.  
2. Maintain high sample diversity, avoiding mode collapse by leveraging the flow objective and active‐uncertainty acquisitions.  
3. Demonstrate robust surrogate accuracy growth over iterations, with uncertainty decreasing in visited regions and guiding exploration elsewhere.  
4. Show generality across discrete domains—synthetic, language, and protein design tasks—validating the broad applicability of our framework.

Scientific Impact. This work bridges surrogate modeling and generative flow networks, establishing the first active-learning GFlowNet framework for black-box discrete sampling. It provides a new algorithmic paradigm for efficient exploration of complex combinatorial spaces with costly evaluation functions.

Practical Applications.  
• Constrained text generation: enabling real-time, constraint-satisfying decoding from large language models under strict query budgets.  
• Protein and molecular engineering: accelerating in silico screening in drug discovery and enzyme design by reducing expensive simulator calls.  
• Combinatorial optimization in hardware and compiler design: rapidly exploring device placements or instruction schedules with fewer simulator runs.

Future Directions. We will explore extensions to continuous–discrete hybrid spaces, integration with reinforcement-learning simulators, and theoretical analysis of convergence guarantees under surrogate bias. By releasing our code and benchmarks, we aim to foster a community around efficient black-box discrete sampling.

In summary, Graph Neural Surrogate–Driven GFlowNets (G²Flow) promise a leap in query‐efficient discrete sampling and optimization, with broad implications for machine learning, scientific discovery, and engineering design.