1. Title  
Optimal Data Epochs in LLM Pretraining: A Theoretical and Empirical Framework for Balancing Efficiency and Representation Quality  

2. Introduction  

Background  
Large language models (LLMs) such as GPT, BERT, and their successors have revolutionized natural language processing by learning rich representations from massive text corpora. Pretraining these models typically involves processing billions to trillions of tokens, often by repeating (“recycling”) the same data for multiple epochs to improve convergence when the raw data supply is insufficient. In practice, data recycling is chosen by heuristic: one to three epochs is common, but there is little theoretical understanding of how repeated passes over the same data affect optimization dynamics, generalization, and representation quality. Trial‐and‐error tuning of epochs at the trillion‐token scale can cost millions of GPU‐hours.  

Research Objectives  
This proposal aims to develop a principled framework for choosing the optimal number of data epochs, $E$, in LLM pretraining. Our three concrete objectives are:  
  • Theoretical Modeling: Build a stochastic‐optimization model that captures the effect of data repetition on gradient variance, gradient autocorrelation across epochs, and convergence rate.  
  • Convergence and Generalization Bounds: Derive upper bounds on expected gradient norm and generalization error as functions of $E$, dataset size $N$, model scale, and gradient‐noise correlation coefficient $\rho$. Identify an effective sample size $N_{\mathrm{eff}}(E)$ that quantifies the diminishing returns of repeated epochs.  
  • Empirical Validation and Heuristics: Validate theoretical predictions through controlled pretraining experiments on Transformer models of varying scale. From theory and empirical results, extract simple heuristics for choosing $E$ given available compute, data diversity, and desired representation quality.  

Significance  
A theoretical understanding of data recycling in LLM pretraining will  
  1. Reduce computational cost by avoiding unnecessary epochs, saving time, energy, and carbon footprint.  
  2. Provide practitioners with plug‐and‐play prescriptions for epoch scheduling under different data and model regimes.  
  3. Advance the theory–practice gap in large‐scale machine learning by reconciling deep learning practice with stochastic optimization theory.  

3. Methodology  

3.1 Theoretical Framework  

Objective Function  
We consider the standard pretraining objective  
$$  
f(\theta)\;=\;\mathbb{E}_{x\sim\mathcal{D}}\bigl[\ell(\theta;x)\bigr],  
$$  
where $\theta\in\mathbb{R}^d$ are model parameters and $\ell(\theta;x)$ is the negative log‐likelihood (or related loss) on token sequence $x$. In practice we have a finite dataset of $N$ training examples $\{x_i\}_{i=1}^N$, and we make $E$ full passes (epochs) over this set using minibatch stochastic gradient descent (SGD).  

Gradient Noise Model  
Let $g_{t}=\nabla\ell(\theta_{t};x_{t})$ be the minibatch gradient at iteration $t$, where $x_t$ is sampled without replacement in each epoch but repeated across epochs. We model  
  – Unbiasedness: $\mathbb{E}[g_t\mid \theta_t]=\nabla f(\theta_t)$.  
  – Variance: $\operatorname{Var}(g_t\mid \theta_t) = \sigma^2$.  
  – Autocorrelation across epochs: If $t$ and $t'$ sample the same data point in different epochs, then  
    $$\operatorname{Cov}\bigl(g_t,g_{t'}\bigr)\;=\;\rho\,\sigma^2,\quad 0\le\rho<1.$$  

Under these assumptions, we can show (see Appendix A for full proof) that for constant stepsize $\eta$ and $T=EN/B$ total minibatch steps (batch size $B$),  
$$  
\min_{t=1,\dots,T}\mathbb{E}\bigl\|\nabla f(\theta_t)\bigr\|^2  
\;\le\;  
\frac{2\bigl(f(\theta_0)-f^*\bigr)}{\eta\,T}  
\;+\;\frac{\eta\,L}{2B}\,\sigma^2\,\bigl[\,1\;+\;\rho\,(E-1)\bigr]  
\,.  
\tag{1}$$  
Here $L$ is the Lipschitz constant of $\nabla f$. This bound suggests defining an effective dataset size  
$$  
N_{\rm eff}(E) \;=\;\frac{N}{\,1+\rho\,(E-1)\,}\,,  
\tag{2}$$  
so that repeated epochs yield diminishing returns once $\rho(E-1)\gg1$.  

Generalization Bound  
Using standard uniform‐stability arguments or PAC‐Bayes techniques (see, e.g., Feldman & Vondrák (2018)), one can similarly derive a generalization gap bound of the form  
$$  
\mathbb{E}\bigl[f(\theta_T)-f_{\rm test}(\theta_T)\bigr]  
\;\le\;  
\mathcal{O}\!\Bigl(\sqrt{\tfrac{\log(1/\delta)}{N_{\rm eff}(E)}}\Bigr)\,.  
\tag{3}$$  
Thus, both convergence speed and generalization are controlled by $N_{\rm eff}(E)$.  

Information‐Geometry Refinement  
To capture representation quality, we further model the parameter update on the statistical manifold of predictive distributions. Let $p_\theta(y\!\mid\!x)$ be the model’s predictive distribution. The Fisher information metric $G(\theta)$ endows curvature to the manifold. Data repetition changes the expected Fisher norm of parameter increments. One can show under mild regularity that  
$$  
\mathbb{E}\bigl[\Delta\theta^\top G(\theta)\,\Delta\theta\bigr]  
\;\approx\;\eta^2\,\bigl(1+\rho(E-1)\bigr)\,\sigma^2_{\rm F}\,,  
\tag{4}$$  
where $\sigma^2_{\rm F}$ is the Fisher‐norm variance of the stochastic gradient. Large $(1+\rho(E-1))$ may overshoot flat minima and hurt representation robustness.  

3.2 Algorithmic Steps  

We propose Algorithm 1 for pretraining with controlled data recycling:  

Algorithm 1: Recycled‐Epoch Pretraining  
Input: Dataset $\mathcal{D}=\{x_i\}_{i=1}^N$, epochs $E$, batch size $B$, stepsize $\eta$, model init $\theta_0$.  
for epoch $e=1$ to $E$ do  
 Shuffle $\mathcal{D}$ into minibatches $\{\mathcal{B}_{e,j}\}_{j=1}^{N/B}$.  
 for each minibatch $\mathcal{B}_{e,j}$ do  
  Compute gradient $g_{e,j} = \frac1B\sum_{x\in\mathcal{B}_{e,j}}\nabla\ell(\theta_{t};x)$.  
  $\theta_{t+1}\;\leftarrow\;\theta_t - \eta\;g_{e,j}$.  
  $t\leftarrow t+1$.  
 end for  
end for  

We will extend this pseudocode to include:  
  • Learning‐rate schedules $\eta_t = \eta_0\,\bigl(1 + t/T\bigr)^{-\alpha}$.  
  • Warmup and decay phases.  
  • Optional gradient‐clipping or weight decay.  

3.3 Experimental Design  

Models and Data  
  – Architectures: Transformer encoder models at scales 110 M, 1.3 B, and 2.7 B parameters.  
  – Pretraining corpora: a 500 GB subset of Common Crawl, plus English Wikipedia (~50 GB). Total $N\approx10^{10}$ tokens.  
  – Epoch schemes: $E\in\{1,2,4,8\}$. For fairness, we hold total number of steps $T=E\times N/B$ fixed to simulate constant compute.  

Control Variables  
  – Batch size $B$ fixed to 1 k tokens.  
  – Stepsize schedule: linear warmup for first 10% of $T$, then cosine decay.  
  – Weight decay and dropout fixed across runs.  

Evaluation Metrics  
 1. Convergence Speed: number of iterations to reach target training loss.  
 2. Test Perplexity: bits‐per‐token on held‐out validation set.  
 3. Downstream Generalization: fine‐tuning on GLUE benchmark, reporting average accuracy.  
 4. Representation Quality:  
 – Linear‐probe accuracy on SentEval tasks (SST2, MRPC, RTE).  
 – Mutual‐information proxy via InfoNCE bound on held‐out pairs.  
 5. Overfitting Risk: gap between training and validation loss curves.  
 6. Compute Efficiency: total GPU‐hour consumption and floating‐point operations (FLOPs).  

Repetitions and Statistical Significance  
Each experiment is repeated three times with different random seeds. We report mean ± standard deviation and perform paired t‐tests to assess significant differences at $p<0.05$.  

3.4 Data Quality Assessment for Recycling  

Following Marion et al. (2023) and White & Black (2024), we will implement a data‐pruning preprocessor that assigns a quality score $q(x)$ to each example using a pre‐trained small LM. We will experiment with two strategies:  
  (a) Uniform Recycling: recycle every example $E$ times.  
  (b) Quality‐Weighted Recycling: each example $x$ is repeated $E_x=\lceil E\cdot q(x)\rceil$ times.  

We will test whether quality‐weighted recycling yields a larger $N_{\rm eff}$ in practice by reducing $\rho$ (the autocorrelation factor) for low‐quality data.  

3.5 Theoretical vs. Empirical Comparison  

For each $E$, we record empirical measures of gradient autocorrelation $\hat\rho$, approximate $N_{\rm eff}^{\rm emp} = N / (1+\hat\rho(E-1))$, and compare with model‐predicted convergence rates from Eq. (1) and generalization gaps from Eq. (3). We will fit $\sigma^2$ and $L$ from small‐scale runs to enhance predictive accuracy of our bounds.  

4. Expected Outcomes & Impact  

4.1 Theoretical Contributions  
  • A closed‐form bound (Eq. 1) linking convergence rate to epoch count $E$, gradient‐noise correlation $\rho$, and batch size $B$.  
  • Definition of effective sample size $N_{\rm eff}(E)$ (Eq. 2) that unifies optimization and generalization perspectives.  
  • Extension of information‐geometry analysis (Eq. 4) that connects data repetition to representation robustness via Fisher‐norm variance.  

4.2 Practical Heuristics  
  From the intersection of theory and empirical validation, we will propose a simple rule of thumb:  
  $$  
  E^* \;\approx\; \frac{1}{\,\rho}\;\Bigl(\sqrt{\tfrac{2(f(\theta_0)-f^*)}{\eta\,T}\big/\bigl(\tfrac{\eta\,L\,\sigma^2}{2B}\bigr)} \;-\;1\Bigr)\,,  
  $$  
  which in many realistic settings simplifies to $E^*\approx 1/\rho \pm 1$. We will release a lightweight “epoch‐planner” tool that estimates $\rho$ from a short warmup run and recommends $E^*$.  

4.3 Empirical Findings  
  – Demonstration that naive multi‐epoch recycling ($E>2$) yields negligible gains beyond 2 × data visits when data autocorrelation $\rho>0.4$, confirming Eq. (2).  
  – Evidence that quality‐weighted recycling reduces overfitting and increases $N_{\rm eff}$ by up to 20% compared to uniform recycling.  
  – Validation that models trained at $E^*$ match or exceed the performance of standard $E=3$ runs at up to 30% fewer GPU‐hours.  

4.4 Broader Impact  
  • By principled reduction of unnecessary epochs, this work will cut LLM pretraining costs, democratizing access to large‐scale language modeling.  
  • The theoretical framework can guide future studies on curriculum learning, adaptive sampling, and federated pretraining.  
  • Our epoch‐planner tool and release of scripts/benchmarks will facilitate reproducible and sustainable LLM research.  

In sum, this proposal bridges a critical gap between stochastic optimization theory and the empirical practice of large‐model pretraining. It delivers both rigorous analysis and actionable guidelines for the next generation of efficient, cost‐effective LLMs.