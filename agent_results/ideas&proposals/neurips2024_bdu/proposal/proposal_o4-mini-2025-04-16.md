1. Title  
“LLM-Guided Prior Elicitation for Efficient Bayesian Optimization in Scientific Discovery”

2. Introduction  
Background  
Bayesian Optimization (BO) is a sample-efficient framework for the global optimization of expensive black-box functions. It proceeds by placing a probabilistic surrogate, typically a Gaussian Process (GP), over the unknown function \(f\colon \mathcal{X}\to\mathbb{R}\), and sequentially querying the true function at points that maximize an acquisition function. The surrogate’s prior—its choice of mean function \(m(x)\), covariance kernel \(k(x,x')\), and hyperpriors on kernel parameters—strongly influences BO’s performance, especially in the early stages when data are scarce. In high-dimensional or scientific discovery domains (e.g., material design, drug discovery), specifying informative priors requires domain expertise and can be a barrier for non-specialists.

Recent advances in Large Language Models (LLMs) have shown they can encode rich, structured knowledge from scientific literature, engineering texts, and empirical studies. Several works (AutoElicit, LLAMBO, LLANA) have begun to harness LLMs for expert prior elicitation in predictive modeling and BO. However, existing methods either require substantial manual prompt engineering, lack systematic mapping from natural language to probabilistic priors, or have been evaluated on narrow sets of problems.

Research Objectives  
We propose a systematic framework, LLM-BO-Prior, that uses LLMs to automatically elicit informative GP priors for Bayesian Optimization from natural-language descriptions of an optimization task. Our objectives are:  
  • To design a prompt-based pipeline that translates textual task descriptions into GP kernel choices, hyperparameter ranges, and input-relevance weights.  
  • To integrate the LLM-elicited priors into a BO loop, combining them with standard acquisition functions (e.g., Expected Improvement, Upper Confidence Bound).  
  • To evaluate the impact of LLM-derived priors on BO sample efficiency and final optimization performance across synthetic benchmarks, hyperparameter tuning, and real-world scientific tasks.  
  • To analyze the reliability, interpretability, and generalization of LLM-elicited priors, and compare against uninformative priors, expert-handcrafted priors, and existing LLM-BO methods.

Significance  
By lowering the barrier to specifying rich, domain-informed priors, our approach democratizes BO for practitioners in science and engineering. It bridges frontier LLM capabilities with Bayesian decision-theoretic methods, potentially accelerating discovery in domains where each function evaluation is costly (e.g., materials screening, biological assays). Moreover, characterizing the quality and limitations of LLM-elicited priors will inform future research on the intersection of large-scale language models and uncertainty-aware decision making.

3. Methodology  
3.1 Problem Formulation  
Let \(f\colon \mathcal{X}\subset \mathbb{R}^d \to \mathbb{R}\) be an expensive-to-evaluate black-box objective (e.g., yield of a chemical process as a function of temperature, pressure, composition). We seek  
\[
x^* = \arg\max_{x\in\mathcal{X}} f(x),
\]  
subject to a budget of \(N\) evaluations. Under BO, we place a Gaussian Process prior  
\[
f \sim \mathcal{GP}\bigl(m(x),\, k(x,x';\theta)\bigr),
\]  
where \(\theta\) are kernel hyperparameters (e.g., length-scales, signal variance). Given data \(\mathcal{D}_t = \{(x_i,y_i)\}_{i=1}^t\), we update the posterior and select the next point by maximizing an acquisition function \(\alpha(x\,|\,\mathcal{D}_t)\).  

3.2 LLM-Based Prior Elicitation  
Our core contribution is ElicitPrior, a function that maps a natural language description \(T\) of the optimization problem to a hyperprior over \(\theta\) and kernel structure:  
  1. **Prompt Engineering**  
     • We design a structured prompt template that asks the LLM to (i) identify relevant input dimensions, (ii) suggest an appropriate kernel family (e.g., RBF, Matérn, periodic), and (iii) recommend plausible ranges or distributions for hyperparameters such as length-scales \(\ell_j\) and signal variance \(\sigma_f^2\).  
     • Example prompt excerpt:  
       “Task: Optimize the yield of a biochemical reaction described in the literature as highly sensitive to temperature and pH. Suggest an appropriate Gaussian Process prior: list kernel type, relevant dimensions, and hyperparameter ranges.”  

  2. **Structured Output Parsing**  
     • We constrain the LLM to output a JSON-like response:  
       {  
         "kernel": "Matérn ν=2.5",  
         "length_scales": {"temp": [0.1, 10], "pH": [0.01, 1]},  
         "signal_variance": [0.5, 5.0],  
         "mean_function": "constant"  
       }  
     • A simple parser converts this into prior distributions, e.g.,  
       \(\ell_j \sim \mathrm{LogNormal}(\mu_j,\sigma_j)\) calibrated so that the 5th–95th percentiles match the suggested range.  

  3. **Hyperprior Construction**  
     • Given parsed ranges \([a_j, b_j]\) for each hyperparameter, we set  
       \[
         \log \ell_j \sim \mathcal{N}\!\bigl(\tfrac{\log a_j+\log b_j}{2},\,(\tfrac{\log b_j-\log a_j}{4})^2\bigr)
       \]  
     • For signal variance \(\sigma_f^2\), we similarly define a log-normal prior matching the suggested interval.

3.3 BO Algorithm with LLM Prior  
Algorithm 1: LLM-BO-Prior  
Input: Task description \(T\), domain \(\mathcal{X}\), evaluation budget \(N\)  
Output: Estimated optimum \(\hat x\)  

1. \(\mathcal{P}(\theta)\leftarrow\) ElicitPrior\((T)\)  
2. Initialize dataset \(\mathcal{D}_0\) with \(t_0\) random samples.  
3. For \(t = t_0,\dots,N-1\):  
     a. Fit GP posterior \(p(f\mid \mathcal{D}_t)\) using prior \(\mathcal{P}(\theta)\).  
     b. Select next query  
        \[
          x_{t+1} = \arg\max_{x\in\mathcal{X}} \alpha(x\,|\,\mathcal{D}_t),
        \]  
        with \(\alpha\) ∈ {EI, UCB, TS}.  
     c. Evaluate \(y_{t+1}=f(x_{t+1})+\epsilon\).  
     d. Augment \(\mathcal{D}_{t+1}=\mathcal{D}_t\cup\{(x_{t+1},y_{t+1})\}\).  
4. Return \(\hat x = \arg\max_{(x_i,y_i)\in\mathcal{D}_N} y_i\).

3.4 Experimental Design  
We will evaluate on three categories of tasks:

A. Synthetic Benchmark Functions  
  • Branin (2D), Hartmann6 (6D), Ackley (4D)  
  • Provide task descriptions (e.g., “multi-modal with three local minima in [−5,10]×[0,15]”), feed to LLM.  

B. Hyperparameter Tuning  
  • Tune a convolutional neural network on CIFAR-10: learning rate, momentum, weight decay.  
  • Task description drawn from model documentation: “optimize accuracy of ResNet-18 on CIFAR-10…”  

C. Scientific Discovery / Material Design  
  • Band gap prediction for perovskite materials as a function of compositional ratios.  
  • Text sourced from material science abstracts describing sensitivity to composition.

Baselines  
  1. Random search  
  2. Standard BO with default weak priors (e.g., length-scales ∼LogNormal(0,1)).  
  3. Expert-handcrafted priors elicited from domain specialists.  
  4. AutoElicit, LLAMBO, LLANA.

Evaluation Metrics  
  • Simple Regret \(r_t = f(x^*) - \max_{i\le t} f(x_i)\) as a function of \(t\).  
  • Number of evaluations to reach \(\epsilon\)-optimal (i.e., \(r_t < \epsilon\)).  
  • Final best value at \(t=N\).  
  • GP predictive log-likelihood and calibration error (e.g., expected calibration error) on held-out points.  

Statistical Analysis  
  • Run each method for 20 independent seeds.  
  • Report mean and standard error of regret curves.  
  • Use paired t-tests to assess significance of differences in evaluation counts to \(\epsilon\)-optimality.  

Implementation Details  
  • LLM: OpenAI GPT-4 or similar, accessed via API. Prompt contexts limited to 1024 tokens.  
  • GP and BO: Implemented in GPyTorch + BoTorch.  
  • Hardware: LLM inference on GPU-accelerated API servers; BO experiments on a CPU cluster.  
  • Code and prompts to be released as open source.

4. Expected Outcomes & Impact  
We anticipate that:  
  1. **Sample Efficiency Gains**: LLM-elicited priors will reduce the number of function evaluations by 10–50% compared to standard BO, especially in the low-data regime.  
  2. **Robustness Across Domains**: Our method will generalize to synthetic, ML hyperparameter, and materials tasks, matching or outperforming expert-chosen priors without manual tuning.  
  3. **Interpretability and Trust**: By parsing structured LLM outputs, we will provide transparent priors, enabling practitioners to inspect and adjust suggested hyperpriors.  
  4. **Insights into LLM Reliability**: Analysis of failure modes (e.g., overly narrow ranges, incorrect kernel type) will inform best practices in prompt design and hybrid human-LLM workflows.

Broader Impact  
By tightly coupling LLM knowledge extraction with Bayesian decision‐making, our framework:  
  • **Democratizes Bayesian Optimization** for non-experts in science and engineering, lowering the barrier to entry.  
  • **Accelerates Discovery** in fields where each experiment or simulation is costly, improving resource utilization.  
  • **Advances Trustworthy AI** by making priors transparent and verifiable, addressing concerns about black-box LLM advice.  
  • **Catalyzes Future Research** at the intersection of natural‐language processing and probabilistic modeling, suggesting new avenues such as multi‐modal prior elicitation.

In summary, LLM-BO-Prior aims to unlock the rich, latent domain expertise embedded in large language models for principled Bayesian optimization, offering a practical and extensible path toward uncertainty‐aware AI systems in critical real‐world applications.