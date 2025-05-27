# LLM-PEBO: Large Language Model-Guided Prior Elicitation for Bayesian Optimization

## 1. Introduction

Bayesian Optimization (BO) has emerged as a powerful methodology for optimizing expensive black-box functions, with applications spanning hyperparameter tuning, experimental design, and scientific discovery. At its core, BO builds a probabilistic surrogate model of the objective function—typically a Gaussian Process (GP)—and uses an acquisition function to balance exploration and exploitation when selecting new points to evaluate. The efficacy of this approach hinges critically on the quality of the prior distributions employed, particularly the kernel selection and hyperparameter initialization of the GP surrogate model.

However, specifying informative priors remains a significant challenge, especially for non-experts or in high-dimensional, complex domains. Domain experts often possess valuable knowledge about the optimization problem, but this knowledge is generally expressed qualitatively rather than as mathematical parameters for prior distributions. The absence of effective mechanisms to translate domain expertise into quantitative priors often forces practitioners to rely on default, uninformative priors, leading to inefficient exploration strategies that require numerous function evaluations before convergence.

This inefficiency is particularly problematic in scientific discovery tasks where each function evaluation may involve costly experiments or simulations. For example, in drug discovery, materials science, or chemical process optimization, a single evaluation might require days of laboratory work or computationally expensive simulations. The cost of these evaluations makes standard BO with uninformative priors prohibitively expensive or time-consuming for many real-world applications.

Recent advances in Large Language Models (LLMs) offer a promising solution to this challenge. Modern LLMs have demonstrated remarkable capabilities in understanding domain-specific knowledge, interpreting natural language descriptions, and making contextually appropriate recommendations. Their ability to extract and synthesize information from vast training corpora—including scientific literature, technical documentation, and domain-specific texts—positions them as potential tools for eliciting informative priors from natural language problem descriptions.

In this research, we propose LLM-PEBO (Large Language Model-guided Prior Elicitation for Bayesian Optimization), a novel framework that leverages LLMs to automatically generate informed priors for Bayesian Optimization from natural language problem descriptions. By bridging the gap between qualitative domain knowledge and quantitative prior specifications, LLM-PEBO aims to significantly enhance BO efficiency, particularly in the early stages of optimization when observations are limited.

The primary research objectives of this study are:

1. To develop a robust methodology for using LLMs to translate natural language problem descriptions into quantitative priors for Bayesian Optimization
2. To implement and evaluate mechanisms for both structural prior elicitation (e.g., kernel selection, relevant dimension identification) and parametric prior elicitation (e.g., length scales, signal variance)
3. To assess the impact of LLM-elicited priors on BO performance across diverse optimization tasks, including standard benchmarks and real-world applications
4. To investigate the interpretability and reliability of LLM-generated priors through ablation studies and comparative analyses

The significance of this research extends beyond theoretical contributions to BO methodology. By reducing the number of function evaluations required for optimization, LLM-PEBO could democratize access to advanced optimization techniques for domains where evaluation costs have previously been prohibitive. Furthermore, the framework could serve as a bridge between domain experts without statistical expertise and sophisticated Bayesian methods, allowing them to leverage their qualitative knowledge more effectively in quantitative optimization tasks.

## 2. Methodology

### 2.1 Overall Framework

The LLM-PEBO framework consists of four main components:

1. **Problem Description Processing**: Transforming natural language descriptions into structured prompts for the LLM
2. **Prior Elicitation Pipeline**: Extracting structural and parametric prior information from LLM outputs
3. **Bayesian Optimization Implementation**: Incorporating elicited priors into the BO process
4. **Feedback Integration Mechanism**: Refining priors based on initial optimization results

Figure 1 illustrates the overall architecture of the LLM-PEBO framework.

### 2.2 Problem Description Processing

The input to our system is a natural language description of the optimization problem, which may include:
- The objective function's characteristics
- Domain constraints and boundary conditions
- Prior knowledge from scientific literature or domain expertise
- Expected behavior or patterns in the response surface

To process this information effectively, we implement a multi-stage prompting strategy:

1. **Initial Parsing**: We extract key elements from the problem description using a structured prompt template that directs the LLM to identify:
   - The dimensionality and nature of the input space
   - Continuity and smoothness properties of the objective function
   - Expected correlations between input dimensions
   - Constraints on the search space

2. **Clarification Dialogue**: For ambiguous or incomplete problem descriptions, we implement an interactive dialogue component where the LLM generates targeted questions to elicit additional information from the user.

3. **Representation Formatting**: The processed information is structured into a standardized format that serves as input to the prior elicitation pipeline.

### 2.3 Prior Elicitation Pipeline

The prior elicitation pipeline consists of two main stages: structural prior elicitation and parametric prior elicitation.

#### 2.3.1 Structural Prior Elicitation

In this stage, the LLM determines the appropriate structure for the GP surrogate model:

1. **Kernel Selection**: The LLM recommends a suitable kernel or composition of kernels based on the expected properties of the objective function. The candidate kernels include:
   - Radial Basis Function (RBF) kernel for smooth functions
   - Matérn kernels for functions with varying degrees of smoothness
   - Periodic kernels for functions with cyclical patterns
   - Linear kernels for approximately linear relationships
   - Composite kernels (sums and products) for complex behavior

2. **Dimension Relevance Assessment**: The LLM identifies which input dimensions are likely to be relevant to the objective function and suggests potential automatic relevance determination (ARD) configurations.

3. **Mean Function Selection**: The LLM recommends an appropriate mean function for the GP based on prior knowledge about the expected trend of the objective function.

The structural elicitation is formalized as follows:

$$k(x, x') = \text{LLM}_{\text{structure}}(D, \mathcal{K})$$

where $D$ is the problem description, $\mathcal{K}$ is the set of available kernel functions, and $\text{LLM}_{\text{structure}}$ represents the LLM's structural prior elicitation component.

#### 2.3.2 Parametric Prior Elicitation

Once the structure is determined, the LLM suggests appropriate hyperparameter priors:

1. **Length Scale Initialization**: For each dimension $i$, the LLM suggests a prior distribution for the length scale parameter $l_i$:

$$p(l_i) = \text{LLM}_{\text{param}}(D, i, \text{range}_i)$$

where $\text{range}_i$ is the range of the $i$-th dimension.

2. **Signal Variance Initialization**: The LLM suggests a prior for the signal variance $\sigma_f^2$ based on the expected variation in the objective function:

$$p(\sigma_f^2) = \text{LLM}_{\text{param}}(D, \text{"variance"}, \text{context})$$

3. **Noise Variance Estimation**: If the objective function evaluations are expected to be noisy, the LLM provides a prior for the noise variance $\sigma_n^2$:

$$p(\sigma_n^2) = \text{LLM}_{\text{param}}(D, \text{"noise"}, \text{context})$$

For each hyperparameter, the LLM generates both a point estimate for initialization and a distribution that captures the uncertainty in this estimate, enabling robust hyperparameter optimization during the BO process.

### 2.4 Bayesian Optimization Implementation

We integrate the elicited priors into a standard Bayesian Optimization framework:

1. **GP Surrogate Model**: We construct a GP surrogate model using the elicited kernel structure and hyperparameter priors:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

where $m(x)$ is the mean function and $k(x, x')$ is the kernel function recommended by the LLM.

2. **Acquisition Function**: We employ a standard acquisition function (Expected Improvement) to select the next evaluation point:

$$x_{t+1} = \arg\max_x \alpha(x | \mathcal{D}_t, \theta_{\text{LLM}})$$

where $\mathcal{D}_t$ is the dataset of observations up to iteration $t$, $\theta_{\text{LLM}}$ represents the hyperparameters elicited by the LLM, and $\alpha$ is the acquisition function.

3. **Hyperparameter Adaptation**: During optimization, we periodically update the GP hyperparameters using Markov Chain Monte Carlo (MCMC) sampling from the posterior distribution, with the LLM-elicited priors serving as the prior distributions in this process.

### 2.5 Feedback Integration Mechanism

To enhance the robustness of our approach, we implement a feedback mechanism that refines the LLM-elicited priors based on initial optimization results:

1. **Initial Exploration Phase**: We conduct a brief initial exploration phase (e.g., 5-10% of the total budget) using the LLM-elicited priors.

2. **Performance Assessment**: We assess the quality of the elicited priors by comparing the observed objective function values with the GP predictions.

3. **Prior Refinement**: Based on this assessment, we prompt the LLM to refine its recommendations, providing it with information about the discrepancy between expectations and observations.

This feedback loop allows the framework to correct potential misalignments between the LLM's interpretation of the problem description and the actual characteristics of the objective function.

### 2.6 Experimental Design

To evaluate the effectiveness of LLM-PEBO, we design a comprehensive experimental protocol encompassing both synthetic benchmarks and real-world applications.

#### 2.6.1 Synthetic Benchmark Functions

We evaluate our approach on standard benchmark functions with varying characteristics:

1. **Low-dimensional functions** (2-3D):
   - Branin, Ackley, Rosenbrock
   - Synthetic functions with known ground truth

2. **Medium-dimensional functions** (5-10D):
   - Hartmann6, Levy, Michalewicz
   - Functions with varying degrees of multimodality and smoothness

3. **High-dimensional functions** (20-100D):
   - Sparse Ackley, Sparse Rosenbrock
   - Functions with intrinsic low-dimensionality embedded in high-dimensional spaces

For each function, we create detailed natural language descriptions that include relevant information about the function's properties, without explicitly providing mathematical formulations.

#### 2.6.2 Real-world Applications

We assess the practical utility of LLM-PEBO in three real-world domains:

1. **Hyperparameter Tuning**:
   - Neural network hyperparameter optimization for image classification tasks
   - Gradient boosting model tuning for tabular data prediction

2. **Materials Design**:
   - Optimization of alloy compositions for specific mechanical properties
   - Polymer design for targeted thermal properties

3. **Molecular Optimization**:
   - Drug-like molecule optimization for binding affinity
   - Chemical reaction yield optimization

For each application, we construct problem descriptions based on scientific literature and domain-specific knowledge, simulating the information a domain expert might provide.

#### 2.6.3 Baseline Methods

We compare LLM-PEBO against the following baselines:

1. **Standard BO** with default priors (uninformative)
2. **Random Search** as a non-Bayesian baseline
3. **Expert-defined BO** with manually specified priors (where available)
4. **LLAMBO** and other existing LLM-enhanced BO approaches from recent literature

#### 2.6.4 Evaluation Metrics

We employ multiple evaluation metrics to assess performance:

1. **Simple Regret**: The difference between the best found value and the global optimum
   $$r_T = f(x^*) - f(x_{\text{best}})$$

2. **Sample Efficiency**: The number of function evaluations required to reach a target performance threshold
   $$N_{\epsilon} = \min\{t : r_t \leq \epsilon\}$$

3. **Area Under Curve (AUC)**: The area under the best-found-value curve over the optimization trajectory
   $$\text{AUC} = \int_0^T (f(x^*) - f(x_{\text{best}}^t)) dt$$

4. **Calibration Error**: The discrepancy between the GP's predicted uncertainty and the empirical error distribution
   $$\text{CE} = \sum_{i=1}^n (f(x_i) - \mu_i)^2 / \sigma_i^2 - 1$$
   where $\mu_i$ and $\sigma_i^2$ are the mean and variance of the GP prediction at $x_i$

#### 2.6.5 Ablation Studies

To understand the contribution of different components of our framework, we conduct ablation studies:

1. **Structural vs. Parametric Priors**: Evaluating the impact of LLM-guided structural priors versus parametric priors
2. **Feedback Mechanism**: Assessing the contribution of the feedback integration component
3. **LLM Model Selection**: Comparing performance across different LLM architectures and sizes
4. **Prompt Engineering**: Investigating the sensitivity of the framework to different prompting strategies

## 3. Expected Outcomes & Impact

### 3.1 Anticipated Results

We anticipate that LLM-PEBO will demonstrate significant improvements in Bayesian Optimization efficiency, particularly in the early stages of optimization when observations are limited. Specifically, we expect to observe:

1. **Improved Sample Efficiency**: A substantial reduction in the number of function evaluations required to reach a specified performance threshold compared to standard BO with uninformative priors, especially for complex functions with structure that can be inferred from natural language descriptions.

2. **Enhanced Early-Stage Performance**: Superior optimization performance during the initial stages (first 10-30% of evaluations), reflecting the value of the LLM-elicited priors in guiding exploration toward promising regions.

3. **Accurate Uncertainty Quantification**: Better calibrated uncertainty estimates in the GP surrogate model, leading to more effective exploration-exploitation trade-offs and more reliable convergence guarantees.

4. **Domain Adaptability**: Consistent performance improvements across diverse application domains, demonstrating the framework's ability to extract and leverage domain-specific knowledge from problem descriptions.

5. **Interpretable Prior Recommendations**: Transparent and interpretable prior recommendations from the LLM, providing insights into how domain knowledge is translated into quantitative priors.

### 3.2 Scientific Impact

The scientific contributions of this research will advance multiple areas:

1. **Bayesian Optimization Methodology**: The introduction of a systematic approach to elicit informative priors from natural language descriptions addresses a fundamental limitation in current BO frameworks, potentially establishing a new paradigm for incorporating domain knowledge into optimization algorithms.

2. **Natural Language Processing for Scientific Applications**: This work demonstrates a novel application of LLMs in scientific computing, specifically for translating qualitative descriptions into quantitative mathematical formulations, which could inspire similar approaches in other computational domains.

3. **Uncertainty Quantification**: By improving prior specification, this research enhances the quality of uncertainty estimates in surrogate models, contributing to more reliable decision-making under uncertainty.

4. **Human-AI Collaborative Optimization**: The framework facilitates a new form of collaboration between domain experts and optimization algorithms, where experts can contribute their knowledge through natural language rather than mathematical formulations.

### 3.3 Practical Impact

The practical implications of this research extend to numerous application domains:

1. **Accelerated Scientific Discovery**: By reducing the number of experiments required for optimization, LLM-PEBO could significantly accelerate scientific discovery processes in fields such as materials science, drug discovery, and chemical engineering.

2. **Democratized Access to Advanced Optimization**: The ability to translate natural language descriptions into effective priors lowers the barrier to entry for sophisticated Bayesian methods, making them accessible to domain experts without extensive statistical expertise.

3. **Resource Efficiency**: The improved sample efficiency translates directly to cost savings in domains where function evaluations are expensive, enabling more efficient allocation of limited experimental or computational resources.

4. **Enhanced Automation**: This approach could facilitate the automation of optimization processes in scientific workflows, reducing the need for manual intervention in iterative experimental design.

### 3.4 Limitations and Future Work

We acknowledge several potential limitations that will guide future work:

1. **LLM Knowledge Boundaries**: The quality of elicited priors is constrained by the knowledge encoded in the LLM's training data, potentially limiting performance in cutting-edge domains not well-represented in the training corpus.

2. **Hallucination Risks**: LLMs may generate plausible but incorrect recommendations, necessitating robust validation mechanisms to identify and mitigate such cases.

3. **Computational Overhead**: The use of LLMs introduces additional computational costs, which must be balanced against the potential savings in function evaluations.

Future research directions stemming from this work include:

1. Extending the framework to handle multimodal inputs, such as combining text descriptions with visual information or preliminary data
2. Developing specialized fine-tuning approaches to enhance LLM performance on specific domains
3. Exploring the integration of LLM-elicited priors with other advanced BO variants, such as multi-fidelity or multi-objective optimization
4. Investigating continual learning approaches to refine the LLM's prior elicitation capabilities based on optimization outcomes across multiple tasks

In conclusion, LLM-PEBO represents a significant step toward bridging the gap between qualitative domain knowledge and quantitative optimization, with far-reaching implications for accelerating scientific discovery and enabling more efficient resource allocation in optimization tasks.