# Graph-Based Surrogate-Assisted GFlowNets for Efficient Black-Box Discrete Sampling and Optimization

## 1. Introduction

Discrete sampling and optimization are central challenges in machine learning, appearing in diverse applications ranging from molecule design and protein engineering to combinatorial optimization and language model inference. While continuous domains benefit from gradient-based methods like Langevin dynamics, discrete spaces pose unique difficulties due to their non-differentiable nature and the combinatorial explosion of possibilities. These challenges are particularly pronounced in black-box settings, where objective functions are expensive to evaluate and gradient information is unavailable.

Recent advances in discrete sampling have included gradient-based discrete MCMC methods, continuous space embeddings, and novel proposal strategies such as Generative Flow Networks (GFlowNets). GFlowNets, in particular, have shown promise by formulating the sampling problem as learning stochastic policies that generate trajectories with probabilities proportional to a given reward function. This approach enables the generation of diverse high-quality samples and has demonstrated success in various domains including molecular generation, combinatorial optimization, and scientific discovery.

However, existing approaches face significant limitations when applied to black-box objectives with long-range, high-order correlations—precisely the characteristics exhibited by modern language models, protein design problems, and complex combinatorial spaces. Current methods either require prohibitively many objective function evaluations or fail to capture the intricate dependencies that define the problem structure. The resulting inefficiency makes many practical applications computationally infeasible.

This research aims to address these limitations by developing a novel framework that combines the generative capabilities of GFlowNets with the approximation power of graph neural networks (GNNs) in a surrogate-driven approach. Our key insight is that a learned surrogate model can guide the sampling process while drastically reducing the number of expensive black-box evaluations required. By iteratively refining this surrogate through active learning and using it to guide a GFlowNet, we can efficiently navigate complex discrete spaces even when direct objective evaluation is costly.

The proposed framework offers several advantages. First, it significantly reduces the number of true objective evaluations needed for effective sampling and optimization. Second, it naturally captures the structural relationships within discrete spaces through graph-based representations. Third, it provides a principled approach to balancing exploration and exploitation in black-box settings. Finally, it establishes a feedback loop between the surrogate model and the flow network, enabling continual improvement of both components.

The significance of this research extends to numerous applications. In computational chemistry, it could accelerate the discovery of molecules with desired properties. In protein engineering, it could enable more efficient design of sequences with specific functions. In machine learning itself, it could improve neural architecture search, hyperparameter optimization, and prompt engineering for large language models.

## 2. Methodology

### 2.1 Problem Formulation

We consider the problem of sampling from or optimizing a discrete black-box objective function $f: \mathcal{X} \rightarrow \mathbb{R}$, where $\mathcal{X}$ is a discrete space of objects. In the sampling setting, our goal is to generate samples from the distribution $p(x) \propto \exp(f(x)/T)$, where $T$ is a temperature parameter. In the optimization setting, we aim to find $x^* = \arg\max_{x \in \mathcal{X}} f(x)$. We assume that evaluating $f(x)$ is expensive, and no gradient information is available.

The discrete objects in $\mathcal{X}$ can be represented as graphs or sequences, and we assume a generation process where objects are constructed sequentially through a series of actions. This construction process defines a directed acyclic graph (DAG) $\mathcal{G} = (\mathcal{S}, \mathcal{E})$, where $\mathcal{S}$ is the set of all possible states (including partial constructions), and $\mathcal{E}$ represents valid transitions between states.

### 2.2 Graph Neural Network Surrogate

We propose to learn a surrogate model $\hat{f}_\theta: \mathcal{X} \rightarrow \mathbb{R}$ parameterized by a graph neural network (GNN) to approximate the true objective function $f$. The GNN processes the graph structure of discrete objects to capture the dependencies and interactions between their components.

For a graph $G = (V, E)$ representing a discrete object, we define node features $h_v$ for each node $v \in V$ and edge features $e_{uv}$ for each edge $(u,v) \in E$. The GNN surrogate model consists of multiple message-passing layers:

$$h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{(h_u^{(l)}, e_{uv}) | u \in \mathcal{N}(v)\}\right)\right)$$

where $\mathcal{N}(v)$ is the neighborhood of node $v$, and AGGREGATE and UPDATE are learnable functions implemented as neural networks. After $L$ layers of message passing, a readout function combines the final node representations to produce the predicted objective value:

$$\hat{f}_\theta(G) = \text{READOUT}\left(\{h_v^{(L)} | v \in V\}\right)$$

The surrogate model is trained to minimize the mean squared error between its predictions and the true objective values on a set of evaluated samples:

$$\mathcal{L}_{\text{surrogate}}(\theta) = \frac{1}{|\mathcal{D}|}\sum_{x \in \mathcal{D}} (\hat{f}_\theta(x) - f(x))^2$$

where $\mathcal{D}$ is the set of samples with known objective values.

### 2.3 GFlowNet Formulation

GFlowNets provide a framework for generating samples with probabilities proportional to a reward function. In our approach, we use the surrogate model to define the reward function for the GFlowNet.

For a complete object $x \in \mathcal{X}$, we define the reward function as:

$$R(x) = \exp(\hat{f}_\theta(x)/T)$$

where $T$ is a temperature parameter that controls the concentration of the distribution.

The GFlowNet learns stochastic policies to construct objects sequentially. Let $P_F(s' | s)$ be the forward policy, which defines the probability of transitioning from state $s$ to state $s'$, and $P_B(s | s')$ be the backward policy for the reverse direction. The GFlowNet is trained to satisfy the flow-matching condition:

$$F(s \rightarrow s') = F(s) \cdot P_F(s' | s) = F(s') \cdot P_B(s | s')$$

where $F(s)$ is the flow (unnormalized probability) through state $s$, and $F(s_0) = Z = \sum_{x \in \mathcal{X}} R(x)$ is the partition function.

We train the GFlowNet using the trajectory balance loss:

$$\mathcal{L}_{\text{TB}}(\phi) = \mathbb{E}_{\tau \sim P_{\text{TB}}}\left[\left(\log \frac{P_F(\tau)R(x_{\tau})}{P_B(\tau)}\right)^2\right]$$

where $P_F(\tau) = \prod_{i=0}^{n-1} P_F(s_{i+1}|s_i)$ is the probability of the forward trajectory, $P_B(\tau) = \prod_{i=0}^{n-1} P_B(s_i|s_{i+1})$ is the probability of the backward trajectory, and $P_{\text{TB}}$ is a training distribution over trajectories.

### 2.4 Iterative Framework

Our proposed method alternates between three main phases:

1. **Surrogate Training**: Update the GNN surrogate model using the available evaluations of the true objective function.
2. **GFlowNet Sampling**: Generate candidate samples using the GFlowNet guided by the current surrogate model.
3. **Active Learning**: Select the most informative candidates for evaluation with the true objective function.

The algorithm proceeds as follows:

1. Initialize a dataset $\mathcal{D}$ with a small set of randomly generated samples and their true objective values.
2. Train the GNN surrogate model $\hat{f}_\theta$ on $\mathcal{D}$.
3. For $t = 1, 2, \ldots, T$ iterations:
   a. Train a GFlowNet using the surrogate model $\hat{f}_\theta$ to define rewards.
   b. Generate a pool of candidate samples $\mathcal{C}$ using the GFlowNet.
   c. Select a batch of candidates $\mathcal{B} \subset \mathcal{C}$ based on an acquisition function.
   d. Evaluate the true objective function $f(x)$ for each $x \in \mathcal{B}$.
   e. Update the dataset: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(x, f(x)) | x \in \mathcal{B}\}$.
   f. Retrain or fine-tune the surrogate model $\hat{f}_\theta$ on the updated dataset $\mathcal{D}$.

### 2.5 Active Learning Strategy

To efficiently improve the surrogate model, we employ an active learning strategy that selects candidates maximizing expected information gain. We propose a composite acquisition function that balances exploration and exploitation:

$$\alpha(x) = \lambda \cdot \text{UCB}(x) + (1-\lambda) \cdot \text{BALD}(x)$$

The Upper Confidence Bound (UCB) component encourages exploitation of promising regions:

$$\text{UCB}(x) = \hat{f}_\theta(x) + \beta \cdot \sigma_\theta(x)$$

where $\sigma_\theta(x)$ is the predicted uncertainty (using MC-Dropout or ensemble methods), and $\beta$ is a hyperparameter controlling the exploration-exploitation trade-off.

The Bayesian Active Learning by Disagreement (BALD) component promotes exploration of uncertain regions:

$$\text{BALD}(x) = \mathbb{H}[\hat{f}_\theta(x)] - \mathbb{E}_{p(\theta|\mathcal{D})}[\mathbb{H}[\hat{f}_\theta(x)|\theta]]$$

where $\mathbb{H}$ denotes information entropy. The parameter $\lambda \in [0, 1]$ controls the balance between these components and can be adjusted during the optimization process.

### 2.6 Surrogate Calibration

To address potential biases in the surrogate model, we incorporate a calibration mechanism that adjusts the surrogate predictions based on observed discrepancies. For each batch of evaluated candidates, we compute a calibration factor:

$$c_t = \frac{1}{|\mathcal{B}_t|}\sum_{x \in \mathcal{B}_t} \frac{f(x)}{\hat{f}_\theta(x)}$$

This factor is used to adjust the surrogate predictions for the next iteration:

$$\hat{f}_{\text{calibrated}}(x) = c_t \cdot \hat{f}_\theta(x)$$

For more sophisticated calibration, we can learn a calibration model $g_\psi$ that maps surrogate predictions to calibrated values:

$$\hat{f}_{\text{calibrated}}(x) = g_\psi(\hat{f}_\theta(x), \sigma_\theta(x))$$

### 2.7 Experimental Design

We will evaluate our method on three categories of problems:

1. **Molecular Design**: Generating molecules with desired properties (e.g., drug-likeness, synthesizability, binding affinity).
2. **Protein Sequence Design**: Designing protein sequences with specific folding properties or functions.
3. **Combinatorial Optimization**: Solving problems such as maximum independent set, minimum vertex cover, and traveling salesman.

For each problem, we will compare our method against:
- Random search
- Bayesian optimization with various acquisition functions
- Standard GFlowNets without surrogate guidance
- MCMC methods (e.g., Metropolis-Hastings)
- Other state-of-the-art methods specific to each domain

We will measure performance using the following metrics:
- Sample efficiency: Number of true objective evaluations needed to reach a specified performance level
- Optimization performance: Best objective value found within a limited budget of evaluations
- Sampling quality: KL divergence from the target distribution (when ground truth is available)
- Diversity: Measuring the coverage of the discrete space using appropriate similarity metrics

## 3. Expected Outcomes & Impact

This research is expected to yield several significant outcomes:

1. **Methodological Advances**: We anticipate that our framework will establish a new paradigm for black-box discrete sampling and optimization, demonstrating that surrogate-assisted GFlowNets can efficiently navigate complex discrete spaces with minimal objective evaluations. The combination of graph neural network surrogates and flow-based generative models represents a novel approach that leverages both the structural understanding capabilities of GNNs and the directed exploration of GFlowNets.

2. **Sample Efficiency**: We expect our method to significantly reduce the number of expensive function evaluations required compared to existing approaches. This will be particularly valuable in domains where evaluation costs are prohibitive, such as computational chemistry, protein design, and large language model optimization. Our preliminary analyses suggest that the active learning component could reduce required evaluations by at least an order of magnitude in complex settings.

3. **Algorithmic Insights**: The research will shed light on the interplay between surrogate modeling and generative flow networks, providing insights into how approximation errors in surrogates affect sampling quality and how to effectively guide exploration based on uncertainty. These insights will inform future algorithm development in this domain.

4. **Application Impact**: We anticipate that our method will enable new applications in several domains:
   - **Computational Chemistry**: Accelerating the discovery of molecules with specific properties for drug development, materials science, and catalysis.
   - **Protein Engineering**: Enabling more efficient design of protein sequences with tailored functions, potentially advancing therapeutic protein development.
   - **Machine Learning Infrastructure**: Improving neural architecture search, compiler optimization, and hardware-software co-design.
   - **Large Language Model Deployment**: Optimizing prompt engineering, fine-tuning strategies, and inference configurations for LLMs.

5. **Software Framework**: We will develop and release an open-source implementation of our methodology, providing researchers and practitioners with tools to apply our approach to their own discrete sampling and optimization problems.

The broader impact of this work extends beyond the specific method proposed. By addressing the fundamental challenge of efficient exploration in discrete spaces, our research contributes to a more general understanding of how to navigate complex combinatorial landscapes with limited computational resources. This has implications for scientific discovery, engineering design, and artificial intelligence research. As discrete optimization underlies many critical problems—from designing energy-efficient materials to optimizing supply chains—advances in this area have the potential to accelerate progress across multiple fields of science and technology.

Furthermore, the methodology developed here may also provide insights into how intelligent systems can efficiently explore and learn in discrete environments, potentially informing research on reinforcement learning, automated reasoning, and artificial general intelligence. By demonstrating how structural understanding (via GNNs) and directed exploration (via GFlowNets) can be combined, our work contributes to the broader quest for more efficient and effective learning algorithms.