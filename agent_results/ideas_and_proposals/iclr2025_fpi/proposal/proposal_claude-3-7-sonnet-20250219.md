# Diffusion-Guided Alignment: A Token-Level Sampling Framework for Real-Time Control of Language Models

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language generation, reasoning, and various downstream tasks. However, ensuring that these models adhere to human preferences, safety guidelines, and quality standards remains challenging. Current alignment methods, particularly Reinforcement Learning from Human Feedback (RLHF), require computationally expensive fine-tuning processes that permanently modify model weights. While effective, these approaches suffer from several limitations: they are costly to implement, introduce instability during training, can lead to over-optimization toward specific metrics, and lack flexibility to adapt to changing requirements or user preferences without retraining.

Recent advances in diffusion models have shown promise in controlled generation across various domains. These approaches leverage iterative denoising processes guided by external signals to steer outputs toward desired characteristics. Notably, works such as DiffPO (Chen et al., 2025), Sampling Demons (Yeh et al., 2024), and SMC-based alignment methods (Kim et al., 2025) have demonstrated that diffusion-inspired techniques can enable inference-time control without modifying underlying model weights. However, existing approaches operate primarily at the sentence level or require complex sampling procedures that introduce significant latency during generation.

The gap in current research lies in developing efficient, token-level diffusion processes that can dynamically align LLM outputs with target distributions during inference while maintaining generation quality and minimizing computational overhead. This research aims to address this gap by proposing a novel diffusion-based sampling framework that enables real-time alignment of language models with arbitrary reward functions during inference.

This work makes several significant contributions to the field:

1. We introduce a token-level diffusion process specifically designed for autoregressive language models, allowing for fine-grained control during the generation process.

2. We develop a learned transition kernel that efficiently samples from the joint distribution of the base LLM and target density using gradient-based updates inspired by Langevin dynamics.

3. We propose an adaptive noise scheduling algorithm that dynamically adjusts the diffusion process based on the current generation state and target reward function.

4. We present a lightweight reward-aware proposal distribution that guides the sampling process toward high-reward regions without requiring backpropagation through the reward function.

5. We demonstrate that our approach enables efficient, controllable alignment across various reward functions and model architectures, with minimal computational overhead compared to traditional fine-tuning methods.

The successful development of this framework would enable on-the-fly adaptation of LLMs to diverse constraints or user preferences without requiring expensive model retraining, representing a significant advancement in the field of language model alignment.

## 2. Methodology

### 2.1 Problem Formulation

We frame the task of inference-time alignment as a target density sampling problem. Given a pre-trained language model $p_\theta(x)$ and a reward function $r(x)$ that quantifies desired properties of generated text (e.g., helpfulness, safety, or alignment with specific instructions), our goal is to sample from the reward-weighted distribution:

$$p_r(x) \propto p_\theta(x) \cdot \exp(\beta r(x))$$

where $\beta$ controls the strength of alignment toward the reward function. Traditional approaches would require fine-tuning the language model to directly optimize this objective. Instead, we develop a diffusion-based sampling framework that allows us to sample from $p_r(x)$ during inference without modifying the base model parameters $\theta$.

### 2.2 Token-Level Diffusion Process

We propose a token-level diffusion process that operates on the embedding space of the language model. Let $x_0 = (x_0^1, x_0^2, ..., x_0^n)$ be a sequence of token embeddings representing the target text. We define a forward diffusion process that gradually adds noise to these embeddings:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

where $t \in \{1, 2, ..., T\}$ represents diffusion steps, and $\beta_t$ is a time-dependent noise schedule. This forward process transforms the clean token embeddings into pure noise as $t$ approaches $T$.

For the reverse process, we train a denoising model $p_\phi(x_{t-1}|x_t)$ that learns to predict the distribution of $x_{t-1}$ given $x_t$:

$$p_\phi(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\phi(x_t, t), \Sigma_\phi(x_t, t))$$

where $\mu_\phi$ and $\Sigma_\phi$ are neural networks parameterized by $\phi$. Crucially, we design these networks to incorporate the base language model $p_\theta$ and the reward function $r(x)$ to guide the denoising process toward high-reward regions.

### 2.3 Reward-Guided Transition Kernel

To effectively sample from the target distribution $p_r(x)$, we develop a Langevin dynamics-inspired transition kernel. For each diffusion step $t$, we update the noisy embeddings using:

$$x_{t-1} = \mu_\phi(x_t, t) + \gamma_t \nabla_{x_t} \log p_r(x_t) + \sqrt{\Sigma_\phi(x_t, t)} \cdot z$$

where $z \sim \mathcal{N}(0, \mathbf{I})$ is standard Gaussian noise, and $\gamma_t$ is a step size parameter. The gradient term $\nabla_{x_t} \log p_r(x_t)$ can be decomposed as:

$$\nabla_{x_t} \log p_r(x_t) = \nabla_{x_t} \log p_\theta(x_t) + \beta \nabla_{x_t} r(x_t)$$

The first term guides the process toward high-probability sequences under the base language model, while the second term steers generation toward high-reward regions.

However, computing exact gradients with respect to discrete tokens is challenging. Instead, we propose a surrogate gradient estimation using:

$$\nabla_{x_t} \log p_\theta(x_t) \approx \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x_t^{(k)})}{p_\theta(x_t)} \cdot (x_t^{(k)} - x_t)$$

where $\{x_t^{(k)}\}_{k=1}^K$ are local perturbations of $x_t$. Similarly, we estimate $\nabla_{x_t} r(x_t)$ using a finite-difference approximation.

### 2.4 Adaptive Noise Scheduling

To balance exploration and exploitation during the diffusion process, we propose an adaptive noise scheduling algorithm that dynamically adjusts the noise levels based on the current state and reward function:

$$\beta_t = \beta_{\text{base}}(t) \cdot \exp(-\lambda \cdot r(x_t))$$

where $\beta_{\text{base}}(t)$ is a baseline noise schedule, and $\lambda$ controls the adaptation strength. This ensures that the diffusion process applies less noise to high-reward regions, preserving beneficial properties while allowing for more exploration in low-reward regions.

### 2.5 Reward-Aware Proposal Distribution

To further improve sampling efficiency, we develop a lightweight reward-aware proposal distribution $q_\psi(x_{t-1}|x_t, r)$ that directly incorporates reward information:

$$q_\psi(x_{t-1}|x_t, r) = \mathcal{N}(x_{t-1}; \mu_\phi(x_t, t) + \delta_\psi(x_t, r), \Sigma_\phi(x_t, t))$$

where $\delta_\psi$ is a small neural network that predicts a correction term based on the current state and reward function. This network is trained to minimize:

$$\mathcal{L}_\psi = \mathbb{E}_{x_t, r} \left[ \| \delta_\psi(x_t, r) - \gamma_t \nabla_{x_t} \log r(x_t) \|^2 \right]$$

This allows us to approximate the reward gradient without requiring backpropagation through the reward function during inference, significantly reducing computational overhead.

### 2.6 Training Procedure

Our training procedure involves two main components:

1. **Denoising Model Training**: We train the denoising model $p_\phi(x_{t-1}|x_t)$ using a combination of reconstruction loss and reward prediction loss:

$$\mathcal{L}_\phi = \mathbb{E}_{x_0, t, \epsilon} \left[ \|\mu_\phi(x_t, t) - \tilde{\mu}(x_0, x_t, t)\|^2 \right] + \alpha \mathbb{E}_{x_t} \left[ (r_\phi(x_t) - r(x_t))^2 \right]$$

where $\tilde{\mu}(x_0, x_t, t)$ is the optimal denoising mean, and $r_\phi(x_t)$ is a reward prediction auxiliary task.

2. **Proposal Network Training**: We train the reward-aware proposal network $\delta_\psi$ using the loss function defined in Equation 9, with gradients estimated using the finite difference approximation.

Both networks are trained on a diverse corpus of text with corresponding reward values, which can be obtained from existing human preference datasets or by evaluating candidate generations using external reward models.

### 2.7 Inference Algorithm

During inference, we implement our diffusion-guided sampling process as follows:

1. Initialize $x_T$ with random noise or with tokens from the base language model.
2. For $t = T, T-1, ..., 1$:
   a. Compute the denoising mean $\mu_\phi(x_t, t)$ and variance $\Sigma_\phi(x_t, t)$.
   b. Compute the reward-aware correction term $\delta_\psi(x_t, r)$.
   c. Sample $x_{t-1} \sim \mathcal{N}(x_{t-1}; \mu_\phi(x_t, t) + \delta_\psi(x_t, r), \Sigma_\phi(x_t, t))$.
   d. Update the adaptive noise schedule $\beta_{t-1}$ based on $r(x_{t-1})$.
3. Return the final denoised sequence $x_0$.

For autoregressive generation, we apply this process incrementally, denoising and sampling one token at a time while conditioning on previously generated tokens.

### 2.8 Experimental Design

To evaluate our approach, we will conduct experiments across several dimensions:

1. **Alignment Quality**: We will measure how effectively our method aligns generated text with target reward functions, compared to baseline approaches including:
   - Standard generation (no alignment)
   - RLHF fine-tuned models
   - Existing inference-time methods (e.g., DiffPO, SMC-based)

2. **Computational Efficiency**: We will analyze the computational overhead introduced by our method in terms of:
   - Generation latency (tokens per second)
   - Memory consumption
   - Energy usage

3. **Controllability**: We will assess the method's ability to adapt to different alignment objectives by evaluating performance across diverse reward functions:
   - Helpfulness/instruction-following
   - Safety/toxicity reduction
   - Truthfulness/factuality
   - Style or persona alignment

4. **Generalization**: We will test how our approach generalizes to:
   - Different model architectures and sizes
   - Out-of-distribution prompts
   - Multiple simultaneous reward objectives

5. **Ablation Studies**: We will conduct ablation studies to analyze the contribution of each component:
   - Token-level vs. sentence-level diffusion
   - Adaptive vs. fixed noise scheduling
   - Reward-aware proposal vs. standard sampling

For each experiment, we will use the following evaluation metrics:

- **Reward Alignment**: Measured by evaluating generated texts using the target reward function.
- **Text Quality**: Assessed using perplexity, BLEU, ROUGE, and BERTScore metrics.
- **Human Evaluation**: Ratings from human evaluators on dimensions including quality, helpfulness, and alignment with specified attributes.
- **Efficiency Metrics**: Generation speed (tokens/second), memory usage, and computational cost.

We will use standard benchmarks including the HarmBench, TruthfulQA, Anthropic Helpful/Harmless dataset, and MT-Bench for evaluation, ensuring comprehensive coverage of different alignment scenarios.

## 3. Expected Outcomes & Impact

This research is expected to yield several significant outcomes with broad implications for the field of language model alignment:

### 3.1 Technical Outcomes

1. **Efficient Inference-Time Alignment**: We anticipate that our diffusion-based sampling framework will enable effective alignment of language model outputs with target reward functions during inference, without requiring model retraining. This will be demonstrated through superior performance on alignment benchmarks compared to existing inference-time methods, while maintaining computational efficiency.

2. **Token-Level Control**: The proposed token-level diffusion process is expected to provide fine-grained control over generation, allowing for more precise alignment than sentence-level approaches. This will be particularly valuable for applications requiring nuanced control over generated content.

3. **Adaptability to Multiple Reward Functions**: Our method should demonstrate strong performance across diverse reward objectives, from safety and helpfulness to style and factuality. This flexibility will enable users to dynamically specify alignment criteria based on their needs.

4. **Scalability to Large Models**: The lightweight nature of our reward-aware proposal distribution should allow the method to scale efficiently to state-of-the-art language models without prohibitive computational overhead.

### 3.2 Practical Applications

1. **On-Demand Alignment**: Our approach will enable applications to dynamically adjust alignment parameters based on context, user preferences, or specific requirements without requiring multiple specialized models.

2. **Safety Enhancements**: By providing real-time steering toward safer content generation, our method could help address critical safety concerns in deployed language models while maintaining their overall utility.

3. **Personalization**: The framework could enable personalized language model interactions by dynamically aligning outputs with individual user preferences or requirements.

4. **Multi-Objective Optimization**: By extending our approach to handle multiple simultaneous reward functions, we can address scenarios requiring balanced optimization of competing objectives (e.g., helpfulness vs. safety).

### 3.3 Scientific Impact

1. **Theoretical Foundations**: Our work will contribute to the theoretical understanding of diffusion processes in discrete domains, particularly as applied to language model alignment.

2. **Bridging Sampling and Learning**: By combining principles from diffusion models and Langevin dynamics with language model inference, our research bridges these traditionally separate approaches to probabilistic inference.

3. **New Research Directions**: We anticipate that our framework will inspire new research into dynamic, inference-time control mechanisms for language models, potentially leading to more sophisticated approaches in the future.

### 3.4 Broader Impact

1. **Reduced Environmental Footprint**: By enabling alignment without retraining, our approach could significantly reduce the computational resources and associated environmental costs of maintaining aligned language models.

2. **Democratization of Alignment**: The ability to align models during inference without expensive retraining could make alignment techniques more accessible to researchers and developers with limited resources.

3. **Adaptability to Evolving Standards**: As societal expectations and alignment criteria evolve, our approach allows for dynamic adaptation without the need to retrain models from scratch.

In summary, this research aims to develop a novel diffusion-based framework for inference-time alignment of language models, enabling efficient, controllable, and adaptive generation that respects user preferences and safety constraints. If successful, this approach could significantly advance the state of the art in language model alignment, providing a more flexible and efficient alternative to traditional fine-tuning methods while maintaining strong performance across diverse alignment objectives.