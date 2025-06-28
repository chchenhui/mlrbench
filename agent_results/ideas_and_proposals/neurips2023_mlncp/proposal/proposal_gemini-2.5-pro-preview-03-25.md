## 1. Title: Physics-Informed and Noise-Aware Training for Robust Deep Learning on Analog Hardware

## 2. Introduction

### 2.1 Background
The relentless progress in machine learning (ML), particularly the advent of large-scale generative models, has led to an exponential increase in computational demand. This trend collides with the escalating challenges faced by traditional digital computing based on von Neumann architectures, namely the saturation of Moore's Law, the memory wall bottleneck, and unsustainable energy consumption (Schuman et al., 2017). Consequently, there is a pressing need to explore alternative computing paradigms capable of delivering high performance and energy efficiency for demanding AI workloads.

Analog and neuromorphic computing platforms, including opto-electronic devices and Processing-in-Memory (PIM) architectures, represent a promising frontier (Marković et al., 2020; Shainline, 2023). These systems perform computations, such as matrix-vector multiplications (MVMs), directly in the physical domain, potentially offering orders-of-magnitude improvements in speed and energy efficiency compared to their digital counterparts. The co-design of ML models with specialized hardware, analogous to the synergy between deep learning and GPUs, holds immense potential for unlocking the next generation of AI accelerators (Misra & Saha, 2010).

However, the transition of ML models to analog hardware is fraught with challenges. Unlike the deterministic environment of digital processors, analog systems are inherently stochastic and imperfect. They suffer from thermal noise, shot noise, device-to-device variations (mismatch), limited precision (low bit-depth), parasitic effects, and non-linearities (Zhou et al., 2020; Wang et al., 2025a). Furthermore, these characteristics can drift over time due to environmental factors like temperature fluctuations and device aging (Wang et al., 2025a). Standard deep learning models, typically trained assuming high-precision, deterministic digital logic, often experience significant performance degradation when deployed onto such noisy, low-precision analog substrates (Gray et al., 2025).

Existing approaches to mitigate these issues often involve post-training quantization, simple noise injection during training (noisy training), or hardware-agnostic normalization techniques (Tsai et al., 2020; Wang et al., 2025b). While showing some promise, these methods may not fully capture the complex, state-dependent nature of analog hardware non-idealities. Recent work highlights the potential benefits of variance-aware noise schedules (Wang et al., 2025a), knowledge distillation (Zhou et al., 2020), and incorporating hardware constraints into model design (White et al., 2023; Green et al., 2024). Yet, a comprehensive framework that deeply integrates the physics of analog computation directly into the neural network training process remains an active area of research. The core idea is shifting from merely *tolerating* hardware imperfections to actively *co-designing* algorithms that leverage or are inherently robust to the underlying physical characteristics.

### 2.2 Research Objectives
This research proposes a novel hybrid training paradigm, termed **Physics-Informed Noise-Aware Training (PINAT)**, designed to enable robust and efficient training of deep neural networks directly targeting noisy analog hardware platforms. The primary goal is to develop models intrinsically resilient to analog imperfections by embedding knowledge of the hardware's physical behaviour into the learning process itself.

The specific objectives are:

1.  **Develop Realistic Physics-Based Noise Models:** Characterize and formulate mathematical models representing dominant noise sources (e.g., thermal noise, read/write noise, mismatch) and non-ideal behaviours (e.g., non-linearity, asymmetry, limited precision) pertinent to target analog hardware platforms (e.g., ReRAM crossbars, photonic processors). These models will be state-dependent, capturing variations based on input signals, weight values, and device states.
2.  **Design Differentiable Stochastic Neural Network Layers:** Introduce "Stochastic Residual Layers" (SRLs) (inspired by Black et al., 2024) and potentially other stochastic computation blocks (e.g., MVMs) that incorporate the developed physics-based noise models directly into the network's forward pass. Crucially, these layers must be differentiable to allow for gradient-based optimization.
3.  **Formulate a Physics-Informed Regularization Framework:** Develop novel regularization terms, integrated into the main learning objective function, that explicitly penalize weight configurations or activation patterns known to be sensitive to analog hardware limitations or difficult to implement reliably. This draws inspiration from physics-informed neural networks (Raissi et al., 2019) but applied to hardware constraints (cf. White et al., 2023).
4.  **Implement a Comprehensive Hybrid Training Algorithm:** Combine the stochastic forward pass (using SRLs/noisy blocks) and the physics-informed regularization within a standard deep learning training loop (e.g., using PyTorch or TensorFlow). This algorithm will simulate the noisy analog computation during training, guiding the network towards robust solutions.
5.  **Validate Robustness and Efficiency:** Rigorously evaluate the performance of models trained using PINAT on standard ML benchmarks. Assess their accuracy and robustness under simulated analog hardware conditions with varying levels of noise, mismatch, and precision. Compare against relevant baselines including standard digital training, post-training quantization, quantization-aware training, and existing noisy training techniques (Wang et al., 2025a; Wang et al., 2025b; Zhou et al., 2020). Estimate potential energy efficiency gains.
6.  **Explore Applicability to Emerging Models:** Investigate the efficacy of the PINAT framework for training models particularly sensitive to computational resources or potentially benefiting from inherent noise, such as Energy-Based Models (EBMs) (cf. Violet et al., 2025).

### 2.3 Significance
This research addresses a critical bottleneck hindering the widespread adoption of energy-efficient analog hardware for accelerating demanding AI tasks. By developing the PINAT framework, we aim to bridge the gap between theoretical ML models and the physical reality of next-generation computing platforms. The significance lies in several key areas:

1.  **Enhanced Robustness:** PINAT promises to produce models significantly more resilient to the inherent non-idealities of analog hardware compared to existing methods, potentially achieving near-digital accuracy even at very low precision and high noise levels. This could eliminate the need for costly post-training calibration or overly conservative hardware design.
2.  **Improved Energy Efficiency:** By enabling reliable computation on low-power analog substrates, this research directly contributes to reducing the substantial energy footprint of AI training and inference, promoting more sustainable AI development.
3.  **Enabling Edge AI:** Robust low-precision models unlocked by PINAT are crucial for deploying sophisticated AI capabilities, including generative models, onto resource-constrained edge devices (e.g., smartphones, IoT sensors, wearables) where power budgets are strictly limited.
4.  **Advancing Hardware-Software Co-Design:** This work pioneers a deeper form of co-design where the learning algorithm intimately adapts to the hardware's physical properties. This perspective can inform the design of future analog accelerators and ML models tailored for them (Green et al., 2024).
5.  **Unlocking New Model Classes:** The ability to train effectively in noisy environments might prove beneficial for certain model classes. For instance, the intrinsic stochasticity of analog hardware could potentially serve as a source of regularization or exploration, particularly advantageous for models like EBMs where sampling or noise injection is often a core component (Violet et al., 2025).

Successfully achieving these objectives would represent a significant step towards practical, scalable, and sustainable AI powered by novel analog computing paradigms.

## 3. Methodology

### 3.1 Overall Approach
The core of this research is the development and validation of the Physics-Informed Noise-Aware Training (PINAT) framework. This framework integrates hardware physics into deep learning training through two primary mechanisms: (1) simulating noisy analog computations within the neural network's forward and backward passes using differentiable, physics-based hardware models, and (2) regularizing the learning process to favour hardware-compatible solutions. The methodology involves theoretical development, algorithmic implementation, and extensive empirical evaluation using simulations targeting realistic analog hardware characteristics.

### 3.2 Data Collection and Preparation
For model training and evaluation, we will utilize standard, publicly available datasets to ensure comparability with existing literature. Initial experiments will focus on image classification tasks using:

*   **CIFAR-10 / CIFAR-100:** Widely used benchmarks for evaluating model robustness (Krizhevsky, 2009).
*   **Tiny ImageNet:** A larger-scale subset of ImageNet, used in related robustness studies (Wang et al., 2025a).
*   **(Optional) ImageNet (subset):** For evaluating scalability on more complex tasks.

We may also explore tasks suitable for EBMs, such as image generation or density estimation on datasets like MNIST or CelebA, following work like Violet et al. (2025). Standard data preprocessing techniques (normalization, augmentation) will be applied as appropriate for each dataset and model architecture.

While we will primarily rely on simulated hardware characteristics drawn from literature values (e.g., ReRAM noise models, mismatch statistics), if access to specific analog hardware or detailed characterization data becomes available, this empirical data will be used to refine the physics-based noise models.

### 3.3 Physics-Informed Noise Modelling and Stochastic Layers

We will model the primary non-idealities of a target analog compute paradigm (e.g., PIM based on crossbar arrays). Key aspects include:

1.  **Noisy MVM Simulation:** The core analog operation is MVM: $y = Wx$. In analog hardware, this is imperfect. We model the effective weight $W_{eff}$ as $W_{ideal} + \Delta W$, and account for input/output noise.
    $$ y_{analog} = f_{act}( (W + \Delta W_{mismatch} + \mathcal{N}_{weight}(W, t)) \cdot (x + \mathcal{N}_{input}(x)) + \mathcal{N}_{readout} ) $$
    where:
    *   $W$ is the ideal weight matrix.
    *   $\Delta W_{mismatch}$ represents static device-to-device variations, often modelled as Gaussian with zero mean and variance $\sigma^2_{mismatch}$.
    *   $\mathcal{N}_{weight}(W, t)$ represents temporal weight noise (e.g., programming noise, drift), potentially dependent on the weight value $W$ and time/temperature $t$. For ReRAM, this can involve conductance fluctuations.
    *   $\mathcal{N}_{input}(x)$ represents noise on the input signals (e.g., DAC noise).
    *   $\mathcal{N}_{readout}$ represents noise during sensing/accumulation (e.g., ADC noise, thermal noise in summation lines).
    *   $f_{act}$ is the activation function, which might also exhibit non-ideal analog behaviour (e.g., asymmetry, saturation levels).

2.  **Stochastic Residual Layers (SRLs):** Building upon Black et al. (2024), we propose incorporating noise directly into residual connections or specific computational blocks. For a layer $l$ with function $F(x_l, W_l)$, the output $x_{l+1}$ becomes:
    $$ x_{l+1} = \text{Activation}( \text{Combine}(x_l, F_{noisy}(x_l, W_l)) ) $$
    Where $F_{noisy}$ represents the analog computation simulation as described above. For instance, if $F$ is a convolution or linear layer, $F_{noisy}$ simulates the noisy MVM. Combine could be addition (for ResNets) or concatenation. The noise models $\mathcal{N}_{(\cdot)}$ will be parameterized based on hardware characteristics (e.g., target SNR, mismatch variance $\sigma^2_{mismatch}$, bit precision affecting quantization noise). Importantly, these noise injection mechanisms must be designed to be *differentiable* with respect to the layer parameters $W_l$ (and potentially inputs $x_l$). Techniques like the reparameterization trick (Kingma & Welling, 2013) will be employed for Gaussian noise sources. Non-differentiable effects like low-bit quantization will be handled using techniques like the Straight-Through Estimator (STE) (Bengio et al., 2013) during the backward pass.

3.  **Differentiable Surrogate Models:** The noisy operations ($F_{noisy}$) act as differentiable surrogate models of the targeted analog hardware components. These surrogates will be implemented as custom layers or functions within standard ML frameworks (PyTorch/TensorFlow).

### 3.4 Physics-Informed Regularization
To further guide the training towards hardware-compatible solutions, we introduce a physics-informed regularization term $L_{physics}$ added to the standard task loss $L_{task}$:
$$ L_{total} = L_{task} + \lambda L_{physics} $$
where $\lambda$ is a hyperparameter controlling the regularization strength. $L_{physics}$ will incorporate penalties based on known hardware limitations:

1.  **Weight Range/Distribution Penalty:** Analog weights often have limited range and specific statistical distributions due to device physics. $L_{physics}$ can include a term penalizing weights outside the reliably programmable range or encouraging weight distributions that align with device characteristics (e.g., log-normal for some memristors). Let $W_{min}, W_{max}$ be the reliable range.
    $$ L_{range} = \sum_{i,j} (\max(0, W_{ij} - W_{max}) + \max(0, W_{min} - W_{ij})) $$
2.  **Sensitivity Penalty:** Penalize weight configurations that are highly sensitive to noise. This could involve penalizing large magnitude weights (often more affected by relative noise) or weights near decision boundaries where small perturbations cause large output changes. One approach might involve adding a small noise perturbation during a secondary forward pass and penalizing the divergence from the original output.
3.  **Non-Linearity/Asymmetry Awareness:** If the hardware activation functions or computations exhibit specific non-linearities or asymmetries, the regularizer could encourage the network to operate in regions where these effects are minimal or predictable, or learn to compensate for them.

The specific formulation of $L_{physics}$ will be adapted based on the target hardware model. Its gradient $\frac{\partial L_{physics}}{\partial W}$ will directly influence weight updates during backpropagation.

### 3.5 PINAT Training Algorithm
The proposed training procedure is as follows:

1.  **Initialization:** Initialize network weights $W$. Define hardware parameters (noise levels $\sigma^2$, precision $P$, mismatch $\sigma^2_{mismatch}$, weight range $[W_{min}, W_{max}]$, etc.) for the surrogate models. Choose regularization strength $\lambda$.
2.  **Training Loop:** For each epoch:
    *   For each mini-batch of data $(X, Y)$:
        *   **Noisy Forward Pass:** Compute the network output $\hat{Y} = \text{Network}_{PINAT}(X, W)$. Inside the network, computations within designated layers (e.g., convolutions, linear layers, SRLs) use the differentiable surrogate models incorporating physics-based noise $\mathcal{N}_{(\cdot)}$ and non-idealities based on current weights $W$ and inputs $X$.
        *   **Task Loss Calculation:** Compute the primary task loss $L_{task} = \text{Criterion}(\hat{Y}, Y)$ (e.g., Cross-Entropy).
        *   **Physics Regularization Calculation:** Compute the physics-informed regularization term $L_{physics}$ based on current weights $W$ and potentially activations.
        *   **Total Loss Calculation:** Compute $L_{total} = L_{task} + \lambda L_{physics}$.
        *   **Backward Pass:** Compute gradients $\frac{\partial L_{total}}{\partial W}$ via backpropagation. Gradients flow through the noisy surrogate models (using reparameterization, STE, etc.).
        *   **Weight Update:** Update weights using an optimizer (e.g., Adam, SGD): $W \leftarrow \text{OptimizerUpdate}(W, \frac{\partial L_{total}}{\partial W})$. Optionally project weights into the allowed hardware range if not handled by $L_{physics}$.
3.  **Validation:** Periodically evaluate the model on a validation set using the *same* noisy forward pass simulation to monitor convergence and prevent overfitting to specific noise instances.

### 3.6 Experimental Design

1.  **Target Hardware Models:** Define specific analog hardware profiles based on literature (e.g., a generic ReRAM crossbar profile with reported noise/mismatch stats, or a photonic processor profile). Key parameters: precision (e.g., 4-bit, 6-bit, 8-bit weights/activations), read noise level (SNR dB), write noise (% variation), mismatch ($\sigma_{mismatch}$ as % of mean), weight range. We will also investigate robustness to *dynamic* noise by varying noise parameters during evaluation.
2.  **Network Architectures:** Evaluate on standard architectures like ResNet-18/34 (for CIFAR/Tiny ImageNet), potentially a small Vision Transformer (ViT), and a simple EBM architecture (e.g., based on convolutional layers).
3.  **Baselines for Comparison:**
    *   **FP32 Digital Baseline:** Standard training on digital hardware (GPU) with 32-bit floating-point precision.
    *   **Post-Training Quantization (PTQ):** Train FP32 model, then quantize weights and activations to target low precision. Evaluate under simulated noise.
    *   **Quantization-Aware Training (QAT):** Simulate quantization effects during training. Evaluate under simulated noise.
    *   **Generic Noisy Training (NT):** Inject simple i.i.d. Gaussian noise during training (e.g., Zhou et al., 2020; Wang et al., 2025b). Evaluate under simulated physics-based noise.
    *   **Variance-Aware Noisy Training (VANT):** Implement Wang et al.'s (2025a) method with dynamic noise schedules. Evaluate under simulated physics-based noise.
    *   **(If applicable) Noise-Aware Normalization:** Implement Tsai et al.'s (2020) method.
4.  **Evaluation Protocol:**
    *   Train models using each method (PINAT and baselines).
    *   Evaluate accuracy on the test set under:
        *   Ideal digital conditions (FP32 baseline only).
        *   Simulated analog conditions: fixed target precision, noise levels, mismatch. Run multiple inference passes per input to average over stochastic noise.
        *   Sweep Analysis: Evaluate accuracy across a range of noise levels, mismatch variances, and bit precisions to assess robustness curves.
        *   Dynamic Noise Evaluation: Test performance when noise parameters fluctuate during inference.
5.  **Evaluation Metrics:**
    *   **Accuracy:** Top-1 and Top-5 classification accuracy. For EBMs, metrics like FID score (for generation) or likelihood estimation.
    *   **Robustness:** Accuracy degradation relative to the FP32 baseline under various levels of simulated hardware imperfections. Area Under the Curve (AUC) for accuracy vs. noise/mismatch plots.
    *   **Training Stability:** Convergence speed (epochs to reach target accuracy), loss curve stability.
    *   **Computational Efficiency Estimation:** Calculate theoretical energy savings based on reduced precision operations (Energy(MAC) scales roughly with precision^2), reduced data movement in PIM. Compare theoretical MACs/Joule for PINAT-trained models on analog vs. baselines on digital/analog.

## 4. Expected Outcomes & Impact

### 4.1 Expected Outcomes
We anticipate that the PINAT framework will yield the following key outcomes:

1.  **Superior Robustness:** Models trained with PINAT are expected to exhibit significantly higher accuracy compared to baseline methods (PTQ, QAT, generic NT, VANT) when evaluated under realistic simulated analog hardware conditions, particularly at low precisions (e.g., 4-6 bits) and high noise/mismatch levels. We hypothesize that PINAT models will retain a larger fraction of their FP32 baseline accuracy (e.g., >90-95% retention) under these challenging conditions, outperforming other techniques.
2.  **Effective Noise/Physics Integration:** We expect to demonstrate that the explicit modeling of hardware noise statistics and physics through SRLs and the $L_{physics}$ term leads to measurably different weight distributions and activation patterns compared to standard or generic noisy training, reflecting adaptation to the hardware constraints. Ablation studies will verify the contribution of both the stochastic layers and the physics-informed regularization.
3.  **Validation across Models and Tasks:** The framework's effectiveness is expected to be demonstrated across different CNN architectures (e.g., ResNets) and potentially extend to other model types like ViTs or EBMs, showcasing its versatility. For EBMs, we expect PINAT might not only enhance robustness but potentially leverage hardware noise beneficially during training/sampling (cf. Violet et al., 2025).
4.  **Robustness to Dynamic Conditions:** PINAT-trained models are expected to show better resilience not only to static noise levels but also to dynamic fluctuations in noise characteristics, mimicking real-world environmental variations, compared to methods trained with fixed noise assumptions.
5.  **Quantifiable Efficiency Gains:** While direct hardware measurement is beyond the scope of simulation, the research will provide strong evidence and quantitative estimates for potential energy efficiency improvements achievable by operating reliably at lower precisions on analog hardware, supported by the robustness results.

### 4.2 Impact

The successful development and validation of the PINAT framework will have significant impacts:

1.  **Scientific Impact:** This research will contribute fundamental knowledge to the nascent field of machine learning on non-traditional hardware. It will provide a novel, principled framework for hardware-software co-design, moving beyond simple noise tolerance towards deep integration of hardware physics into learning algorithms. It will advance the understanding of training dynamics in highly stochastic, low-precision environments. The development of differentiable, physics-based surrogates for analog components will be a valuable contribution in itself.
2.  **Technological Impact:** PINAT can directly enable the practical deployment of complex deep learning models onto energy-efficient analog accelerators. This could unlock significant power savings (10x-100x) for AI computations, crucial for large-scale data centers and battery-powered edge devices. It facilitates the realization of compact, low-power AI systems for applications in autonomous systems, healthcare monitoring, real-time sensor fusion, and on-device generative AI, which are currently limited by the power envelope of digital hardware.
3.  **Societal Impact:** By drastically reducing the energy consumption of AI, this research contributes to making AI technology more sustainable and environmentally friendly. Furthermore, by enabling powerful AI on low-cost, low-power hardware, it can help democratize access to advanced AI capabilities, potentially bridging the digital divide and fostering innovation in resource-constrained settings.

In summary, this project aims to deliver a breakthrough in training methodologies for analog AI hardware, paving the way for a new generation of robust, efficient, and sustainable machine learning systems. It directly addresses the core challenges outlined in the workshop call, fostering synergy between ML algorithms and emerging compute paradigms.

## References (Sorted Alphabetically for convention, includes cited literature review items)

*   Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. *arXiv preprint arXiv:1308.3432*.
*   Black, D., Blue, S., & Red, K. (2024). Stochastic Residual Layers for Noise-Tolerant Neural Networks. *arXiv preprint arXiv:2402.34567*.
*   Doe, J., Smith, J., & Johnson, A. (2023). Noise-Aware Training for Robust Neural Networks on Analog Hardware. *arXiv preprint arXiv:2305.12345*.
*   Gray, P., Orange, L., & Cyan, T. (2025). Training Deep Neural Networks on Noisy Analog Hardware: A Survey. *arXiv preprint arXiv:2501.23456*.
*   Green, L., Yellow, M., & Purple, N. (2024). Co-Designing Neural Architectures and Analog Hardware for Energy-Efficient AI. *arXiv preprint arXiv:2407.45678*.
*   Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.
*   Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report, University of Toronto*.
*   Marković, D., Grollier, J., & Querlioz, D. (2020). Physics for neuromorphic computing. *Nature Reviews Physics*, 2(9), 499-510.
*   Misra, J., & Saha, I. (2010). Artificial neural networks in hardware: A survey of two decades of progress. *Neurocomputing*, 74(1-3), 239-255.
*   Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
*   Schuman, C. D., Potok, T. E., Patton, R. M., Birdwell, J. D., Dean, M. E., Rose, G. S., & Plank, J. S. (2017). A survey of neuromorphic computing and neural networks in hardware. *arXiv preprint arXiv:1705.06963*.
*   Shainline, J. M. (2023). Optoelectronic hardware for neuromorphic computing. *APL Photonics*, 8(10).
*   Tsai, L. H., Chang, S. C., Chen, Y. T., Pan, J. Y., Wei, W., & Juan, D. C. (2020). Robust Processing-In-Memory Neural Networks via Noise-Aware Normalization. *arXiv preprint arXiv:2007.03230*.
*   Violet, R., Indigo, S., & Magenta, A. (2025). Energy-Based Models on Analog Accelerators: Leveraging Noise for Regularization. *arXiv preprint arXiv:2504.56789*.
*   Wang, X., Borras, H., Klein, B., & Fröning, H. (2025a). Variance-Aware Noisy Training: Hardening DNNs against Unstable Analog Computations. *arXiv preprint arXiv:2503.16183*.
*   Wang, X., Borras, H., Klein, B., & Fröning, H. (2025b). On Hardening DNNs against Noisy Computations. *arXiv preprint arXiv:2501.14531*.
*   White, E., Brown, R., & Green, M. (2023). Physics-Informed Neural Networks for Analog Hardware Acceleration. *arXiv preprint arXiv:2310.67890*.
*   Zhou, C., Kadambi, P., Mattina, M., & Whatmough, P. N. (2020). Noisy Machines: Understanding Noisy Neural Networks and Enhancing Robustness to Analog Hardware Errors Using Distillation. *arXiv preprint arXiv:2001.04974*.