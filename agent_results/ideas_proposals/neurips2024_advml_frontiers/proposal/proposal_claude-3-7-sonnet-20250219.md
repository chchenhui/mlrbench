# Cross-Modal Adversarial Immunization: A Unified Defense Framework for Large Multimodal Models

## 1. Introduction

### Background

Large Multimodal Models (LMMs) have revolutionized artificial intelligence by seamlessly integrating information across diverse perceptual domains, enabling sophisticated reasoning that bridges vision, language, and other modalities. Models like GPT-4V, CLIP, and LLaVA represent significant advances in AI's ability to understand and generate content that spans multiple sensory channels. However, this integration creates novel attack surfaces that traditional security paradigms fail to address adequately.

Cross-modal adversarial attacks represent an emerging threat vector where perturbations in one modality (e.g., subtle image manipulations) trigger unexpected failures in another modality (e.g., text reasoning). These attacks exploit the complex integration points between different perceptual domains within LMMs, presenting unique challenges not encountered in single-modality systems. Recent research by Rahmatullaev et al. (2025) demonstrated universal adversarial attacks on aligned multimodal LLMs, while Dou et al. (2024) introduced CrossFire, an attack methodology that effectively manipulates multi-modal models by transforming inputs across modality boundaries.

The literature reveals a concerning vulnerability: current defenses typically focus on single-modality protection, leaving models exposed at the critical junctions where modalities interface. While ProEAT (Lu et al., 2025) has made advances in defending against jailbreak attacks through projector-based adversarial training, comprehensive cross-modal defense mechanisms remain underdeveloped. Wei et al. (2021) revealed the feasibility of transferring adversarial perturbations across modalities (from images to videos), highlighting the transferability of these attacks. Despite progress in cross-modal consistency training (White et al., 2023) and adaptive defense mechanisms (Black et al., 2024), there remains a critical need for a unified framework that specifically addresses the unique characteristics of cross-modal vulnerabilities.

### Research Objectives

This research proposes a novel defensive framework called "Cross-Modal Adversarial Immunization" (CMAI) with the following objectives:

1. Develop a cross-modal consistency verification module that can efficiently detect misalignments between representations across modalities, serving as an early warning system for potential adversarial interventions.

2. Design and implement modality-bridging adversarial training techniques that explicitly generate perturbations targeting cross-modal transfer points to strengthen these vulnerable integration zones.

3. Create an adaptive robustness mechanism that dynamically adjusts defensive priorities based on detected attack patterns, enabling real-time response to emerging threats.

4. Evaluate the proposed framework's effectiveness across multiple state-of-the-art LMMs and diverse attack scenarios to establish empirical evidence of improved robustness.

5. Analyze the trade-offs between adversarial robustness and model performance on benign inputs to develop optimization strategies that minimize performance degradation.

### Significance

The significance of this research extends across several dimensions:

**Security Implications**: As LMMs increasingly power critical systems in healthcare, autonomous transportation, content moderation, and financial services, addressing cross-modal vulnerabilities becomes essential to preventing potentially catastrophic system failures or manipulation.

**Theoretical Advancement**: The proposed framework will advance our understanding of adversarial vulnerabilities at the intersection of multiple modalities, potentially revealing fundamental principles about cross-modal reasoning in both artificial and biological intelligence systems.

**Practical Applications**: By developing a defensive framework that can be integrated with minimal modifications to existing LMMs, this research will provide an accessible security enhancement for a wide range of applications, particularly those in high-stakes environments where reliability across multiple modalities is critical.

**Model Development**: Insights from this research will inform future LMM architectures, potentially leading to inherently more robust designs that preemptively address cross-modal vulnerabilities.

The proposed Cross-Modal Adversarial Immunization framework addresses a critical gap in the current literature on adversarial machine learning for multimodal systems, with potential impacts spanning theoretical understanding, practical security, and future model development.

## 2. Methodology

Our proposed Cross-Modal Adversarial Immunization (CMAI) framework comprises three integrated components designed to comprehensively address cross-modal vulnerabilities in Large Multimodal Models (LMMs). The methodology details the architecture and algorithmic approach for each component, followed by the experimental design to validate our approach.

### 2.1 Cross-Modal Consistency Verification Module (CMCVM)

The CMCVM serves as the first line of defense by monitoring the alignment between representations across different modalities. 

#### 2.1.1 Architecture

The CMCVM consists of:

1. **Modality-Specific Feature Extractors**: For each modality $m \in M$ (where $M$ is the set of all modalities supported by the LMM), we utilize the existing embedding layers to extract representations $\mathbf{z}_m \in \mathbb{R}^{d_m}$.

2. **Alignment Projection Networks**: For each modality pair $(i,j) \in M \times M$, we train projection networks $f_{i \to j}: \mathbb{R}^{d_i} \to \mathbb{R}^{d_j}$ that map representations from modality $i$ to modality $j$:

$$f_{i \to j}(\mathbf{z}_i) = \mathbf{W}_{i \to j}\mathbf{z}_i + \mathbf{b}_{i \to j}$$

where $\mathbf{W}_{i \to j} \in \mathbb{R}^{d_j \times d_i}$ and $\mathbf{b}_{i \to j} \in \mathbb{R}^{d_j}$ are learned parameters.

3. **Consistency Scoring Function**: For inputs with multiple modalities, we compute a consistency score $C(x_i, x_j)$ between modalities $i$ and $j$:

$$C(x_i, x_j) = \cos(f_{i \to j}(\mathbf{z}_i), \mathbf{z}_j) = \frac{f_{i \to j}(\mathbf{z}_i) \cdot \mathbf{z}_j}{||f_{i \to j}(\mathbf{z}_i)|| \cdot ||\mathbf{z}_j||}$$

where $\cos(\cdot,\cdot)$ denotes cosine similarity.

#### 2.1.2 Training Procedure

The CMCVM is trained on benign multimodal data to learn the expected patterns of cross-modal consistency:

1. For a training dataset $\mathcal{D} = \{(x_1^{(n)}, x_2^{(n)}, ..., x_{|M|}^{(n)})\}_{n=1}^N$ containing aligned inputs across modalities:

2. Minimize the consistency loss:

$$\mathcal{L}_{\text{consist}} = \sum_{(i,j) \in M \times M} \alpha_{i,j} \left( 1 - \frac{1}{N} \sum_{n=1}^N C(x_i^{(n)}, x_j^{(n)}) \right)$$

where $\alpha_{i,j}$ are learnable importance weights for each modality pair.

3. Additionally, incorporate contrastive learning to distinguish between aligned and misaligned pairs:

$$\mathcal{L}_{\text{contrastive}} = -\log\frac{\exp(C(x_i^{(n)}, x_j^{(n)})/\tau)}{\sum_{k \neq n} \exp(C(x_i^{(n)}, x_j^{(k)})/\tau)}$$

where $\tau$ is a temperature parameter.

4. The final training objective is:

$$\mathcal{L}_{\text{CMCVM}} = \mathcal{L}_{\text{consist}} + \lambda_{\text{contrastive}} \mathcal{L}_{\text{contrastive}}$$

where $\lambda_{\text{contrastive}}$ is a hyperparameter controlling the contribution of the contrastive loss.

### 2.2 Modality-Bridging Adversarial Training (MBAT)

The MBAT component explicitly generates adversarial perturbations targeting cross-modal transfer points to strengthen model robustness at these vulnerable junctions.

#### 2.2.1 Cross-Modal Adversarial Example Generation

For each modality pair $(i,j) \in M \times M$, we generate adversarial examples that maximize cross-modal inconsistency:

1. For input $x_i$ in modality $i$, generate adversarial perturbation $\delta_i$ by:

$$\delta_i = \arg\max_{\delta: ||\delta||_p \leq \epsilon} \mathcal{L}_{\text{attack}}(x_i + \delta, x_j)$$

where $\mathcal{L}_{\text{attack}}$ is defined as:

$$\mathcal{L}_{\text{attack}}(x_i, x_j) = -C(x_i, x_j) + \beta D_{\text{KL}}(p(y|x_i, x_j) || p(y|x_i+\delta, x_j))$$

Here, $p(y|x_i, x_j)$ is the model's prediction given inputs from modalities $i$ and $j$, $D_{\text{KL}}$ is the Kullback-Leibler divergence, and $\beta$ controls the trade-off between decreasing cross-modal consistency and changing the model's predictions.

2. We solve this optimization problem using Projected Gradient Descent (PGD):

$$\delta_i^{t+1} = \Pi_{||\delta||_p \leq \epsilon} \left( \delta_i^t + \alpha \cdot \text{sign}(\nabla_{\delta} \mathcal{L}_{\text{attack}}(x_i + \delta_i^t, x_j)) \right)$$

where $\Pi$ denotes projection onto the $\ell_p$-ball of radius $\epsilon$, $\alpha$ is the step size, and $t$ is the iteration number.

#### 2.2.2 Bidirectional Adversarial Training

To comprehensively address cross-modal vulnerabilities, we implement bidirectional adversarial training:

1. For each minibatch $\{(x_1^{(n)}, x_2^{(n)}, ..., x_{|M|}^{(n)})\}_{n=1}^B$:

2. Generate adversarial perturbations $\delta_i^{(n)}$ for each modality $i$ and sample $n$.

3. Update model parameters $\theta$ by minimizing:

$$\mathcal{L}_{\text{MBAT}}(\theta) = \frac{1}{B} \sum_{n=1}^B \left[ \sum_{i \in M} \gamma_i \mathcal{L}_{\text{task}}(\theta; x_i^{(n)} + \delta_i^{(n)}, \{x_j^{(n)}\}_{j \neq i}) \right]$$

where $\mathcal{L}_{\text{task}}$ is the original task loss (e.g., cross-entropy for classification), and $\gamma_i$ are modality-specific weights.

### 2.3 Adaptive Robustness Mechanism (ARM)

The ARM dynamically adjusts defensive priorities based on detected attack patterns, enabling real-time response to emerging threats.

#### 2.3.1 Attack Pattern Detection

1. Maintain a sliding window of recent inputs $\mathcal{W} = \{(x_1^{(t)}, x_2^{(t)}, ..., x_{|M|}^{(t)})\}_{t=t-k+1}^t$ where $k$ is the window size.

2. For each new input, compute the consistency scores $\{C(x_i, x_j)\}_{(i,j) \in M \times M}$.

3. Detect potential attack patterns by fitting a Gaussian Mixture Model to the distribution of consistency scores, where anomalies represent potential attacks.

4. For each modality pair $(i,j)$, compute an attack suspicion score:

$$S_{i,j} = 1 - \frac{\sum_{t=t-k+1}^t C(x_i^{(t)}, x_j^{(t)})}{k}$$

#### 2.3.2 Dynamic Defense Adaptation

1. Adjust the importance weights $\alpha_{i,j}$ in the CMCVM based on attack suspicion scores:

$$\alpha_{i,j}^{\text{new}} = \alpha_{i,j}^{\text{base}} \cdot (1 + \eta \cdot S_{i,j})$$

where $\eta$ is a scaling hyperparameter.

2. Similarly, adjust the modality-specific weights $\gamma_i$ in MBAT:

$$\gamma_i^{\text{new}} = \gamma_i^{\text{base}} \cdot \left(1 + \eta \cdot \max_{j \neq i} S_{i,j} \right)$$

3. Implement an exponential moving average update for the weights to ensure stability:

$$\alpha_{i,j}^t = (1 - \rho) \alpha_{i,j}^{t-1} + \rho \alpha_{i,j}^{\text{new}}$$
$$\gamma_i^t = (1 - \rho) \gamma_i^{t-1} + \rho \gamma_i^{\text{new}}$$

where $\rho$ controls the adaptation rate.

### 2.4 Unified Framework Integration

The three components are integrated into a unified defense framework:

1. During inference, incoming inputs are first passed through the CMCVM to detect potential cross-modal inconsistencies.

2. If inconsistencies exceed a threshold, the ARM activates to dynamically adjust the defensive weights.

3. The model is continuously fine-tuned using MBAT to maintain robustness against evolving attack patterns.

The complete loss function for training the CMAI framework is:

$$\mathcal{L}_{\text{CMAI}} = \mathcal{L}_{\text{task}} + \lambda_{\text{CMCVM}} \mathcal{L}_{\text{CMCVM}} + \lambda_{\text{MBAT}} \mathcal{L}_{\text{MBAT}}$$

where $\lambda_{\text{CMCVM}}$ and $\lambda_{\text{MBAT}}$ are hyperparameters controlling the contribution of each component.

### 2.5 Experimental Design

#### 2.5.1 Datasets

We will evaluate our approach on multiple multimodal datasets:

1. **MS-COCO**: Contains image-text pairs for evaluation of vision-language models.
2. **AudioSet**: Audio-visual dataset for evaluating audio-visual cross-modal attacks.
3. **HatefulMemes**: Multimodal dataset requiring both image and text understanding to identify harmful content.
4. **CLEVR-Dialog**: Complex reasoning dataset requiring visual and linguistic understanding.
5. **Multi-Avengers**: A custom dataset we will create specifically for cross-modal adversarial testing, containing deliberately misaligned cross-modal content.

#### 2.5.2 Models

We will implement and evaluate CMAI on several state-of-the-art LMMs:

1. **GPT-4V**: Representing vision-language capable large language models.
2. **LLaVA**: Open-source vision-language model.
3. **CLIP**: Contrastive Language-Image Pre-training model.
4. **BLIP-2**: Bootstrapping Language-Image Pre-training model.
5. **ImageBind**: Model that connects multiple modalities including vision, language, audio, etc.

#### 2.5.3 Attack Scenarios

We will evaluate against a comprehensive suite of cross-modal adversarial attacks:

1. **CrossFire** (Dou et al., 2024): Transforming inputs across modality boundaries.
2. **Universal adversarial attacks** (Rahmatullaev et al., 2025): Using optimized images to bypass alignment safeguards.
3. **I2V attacks** (Wei et al., 2021): Transferring adversarial perturbations from images to videos.
4. **Gradient-based cross-modal attacks**: Our own implementation of PGD attacks targeting cross-modal transfer points.
5. **Jailbreak attacks**: Testing against instruction-following violations through cross-modal inputs.

#### 2.5.4 Evaluation Metrics

We will use the following metrics to evaluate our approach:

1. **Robustness Metrics**:
   - Attack Success Rate (ASR): Percentage of successful attacks.
   - Cross-Modal Consistency Score (CMCS): Measure of alignment between modalities.
   - Robust Accuracy: Model performance under adversarial attack conditions.

2. **Performance Metrics**:
   - Clean Accuracy: Model performance on benign inputs.
   - Performance Drop: Difference between clean and robust accuracy.

3. **Efficiency Metrics**:
   - Computational Overhead: Additional computation time required by the defense.
   - Memory Overhead: Additional memory required by the defense.

4. **Adaptation Metrics**:
   - Response Time: Time taken to adapt defenses to new attack patterns.
   - False Positive Rate: Rate of benign inputs incorrectly classified as adversarial.

#### 2.5.5 Ablation Studies

We will conduct ablation studies to understand the contribution of each component:

1. CMAI without CMCVM
2. CMAI without MBAT
3. CMAI without ARM
4. Various combinations of components with different hyperparameter settings

#### 2.5.6 Comparative Analysis

We will compare CMAI against existing defense methods:

1. ProEAT (Lu et al., 2025)
2. Standard adversarial training
3. Cross-modal consistency training (White et al., 2023)
4. Adaptive defense mechanisms (Black et al., 2024)

This comprehensive experimental design will allow us to thoroughly evaluate the effectiveness of our proposed Cross-Modal Adversarial Immunization framework across diverse LMMs, datasets, and attack scenarios.

## 3. Expected Outcomes & Impact

### 3.1 Expected Outcomes

The successful implementation and evaluation of the Cross-Modal Adversarial Immunization (CMAI) framework is expected to yield several significant outcomes:

1. **Enhanced Cross-Modal Robustness**: We anticipate a substantial reduction in Attack Success Rate (ASR) for cross-modal adversarial attacks, with at least a 30-50% decrease compared to undefended models and a 15-25% improvement over current state-of-the-art defenses like ProEAT. This enhancement will be particularly pronounced for attacks that target modality integration points.

2. **Minimal Performance Trade-off**: The adaptive nature of our framework is expected to maintain high performance on benign inputs, with clean accuracy degradation of less than 5% across benchmark datasets. This represents a significant improvement over conventional adversarial training approaches that often suffer from larger performance drops.

3. **Transferable Defense Methodology**: Our framework should demonstrate effectiveness across multiple LMM architectures (GPT-4V, LLaVA, CLIP, BLIP-2, ImageBind), indicating the generalizability of the approach and its potential as a standard security enhancement for diverse multimodal systems.

4. **Real-time Adaptation Capabilities**: The Adaptive Robustness Mechanism should demonstrate the ability to detect and respond to novel attack patterns within a small number of examples (5-10), providing dynamic defense capabilities not present in static defense strategies.

5. **Comprehensive Cross-Modal Vulnerability Map**: Through extensive testing across modality pairs and attack types, we will develop a comprehensive mapping of cross-modal vulnerabilities in current LMMs, providing valuable insights for future model development.

6. **Efficient Implementation**: The architectural design should introduce minimal computational overhead (targeting less than 10% increase in inference time) while providing substantial security benefits, making it practical for real-world deployment.

### 3.2 Research Impact

The potential impact of this research extends across multiple domains:

#### 3.2.1 Technical Impact

The proposed CMAI framework represents a significant advancement in adversarial machine learning defenses by specifically addressing the unique challenges of cross-modal vulnerabilities. The technical contributions include:

- Novel methods for verifying cross-modal consistency that can detect subtle misalignments between modalities.
- Advanced adversarial training techniques that specifically target the vulnerable points where modalities interface.
- Adaptive defense mechanisms that can respond to emerging threats without requiring manual intervention.
- A unified framework that integrates these components in a computationally efficient manner.

These technical innovations advance the state of the art in defensive techniques for multimodal AI systems and provide a foundation for future research in cross-modal security.

#### 3.2.2 Practical Impact

The practical implications of this research are substantial for organizations deploying LMMs in high-stakes environments:

- **Autonomous Vehicles**: Enhanced safety through robust multimodal perception systems that resist adversarial manipulation of visual or auditory inputs.
- **Healthcare**: More reliable medical diagnostics and treatment recommendations based on multimodal patient data (images, text reports, sensor readings).
- **Content Moderation**: Improved detection of harmful content that attempts to evade filters by exploiting cross-modal vulnerabilities.
- **Financial Systems**: Strengthened security for authentication and fraud detection systems that rely on multiple input modalities.
- **Critical Infrastructure**: Enhanced protection for systems that use multimodal inputs for monitoring and control functions.

By providing a practical, deployable defense framework, this research directly addresses the security needs of real-world AI applications where reliability across multiple modalities is essential.

#### 3.2.3 Theoretical Impact

Beyond its practical applications, this research contributes to the theoretical understanding of cross-modal reasoning and adversarial vulnerabilities:

- Insights into the nature of cross-modal representations and how they can be manipulated or protected.
- New understanding of the integration points between different perceptual domains in neural architectures.
- Theoretical frameworks for evaluating and ensuring consistency across diverse representational spaces.
- Potential connections to cognitive science research on multisensory integration and its robustness properties in biological systems.

These theoretical contributions extend beyond security applications, potentially influencing fundamental research in multimodal learning and representation.

#### 3.2.4 Societal Impact

The broader societal impact of this research lies in its contribution to building more trustworthy AI systems:

- **Reliability**: More reliable AI systems that maintain consistent behavior across different input modalities, reducing unexpected failures.
- **Trustworthiness**: Enhanced user trust through demonstrably robust behavior in adversarial conditions.
- **Safety**: Reduced vulnerability to malicious manipulation, particularly important as LMMs are increasingly deployed in safety-critical applications.
- **Accessibility**: More robust multimodal systems can better serve users who rely on multiple modalities for interaction, including those with disabilities.

By addressing cross-modal vulnerabilities, this research contributes to the development of AI systems that can be more safely and confidently deployed in diverse societal contexts.

### 3.3 Future Research Directions

This work opens several promising avenues for future research:

1. **Expansion to Additional Modalities**: Extending the framework to encompass emerging modalities such as tactile inputs, 3D representations, or brain-computer interfaces.

2. **Certified Robustness for Multimodal Systems**: Developing theoretical guarantees of robustness for cross-modal systems, building on the empirical robustness demonstrated in this work.

3. **Self-Supervised Cross-Modal Robustness**: Exploring how self-supervised learning techniques can be integrated with the CMAI framework to enable robustness training with less reliance on labeled data.

4. **Cognitive-Inspired Defenses**: Investigating biological multisensory integration mechanisms for inspiration in designing more naturally robust cross-modal systems.

5. **Cross-Modal Explainability**: Developing methods to explain cross-modal decisions and identify potential vulnerabilities, making the defensive process more transparent.

In conclusion, the Cross-Modal Adversarial Immunization framework represents a significant step forward in addressing a critical vulnerability in contemporary AI systems. Its successful implementation will contribute to more secure, reliable, and trustworthy multimodal models across a wide range of applications.