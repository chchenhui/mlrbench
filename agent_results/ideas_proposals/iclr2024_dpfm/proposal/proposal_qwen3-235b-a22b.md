# **Reinforcement Learning–Guided Data Curation for Safety-Aligned Foundation Models**

---

## 1. Introduction

### Background
Foundation Models (FMs) such as GPT-4, LLaMA, and Stable Diffusion have revolutionized tasks in natural language processing, computer vision, and multimodal reasoning. However, their reliance on massive, unlabeled corpora to absorb toxic, biased, or misaligned content has led to alarming downstream consequences, including harmful outputs, privacy breaches, and ethical violations. While prior efforts have addressed safety through post-hoc alignment techniques like reinforcement learning (RL) with human feedback (RLHF) or reward modeling on downstream data, these approaches often struggle with scalability, overlook upstream data biases, and fail to address safety as a core training priority. Recent advances in data-centric AI—such as *safety pretraining* (Maini et al., 2025) and *automated safety alignment* (Shi et al., 2023)—highlight the critical need for dynamic, data-driven mechanisms to curate high-quality training content.

### Research Objectives
This proposal seeks to address the following objectives:
1. **Automated Data Curation**: Develop a reinforcement learning (RL)-based framework to dynamically prioritize safer, alignment-friendly training samples.
2. **Safety-Performance Balance**: Ensure that improved safety metrics do not degrade linguistic or task-specific capabilities.
3. **Closed-Loop Learning**: Enable iterative refinement of selection policies through feedback from fine-tuned models.
4. **Scalability**: Reduce dependency on manual filtering by leveraging automated safety detectors and lightweight model evaluations.

### Significance
By embedding safety into the pretraining phase via RL-driven curation, this work directly tackles key challenges highlighted in the literature:
- **Scalability of Data Curation**: Manual filtering (e.g., Safety Pretraining, Maini et al., 2025) is labor-intensive; automation via RL streamlines the process.
- **Alignment with Human Values**: Proxy alignment signals from labeled probes (similar to Safer-Instruct, Shi et al., 2023) provide dynamic adjustments for evolving societal norms.
- **Safety-Performance Trade-offs**: RAFT (Dong et al., 2023) focuses on post-alignment fine-tuning; our approach prioritizes safe samples upfront, preserving performance through iterative validation.

---

## 2. Methodology

### 2.1 Problem Formulation
Let $\mathcal{D}_{\text{raw}}$ be a massive unlabeled corpus and $\mathcal{P}(x)$ the probability distribution over its samples $x$. Our goal is to derive a curated subset $\mathcal{D}_{\text{safe}} \subset \mathcal{D}_{\text{raw}}$ with optimal safety-to-misalignment ratio while retaining linguistic diversity. The problem is cast as a Markov Decision Process (MDP):
- **State**: Features of the current training batch (e.g., toxicity scores, alignment proxy metrics).
- **Action**: Assign selection probabilities to samples in $\mathcal{D}_{\text{raw}}$.
- **Reward**: Composite metric combining safety and alignment signals.

### 2.2 Data Collection and Preprocessing
1. **Raw Corpus Curation**: Start with $\mathcal{D}_{\text{raw}}$, combining sources like Common Crawl, GitHub, and web-scraped dialogues.
2. **Initial Filtering**: Apply off-the-shelf detectors (e.g., Perspective API) to remove overtly toxic samples, forming a candidate pool $\mathcal{D}_{\text{pool}}$.
3. **Labeled Probes for Alignment**: Construct a small labeled dataset $\mathcal{D}_{\text{probe}}$ using active learning over $\mathcal{D}_{\text{pool}}$. Label examples based on Jigsaw’s toxicity taxonomy and proxy alignment tasks (e.g., instruction following, factuality).

### 2.3 Reinforcement Learning Framework

#### 2.3.1 Reward Design
Define a composite reward $R(x)$ for each sample $x \in \mathcal{D}_{\text{pool}}$:
$$
R(x) = \alpha \cdot \underbrace{[-T(x)]}_{\text{Safety}} + \beta \cdot \underbrace{\mathbb{E}_{y \sim \pi_{\theta}}}[A(x, y)]_{\text{Alignment}},
$$
where:
- $T(x)$: Toxicity score from Perspective API (normalized between 0–1).
- $A(x, y)$: Alignment signal for input $x$ and model output $y$, inferred using a lightweight probe (e.g., logistic regression on features from $\mathcal{D}_{\text{probe}}$).
- $\alpha + \beta = 1$: Hyperparameters balancing safety and alignment.

#### 2.3.2 Agent Architecture
- **Policy Model**: A transformer-based actor-critic network $\pi_{\theta}$ parameterizing the selection probability $p(a_t | s_t)$, where actions $a_t$ are weights assigned to samples in $\mathcal{D}_{\text{pool}}$.
- **Training Algorithm**: Proximal Policy Optimization (PPO) with clipped surrogate objective:
$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \left[ \min\left( r_t(\epsilon) \cdot \hat{A}_t, \text{clip}(r_t(\epsilon), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t \right) \right],
$$
where $r_t(\epsilon)$ is the ratio of new to old policy probabilities and $\hat{A}_t$ is the advantage function.

### 2.4 Algorithmic Workflow
1. **Initialize**: $\mathcal{D}_{\text{pool}}$, $\pi_{\theta}$, and the lightweight foundation model $\mathcal{M}_{\phi}$.
2. **Iterate**:
   - **Step 1**: Use $\pi_{\theta}$ to sample a training batch $\mathcal{B} \subset \mathcal{D}_{\text{pool}}$.
   - **Step 2**: Pretrain $\mathcal{M}_{\phi}$ on $\mathcal{B}$ using causal language modeling:
   $$
   \mathcal{L}_{\text{LM}} = -\sum_{i=1}^n \log p_{\phi}(x_i | x_{<i}).
   $$
   - **Step 3**: Evaluate $\mathcal{M}_{\phi}$ on safety metrics (e.g., Per-Batch Toxicity Reduction) and alignment tasks (e.g., AlpacaEval). Update $\pi_{\theta}$ using PPO.
   - **Step 4**: Retrain probes on new $\mathcal{M}_{\phi}$ outputs to refine $A(x, y)$.

### 2.5 Experimental Design

#### 2.5.1 Datasets
- **Base Corpus**: 1,000B tokens from Common Crawl, filtered to 500B tokens for $\mathcal{D}_{\text{pool}}$.
- **Evaluation Sets**:
  - **Toxicity**: RealToxicityPrompts (Gehman et al., 2020).
  - **Bias**: Winogender (Rudinger et al., 2018), Ceval (Zhao et al., 2021).
  - **Alignment**: TruthfulQA (Lin et al., 2022), AlpacaEval (Dubois et al., 2023).

#### 2.5.2 Baselines
- **Random Curation**: Uniform sampling from $\mathcal{D}_{\text{pool}}$.
- **Filter-Only**: Toxicity thresholding without RL.
- **RAFT** (Dong et al., 2023): Reward-ranked fine-tuning on model outputs.
- **Safer-Instruct** (Shi et al., 2023): Automated preference data generation.

#### 2.5.3 Metrics
- **Safety**: 
  - Average toxicity score (Perspective API).
  - Per-text false refusal rate (outputs refusing neutral prompts).
- **Alignment**:
  - Human preference rate (4k models in pairwise comparisons).
  - TruthfulQA score (grounded answers vs. hallucinations).
- **Capabilities**: 
  - Perplexity on PTB and LM-Wiki (Lin et al., 2021).

#### 2.5.4 Training Protocol
- **Models**: LLaMA-7B for $\mathcal{M}_{\phi}$, distilled RoBERTa for probes.
- **Hardware**: 8× A100 GPUs, 128 accelerators for distributed PPO.
- **Stopping Criteria**: 10 epochs of stable reward score plateau or 1M PPO optimization steps.

#### 2.5.5 Ablation Studies
- **Reward Components**: Test $\alpha, \beta$ with values in [0.0, 0.2, 0.5, 0.8].
- **Probe Size**: Evaluate performance with 1k, 10k, and 100k labeled examples.

#### 2.5.6 Ethical Considerations
- **Bias Mitigation**: Audit $\mathcal{D}_{\text{pool}}$ for underrepresented populations using Negative Effect Filtering.
- **Data Sensitivity**: Exclude samples with personally identifiable information (PII) using regex-based sanitization.

---

## 3. Expected Outcomes & Impact

### Expected Outcomes
1. **Safety Improvement**: Reduce toxicity scores by ≥40% compared to filter-only baselines while maintaining strict alignment with $\mathcal{D}_{\text{probe}}$ preferences.
2. **Closed-Loop Validation**: Demonstrate that periodic retraining of $\pi_{\theta}$ using fine-tuned $\mathcal{M}_{\phi}$ outperforms static RL policies by ≤15% in alignment accuracy.
3. **Performance Preservation**: Achieve perplexity metrics within ±5% of a model trained on $\mathcal{D}_{\text{raw}}$, ensuring linguistic capabilities are unperturbed.
4. **Benchmark Benchmarking**: Cross-evaluate with existing metrics like SELF-PX (Dass et al., 2023) and LegalEvaluation (Zheng et al., 2023) to assess legal-entity handling.

### Broader Impact
- **Scalable Data-Centric AI**: Enable organizations to curate petabyte-scale datasets without manual labeling bottlenecks.
- **Automated Governance**: Depersonalize curation workflows, reducing exposure of auditors to harmful content.
- **Data Economics**: Quantify the ROI of alignment-aware data curation, potentially shifting industry focus from model size to data quality.
- **Policy Influence**: Provide toolkits for regulatory compliance (e.g., GDPR Article 22 for profiling, CHIP Act requirements).

--- 

### Conclusion
This proposal pioneers a reinforcement learning–guided, data-centric paradigm for FM training that prioritizes safety upfront while maintaining performance. By integrating automated feedback loops, compositional reward design, and scalable validation, our framework directly addresses the workshop’s focus areas: data quality, safety alignment, and efficiency. The resultant pipeline offers a critical step toward democratizing safe AI development.